"""
Clone Hero Content Manager - JSON API Routes

Provides all REST API endpoints for:
- Songs CRUD (list, get, update, delete) — backed by Nextcloud WebDAV
- Content upload (archives and direct files) — songs go to Nextcloud
- Song generation from audio files — output uploaded to Nextcloud
- Nextcloud WebDAV operations (browse, download, upload, mkdir, delete)
- Library sync (scan Nextcloud and refresh local metadata cache)
- Library sync with SSE streaming for real-time progress
- System status and background activity reporting
- Health check
"""

import asyncio
import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import aiofiles
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel

from src.config import (
    ALLOWED_AUDIO_EXTENSIONS,
    ALLOWED_UPLOAD_EXTENSIONS,
    APP_VERSION,
    DB_PATH,
    DB_SYNC_INTERVAL,
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_GB,
    NEXTCLOUD_DB_PATH,
)
from src.database import (
    count_songs,
    delete_song,
    get_song_by_id,
    get_songs,
    update_song,
)
from src.services.content_manager import (
    delete_song_from_nextcloud,
    get_sync_state,
    process_upload,
    sync_library_from_nextcloud,
    sync_library_from_nextcloud_stream,
)
from src.services.song_generator import process_and_upload_song
from src.webdav import (
    check_connection,
    delete_remote,
    is_configured,
    list_directory,
    mkdir,
    upload_file,
    write_remote_song_ini,
)

router = APIRouter(prefix="/api", tags=["API"])

# Track startup time for health check
_START_TIME = time.time()
# Track last DB upload time (updated by main.py after each upload)
_last_db_upload: float | None = None


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class SongUpdate(BaseModel):
    title: Optional[str] = None
    artist: Optional[str] = None
    album: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class WebDAVDownloadRequest(BaseModel):
    remote_path: str
    content_type: str = "songs"


class WebDAVUploadRequest(BaseModel):
    remote_path: str


class GenerateSongRequest(BaseModel):
    song_name: Optional[str] = None
    artist: Optional[str] = None
    difficulty: str = "expert"
    enable_lyrics: bool = True
    enable_album_art: bool = True


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@router.get("/health")
async def health_check():
    """Health check endpoint for the service."""
    uptime = round(time.time() - _START_TIME, 2)
    db_ok = DB_PATH.exists()
    webdav_ok = is_configured()

    status = "ok" if db_ok else "degraded"

    return {
        "status": status,
        "database": "ok" if db_ok else "missing",
        "webdav_configured": webdav_ok,
        "uptime_seconds": uptime,
        "version": APP_VERSION,
    }


# ---------------------------------------------------------------------------
# System Status (background activity reporting)
# ---------------------------------------------------------------------------
@router.get("/system/status")
async def api_system_status():
    """
    Return system status including background task activity.

    Reports on: uptime, DB sync state, library sync state,
    Nextcloud connection, and song count.
    """
    uptime = round(time.time() - _START_TIME, 2)
    db_size = DB_PATH.stat().st_size if DB_PATH.exists() else 0
    sync_state = get_sync_state()
    webdav_ok = is_configured()

    result = {
        "uptime_seconds": uptime,
        "version": APP_VERSION,
        "database": {
            "path": str(DB_PATH),
            "exists": DB_PATH.exists(),
            "size_bytes": db_size,
            "nextcloud_path": NEXTCLOUD_DB_PATH,
        },
        "db_sync": {
            "interval_seconds": DB_SYNC_INTERVAL,
            "last_upload": _last_db_upload,
            "last_upload_ago": (
                round(time.time() - _last_db_upload, 1) if _last_db_upload else None
            ),
        },
        "library_sync": sync_state,
        "webdav_configured": webdav_ok,
        "total_songs": await count_songs(),
    }

    return result


@router.get("/system/sync-state")
async def api_sync_state():
    """
    Lightweight endpoint returning only the current library sync state.

    Designed for polling from the UI to show real-time sync activity
    without the overhead of the full system status.
    """
    return get_sync_state()


# ---------------------------------------------------------------------------
# Songs CRUD (metadata cached locally, files on Nextcloud)
# ---------------------------------------------------------------------------
@router.get("/songs")
async def api_list_songs(
    search: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List songs with optional search and pagination."""
    search_query = search.strip() if search else None
    total = await count_songs(search=search_query)
    songs = await get_songs(search=search_query, limit=limit, offset=offset)

    # Parse metadata JSON strings into dicts
    for song in songs:
        meta = song.get("metadata")
        if isinstance(meta, str):
            try:
                song["metadata"] = json.loads(meta)
            except (json.JSONDecodeError, TypeError):
                song["metadata"] = {}

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "songs": songs,
    }


@router.get("/songs/{song_id}")
async def api_get_song(song_id: int):
    """Get a single song by ID."""
    song = await get_song_by_id(song_id)
    if not song:
        raise HTTPException(status_code=404, detail="Song not found")

    meta = song.get("metadata")
    if isinstance(meta, str):
        try:
            song["metadata"] = json.loads(meta)
        except (json.JSONDecodeError, TypeError):
            song["metadata"] = {}

    return song


@router.put("/songs/{song_id}")
async def api_update_song(song_id: int, body: SongUpdate):
    """
    Update a song's metadata.

    Updates the local database cache AND the song.ini file on Nextcloud
    so the two stay in sync.
    """
    song = await get_song_by_id(song_id)
    if not song:
        raise HTTPException(status_code=404, detail="Song not found")

    fields: Dict[str, Any] = {}
    if body.title is not None:
        fields["title"] = body.title
    if body.artist is not None:
        fields["artist"] = body.artist
    if body.album is not None:
        fields["album"] = body.album
    if body.metadata is not None:
        fields["metadata"] = json.dumps(body.metadata)

    if not fields:
        raise HTTPException(status_code=400, detail="No fields to update")

    updated = await update_song(song_id, **fields)
    if not updated:
        raise HTTPException(status_code=500, detail="Failed to update song")

    # Also update the song.ini on Nextcloud
    remote_path = song.get("remote_path", "")
    if remote_path and is_configured():
        updated_song = await get_song_by_id(song_id)
        if updated_song:
            meta = updated_song.get("metadata", "{}")
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except (json.JSONDecodeError, TypeError):
                    meta = {}
            song_data = {
                "title": updated_song.get("title", ""),
                "artist": updated_song.get("artist", ""),
                "album": updated_song.get("album", ""),
                "metadata": meta,
            }
            ok = await write_remote_song_ini(remote_path, song_data)
            if not ok:
                logger.warning(
                    "⚠️ Updated DB but failed to update song.ini on Nextcloud for song {}",
                    song_id,
                )

    return {"message": f"Song {song_id} updated successfully"}


@router.delete("/songs/{song_id}")
async def api_delete_song(song_id: int):
    """Delete a song from the database AND from Nextcloud."""
    song = await get_song_by_id(song_id)
    if not song:
        raise HTTPException(status_code=404, detail="Song not found")

    # Delete from Nextcloud
    remote_path = song.get("remote_path", "")
    if remote_path and is_configured():
        deleted_remote = await delete_song_from_nextcloud(remote_path)
        if not deleted_remote:
            logger.warning(
                "⚠️ Could not delete song from Nextcloud (remote_path={}). "
                "Removing from local DB anyway.",
                remote_path,
            )

    # Delete from database
    deleted = await delete_song(song_id)
    if not deleted:
        raise HTTPException(
            status_code=500, detail="Failed to delete song from database"
        )

    return {"message": f"Song {song_id} deleted successfully"}


# ---------------------------------------------------------------------------
# Content Upload (songs go to Nextcloud)
# ---------------------------------------------------------------------------
def _get_temp_path(filename: str) -> str:
    """Generate a secure temporary file path."""
    return os.path.join(tempfile.gettempdir(), f"ch_{uuid.uuid4().hex}_{filename}")


def _validate_extension(filename: str, allowed: set) -> str:
    """Validate and return the file extension, raising HTTPException if invalid."""
    ext = Path(filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file extension: {ext}. Allowed: {', '.join(sorted(allowed))}",
        )
    return ext


@router.post("/upload")
async def api_upload_content(
    file: UploadFile = File(...),
    content_type: str = Form("songs"),
):
    """
    Upload a content file (ZIP/RAR archive for songs, or direct files for assets).

    For songs: extracts archives, parses song.ini files, uploads to Nextcloud,
    and registers metadata in the local database cache.

    For other content types (backgrounds, colors, highways): stores files
    locally as before.
    """
    valid_types = {"songs", "backgrounds", "colors", "highways"}
    if content_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content_type: {content_type}. Must be one of: {', '.join(valid_types)}",
        )

    if content_type == "songs" and not is_configured():
        raise HTTPException(
            status_code=503,
            detail="Nextcloud WebDAV is not configured. Songs are stored on Nextcloud. "
            "Set NEXTCLOUD_URL, NEXTCLOUD_USERNAME, and NEXTCLOUD_PASSWORD in your .env file.",
        )

    filename = file.filename
    if not filename:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must have a filename.",
        )

    _validate_extension(filename, ALLOWED_UPLOAD_EXTENSIONS)

    # Save to temp file with size check
    temp_path = _get_temp_path(filename)
    total_size = 0

    try:
        async with aiofiles.open(temp_path, "wb") as f:
            while chunk := await file.read(65536):
                total_size += len(chunk)
                if total_size > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size is {MAX_FILE_SIZE_GB}GB.",
                    )
                await f.write(chunk)

        logger.info(
            f"📤 Upload received: {filename} ({total_size} bytes) -> {content_type}"
        )

        # Process the uploaded file (songs are uploaded to Nextcloud inside)
        result = await process_upload(temp_path, content_type)

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"❌ Upload processing error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Upload processing failed: {str(e)}"
        )
    finally:
        try:
            os.remove(temp_path)
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# Song Generator (output uploaded to Nextcloud)
# ---------------------------------------------------------------------------
@router.post("/generate")
async def api_generate_song(
    file: UploadFile = File(...),
    song_name: Optional[str] = Form(None),
    artist: Optional[str] = Form(None),
    difficulty: str = Form("expert"),
    enable_lyrics: bool = Form(True),
    enable_album_art: bool = Form(True),
):
    """
    Upload an audio file and generate a Clone Hero chart from it.

    The generated chart and song.ini are uploaded to Nextcloud and
    registered in the local metadata cache.

    Supports MP3, OGG, WAV, FLAC, and OPUS formats.
    """
    if not is_configured():
        raise HTTPException(
            status_code=503,
            detail="Nextcloud WebDAV is not configured. Generated songs are stored on Nextcloud. "
            "Set NEXTCLOUD_URL, NEXTCLOUD_USERNAME, and NEXTCLOUD_PASSWORD in your .env file.",
        )

    filename = file.filename
    if not filename:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must have a filename.",
        )

    ext = _validate_extension(filename, ALLOWED_AUDIO_EXTENSIONS)

    temp_path = _get_temp_path(f"gen_{uuid.uuid4().hex}{ext}")

    try:
        async with aiofiles.open(temp_path, "wb") as f:
            while chunk := await file.read(65536):
                await f.write(chunk)

        logger.info(f"🎵 Generate request: {filename} (difficulty={difficulty})")

        result = await process_and_upload_song(
            temp_path,
            song_name=song_name or Path(filename).stem,
            artist=artist,
            difficulty=difficulty,
            enable_lyrics=enable_lyrics,
            enable_album_art=enable_album_art,
        )

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"❌ Song generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Song generation failed: {str(e)}")
    finally:
        try:
            os.remove(temp_path)
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# Library Sync (scan Nextcloud → refresh local DB cache)
# ---------------------------------------------------------------------------
@router.post("/library/sync")
async def api_sync_library():
    """
    Scan the Nextcloud songs directory and refresh the local metadata cache.

    - Discovers all song.ini files under the configured songs path
    - Parses metadata and upserts into the local SQLite database
    - Removes DB entries for songs no longer on Nextcloud
    """
    if not is_configured():
        raise HTTPException(
            status_code=503,
            detail="Nextcloud WebDAV is not configured. "
            "Set NEXTCLOUD_URL, NEXTCLOUD_USERNAME, and NEXTCLOUD_PASSWORD in your .env file.",
        )

    result = await sync_library_from_nextcloud()

    if "error" in result:
        raise HTTPException(status_code=502, detail=result["error"])

    return result


@router.get("/library/sync/stream")
async def api_sync_library_stream():
    """
    Stream library sync progress as Server-Sent Events (SSE).

    Each event is a JSON object with a ``type`` field describing the
    event kind (start, discovering, discovered, parsing, parsed,
    parse_error, purging, purged, complete, error).

    Usage from JavaScript::

        const es = new EventSource('/api/library/sync/stream');
        es.onmessage = (e) => {
            const data = JSON.parse(e.data);
            console.log(data.type, data.message);
        };
    """
    if not is_configured():

        async def error_stream():
            payload = json.dumps(
                {
                    "type": "error",
                    "message": "Nextcloud WebDAV is not configured.",
                }
            )
            yield f"data: {payload}\n\n"

        return StreamingResponse(
            error_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Check if a sync is already running
    state = get_sync_state()
    if state.get("running"):

        async def already_running():
            payload = json.dumps(
                {
                    "type": "error",
                    "message": "A library sync is already in progress.",
                }
            )
            yield f"data: {payload}\n\n"

        return StreamingResponse(
            already_running(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    async def event_stream():
        try:
            async for event in sync_library_from_nextcloud_stream():
                payload = json.dumps(event)
                yield f"data: {payload}\n\n"
                # Small yield to let the event loop flush
                await asyncio.sleep(0.01)
        except Exception as e:
            logger.error("❌ SSE sync stream error: {}", e)
            payload = json.dumps(
                {
                    "type": "error",
                    "message": f"Stream error: {str(e)}",
                }
            )
            yield f"data: {payload}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/library/status")
async def api_library_status():
    """Return library statistics and Nextcloud connection info."""
    total = await count_songs()
    webdav_ok = is_configured()

    result = {
        "total_songs": total,
        "webdav_configured": webdav_ok,
        "version": APP_VERSION,
    }

    if webdav_ok:
        status = await check_connection()
        result["webdav_connected"] = status.get("connected", False)
        result["webdav_url"] = status.get("url", "")
        if status.get("error"):
            result["webdav_error"] = status["error"]

    return result


# ---------------------------------------------------------------------------
# WebDAV / Nextcloud endpoints (general file browser)
# ---------------------------------------------------------------------------
@router.get("/webdav/status")
async def api_webdav_status():
    """Check the Nextcloud WebDAV connection status."""
    if not is_configured():
        return {
            "configured": False,
            "connected": False,
            "error": "Nextcloud WebDAV is not configured. Set NEXTCLOUD_URL, NEXTCLOUD_USERNAME, and NEXTCLOUD_PASSWORD in your .env file.",
        }

    status = await check_connection()
    return {
        "configured": True,
        "connected": status.get("connected", False),
        "url": status.get("url", ""),
        "error": status.get("error"),
    }


@router.get("/webdav/browse")
async def api_webdav_browse(path: str = Query("/", alias="path")):
    """List files and directories at a given Nextcloud path."""
    if not is_configured():
        raise HTTPException(
            status_code=503, detail="Nextcloud WebDAV is not configured"
        )

    items = await list_directory(path)
    return {
        "path": path,
        "total": len(items),
        "items": [item.to_dict() for item in items],
    }


@router.post("/webdav/import")
async def api_webdav_import_song(remote_path: str = Form(...)):
    """
    Import a song folder from Nextcloud into the local metadata cache.

    Expects *remote_path* to point to a folder containing a song.ini.
    The song stays on Nextcloud — only its metadata is cached locally.
    """
    if not is_configured():
        raise HTTPException(
            status_code=503, detail="Nextcloud WebDAV is not configured"
        )

    from src.database import upsert_song
    from src.webdav import parse_remote_song_ini

    parsed = await parse_remote_song_ini(remote_path)
    if not parsed:
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse song.ini in {remote_path}. "
            "Ensure the folder contains a valid song.ini with at least name and artist fields.",
        )

    metadata_json = json.dumps(parsed.get("metadata", {}))
    song_id = await upsert_song(
        title=parsed["title"],
        artist=parsed["artist"],
        album=parsed["album"],
        remote_path=remote_path.rstrip("/"),
        metadata=metadata_json,
    )

    return {
        "message": f"Imported song: {parsed['title']} - {parsed['artist']}",
        "song_id": song_id,
        "remote_path": remote_path,
    }


@router.post("/webdav/upload")
async def api_webdav_upload(
    file: UploadFile = File(...),
    remote_path: str = Form("/"),
):
    """Upload a file to Nextcloud via WebDAV."""
    if not is_configured():
        raise HTTPException(
            status_code=503, detail="Nextcloud WebDAV is not configured"
        )

    filename = file.filename
    if not filename:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must have a filename.",
        )

    try:
        content = await file.read()

        # Determine content type
        content_type = file.content_type or "application/octet-stream"

        # Build the full remote path
        dest = remote_path.rstrip("/") + "/" + filename

        success = await upload_file(dest, content, content_type)
        if not success:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to upload {filename} to Nextcloud",
            )

        return {
            "message": f"Uploaded {filename} to Nextcloud",
            "remote_path": dest,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"❌ WebDAV upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/webdav/mkdir")
async def api_webdav_mkdir(remote_path: str = Form(...)):
    """Create a directory on Nextcloud."""
    if not is_configured():
        raise HTTPException(
            status_code=503, detail="Nextcloud WebDAV is not configured"
        )

    success = await mkdir(remote_path)
    if not success:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to create directory: {remote_path}",
        )

    return {"message": f"Directory created: {remote_path}"}


@router.delete("/webdav/delete")
async def api_webdav_delete(path: str = Query(...)):
    """Delete a file or directory on Nextcloud."""
    if not is_configured():
        raise HTTPException(
            status_code=503, detail="Nextcloud WebDAV is not configured"
        )

    success = await delete_remote(path)
    if not success:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to delete: {path}",
        )

    return {"message": f"Deleted: {path}"}
