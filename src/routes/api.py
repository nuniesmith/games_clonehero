"""
Clone Hero Content Manager - JSON API Routes

Provides all REST API endpoints for:
- Songs CRUD (list, get, update, delete)
- Content upload (archives and direct files)
- Song generation from audio files
- Nextcloud WebDAV operations (browse, download, upload, sync)
- Health check
"""

import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import aiofiles
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from loguru import logger
from pydantic import BaseModel

from src.config import (
    ALLOWED_AUDIO_EXTENSIONS,
    ALLOWED_UPLOAD_EXTENSIONS,
    APP_VERSION,
    DB_PATH,
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_GB,
)
from src.database import (
    count_songs,
    delete_song,
    get_song_by_id,
    get_songs,
    update_song,
)
from src.services.content_manager import (
    delete_song_files,
    parse_song_ini,
    process_upload,
    scan_local_songs,
    write_song_ini,
)
from src.services.song_generator import process_song_file
from src.webdav import (
    check_connection,
    delete_remote,
    download_file,
    download_file_stream,
    is_configured,
    list_directory,
    mkdir,
    upload_file,
)

router = APIRouter(prefix="/api", tags=["API"])

# Track startup time for health check
_START_TIME = time.time()


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
# Songs CRUD
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
    """Update a song's metadata."""
    # Fetch current song to ensure it exists
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

    # Also update the song.ini file on disk if it exists
    file_path = song.get("file_path", "")
    ini_path = Path(file_path) / "song.ini" if file_path else None
    if ini_path and ini_path.exists():
        # Merge current data with updates
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
            write_song_ini(ini_path, song_data)

    return {"message": f"Song {song_id} updated successfully"}


@router.delete("/songs/{song_id}")
async def api_delete_song(song_id: int):
    """Delete a song from the database and optionally from disk."""
    song = await get_song_by_id(song_id)
    if not song:
        raise HTTPException(status_code=404, detail="Song not found")

    # Delete from disk
    file_path = song.get("file_path", "")
    if file_path:
        delete_song_files(file_path)

    # Delete from database
    deleted = await delete_song(song_id)
    if not deleted:
        raise HTTPException(
            status_code=500, detail="Failed to delete song from database"
        )

    return {"message": f"Song {song_id} deleted successfully"}


# ---------------------------------------------------------------------------
# Content Upload
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

    Extracts archives, parses song.ini files, and registers songs in the database.
    """
    valid_types = {"songs", "backgrounds", "colors", "highways"}
    if content_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content_type: {content_type}. Must be one of: {', '.join(valid_types)}",
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

        # Process the uploaded file
        result = process_upload(temp_path, content_type)

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
# Song Generator
# ---------------------------------------------------------------------------
@router.post("/generate")

async def api_generate_song(
    file: UploadFile = File(...),
    song_name: Optional[str] = Form(None),
    artist: Optional[str] = Form(None),
    difficulty: str = Form("expert"),
):
    """
    Upload an audio file and generate a Clone Hero chart from it.

    Supports MP3, OGG, WAV, FLAC, and OPUS formats.
    """
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

        # Process the song
        import asyncio

        result = await asyncio.to_thread(
            process_song_file,
            temp_path,
            song_name=song_name or Path(filename).stem,
            artist=artist,
            difficulty=difficulty,
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
# Local content scan
# ---------------------------------------------------------------------------
@router.get("/scan")
async def api_scan_local():
    """Scan the local content directory for song.ini files."""
    import asyncio

    songs = await asyncio.to_thread(scan_local_songs)
    return {
        "total": len(songs),
        "songs": songs,
    }


# ---------------------------------------------------------------------------
# WebDAV / Nextcloud endpoints
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


@router.post("/webdav/download")
async def api_webdav_download(body: WebDAVDownloadRequest):
    """
    Download a file from Nextcloud and process it as content.

    For archives (.zip/.rar), extracts and processes songs.
    For other files, saves them to the appropriate content directory.
    """
    if not is_configured():
        raise HTTPException(
            status_code=503, detail="Nextcloud WebDAV is not configured"
        )

    remote_path = body.remote_path.strip()
    filename = Path(remote_path).name
    ext = Path(filename).suffix.lower()

    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}",
        )

    temp_path = _get_temp_path(filename)

    try:
        # Download from Nextcloud
        success = await download_file_stream(remote_path, temp_path)
        if not success:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to download {remote_path} from Nextcloud",
            )

        logger.info(f"⬇️ Downloaded from Nextcloud: {remote_path}")

        # Process the downloaded file
        result = process_upload(temp_path, body.content_type)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"❌ WebDAV download error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.remove(temp_path)
        except FileNotFoundError:
            pass


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


@router.post("/webdav/sync-to-nextcloud")
async def api_sync_to_nextcloud(song_id: int = Form(...)):
    """
    Sync a local song to Nextcloud.

    Uploads the song's entire folder contents to Nextcloud under the
    configured remote path, preserving the folder structure.
    """
    if not is_configured():
        raise HTTPException(
            status_code=503, detail="Nextcloud WebDAV is not configured"
        )

    song = await get_song_by_id(song_id)
    if not song:
        raise HTTPException(status_code=404, detail="Song not found")

    file_path = song.get("file_path", "")
    song_dir = Path(file_path)
    if not song_dir.exists() or not song_dir.is_dir():
        raise HTTPException(status_code=404, detail="Song folder not found on disk")

    artist = song.get("artist", "Unknown")
    title = song.get("title", "Unknown")
    remote_base = f"/songs/{artist}/{title}"

    uploaded_files = []
    errors = []

    # Ensure remote directory exists
    await mkdir(remote_base)

    # Upload all files in the song directory
    for local_file in song_dir.rglob("*"):
        if local_file.is_file():
            relative = local_file.relative_to(song_dir)
            remote_file = f"{remote_base}/{relative.as_posix()}"

            try:
                content = local_file.read_bytes()
                success = await upload_file(remote_file, content)
                if success:
                    uploaded_files.append(remote_file)
                else:
                    errors.append(f"Failed to upload: {relative}")
            except Exception as e:
                errors.append(f"Error uploading {relative}: {str(e)}")

    result = {
        "message": f"Synced {len(uploaded_files)} file(s) to Nextcloud",
        "remote_path": remote_base,
        "uploaded": uploaded_files,
    }

    if errors:
        result["errors"] = errors

    return result


@router.post("/webdav/sync-from-nextcloud")
async def api_sync_from_nextcloud(remote_path: str = Form(...)):
    """
    Download a song folder from Nextcloud and import it locally.

    Expects a folder containing at minimum a song.ini file.
    Downloads all files in the remote folder and processes them as a song.
    """
    if not is_configured():
        raise HTTPException(
            status_code=503, detail="Nextcloud WebDAV is not configured"
        )

    # List all files in the remote path
    items = await list_directory(remote_path)
    if not items:
        raise HTTPException(
            status_code=404,
            detail=f"No files found at {remote_path}",
        )

    # Create a temp directory and download everything
    temp_dir = Path(tempfile.mkdtemp(prefix="ch_ncsync_"))

    try:
        download_count = 0
        for item in items:
            if item.is_directory:
                # Skip subdirectories for now (flat import)
                continue

            content = await download_file(item.path)
            if content is not None:
                dest = temp_dir / item.name
                dest.write_bytes(content)
                download_count += 1

        if download_count == 0:
            raise HTTPException(
                status_code=400,
                detail="No files could be downloaded from the remote path",
            )

        logger.info(f"⬇️ Downloaded {download_count} files from Nextcloud:{remote_path}")

        # Check if there's a song.ini
        ini_path = temp_dir / "song.ini"
        if ini_path.exists():
            parsed = parse_song_ini(ini_path)
            if parsed:
                import shutil

                from src.services.content_manager import get_content_directory

                songs_dir = get_content_directory("songs")
                safe_name = parsed["artist"].replace("/", "_")
                safe_title = parsed["title"].replace("/", "_")
                dest_dir = (
                    songs_dir / safe_name / f"{safe_title}_{uuid.uuid4().hex[:8]}"
                )
                dest_dir.mkdir(parents=True, exist_ok=True)

                # Move all files to the song directory
                for f in temp_dir.iterdir():
                    if f.is_file():
                        shutil.move(str(f), str(dest_dir / f.name))

                # Register in database
                import asyncio

                from src.database import insert_song_sync

                metadata_json = json.dumps(parsed.get("metadata", {}))
                song_id = await asyncio.to_thread(
                    insert_song_sync,
                    title=parsed["title"],
                    artist=parsed["artist"],
                    album=parsed.get("album", "Unknown"),
                    file_path=str(dest_dir),
                    metadata=metadata_json,
                )

                return {
                    "message": f"Imported song: {parsed['title']} - {parsed['artist']}",
                    "song_id": song_id,
                    "folder_path": str(dest_dir),
                }
            else:
                raise HTTPException(
                    status_code=400,
                    detail="song.ini found but could not be parsed (missing required fields)",
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="No song.ini found in the remote directory. Cannot import as a song.",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"❌ Sync from Nextcloud error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        import shutil

        shutil.rmtree(str(temp_dir), ignore_errors=True)
