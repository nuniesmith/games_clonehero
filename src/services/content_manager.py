"""
Clone Hero Content Manager - Content Management Service

Handles:
- Parsing song.ini files to extract metadata
- Extracting ZIP/RAR archives to a temporary staging area
- Uploading song folders to Nextcloud via WebDAV
- Uploading non-song assets (backgrounds, colors, highways) to Nextcloud
- Registering songs in the local SQLite metadata cache
- Cleaning up temporary files after upload

All persistent storage lives on Nextcloud.  Local disk is only used as a
transient staging area for extraction and processing before upload.
"""

import configparser
import json
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import rarfile
from loguru import logger

from src.config import (
    NEXTCLOUD_FOLDERS,
    NEXTCLOUD_SONGS_PATH,
    OPTIONAL_SONG_FIELDS,
    TEMP_DIR,
)


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------
def get_temp_staging_dir(prefix: str = "ch_stage_") -> Path:
    """Create and return a temporary staging directory."""
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=str(TEMP_DIR)))


# ---------------------------------------------------------------------------
# Song.ini parsing
# ---------------------------------------------------------------------------
def parse_song_ini(ini_path: Path) -> Optional[Dict[str, Any]]:
    """
    Parse a song.ini file and return structured metadata.

    Returns None if the file is invalid or missing required fields.
    Required fields: name, artist.
    """
    config = configparser.ConfigParser(strict=False)
    try:
        with ini_path.open("r", encoding="utf-8-sig") as f:
            config.read_file(f)
    except Exception as e:
        logger.error(f"❌ Failed to read {ini_path}: {e}")
        return None

    if not config.has_section("song"):
        logger.warning(f"⚠️ Missing [song] section in {ini_path}")
        return None

    name = config.get("song", "name", fallback=None)
    artist = config.get("song", "artist", fallback=None)
    album = config.get("song", "album", fallback=None)

    if not name or not artist:
        logger.warning(f"⚠️ Missing required fields (name/artist) in {ini_path}")
        return None

    # Collect optional metadata
    metadata = {}
    for field_name in OPTIONAL_SONG_FIELDS:
        if config.has_option("song", field_name):
            value = config.get("song", field_name, fallback=None)
            if value is not None:
                metadata[field_name] = value.strip()

    return {
        "title": name.strip(),
        "artist": artist.strip(),
        "album": album.strip() if album else "Unknown",
        "metadata": metadata,
    }


def write_song_ini(ini_path: Path, song_data: Dict[str, Any]) -> bool:
    """
    Write or update a song.ini file with the provided data.

    song_data should contain: title, artist, album, and optionally metadata dict.
    """
    config = configparser.ConfigParser()
    config.add_section("song")

    config.set("song", "name", song_data.get("title", ""))
    config.set("song", "artist", song_data.get("artist", ""))
    config.set("song", "album", song_data.get("album", ""))

    # Write optional metadata fields
    metadata = song_data.get("metadata", {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except (json.JSONDecodeError, TypeError):
            metadata = {}

    for key, value in metadata.items():
        if value is not None:
            config.set("song", key, str(value))

    try:
        ini_path.parent.mkdir(parents=True, exist_ok=True)
        with ini_path.open("w", encoding="utf-8") as f:
            config.write(f)
        logger.info(f"✅ Wrote song.ini: {ini_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to write song.ini at {ini_path}: {e}")
        return False


# ---------------------------------------------------------------------------
# Archive extraction
# ---------------------------------------------------------------------------
def extract_archive(file_path: str, extract_dir: str) -> Dict[str, Any]:
    """
    Extract a .zip or .rar archive to the given directory.

    Returns a dict with 'success': True or 'error': str.
    """
    file_ext = Path(file_path).suffix.lower()

    try:
        if file_ext == ".zip":
            with zipfile.ZipFile(file_path, "r") as zf:
                zf.extractall(extract_dir)
            logger.info(f"📦 Extracted ZIP: {file_path}")
            return {"success": True}

        elif file_ext == ".rar":
            try:
                with rarfile.RarFile(file_path) as rf:
                    rf.extractall(extract_dir)
                logger.info(f"📦 Extracted RAR: {file_path}")
                return {"success": True}
            except rarfile.NeedFirstVolume:
                logger.error(f"🚨 Multi-part RAR not supported: {file_path}")
                return {"error": "Multi-part RAR archives are not supported"}
            except rarfile.RarCannotExec as e:
                logger.error(f"❌ RAR extraction needs 'unrar' binary: {e}")
                return {"error": "RAR extraction failed. Ensure 'unrar' is installed."}

        else:
            return {"error": f"Unsupported archive format: {file_ext}"}

    except zipfile.BadZipFile:
        logger.error(f"❌ Corrupted ZIP file: {file_path}")
        return {"error": "The ZIP file is corrupted or invalid"}
    except Exception as e:
        logger.exception(f"❌ Error extracting {file_path}: {e}")
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Non-song asset upload to Nextcloud
# ---------------------------------------------------------------------------
async def _upload_asset_to_nextcloud(
    local_path: Path,
    content_type: str,
) -> Optional[str]:
    """
    Upload a single non-song asset file to the appropriate Nextcloud folder.

    Returns the remote path on success, None on failure.
    """
    from src.webdav import is_configured, upload_file

    if not is_configured():
        logger.error(
            "❌ Nextcloud is not configured — cannot store {} asset", content_type
        )
        return None

    nextcloud_folder = NEXTCLOUD_FOLDERS.get(content_type)
    if not nextcloud_folder:
        logger.error(
            "❌ No Nextcloud folder mapping for content type: {}", content_type
        )
        return None

    remote_path = f"{nextcloud_folder.rstrip('/')}/{local_path.name}"

    try:
        file_bytes = local_path.read_bytes()
        ok = await upload_file(remote_path, file_bytes)
        if ok:
            logger.info(f"⬆️ Uploaded {content_type} asset to Nextcloud: {remote_path}")
            return remote_path
        else:
            logger.error(f"❌ Failed to upload asset to Nextcloud: {remote_path}")
            return None
    except Exception as e:
        logger.error(f"❌ Error uploading asset {local_path.name}: {e}")
        return None


# ---------------------------------------------------------------------------
# Content processing pipeline  (Nextcloud-first)
# ---------------------------------------------------------------------------
async def process_extracted_songs(extract_dir: str) -> List[Dict[str, Any]]:
    """
    Walk an extracted directory, find all song.ini files, parse them,
    upload each song folder to Nextcloud, and register in the DB.

    Returns a list of successfully stored song info dicts.
    """
    from src.database import upsert_song
    from src.webdav import is_configured, upload_song_folder

    stored_songs: List[Dict[str, Any]] = []
    extract_path = Path(extract_dir)

    for ini_path in extract_path.rglob("song.ini"):
        parsed = parse_song_ini(ini_path)
        if not parsed:
            continue

        title = parsed["title"]
        artist = parsed["artist"]
        album = parsed["album"]
        metadata = parsed["metadata"]
        song_source = ini_path.parent

        if not is_configured():
            logger.error(
                "❌ Nextcloud is not configured — cannot store song '{}'. "
                "Set NEXTCLOUD_URL, NEXTCLOUD_USERNAME, and NEXTCLOUD_PASSWORD.",
                title,
            )
            continue

        # Upload to Nextcloud
        unique_suffix = uuid.uuid4().hex[:8]
        remote_path = await upload_song_folder(
            local_dir=str(song_source),
            artist=artist,
            title=title,
            suffix=unique_suffix,
        )

        if not remote_path:
            logger.error(f"❌ Failed to upload song '{title}' to Nextcloud")
            continue

        # Register in database with the Nextcloud remote path
        metadata_json = json.dumps(metadata)
        song_id = await upsert_song(
            title=title,
            artist=artist,
            album=album,
            remote_path=remote_path,
            metadata=metadata_json,
        )

        if song_id and song_id > 0:
            stored_songs.append(
                {
                    "id": song_id,
                    "title": title,
                    "artist": artist,
                    "album": album,
                    "remote_path": remote_path,
                    "metadata": metadata,
                }
            )

    return stored_songs


async def process_upload(file_path: str, content_type: str = "songs") -> Dict[str, Any]:
    """
    Full pipeline: extract archive -> process content -> upload to Nextcloud -> cleanup.

    For songs: extracts archives, parses song.ini files, uploads song folders
    to Nextcloud, and registers metadata in the local database cache.

    For non-song content types (backgrounds, colors, highways): extracts
    archives or takes direct files and uploads them to the appropriate
    Nextcloud folder.

    Returns a result dict with stored items or error.
    """
    file_ext = Path(file_path).suffix.lower()

    # For direct files (non-archives), upload straight to Nextcloud
    if file_ext not in (".zip", ".rar"):
        if content_type == "songs":
            return {
                "error": "Songs must be uploaded as .zip or .rar archives containing a song.ini"
            }

        local_path = Path(file_path)
        remote = await _upload_asset_to_nextcloud(local_path, content_type)

        # Clean up the local temp file
        try:
            local_path.unlink(missing_ok=True)
        except OSError:
            pass

        if remote:
            return {
                "message": f"File uploaded to Nextcloud: {remote}",
                "remote_path": remote,
            }
        else:
            return {"error": "Failed to upload file to Nextcloud"}

    # For archives, extract and process
    temp_dir = get_temp_staging_dir(prefix="ch_extract_")

    try:
        # Step 1: Extract the archive
        result = extract_archive(file_path, str(temp_dir))
        if "error" in result:
            return result

        # Step 2: Remove original archive
        try:
            Path(file_path).unlink()
        except OSError:
            pass

        # Step 3: Process based on content type
        if content_type == "songs":
            stored = await process_extracted_songs(str(temp_dir))
            if stored:
                return {
                    "message": f"✅ Successfully processed {len(stored)} song(s) and uploaded to Nextcloud",
                    "songs": stored,
                }
            else:
                return {
                    "error": "No valid songs found in the archive (missing or invalid song.ini), "
                    "or Nextcloud upload failed."
                }
        else:
            # For non-song content, upload all extracted files to Nextcloud
            uploaded_files: List[str] = []
            failed_files: List[str] = []

            for item in temp_dir.rglob("*"):
                if item.is_file():
                    remote = await _upload_asset_to_nextcloud(item, content_type)
                    if remote:
                        uploaded_files.append(remote)
                    else:
                        failed_files.append(item.name)

            if uploaded_files:
                msg = f"✅ Uploaded {len(uploaded_files)} file(s) to Nextcloud"
                if failed_files:
                    msg += f" ({len(failed_files)} failed)"
                return {
                    "message": msg,
                    "remote_paths": uploaded_files,
                    "failed": failed_files if failed_files else None,
                }
            else:
                return {"error": "No files could be uploaded to Nextcloud"}

    except Exception as e:
        logger.exception(f"❌ Error processing upload: {e}")
        return {"error": str(e)}

    finally:
        # Always clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Nextcloud library sync
# ---------------------------------------------------------------------------

# Module-level sync state so the UI can query what's happening
_sync_state: Dict[str, Any] = {
    "running": False,
    "phase": "",  # "idle", "discovering", "parsing", "purging", "done"
    "total_folders": 0,
    "processed": 0,
    "synced": 0,
    "failed": 0,
    "purged": 0,
    "current_folder": "",
    "last_synced_song": "",
    "errors": [],
}


def get_sync_state() -> Dict[str, Any]:
    """Return a snapshot of the current sync state."""
    return dict(_sync_state)


async def sync_library_from_nextcloud_stream():
    """
    Async generator version of library sync that yields progress dicts.

    Each yielded dict has a ``type`` key indicating the event kind:

    - ``start``        — sync has begun
    - ``discovering``  — scanning Nextcloud for song folders
    - ``discovered``   — folder scan complete, reports total found
    - ``parsing``      — about to parse a specific folder
    - ``parsed``       — successfully parsed & upserted a song
    - ``parse_error``  — failed to parse a folder
    - ``purging``      — about to purge stale DB entries
    - ``purged``       — purge complete
    - ``complete``     — final summary

    The UI can consume these via SSE to show real-time progress.
    """
    global _sync_state
    from src.database import purge_stale_songs, upsert_song
    from src.webdav import find_song_folders, is_configured, parse_remote_song_ini

    if not is_configured():
        yield {
            "type": "error",
            "message": "Nextcloud WebDAV is not configured. "
            "Set NEXTCLOUD_URL, NEXTCLOUD_USERNAME, and NEXTCLOUD_PASSWORD.",
        }
        return

    # Reset state
    _sync_state = {
        "running": True,
        "phase": "discovering",
        "total_folders": 0,
        "processed": 0,
        "synced": 0,
        "failed": 0,
        "purged": 0,
        "current_folder": "",
        "last_synced_song": "",
        "errors": [],
    }

    try:
        logger.info("🔄 Starting library sync from Nextcloud …")
        yield {"type": "start", "message": "Starting library sync from Nextcloud…"}

        # --- Step 1: Discover song folders ---
        _sync_state["phase"] = "discovering"
        yield {"type": "discovering", "message": "Scanning Nextcloud for song folders…"}

        try:
            folders = await find_song_folders()
        except Exception as e:
            logger.error("❌ Error discovering song folders: {}", e)
            _sync_state["phase"] = "idle"
            yield {"type": "error", "message": f"Failed to scan Nextcloud: {e}"}
            return

        total = len(folders)
        _sync_state["total_folders"] = total
        _sync_state["phase"] = "parsing"
        yield {
            "type": "discovered",
            "total": total,
            "message": f"Found {total} song folder{'s' if total != 1 else ''} on Nextcloud",
        }

        # --- Step 2: Parse each song.ini and upsert ---
        synced = 0
        failed = 0
        errors: List[str] = []

        for i, folder in enumerate(folders, 1):
            _sync_state["processed"] = i
            _sync_state["current_folder"] = folder

            # Extract a short display name from the folder path
            folder_display = (
                folder.rstrip("/").rsplit("/", 1)[-1] if "/" in folder else folder
            )

            yield {
                "type": "parsing",
                "index": i,
                "total": total,
                "folder": folder,
                "folder_display": folder_display,
                "message": f"Parsing {i}/{total}: {folder_display}",
            }

            try:
                parsed = await parse_remote_song_ini(folder)
                if not parsed:
                    failed += 1
                    err = f"Could not parse song.ini in {folder_display}"
                    errors.append(err)
                    _sync_state["failed"] = failed
                    _sync_state["errors"] = errors
                    yield {
                        "type": "parse_error",
                        "index": i,
                        "total": total,
                        "folder": folder,
                        "folder_display": folder_display,
                        "message": err,
                    }
                    continue

                metadata_json = json.dumps(parsed.get("metadata", {}))
                await upsert_song(
                    title=parsed["title"],
                    artist=parsed["artist"],
                    album=parsed["album"],
                    remote_path=folder,
                    metadata=metadata_json,
                )
                synced += 1
                _sync_state["synced"] = synced
                song_label = f"{parsed['artist']} — {parsed['title']}"
                _sync_state["last_synced_song"] = song_label

                yield {
                    "type": "parsed",
                    "index": i,
                    "total": total,
                    "folder": folder,
                    "title": parsed["title"],
                    "artist": parsed["artist"],
                    "album": parsed.get("album", ""),
                    "message": f"Synced {i}/{total}: {song_label}",
                }
            except Exception as e:
                failed += 1
                err = f"Error syncing {folder_display}: {str(e)}"
                errors.append(err)
                _sync_state["failed"] = failed
                _sync_state["errors"] = errors
                logger.error("❌ Error syncing {}: {}", folder, e)
                yield {
                    "type": "parse_error",
                    "index": i,
                    "total": total,
                    "folder": folder,
                    "folder_display": folder_display,
                    "message": err,
                }

        # --- Step 3: Purge stale entries ---
        _sync_state["phase"] = "purging"
        yield {"type": "purging", "message": "Removing songs no longer on Nextcloud…"}

        purged = await purge_stale_songs(folders)
        _sync_state["purged"] = purged

        if purged > 0:
            yield {
                "type": "purged",
                "purged": purged,
                "message": f"Removed {purged} stale song{'s' if purged != 1 else ''} from library",
            }

        # --- Done ---
        _sync_state["phase"] = "done"

        summary = {
            "type": "complete",
            "synced": synced,
            "purged": purged,
            "failed": failed,
            "total_on_nextcloud": total,
            "message": f"Library sync complete: {synced} synced, {purged} removed, {failed} failed",
        }
        if errors:
            summary["errors"] = errors

        logger.info(
            "🔄 Library sync complete: {} synced, {} purged, {} failed",
            synced,
            purged,
            failed,
        )
        yield summary
    finally:
        # Always reset the running flag, even if the SSE client disconnects
        # mid-stream (GeneratorExit) or an unhandled exception occurs.
        # Without this, a dropped connection permanently locks out all
        # future sync attempts.
        _sync_state["running"] = False
        if _sync_state["phase"] != "done":
            logger.warning(
                "⚠️  Library sync interrupted (phase was '{}') — "
                "resetting state so a new sync can start",
                _sync_state["phase"],
            )
            _sync_state["phase"] = "idle"


async def sync_library_from_nextcloud() -> Dict[str, Any]:
    """
    Scan the Nextcloud songs directory, parse every song.ini found,
    and upsert the metadata into the local database.

    Songs that exist in the DB but are no longer on Nextcloud are purged.

    Returns a summary dict with counts.

    This is the non-streaming wrapper around
    :func:`sync_library_from_nextcloud_stream`.  It consumes all progress
    events and returns only the final summary.
    """
    last_event: Dict[str, Any] = {}
    async for event in sync_library_from_nextcloud_stream():
        last_event = event

    if not last_event:
        return {"error": "Sync produced no events"}

    # The generator's final event is either "complete" or "error"
    if last_event.get("type") == "error":
        return {"error": last_event.get("message", "Unknown sync error")}

    return {
        "message": last_event.get("message", "Sync complete"),
        "synced": last_event.get("synced", 0),
        "purged": last_event.get("purged", 0),
        "failed": last_event.get("failed", 0),
        "total_on_nextcloud": last_event.get("total_on_nextcloud", 0),
        "errors": last_event.get("errors", []),
    }


async def delete_song_from_nextcloud(remote_path: str) -> bool:
    """
    Delete a song's folder from Nextcloud.
    Returns True if the remote folder was successfully removed.
    """
    from src.webdav import delete_remote, is_configured

    if not is_configured():
        logger.error("❌ Cannot delete from Nextcloud — not configured")
        return False

    if not remote_path or remote_path == "/":
        logger.error("❌ Refusing to delete root path")
        return False

    # Safety: only delete paths under the songs root
    songs_root = NEXTCLOUD_SONGS_PATH.rstrip("/")
    if not remote_path.startswith(songs_root):
        logger.error(
            "❌ Refusing to delete path outside songs directory: {}", remote_path
        )
        return False

    ok = await delete_remote(remote_path)
    if ok:
        logger.info("🗑️ Deleted song folder from Nextcloud: {}", remote_path)
    else:
        logger.error("❌ Failed to delete from Nextcloud: {}", remote_path)
    return ok


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def _sanitize_filename(name: str) -> str:
    """
    Sanitize a string for use as a filename/directory name.
    Removes or replaces characters that are problematic on most file systems.
    """
    replacements = {
        "/": "_",
        "\\": "_",
        ":": "-",
        "*": "",
        "?": "",
        '"': "",
        "<": "",
        ">": "",
        "|": "",
    }
    result = name
    for old, new in replacements.items():
        result = result.replace(old, new)

    # Strip leading/trailing whitespace and dots
    result = result.strip(" .")

    # Fallback for empty result
    if not result:
        result = "unknown"

    return result
