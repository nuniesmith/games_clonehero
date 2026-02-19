"""
Clone Hero Content Manager - Content Management Service

Handles:
- Parsing song.ini files to extract metadata
- Extracting ZIP/RAR archives
- Storing songs on disk and registering them in the database
- Moving/organizing content into the correct directory structure
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

from src.config import CONTENT_DIR, CONTENT_FOLDERS, OPTIONAL_SONG_FIELDS
from src.database import insert_song_sync


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------
def get_content_directory(content_type: str) -> Path:
    """Return the content directory for a given type, creating it if needed."""
    subfolder = CONTENT_FOLDERS.get(content_type, content_type)
    directory = CONTENT_DIR / subfolder
    directory.mkdir(parents=True, exist_ok=True)
    return directory


# ---------------------------------------------------------------------------
# Song.ini parsing
# ---------------------------------------------------------------------------
def parse_song_ini(ini_path: Path) -> Optional[Dict[str, Any]]:
    """
    Parse a song.ini file and return structured metadata.

    Returns None if the file is invalid or missing required fields.
    Required fields: name, artist, album.
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
# Content processing pipeline
# ---------------------------------------------------------------------------
def process_extracted_songs(extract_dir: str) -> List[Dict[str, Any]]:
    """
    Walk an extracted directory, find all song.ini files, parse them,
    move the song folders to the content directory, and register in the DB.

    Returns a list of successfully stored song info dicts.
    """
    stored_songs: List[Dict[str, Any]] = []
    extract_path = Path(extract_dir)
    songs_dir = get_content_directory("songs")

    for ini_path in extract_path.rglob("song.ini"):
        parsed = parse_song_ini(ini_path)
        if not parsed:
            continue

        title = parsed["title"]
        artist = parsed["artist"]
        album = parsed["album"]
        metadata = parsed["metadata"]

        # Build the destination directory: songs/<Artist>/<Title>_<short_uuid>
        safe_artist = _sanitize_filename(artist)
        safe_title = _sanitize_filename(title)
        unique_suffix = uuid.uuid4().hex[:8]
        dest_dir = songs_dir / safe_artist / f"{safe_title}_{unique_suffix}"

        try:
            # Move the entire song folder (parent of song.ini)
            song_source = ini_path.parent
            shutil.move(str(song_source), str(dest_dir))
            logger.info(f"📁 Moved song to {dest_dir}")

            # Register in database
            metadata_json = json.dumps(metadata)
            song_id = insert_song_sync(
                title=title,
                artist=artist,
                album=album,
                file_path=str(dest_dir),
                metadata=metadata_json,
            )

            if song_id and song_id > 0:
                stored_songs.append(
                    {
                        "id": song_id,
                        "title": title,
                        "artist": artist,
                        "album": album,
                        "folder_path": str(dest_dir),
                        "metadata": metadata,
                    }
                )

        except Exception as e:
            logger.error(f"❌ Error storing song '{title}': {e}")

    return stored_songs


def process_upload(file_path: str, content_type: str = "songs") -> Dict[str, Any]:
    """
    Full pipeline: extract archive -> process songs -> cleanup.

    For non-song content types (backgrounds, colors, highways), files are
    simply moved to the appropriate content directory.

    Returns a result dict with stored items or error.
    """
    file_ext = Path(file_path).suffix.lower()

    # For direct files (non-archives), just move them
    if file_ext not in (".zip", ".rar"):
        if content_type == "songs":
            return {
                "error": "Songs must be uploaded as .zip or .rar archives containing a song.ini"
            }

        final_dir = get_content_directory(content_type)
        file_name = Path(file_path).name
        dest = final_dir / file_name

        try:
            shutil.move(file_path, str(dest))
            logger.info(f"✅ Stored {content_type} file: {dest}")
            return {"message": f"File stored: {file_name}", "file": str(dest)}
        except Exception as e:
            logger.error(f"❌ Error moving file to {dest}: {e}")
            return {"error": str(e)}

    # For archives, extract and process
    temp_dir = Path(tempfile.mkdtemp(prefix="ch_extract_"))

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
            stored = process_extracted_songs(str(temp_dir))
            if stored:
                return {
                    "message": f"✅ Successfully processed {len(stored)} song(s)",
                    "songs": stored,
                }
            else:
                return {
                    "error": "No valid songs found in the archive (missing or invalid song.ini)"
                }
        else:
            # For non-song content, move all extracted files to the content dir
            final_dir = get_content_directory(content_type)
            moved_files = []
            for item in temp_dir.rglob("*"):
                if item.is_file():
                    dest = final_dir / item.name
                    shutil.move(str(item), str(dest))
                    moved_files.append(str(dest))

            return {
                "message": f"✅ Stored {len(moved_files)} file(s)",
                "files": moved_files,
            }

    except Exception as e:
        logger.exception(f"❌ Error processing upload: {e}")
        return {"error": str(e)}

    finally:
        # Always clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def scan_local_songs() -> List[Dict[str, Any]]:
    """
    Scan the local songs directory for song.ini files and return parsed data.
    Does NOT insert into the database — useful for discovery/review.
    """
    songs_dir = get_content_directory("songs")
    results: List[Dict[str, Any]] = []

    for ini_path in songs_dir.rglob("song.ini"):
        parsed = parse_song_ini(ini_path)
        if parsed:
            parsed["folder_path"] = str(ini_path.parent)
            parsed["ini_path"] = str(ini_path)
            results.append(parsed)

    logger.info(f"🔍 Scanned {len(results)} local songs")
    return results


def delete_song_files(folder_path: str) -> bool:
    """
    Delete a song's folder from disk.
    Returns True if the folder was successfully removed.
    """
    path = Path(folder_path)
    if not path.exists():
        logger.warning(f"⚠️ Song folder not found: {folder_path}")
        return False

    # Safety check: make sure it's inside the content directory
    try:
        path.resolve().relative_to(CONTENT_DIR.resolve())
    except ValueError:
        logger.error(f"❌ Refusing to delete path outside content dir: {folder_path}")
        return False

    try:
        shutil.rmtree(str(path))
        logger.info(f"🗑️ Deleted song folder: {folder_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Error deleting {folder_path}: {e}")
        return False


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def _sanitize_filename(name: str) -> str:
    """
    Sanitize a string for use as a filename/directory name.
    Removes or replaces characters that are problematic on most file systems.
    """
    # Replace problematic characters
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
