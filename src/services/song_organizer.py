"""
Clone Hero Content Manager - Song Organizer Service

Provides song library organization utilities for the web application,
extracted from scripts/organize_clonehero.py.  Handles:

- Scanning Nextcloud for song folders and validating structure
- Organizing songs into a clean Artist/Song/ directory layout
- Detecting and cleaning up duplicate songs
- Fetching missing album art from MusicBrainz / Cover Art Archive
- Sanitizing filenames for cross-platform compatibility

All operations work against Nextcloud via WebDAV.  Local disk is only
used as a transient staging area when needed.

Key entry points:
- ``organize_song_folder()``       — move/rename a single song to proper structure
- ``scan_library_issues()``        — scan library for organizational problems
- ``find_duplicates_on_nextcloud()`` — detect duplicate songs
- ``cleanup_duplicates()``         — remove lower-quality duplicate copies
- ``fetch_missing_album_art()``    — download album art from MusicBrainz
- ``organize_library_stream()``    — full library organize with SSE progress
"""

import configparser
import hashlib
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UNKNOWN_ARTIST = "_Unknown Artist"
UNKNOWN_SONG = "_Unknown Song"
MAX_FILENAME_LEN = 120

# Map loose files to their standard name (case-insensitive matching).
RENAME_MAP = {
    "album.png": "album.png",
    "album.jpg": "album.jpg",
    "album.jpeg": "album.jpg",
    "background.png": "background.png",
    "background.jpg": "background.jpg",
    "background.jpeg": "background.jpg",
    "notes.chart": "notes.chart",
    "notes.mid": "notes.mid",
    "song.ogg": "song.ogg",
    "song.mp3": "song.mp3",
    "song.opus": "song.opus",
    "lyrics.txt": "Lyrics.txt",
    "song.ini": "song.ini",
}

# MusicBrainz / Cover Art Archive
MB_BASE = "https://musicbrainz.org/ws/2"
CAA_BASE = "https://coverartarchive.org"
MB_USER_AGENT = "CloneHeroManager/1.0 (https://github.com/nuniesmith/games_clonehero)"
_last_mb_request: float = 0.0


# ---------------------------------------------------------------------------
# Filename / path helpers
# ---------------------------------------------------------------------------


def sanitize_filename(name: str, max_len: int = MAX_FILENAME_LEN) -> str:
    """
    Remove characters that are invalid in Windows filenames and truncate.

    This ensures safe directory and file names across all platforms.
    """
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", name)
    sanitized = sanitized.strip(". ")
    if not sanitized:
        return "_"
    if len(sanitized) > max_len:
        sanitized = sanitized[:max_len].rstrip(". ")
    return sanitized


def normalize_for_matching(s: str) -> str:
    """
    Normalize a string for fuzzy matching — lowercase, strip punctuation,
    remove parentheticals like (Remaster) or [charter name].
    """
    s = s.lower().strip()
    s = re.sub(r"\s*\(.*?\)\s*", " ", s)  # remove parenthetical
    s = re.sub(r"\s*\[.*?\]\s*", " ", s)  # remove brackets
    s = re.sub(r"[^a-z0-9\s]", "", s)  # strip punctuation
    s = re.sub(r"\s+", " ", s).strip()  # collapse whitespace
    return s


def get_organized_path(
    artist: str,
    song_name: str,
    base_path: str = "/Songs",
) -> str:
    """
    Compute the canonical organized remote path for a song.

    Returns a path like ``/Songs/Artist Name/Song Title``.
    """
    safe_artist = sanitize_filename(artist) if artist else UNKNOWN_ARTIST
    safe_song = sanitize_filename(song_name) if song_name else UNKNOWN_SONG
    return f"{base_path.rstrip('/')}/{safe_artist}/{safe_song}"


# ---------------------------------------------------------------------------
# Song.ini parsing (from text — no filesystem access needed)
# ---------------------------------------------------------------------------


def parse_song_ini_text(ini_text: str) -> Dict[str, str]:
    """
    Parse song.ini content and return a metadata dict.

    Returns keys: artist, name, charter, album (all strings, possibly empty).
    """
    meta = {"artist": "", "name": "", "charter": "", "album": ""}

    try:
        config = configparser.ConfigParser(strict=False, interpolation=None)
        config.read_string(ini_text)

        section = None
        for s in config.sections():
            if s.lower() == "song":
                section = s
                break

        if section:
            meta["artist"] = config.get(section, "artist", fallback="").strip()
            meta["name"] = config.get(section, "name", fallback="").strip()
            meta["charter"] = config.get(section, "charter", fallback="").strip()
            meta["album"] = config.get(section, "album", fallback="").strip()
    except Exception as e:
        logger.warning("Failed to parse song.ini text: {}", e)

    return meta


def extract_artist_song_from_dirname(dirname: str) -> Tuple[str, str]:
    """
    Try to extract artist and song name from a directory name.

    Common patterns:
    - ``Artist - Song Title``
    - ``Artist_-_Song Title``
    - Just ``Song Title`` (artist unknown)
    """
    if " - " in dirname:
        parts = dirname.split(" - ", 1)
        return parts[0].strip(), parts[1].strip()

    if "_-_" in dirname:
        parts = dirname.split("_-_", 1)
        return parts[0].strip().replace("_", " "), parts[1].strip().replace("_", " ")

    return "", dirname.strip()


# ---------------------------------------------------------------------------
# Song validation (quick structural checks)
# ---------------------------------------------------------------------------


class SongIssue:
    """Represents a structural issue with a song folder."""

    CRITICAL = "critical"
    COSMETIC = "cosmetic"

    def __init__(self, severity: str, message: str):
        self.severity = severity
        self.message = message

    def to_dict(self) -> Dict[str, str]:
        return {"severity": self.severity, "message": self.message}


def validate_song_files(file_list: List[str]) -> Tuple[List[str], List[str]]:
    """
    Check a list of filenames for missing critical/cosmetic files.

    Returns ``(critical_issues, cosmetic_issues)`` where:
    - Critical = song is unplayable in Clone Hero
    - Cosmetic = works but missing nice-to-haves

    Parameters
    ----------
    file_list : list[str]
        Filenames in the song folder (not full paths).
    """
    critical: List[str] = []
    cosmetic: List[str] = []
    lower_files = {f.lower() for f in file_list}

    if "song.ini" not in lower_files:
        critical.append("Missing song.ini")
    if "notes.chart" not in lower_files and "notes.mid" not in lower_files:
        critical.append("Missing notes.chart / notes.mid")
    if not (lower_files & {"song.ogg", "song.mp3", "song.opus"}):
        critical.append("Missing audio (song.ogg/mp3/opus)")
    if not (lower_files & {"album.png", "album.jpg"}):
        cosmetic.append("Missing album art (album.png/jpg)")
    if not (lower_files & {"background.png", "background.jpg"}):
        cosmetic.append("Missing background (background.png/jpg)")

    return critical, cosmetic


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------


def compute_chart_hash(chart_bytes: bytes) -> str:
    """
    Compute a truncated SHA-256 hash of chart file content.

    Used for detecting identical charts across different folder names.
    """
    h = hashlib.sha256(chart_bytes)
    return h.hexdigest()[:16]


def score_song_files(
    file_list: List[str], file_sizes: Optional[Dict[str, int]] = None
) -> int:
    """
    Score a song folder by completeness.  Higher = better quality copy to keep.

    Parameters
    ----------
    file_list : list[str]
        Filenames in the song folder.
    file_sizes : dict, optional
        Mapping of filename -> size in bytes.  If provided, larger audio
        files contribute more to the score.
    """
    score = 0
    lower_files = {f.lower() for f in file_list}

    # Critical files
    if lower_files & {"notes.chart", "notes.mid"}:
        score += 10
    if lower_files & {"song.ogg", "song.mp3", "song.opus"}:
        score += 10
        # Prefer larger audio (likely higher quality)
        if file_sizes:
            for ext in ("song.ogg", "song.mp3", "song.opus"):
                if ext in lower_files and ext in file_sizes:
                    size_mb = file_sizes[ext] / (1024 * 1024)
                    score += min(int(size_mb), 5)
                    break
    if "song.ini" in lower_files:
        score += 5
    # Cosmetic files
    if lower_files & {"album.png", "album.jpg"}:
        score += 3
    if lower_files & {"background.png", "background.jpg"}:
        score += 2
    if "lyrics.txt" in lower_files:
        score += 1
    # More total files = more complete chart
    score += min(len(file_list), 5)

    return score


class DuplicateGroup:
    """A group of songs identified as duplicates of each other."""

    def __init__(self, key: str):
        self.key = key
        self.entries: List[Dict[str, Any]] = []

    def add(
        self,
        remote_path: str,
        song_id: Optional[int],
        score: int,
        chart_hash: Optional[str] = None,
        title: str = "",
        artist: str = "",
    ) -> None:
        self.entries.append(
            {
                "remote_path": remote_path,
                "song_id": song_id,
                "score": score,
                "chart_hash": chart_hash,
                "title": title,
                "artist": artist,
            }
        )

    @property
    def keep(self) -> Dict[str, Any]:
        """The highest-scored entry (the one to keep)."""
        return sorted(self.entries, key=lambda e: -e["score"])[0]

    @property
    def duplicates(self) -> List[Dict[str, Any]]:
        """All entries except the one to keep."""
        sorted_entries = sorted(self.entries, key=lambda e: -e["score"])
        return sorted_entries[1:]

    @property
    def count(self) -> int:
        return len(self.entries)

    @property
    def duplicate_count(self) -> int:
        return max(0, len(self.entries) - 1)

    def to_dict(self) -> Dict[str, Any]:
        sorted_entries = sorted(self.entries, key=lambda e: -e["score"])
        return {
            "key": self.key,
            "keep": sorted_entries[0] if sorted_entries else None,
            "duplicates": sorted_entries[1:],
            "count": self.count,
        }


def find_duplicates_from_metadata(
    songs: List[Dict[str, Any]],
) -> List[DuplicateGroup]:
    """
    Detect duplicate songs by normalized artist+title matching.

    Parameters
    ----------
    songs : list[dict]
        Song records from the database, each with at least 'id', 'title',
        'artist', and 'remote_path' keys.

    Returns
    -------
    list[DuplicateGroup]
        Groups with 2+ entries (actual duplicates).
    """
    by_name: Dict[str, List[Dict[str, Any]]] = {}

    for song in songs:
        artist = song.get("artist", "").strip()
        title = song.get("title", "").strip()

        if not artist and not title:
            remote = song.get("remote_path", "")
            dirname = remote.rstrip("/").rsplit("/", 1)[-1] if "/" in remote else remote
            artist, title = extract_artist_song_from_dirname(dirname)

        key = f"{normalize_for_matching(artist)}|{normalize_for_matching(title)}"
        by_name.setdefault(key, []).append(song)

    groups: List[DuplicateGroup] = []
    for key, entries in by_name.items():
        if len(entries) < 2:
            continue
        group = DuplicateGroup(key)
        for song in entries:
            group.add(
                remote_path=song.get("remote_path", ""),
                song_id=song.get("id"),
                score=0,  # Will be scored later with file info
                title=song.get("title", ""),
                artist=song.get("artist", ""),
            )
        groups.append(group)

    return groups


# ---------------------------------------------------------------------------
# MusicBrainz / Cover Art Archive helpers
# ---------------------------------------------------------------------------


def _mb_request(url: str) -> Optional[dict]:
    """Make a rate-limited request to MusicBrainz API (max 1 req/sec)."""
    global _last_mb_request
    elapsed = time.monotonic() - _last_mb_request
    if elapsed < 1.1:
        time.sleep(1.1 - elapsed)

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": MB_USER_AGENT,
            "Accept": "application/json",
        },
    )
    try:
        _last_mb_request = time.monotonic()
        with urllib.request.urlopen(req, timeout=10) as resp:
            import json

            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        logger.debug("MusicBrainz request failed: {}", e)
        return None


def _search_release(artist: str, song: str, album: str = "") -> Optional[str]:
    """Search MusicBrainz for a release MBID.  Returns MBID or None."""
    # Strategy 1: artist + album
    if album:
        query = f'artist:"{artist}" AND release:"{album}"'
        encoded = urllib.parse.quote(query)
        url = f"{MB_BASE}/release/?query={encoded}&limit=5&fmt=json"
        data = _mb_request(url)
        if data and data.get("releases"):
            mbid = data["releases"][0].get("id")
            score = data["releases"][0].get("score", 0)
            if mbid and score >= 80:
                logger.debug("Found release by album: {} (score={})", mbid, score)
                return mbid

    # Strategy 2: artist + song title -> recording -> release
    query = f'artist:"{artist}" AND recording:"{song}"'
    encoded = urllib.parse.quote(query)
    url = f"{MB_BASE}/recording/?query={encoded}&limit=5&fmt=json"
    data = _mb_request(url)
    if data and data.get("recordings"):
        for recording in data["recordings"]:
            score = recording.get("score", 0)
            if score < 70:
                continue
            for release in recording.get("releases", []):
                mbid = release.get("id")
                if mbid:
                    logger.debug(
                        "Found release via recording: {} (score={})", mbid, score
                    )
                    return mbid

    # Strategy 3: broad release search
    query = f'artist:"{artist}" AND release:"{song}"'
    encoded = urllib.parse.quote(query)
    url = f"{MB_BASE}/release/?query={encoded}&limit=3&fmt=json"
    data = _mb_request(url)
    if data and data.get("releases"):
        mbid = data["releases"][0].get("id")
        score = data["releases"][0].get("score", 0)
        if mbid and score >= 70:
            logger.debug("Found release by song title: {} (score={})", mbid, score)
            return mbid

    return None


def _download_cover_bytes(mbid: str) -> Optional[bytes]:
    """Download front cover from Cover Art Archive.  Returns image bytes or None."""
    url = f"{CAA_BASE}/release/{mbid}/front-500"
    req = urllib.request.Request(url, headers={"User-Agent": MB_USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            img_data = resp.read()
            if len(img_data) < 1000:
                logger.debug("Cover art too small, likely not valid")
                return None
            return img_data
    except urllib.error.HTTPError as e:
        if e.code == 404:
            logger.debug("No cover art available for release {}", mbid)
        else:
            logger.debug("Cover art download failed: {}", e)
        return None
    except (urllib.error.URLError, TimeoutError) as e:
        logger.debug("Cover art download failed: {}", e)
        return None


def fetch_album_art_bytes(artist: str, song: str, album: str = "") -> Optional[bytes]:
    """
    Search MusicBrainz for album art and return the image bytes.

    Returns None if no art is found.  This is the synchronous core that
    can be called from an async wrapper via ``asyncio.to_thread()``.
    """
    if not artist or not song:
        return None

    mbid = _search_release(artist, song, album)
    if not mbid:
        logger.debug("No MusicBrainz release found for: {} - {}", artist, song)
        return None

    img = _download_cover_bytes(mbid)
    if img:
        logger.info("Downloaded album art for: {} - {}", artist, song)
    return img


# ---------------------------------------------------------------------------
# Async Nextcloud integration
# ---------------------------------------------------------------------------


async def organize_song_folder(
    song_id: int,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Move a song to the correct ``Artist/Song Title`` path on Nextcloud.

    If the song is already at the correct path, this is a no-op.

    Parameters
    ----------
    song_id : int
        Database ID of the song.
    dry_run : bool
        If True, report what would change without modifying anything.

    Returns
    -------
    dict
        Result with keys: moved (bool), old_path, new_path, message.
    """
    from src.config import NEXTCLOUD_SONGS_PATH
    from src.database import get_song_by_id, update_song
    from src.webdav import is_configured, mkdir, move_remote

    if not is_configured():
        return {"error": "Nextcloud WebDAV is not configured"}

    song = await get_song_by_id(song_id)
    if not song:
        return {"error": f"Song {song_id} not found"}

    artist = song.get("artist", "").strip()
    title = song.get("title", "").strip()
    current_path = song.get("remote_path", "").rstrip("/")

    if not current_path:
        return {"error": "Song has no remote path"}

    # Compute the canonical path
    target_path = get_organized_path(artist, title, NEXTCLOUD_SONGS_PATH)

    if current_path.rstrip("/") == target_path.rstrip("/"):
        return {
            "moved": False,
            "old_path": current_path,
            "new_path": target_path,
            "message": "Song is already at the correct path",
        }

    if dry_run:
        return {
            "moved": False,
            "dry_run": True,
            "old_path": current_path,
            "new_path": target_path,
            "message": f"Would move: {current_path} → {target_path}",
        }

    # Create the artist directory
    artist_dir = f"{NEXTCLOUD_SONGS_PATH.rstrip('/')}/{sanitize_filename(artist) if artist else UNKNOWN_ARTIST}"
    try:
        await mkdir(artist_dir)
    except Exception as e:
        logger.warning("Could not create artist dir {}: {}", artist_dir, e)

    # Move the folder
    try:
        ok = await move_remote(current_path, target_path)
        if not ok:
            return {
                "error": f"Failed to move {current_path} → {target_path}",
                "old_path": current_path,
                "new_path": target_path,
            }
    except Exception as e:
        return {
            "error": f"Move failed: {e}",
            "old_path": current_path,
            "new_path": target_path,
        }

    # Update the database
    try:
        await update_song(song_id, remote_path=target_path)
    except Exception as e:
        logger.error("Failed to update DB after move: {}", e)

    logger.info("📁 Organized song {}: {} → {}", song_id, current_path, target_path)

    return {
        "moved": True,
        "old_path": current_path,
        "new_path": target_path,
        "message": f"Moved to {target_path}",
    }


async def scan_library_issues() -> Dict[str, Any]:
    """
    Scan the entire Nextcloud song library for organizational issues.

    Checks for:
    - Songs not in Artist/Title structure
    - Missing critical files (song.ini, notes.chart, audio)
    - Missing cosmetic files (album art, background)
    - Potential duplicates

    Returns a summary dict with categorized issues.
    """
    from src.config import NEXTCLOUD_SONGS_PATH
    from src.database import get_songs
    from src.webdav import is_configured, list_song_folder_files

    if not is_configured():
        return {"error": "Nextcloud WebDAV is not configured"}

    songs = await get_songs(limit=10000)
    total = len(songs)

    misplaced: List[Dict[str, Any]] = []
    missing_files: List[Dict[str, Any]] = []
    missing_art: List[Dict[str, Any]] = []
    healthy: int = 0

    for song in songs:
        song_id = song.get("id")
        artist = song.get("artist", "").strip()
        title = song.get("title", "").strip()
        current_path = song.get("remote_path", "").rstrip("/")

        # Check path structure
        expected = get_organized_path(artist, title, NEXTCLOUD_SONGS_PATH)
        if current_path != expected.rstrip("/"):
            misplaced.append(
                {
                    "song_id": song_id,
                    "title": title,
                    "artist": artist,
                    "current_path": current_path,
                    "expected_path": expected,
                }
            )

        # Check files
        try:
            items = await list_song_folder_files(current_path)
            if items:
                file_names = [item.name for item in items]
                critical, cosmetic = validate_song_files(file_names)
                if critical:
                    missing_files.append(
                        {
                            "song_id": song_id,
                            "title": title,
                            "artist": artist,
                            "issues": critical,
                        }
                    )
                elif cosmetic:
                    lower_files = {f.lower() for f in file_names}
                    if not (lower_files & {"album.png", "album.jpg"}):
                        missing_art.append(
                            {
                                "song_id": song_id,
                                "title": title,
                                "artist": artist,
                            }
                        )
                    healthy += 1
                else:
                    healthy += 1
            else:
                missing_files.append(
                    {
                        "song_id": song_id,
                        "title": title,
                        "artist": artist,
                        "issues": ["Could not list folder contents"],
                    }
                )
        except Exception as e:
            logger.warning("Error scanning song {}: {}", song_id, e)

    # Check for duplicates
    dup_groups = find_duplicates_from_metadata(songs)
    total_dupes = sum(g.duplicate_count for g in dup_groups)

    return {
        "total_songs": total,
        "healthy": healthy,
        "misplaced": misplaced,
        "misplaced_count": len(misplaced),
        "missing_files": missing_files,
        "missing_files_count": len(missing_files),
        "missing_art": missing_art,
        "missing_art_count": len(missing_art),
        "duplicate_groups": [g.to_dict() for g in dup_groups],
        "duplicate_count": total_dupes,
    }


async def find_duplicates_on_nextcloud() -> Dict[str, Any]:
    """
    Scan the Nextcloud library for duplicate songs.

    Uses name-based matching and (when feasible) chart hash comparison
    to detect duplicates.

    Returns a dict with duplicate groups and summary statistics.
    """
    from src.database import get_songs
    from src.webdav import download_file, is_configured, list_song_folder_files

    if not is_configured():
        return {"error": "Nextcloud WebDAV is not configured"}

    songs = await get_songs(limit=10000)
    groups = find_duplicates_from_metadata(songs)

    # Enrich with file-level scoring and chart hashes
    for group in groups:
        for entry in group.entries:
            remote_path = entry["remote_path"]
            try:
                items = await list_song_folder_files(remote_path)
                if items:
                    file_names = [item.name for item in items]
                    # Build approximate file sizes from WebDAVItem
                    file_sizes = {item.name.lower(): item.size for item in items}
                    entry["score"] = score_song_files(file_names, file_sizes)
                    entry["files"] = file_names

                    # Try to get chart hash for exact-match detection
                    chart_remote = f"{remote_path.rstrip('/')}/notes.chart"
                    try:
                        chart_bytes = await download_file(chart_remote)
                        if chart_bytes:
                            entry["chart_hash"] = compute_chart_hash(chart_bytes)
                    except Exception:
                        pass
                else:
                    entry["score"] = 0
            except Exception:
                entry["score"] = 0

    total_dupes = sum(g.duplicate_count for g in groups)

    return {
        "groups": [g.to_dict() for g in groups],
        "group_count": len(groups),
        "duplicate_count": total_dupes,
        "total_songs_scanned": len(songs),
    }


async def cleanup_duplicates(
    group_keys: Optional[List[str]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Remove duplicate songs, keeping the highest-scored copy in each group.

    Parameters
    ----------
    group_keys : list[str], optional
        Specific duplicate group keys to clean.  If None, cleans all.
    dry_run : bool
        If True, report what would be deleted without actually deleting.

    Returns
    -------
    dict
        Summary with deleted count and details.
    """
    from src.database import delete_song
    from src.webdav import delete_remote, is_configured

    if not is_configured():
        return {"error": "Nextcloud WebDAV is not configured"}

    dup_result = await find_duplicates_on_nextcloud()
    if "error" in dup_result:
        return dup_result

    groups_data = dup_result.get("groups", [])
    deleted: List[Dict[str, Any]] = []
    errors: List[str] = []

    for group_data in groups_data:
        key = group_data.get("key", "")
        if group_keys is not None and key not in group_keys:
            continue

        dupes = group_data.get("duplicates", [])
        keep = group_data.get("keep")

        for dupe in dupes:
            remote_path = dupe.get("remote_path", "")
            song_id = dupe.get("song_id")

            if dry_run:
                deleted.append(
                    {
                        "song_id": song_id,
                        "remote_path": remote_path,
                        "dry_run": True,
                    }
                )
                continue

            try:
                # Delete from Nextcloud
                if remote_path:
                    ok = await delete_remote(remote_path)
                    if not ok:
                        errors.append(f"Failed to delete {remote_path}")
                        continue

                # Delete from database
                if song_id:
                    await delete_song(song_id)

                deleted.append(
                    {
                        "song_id": song_id,
                        "remote_path": remote_path,
                    }
                )
                logger.info(
                    "🗑️  Deleted duplicate: {} (kept {})",
                    remote_path,
                    keep.get("remote_path", "?") if keep else "?",
                )
            except Exception as e:
                errors.append(f"Error deleting {remote_path}: {e}")

    action = "Would delete" if dry_run else "Deleted"
    return {
        "message": f"{action} {len(deleted)} duplicate(s)",
        "deleted": deleted,
        "deleted_count": len(deleted),
        "errors": errors,
        "dry_run": dry_run,
    }


async def fetch_missing_art_for_song(song_id: int) -> Dict[str, Any]:
    """
    Attempt to download album art for a single song from MusicBrainz.

    If art is found, uploads it to the song's folder on Nextcloud.

    Returns a result dict.
    """
    import asyncio

    from src.database import get_song_by_id
    from src.webdav import is_configured, list_song_folder_files, upload_file

    if not is_configured():
        return {"error": "Nextcloud WebDAV is not configured"}

    song = await get_song_by_id(song_id)
    if not song:
        return {"error": f"Song {song_id} not found"}

    remote_path = song.get("remote_path", "")
    artist = song.get("artist", "").strip()
    title = song.get("title", "").strip()
    album = ""

    # Check metadata for album name
    metadata = song.get("metadata")
    if metadata:
        if isinstance(metadata, str):
            import json

            try:
                metadata = json.loads(metadata)
            except (json.JSONDecodeError, TypeError):
                metadata = {}
        if isinstance(metadata, dict):
            album = metadata.get("album", "")

    # Check if album art already exists
    try:
        items = await list_song_folder_files(remote_path)
        if items:
            lower_files = {item.name.lower() for item in items}
            if lower_files & {"album.png", "album.jpg"}:
                return {
                    "found": True,
                    "already_exists": True,
                    "message": "Album art already exists",
                }
    except Exception:
        pass

    if not artist or not title:
        return {"found": False, "message": "Missing artist or title for lookup"}

    # Fetch art (blocking I/O — run in thread)
    img_bytes = await asyncio.to_thread(fetch_album_art_bytes, artist, title, album)

    if not img_bytes:
        return {
            "found": False,
            "message": f"No album art found for {artist} - {title}",
        }

    # Upload to Nextcloud
    art_remote = f"{remote_path.rstrip('/')}/album.jpg"
    try:
        ok = await upload_file(art_remote, img_bytes)
        if ok:
            logger.info(
                "🎨 Uploaded album art for {} - {} ({} bytes)",
                artist,
                title,
                len(img_bytes),
            )
            return {
                "found": True,
                "uploaded": True,
                "remote_path": art_remote,
                "size": len(img_bytes),
                "message": f"Downloaded and uploaded album art ({len(img_bytes)} bytes)",
            }
        else:
            return {"error": f"Failed to upload album art to {art_remote}"}
    except Exception as e:
        return {"error": f"Upload error: {e}"}


async def organize_library_stream(
    dry_run: bool = False,
    fix_paths: bool = True,
    fetch_art: bool = False,
    clean_dupes: bool = False,
):
    """
    Async generator that organizes the full library with streaming progress.

    Yields dicts suitable for SSE consumption.

    Parameters
    ----------
    dry_run : bool
        Preview changes without modifying anything.
    fix_paths : bool
        Move songs to correct Artist/Title structure.
    fetch_art : bool
        Download missing album art.
    clean_dupes : bool
        Remove duplicate songs.
    """
    from src.database import get_songs
    from src.webdav import is_configured

    if not is_configured():
        yield {"type": "error", "message": "Nextcloud WebDAV is not configured"}
        return

    songs = await get_songs(limit=10000)
    total = len(songs)

    yield {
        "type": "start",
        "total": total,
        "dry_run": dry_run,
        "message": f"Starting library organization ({total} songs)…",
    }

    organized = 0
    art_found = 0
    art_missing = 0
    dupes_removed = 0
    errors: List[str] = []

    # Step 1: Organize paths
    if fix_paths:
        yield {
            "type": "phase",
            "phase": "organizing",
            "message": "Organizing song paths…",
        }

        for i, song in enumerate(songs, 1):
            song_id = song.get("id")
            title = song.get("title", "Unknown")
            artist = song.get("artist", "Unknown")
            label = f"{artist} — {title}"

            yield {
                "type": "organizing",
                "index": i,
                "total": total,
                "label": label,
                "message": f"Checking {i}/{total}: {label}",
            }

            try:
                if song_id is None:
                    continue
                result = await organize_song_folder(song_id, dry_run=dry_run)
                if result.get("moved"):
                    organized += 1
                    yield {
                        "type": "organized",
                        "index": i,
                        "total": total,
                        "label": label,
                        "old_path": result.get("old_path", ""),
                        "new_path": result.get("new_path", ""),
                        "message": f"📁 Moved: {label}",
                    }
                elif result.get("dry_run") and result.get("old_path") != result.get(
                    "new_path"
                ):
                    organized += 1
                    yield {
                        "type": "organized",
                        "index": i,
                        "total": total,
                        "label": label,
                        "dry_run": True,
                        "message": f"📁 Would move: {label}",
                    }
            except Exception as e:
                errors.append(f"Error organizing {label}: {e}")

    # Step 2: Fetch missing album art
    if fetch_art:
        yield {
            "type": "phase",
            "phase": "album_art",
            "message": "Fetching missing album art…",
        }

        for i, song in enumerate(songs, 1):
            song_id = song.get("id")
            title = song.get("title", "Unknown")
            artist = song.get("artist", "Unknown")
            label = f"{artist} — {title}"

            if not artist or not title or song_id is None:
                continue

            yield {
                "type": "fetching_art",
                "index": i,
                "total": total,
                "label": label,
                "message": f"🎨 Checking art {i}/{total}: {label}",
            }

            try:
                result = await fetch_missing_art_for_song(song_id)
                if result.get("found") and not result.get("already_exists"):
                    art_found += 1
                elif not result.get("found") and not result.get("already_exists"):
                    art_missing += 1
            except Exception as e:
                errors.append(f"Error fetching art for {label}: {e}")

    # Step 3: Clean duplicates
    if clean_dupes:
        yield {"type": "phase", "phase": "dedup", "message": "Scanning for duplicates…"}

        try:
            dup_result = await cleanup_duplicates(dry_run=dry_run)
            dupes_removed = dup_result.get("deleted_count", 0)
            if dup_result.get("errors"):
                errors.extend(dup_result["errors"])
        except Exception as e:
            errors.append(f"Error during dedup: {e}")

    # Summary
    yield {
        "type": "complete",
        "total": total,
        "organized": organized,
        "art_found": art_found,
        "art_missing": art_missing,
        "dupes_removed": dupes_removed,
        "errors": errors,
        "dry_run": dry_run,
        "message": (
            f"Organization complete: {organized} moved, "
            f"{art_found} art downloaded, {dupes_removed} dupes removed"
        ),
    }
