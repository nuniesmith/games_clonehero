"""
Clone Hero Song Organizer
=========================
Scans a directory of Clone Hero songs, extracts any archives (.7z, .zip, .rar),
and organizes everything into a clean Artist/Song/ structure.

Standard structure:
  Artist/
    Song Title/
      album.png
      background.png
      Lyrics.txt
      notes.chart
      song.ini
      song.ogg

Usage:
  python organize_clonehero.py --source /path/to/messy/songs --dest /path/to/organized/songs

Optional:
  --dry-run         Preview changes without moving anything
  --move            Move files instead of copying (saves disk space)
  --clean           Auto-delete songs with critical issues (unplayable)
  --skip-extract    Skip archive extraction step
  --fetch-art       Download missing album art during organize
  --fetch-art-only  Scan an already-organized library and fetch missing art (no --source needed)
  --dedup           Scan dest for duplicate songs and print a report
  --dedup-clean     Scan dest for duplicates and auto-delete lower-quality copies

Cross-platform: uses 7z/unrar on Linux/macOS, WinRAR on Windows (auto-detected).
"""

import argparse
import configparser
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Map loose files to their standard name (case-insensitive matching).
# NOTE: We never cross-rename extensions (e.g. jpg->png) because that
# produces a file whose content doesn't match its extension.
RENAME_MAP = {
    # Images — keep original format
    "album.png": "album.png",
    "album.jpg": "album.jpg",
    "album.jpeg": "album.jpg",
    # Background — keep original format
    "background.png": "background.png",
    "background.jpg": "background.jpg",
    "background.jpeg": "background.jpg",
    # Chart
    "notes.chart": "notes.chart",
    "notes.mid": "notes.mid",
    # Audio
    "song.ogg": "song.ogg",
    "song.mp3": "song.mp3",
    "song.opus": "song.opus",
    # Lyrics
    "lyrics.txt": "Lyrics.txt",
    # Config
    "song.ini": "song.ini",
}

UNKNOWN_ARTIST = "_Unknown Artist"
UNKNOWN_SONG = "_Unknown Song"

# ANSI color codes
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

SPINNER_FRAMES = [
    "\u280b",
    "\u2819",
    "\u2839",
    "\u2838",
    "\u283c",
    "\u2834",
    "\u2826",
    "\u2827",
    "\u2807",
    "\u280f",
]
BAR_WIDTH = 30


# ---------------------------------------------------------------------------
# Progress display
# ---------------------------------------------------------------------------
class Progress:
    """Live terminal progress with spinner, bar, ETA, and current item."""

    def __init__(self, total: int, label: str = "Processing"):
        self.total = total
        self.label = label
        self.current = 0
        self.success = 0
        self.warnings = 0
        self.errors = 0
        self.current_item = ""
        self.start_time = time.monotonic()
        self._frame = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._lock = threading.Lock()
        self._term_width = shutil.get_terminal_size((80, 20)).columns
        self._term_width_updated = time.monotonic()

    def start(self):
        self._hide_cursor()
        self._thread.start()

    def stop(self, final_msg: str = ""):
        self._stop.set()
        self._thread.join()
        self._clear_line()
        if final_msg:
            print(final_msg)
        self._show_cursor()

    def update(self, item: str = "", advance: bool = True):
        with self._lock:
            if advance:
                self.current += 1
            self.current_item = item

    def mark_success(self):
        with self._lock:
            self.success += 1

    def mark_warning(self):
        with self._lock:
            self.warnings += 1

    def mark_error(self):
        with self._lock:
            self.errors += 1

    def log_line(self, msg: str):
        """Print a line above the spinner (for important messages)."""
        with self._lock:
            self._clear_line()
            print(msg)

    def _spin(self):
        while not self._stop.is_set():
            now = time.monotonic()
            if now - self._term_width_updated > 2.0:
                self._term_width = shutil.get_terminal_size((80, 20)).columns
                self._term_width_updated = now
            self._draw()
            self._stop.wait(0.08)

    def _draw(self):
        with self._lock:
            pct = (self.current / self.total * 100) if self.total else 0
            filled = int(BAR_WIDTH * self.current / self.total) if self.total else 0
            bar = f"{'\u2588' * filled}{'\u2591' * (BAR_WIDTH - filled)}"

            elapsed = time.monotonic() - self.start_time
            if self.current > 0 and self.current < self.total:
                rate = elapsed / self.current
                remaining = rate * (self.total - self.current)
                eta = str(timedelta(seconds=int(remaining)))
            elif self.current >= self.total:
                eta = "done!"
            else:
                eta = "calculating..."

            spinner = SPINNER_FRAMES[self._frame % len(SPINNER_FRAMES)]
            self._frame += 1

            item = self.current_item
            max_item_len = max(20, self._term_width - 60)
            if len(item) > max_item_len:
                item = item[: max_item_len - 3] + "..."

            stats = f"{GREEN}\u2713{self.success}{RESET}"
            if self.warnings:
                stats += f" {YELLOW}\u26a0{self.warnings}{RESET}"
            if self.errors:
                stats += f" {RED}\u2717{self.errors}{RESET}"

            line = (
                f"\r  {CYAN}{spinner}{RESET} "
                f"{bar} "
                f"{BOLD}{self.current}{RESET}/{self.total} "
                f"({pct:5.1f}%) "
                f"{DIM}ETA {eta}{RESET} "
                f"[{stats}]"
                f"\n  {DIM}\u25ba {item}{RESET}"
            )

            sys.stdout.write(f"\033[2A\033[J{line}")
            sys.stdout.flush()

    def _clear_line(self):
        sys.stdout.write("\r\033[K\r\033[K")
        sys.stdout.flush()

    @staticmethod
    def _hide_cursor():
        sys.stdout.write("\033[?25l\n\n")
        sys.stdout.flush()

    @staticmethod
    def _show_cursor():
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()


class SimpleSpinner:
    """Indeterminate spinner for tasks without a known total."""

    def __init__(self, message: str = "Working"):
        self.message = message
        self._frame = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def start(self):
        sys.stdout.write("\033[?25l")
        sys.stdout.flush()
        self._thread.start()

    def stop(self, final_msg: str = ""):
        self._stop.set()
        self._thread.join()
        sys.stdout.write("\r\033[K")
        if final_msg:
            print(final_msg)
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()

    def update(self, message: str):
        self.message = message

    def _spin(self):
        term_width = shutil.get_terminal_size((80, 20)).columns
        last_resize_check = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            if now - last_resize_check > 2.0:
                term_width = shutil.get_terminal_size((80, 20)).columns
                last_resize_check = now
            spinner = SPINNER_FRAMES[self._frame % len(SPINNER_FRAMES)]
            self._frame += 1
            msg = self.message
            if len(msg) > term_width - 10:
                msg = msg[: term_width - 13] + "..."
            sys.stdout.write(f"\r  {CYAN}{spinner}{RESET} {msg}\033[K")
            sys.stdout.flush()
            self._stop.wait(0.08)


# ---------------------------------------------------------------------------
# Logging (file-only — terminal is used by the spinner)
# ---------------------------------------------------------------------------
log = logging.getLogger("ch_organizer")


def setup_logging(dest: Path):
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    log_dir = dest / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(log_dir / f"organize_{ts}.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    log.addHandler(fh)


# ---------------------------------------------------------------------------
# Archive extraction
# ---------------------------------------------------------------------------
def find_archives(source: Path) -> list[Path]:
    """Recursively find all .7z, .zip, .rar archives."""
    exts = {".7z", ".zip", ".rar"}
    archives = []
    for p in source.rglob("*"):
        if p.suffix.lower() in exts and p.is_file():
            archives.append(p)
    return archives


# ---------------------------------------------------------------------------
# Cross-platform extraction helpers
# ---------------------------------------------------------------------------
def _find_extract_tool() -> dict:
    """Detect available extraction tools on this platform."""
    tools: dict[str, str | None] = {"7z": None, "unrar": None, "winrar": None}

    if platform.system() == "Windows":
        for candidate in [
            r"C:\Program Files\WinRAR\WinRAR.exe",
            r"C:\Program Files (x86)\WinRAR\WinRAR.exe",
        ]:
            if Path(candidate).is_file():
                tools["winrar"] = candidate
                break

    for name in ("7z", "7za", "7zz"):
        path = shutil.which(name)
        if path:
            tools["7z"] = path
            break

    for name in ("unrar", "unrar-free"):
        path = shutil.which(name)
        if path:
            tools["unrar"] = path
            break

    return tools


def _extract_zip(archive: Path, dest: Path) -> bool:
    """Extract a .zip using Python's zipfile (no external tool needed)."""
    try:
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(dest)
        return True
    except zipfile.BadZipFile:
        log.error(f"  Corrupted ZIP file: {archive.name}")
        return False
    except Exception as e:
        log.error(f"  ZIP extraction failed for {archive.name}: {e}")
        return False


def _extract_with_7z(archive: Path, dest: Path, tool_path: str) -> bool:
    """Extract any archive (.7z, .rar, .zip) using 7z."""
    try:
        result = subprocess.run(
            [tool_path, "x", "-y", f"-o{dest}", str(archive)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            log.warning(f"  7z returned code {result.returncode} for {archive.name}")
            log.debug(f"  stderr: {result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        log.error(f"  7z extraction timed out for: {archive.name}")
        return False
    except FileNotFoundError:
        log.error(f"  7z not found at: {tool_path}")
        return False


def _extract_with_unrar(archive: Path, dest: Path, tool_path: str) -> bool:
    """Extract a .rar using unrar / unrar-free."""
    try:
        result = subprocess.run(
            [tool_path, "x", "-o+", "-y", str(archive), str(dest) + os.sep],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            log.warning(f"  unrar returned code {result.returncode} for {archive.name}")
            log.debug(f"  stderr: {result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        log.error(f"  unrar extraction timed out for: {archive.name}")
        return False
    except FileNotFoundError:
        log.error(f"  unrar not found at: {tool_path}")
        return False


def _extract_with_winrar(archive: Path, dest: Path, tool_path: str) -> bool:
    """Extract any archive using WinRAR (Windows)."""
    try:
        result = subprocess.run(
            [tool_path, "x", "-o+", "-y", str(archive), str(dest) + "\\"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            log.warning(
                f"  WinRAR returned code {result.returncode} for {archive.name}"
            )
            log.debug(f"  stderr: {result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        log.error(f"  WinRAR extraction timed out for: {archive.name}")
        return False
    except FileNotFoundError:
        log.error(f"  WinRAR not found at: {tool_path}")
        return False


def extract_archive(archive: Path, tools: dict, dry_run: bool) -> Path | None:
    """Extract archive into a folder next to it, return the extraction dir."""
    extract_dir = archive.parent / archive.stem
    if extract_dir.exists():
        log.info(f"  Already extracted: {extract_dir}")
        return extract_dir

    if dry_run:
        log.info(f"  [DRY RUN] Would extract: {archive} -> {extract_dir}")
        return extract_dir

    extract_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"  Extracting: {archive.name} -> {extract_dir}")

    ext = archive.suffix.lower()
    ok = False

    if ext == ".zip":
        ok = _extract_zip(archive, extract_dir)
    elif ext == ".rar":
        if tools.get("unrar"):
            ok = _extract_with_unrar(archive, extract_dir, tools["unrar"])
        elif tools.get("7z"):
            ok = _extract_with_7z(archive, extract_dir, tools["7z"])
        elif tools.get("winrar"):
            ok = _extract_with_winrar(archive, extract_dir, tools["winrar"])
        else:
            log.error(f"  No tool available to extract .rar: {archive.name}")
            log.error("  Install 7z, unrar, or WinRAR.")
    elif ext == ".7z":
        if tools.get("7z"):
            ok = _extract_with_7z(archive, extract_dir, tools["7z"])
        elif tools.get("winrar"):
            ok = _extract_with_winrar(archive, extract_dir, tools["winrar"])
        else:
            log.error(f"  No tool available to extract .7z: {archive.name}")
            log.error("  Install 7z (p7zip-full) or WinRAR.")

    if not ok:
        try:
            extract_dir.rmdir()
        except OSError:
            pass
        return None

    return extract_dir


def extract_all(source: Path, tools: dict, dry_run: bool):
    """Find and extract all archives in the source tree."""
    archives = find_archives(source)
    if not archives:
        print(f"  {DIM}No archives found.{RESET}")
        return

    extracted = 0
    failed = 0
    prog = Progress(len(archives), "Extracting")
    prog.start()
    for a in archives:
        prog.update(a.name)
        result = extract_archive(a, tools, dry_run)
        if result is not None:
            prog.mark_success()
            extracted += 1
        else:
            prog.mark_error()
            failed += 1

    msg = f"  {GREEN}\u2713{RESET} Extracted {extracted} archive(s)."
    if failed:
        msg += f"  {RED}\u2717 {failed} failed.{RESET}"
    prog.stop(msg)


# ---------------------------------------------------------------------------
# Song discovery
# ---------------------------------------------------------------------------
def is_song_dir(d: Path) -> bool:
    """A directory counts as a song if it has notes.chart/notes.mid OR song.ini."""
    files_lower = {f.name.lower() for f in d.iterdir() if f.is_file()}
    return bool(files_lower & {"notes.chart", "notes.mid", "song.ini"})


def find_song_dirs(source: Path) -> list[Path]:
    """Recursively find all directories that look like Clone Hero song folders."""
    songs = []
    for root, dirs, files in os.walk(source):
        p = Path(root)
        if is_song_dir(p):
            songs.append(p)
            dirs.clear()
    return songs


# ---------------------------------------------------------------------------
# Metadata parsing
# ---------------------------------------------------------------------------
def parse_song_ini(song_dir: Path) -> dict:
    """Parse song.ini and return metadata dict with artist, name, etc."""
    ini_path = song_dir / "song.ini"
    if not ini_path.exists():
        for f in song_dir.iterdir():
            if f.name.lower() == "song.ini":
                ini_path = f
                break

    meta = {"artist": "", "name": "", "charter": "", "album": ""}

    if not ini_path.exists():
        log.debug(f"  No song.ini found in {song_dir}")
        return meta

    try:
        config = configparser.ConfigParser(strict=False, interpolation=None)
        config.read(str(ini_path), encoding="utf-8-sig")

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
        log.warning(f"  Failed to parse {ini_path}: {e}")

    return meta


MAX_FILENAME_LEN = 120


def sanitize_filename(name: str, max_len: int = MAX_FILENAME_LEN) -> str:
    """Remove characters that are invalid in Windows filenames and truncate."""
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", name)
    sanitized = sanitized.strip(". ")
    if not sanitized:
        return "_"
    if len(sanitized) > max_len:
        sanitized = sanitized[:max_len].rstrip(". ")
        log.warning(f"  Truncated long name: '{name[:60]}...' -> '{sanitized[:60]}...'")
    return sanitized


# ---------------------------------------------------------------------------
# File organization
# ---------------------------------------------------------------------------
def get_dest_path(meta: dict, dest: Path, song_dir: Path) -> Path:
    """Determine the destination Artist/Song/ path."""
    artist = meta.get("artist", "").strip()
    song_name = meta.get("name", "").strip()

    if not artist:
        dirname = song_dir.name
        if " - " in dirname:
            parts = dirname.split(" - ", 1)
            artist = parts[0].strip()
            if not song_name:
                song_name = parts[1].strip()

    artist = sanitize_filename(artist) if artist else UNKNOWN_ARTIST
    song_name = (
        sanitize_filename(song_name)
        if song_name
        else sanitize_filename(song_dir.name) or UNKNOWN_SONG
    )

    return dest / artist / song_name


def copy_song(
    song_dir: Path, dest_song_dir: Path, dry_run: bool, move: bool = False
) -> dict:
    """Copy (or move) song files to destination. Returns stats dict."""
    stats = {"copied": 0, "skipped": 0, "extra": []}
    transfer = shutil.move if move else shutil.copy2
    verb = "move" if move else "copy"

    if dry_run:
        log.info(f"  [DRY RUN] Would {verb} to: {dest_song_dir}")
        return stats

    dest_song_dir.mkdir(parents=True, exist_ok=True)

    for f in song_dir.iterdir():
        if not f.is_file():
            continue

        lower_name = f.name.lower()
        standard_name = RENAME_MAP.get(lower_name)

        if standard_name:
            dest_file = dest_song_dir / standard_name
            if dest_file.exists():
                log.debug(f"    Already exists, skipping: {standard_name}")
                stats["skipped"] += 1
                continue
            transfer(f, dest_file)
            log.debug(f"    {f.name} -> {standard_name}")
            stats["copied"] += 1
        else:
            dest_file = dest_song_dir / f.name
            if not dest_file.exists():
                transfer(f, dest_file)
                stats["extra"].append(f.name)
                stats["copied"] += 1

    return stats


# ---------------------------------------------------------------------------
# Duplicate / conflict handling
# ---------------------------------------------------------------------------
def handle_duplicate(dest_path: Path) -> Path:
    """If dest already exists, append a number."""
    if not dest_path.exists():
        return dest_path

    base = dest_path
    i = 2
    while dest_path.exists():
        dest_path = base.parent / f"{base.name} ({i})"
        i += 1

    log.info(f"  Duplicate detected, using: {dest_path.name}")
    return dest_path


# ---------------------------------------------------------------------------
# Validation / reporting
# ---------------------------------------------------------------------------
def validate_song(song_dir: Path) -> tuple[list[str], list[str]]:
    """Check a song directory for missing files.

    Returns (critical_issues, cosmetic_issues).
    Critical = song is unplayable in Clone Hero.
    Cosmetic = works but missing nice-to-haves.
    """
    critical = []
    cosmetic = []
    files_lower = {f.name.lower() for f in song_dir.iterdir() if f.is_file()}

    if "song.ini" not in files_lower:
        critical.append("Missing song.ini")
    if "notes.chart" not in files_lower and "notes.mid" not in files_lower:
        critical.append("Missing notes.chart / notes.mid")
    if not (files_lower & {"song.ogg", "song.mp3", "song.opus"}):
        critical.append("Missing audio (song.ogg/mp3/opus)")
    if not (files_lower & {"album.png", "album.jpg"}):
        cosmetic.append("Missing album art (album.png/jpg)")
    if not (files_lower & {"background.png", "background.jpg"}):
        cosmetic.append("Missing background (background.png/jpg)")

    return critical, cosmetic


def cleanup_song(dest_dir: Path, dry_run: bool) -> bool:
    """Delete a broken song directory. Returns True if deleted."""
    if dry_run:
        log.info(f"  [DRY RUN] Would delete: {dest_dir}")
        return False

    try:
        shutil.rmtree(dest_dir)
        log.info(f"  Cleaned up: {dest_dir}")

        artist_dir = dest_dir.parent
        if artist_dir.exists() and not any(artist_dir.iterdir()):
            artist_dir.rmdir()
            log.info(f"  Removed empty artist dir: {artist_dir.name}")

        return True
    except Exception as e:
        log.error(f"  Failed to clean up {dest_dir}: {e}")
        return False


# ---------------------------------------------------------------------------
# Album art fetching (MusicBrainz + Cover Art Archive)
# ---------------------------------------------------------------------------
MB_BASE = "https://musicbrainz.org/ws/2"
CAA_BASE = "https://coverartarchive.org"
MB_USER_AGENT = "CloneHeroOrganizer/1.0 (https://github.com/clonehero-organizer)"
_last_mb_request = 0.0


def _mb_request(url: str) -> dict | None:
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
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        log.debug(f"  MusicBrainz request failed: {e}")
        return None


def _search_release(artist: str, song: str, album: str = "") -> str | None:
    """Search MusicBrainz for a release MBID. Returns MBID or None."""
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
                log.debug(f"  Found release by album: {mbid} (score={score})")
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
                    log.debug(f"  Found release via recording: {mbid} (score={score})")
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
            log.debug(f"  Found release by song title: {mbid} (score={score})")
            return mbid

    return None


def _download_cover(mbid: str, dest_file: Path) -> bool:
    """Download front cover from Cover Art Archive. Returns True on success."""
    url = f"{CAA_BASE}/release/{mbid}/front-500"
    req = urllib.request.Request(url, headers={"User-Agent": MB_USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            img_data = resp.read()
            if len(img_data) < 1000:
                log.debug("  Cover art too small, likely not valid")
                return False
            dest_file.write_bytes(img_data)
            return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            log.debug(f"  No cover art available for release {mbid}")
        else:
            log.debug(f"  Cover art download failed: {e}")
        return False
    except (urllib.error.URLError, TimeoutError) as e:
        log.debug(f"  Cover art download failed: {e}")
        return False


def fetch_album_art(dest_dir: Path, meta: dict, dry_run: bool) -> bool:
    """Try to find and download album art for a song. Returns True on success."""
    if (dest_dir / "album.png").exists() or (dest_dir / "album.jpg").exists():
        return True

    artist = meta.get("artist", "").strip()
    song = meta.get("name", "").strip()
    album = meta.get("album", "").strip()

    if not artist or not song:
        log.debug("  Can't fetch art - missing artist or song name")
        return False

    if dry_run:
        log.info(f"  [DRY RUN] Would search for album art: {artist} - {song}")
        return False

    mbid = _search_release(artist, song, album)
    if not mbid:
        log.debug(f"  No MusicBrainz release found for: {artist} - {song}")
        return False

    album_path = dest_dir / "album.jpg"
    if _download_cover(mbid, album_path):
        log.info(f"  Downloaded album art for: {artist} - {song}")
        return True

    return False


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------
def _normalize(s: str) -> str:
    """Normalize a string for fuzzy matching - lowercase, strip punctuation."""
    s = s.lower().strip()
    s = re.sub(r"\s*\(.*?\)\s*", " ", s)  # remove parenthetical (remaster), (live)
    s = re.sub(r"\s*\[.*?\]\s*", " ", s)  # remove brackets [charter name]
    s = re.sub(r"[^a-z0-9\s]", "", s)  # strip punctuation
    s = re.sub(r"\s+", " ", s).strip()  # collapse whitespace
    return s


def _chart_hash(song_dir: Path) -> str | None:
    """Hash the notes.chart / notes.mid file to detect identical charts."""
    for name in ("notes.chart", "notes.mid"):
        chart = song_dir / name
        if chart.exists():
            try:
                h = hashlib.sha256()
                with open(chart, "rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        h.update(chunk)
                return h.hexdigest()[:16]
            except OSError:
                pass
    return None


def _score_song(song_dir: Path) -> int:
    """Score a song directory by completeness. Higher = better copy to keep."""
    score = 0
    files = {f.name.lower(): f for f in song_dir.iterdir() if f.is_file()}

    # Critical files
    if "notes.chart" in files or "notes.mid" in files:
        score += 10
    if files.keys() & {"song.ogg", "song.mp3", "song.opus"}:
        score += 10
        # Prefer larger audio (likely higher quality)
        for ext in ("song.ogg", "song.mp3", "song.opus"):
            if ext in files:
                size_mb = files[ext].stat().st_size / (1024 * 1024)
                score += min(int(size_mb), 5)
                break
    if "song.ini" in files:
        score += 5
    # Cosmetic files
    if files.keys() & {"album.png", "album.jpg"}:
        score += 3
    if files.keys() & {"background.png", "background.jpg"}:
        score += 2
    if "lyrics.txt" in files:
        score += 1
    # More total files = more complete chart
    score += min(len(files), 5)

    return score


def find_duplicates(
    song_dirs: list[Path], progress: Progress | None = None
) -> dict[str, list[tuple[Path, int, str | None]]]:
    """Find duplicate songs by normalized artist+name and chart hash.

    Returns dict of group_key -> [(path, score, chart_hash), ...] with 2+ entries.
    """
    # Pass 1: Group by normalized artist + song name
    by_name: dict[str, list[tuple[Path, dict]]] = {}
    for sd in song_dirs:
        meta = parse_song_ini(sd)
        artist = meta.get("artist", "").strip()
        song = meta.get("name", "").strip()

        if not artist and not song:
            dirname = sd.name
            if " - " in dirname:
                parts = dirname.split(" - ", 1)
                artist = parts[0]
                song = parts[1]
            else:
                song = dirname

        key = f"{_normalize(artist)}|{_normalize(song)}"
        by_name.setdefault(key, []).append((sd, meta))

        if progress:
            display_artist = artist or "(unknown)"
            display_song = song or sd.name
            progress.update(f"{display_artist} - {display_song}")

    # Pass 2: Also group by chart hash to catch renamed dupes
    by_hash: dict[str, list[Path]] = {}
    for sd in song_dirs:
        ch = _chart_hash(sd)
        if ch:
            by_hash.setdefault(ch, []).append(sd)

    # Merge: build final duplicate groups
    seen_paths: set[Path] = set()
    groups: dict[str, list[tuple[Path, int, str | None]]] = {}

    for key, entries in by_name.items():
        if len(entries) < 2:
            continue
        group = []
        for sd, meta in entries:
            score = _score_song(sd)
            ch = _chart_hash(sd)
            group.append((sd, score, ch))
            seen_paths.add(sd)
        groups[key] = group

    # Add hash-based groups for songs not already caught by name
    for ch, paths in by_hash.items():
        if len(paths) < 2:
            continue
        unseen = [p for p in paths if p not in seen_paths]
        if len(unseen) < 2:
            already_grouped = [p for p in paths if p in seen_paths]
            if already_grouped and unseen:
                for key, group in groups.items():
                    group_paths = {g[0] for g in group}
                    if group_paths & set(already_grouped):
                        for p in unseen:
                            score = _score_song(p)
                            group.append((p, score, ch))
                            seen_paths.add(p)
                        break
            continue

        key = f"chart_hash:{ch}"
        group = []
        for p in paths:
            score = _score_song(p)
            group.append((p, score, ch))
            seen_paths.add(p)
        groups[key] = group

    return groups


def dedup_clean(
    groups: dict[str, list[tuple[Path, int, str | None]]],
    dry_run: bool,
    progress: Progress | None = None,
) -> int:
    """Delete duplicate songs, keeping the highest-scored copy. Returns count deleted."""
    deleted = 0

    for key, entries in groups.items():
        entries.sort(key=lambda x: -x[1])
        keep = entries[0]
        dupes = entries[1:]

        for path, score, ch in dupes:
            log.info(f"  Deleting dupe: {path} (score={score}, kept={keep[0]})")

            if progress:
                progress.update(f"Removing {path.name}")

            if dry_run:
                log.info(f"  [DRY RUN] Would delete: {path}")
                deleted += 1
                if progress:
                    progress.mark_success()
                continue

            try:
                shutil.rmtree(path)
                deleted += 1

                artist_dir = path.parent
                if artist_dir.exists() and not any(artist_dir.iterdir()):
                    artist_dir.rmdir()
                    log.info(f"  Removed empty artist dir: {artist_dir.name}")

                if progress:
                    progress.mark_success()

            except Exception as e:
                log.error(f"  Failed to delete {path}: {e}")
                if progress:
                    progress.mark_error()

    return deleted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Organize Clone Hero songs into Artist/Song/ structure."
    )
    parser.add_argument(
        "--source",
        help="Source directory containing messy Clone Hero songs",
    )
    parser.add_argument(
        "--dest",
        required=True,
        help="Destination directory for organized songs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without actually moving/copying files",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying (saves disk space, modifies source)",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip archive extraction step",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Auto-delete songs with critical issues (missing audio/chart - unplayable)",
    )
    parser.add_argument(
        "--fetch-art",
        action="store_true",
        help="Auto-download missing album art from MusicBrainz/Cover Art Archive",
    )
    parser.add_argument(
        "--fetch-art-only",
        action="store_true",
        help="Only scan dest for missing album art (skip organize/extract)",
    )
    parser.add_argument(
        "--dedup",
        action="store_true",
        help="Scan dest for duplicate songs and print a report",
    )
    parser.add_argument(
        "--dedup-clean",
        action="store_true",
        help="Scan dest for duplicates and auto-delete lower-quality copies",
    )

    args = parser.parse_args()
    dest = Path(args.dest).resolve()

    # -----------------------------------------------------------------------
    # Mode: --fetch-art-only
    # -----------------------------------------------------------------------
    if args.fetch_art_only:
        setup_logging(dest)
        print(f"\n  {BOLD}🎨 Clone Hero Album Art Fetcher{RESET}")
        print(f"  {DIM}{'─' * 45}{RESET}")
        print(f"  Scanning: {dest}")
        if args.dry_run:
            print(f"  {YELLOW}⚠  DRY RUN{RESET}")
        print()

        scan_spinner = SimpleSpinner("Scanning organized songs...")
        scan_spinner.start()
        song_dirs = find_song_dirs(dest)
        scan_spinner.stop(
            f"  {GREEN}\u2713{RESET} Found {BOLD}{len(song_dirs)}{RESET} song(s) in library."
        )

        if not song_dirs:
            print(f"  {YELLOW}No songs found in dest.{RESET}")
            return

        missing_art = [
            sd
            for sd in song_dirs
            if not (sd / "album.png").exists() and not (sd / "album.jpg").exists()
        ]

        if not missing_art:
            print(
                f"\n  {GREEN}\u2713 All {len(song_dirs)} songs already have album art!{RESET}\n"
            )
            return

        print(f"  {YELLOW}{len(missing_art)}{RESET} song(s) missing album art.\n")

        art_found = 0
        art_missing = 0
        prog = Progress(len(missing_art), "Fetching art")
        prog.start()

        for sd in missing_art:
            meta = parse_song_ini(sd)
            artist = meta.get("artist", "") or "(unknown)"
            song = meta.get("name", "") or sd.name
            prog.update(f"🎨 {artist} - {song}")

            if fetch_album_art(sd, meta, args.dry_run):
                art_found += 1
                prog.mark_success()
            else:
                art_missing += 1
                prog.mark_warning()

        prog.stop("")

        print(f"\n  {BOLD}{'═' * 50}{RESET}")
        print(f"  {BOLD}🎨 Album Art Fetch — Summary{RESET}")
        print(f"  {BOLD}{'═' * 50}{RESET}")
        print(f"  Songs scanned:     {BOLD}{len(missing_art)}{RESET}")
        print(f"  Art downloaded:    {GREEN}{art_found}{RESET}")
        print(f"  Art not found:     {YELLOW}{art_missing}{RESET}")
        print(f"  {BOLD}{'═' * 50}{RESET}")
        print(f"  {DIM}Full log: {dest / '_logs'}{RESET}\n")
        return

    # -----------------------------------------------------------------------
    # Mode: --dedup / --dedup-clean
    # -----------------------------------------------------------------------
    if args.dedup or args.dedup_clean:
        setup_logging(dest)
        print(f"\n  {BOLD}🔍 Clone Hero Duplicate Scanner{RESET}")
        print(f"  {DIM}{'─' * 45}{RESET}")
        print(f"  Scanning: {dest}")
        print(
            f"  Mode:     "
            f"{'🗑️  Auto-clean (delete lower-quality dupes)' if args.dedup_clean else '📋 Report only'}"
        )
        if args.dry_run:
            print(f"  {YELLOW}⚠  DRY RUN{RESET}")
        print()

        scan_spinner = SimpleSpinner("Scanning organized songs...")
        scan_spinner.start()
        song_dirs = find_song_dirs(dest)
        scan_spinner.stop(
            f"  {GREEN}\u2713{RESET} Found {BOLD}{len(song_dirs)}{RESET} song(s) in library."
        )

        if not song_dirs:
            print(f"  {YELLOW}No songs found in dest.{RESET}")
            return

        print(f"\n  Analyzing for duplicates...\n")
        prog = Progress(len(song_dirs), "Scanning")
        prog.start()
        groups = find_duplicates(song_dirs, progress=prog)
        total_dupes = sum(len(g) - 1 for g in groups.values())
        prog.stop("")

        if not groups:
            print(
                f"\n  {GREEN}\u2713 No duplicates found across {len(song_dirs)} songs!{RESET}\n"
            )
            return

        # --- Report ---
        print(
            f"\n  Found {YELLOW}{total_dupes}{RESET} duplicate(s) "
            f"across {BOLD}{len(groups)}{RESET} group(s).\n"
        )

        waste = 0
        shown = 0
        for key, entries in sorted(groups.items(), key=lambda x: -len(x[1])):
            entries.sort(key=lambda x: -x[1])

            if key.startswith("chart_hash:"):
                display = f"[identical chart] {entries[0][0].name}"
            else:
                parts = key.split("|", 1)
                display = f"{parts[0]} — {parts[1]}" if len(parts) == 2 else key

            if shown < 30:
                print(f"  {BOLD}{display}{RESET}")
                for i, (path, score, ch) in enumerate(entries):
                    try:
                        rel = path.relative_to(dest)
                    except ValueError:
                        rel = path
                    if i == 0:
                        print(f"    {GREEN}★ KEEP{RESET}  score={score:<3}  {rel}")
                    else:
                        print(f"    {RED}✗ DUPE{RESET}  score={score:<3}  {rel}")
                        try:
                            for f in path.rglob("*"):
                                if f.is_file():
                                    waste += f.stat().st_size
                        except OSError:
                            pass
                print()
                shown += 1
            else:
                for i, (path, score, ch) in enumerate(entries):
                    if i > 0:
                        try:
                            for f in path.rglob("*"):
                                if f.is_file():
                                    waste += f.stat().st_size
                        except OSError:
                            pass

        if shown < len(groups):
            print(
                f"  {DIM}...and {len(groups) - shown} more group(s) (see log file){RESET}\n"
            )

        waste_gb = waste / (1024 * 1024 * 1024)
        waste_mb = waste / (1024 * 1024)
        waste_str = f"{waste_gb:.2f} GB" if waste_gb >= 1 else f"{waste_mb:.1f} MB"
        print(f"  {DIM}Estimated wasted space: {YELLOW}{waste_str}{RESET}\n")

        # Log all groups to file for reference
        for key, entries in sorted(groups.items()):
            entries.sort(key=lambda x: -x[1])
            log.info(f"Duplicate group: {key}")
            for i, (path, score, ch) in enumerate(entries):
                marker = "KEEP" if i == 0 else "DUPE"
                log.info(f"  [{marker}] score={score}  hash={ch}  {path}")

        # --- Clean ---
        if args.dedup_clean:
            print(f"  {BOLD}Cleaning duplicates...{RESET}\n")
            clean_prog = Progress(total_dupes, "Cleaning")
            clean_prog.start()
            deleted = dedup_clean(groups, args.dry_run, progress=clean_prog)
            clean_prog.stop("")

            action = "Would delete" if args.dry_run else "Deleted"
            print(f"\n  {BOLD}{'═' * 50}{RESET}")
            print(f"  {BOLD}🔍 Duplicate Scan — Summary{RESET}")
            print(f"  {BOLD}{'═' * 50}{RESET}")
            print(f"  Duplicate groups:  {BOLD}{len(groups)}{RESET}")
            print(f"  {action}:         {RED}{deleted}{RESET} song(s)")
            print(f"  Space reclaimed:   {GREEN}{waste_str}{RESET}")
            print(f"  {BOLD}{'═' * 50}{RESET}")
        else:
            print(
                f"  {DIM}Run with --dedup-clean to auto-delete the lower-quality copies.{RESET}"
            )

        print(f"  {DIM}Full log: {dest / '_logs'}{RESET}\n")
        return

    # -----------------------------------------------------------------------
    # Normal organize mode (requires --source)
    # -----------------------------------------------------------------------
    if not args.source:
        parser.error(
            "--source is required (unless using --fetch-art-only, --dedup, or --dedup-clean)"
        )

    source = Path(args.source).resolve()

    if not source.exists():
        print(f"Error: Source directory not found: {source}")
        sys.exit(1)

    setup_logging(dest)

    tools = _find_extract_tool()

    print(f"\n  {BOLD}🎸 Clone Hero Song Organizer{RESET}")
    print(f"  {DIM}{'─' * 45}{RESET}")
    print(f"  Source:   {source}")
    print(f"  Dest:     {dest}")
    print(f"  Platform: {platform.system()}")

    available = [name for name, path in tools.items() if path]
    if available:
        print(f"  Tools:    {', '.join(available)}")
    else:
        print(f"  Tools:    {YELLOW}none detected (ZIP only){RESET}")

    if args.dry_run:
        print(f"  {YELLOW}⚠  DRY RUN — no files will be changed{RESET}")
    if args.move:
        print(f"  {YELLOW}⚠  MOVE MODE — source files will be relocated{RESET}")
    if args.clean:
        print(f"  {RED}🧹 CLEAN MODE — unplayable songs will be deleted{RESET}")
    if args.fetch_art:
        print(f"  {CYAN}🎨 FETCH ART — will download missing album covers{RESET}")
    print()

    log.info(f"Source:   {source}")
    log.info(f"Dest:     {dest}")
    log.info(f"Platform: {platform.system()}")
    log.info(f"Tools:    {tools}")
    log.info(f"Dry run:  {args.dry_run}")
    log.info(f"Move:     {args.move}")
    log.info(f"Fetch art: {args.fetch_art}")

    # Step 1: Extract archives
    if not args.skip_extract:
        print(f"  {BOLD}Step 1:{RESET} Extracting archives...")
        log.info("--- Step 1: Extracting archives ---")
        extract_all(source, tools, args.dry_run)
    else:
        print(f"  {DIM}Step 1: Skipped (--skip-extract){RESET}")

    # Step 2: Discover song directories
    print(f"\n  {BOLD}Step 2:{RESET} Scanning for songs...")
    log.info("--- Step 2: Discovering songs ---")

    scan_spinner = SimpleSpinner("Scanning directories...")
    scan_spinner.start()
    song_dirs = find_song_dirs(source)
    scan_spinner.stop(
        f"  {GREEN}\u2713{RESET} Found {BOLD}{len(song_dirs)}{RESET} song folder(s)."
    )
    log.info(f"Found {len(song_dirs)} song folder(s).")

    if not song_dirs:
        print(f"  {YELLOW}No songs found! Check your source directory.{RESET}")
        return

    # Step 3: Organize
    transfer_verb = "Moving" if args.move else "Copying"
    print(f"\n  {BOLD}Step 3:{RESET} {transfer_verb} & organizing songs...\n")
    log.info(f"--- Step 3: {transfer_verb} & organizing songs ---")

    success = 0
    skipped_broken = 0
    all_issues = {}
    cleaned = 0
    art_found = 0
    art_missing = 0
    failed = []

    prog = Progress(len(song_dirs), "Organizing")
    prog.start()

    for i, sd in enumerate(song_dirs, 1):
        log.info(f"\n[{i}/{len(song_dirs)}] {sd.name}")

        try:
            meta = parse_song_ini(sd)
            artist_display = meta["artist"] or "(unknown)"
            name_display = meta["name"] or sd.name
            log.info(f"  Artist: {artist_display}  |  Song: {name_display}")

            prog.update(f"{artist_display} — {name_display}")

            # Pre-validate source before copying
            critical, cosmetic = validate_song(sd)

            if critical and args.clean:
                log.info(f"  Skipping (unplayable): {', '.join(critical)}")
                skipped_broken += 1
                prog.mark_warning()
                continue

            dest_song = get_dest_path(meta, dest, sd)
            dest_song = handle_duplicate(dest_song)

            stats = copy_song(sd, dest_song, args.dry_run, move=args.move)
            log.info(
                f"  {'Moved' if args.move else 'Copied'}: {stats['copied']}  "
                f"Skipped: {stats['skipped']}"
            )
            if stats["extra"]:
                log.debug(f"  Extra files: {', '.join(stats['extra'])}")

            # Post-validate destination
            if not args.dry_run and dest_song.exists():
                post_critical, post_cosmetic = validate_song(dest_song)

                if post_critical:
                    log.warning(f"  Critical: {', '.join(post_critical)}")

                    if args.clean:
                        if cleanup_song(dest_song, args.dry_run):
                            cleaned += 1
                            log.info(f"  Cleaned: {dest_song.name}")
                            prog.mark_warning()
                            continue
                    else:
                        all_issues[str(dest_song.relative_to(dest))] = {
                            "critical": post_critical,
                            "cosmetic": post_cosmetic,
                        }
                        prog.mark_warning()

                # Fetch missing album art
                if args.fetch_art and not post_critical:
                    has_album = (dest_song / "album.png").exists() or (
                        dest_song / "album.jpg"
                    ).exists()
                    if not has_album:
                        prog.update(
                            f"🎨 {artist_display} — {name_display}", advance=False
                        )
                        if fetch_album_art(dest_song, meta, args.dry_run):
                            art_found += 1
                        else:
                            art_missing += 1

                elif post_cosmetic:
                    log.debug(f"  Cosmetic: {', '.join(post_cosmetic)}")

            elif critical and not args.clean:
                all_issues[f"(source) {sd.name}"] = {
                    "critical": critical,
                    "cosmetic": cosmetic,
                }
                prog.mark_warning()

            prog.mark_success()
            success += 1

        except Exception as e:
            log.error(f"  FAILED: {e}")
            failed.append((sd.name, str(e)))
            prog.mark_error()
            continue

    elapsed = time.monotonic() - prog.start_time
    elapsed_str = str(timedelta(seconds=int(elapsed)))
    prog.stop("")

    # --- Summary ---
    print(f"\n  {BOLD}{'═' * 50}{RESET}")
    print(f"  {BOLD}🎸 Clone Hero Song Organizer — Summary{RESET}")
    print(f"  {BOLD}{'═' * 50}{RESET}")
    print(f"  Songs found:       {BOLD}{len(song_dirs)}{RESET}")
    print(f"  Songs organized:   {GREEN}{success}{RESET}")
    if skipped_broken:
        print(
            f"  Skipped (broken):  {YELLOW}{skipped_broken}{RESET} {DIM}(unplayable, not copied){RESET}"
        )
    if cleaned:
        print(
            f"  Songs cleaned up:  {RED}{cleaned}{RESET} {DIM}(unplayable, deleted){RESET}"
        )
    if args.fetch_art:
        print(f"  Album art found:   {GREEN}{art_found}{RESET} {DIM}downloaded{RESET}")
        if art_missing:
            print(
                f"  Album art missing: {YELLOW}{art_missing}{RESET} {DIM}(not on MusicBrainz){RESET}"
            )
    if all_issues:
        print(
            f"  Songs with issues: {YELLOW}{len(all_issues)}{RESET} {DIM}(use --clean to skip){RESET}"
        )
    print(f"  Songs failed:      {RED}{len(failed)}{RESET}")
    print(f"  Total time:        {elapsed_str}")

    if all_issues:
        print(f"\n  {YELLOW}Unplayable songs (missing critical files):{RESET}")
        shown = 0
        for song, details in sorted(all_issues.items()):
            if shown >= 20:
                remaining = len(all_issues) - 20
                print(f"    {DIM}...and {remaining} more (see log file){RESET}")
                break
            print(f"    {DIM}{song}{RESET}")
            for p in details["critical"]:
                print(f"      {RED}\u2717 {p}{RESET}")
            shown += 1

    if failed:
        print(f"\n  {RED}{len(failed)} song(s) failed to process:{RESET}")
        for name, err in failed[:20]:
            print(f"    {RED}\u2717{RESET} {name}: {DIM}{err}{RESET}")
        if len(failed) > 20:
            print(f"    {DIM}...and {len(failed) - 20} more (see log file){RESET}")

    print(f"\n  {BOLD}{'═' * 50}{RESET}")
    print(f"  {DIM}Full log: {dest / '_logs'}{RESET}\n")


if __name__ == "__main__":
    main()
