"""
Clone Hero Song Organizer
=========================
Scans a directory of Clone Hero songs, extracts any .7z archives,
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
  python organize_clonehero.py --source "C:\path\to\messy\songs" --dest "C:\path\to\organized\songs"

Optional:
  --dry-run       Preview changes without moving anything
  --winrar-path   Path to WinRAR.exe (default: C:\Program Files\WinRAR\WinRAR.exe)
"""

import os
import sys
import shutil
import argparse
import subprocess
import configparser
import re
import logging
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
STANDARD_FILES = {
    "album.png",
    "background.png",
    "Lyrics.txt",       # not present in every song — that's fine
    "notes.chart",
    "notes.mid",        # some songs use MIDI instead of .chart
    "song.ini",
    "song.ogg",
    "song.mp3",         # alternate audio format
    "song.opus",        # alternate audio format
    "video.mp4",        # some songs have background video
    "video.avi",
    "video.webm",
}

# Map loose files to their standard name (case-insensitive matching)
RENAME_MAP = {
    # Images
    "album.png": "album.png",
    "album.jpg": "album.png",
    "album.jpeg": "album.png",
    # Background
    "background.png": "background.png",
    "background.jpg": "background.png",
    "background.jpeg": "background.png",
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

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger("ch_organizer")


def setup_logging(dest: Path):
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    log.addHandler(console)

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


def extract_archive(archive: Path, winrar: str, dry_run: bool) -> Path | None:
    """Extract archive into a folder next to it, return the extraction dir."""
    extract_dir = archive.parent / archive.stem
    if extract_dir.exists():
        log.info(f"  Already extracted: {extract_dir}")
        return extract_dir

    if dry_run:
        log.info(f"  [DRY RUN] Would extract: {archive} -> {extract_dir}")
        return None

    extract_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"  Extracting: {archive.name} -> {extract_dir}")

    try:
        # WinRAR command-line: x = extract with paths, -o+ = overwrite, -y = yes to all
        result = subprocess.run(
            [winrar, "x", "-o+", "-y", str(archive), str(extract_dir) + "\\"],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            log.warning(f"  WinRAR returned code {result.returncode} for {archive.name}")
            log.debug(f"  stderr: {result.stderr}")
        return extract_dir
    except FileNotFoundError:
        log.error(f"  WinRAR not found at: {winrar}")
        log.error("  Use --winrar-path to specify the correct location.")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        log.error(f"  Extraction timed out for: {archive.name}")
        return None


def extract_all(source: Path, winrar: str, dry_run: bool):
    """Find and extract all archives in the source tree."""
    archives = find_archives(source)
    if not archives:
        log.info("No archives found.")
        return

    log.info(f"Found {len(archives)} archive(s) to extract.")
    for a in archives:
        extract_archive(a, winrar, dry_run)


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
            dirs.clear()  # don't recurse deeper into a song folder
    return songs


# ---------------------------------------------------------------------------
# Metadata parsing
# ---------------------------------------------------------------------------
def parse_song_ini(song_dir: Path) -> dict:
    """Parse song.ini and return metadata dict with artist, name, etc."""
    ini_path = song_dir / "song.ini"
    if not ini_path.exists():
        # Try case-insensitive
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
        # Clone Hero ini files use [Song] or [song] section
        config.read(str(ini_path), encoding="utf-8-sig")

        # Find the song section (case-insensitive)
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


def sanitize_filename(name: str) -> str:
    """Remove characters that are invalid in Windows filenames."""
    # Replace invalid chars with underscore
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
    sanitized = sanitized.strip('. ')
    return sanitized or "_"


# ---------------------------------------------------------------------------
# File organization
# ---------------------------------------------------------------------------
def get_dest_path(meta: dict, dest: Path, song_dir: Path) -> Path:
    """Determine the destination Artist/Song/ path."""
    artist = meta.get("artist", "").strip()
    song_name = meta.get("name", "").strip()

    if not artist:
        # Try to infer from directory structure: often "Artist - Song"
        dirname = song_dir.name
        if " - " in dirname:
            parts = dirname.split(" - ", 1)
            artist = parts[0].strip()
            if not song_name:
                song_name = parts[1].strip()

    artist = sanitize_filename(artist) if artist else UNKNOWN_ARTIST
    song_name = sanitize_filename(song_name) if song_name else sanitize_filename(song_dir.name) or UNKNOWN_SONG

    return dest / artist / song_name


def copy_song(song_dir: Path, dest_song_dir: Path, dry_run: bool) -> dict:
    """Copy/rename song files to destination. Returns stats dict."""
    stats = {"copied": 0, "skipped": 0, "extra": []}

    if dry_run:
        log.info(f"  [DRY RUN] Would copy to: {dest_song_dir}")
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

            # Handle jpg -> png conversion note (just copy, rename ext)
            shutil.copy2(f, dest_file)
            log.debug(f"    {f.name} -> {standard_name}")
            stats["copied"] += 1
        else:
            # Copy extra files too (video, highway.png, etc.)
            dest_file = dest_song_dir / f.name
            if not dest_file.exists():
                shutil.copy2(f, dest_file)
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
def validate_song(dest_dir: Path) -> list[str]:
    """Check a destination song dir for missing essential files."""
    issues = []
    files_lower = {f.name.lower() for f in dest_dir.iterdir() if f.is_file()}

    if "song.ini" not in files_lower:
        issues.append("Missing song.ini")
    if "notes.chart" not in files_lower and "notes.mid" not in files_lower:
        issues.append("Missing notes.chart / notes.mid")
    if not (files_lower & {"song.ogg", "song.mp3", "song.opus"}):
        issues.append("Missing audio (song.ogg/mp3/opus)")
    if "album.png" not in files_lower:
        issues.append("Missing album.png (cosmetic)")

    return issues


def print_summary(total: int, success: int, issues: dict[str, list[str]]):
    """Print a final summary of the organization run."""
    print("\n" + "=" * 60)
    print(f"  Clone Hero Song Organizer — Summary")
    print("=" * 60)
    print(f"  Songs found:      {total}")
    print(f"  Songs organized:  {success}")
    print(f"  Songs with issues: {len(issues)}")

    if issues:
        print("\n  Songs with missing files:")
        for song, problems in sorted(issues.items()):
            print(f"    {song}")
            for p in problems:
                print(f"      - {p}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Organize Clone Hero songs into Artist/Song/ structure."
    )
    parser.add_argument(
        "--source", required=True,
        help="Source directory containing messy Clone Hero songs"
    )
    parser.add_argument(
        "--dest", required=True,
        help="Destination directory for organized songs"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview changes without actually moving/copying files"
    )
    parser.add_argument(
        "--winrar-path",
        default=r"C:\Program Files\WinRAR\WinRAR.exe",
        help="Path to WinRAR.exe"
    )
    parser.add_argument(
        "--skip-extract", action="store_true",
        help="Skip archive extraction step"
    )

    args = parser.parse_args()

    source = Path(args.source).resolve()
    dest = Path(args.dest).resolve()

    if not source.exists():
        print(f"Error: Source directory not found: {source}")
        sys.exit(1)

    setup_logging(dest)

    log.info(f"Source:  {source}")
    log.info(f"Dest:    {dest}")
    log.info(f"Dry run: {args.dry_run}")

    # Step 1: Extract archives
    if not args.skip_extract:
        log.info("\n--- Step 1: Extracting archives ---")
        extract_all(source, args.winrar_path, args.dry_run)
    else:
        log.info("\n--- Skipping extraction ---")

    # Step 2: Discover song directories
    log.info("\n--- Step 2: Discovering songs ---")
    song_dirs = find_song_dirs(source)
    log.info(f"Found {len(song_dirs)} song folder(s).")

    if not song_dirs:
        log.warning("No songs found! Check that your source directory is correct.")
        return

    # Step 3: Organize
    log.info("\n--- Step 3: Organizing songs ---")
    success = 0
    all_issues = {}

    for i, sd in enumerate(song_dirs, 1):
        log.info(f"\n[{i}/{len(song_dirs)}] {sd.name}")

        # Parse metadata
        meta = parse_song_ini(sd)
        artist_display = meta['artist'] or '(unknown)'
        name_display = meta['name'] or sd.name
        log.info(f"  Artist: {artist_display}  |  Song: {name_display}")

        # Determine destination
        dest_song = get_dest_path(meta, dest, sd)
        dest_song = handle_duplicate(dest_song)

        # Copy files
        stats = copy_song(sd, dest_song, args.dry_run)
        log.info(f"  Copied: {stats['copied']}  Skipped: {stats['skipped']}")
        if stats["extra"]:
            log.debug(f"  Extra files: {', '.join(stats['extra'])}")

        # Validate
        if not args.dry_run and dest_song.exists():
            issues = validate_song(dest_song)
            if issues:
                all_issues[str(dest_song.relative_to(dest))] = issues
                log.warning(f"  Issues: {', '.join(issues)}")

        success += 1

    print_summary(len(song_dirs), success, all_issues)
    log.info("Done! Check the log file in _logs/ for full details.")


if __name__ == "__main__":
    main()
