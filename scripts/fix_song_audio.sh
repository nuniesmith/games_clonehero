#!/bin/bash
# fix_song_audio.sh — Fix generated Clone Hero songs that use unsupported audio formats
#
# Clone Hero does not reliably support FLAC across all versions.
# This script converts FLAC (or WAV) audio to OGG Vorbis and patches
# the notes.chart and song.ini files accordingly.
#
# Usage:
#   ./scripts/fix_song_audio.sh <song_directory>
#   ./scripts/fix_song_audio.sh songs/Go\ To\ Sleep_2nd
#   ./scripts/fix_song_audio.sh   # no args → fix all songs in ./songs/

set -euo pipefail

BITRATE="${BITRATE:-192k}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log()  { printf "\033[1;32m[FIX]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[WARN]\033[0m %s\n" "$*"; }
err()  { printf "\033[1;31m[ERR]\033[0m %s\n" "$*" >&2; }

check_deps() {
    if ! command -v ffmpeg &>/dev/null; then
        err "ffmpeg is required but not found. Install it first:"
        err "  sudo apt install ffmpeg   # Debian/Ubuntu"
        err "  sudo pacman -S ffmpeg     # Arch"
        err "  winget install ffmpeg     # Windows"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Fix a single song directory
# ---------------------------------------------------------------------------

fix_song_dir() {
    local dir="$1"

    # Find FLAC or WAV files named song.* (the ones Clone Hero tries to load)
    local src_audio=""
    local src_ext=""

    for ext in flac wav aac m4a wma; do
        if [[ -f "$dir/song.$ext" ]]; then
            src_audio="$dir/song.$ext"
            src_ext="$ext"
            break
        fi
    done

    if [[ -z "$src_audio" ]]; then
        # Check if already OGG/OPUS/MP3 — nothing to do
        if [[ -f "$dir/song.ogg" || -f "$dir/song.opus" || -f "$dir/song.mp3" ]]; then
            log "$dir — already using a compatible format, skipping"
            return 0
        fi
        warn "$dir — no song audio file found, skipping"
        return 0
    fi

    log "$dir — converting song.$src_ext → song.ogg ($BITRATE)"

    # Convert to OGG Vorbis
    if ! ffmpeg -y -i "$src_audio" -vn -c:a libvorbis -b:a "$BITRATE" "$dir/song.ogg" \
         -loglevel warning 2>&1; then
        err "$dir — ffmpeg conversion failed!"
        return 1
    fi

    if [[ ! -f "$dir/song.ogg" ]]; then
        err "$dir — song.ogg was not created"
        return 1
    fi

    # Patch notes.chart if it exists
    if [[ -f "$dir/notes.chart" ]]; then
        log "$dir — patching notes.chart (MusicStream → song.ogg)"

        # 1) Fix MusicStream reference
        if grep -q "MusicStream" "$dir/notes.chart"; then
            sed -i "s|MusicStream = \"song\.$src_ext\"|MusicStream = \"song.ogg\"|g" "$dir/notes.chart"
        else
            # MusicStream line is missing entirely — add it before the closing brace
            # of the [Song] section
            sed -i '/^\[Song\]/,/^}/ {
                /^}/ i\  MusicStream = "song.ogg"
            }' "$dir/notes.chart"
            log "$dir — added missing MusicStream field"
        fi

        # 2) Ensure Offset = 0 exists (some old charts miss it)
        if ! grep -q "Offset" "$dir/notes.chart"; then
            sed -i '/Resolution/a\  Offset = 0' "$dir/notes.chart"
            log "$dir — added missing Offset field"
        fi

        # 3) Ensure PreviewEnd exists
        if ! grep -q "PreviewEnd" "$dir/notes.chart"; then
            sed -i '/PreviewStart/a\  PreviewEnd = 0' "$dir/notes.chart"
            log "$dir — added missing PreviewEnd field"
        fi

        # 4) Ensure Album exists
        if ! grep -q "Album" "$dir/notes.chart"; then
            sed -i '/Artist/a\  Album = "Generated"' "$dir/notes.chart"
            log "$dir — added missing Album field"
        fi

        # 5) Ensure Genre exists
        if ! grep -q "Genre" "$dir/notes.chart"; then
            sed -i '/MediaType/i\  Genre = "Generated"' "$dir/notes.chart"
            log "$dir — added missing Genre field"
        fi

        # 6) Ensure Year exists
        if ! grep -q "Year" "$dir/notes.chart"; then
            sed -i '/Album/a\  Year = ", "' "$dir/notes.chart"
            log "$dir — added missing Year field"
        fi
    else
        warn "$dir — no notes.chart found"
    fi

    # Patch song.ini if it exists
    if [[ -f "$dir/song.ini" ]]; then
        # Ensure delay = 0 is present
        if ! grep -q "delay" "$dir/song.ini"; then
            echo "delay = 0" >> "$dir/song.ini"
            log "$dir — added delay=0 to song.ini"
        fi

        # Ensure diff_guitar is present
        if ! grep -q "diff_guitar" "$dir/song.ini"; then
            echo "diff_guitar = 3" >> "$dir/song.ini"
            log "$dir — added diff_guitar=3 to song.ini"
        fi

        # Ensure preview_start_time is present
        if ! grep -q "preview_start_time" "$dir/song.ini"; then
            echo "preview_start_time = 0" >> "$dir/song.ini"
            log "$dir — added preview_start_time to song.ini"
        fi
    fi

    # Remove the old audio file (keep a backup just in case)
    mv "$src_audio" "$src_audio.bak"
    log "$dir — original song.$src_ext backed up to song.$src_ext.bak"

    log "$dir — ✅ done!"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

check_deps

if [[ $# -ge 1 ]]; then
    # Fix specific directories passed as arguments
    for arg in "$@"; do
        if [[ -d "$arg" ]]; then
            fix_song_dir "$arg"
        else
            err "Not a directory: $arg"
        fi
    done
else
    # No arguments — fix all song dirs under ./songs/
    songs_root="./songs"
    if [[ ! -d "$songs_root" ]]; then
        # Try relative to script location
        script_dir="$(cd "$(dirname "$0")" && pwd)"
        songs_root="$script_dir/../songs"
    fi

    if [[ ! -d "$songs_root" ]]; then
        err "Cannot find songs directory. Pass song directories as arguments."
        exit 1
    fi

    log "Scanning $songs_root for songs with unsupported audio formats..."
    found=0
    for song_dir in "$songs_root"/*/; do
        if [[ -d "$song_dir" ]]; then
            fix_song_dir "$song_dir"
            found=$((found + 1))
        fi
    done

    if [[ $found -eq 0 ]]; then
        warn "No song directories found in $songs_root"
    else
        log "Processed $found song directories"
    fi
fi
