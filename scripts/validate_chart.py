#!/usr/bin/env python3
"""
validate_chart.py — Validate and fix Clone Hero .chart files

Checks a notes.chart file for common issues that cause "Couldn't load that song"
errors in Clone Hero, and optionally applies fixes.

Usage:
    python scripts/validate_chart.py <song_directory_or_chart_file>
    python scripts/validate_chart.py "songs/Go To Sleep_2nd/"
    python scripts/validate_chart.py "songs/Go To Sleep_2nd/notes.chart"
    python scripts/validate_chart.py --fix "songs/Go To Sleep_2nd/"

Flags:
    --fix       Apply automatic fixes (writes changes to disk)
    --verbose   Show detailed diagnostic output
    --json      Output results as JSON
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESOLUTION_DEFAULT = 192
CLONE_HERO_SAFE_AUDIO = {".ogg", ".opus", ".mp3"}
CLONE_HERO_ALL_AUDIO = {".ogg", ".opus", ".mp3", ".wav", ".flac"}

REQUIRED_SONG_FIELDS = {
    "Name",
    "Resolution",
    "MusicStream",
}

RECOMMENDED_SONG_FIELDS = {
    "Artist",
    "Album",
    "Year",
    "Charter",
    "Offset",
    "Player2",
    "Difficulty",
    "PreviewStart",
    "PreviewEnd",
    "Genre",
    "MediaType",
}

VALID_DIFFICULTIES = {
    "EasySingle",
    "MediumSingle",
    "HardSingle",
    "ExpertSingle",
    "EasyDoubleBass",
    "MediumDoubleBass",
    "HardDoubleBass",
    "ExpertDoubleBass",
    "EasyDoubleRhythm",
    "MediumDoubleRhythm",
    "HardDoubleRhythm",
    "ExpertDoubleRhythm",
    "EasyDrums",
    "MediumDrums",
    "HardDrums",
    "ExpertDrums",
    "EasyKeyboard",
    "MediumKeyboard",
    "HardKeyboard",
    "ExpertKeyboard",
    "EasyGHLGuitar",
    "MediumGHLGuitar",
    "HardGHLGuitar",
    "ExpertGHLGuitar",
    "EasyGHLBass",
    "MediumGHLBass",
    "HardGHLBass",
    "ExpertGHLBass",
}

GUITAR_VALID_NOTES = set(range(8))  # 0-7 (0-4 frets, 5=force, 6=tap, 7=open)
DRUMS_VALID_NOTES = set(range(6))  # 0-5 (kick, red, yellow, blue, orange, green)
# Pro drums / accent / ghost markers go higher (32-68 range)
DRUMS_EXTENDED_NOTES = DRUMS_VALID_NOTES | set(range(32, 69))


# ---------------------------------------------------------------------------
# Diagnostic classes
# ---------------------------------------------------------------------------


class Issue:
    """Represents a single validation issue."""

    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"

    def __init__(
        self, severity: str, code: str, message: str, line: Optional[int] = None
    ):
        self.severity = severity
        self.code = code
        self.message = message
        self.line = line

    def __str__(self) -> str:
        sev_icon = {"critical": "❌", "warning": "⚠️", "info": "ℹ️"}.get(
            self.severity, "?"
        )
        loc = f" (line {self.line})" if self.line else ""
        return f"{sev_icon} [{self.code}]{loc} {self.message}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "line": self.line,
        }


class ValidationResult:
    """Aggregated validation results."""

    def __init__(self, chart_path: str):
        self.chart_path = chart_path
        self.issues: List[Issue] = []
        self.fixes_applied: List[str] = []
        self.sections_found: List[str] = []
        self.metadata: Dict[str, str] = {}

    @property
    def has_critical(self) -> bool:
        return any(i.severity == Issue.CRITICAL for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == Issue.WARNING for i in self.issues)

    @property
    def is_valid(self) -> bool:
        return not self.has_critical

    def add(
        self, severity: str, code: str, message: str, line: Optional[int] = None
    ) -> None:
        self.issues.append(Issue(severity, code, message, line))

    def summary(self) -> str:
        critical = sum(1 for i in self.issues if i.severity == Issue.CRITICAL)
        warnings = sum(1 for i in self.issues if i.severity == Issue.WARNING)
        info = sum(1 for i in self.issues if i.severity == Issue.INFO)
        status = "✅ VALID" if self.is_valid else "❌ INVALID"
        parts = [f"{status}: {self.chart_path}"]
        if critical:
            parts.append(f"  {critical} critical issue(s)")
        if warnings:
            parts.append(f"  {warnings} warning(s)")
        if info:
            parts.append(f"  {info} info")
        if self.fixes_applied:
            parts.append(f"  {len(self.fixes_applied)} fix(es) applied")
        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chart_path": self.chart_path,
            "valid": self.is_valid,
            "sections": self.sections_found,
            "metadata": self.metadata,
            "issues": [i.to_dict() for i in self.issues],
            "fixes_applied": self.fixes_applied,
        }


# ---------------------------------------------------------------------------
# Chart parser (lightweight, for validation only)
# ---------------------------------------------------------------------------


def parse_chart_sections(lines: List[str]) -> Dict[str, Tuple[int, int, List[str]]]:
    """
    Parse a .chart file into sections.

    Returns a dict mapping section name (e.g. "[Song]") to a tuple of
    (start_line, end_line, content_lines).
    """
    sections: Dict[str, Tuple[int, int, List[str]]] = {}
    current_section: Optional[str] = None
    section_start = 0
    section_lines: List[str] = []
    in_section = False

    for i, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("[") and line.endswith("]"):
            current_section = line
            section_start = i
            section_lines = []
            in_section = False
        elif line == "{":
            in_section = True
        elif line == "}":
            if current_section is not None:
                sections[current_section] = (section_start, i, section_lines)
            current_section = None
            in_section = False
            section_lines = []
        elif in_section and current_section is not None:
            section_lines.append(line)

    return sections


def parse_song_metadata(content_lines: List[str]) -> Dict[str, str]:
    """Parse key-value pairs from the [Song] section."""
    metadata: Dict[str, str] = {}
    for line in content_lines:
        match = re.match(r"^\s*(\w+)\s*=\s*(.+)$", line)
        if match:
            key = match.group(1).strip()
            value = match.group(2).strip().strip('"')
            metadata[key] = value
    return metadata


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def validate_file_basics(
    chart_path: Path, song_dir: Path, result: ValidationResult
) -> Optional[List[str]]:
    """Validate that the file exists, is readable, and has correct encoding."""
    if not chart_path.exists():
        result.add(
            Issue.CRITICAL, "FILE_MISSING", f"Chart file not found: {chart_path}"
        )
        return None

    if chart_path.stat().st_size == 0:
        result.add(Issue.CRITICAL, "FILE_EMPTY", "Chart file is empty")
        return None

    # Check encoding
    raw_bytes = chart_path.read_bytes()

    # Check for UTF-8 BOM
    if raw_bytes[:3] == b"\xef\xbb\xbf":
        result.add(
            Issue.INFO,
            "ENCODING_BOM",
            "File has UTF-8 BOM (good — matches Clone Hero convention)",
        )
        raw_bytes = raw_bytes[3:]
    else:
        result.add(
            Issue.WARNING,
            "ENCODING_NO_BOM",
            "File lacks UTF-8 BOM. Clone Hero charts typically use UTF-8 with BOM.",
        )

    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = raw_bytes.decode("latin-1")
            result.add(
                Issue.WARNING,
                "ENCODING_LATIN1",
                "File uses Latin-1 encoding instead of UTF-8",
            )
        except Exception:
            result.add(
                Issue.CRITICAL, "ENCODING_BAD", "File encoding is unrecognizable"
            )
            return None

    lines = text.splitlines()
    if len(lines) < 5:
        result.add(
            Issue.CRITICAL,
            "FILE_TOO_SHORT",
            f"Chart has only {len(lines)} lines — likely incomplete",
        )

    return lines


def validate_sections(
    lines: List[str], result: ValidationResult
) -> Dict[str, Tuple[int, int, List[str]]]:
    """Validate that required sections exist and braces are balanced."""
    sections = parse_chart_sections(lines)
    result.sections_found = list(sections.keys())

    # Check required sections
    if "[Song]" not in sections:
        result.add(Issue.CRITICAL, "MISSING_SONG", "Missing [Song] section")
    if "[SyncTrack]" not in sections:
        result.add(Issue.CRITICAL, "MISSING_SYNCTRACK", "Missing [SyncTrack] section")
    if "[Events]" not in sections:
        result.add(
            Issue.WARNING,
            "MISSING_EVENTS",
            "Missing [Events] section (optional but recommended)",
        )

    # Check for at least one note section
    note_sections = [
        s for s in sections if s not in ("[Song]", "[SyncTrack]", "[Events]")
    ]
    if not note_sections:
        result.add(
            Issue.CRITICAL,
            "NO_NOTE_SECTIONS",
            "No note sections found (need at least one difficulty)",
        )
    else:
        for ns in note_sections:
            section_name = ns.strip("[]")
            if section_name not in VALID_DIFFICULTIES:
                result.add(Issue.INFO, "UNKNOWN_SECTION", f"Non-standard section: {ns}")

    # Check brace balance
    open_count = sum(1 for line in lines if line.strip() == "{")
    close_count = sum(1 for line in lines if line.strip() == "}")
    if open_count != close_count:
        result.add(
            Issue.CRITICAL,
            "BRACE_MISMATCH",
            f"Unbalanced braces: {open_count} opening vs {close_count} closing",
        )

    return sections


def validate_song_section(
    sections: Dict[str, Tuple[int, int, List[str]]],
    song_dir: Path,
    result: ValidationResult,
) -> Dict[str, str]:
    """Validate the [Song] metadata section."""
    if "[Song]" not in sections:
        return {}

    start_line, _, content = sections["[Song]"]
    metadata = parse_song_metadata(content)
    result.metadata = metadata

    # Check required fields
    for field in REQUIRED_SONG_FIELDS:
        if field not in metadata:
            result.add(
                Issue.CRITICAL,
                f"MISSING_{field.upper()}",
                f"Missing required field '{field}' in [Song] section",
                start_line,
            )

    # Check recommended fields
    for field in RECOMMENDED_SONG_FIELDS:
        if field not in metadata:
            result.add(
                Issue.WARNING,
                f"MISSING_{field.upper()}",
                f"Missing recommended field '{field}' in [Song] section",
                start_line,
            )

    # Check Resolution
    resolution_str = metadata.get("Resolution", "")
    if resolution_str:
        try:
            resolution = int(resolution_str)
            if resolution not in (192, 480, 960):
                result.add(
                    Issue.WARNING,
                    "ODD_RESOLUTION",
                    f"Unusual resolution {resolution} (standard is 192)",
                )
        except ValueError:
            result.add(
                Issue.CRITICAL,
                "BAD_RESOLUTION",
                f"Resolution is not a number: '{resolution_str}'",
            )

    # Check MusicStream
    music_stream = metadata.get("MusicStream", "")
    if music_stream:
        audio_path = song_dir / music_stream
        ext = Path(music_stream).suffix.lower()

        if not audio_path.exists():
            result.add(
                Issue.CRITICAL, "AUDIO_MISSING", f"Audio file not found: {music_stream}"
            )
        elif ext not in CLONE_HERO_ALL_AUDIO:
            result.add(
                Issue.CRITICAL, "AUDIO_FORMAT_UNKNOWN", f"Unknown audio format: {ext}"
            )
        elif ext not in CLONE_HERO_SAFE_AUDIO:
            result.add(
                Issue.WARNING,
                "AUDIO_FORMAT_UNSAFE",
                f"Audio format '{ext}' is not reliably supported by all Clone Hero versions. "
                f"Consider converting to .ogg or .opus",
            )

    # Check Offset
    offset_str = metadata.get("Offset", "")
    if offset_str:
        try:
            float(offset_str)
        except ValueError:
            result.add(
                Issue.WARNING, "BAD_OFFSET", f"Offset is not a number: '{offset_str}'"
            )

    return metadata


def validate_sync_track(
    sections: Dict[str, Tuple[int, int, List[str]]], result: ValidationResult
) -> None:
    """Validate the [SyncTrack] section for BPM and time signature entries."""
    if "[SyncTrack]" not in sections:
        return

    start_line, _, content = sections["[SyncTrack]"]
    has_initial_ts = False
    has_initial_bpm = False
    bpm_count = 0
    ts_count = 0
    prev_tick = -1

    for i, line in enumerate(content):
        line_num = start_line + i + 2  # +2 for section header and opening brace

        parts = line.split()
        if len(parts) < 4:
            result.add(
                Issue.WARNING,
                "SYNC_BAD_LINE",
                f"Malformed SyncTrack line: '{line}'",
                line_num,
            )
            continue

        try:
            tick = int(parts[0])
        except ValueError:
            result.add(
                Issue.WARNING,
                "SYNC_BAD_TICK",
                f"Non-numeric tick: '{parts[0]}'",
                line_num,
            )
            continue

        if tick < prev_tick:
            result.add(
                Issue.WARNING,
                "SYNC_OUT_OF_ORDER",
                f"Tick {tick} is before previous tick {prev_tick}",
                line_num,
            )
        prev_tick = tick

        event_type = parts[2].upper()

        if event_type == "TS":
            ts_count += 1
            if tick == 0:
                has_initial_ts = True
            try:
                numerator = int(parts[3])
                if numerator < 1 or numerator > 32:
                    result.add(
                        Issue.WARNING,
                        "SYNC_ODD_TS",
                        f"Unusual time signature numerator: {numerator}",
                        line_num,
                    )
            except ValueError:
                result.add(
                    Issue.WARNING,
                    "SYNC_BAD_TS",
                    f"Non-numeric TS value: '{parts[3]}'",
                    line_num,
                )

        elif event_type == "B":
            bpm_count += 1
            if tick == 0:
                has_initial_bpm = True
            try:
                milli_bpm = int(parts[3])
                bpm = milli_bpm / 1000.0
                if bpm < 10 or bpm > 1000:
                    result.add(
                        Issue.WARNING,
                        "SYNC_EXTREME_BPM",
                        f"Extreme BPM value: {bpm:.1f} at tick {tick}",
                        line_num,
                    )
                elif bpm > 400:
                    result.add(
                        Issue.INFO,
                        "SYNC_HIGH_BPM",
                        f"Very high BPM: {bpm:.1f} at tick {tick}",
                        line_num,
                    )
            except ValueError:
                result.add(
                    Issue.CRITICAL,
                    "SYNC_BAD_BPM",
                    f"Non-numeric BPM value: '{parts[3]}'",
                    line_num,
                )

    if not has_initial_ts:
        result.add(
            Issue.CRITICAL,
            "SYNC_NO_INITIAL_TS",
            "Missing initial time signature at tick 0",
        )
    if not has_initial_bpm:
        result.add(
            Issue.CRITICAL, "SYNC_NO_INITIAL_BPM", "Missing initial BPM at tick 0"
        )

    if bpm_count > 50:
        result.add(
            Issue.WARNING,
            "SYNC_TOO_MANY_BPM",
            f"Excessive tempo changes ({bpm_count}). "
            f"This may indicate bad beat analysis. Consider re-generating.",
        )

    result.add(
        Issue.INFO, "SYNC_STATS", f"{bpm_count} BPM markers, {ts_count} TS markers"
    )


def validate_note_sections(
    sections: Dict[str, Tuple[int, int, List[str]]], result: ValidationResult
) -> None:
    """Validate note data in difficulty sections."""
    for section_name, (start_line, _, content) in sections.items():
        clean_name = section_name.strip("[]")
        if clean_name in ("Song", "SyncTrack", "Events"):
            continue

        note_count = 0
        sp_count = 0
        event_count = 0
        prev_tick = -1
        is_drums = "Drums" in clean_name

        valid_notes = DRUMS_EXTENDED_NOTES if is_drums else GUITAR_VALID_NOTES

        for i, line in enumerate(content):
            line_num = start_line + i + 2
            parts = line.split()
            if len(parts) < 5:
                if len(parts) >= 3 and parts[2].upper() == "E":
                    event_count += 1
                    continue
                result.add(
                    Issue.WARNING,
                    "NOTE_BAD_LINE",
                    f"Malformed line in {section_name}: '{line}'",
                    line_num,
                )
                continue

            try:
                tick = int(parts[0])
            except ValueError:
                continue

            if tick < prev_tick:
                result.add(
                    Issue.WARNING,
                    "NOTE_OUT_OF_ORDER",
                    f"Tick {tick} < previous {prev_tick} in {section_name}",
                    line_num,
                )
            prev_tick = tick

            event_type = parts[2].upper()

            if event_type == "N":
                note_count += 1
                try:
                    note_num = int(parts[3])
                    sustain = int(parts[4])
                    if note_num not in valid_notes:
                        result.add(
                            Issue.WARNING,
                            "NOTE_INVALID_NUM",
                            f"Invalid note number {note_num} in {section_name}",
                            line_num,
                        )
                    if sustain < 0:
                        result.add(
                            Issue.WARNING,
                            "NOTE_NEG_SUSTAIN",
                            f"Negative sustain {sustain} in {section_name}",
                            line_num,
                        )
                except ValueError:
                    result.add(
                        Issue.WARNING,
                        "NOTE_BAD_VALUES",
                        f"Non-numeric note values in {section_name}",
                        line_num,
                    )

            elif event_type == "S":
                sp_count += 1
                try:
                    sp_type = int(parts[3])
                    sp_length = int(parts[4])
                    if sp_type != 2:
                        result.add(
                            Issue.INFO,
                            "SP_NONSTANDARD",
                            f"Non-standard SP type {sp_type} in {section_name}",
                            line_num,
                        )
                    if sp_length <= 0:
                        result.add(
                            Issue.WARNING,
                            "SP_ZERO_LENGTH",
                            f"Star power with length {sp_length} in {section_name}",
                            line_num,
                        )
                except ValueError:
                    pass

        if note_count == 0:
            result.add(
                Issue.WARNING, "EMPTY_SECTION", f"Section {section_name} has no notes"
            )
        else:
            result.add(
                Issue.INFO,
                "SECTION_STATS",
                f"{section_name}: {note_count} notes, {sp_count} star power, {event_count} events",
            )


def validate_events(
    sections: Dict[str, Tuple[int, int, List[str]]], result: ValidationResult
) -> None:
    """Validate the [Events] section."""
    if "[Events]" not in sections:
        return

    start_line, _, content = sections["[Events]"]
    section_count = 0
    lyric_count = 0
    phrase_start_count = 0
    phrase_end_count = 0
    prev_tick = -1
    out_of_order_count = 0

    for i, line in enumerate(content):
        line_num = start_line + i + 2  # +2 for section header and opening brace

        if '"section ' in line:
            section_count += 1
        elif '"lyric ' in line:
            lyric_count += 1
        elif '"phrase_start"' in line:
            phrase_start_count += 1
        elif '"phrase_end"' in line:
            phrase_end_count += 1

        # Check tick ordering — Clone Hero rejects charts with out-of-order
        # events as "corrupt (or broken)".
        parts = line.split()
        if len(parts) >= 3:
            try:
                tick = int(parts[0])
                if tick < prev_tick:
                    out_of_order_count += 1
                    if out_of_order_count <= 3:
                        result.add(
                            Issue.CRITICAL,
                            "EVENT_OUT_OF_ORDER",
                            f"Event tick {tick} is before previous tick {prev_tick} "
                            f"— Clone Hero will reject the chart as corrupt",
                            line_num,
                        )
                prev_tick = tick
            except ValueError:
                pass

    if out_of_order_count > 3:
        result.add(
            Issue.CRITICAL,
            "EVENT_OUT_OF_ORDER_MANY",
            f"{out_of_order_count} total out-of-order events in [Events] "
            f"(only first 3 shown). Section markers and lyric events must be "
            f"merged and sorted by tick.",
        )

    if section_count == 0:
        result.add(
            Issue.WARNING,
            "NO_SECTIONS",
            "No section markers in [Events] (practice mode sections won't work)",
        )

    if phrase_start_count != phrase_end_count:
        result.add(
            Issue.WARNING,
            "PHRASE_MISMATCH",
            f"Mismatched phrase markers: {phrase_start_count} starts vs {phrase_end_count} ends",
        )

    result.add(
        Issue.INFO,
        "EVENT_STATS",
        f"{section_count} sections, {lyric_count} lyrics, "
        f"{phrase_start_count} phrase markers",
    )


def validate_audio_files(
    song_dir: Path, metadata: Dict[str, str], result: ValidationResult
) -> None:
    """Check that audio files exist and are in a supported format."""
    music_stream = metadata.get("MusicStream", "")

    # Look for any audio files in the directory
    audio_files = []
    for ext in CLONE_HERO_ALL_AUDIO:
        for f in song_dir.glob(f"*{ext}"):
            audio_files.append(f)

    if not audio_files and not music_stream:
        result.add(Issue.CRITICAL, "NO_AUDIO", "No audio files found in song directory")
        return

    for af in audio_files:
        if af.suffix.lower() not in CLONE_HERO_SAFE_AUDIO:
            result.add(
                Issue.WARNING,
                "AUDIO_COMPAT",
                f"Audio file '{af.name}' uses format '{af.suffix}' which may not be supported. "
                f'Convert to .ogg: ffmpeg -i "{af.name}" -c:a libvorbis -b:a 192k song.ogg',
            )


def validate_song_ini(song_dir: Path, result: ValidationResult) -> None:
    """Validate song.ini if it exists."""
    ini_path = song_dir / "song.ini"
    if not ini_path.exists():
        result.add(
            Issue.WARNING, "NO_SONG_INI", "No song.ini found (optional but recommended)"
        )
        return

    try:
        ini_text = ini_path.read_text(encoding="utf-8-sig")
    except Exception:
        ini_text = ini_path.read_text(encoding="latin-1")

    ini_lines = ini_text.splitlines()

    has_song_section = False
    has_name = False
    has_artist = False
    has_diff_guitar = False
    has_delay = False

    for line in ini_lines:
        stripped = line.strip().lower()
        if stripped == "[song]":
            has_song_section = True
        elif stripped.startswith("name"):
            has_name = True
        elif stripped.startswith("artist"):
            has_artist = True
        elif stripped.startswith("diff_guitar"):
            has_diff_guitar = True
        elif stripped.startswith("delay"):
            has_delay = True

    if not has_song_section:
        result.add(
            Issue.WARNING, "INI_NO_SECTION", "song.ini missing [song] section header"
        )
    if not has_name:
        result.add(Issue.WARNING, "INI_NO_NAME", "song.ini missing 'name' field")
    if not has_diff_guitar:
        result.add(
            Issue.WARNING,
            "INI_NO_DIFF",
            "song.ini missing 'diff_guitar' — Clone Hero may not display difficulty rating",
        )
    if not has_delay:
        result.add(
            Issue.INFO, "INI_NO_DELAY", "song.ini missing 'delay' field (defaults to 0)"
        )


# ---------------------------------------------------------------------------
# Fixers
# ---------------------------------------------------------------------------


def apply_fixes(
    chart_path: Path,
    song_dir: Path,
    lines: List[str],
    sections: Dict[str, Tuple[int, int, List[str]]],
    metadata: Dict[str, str],
    result: ValidationResult,
) -> List[str]:
    """Apply automatic fixes to the chart lines. Returns modified lines."""
    fixed_lines = list(lines)
    changes: List[str] = []

    # Fix 1: Add UTF-8 BOM if missing
    raw = chart_path.read_bytes()
    needs_bom = raw[:3] != b"\xef\xbb\xbf"

    # Fix 2: Add missing MusicStream
    if "MusicStream" not in metadata and "[Song]" in sections:
        # Find the best available audio file
        audio_file = None
        for ext in [".ogg", ".opus", ".mp3", ".wav", ".flac"]:
            candidate = song_dir / f"song{ext}"
            if candidate.exists():
                audio_file = f"song{ext}"
                break

        if audio_file:
            start, end, _ = sections["[Song]"]
            # Insert before the closing brace
            insert_line = f'  MusicStream = "{audio_file}"'
            for i in range(len(fixed_lines) - 1, start, -1):
                if fixed_lines[i].strip() == "}":
                    fixed_lines.insert(i, insert_line)
                    changes.append(f'Added MusicStream = "{audio_file}"')
                    break

    # Fix 3: Add missing Offset
    if "Offset" not in metadata and "[Song]" in sections:
        start, end, _ = sections["[Song]"]
        for i in range(start, min(start + 30, len(fixed_lines))):
            if "Resolution" in fixed_lines[i]:
                fixed_lines.insert(i, "  Offset = 0")
                changes.append("Added Offset = 0")
                break

    # Fix 4: Add missing PreviewEnd
    if (
        "PreviewEnd" not in metadata
        and "PreviewStart" in metadata
        and "[Song]" in sections
    ):
        start, _, _ = sections["[Song]"]
        for i in range(start, min(start + 30, len(fixed_lines))):
            if "PreviewStart" in fixed_lines[i]:
                fixed_lines.insert(i + 1, "  PreviewEnd = 0")
                changes.append("Added PreviewEnd = 0")
                break

    # Fix 5: Add missing Album
    if "Album" not in metadata and "[Song]" in sections:
        start, _, _ = sections["[Song]"]
        for i in range(start, min(start + 30, len(fixed_lines))):
            if "Artist" in fixed_lines[i]:
                fixed_lines.insert(i + 1, '  Album = "Generated"')
                changes.append('Added Album = "Generated"')
                break

    # Fix 6: Sort out-of-order events in [Events] section
    # Clone Hero rejects charts with out-of-order events as "corrupt".
    if "[Events]" in sections:
        ev_start, ev_end, ev_content = sections["[Events]"]
        # Check if any events are out of order
        ev_ticks = []
        needs_sort = False
        prev_ev_tick = -1
        for ev_line in ev_content:
            ev_parts = ev_line.split()
            if len(ev_parts) >= 3:
                try:
                    t = int(ev_parts[0])
                    if t < prev_ev_tick:
                        needs_sort = True
                    prev_ev_tick = t
                    ev_ticks.append(t)
                except ValueError:
                    pass

        if needs_sort:
            # Find the [Events] block boundaries in fixed_lines
            ev_block_start = None
            ev_block_end = None
            for i in range(len(fixed_lines)):
                if fixed_lines[i].strip() == "[Events]":
                    ev_block_start = i + 2  # skip "[Events]" and "{"
                elif ev_block_start is not None and fixed_lines[i].strip() == "}":
                    ev_block_end = i
                    break

            if ev_block_start is not None and ev_block_end is not None:
                # Extract event lines, sort by tick, replace
                event_block = fixed_lines[ev_block_start:ev_block_end]
                sorted_events = []
                for el in event_block:
                    el_parts = el.strip().split()
                    try:
                        sort_tick = int(el_parts[0]) if len(el_parts) >= 3 else 0
                    except ValueError:
                        sort_tick = 0
                    sorted_events.append((sort_tick, el))
                sorted_events.sort(key=lambda x: x[0])
                fixed_lines[ev_block_start:ev_block_end] = [
                    se[1] for se in sorted_events
                ]
                changes.append(
                    f"Sorted {len(sorted_events)} events in [Events] by tick "
                    f"(was out-of-order — Clone Hero would reject as corrupt)"
                )

    # Fix 7: Add missing Genre
    if "Genre" not in metadata and "[Song]" in sections:
        start, _, _ = sections["[Song]"]
        for i in range(start, min(start + 30, len(fixed_lines))):
            if "MediaType" in fixed_lines[i]:
                fixed_lines.insert(i, '  Genre = "Generated"')
                changes.append('Added Genre = "Generated"')
                break

    # Write fixed file
    if changes or needs_bom:
        backup_path = chart_path.with_suffix(".chart.bak")
        if not backup_path.exists():
            shutil.copy2(chart_path, backup_path)
            changes.insert(0, f"Created backup at {backup_path.name}")

        content = "\n".join(fixed_lines)
        if not content.endswith("\n"):
            content += "\n"

        # Write with BOM
        with open(chart_path, "wb") as f:
            f.write(b"\xef\xbb\xbf")
            f.write(content.encode("utf-8"))

        if needs_bom:
            changes.append("Added UTF-8 BOM")

    result.fixes_applied = changes
    return fixed_lines


def try_convert_audio(song_dir: Path, result: ValidationResult) -> bool:
    """Attempt to convert unsupported audio to OGG using ffmpeg."""
    for ext in (".flac", ".wav", ".aac", ".m4a", ".wma"):
        src = song_dir / f"song{ext}"
        if src.exists():
            dest = song_dir / "song.ogg"
            if dest.exists():
                result.add(
                    Issue.INFO,
                    "AUDIO_OGG_EXISTS",
                    "song.ogg already exists alongside unsupported format",
                )
                return True

            # Check for ffmpeg
            try:
                subprocess.run(
                    ["ffmpeg", "-version"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                result.add(
                    Issue.WARNING,
                    "NO_FFMPEG",
                    f"ffmpeg not found — cannot auto-convert {src.name} to OGG. "
                    f"Install ffmpeg and re-run with --fix",
                )
                return False

            print(f"  Converting {src.name} → song.ogg ...")
            try:
                proc = subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        str(src),
                        "-vn",
                        "-c:a",
                        "libvorbis",
                        "-b:a",
                        "192k",
                        str(dest),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=120,
                )
                if proc.returncode == 0 and dest.exists():
                    result.fixes_applied.append(f"Converted {src.name} → song.ogg")
                    # Backup original
                    backup = src.with_suffix(src.suffix + ".bak")
                    src.rename(backup)
                    result.fixes_applied.append(f"Backed up {src.name} → {backup.name}")
                    return True
                else:
                    stderr_text = proc.stderr.decode(errors="replace")[-300:]
                    result.add(
                        Issue.WARNING,
                        "CONVERT_FAIL",
                        f"ffmpeg conversion failed: {stderr_text}",
                    )
                    return False
            except Exception as e:
                result.add(Issue.WARNING, "CONVERT_ERROR", f"Conversion error: {e}")
                return False

    return False


# ---------------------------------------------------------------------------
# Main validation entry point
# ---------------------------------------------------------------------------


def validate_chart(
    chart_path: Path,
    song_dir: Optional[Path] = None,
    fix: bool = False,
    verbose: bool = False,
) -> ValidationResult:
    """
    Run all validators on a chart file.

    Parameters
    ----------
    chart_path : Path
        Path to the notes.chart file.
    song_dir : Path, optional
        Song directory (defaults to chart_path.parent).
    fix : bool
        If True, attempt to apply automatic fixes.
    verbose : bool
        If True, include extra diagnostic detail.

    Returns
    -------
    ValidationResult
    """
    if song_dir is None:
        song_dir = chart_path.parent

    result = ValidationResult(str(chart_path))

    # Step 1: File basics
    lines = validate_file_basics(chart_path, song_dir, result)
    if lines is None:
        return result

    # Step 2: Section structure
    sections = validate_sections(lines, result)

    # Step 3: [Song] metadata
    metadata = validate_song_section(sections, song_dir, result)

    # Step 4: [SyncTrack]
    validate_sync_track(sections, result)

    # Step 5: Note sections
    validate_note_sections(sections, result)

    # Step 6: [Events]
    validate_events(sections, result)

    # Step 7: Audio files
    validate_audio_files(song_dir, metadata, result)

    # Step 8: song.ini
    validate_song_ini(song_dir, result)

    # Step 9: Apply fixes if requested
    if fix:
        # Try audio conversion first
        for ext in (".flac", ".wav"):
            if (song_dir / f"song{ext}").exists() and not (
                song_dir / "song.ogg"
            ).exists():
                converted = try_convert_audio(song_dir, result)
                if converted:
                    # Re-parse to pick up new audio file reference
                    metadata["MusicStream_needs_update"] = "song.ogg"
                break

        # Apply chart fixes
        apply_fixes(chart_path, song_dir, lines, sections, metadata, result)

        # If audio was converted, update MusicStream in the fixed file
        if "MusicStream_needs_update" in metadata:
            chart_text = chart_path.read_text(encoding="utf-8-sig")
            for ext in (".flac", ".wav", ".aac", ".m4a", ".wma"):
                chart_text = chart_text.replace(
                    f'MusicStream = "song{ext}"', 'MusicStream = "song.ogg"'
                )
            with open(chart_path, "wb") as f:
                f.write(b"\xef\xbb\xbf")
                f.write(chart_text.encode("utf-8"))
            result.fixes_applied.append("Updated MusicStream to song.ogg")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and fix Clone Hero .chart files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "path", nargs="+", help="Song directory or notes.chart file(s) to validate"
    )
    parser.add_argument("--fix", action="store_true", help="Attempt to auto-fix issues")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show all diagnostic messages including info-level",
    )
    parser.add_argument("--json", action="store_true", help="Output results as JSON")

    args = parser.parse_args()
    all_results: List[ValidationResult] = []

    for target in args.path:
        target_path = Path(target)

        if target_path.is_dir():
            chart_path = target_path / "notes.chart"
            song_dir = target_path
        elif target_path.is_file() and target_path.suffix == ".chart":
            chart_path = target_path
            song_dir = target_path.parent
        else:
            print(f"❌ Not a valid target: {target}")
            continue

        result = validate_chart(
            chart_path, song_dir, fix=args.fix, verbose=args.verbose
        )
        all_results.append(result)

    if args.json:
        output = [r.to_dict() for r in all_results]
        print(json.dumps(output, indent=2))
    else:
        for result in all_results:
            print()
            print("=" * 60)
            print(result.summary())
            print("-" * 60)
            for issue in result.issues:
                if issue.severity == Issue.INFO and not args.verbose:
                    continue
                print(f"  {issue}")
            if result.fixes_applied:
                print()
                print("  Fixes applied:")
                for fix_msg in result.fixes_applied:
                    print(f"    ✅ {fix_msg}")
            print()

    # Return non-zero if any chart has critical issues (after fixes)
    if any(r.has_critical for r in all_results):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
