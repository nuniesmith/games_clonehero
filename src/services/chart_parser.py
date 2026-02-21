"""
Clone Hero Content Manager - Chart Parser Service

Parses Clone Hero ``notes.chart`` files into structured Python dicts that can
be serialised to JSON for the web-based chart viewer / editor.

The ``.chart`` format is a plain-text INI-like file with bracketed sections:

    [Song]        – metadata key/value pairs
    [SyncTrack]   – tempo (B) and time-signature (TS) events
    [Events]      – section markers, lyrics, and other events
    [ExpertSingle] / [HardSingle] / [MediumSingle] / [EasySingle]
                  – note data for each difficulty

This module provides:
    - Full round-trip parsing: chart file → structured dict → chart file
    - Individual section parsers with robust error handling
    - Tick ↔ seconds conversion using the parsed tempo map
    - Utility helpers for the viewer / editor API layer
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_RESOLUTION = 192  # ticks per quarter note

# Canonical section names that correspond to guitar difficulties
DIFFICULTY_SECTION_MAP: dict[str, str] = {
    "ExpertSingle": "expert",
    "HardSingle": "hard",
    "MediumSingle": "medium",
    "EasySingle": "easy",
}

# Reverse map: difficulty label → chart section name
DIFFICULTY_TO_SECTION: dict[str, str] = {
    v: k for k, v in DIFFICULTY_SECTION_MAP.items()
}

# Lane names for display
LANE_NAMES: dict[int, str] = {
    0: "Green",
    1: "Red",
    2: "Yellow",
    3: "Blue",
    4: "Orange",
    5: "Force",
    6: "Tap",
    7: "Open",
}

# Note type constants
NOTE_TYPE_NORMAL = "note"
NOTE_TYPE_HOPO = "hopo"
NOTE_TYPE_TAP = "tap"
NOTE_TYPE_STAR_POWER = "star_power"

# Regex patterns for parsing chart lines
_RE_SECTION_HEADER = re.compile(r"^\[(.+)\]\s*$")
_RE_KV_QUOTED = re.compile(r'^\s*(\S+)\s*=\s*"(.*)"\s*$')
_RE_KV_UNQUOTED = re.compile(r"^\s*(\S+)\s*=\s*(.*?)\s*$")
_RE_EVENT = re.compile(r"^\s*(\d+)\s*=\s*(\S+)\s*(.*?)\s*$")


# ---------------------------------------------------------------------------
# Tempo map utilities
# ---------------------------------------------------------------------------


class TempoMap:
    """
    Maintains a sorted list of tempo markers and provides tick ↔ time
    conversion.

    Each entry is ``(tick, bpm)`` where *bpm* is the actual BPM (i.e. the
    chart's milli-BPM value divided by 1000).
    """

    resolution: int

    def __init__(
        self, markers: list[tuple[int, float]], resolution: int = DEFAULT_RESOLUTION
    ):
        self.resolution = resolution
        # Sort by tick ascending; ensure at least one marker at tick 0
        if not markers:
            markers = [(0, 120.0)]
        self.markers: list[tuple[int, float]] = sorted(markers, key=lambda m: m[0])
        if self.markers[0][0] != 0:
            self.markers.insert(0, (0, 120.0))

        # Pre-compute cumulative time at each marker for fast lookup
        self._times: list[float] = [0.0]
        for i in range(1, len(self.markers)):
            prev_tick, prev_bpm = self.markers[i - 1]
            curr_tick, _ = self.markers[i]
            dt = self._ticks_to_seconds(curr_tick - prev_tick, prev_bpm)
            self._times.append(self._times[-1] + dt)

    def _ticks_to_seconds(self, ticks: int, bpm: float) -> float:
        """Convert a tick delta to seconds at the given BPM."""
        if bpm <= 0:
            return 0.0
        beats = ticks / self.resolution
        return beats * (60.0 / bpm)

    def tick_to_time(self, tick: int) -> float:
        """Convert an absolute tick position to a time in seconds."""
        # Binary-search for the relevant tempo segment
        lo, hi = 0, len(self.markers) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.markers[mid][0] <= tick:
                lo = mid
            else:
                hi = mid - 1

        seg_tick, seg_bpm = self.markers[lo]
        seg_time = self._times[lo]
        dt = self._ticks_to_seconds(tick - seg_tick, seg_bpm)
        return seg_time + dt

    def time_to_tick(self, time_s: float) -> int:
        """Convert a time in seconds to the nearest absolute tick."""
        if time_s <= 0:
            return 0

        # Find the segment that contains this time
        lo, hi = 0, len(self._times) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._times[mid] <= time_s:
                lo = mid
            else:
                hi = mid - 1

        seg_tick, seg_bpm = self.markers[lo]
        seg_time = self._times[lo]
        remaining = time_s - seg_time

        if seg_bpm <= 0:
            return seg_tick

        beats = remaining / (60.0 / seg_bpm)
        return seg_tick + int(round(beats * self.resolution))

    @property
    def total_time(self) -> float:
        """Estimated total time based on the last marker (not authoritative)."""
        if not self.markers:
            return 0.0
        return self._times[-1]

    def bpm_at_tick(self, tick: int) -> float:
        """Return the BPM in effect at the given tick."""
        result_bpm = self.markers[0][1]
        for t, bpm in self.markers:
            if t <= tick:
                result_bpm = bpm
            else:
                break
        return result_bpm


# ---------------------------------------------------------------------------
# Low-level line parsers
# ---------------------------------------------------------------------------


def _parse_song_section(lines: list[str]) -> dict[str, Any]:
    """Parse the [Song] section into a metadata dict."""
    meta: dict[str, Any] = {}
    for line in lines:
        m = _RE_KV_QUOTED.match(line)
        if m:
            key, value = m.group(1), m.group(2)
            meta[key] = value
            continue
        m = _RE_KV_UNQUOTED.match(line)
        if m:
            key, value = m.group(1), m.group(2)
            # Try to parse numeric values
            try:
                if "." in value:
                    meta[key] = float(value)
                else:
                    meta[key] = int(value)
            except (ValueError, TypeError):
                meta[key] = value
    return meta


def _parse_sync_track(lines: list[str]) -> dict[str, Any]:
    """
    Parse the [SyncTrack] section.

    Returns a dict with:
        tempo_markers : list of {tick, bpm, milli_bpm}
        time_signatures : list of {tick, numerator, denominator_power}
    """
    tempo_markers: list[dict[str, Any]] = []
    time_signatures: list[dict[str, Any]] = []

    for line in lines:
        m = _RE_EVENT.match(line)
        if not m:
            continue
        tick = int(m.group(1))
        event_type = m.group(2)
        args = m.group(3).strip()

        if event_type == "B":
            # Tempo marker: B {milli_bpm}
            try:
                milli_bpm = int(args)
                bpm = milli_bpm / 1000.0
                tempo_markers.append(
                    {
                        "tick": tick,
                        "bpm": round(bpm, 3),
                        "milli_bpm": milli_bpm,
                    }
                )
            except (ValueError, TypeError):
                logger.warning("⚠️ Malformed tempo marker at tick {}: {}", tick, args)

        elif event_type == "TS":
            # Time signature: TS {numerator} [denominator_power]
            parts = args.split()
            try:
                numerator = int(parts[0])
                denom_power = int(parts[1]) if len(parts) > 1 else 2
                time_signatures.append(
                    {
                        "tick": tick,
                        "numerator": numerator,
                        "denominator": 2**denom_power,
                        "denominator_power": denom_power,
                    }
                )
            except (ValueError, IndexError, TypeError):
                logger.warning("⚠️ Malformed time signature at tick {}: {}", tick, args)

    return {
        "tempo_markers": tempo_markers,
        "time_signatures": time_signatures,
    }


def _parse_events_section(lines: list[str]) -> list[dict[str, Any]]:
    """
    Parse the [Events] section.

    Returns a list of event dicts:
        {tick, type, value}

    Known event types:
        section     — "section Intro", "section Verse 1", etc.
        phrase_start — lyric phrase boundary start
        phrase_end   — lyric phrase boundary end
        lyric        — individual lyric word
        other        — anything else
    """
    events: list[dict[str, Any]] = []

    for line in lines:
        m = _RE_EVENT.match(line)
        if not m:
            continue
        tick = int(m.group(1))
        event_type = m.group(2)
        args = m.group(3).strip()

        if event_type == "E":
            # Strip surrounding quotes
            raw = args.strip('"').strip()

            if raw.startswith("section "):
                events.append(
                    {
                        "tick": tick,
                        "type": "section",
                        "value": raw[len("section ") :],
                    }
                )
            elif raw == "phrase_start":
                events.append(
                    {
                        "tick": tick,
                        "type": "phrase_start",
                        "value": "",
                    }
                )
            elif raw == "phrase_end":
                events.append(
                    {
                        "tick": tick,
                        "type": "phrase_end",
                        "value": "",
                    }
                )
            elif raw.startswith("lyric "):
                events.append(
                    {
                        "tick": tick,
                        "type": "lyric",
                        "value": raw[len("lyric ") :],
                    }
                )
            else:
                events.append(
                    {
                        "tick": tick,
                        "type": "other",
                        "value": raw,
                    }
                )

    return events


def _parse_note_section(lines: list[str]) -> dict[str, Any]:
    """
    Parse a note section (e.g. [ExpertSingle]).

    Returns a dict with:
        notes       : list of note dicts
        star_power  : list of {tick, duration}
        raw_events  : count of raw parsed events

    Each note dict:
        {tick, lane, duration, is_hopo, is_tap, time_s (filled later)}
    """
    # First pass: collect raw events grouped by tick
    tick_events: dict[int, list[tuple[str, int, int]]] = {}

    for line in lines:
        m = _RE_EVENT.match(line)
        if not m:
            continue
        tick = int(m.group(1))
        event_type = m.group(2)
        args = m.group(3).strip()

        if event_type in ("N", "S"):
            parts = args.split()
            if len(parts) >= 2:
                try:
                    code = int(parts[0])
                    duration = int(parts[1])
                    if tick not in tick_events:
                        tick_events[tick] = []
                    tick_events[tick].append((event_type, code, duration))
                except (ValueError, TypeError):
                    pass

    # Second pass: assemble notes and star power
    notes: list[dict[str, Any]] = []
    star_power: list[dict[str, Any]] = []
    total_events = 0

    for tick in sorted(tick_events.keys()):
        events = tick_events[tick]
        total_events += len(events)

        # Check for modifier flags at this tick
        has_hopo = False
        has_tap = False
        note_events: list[tuple[int, int]] = []
        sp_events: list[tuple[int, int]] = []

        for evt_type, code, duration in events:
            if evt_type == "N":
                if code == 5:
                    has_hopo = True
                elif code == 6:
                    has_tap = True
                elif 0 <= code <= 4 or code == 7:
                    note_events.append((code, duration))
            elif evt_type == "S":
                if code == 2:
                    sp_events.append((code, duration))

        # Create note entries
        for lane, dur in note_events:
            notes.append(
                {
                    "tick": tick,
                    "lane": lane,
                    "lane_name": LANE_NAMES.get(lane, f"Lane {lane}"),
                    "duration": dur,
                    "is_hopo": has_hopo,
                    "is_tap": has_tap,
                    "time_s": 0.0,  # will be filled by tempo map
                }
            )

        for _, dur in sp_events:
            star_power.append(
                {
                    "tick": tick,
                    "duration": dur,
                }
            )

    return {
        "notes": notes,
        "star_power": star_power,
        "raw_events": total_events,
    }


# ---------------------------------------------------------------------------
# High-level chart parsing
# ---------------------------------------------------------------------------


def _split_sections(text: str) -> dict[str, list[str]]:
    """
    Split a chart file's text into named sections.

    Returns a dict mapping section name to the list of lines inside
    the braces (excluding the ``{`` and ``}`` delimiters).
    """
    sections: dict[str, list[str]] = {}
    current_section: str | None = None
    current_lines: list[str] = []
    inside = False

    for raw_line in text.splitlines():
        line = raw_line.strip()

        # Check for section header
        m = _RE_SECTION_HEADER.match(line)
        if m:
            # Save previous section if any
            if current_section is not None and current_lines:
                sections[current_section] = current_lines
            current_section = m.group(1)
            current_lines = []
            inside = False
            continue

        if line == "{":
            inside = True
            continue

        if line == "}":
            if current_section is not None:
                sections[current_section] = current_lines
                current_section = None
                current_lines = []
            inside = False
            continue

        if inside and current_section is not None:
            current_lines.append(raw_line)

    # Handle trailing section without closing brace (lenient)
    if current_section is not None and current_lines:
        sections[current_section] = current_lines

    return sections


def parse_chart_file(path: str | Path) -> dict[str, Any]:
    """
    Parse a Clone Hero ``.chart`` file into a structured dict.

    Parameters
    ----------
    path : str or Path
        Path to the ``notes.chart`` file.

    Returns
    -------
    dict
        A dict with keys:

        - **song** *(dict)* – [Song] metadata (Name, Artist, Resolution, etc.)
        - **sync_track** *(dict)* – tempo_markers and time_signatures
        - **events** *(list)* – section markers, lyrics, etc.
        - **difficulties** *(dict)* – keyed by difficulty label (easy/medium/hard/expert),
          each containing ``notes``, ``star_power``, ``note_count``, ``section_name``
        - **resolution** *(int)* – ticks per quarter note
        - **tempo_map** *(TempoMap)* – for tick/time conversion (not JSON-serialisable;
          use :func:`chart_to_json` for a fully serialisable version)
        - **raw_sections** *(list of str)* – names of all sections found in the file
        - **has_lyrics** *(bool)* – whether any lyric events were found
        - **duration_s** *(float)* – estimated song duration in seconds

    Raises
    ------
    FileNotFoundError
        If the chart file does not exist.
    ValueError
        If the file cannot be parsed at all.
    """
    chart_path = Path(path)
    if not chart_path.exists():
        raise FileNotFoundError(f"Chart file not found: {path}")

    # Read with BOM-aware encoding
    text = chart_path.read_text(encoding="utf-8-sig", errors="replace")

    if not text.strip():
        raise ValueError(f"Chart file is empty: {path}")

    sections = _split_sections(text)

    if not sections:
        raise ValueError(f"No sections found in chart file: {path}")

    # ── [Song] ──
    song_meta = _parse_song_section(sections.get("Song", []))
    resolution = int(song_meta.get("Resolution", DEFAULT_RESOLUTION))

    # ── [SyncTrack] ──
    sync_data = _parse_sync_track(sections.get("SyncTrack", []))
    tempo_markers = sync_data["tempo_markers"]

    # Build TempoMap
    tm_entries = [(m["tick"], m["bpm"]) for m in tempo_markers]
    tempo_map = TempoMap(tm_entries, resolution=resolution)

    # ── [Events] ──
    events = _parse_events_section(sections.get("Events", []))

    # Annotate events with time
    for evt in events:
        evt["time_s"] = round(tempo_map.tick_to_time(evt["tick"]), 4)

    # Check for lyrics
    has_lyrics = any(e["type"] in ("lyric", "phrase_start") for e in events)

    # ── Difficulty sections ──
    difficulties: dict[str, Any] = {}
    max_tick = 0

    for section_name, diff_label in DIFFICULTY_SECTION_MAP.items():
        if section_name not in sections:
            continue
        note_data = _parse_note_section(sections[section_name])

        # Annotate notes with time
        for note in note_data["notes"]:
            note["time_s"] = round(tempo_map.tick_to_time(note["tick"]), 4)
            end_tick = note["tick"] + note["duration"]
            if end_tick > max_tick:
                max_tick = end_tick

        # Annotate star power with time
        for sp in note_data["star_power"]:
            sp["time_s"] = round(tempo_map.tick_to_time(sp["tick"]), 4)
            sp["end_time_s"] = round(
                tempo_map.tick_to_time(sp["tick"] + sp["duration"]), 4
            )

        difficulties[diff_label] = {
            "section_name": section_name,
            "notes": note_data["notes"],
            "star_power": note_data["star_power"],
            "note_count": len(note_data["notes"]),
            "raw_events": note_data["raw_events"],
        }

    # Estimate duration
    duration_s = tempo_map.tick_to_time(max_tick) if max_tick > 0 else 0.0

    logger.info(
        "📊 Parsed chart: {} | resolution={} | {} tempo markers | {} events | "
        "difficulties: {} | lyrics: {} | ~{:.1f}s",
        chart_path.name,
        resolution,
        len(tempo_markers),
        len(events),
        ", ".join(sorted(difficulties.keys())) or "none",
        has_lyrics,
        duration_s,
    )

    return {
        "song": song_meta,
        "sync_track": sync_data,
        "events": events,
        "difficulties": difficulties,
        "resolution": resolution,
        "tempo_map": tempo_map,
        "raw_sections": list(sections.keys()),
        "has_lyrics": has_lyrics,
        "duration_s": round(duration_s, 3),
    }


# ---------------------------------------------------------------------------
# JSON-safe serialisation
# ---------------------------------------------------------------------------


def chart_to_json(parsed: dict[str, Any]) -> dict[str, Any]:
    """
    Convert the output of :func:`parse_chart_file` into a fully
    JSON-serialisable dict.

    The ``tempo_map`` (a :class:`TempoMap` instance) is replaced with a
    plain dict of its data.  Everything else is already serialisable.
    """
    result = dict(parsed)

    # Replace TempoMap with serialisable representation
    tm: TempoMap | None = result.pop("tempo_map", None)
    if tm is not None:
        result["tempo_map"] = {
            "resolution": tm.resolution,
            "markers": [
                {"tick": tick, "bpm": round(bpm, 3)} for tick, bpm in tm.markers
            ],
        }

    return result


def chart_to_viewer_json(
    parsed: dict[str, Any],
    difficulty: str | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
    max_notes: int = 5000,
) -> dict[str, Any]:
    """
    Produce a compact JSON payload optimised for the chart viewer/editor.

    This slices the data to only the requested difficulty and time range,
    keeping the payload small for the frontend.

    Parameters
    ----------
    parsed : dict
        Output of :func:`parse_chart_file`.
    difficulty : str, optional
        Which difficulty to include (``easy``/``medium``/``hard``/``expert``).
        If ``None``, the highest available difficulty is selected.
    start_time : float, optional
        Start of the time window in seconds.  Defaults to 0.
    end_time : float, optional
        End of the time window in seconds.  Defaults to full song.
    max_notes : int
        Maximum number of notes to return (to prevent giant payloads).

    Returns
    -------
    dict
        A compact JSON-ready dict for the frontend viewer.
    """
    song = parsed.get("song", {})
    sync = parsed.get("sync_track", {})
    events = parsed.get("events", [])
    difficulties = parsed.get("difficulties", {})
    resolution = parsed.get("resolution", DEFAULT_RESOLUTION)
    duration = parsed.get("duration_s", 0.0)

    # Select difficulty
    if difficulty and difficulty in difficulties:
        selected_diff = difficulty
    else:
        # Pick highest available
        for pref in ("expert", "hard", "medium", "easy"):
            if pref in difficulties:
                selected_diff = pref
                break
        else:
            selected_diff = None

    # Time window
    t_start = start_time if start_time is not None else 0.0
    t_end = end_time if end_time is not None else duration if duration > 0 else 999999.0

    # Filter events to time range
    _filtered_events = [e for e in events if t_start <= e.get("time_s", 0) <= t_end]

    # Sections (always include all for the section list)
    sections = [e for e in events if e["type"] == "section"]

    # Lyrics
    lyrics = [e for e in events if e["type"] in ("lyric", "phrase_start", "phrase_end")]
    filtered_lyrics = [ly for ly in lyrics if t_start <= ly.get("time_s", 0) <= t_end]

    # Notes for selected difficulty
    diff_data = None
    filtered_notes: list[dict[str, Any]] = []
    star_power: list[dict[str, Any]] = []
    note_count = 0

    if selected_diff and selected_diff in difficulties:
        diff_info = difficulties[selected_diff]
        all_notes = diff_info.get("notes", [])
        note_count = diff_info.get("note_count", len(all_notes))

        # Filter by time window
        filtered_notes = [
            n for n in all_notes if t_start <= n.get("time_s", 0) <= t_end
        ]

        # Cap note count
        if len(filtered_notes) > max_notes:
            filtered_notes = filtered_notes[:max_notes]

        star_power = diff_info.get("star_power", [])
        star_power = [
            sp for sp in star_power if t_start <= sp.get("time_s", 0) <= t_end
        ]

        diff_data = {
            "name": selected_diff,
            "section_name": diff_info.get("section_name", ""),
            "total_notes": note_count,
            "notes": filtered_notes,
            "star_power": star_power,
            "notes_in_view": len(filtered_notes),
        }

    # Tempo markers
    tempo_markers = sync.get("tempo_markers", [])
    time_sigs = sync.get("time_signatures", [])

    # Build summary of all difficulties
    diff_summary = {}
    for d_name, d_info in difficulties.items():
        diff_summary[d_name] = {
            "note_count": d_info.get("note_count", 0),
            "star_power_count": len(d_info.get("star_power", [])),
        }

    return {
        "song": {
            "name": song.get("Name", "Unknown"),
            "artist": song.get("Artist", "Unknown"),
            "album": song.get("Album", ""),
            "year": song.get("Year", ""),
            "charter": song.get("Charter", ""),
            "genre": song.get("Genre", ""),
            "resolution": resolution,
            "offset": song.get("Offset", 0),
        },
        "duration_s": duration,
        "has_lyrics": parsed.get("has_lyrics", False),
        "tempo_markers": tempo_markers,
        "time_signatures": time_sigs,
        "sections": sections,
        "lyrics": filtered_lyrics,
        "difficulty": diff_data,
        "available_difficulties": diff_summary,
        "view": {
            "start_time": t_start,
            "end_time": t_end,
            "max_notes": max_notes,
            "truncated": len(filtered_notes) >= max_notes,
        },
    }


# ---------------------------------------------------------------------------
# Chart modification helpers (for the editor)
# ---------------------------------------------------------------------------


def add_note(
    parsed: dict[str, Any],
    difficulty: str,
    tick: int,
    lane: int,
    duration: int = 0,
    is_hopo: bool = False,
) -> bool:
    """
    Add a note to the parsed chart data structure.

    Returns True on success, False if the difficulty doesn't exist or
    the parameters are invalid.
    """
    if difficulty not in parsed.get("difficulties", {}):
        return False
    if lane < 0 or lane > 7:
        return False
    if tick < 0:
        return False

    tm: TempoMap | None = parsed.get("tempo_map")
    time_s = tm.tick_to_time(tick) if tm else 0.0

    note = {
        "tick": tick,
        "lane": lane,
        "lane_name": LANE_NAMES.get(lane, f"Lane {lane}"),
        "duration": max(0, duration),
        "is_hopo": is_hopo,
        "is_tap": False,
        "time_s": round(time_s, 4),
    }

    notes = parsed["difficulties"][difficulty]["notes"]
    notes.append(note)
    # Keep sorted by tick, then lane
    notes.sort(key=lambda n: (n["tick"], n["lane"]))
    parsed["difficulties"][difficulty]["note_count"] = len(notes)

    return True


def remove_note(
    parsed: dict[str, Any],
    difficulty: str,
    tick: int,
    lane: int,
) -> bool:
    """
    Remove a note (by tick + lane) from the parsed chart data.

    Returns True if a note was removed, False otherwise.
    """
    if difficulty not in parsed.get("difficulties", {}):
        return False

    notes = parsed["difficulties"][difficulty]["notes"]
    original_len = len(notes)
    parsed["difficulties"][difficulty]["notes"] = [
        n for n in notes if not (n["tick"] == tick and n["lane"] == lane)
    ]
    new_len = len(parsed["difficulties"][difficulty]["notes"])
    parsed["difficulties"][difficulty]["note_count"] = new_len

    return new_len < original_len


def move_note(
    parsed: dict[str, Any],
    difficulty: str,
    old_tick: int,
    old_lane: int,
    new_tick: int,
    new_lane: int,
) -> bool:
    """
    Move a note to a new tick and/or lane.

    Returns True if the note was found and moved, False otherwise.
    """
    if difficulty not in parsed.get("difficulties", {}):
        return False

    notes = parsed["difficulties"][difficulty]["notes"]
    tm: TempoMap | None = parsed.get("tempo_map")

    for note in notes:
        if note["tick"] == old_tick and note["lane"] == old_lane:
            note["tick"] = new_tick
            note["lane"] = new_lane
            note["lane_name"] = LANE_NAMES.get(new_lane, f"Lane {new_lane}")
            note["time_s"] = round(tm.tick_to_time(new_tick), 4) if tm else 0.0
            # Re-sort
            notes.sort(key=lambda n: (n["tick"], n["lane"]))
            return True

    return False


# ---------------------------------------------------------------------------
# Write chart back to file
# ---------------------------------------------------------------------------


def write_chart_file(parsed: dict[str, Any], output_path: str | Path) -> bool:
    """
    Write a parsed chart structure back to a ``.chart`` file.

    This enables round-trip editing: parse → modify → write.

    Parameters
    ----------
    parsed : dict
        A chart structure as returned by :func:`parse_chart_file`, possibly
        modified via the add/remove/move helpers.
    output_path : str or Path
        Where to write the ``.chart`` file.

    Returns
    -------
    bool
        True on success, False on failure.
    """
    try:
        lines: list[str] = []
        song = parsed.get("song", {})
        sync = parsed.get("sync_track", {})
        events = parsed.get("events", [])
        difficulties = parsed.get("difficulties", {})
        _resolution = parsed.get("resolution", DEFAULT_RESOLUTION)

        # ── [Song] ──
        lines.append("[Song]")
        lines.append("{")
        # Write known fields in a sensible order
        field_order = [
            "Name",
            "Artist",
            "Album",
            "Year",
            "Charter",
            "Offset",
            "Resolution",
            "Player2",
            "Difficulty",
            "PreviewStart",
            "PreviewEnd",
            "Genre",
            "MediaType",
            "MusicStream",
        ]
        written_keys: set[str] = set()
        for key in field_order:
            if key in song:
                val = song[key]
                if isinstance(val, str):
                    lines.append(f'  {key} = "{val}"')
                else:
                    lines.append(f"  {key} = {val}")
                written_keys.add(key)
        # Write any remaining keys not in the order list
        for key, val in song.items():
            if key not in written_keys:
                if isinstance(val, str):
                    lines.append(f'  {key} = "{val}"')
                else:
                    lines.append(f"  {key} = {val}")
        lines.append("}")
        lines.append("")

        # ── [SyncTrack] ──
        lines.append("[SyncTrack]")
        lines.append("{")
        # Time signatures first, then tempo markers, merged by tick
        ts_list = sync.get("time_signatures", [])
        tempo_list = sync.get("tempo_markers", [])

        # Build a combined event list sorted by tick
        sync_events: list[tuple[int, str]] = []
        for ts in ts_list:
            tick = ts["tick"]
            num = ts["numerator"]
            denom_pow = ts.get("denominator_power", 2)
            if denom_pow == 2:
                sync_events.append((tick, f"  {tick} = TS {num}"))
            else:
                sync_events.append((tick, f"  {tick} = TS {num} {denom_pow}"))
        for tm in tempo_list:
            tick = tm["tick"]
            mbpm = tm["milli_bpm"]
            sync_events.append((tick, f"  {tick} = B {mbpm}"))

        sync_events.sort(key=lambda x: x[0])
        for _, line in sync_events:
            lines.append(line)

        lines.append("}")
        lines.append("")

        # ── [Events] ──
        lines.append("[Events]")
        lines.append("{")
        for evt in sorted(events, key=lambda e: e["tick"]):
            evt_type = evt["type"]
            value = evt["value"]
            tick = evt["tick"]
            if evt_type == "section":
                lines.append(f'  {tick} = E "section {value}"')
            elif evt_type == "phrase_start":
                lines.append(f'  {tick} = E "phrase_start"')
            elif evt_type == "phrase_end":
                lines.append(f'  {tick} = E "phrase_end"')
            elif evt_type == "lyric":
                lines.append(f'  {tick} = E "lyric {value}"')
            elif evt_type == "other":
                lines.append(f'  {tick} = E "{value}"')
        lines.append("}")
        lines.append("")

        # ── Note sections ──
        # Write in order: Easy, Medium, Hard, Expert
        for diff_label in ("easy", "medium", "hard", "expert"):
            if diff_label not in difficulties:
                continue
            diff = difficulties[diff_label]
            section_name = diff.get(
                "section_name",
                DIFFICULTY_TO_SECTION.get(diff_label, f"{diff_label.title()}Single"),
            )

            lines.append(f"[{section_name}]")
            lines.append("{")

            # Collect all events for this section grouped by tick
            tick_lines: dict[int, list[str]] = {}

            for note in diff.get("notes", []):
                tick = note["tick"]
                lane = note["lane"]
                dur = note["duration"]
                if tick not in tick_lines:
                    tick_lines[tick] = []
                tick_lines[tick].append(f"  {tick} = N {lane} {dur}")

                # Add HOPO flag
                if note.get("is_hopo"):
                    hopo_line = f"  {tick} = N 5 0"
                    if hopo_line not in tick_lines[tick]:
                        tick_lines[tick].append(hopo_line)

                # Add tap flag
                if note.get("is_tap"):
                    tap_line = f"  {tick} = N 6 0"
                    if tap_line not in tick_lines[tick]:
                        tick_lines[tick].append(tap_line)

            for sp in diff.get("star_power", []):
                tick = sp["tick"]
                dur = sp["duration"]
                if tick not in tick_lines:
                    tick_lines[tick] = []
                tick_lines[tick].append(f"  {tick} = S 2 {dur}")

            for tick in sorted(tick_lines.keys()):
                for line in tick_lines[tick]:
                    lines.append(line)

            lines.append("}")
            lines.append("")

        # Write to file with UTF-8 BOM
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8-sig") as f:
            f.write("\n".join(lines))

        logger.info("💾 Wrote chart file: {} ({} lines)", out, len(lines))
        return True

    except Exception as e:
        logger.error("❌ Failed to write chart file {}: {}", output_path, e)
        return False


# ---------------------------------------------------------------------------
# Convenience / summary helpers
# ---------------------------------------------------------------------------


def get_chart_summary(parsed: dict[str, Any]) -> dict[str, Any]:
    """
    Return a lightweight summary of a parsed chart (no note data).

    Useful for listing charts in the UI without loading full note arrays.
    """
    song = parsed.get("song", {})
    difficulties = parsed.get("difficulties", {})
    events = parsed.get("events", [])

    section_names = [e["value"] for e in events if e["type"] == "section"]

    diff_summary = {}
    for name, info in difficulties.items():
        diff_summary[name] = {
            "note_count": info.get("note_count", 0),
            "star_power_sections": len(info.get("star_power", [])),
        }

    tempo_markers = parsed.get("sync_track", {}).get("tempo_markers", [])
    bpms = [m["bpm"] for m in tempo_markers] if tempo_markers else [120.0]

    return {
        "name": song.get("Name", "Unknown"),
        "artist": song.get("Artist", "Unknown"),
        "album": song.get("Album", ""),
        "charter": song.get("Charter", ""),
        "genre": song.get("Genre", ""),
        "resolution": parsed.get("resolution", DEFAULT_RESOLUTION),
        "duration_s": parsed.get("duration_s", 0.0),
        "has_lyrics": parsed.get("has_lyrics", False),
        "sections": section_names,
        "difficulties": diff_summary,
        "bpm_range": {
            "min": round(min(bpms), 1),
            "max": round(max(bpms), 1),
            "primary": round(bpms[0], 1),
        },
        "available_difficulties": list(diff_summary.keys()),
    }


def parse_chart_from_string(text: str) -> dict[str, Any]:
    """
    Parse chart data from a string (rather than a file).

    Behaves identically to :func:`parse_chart_file` but accepts raw text.
    Useful for testing or when the chart content is already in memory.
    """
    if not text.strip():
        raise ValueError("Chart text is empty")

    # Remove BOM if present
    if text.startswith("\ufeff"):
        text = text[1:]

    sections = _split_sections(text)
    if not sections:
        raise ValueError("No sections found in chart text")

    # Reuse the same logic as parse_chart_file
    song_meta = _parse_song_section(sections.get("Song", []))
    resolution = int(song_meta.get("Resolution", DEFAULT_RESOLUTION))

    sync_data = _parse_sync_track(sections.get("SyncTrack", []))
    tempo_markers = sync_data["tempo_markers"]

    tm_entries = [(m["tick"], m["bpm"]) for m in tempo_markers]
    tempo_map = TempoMap(tm_entries, resolution=resolution)

    events = _parse_events_section(sections.get("Events", []))
    for evt in events:
        evt["time_s"] = round(tempo_map.tick_to_time(evt["tick"]), 4)

    has_lyrics = any(e["type"] in ("lyric", "phrase_start") for e in events)

    difficulties: dict[str, Any] = {}
    max_tick = 0

    for section_name, diff_label in DIFFICULTY_SECTION_MAP.items():
        if section_name not in sections:
            continue
        note_data = _parse_note_section(sections[section_name])
        for note in note_data["notes"]:
            note["time_s"] = round(tempo_map.tick_to_time(note["tick"]), 4)
            end_tick = note["tick"] + note["duration"]
            if end_tick > max_tick:
                max_tick = end_tick
        for sp in note_data["star_power"]:
            sp["time_s"] = round(tempo_map.tick_to_time(sp["tick"]), 4)
            sp["end_time_s"] = round(
                tempo_map.tick_to_time(sp["tick"] + sp["duration"]), 4
            )
        difficulties[diff_label] = {
            "section_name": section_name,
            "notes": note_data["notes"],
            "star_power": note_data["star_power"],
            "note_count": len(note_data["notes"]),
            "raw_events": note_data["raw_events"],
        }

    duration_s = tempo_map.tick_to_time(max_tick) if max_tick > 0 else 0.0

    return {
        "song": song_meta,
        "sync_track": sync_data,
        "events": events,
        "difficulties": difficulties,
        "resolution": resolution,
        "tempo_map": tempo_map,
        "raw_sections": list(sections.keys()),
        "has_lyrics": has_lyrics,
        "duration_s": round(duration_s, 3),
    }
