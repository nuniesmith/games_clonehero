"""
Clone Hero Content Manager - MIDI Parser Service

Parses MIDI files (.mid / .midi) to extract musical information that can be
used to generate higher-quality Clone Hero charts compared to audio-only
analysis.

MIDI files contain precise note-on/note-off events with pitch and velocity,
tempo changes, time signatures, track names, and text/marker events — all of
which map directly to Clone Hero chart concepts:

    MIDI concept              → Clone Hero concept
    ─────────────────────────────────────────────────
    Note-on / note-off        → Note placement + sustain
    Note pitch (0-127)        → Lane assignment (0-4 + open)
    Note velocity (0-127)     → Difficulty filtering / HOPO / dynamics
    Tempo changes             → SyncTrack BPM markers
    Time signature            → SyncTrack TS markers
    Marker / text events      → Section labels (Intro, Verse, Chorus…)
    Track names               → Instrument identification

Supported MIDI sources:
    - Guitar Pro exports (.mid from GP5/GP6/GP7)
    - DAW exports (Reaper, FL Studio, Ableton, etc.)
    - Rock Band / Guitar Hero MIDI charts (pre-charted)
    - General MIDI files with instrument tracks

The parser produces a ``MidiSongData`` dataclass that the song generator
can consume directly, replacing or supplementing audio-derived onset data
with precise MIDI note information.

Dependencies: ``mido`` (pure-Python MIDI parser, no C dependencies).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

try:
    import importlib.util

    MIDO_AVAILABLE = importlib.util.find_spec("mido") is not None
except Exception:
    MIDO_AVAILABLE = False

if not MIDO_AVAILABLE:
    logger.warning("⚠️ mido not installed — MIDI import disabled. pip install mido")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Clone Hero lane constants (same as song_generator.py)
LANE_GREEN = 0
LANE_RED = 1
LANE_YELLOW = 2
LANE_BLUE = 3
LANE_ORANGE = 4
LANE_OPEN = 7

# Standard Clone Hero chart resolution
CHART_RESOLUTION = 192  # ticks per quarter note

# Default MIDI ticks per beat (common values: 480, 960, 120, 96)
DEFAULT_MIDI_TPB = 480

# ---------------------------------------------------------------------------
# MIDI note ranges for instrument detection
#
# General MIDI instruments typically use these note ranges:
#   Bass guitar  : E1 (28) – G3 (55)
#   Guitar       : E2 (40) – E6 (88)
#   Vocals       : C2 (36) – C6 (84)
#   Drums        : Channel 10, notes 35-81 (GM drum map)
#
# Rock Band / Guitar Hero MIDI charts use specific note numbers:
#   Expert guitar: 96-100 (Green=96, Red=97, Yellow=98, Blue=99, Orange=100)
#   Hard guitar  : 84-88
#   Medium guitar: 72-76
#   Easy guitar  : 60-64
#   Expert bass  : 96-100 (on bass track)
#   Expert drums : 96-100 (on drums track)
# ---------------------------------------------------------------------------

# Rock Band MIDI note numbers for guitar/bass (per difficulty)
RB_EXPERT_NOTES = {96: 0, 97: 1, 98: 2, 99: 3, 100: 4}  # G R Y B O
RB_HARD_NOTES = {84: 0, 85: 1, 86: 2, 87: 3, 88: 4}
RB_MEDIUM_NOTES = {72: 0, 73: 1, 74: 2, 75: 3, 76: 4}
RB_EASY_NOTES = {60: 0, 61: 1, 62: 2, 63: 3, 64: 4}

RB_DIFFICULTY_MAPS = {
    "expert": RB_EXPERT_NOTES,
    "hard": RB_HARD_NOTES,
    "medium": RB_MEDIUM_NOTES,
    "easy": RB_EASY_NOTES,
}

# Rock Band open note markers (special MIDI notes)
RB_OPEN_NOTE_MARKERS = {
    "expert": 95,  # Note just below expert green
    "hard": 83,
    "medium": 71,
    "easy": 59,
}

# Rock Band MIDI note numbers for drums
RB_DRUM_NOTES = {
    96: 0,  # Expert kick
    97: 1,  # Expert red (snare)
    98: 2,  # Expert yellow (hi-hat)
    99: 3,  # Expert blue (tom)
    100: 4,  # Expert green (floor tom / cymbal)
}

# General MIDI drum map (channel 10) → Clone Hero drum lanes
GM_DRUM_MAP: dict[int, int] = {
    35: 0,
    36: 0,  # Acoustic/Electric Bass Drum → kick
    38: 1,
    40: 1,
    37: 1,  # Snare / Side Stick → red (snare)
    42: 2,
    44: 2,
    46: 2,  # Hi-hat (closed/pedal/open) → yellow
    41: 3,
    43: 3,
    45: 3,
    47: 3,  # Low/Mid Toms → blue (tom)
    48: 4,
    50: 4,  # High Tom → green
    49: 4,
    51: 2,
    52: 4,  # Crash/Ride → yellow/green
    53: 2,
    55: 4,
    57: 4,
    59: 4,  # Various cymbals
}

# Track name patterns for instrument identification (case-insensitive)
GUITAR_TRACK_PATTERNS = [
    "guitar",
    "gtr",
    "lead",
    "rhythm",
    "distortion",
    "clean",
    "overdrive",
    "electric",
    "acoustic guitar",
    "part guitar",
    "t1 gems",
]
BASS_TRACK_PATTERNS = [
    "bass",
    "bass guitar",
    "part bass",
]
DRUM_TRACK_PATTERNS = [
    "drum",
    "drums",
    "percussion",
    "perc",
    "part drums",
]
VOCAL_TRACK_PATTERNS = [
    "vocal",
    "vocals",
    "voice",
    "sing",
    "melody",
    "part vocals",
    "harm1",
    "harm2",
    "harm3",
]
SECTION_TRACK_PATTERNS = [
    "section",
    "marker",
    "events",
    "part events",
    "beat",
    "venue",
]

# Section label mapping for common MIDI marker text
SECTION_LABEL_MAP: dict[str, str] = {
    "intro": "Intro",
    "verse": "Verse",
    "verse_1": "Verse 1",
    "verse_2": "Verse 2",
    "verse_3": "Verse 3",
    "prechorus": "Pre-Chorus",
    "pre_chorus": "Pre-Chorus",
    "pre-chorus": "Pre-Chorus",
    "chorus": "Chorus",
    "chorus_1": "Chorus 1",
    "chorus_2": "Chorus 2",
    "bridge": "Bridge",
    "solo": "Solo",
    "guitar_solo": "Guitar Solo",
    "outro": "Outro",
    "breakdown": "Breakdown",
    "interlude": "Interlude",
    "riff": "Riff",
    "coda": "Coda",
    "ending": "Outro",
    "silence": "Silence",
    "big_rock_ending": "Big Rock Ending",
    "bre": "Big Rock Ending",
    "drum_solo": "Drum Solo",
    "bass_solo": "Bass Solo",
    "no_crowd": "Interlude",
    "crowd_intense": "Chorus",
    "crowd_normal": "Verse",
    "crowd_mellow": "Verse",
    "music_start": "Intro",
    "music_end": "Outro",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MidiNote:
    """A single MIDI note event with timing and musical information."""

    # Timing (in MIDI ticks from the start of the file)
    tick: int
    # Duration in MIDI ticks (note-off minus note-on)
    duration_ticks: int
    # Timing in seconds (computed from tempo map)
    time_seconds: float
    # Duration in seconds
    duration_seconds: float

    # Musical properties
    pitch: int  # MIDI note number (0-127)
    velocity: int  # Note velocity (0-127)
    channel: int  # MIDI channel (0-15, drums typically on 9)

    # Clone Hero mapping (computed during parsing)
    lane: int = 0  # Mapped CH lane (0-4, 7=open)
    is_chord: bool = False  # Part of a chord (simultaneous notes)

    def __repr__(self) -> str:
        note_name = _midi_note_to_name(self.pitch)
        return (
            f"MidiNote(tick={self.tick}, time={self.time_seconds:.3f}s, "
            f"pitch={self.pitch}({note_name}), vel={self.velocity}, "
            f"lane={self.lane}, dur={self.duration_ticks}ticks)"
        )


@dataclass
class MidiTempoEvent:
    """A tempo change event from the MIDI file."""

    tick: int
    tempo_bpm: float
    # Microseconds per beat (raw MIDI tempo value)
    microseconds_per_beat: int
    # Time in seconds when this tempo takes effect
    time_seconds: float = 0.0


@dataclass
class MidiTimeSignature:
    """A time signature event from the MIDI file."""

    tick: int
    numerator: int
    denominator: int
    time_seconds: float = 0.0


@dataclass
class MidiSection:
    """A structural section marker from the MIDI file."""

    tick: int
    time_seconds: float
    label: str
    # Original raw text from the MIDI marker/text event
    raw_text: str = ""


@dataclass
class MidiTrackInfo:
    """Metadata about a single MIDI track."""

    index: int
    name: str
    instrument: str  # detected instrument: guitar/bass/drums/vocals/unknown
    note_count: int
    channel: int | None  # primary channel used (-1 if mixed)
    pitch_range: tuple[int, int] = (0, 127)  # (min, max) pitch
    is_drum: bool = False


@dataclass
class MidiSongData:
    """
    Complete parsed MIDI song data, ready for chart generation.

    This is the primary output of the MIDI parser. The song generator
    consumes this to produce notes.chart files with MIDI-accurate note
    placement instead of (or supplemented by) audio-derived onsets.
    """

    # Source file info
    source_file: str = ""
    midi_format: int = 1  # MIDI format (0, 1, or 2)
    ticks_per_beat: int = 480  # MIDI file resolution

    # Tempo information
    tempo_events: list[MidiTempoEvent] = field(default_factory=list)
    time_signatures: list[MidiTimeSignature] = field(default_factory=list)
    initial_tempo: float = 120.0  # BPM

    # Section markers
    sections: list[MidiSection] = field(default_factory=list)

    # Notes per instrument track (instrument_name → list of notes)
    tracks: dict[str, list[MidiNote]] = field(default_factory=dict)

    # Track metadata
    track_info: list[MidiTrackInfo] = field(default_factory=list)

    # Duration in seconds (computed from last note-off)
    duration: float = 0.0

    # Whether this MIDI uses Rock Band / Guitar Hero note mapping
    is_rb_format: bool = False

    # Beat times derived from the MIDI tempo map (seconds)
    beat_times: list[float] = field(default_factory=list)

    # Available instruments detected in the MIDI
    available_instruments: list[str] = field(default_factory=list)

    def get_notes_for_instrument(self, instrument: str) -> list[MidiNote]:
        """Get notes for a specific instrument, with fallbacks.

        If the exact instrument track isn't found, tries common
        alternatives (e.g., 'lead' for guitar, or the track with
        the most notes).
        """
        # Direct match
        if instrument in self.tracks:
            return self.tracks[instrument]

        # Try aliases
        aliases: dict[str, list[str]] = {
            "guitar": [
                "lead",
                "rhythm",
                "electric",
                "acoustic guitar",
                "clean",
                "distortion",
                "overdrive",
            ],
            "bass": ["bass guitar"],
            "drums": ["percussion", "perc"],
            "vocals": ["voice", "melody", "sing"],
        }
        for alias in aliases.get(instrument, []):
            if alias in self.tracks:
                return self.tracks[alias]

        # Fallback: return the track with the most notes (excluding drums
        # if we're looking for a melodic instrument)
        if self.tracks:
            candidates = self.tracks
            if instrument in ("guitar", "bass", "vocals"):
                candidates = {
                    k: v
                    for k, v in self.tracks.items()
                    if k != "drums"
                    and not any(ti.is_drum for ti in self.track_info if ti.name == k)
                }
            if candidates:
                best = max(candidates, key=lambda k: len(candidates[k]))
                return candidates[best]

        return []

    def get_onset_times(self, instrument: str) -> list[float]:
        """Get onset times (in seconds) for an instrument."""
        notes = self.get_notes_for_instrument(instrument)
        return [n.time_seconds for n in notes]

    def get_onset_strengths(self, instrument: str) -> list[float]:
        """Get normalised onset strengths (from MIDI velocity)."""
        notes = self.get_notes_for_instrument(instrument)
        if not notes:
            return []
        max_vel = max(n.velocity for n in notes) or 1
        return [n.velocity / max_vel for n in notes]

    def get_lanes(self, instrument: str) -> list[int]:
        """Get pre-computed lane assignments for an instrument."""
        notes = self.get_notes_for_instrument(instrument)
        return [n.lane for n in notes]

    def get_sustain_ticks(
        self, instrument: str, chart_resolution: int = CHART_RESOLUTION
    ) -> list[int]:
        """Get sustain durations converted to chart ticks."""
        notes = self.get_notes_for_instrument(instrument)
        if not notes:
            return []

        ratio = chart_resolution / self.ticks_per_beat
        result: list[int] = []
        for n in notes:
            chart_dur = int(round(n.duration_ticks * ratio))
            # Only emit sustain if it's at least half a beat
            if chart_dur >= chart_resolution // 2:
                result.append(chart_dur)
            else:
                result.append(0)
        return result

    def to_segments(self) -> list[dict[str, Any]]:
        """Convert MIDI sections to the segment format used by the generator."""
        if not self.sections:
            return []
        return [
            {
                "time": round(s.time_seconds, 3),
                "label": s.label,
                "energy": 0.5,  # MIDI doesn't have energy; audio can fill this
            }
            for s in self.sections
        ]

    def to_tempo_map(
        self, chart_resolution: int = CHART_RESOLUTION
    ) -> list[tuple[int, int]]:
        """Convert MIDI tempo events to chart tempo map format.

        Returns list of (chart_tick, milli_bpm) tuples compatible with
        the song generator's SyncTrack output.
        """
        if not self.tempo_events:
            return [(0, int(round(self.initial_tempo * 1000)))]

        ratio = chart_resolution / self.ticks_per_beat
        result: list[tuple[int, int]] = []
        for te in self.tempo_events:
            chart_tick = int(round(te.tick * ratio))
            milli_bpm = int(round(te.tempo_bpm * 1000))
            result.append((chart_tick, milli_bpm))

        return result

    def to_time_signature_events(
        self, chart_resolution: int = CHART_RESOLUTION
    ) -> list[tuple[int, int, int]]:
        """Convert MIDI time signatures to chart format.

        Returns list of (chart_tick, numerator, denominator) tuples.
        """
        if not self.time_signatures:
            return [(0, 4, 4)]

        ratio = chart_resolution / self.ticks_per_beat
        result: list[tuple[int, int, int]] = []
        for ts in self.time_signatures:
            chart_tick = int(round(ts.tick * ratio))
            result.append((chart_tick, ts.numerator, ts.denominator))

        return result


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _midi_note_to_name(note: int) -> str:
    """Convert MIDI note number to note name (e.g. 60 → 'C4')."""
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = (note // 12) - 1
    name = names[note % 12]
    return f"{name}{octave}"


def _identify_instrument_from_name(track_name: str) -> str:
    """Guess the instrument from a MIDI track name."""
    name_lower = track_name.lower().strip()

    # Check each instrument pattern set (order matters: bass before guitar
    # since "bass guitar" contains "guitar")
    for pattern in DRUM_TRACK_PATTERNS:
        if pattern in name_lower:
            return "drums"
    for pattern in BASS_TRACK_PATTERNS:
        if pattern in name_lower:
            return "bass"
    for pattern in GUITAR_TRACK_PATTERNS:
        if pattern in name_lower:
            return "guitar"
    for pattern in VOCAL_TRACK_PATTERNS:
        if pattern in name_lower:
            return "vocals"
    for pattern in SECTION_TRACK_PATTERNS:
        if pattern in name_lower:
            return "sections"

    return "unknown"


def _identify_instrument_from_notes(
    notes: list[MidiNote],
    channel: int | None,
) -> str:
    """Guess the instrument from note characteristics."""
    if not notes:
        return "unknown"

    # Channel 9 (0-indexed) / 10 (1-indexed) is the GM drum channel
    if channel == 9:
        return "drums"

    pitches = [n.pitch for n in notes]
    min_pitch = min(pitches)
    max_pitch = max(pitches)
    avg_pitch = sum(pitches) / len(pitches)

    # Check for Rock Band note mapping
    rb_notes = (
        set(RB_EXPERT_NOTES.keys())
        | set(RB_HARD_NOTES.keys())
        | set(RB_MEDIUM_NOTES.keys())
        | set(RB_EASY_NOTES.keys())
    )
    pitch_set = set(pitches)
    rb_overlap = len(pitch_set & rb_notes) / max(len(pitch_set), 1)
    if rb_overlap > 0.8:
        return "guitar"  # Likely a Rock Band guitar/bass track

    # Heuristic based on pitch range
    if max_pitch <= 55 and avg_pitch < 48:
        return "bass"
    if min_pitch >= 36 and max_pitch <= 84 and avg_pitch > 55:
        return "vocals"
    if 40 <= min_pitch <= 88:
        return "guitar"

    return "unknown"


def _is_rb_format(notes: list[MidiNote]) -> bool:
    """Detect whether notes use Rock Band MIDI note mapping."""
    if not notes:
        return False

    rb_all = set()
    for note_map in RB_DIFFICULTY_MAPS.values():
        rb_all.update(note_map.keys())
    # Also include open note markers
    rb_all.update(RB_OPEN_NOTE_MARKERS.values())

    pitch_set = set(n.pitch for n in notes)
    overlap = len(pitch_set & rb_all) / max(len(pitch_set), 1)
    return overlap > 0.7


def _normalise_section_label(raw_text: str) -> str:
    """Normalise a MIDI marker/text into a clean section label.

    Handles various formats:
        - "[section Intro]" (Rock Band format)
        - "Intro"
        - "section_intro"
        - "[prc_intro]"
    """
    text = raw_text.strip()

    # Strip surrounding brackets
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()

    # Remove common prefixes
    for prefix in ("section ", "section_", "prc_", "sect_", "event_"):
        if text.lower().startswith(prefix):
            text = text[len(prefix) :]
            break

    # Look up in our label map
    key = text.lower().strip().replace(" ", "_").replace("-", "_")
    if key in SECTION_LABEL_MAP:
        return SECTION_LABEL_MAP[key]

    # If it's already a reasonable label, title-case it
    if text and not text[0].isdigit():
        # Replace underscores with spaces and title-case
        return text.replace("_", " ").strip().title()

    return text or "Section"


def _build_tempo_map_seconds(
    tempo_events: list[MidiTempoEvent],
    ticks_per_beat: int,
) -> list[MidiTempoEvent]:
    """Compute absolute time in seconds for each tempo event.

    MIDI files store tempo as microseconds-per-beat at specific tick
    positions. To convert any tick to seconds we need to walk through
    the tempo changes cumulatively.
    """
    if not tempo_events:
        return []

    # Sort by tick (should already be sorted, but be safe)
    events = sorted(tempo_events, key=lambda e: e.tick)

    current_time = 0.0
    prev_tick = 0
    prev_uspb = events[0].microseconds_per_beat  # microseconds per beat

    for evt in events:
        # Compute elapsed time from previous tempo event to this one
        delta_ticks = evt.tick - prev_tick
        if delta_ticks > 0 and ticks_per_beat > 0:
            # Seconds = (delta_ticks / ticks_per_beat) * (uspb / 1_000_000)
            current_time += (delta_ticks / ticks_per_beat) * (prev_uspb / 1_000_000)

        evt.time_seconds = current_time
        prev_tick = evt.tick
        prev_uspb = evt.microseconds_per_beat

    return events


def _tick_to_seconds(
    tick: int,
    tempo_events: list[MidiTempoEvent],
    ticks_per_beat: int,
) -> float:
    """Convert a MIDI tick position to seconds using the tempo map."""
    if not tempo_events:
        # Default 120 BPM
        return tick / ticks_per_beat * 0.5

    # Find the tempo event that's active at this tick
    active_event = tempo_events[0]
    for evt in tempo_events:
        if evt.tick <= tick:
            active_event = evt
        else:
            break

    # Compute time from the active event's position
    delta_ticks = tick - active_event.tick
    if ticks_per_beat > 0:
        delta_seconds = (delta_ticks / ticks_per_beat) * (
            active_event.microseconds_per_beat / 1_000_000
        )
    else:
        delta_seconds = 0.0

    return active_event.time_seconds + delta_seconds


def _compute_beat_times(
    tempo_events: list[MidiTempoEvent],
    ticks_per_beat: int,
    duration_seconds: float,
) -> list[float]:
    """Generate beat time positions from the MIDI tempo map.

    Walks the tempo map and emits a beat timestamp every
    ``ticks_per_beat`` ticks, converting to seconds.
    """
    if not tempo_events or duration_seconds <= 0:
        return []

    beats: list[float] = []
    current_time = 0.0
    current_tick = 0
    tempo_idx = 0
    uspb = tempo_events[0].microseconds_per_beat if tempo_events else 500_000

    # Walk tick by tick in beat-sized increments
    max_ticks = int(duration_seconds * ticks_per_beat * 4)  # generous upper bound
    safety = 0

    while current_time < duration_seconds and safety < max_ticks:
        beats.append(round(current_time, 6))

        # Advance one beat
        next_tick = current_tick + ticks_per_beat

        # Check for tempo changes within this beat
        while (
            tempo_idx + 1 < len(tempo_events)
            and tempo_events[tempo_idx + 1].tick <= next_tick
        ):
            # Partial beat at old tempo, then switch
            mid_tick = tempo_events[tempo_idx + 1].tick
            partial_ticks = mid_tick - current_tick
            if ticks_per_beat > 0:
                current_time += (partial_ticks / ticks_per_beat) * (uspb / 1_000_000)
            current_tick = mid_tick
            tempo_idx += 1
            uspb = tempo_events[tempo_idx].microseconds_per_beat

        # Remaining ticks in this beat at current tempo
        remaining_ticks = next_tick - current_tick
        if remaining_ticks > 0 and ticks_per_beat > 0:
            current_time += (remaining_ticks / ticks_per_beat) * (uspb / 1_000_000)

        current_tick = next_tick
        safety += 1

    return beats


# ---------------------------------------------------------------------------
# Lane mapping strategies
# ---------------------------------------------------------------------------


def map_rb_notes_to_lanes(
    notes: list[MidiNote],
    difficulty: str = "expert",
) -> list[MidiNote]:
    """Map Rock Band MIDI note numbers to Clone Hero lanes.

    Rock Band uses specific MIDI note numbers for each difficulty:
        Expert: 96-100 → Green(0) to Orange(4)
        Hard:   84-88
        Medium: 72-76
        Easy:   60-64

    Notes that don't match any difficulty mapping are dropped.
    """
    note_map = RB_DIFFICULTY_MAPS.get(difficulty, RB_EXPERT_NOTES)
    open_marker = RB_OPEN_NOTE_MARKERS.get(difficulty)

    # Check for open note marker notes (they're separate MIDI notes that
    # overlap with regular notes to indicate open strums)
    open_ticks: set[int] = set()
    if open_marker is not None:
        for n in notes:
            if n.pitch == open_marker:
                open_ticks.add(n.tick)

    result: list[MidiNote] = []
    for n in notes:
        if n.pitch in note_map:
            n.lane = note_map[n.pitch]
            # Check if this tick has an open note marker
            if n.tick in open_ticks and n.lane == 0:
                n.lane = LANE_OPEN
            result.append(n)

    return result


def map_pitch_range_to_lanes(
    notes: list[MidiNote],
    max_lane: int = 4,
    open_threshold_semitones: int = 3,
) -> list[MidiNote]:
    """Map arbitrary MIDI pitches to Clone Hero lanes based on pitch range.

    Divides the pitch range of the notes into equal-sized bins, one per
    lane. The lowest pitches can optionally be mapped to open notes
    (lane 7) if they fall within ``open_threshold_semitones`` of the
    minimum pitch.

    This is used for non-Rock-Band MIDI files (DAW exports, Guitar Pro, etc.)
    where the note numbers represent actual musical pitches.
    """
    if not notes:
        return notes

    pitches = [n.pitch for n in notes]
    min_pitch = min(pitches)
    max_pitch = max(pitches)
    pitch_range = max_pitch - min_pitch

    if pitch_range == 0:
        # All notes on the same pitch → put them on green
        for n in notes:
            n.lane = LANE_GREEN
        return notes

    num_lanes = max_lane + 1  # 0 to max_lane inclusive

    for n in notes:
        # Check for open note (very lowest pitches)
        if n.pitch - min_pitch <= open_threshold_semitones:
            n.lane = LANE_OPEN
        else:
            # Map pitch to lane (0 = low, max_lane = high)
            relative = (n.pitch - min_pitch) / pitch_range
            lane = int(relative * (num_lanes - 1))
            n.lane = min(lane, max_lane)

    return notes


def map_drum_notes_to_lanes(
    notes: list[MidiNote],
    is_gm: bool = True,
) -> list[MidiNote]:
    """Map MIDI drum notes to Clone Hero drum lanes.

    For General MIDI drums (channel 10), uses the standard GM drum map.
    For Rock Band drums, uses the RB note number mapping.
    """
    for n in notes:
        if is_gm and n.pitch in GM_DRUM_MAP:
            n.lane = GM_DRUM_MAP[n.pitch]
        elif n.pitch in RB_DRUM_NOTES:
            n.lane = RB_DRUM_NOTES[n.pitch]
        else:
            # Fallback: map by pitch range
            if n.pitch < 40:
                n.lane = 0  # kick
            elif n.pitch < 50:
                n.lane = 1  # snare
            elif n.pitch < 55:
                n.lane = 2  # hi-hat
            elif n.pitch < 60:
                n.lane = 3  # tom
            else:
                n.lane = 4  # cymbal
    return notes


def _mark_chords(notes: list[MidiNote], tolerance_ticks: int = 5) -> list[MidiNote]:
    """Mark notes that are part of a chord (simultaneous notes).

    Notes within ``tolerance_ticks`` of each other are considered
    simultaneous (same strum).
    """
    if not notes:
        return notes

    # Group notes by approximate tick position
    groups: dict[int, list[int]] = defaultdict(list)
    for i, n in enumerate(notes):
        # Quantise tick to group simultaneous notes
        quantised = n.tick // max(tolerance_ticks, 1) * max(tolerance_ticks, 1)
        groups[quantised].append(i)

    for indices in groups.values():
        if len(indices) > 1:
            for idx in indices:
                notes[idx].is_chord = True

    return notes


# ---------------------------------------------------------------------------
# Difficulty filtering
# ---------------------------------------------------------------------------


def filter_notes_for_difficulty(
    notes: list[MidiNote],
    difficulty: str,
    max_lane: int | None = None,
) -> list[MidiNote]:
    """Filter and simplify notes for a given difficulty level.

    Applies difficulty-appropriate thinning:
        - Easy: every 4th note, max 3 lanes, no chords
        - Medium: every 3rd note, max 4 lanes, rare chords
        - Hard: every 2nd note, all lanes, some chords
        - Expert: all notes

    Also enforces minimum note gaps appropriate to the difficulty.
    """
    if not notes:
        return []

    difficulty_config = {
        "easy": {"skip": 4, "max_lane": max_lane or 2, "drop_chords": True},
        "medium": {"skip": 3, "max_lane": max_lane or 3, "drop_chords": False},
        "hard": {"skip": 2, "max_lane": max_lane or 4, "drop_chords": False},
        "expert": {"skip": 1, "max_lane": max_lane or 4, "drop_chords": False},
    }

    config = difficulty_config.get(difficulty, difficulty_config["expert"])
    skip = config["skip"]
    lane_cap = config["max_lane"]
    drop_chords = config["drop_chords"]

    result: list[MidiNote] = []

    # For non-expert, we thin by selecting every Nth non-chord note
    # (or every note for expert)
    counter = 0
    seen_ticks: set[int] = set()

    for note in notes:
        # Cap the lane
        if note.lane != LANE_OPEN:
            note.lane = min(note.lane, lane_cap)
        elif difficulty not in ("hard", "expert"):
            note.lane = LANE_GREEN  # downgrade open to green on easy/medium

        # Handle chords
        if note.is_chord:
            if drop_chords:
                # On easy, only keep one note from each chord
                if note.tick in seen_ticks:
                    continue
            seen_ticks.add(note.tick)

        # Thin notes for lower difficulties
        if note.tick not in seen_ticks:
            # This is a "new" note position
            counter += 1
            if counter % skip != 0:
                continue

        seen_ticks.add(note.tick)
        result.append(note)

    return result


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------


def is_available() -> bool:
    """Check if MIDI parsing is available (mido installed)."""
    return MIDO_AVAILABLE


def parse_midi_file(
    file_path: str | Path,
    target_instrument: str | None = None,
) -> MidiSongData:
    """
    Parse a MIDI file and extract all musical information.

    Parameters
    ----------
    file_path : str or Path
        Path to the .mid / .midi file.
    target_instrument : str, optional
        If specified, only extract notes for this instrument.
        Otherwise, all detected instrument tracks are parsed.

    Returns
    -------
    MidiSongData
        Complete parsed song data ready for chart generation.

    Raises
    ------
    ImportError
        If mido is not installed.
    FileNotFoundError
        If the MIDI file doesn't exist.
    ValueError
        If the file is not a valid MIDI file.
    """
    if not MIDO_AVAILABLE:
        raise ImportError(
            "mido is required for MIDI parsing. Install with: pip install mido"
        )

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"MIDI file not found: {file_path}")

    logger.info("🎹 Parsing MIDI file: {}", file_path.name)

    try:
        import mido as _mido

        midi = _mido.MidiFile(str(file_path))
    except Exception as e:
        raise ValueError(f"Failed to parse MIDI file: {e}") from e

    tpb = midi.ticks_per_beat or DEFAULT_MIDI_TPB
    song_data = MidiSongData(
        source_file=str(file_path),
        midi_format=midi.type,
        ticks_per_beat=tpb,
    )

    # ── Pass 1: Extract tempo map, time signatures, and section markers ──
    tempo_events: list[MidiTempoEvent] = []
    time_signatures: list[MidiTimeSignature] = []
    sections: list[MidiSection] = []
    raw_track_notes: dict[int, list[MidiNote]] = defaultdict(list)
    track_names: dict[int, str] = {}
    track_channels: dict[int, set[int]] = defaultdict(set)

    for track_idx, track in enumerate(midi.tracks):
        abs_tick = 0
        track_name = track.name or f"Track {track_idx}"
        pending_notes: dict[tuple[int, int], tuple[int, int]] = {}
        # (channel, pitch) → (start_tick, velocity)

        for msg in track:
            abs_tick += msg.time  # msg.time is delta ticks

            if msg.type == "track_name":
                track_name = msg.name or track_name
                track_names[track_idx] = track_name

            elif msg.type == "set_tempo":
                import mido as _mido

                uspb = msg.tempo
                bpm = _mido.tempo2bpm(uspb)
                tempo_events.append(
                    MidiTempoEvent(
                        tick=abs_tick,
                        tempo_bpm=round(bpm, 3),
                        microseconds_per_beat=uspb,
                    )
                )

            elif msg.type == "time_signature":
                time_signatures.append(
                    MidiTimeSignature(
                        tick=abs_tick,
                        numerator=msg.numerator,
                        denominator=msg.denominator,
                    )
                )

            elif msg.type in ("marker", "text"):
                text = msg.text.strip() if hasattr(msg, "text") else ""
                if text:
                    label = _normalise_section_label(text)
                    sections.append(
                        MidiSection(
                            tick=abs_tick,
                            time_seconds=0.0,  # computed later
                            label=label,
                            raw_text=text,
                        )
                    )

            elif msg.type == "note_on" and msg.velocity > 0:
                key = (msg.channel, msg.note)
                pending_notes[key] = (abs_tick, msg.velocity)
                track_channels[track_idx].add(msg.channel)

            elif msg.type == "note_off" or (
                msg.type == "note_on" and msg.velocity == 0
            ):
                key = (msg.channel, msg.note)
                if key in pending_notes:
                    start_tick, velocity = pending_notes.pop(key)
                    duration = abs_tick - start_tick
                    raw_track_notes[track_idx].append(
                        MidiNote(
                            tick=start_tick,
                            duration_ticks=max(duration, 1),
                            time_seconds=0.0,  # computed later
                            duration_seconds=0.0,
                            pitch=msg.note,
                            velocity=velocity,
                            channel=msg.channel,
                        )
                    )

        # Store track name if we didn't get one from a track_name event
        if track_idx not in track_names:
            track_names[track_idx] = track_name

    # ── Default tempo if none found ──
    if not tempo_events:
        tempo_events.append(
            MidiTempoEvent(
                tick=0,
                tempo_bpm=120.0,
                microseconds_per_beat=500_000,  # 120 BPM
            )
        )

    # Sort tempo events by tick
    tempo_events.sort(key=lambda e: e.tick)
    song_data.initial_tempo = tempo_events[0].tempo_bpm

    # ── Compute absolute times for tempo events ──
    tempo_events = _build_tempo_map_seconds(tempo_events, tpb)
    song_data.tempo_events = tempo_events

    # ── Sort and store time signatures ──
    time_signatures.sort(key=lambda e: e.tick)
    for ts in time_signatures:
        ts.time_seconds = _tick_to_seconds(ts.tick, tempo_events, tpb)
    song_data.time_signatures = time_signatures

    if not time_signatures:
        song_data.time_signatures = [
            MidiTimeSignature(tick=0, numerator=4, denominator=4)
        ]

    # ── Compute times for section markers ──
    for sec in sections:
        sec.time_seconds = _tick_to_seconds(sec.tick, tempo_events, tpb)
    sections.sort(key=lambda s: s.time_seconds)

    # Deduplicate sections at very close times (within 0.1s)
    deduped_sections: list[MidiSection] = []
    for sec in sections:
        if (
            not deduped_sections
            or abs(sec.time_seconds - deduped_sections[-1].time_seconds) > 0.1
        ):
            deduped_sections.append(sec)
    song_data.sections = deduped_sections

    # ── Pass 2: Process tracks — identify instruments, map lanes ──
    max_end_time = 0.0
    all_instruments: set[str] = set()

    for track_idx, notes in raw_track_notes.items():
        if not notes:
            continue

        # Sort notes by tick
        notes.sort(key=lambda n: (n.tick, n.pitch))

        # Compute times
        for n in notes:
            n.time_seconds = _tick_to_seconds(n.tick, tempo_events, tpb)
            end_tick = n.tick + n.duration_ticks
            n.duration_seconds = max(
                _tick_to_seconds(end_tick, tempo_events, tpb) - n.time_seconds,
                0.001,
            )
            end_time = n.time_seconds + n.duration_seconds
            if end_time > max_end_time:
                max_end_time = end_time

        # Identify instrument
        track_name = track_names.get(track_idx, f"Track {track_idx}")
        channels = track_channels.get(track_idx, set())
        primary_channel = min(channels) if channels else None

        instrument = _identify_instrument_from_name(track_name)
        if instrument in ("unknown", "sections"):
            instrument = _identify_instrument_from_notes(notes, primary_channel)

        # Skip section/marker-only tracks
        if instrument == "sections":
            continue

        # Apply target instrument filter
        if target_instrument and instrument != target_instrument:
            # Still process if it's "unknown" (might be our target)
            if instrument != "unknown":
                continue

        # Detect Rock Band format
        is_rb = _is_rb_format(notes)
        if is_rb:
            song_data.is_rb_format = True

        # Determine if this is a drum track
        is_drum = (
            instrument == "drums"
            or primary_channel == 9
            or (is_rb and instrument == "drums")
        )

        # Map notes to lanes
        if is_rb:
            notes = map_rb_notes_to_lanes(notes, difficulty="expert")
        elif is_drum:
            is_gm = primary_channel == 9
            notes = map_drum_notes_to_lanes(notes, is_gm=is_gm)
        else:
            # Pitch-range mapping for melodic instruments
            max_lane = 4
            notes = map_pitch_range_to_lanes(
                notes,
                max_lane=max_lane,
                open_threshold_semitones=3 if instrument in ("guitar", "bass") else 0,
            )

        # Mark chords
        notes = _mark_chords(notes, tolerance_ticks=max(tpb // 32, 5))

        # Build pitch range info
        pitches = [n.pitch for n in notes]
        pitch_range = (min(pitches), max(pitches)) if pitches else (0, 0)

        # Store track info
        track_info = MidiTrackInfo(
            index=track_idx,
            name=track_name,
            instrument=instrument,
            note_count=len(notes),
            channel=primary_channel,
            pitch_range=pitch_range,
            is_drum=is_drum,
        )
        song_data.track_info.append(track_info)

        # Resolve instrument name for storage (handle duplicates)
        store_name = instrument
        if store_name in song_data.tracks:
            # Multiple tracks for same instrument — merge or disambiguate
            existing = song_data.tracks[store_name]
            if len(notes) > len(existing):
                # New track has more notes — replace
                song_data.tracks[store_name] = notes
            else:
                # Keep existing, store new under qualified name
                qualified = f"{instrument}_{track_name.lower().replace(' ', '_')}"
                song_data.tracks[qualified] = notes
        else:
            song_data.tracks[store_name] = notes

        all_instruments.add(instrument)

    song_data.duration = max_end_time
    song_data.available_instruments = sorted(all_instruments)

    # ── Compute beat times from tempo map ──
    song_data.beat_times = _compute_beat_times(tempo_events, tpb, max_end_time)

    # ── Summary logging ──
    logger.info(
        "✅ MIDI parsed: format={}, tpb={}, tempo={:.1f} BPM, "
        "{} tempo changes, {} time sigs, {} sections, "
        "{} tracks with notes, duration={:.1f}s, instruments={}",
        midi.type,
        tpb,
        song_data.initial_tempo,
        len(tempo_events),
        len(time_signatures),
        len(song_data.sections),
        len(song_data.tracks),
        song_data.duration,
        song_data.available_instruments,
    )

    for ti in song_data.track_info:
        logger.debug(
            "  📌 Track {}: '{}' → {} ({} notes, ch={}, range={}-{}{})",
            ti.index,
            ti.name,
            ti.instrument,
            ti.note_count,
            ti.channel,
            _midi_note_to_name(ti.pitch_range[0]),
            _midi_note_to_name(ti.pitch_range[1]),
            " [DRUMS]" if ti.is_drum else "",
        )

    return song_data


# ---------------------------------------------------------------------------
# High-level helpers for the song generator integration
# ---------------------------------------------------------------------------


def midi_to_chart_events(
    midi_data: MidiSongData,
    instrument: str = "guitar",
    difficulty: str = "expert",
    chart_resolution: int = CHART_RESOLUTION,
) -> dict[str, Any]:
    """
    Convert parsed MIDI data into the format expected by the chart generator.

    Returns a dict compatible with what ``analyze_audio`` returns, but with
    MIDI-derived precision:

        - tempo (float): initial BPM
        - beat_times (list[float]): beat positions in seconds
        - onset_times (list[float]): note onset positions in seconds
        - onset_strengths (list[float]): normalised velocities (0-1)
        - duration (float): song duration in seconds
        - segments (list[dict]): section markers
        - midi_lanes (list[int]): pre-computed lane assignments
        - midi_sustains (list[int]): sustain durations in chart ticks
        - midi_chords (list[bool]): whether each note is part of a chord
        - midi_velocities (list[int]): raw MIDI velocities (0-127)
        - tempo_map (list[tuple]): (chart_tick, milli_bpm) pairs
        - time_signatures (list[tuple]): (chart_tick, num, denom) tuples
    """
    notes = midi_data.get_notes_for_instrument(instrument)

    if not notes:
        logger.warning(
            "⚠️ No {} notes found in MIDI. Available instruments: {}",
            instrument,
            midi_data.available_instruments,
        )
        return {
            "tempo": midi_data.initial_tempo,
            "beat_times": midi_data.beat_times,
            "onset_times": [],
            "onset_strengths": [],
            "duration": midi_data.duration,
            "segments": midi_data.to_segments(),
            "midi_lanes": [],
            "midi_sustains": [],
            "midi_chords": [],
            "midi_velocities": [],
            "tempo_map": midi_data.to_tempo_map(chart_resolution),
            "time_signatures": midi_data.to_time_signature_events(chart_resolution),
        }

    # Filter notes for the requested difficulty
    filtered = filter_notes_for_difficulty(notes, difficulty)

    # Extract data arrays
    onset_times = [n.time_seconds for n in filtered]
    max_vel = max((n.velocity for n in filtered), default=1) or 1
    onset_strengths = [n.velocity / max_vel for n in filtered]
    lanes = [n.lane for n in filtered]
    velocities = [n.velocity for n in filtered]
    chords = [n.is_chord for n in filtered]

    # Compute sustain durations in chart ticks
    tick_ratio = chart_resolution / midi_data.ticks_per_beat
    sustains: list[int] = []
    for n in filtered:
        chart_dur = int(round(n.duration_ticks * tick_ratio))
        # Only emit sustain if it's meaningful (>= half a beat)
        if chart_dur >= chart_resolution // 2:
            sustains.append(chart_dur)
        else:
            sustains.append(0)

    segments = midi_data.to_segments()
    if not segments and midi_data.duration > 0:
        # Generate basic segments from the note distribution
        segments = _auto_segments_from_notes(filtered, midi_data.duration)

    return {
        "tempo": midi_data.initial_tempo,
        "beat_times": midi_data.beat_times,
        "onset_times": onset_times,
        "onset_strengths": onset_strengths,
        "duration": midi_data.duration,
        "segments": segments,
        "midi_lanes": lanes,
        "midi_sustains": sustains,
        "midi_chords": chords,
        "midi_velocities": velocities,
        "tempo_map": midi_data.to_tempo_map(chart_resolution),
        "time_signatures": midi_data.to_time_signature_events(chart_resolution),
    }


def _auto_segments_from_notes(
    notes: list[MidiNote],
    duration: float,
) -> list[dict[str, Any]]:
    """Generate rough section markers from note density changes.

    When the MIDI file has no marker events, this analyses the note
    distribution to create reasonable section boundaries.
    """
    if not notes or duration <= 0:
        return []

    # Compute note density in 5-second windows
    window = 5.0
    n_windows = max(1, int(math.ceil(duration / window)))
    densities: list[float] = []

    for i in range(n_windows):
        t_start = i * window
        t_end = t_start + window
        count = sum(1 for n in notes if t_start <= n.time_seconds < t_end)
        densities.append(count / window)

    if not densities:
        return []

    avg_density = sum(densities) / len(densities)

    # Find windows where density changes significantly
    segments: list[dict[str, Any]] = []
    section_labels = [
        "Intro",
        "Verse",
        "Chorus",
        "Bridge",
        "Solo",
        "Verse 2",
        "Chorus 2",
        "Outro",
    ]
    label_idx = 0

    segments.append(
        {
            "time": 0.0,
            "label": section_labels[0],
            "energy": 0.5,
        }
    )
    label_idx = 1

    prev_density = densities[0] if densities else 0
    for i in range(1, len(densities)):
        change = abs(densities[i] - prev_density)
        if change > avg_density * 0.5 and densities[i] != prev_density:
            label = section_labels[label_idx % len(section_labels)]
            label_idx += 1
            rel_energy = min(densities[i] / (max(densities) or 1), 1.0)
            segments.append(
                {
                    "time": round(i * window, 3),
                    "label": label,
                    "energy": round(rel_energy, 3),
                }
            )
        prev_density = densities[i]

    return segments


def merge_midi_and_audio_analysis(
    midi_data: dict[str, Any],
    audio_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Merge MIDI-derived data with audio analysis results.

    MIDI provides precise note placement and lane assignments.
    Audio provides energy/dynamics information, verified tempo, and
    onset strengths that reflect the actual recorded performance.

    Priority:
        - Tempo: use MIDI (exact) unless audio differs significantly
          (which might indicate the MIDI is out of sync)
        - Note timing: use MIDI onset_times (precise note positions)
        - Lane assignments: use MIDI midi_lanes (from actual pitches)
        - Onset strengths: blend MIDI velocity with audio onset strength
        - Segments: prefer MIDI sections, fall back to audio if empty
        - Beat times: use MIDI (derived from exact tempo map)
        - Sustains: use MIDI midi_sustains (from note durations)
        - Duration: use audio (more reliable than MIDI end-of-track)
    """
    result = dict(midi_data)  # start with MIDI as base

    # ── Tempo sanity check ──
    midi_tempo = midi_data.get("tempo", 120.0)
    audio_tempo = audio_data.get("tempo", 120.0)

    if audio_tempo > 0 and midi_tempo > 0:
        tempo_ratio = midi_tempo / audio_tempo
        # If they're within 5%, trust MIDI. Otherwise log a warning.
        if not (0.95 <= tempo_ratio <= 1.05):
            # Check for common tempo detection errors (half/double time)
            if 0.45 <= tempo_ratio <= 0.55:
                logger.warning(
                    "⚠️ MIDI tempo ({:.1f}) is ~half of audio ({:.1f}). "
                    "Audio detector may have doubled the tempo.",
                    midi_tempo,
                    audio_tempo,
                )
            elif 1.9 <= tempo_ratio <= 2.1:
                logger.warning(
                    "⚠️ MIDI tempo ({:.1f}) is ~double audio ({:.1f}). "
                    "Audio detector may have halved the tempo.",
                    midi_tempo,
                    audio_tempo,
                )
            else:
                logger.warning(
                    "⚠️ Tempo mismatch: MIDI={:.1f}, audio={:.1f}. "
                    "Using MIDI tempo (assumed correct). "
                    "Check that MIDI and audio are the same song.",
                    midi_tempo,
                    audio_tempo,
                )

    # ── Duration: prefer audio (more reliable end detection) ──
    audio_duration = audio_data.get("duration", 0.0)
    if audio_duration > 0:
        result["duration"] = audio_duration

    # ── Segments: prefer MIDI, fall back to audio ──
    midi_segments = midi_data.get("segments", [])
    audio_segments = audio_data.get("segments", [])

    if midi_segments and len(midi_segments) >= 2:
        # Enrich MIDI segments with audio energy data
        result["segments"] = _enrich_segments_with_energy(midi_segments, audio_data)
    elif audio_segments:
        result["segments"] = audio_segments

    # ── Onset strengths: blend MIDI velocity with audio energy ──
    midi_strengths = midi_data.get("onset_strengths", [])
    audio_rms_values = audio_data.get("rms_values", [])
    audio_rms_times = audio_data.get("rms_times", [])
    midi_onsets = midi_data.get("onset_times", [])

    if midi_strengths and audio_rms_values and audio_rms_times:
        blended = _blend_strengths(
            midi_onsets,
            midi_strengths,
            audio_rms_times,
            audio_rms_values,
            blend_ratio=0.7,  # 70% MIDI, 30% audio
        )
        result["onset_strengths"] = blended

    return result


def _enrich_segments_with_energy(
    segments: list[dict[str, Any]],
    audio_data: dict[str, Any],
) -> list[dict[str, Any]]:
    """Add audio-derived energy values to MIDI-derived sections."""
    rms_values = audio_data.get("rms_values", [])
    rms_times = audio_data.get("rms_times", [])

    if not rms_values or not rms_times:
        return segments

    max_rms = max(rms_values) if rms_values else 1.0
    if max_rms <= 0:
        max_rms = 1.0

    enriched = []
    for i, seg in enumerate(segments):
        seg_start = seg["time"]
        seg_end = (
            segments[i + 1]["time"]
            if i + 1 < len(segments)
            else audio_data.get("duration", seg_start + 30)
        )

        # Compute average RMS in this segment's time range
        rms_sum = 0.0
        rms_count = 0
        for t, v in zip(rms_times, rms_values):
            if seg_start <= t < seg_end:
                rms_sum += v
                rms_count += 1

        energy = (rms_sum / rms_count / max_rms) if rms_count > 0 else 0.5

        enriched.append(
            {
                "time": seg["time"],
                "label": seg["label"],
                "energy": round(energy, 4),
            }
        )

    return enriched


def _blend_strengths(
    midi_times: list[float],
    midi_strengths: list[float],
    rms_times: list[float],
    rms_values: list[float],
    blend_ratio: float = 0.7,
) -> list[float]:
    """Blend MIDI velocities with audio RMS energy.

    For each MIDI note onset, find the nearest RMS value and blend:
        blended = blend_ratio * midi_strength + (1 - blend_ratio) * audio_strength
    """
    if not midi_times or not midi_strengths:
        return midi_strengths

    max_rms = max(rms_values) if rms_values else 1.0
    if max_rms <= 0:
        max_rms = 1.0

    blended: list[float] = []
    rms_idx = 0

    for t, ms in zip(midi_times, midi_strengths):
        # Find nearest RMS frame
        while rms_idx < len(rms_times) - 1 and rms_times[rms_idx + 1] <= t:
            rms_idx += 1

        if rms_idx < len(rms_values):
            audio_strength = rms_values[rms_idx] / max_rms
        else:
            audio_strength = 0.5

        combined = blend_ratio * ms + (1.0 - blend_ratio) * audio_strength
        blended.append(round(min(combined, 1.0), 4))

    return blended


# ---------------------------------------------------------------------------
# Convenience: parse and summarise for the UI
# ---------------------------------------------------------------------------


def get_midi_summary(file_path: str | Path) -> dict[str, Any]:
    """
    Parse a MIDI file and return a JSON-serialisable summary for the UI.

    This is used by the API to show MIDI information in the generator
    form before the user clicks Generate.
    """
    try:
        data = parse_midi_file(file_path)
        return {
            "valid": True,
            "format": data.midi_format,
            "ticks_per_beat": data.ticks_per_beat,
            "initial_tempo": data.initial_tempo,
            "duration": round(data.duration, 2),
            "tempo_changes": len(data.tempo_events),
            "time_signatures": [
                f"{ts.numerator}/{ts.denominator}" for ts in data.time_signatures
            ],
            "sections": len(data.sections),
            "section_labels": [s.label for s in data.sections],
            "instruments": data.available_instruments,
            "tracks": [
                {
                    "index": ti.index,
                    "name": ti.name,
                    "instrument": ti.instrument,
                    "notes": ti.note_count,
                    "is_drum": ti.is_drum,
                    "pitch_range": f"{_midi_note_to_name(ti.pitch_range[0])}-{_midi_note_to_name(ti.pitch_range[1])}",
                }
                for ti in data.track_info
            ],
            "is_rb_format": data.is_rb_format,
            "total_notes": sum(len(notes) for notes in data.tracks.values()),
            "beat_count": len(data.beat_times),
        }
    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
        }
