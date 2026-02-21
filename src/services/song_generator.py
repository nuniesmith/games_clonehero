"""
Clone Hero Content Manager - Song Generator Service

Analyses audio files using librosa to detect tempo, beats, and rhythm patterns,
then generates Clone Hero compatible notes.chart files.

Audio files are automatically converted to OGG Vorbis format for maximum
compatibility across Clone Hero versions (FLAC, WAV, etc. are not reliably
supported by all builds).

Generated songs are uploaded to Nextcloud via WebDAV and registered in the
local metadata cache.  Local disk is only used as a transient staging area.
"""

import json
import random
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
from loguru import logger

from src.config import TEMP_DIR
from src.services.album_art_generator import generate_album_art
from src.services.album_art_generator import is_available as album_art_available
from src.services.content_manager import write_song_ini
from src.services.lyrics_generator import generate_lyrics_for_chart
from src.services.stem_separator import (
    StemAnalysis,
    analyze_instrument,
    get_difficulty_profile_for_instrument,
    pitch_contour_to_lanes,
)

# ---------------------------------------------------------------------------
# Audio format helpers
# ---------------------------------------------------------------------------

# Formats that Clone Hero reliably supports across all versions.
# FLAC support is version-dependent, so we always convert to OGG.
CLONE_HERO_SAFE_EXTENSIONS = {".ogg", ".opus", ".mp3"}


def _ffmpeg_available() -> bool:
    """Return True if ffmpeg is on the system PATH."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def convert_audio_to_ogg(
    source_path: Path,
    dest_dir: Path,
    target_filename: str = "song.ogg",
    bitrate: str = "192k",
) -> Optional[Path]:
    """
    Convert an audio file to OGG Vorbis using ffmpeg.

    Parameters
    ----------
    source_path : Path
        Input audio file (FLAC, WAV, MP3, etc.).
    dest_dir : Path
        Directory where the output file will be written.
    target_filename : str
        Name of the output file (default ``song.ogg``).
    bitrate : str
        Target audio bitrate for the OGG encoder (default ``192k``).

    Returns
    -------
    Path or None
        Path to the converted file on success, ``None`` on failure.
    """
    dest_path = dest_dir / target_filename
    cmd = [
        "ffmpeg",
        "-y",  # overwrite without asking
        "-i",
        str(source_path),
        "-vn",  # strip any embedded artwork / video
        "-c:a",
        "libvorbis",
        "-b:a",
        bitrate,
        str(dest_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
        )
        if result.returncode == 0 and dest_path.exists():
            logger.info(
                "🔄 Converted {} → {} ({})",
                source_path.name,
                target_filename,
                bitrate,
            )
            return dest_path
        else:
            stderr_text = result.stderr.decode(errors="replace")[-500:]
            logger.error(
                "❌ ffmpeg conversion failed (rc={}): {}",
                result.returncode,
                stderr_text,
            )
            return None
    except FileNotFoundError:
        logger.error("❌ ffmpeg not found — cannot convert audio")
        return None
    except subprocess.TimeoutExpired:
        logger.error("❌ ffmpeg conversion timed out (>120s)")
        return None
    except Exception as e:
        logger.error("❌ Audio conversion error: {}", e)
        return None


# ---------------------------------------------------------------------------
# Audio analysis
# ---------------------------------------------------------------------------
def analyze_audio(file_path: str) -> Dict[str, Any]:
    """
    Analyse an audio file to extract musical features.

    Returns a dict with:
        - tempo (float): detected BPM
        - beat_times (list[float]): timestamps of detected beats in seconds
        - duration (float): total duration in seconds
        - onset_times (list[float]): onset event timestamps
        - onset_strengths (list[float]): strength/energy of each onset
        - segments (list[dict]): detected structural segments
    """
    try:
        logger.info("🎵 Analysing audio: {}", file_path)

        y, sr = librosa.load(file_path, sr=None)
        duration = float(librosa.get_duration(y=y, sr=sr))

        # Beat tracking
        tempo_arr, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        # librosa may return tempo as an ndarray; extract scalar
        if isinstance(tempo_arr, np.ndarray):
            tempo = (
                float(tempo_arr.item()) if tempo_arr.size == 1 else float(tempo_arr[0])
            )
        else:
            tempo = float(tempo_arr)

        beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

        # Onset detection for note placement
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=onset_env)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr).tolist()

        # Get onset strengths for each detected onset (normalised 0-1)
        if len(onset_frames) > 0:
            raw_strengths = onset_env[onset_frames]
            max_s = float(np.max(raw_strengths))
            if max_s > 0:
                onset_strengths = (raw_strengths / max_s).tolist()
            else:
                onset_strengths = [0.5] * len(onset_frames)
        else:
            onset_strengths: list[float] = []

        # RMS energy for dynamics
        rms = librosa.feature.rms(y=y)[0]
        rms_times = librosa.frames_to_time(range(len(rms)), sr=sr).tolist()

        # Try to detect structural segments via spectral clustering
        segments = _detect_segments(y, int(sr), duration, beat_times)

        logger.info(
            "✅ Audio analysis complete: tempo={:.1f} BPM, "
            "{} beats, {} onsets, duration={:.1f}s, {} segments",
            tempo,
            len(beat_times),
            len(onset_times),
            duration,
            len(segments),
        )

        return {
            "tempo": round(tempo, 2),
            "beat_times": beat_times,
            "onset_times": onset_times,
            "onset_strengths": onset_strengths,
            "duration": round(duration, 2),
            "rms_times": rms_times,
            "rms_values": rms.tolist(),
            "segments": segments,
        }

    except Exception as e:
        logger.error("❌ Error analysing audio {}: {}", file_path, e)
        raise


def _detect_segments(
    y: np.ndarray, sr: int, duration: float, beat_times: List[float]
) -> List[Dict[str, Any]]:
    """
    Detect structural segments in the audio (intro, verse, chorus, etc.).

    Falls back to evenly-spaced sections if segmentation fails.
    """
    try:
        # Use spectral features to detect change points
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        bounds = librosa.segment.agglomerative(chroma, k=8)
        bound_times = librosa.frames_to_time(bounds, sr=sr).tolist()

        # Label segments based on position and energy
        rms = librosa.feature.rms(y=y)[0]
        rms_frames = np.array(range(len(rms)))

        segments = []
        section_labels = _generate_section_labels(len(bound_times) + 1)

        # Add the start
        prev_time = 0.0
        for i, bt in enumerate(bound_times):
            if bt - prev_time < 2.0:
                # Skip very short segments
                continue

            # Compute average energy for this segment
            start_frame = librosa.time_to_frames(prev_time, sr=sr)
            end_frame = librosa.time_to_frames(bt, sr=sr)
            end_frame = min(end_frame, len(rms) - 1)
            seg_rms = (
                float(np.mean(rms[start_frame : end_frame + 1]))
                if end_frame > start_frame
                else 0.0
            )

            label = section_labels[min(i, len(section_labels) - 1)]
            segments.append(
                {
                    "time": round(prev_time, 3),
                    "label": label,
                    "energy": round(seg_rms, 6),
                }
            )
            prev_time = bt

        # Add the final segment
        if prev_time < duration - 2.0:
            segments.append(
                {
                    "time": round(prev_time, 3),
                    "label": "Outro",
                    "energy": 0.0,
                }
            )

        if len(segments) < 3:
            raise ValueError("Too few segments detected, falling back")

        return segments

    except Exception:
        # Fallback: evenly-spaced generic sections
        return _fallback_segments(duration)


def _generate_section_labels(count: int) -> List[str]:
    """Generate musically sensible section labels for a given segment count."""
    # Templates for different song lengths
    templates = {
        3: ["Intro", "Main", "Outro"],
        4: ["Intro", "Verse", "Chorus", "Outro"],
        5: ["Intro", "Verse 1", "Chorus", "Verse 2", "Outro"],
        6: ["Intro", "Verse 1", "Pre-Chorus", "Chorus", "Verse 2", "Outro"],
        7: ["Intro", "Verse 1", "Pre-Chorus", "Chorus", "Verse 2", "Bridge", "Outro"],
        8: [
            "Intro",
            "Verse 1",
            "Pre-Chorus",
            "Chorus",
            "Verse 2",
            "Bridge",
            "Final Chorus",
            "Outro",
        ],
        9: [
            "Intro",
            "Verse 1",
            "Pre-Chorus",
            "Chorus 1",
            "Verse 2",
            "Chorus 2",
            "Bridge",
            "Final Chorus",
            "Outro",
        ],
        10: [
            "Intro",
            "Verse 1",
            "Pre-Chorus",
            "Chorus 1",
            "Verse 2",
            "Pre-Chorus 2",
            "Chorus 2",
            "Bridge",
            "Final Chorus",
            "Outro",
        ],
    }

    clamped = max(3, min(count, 10))
    labels = templates[clamped]

    # Extend if needed
    while len(labels) < count:
        labels.insert(-1, f"Section {len(labels)}")

    return labels[:count]


def _fallback_segments(duration: float) -> List[Dict[str, Any]]:
    """Generate evenly-spaced fallback segments."""
    num_sections = max(4, min(8, int(duration / 30)))
    interval = duration / num_sections
    labels = _generate_section_labels(num_sections)
    segments = []
    for i in range(num_sections):
        segments.append(
            {
                "time": round(i * interval, 3),
                "label": labels[i],
                "energy": 0.5,
            }
        )
    return segments


def analyze_audio_detailed(file_path: str) -> Dict[str, Any]:
    """
    Extended audio analysis including spectral features.
    Useful for more intelligent note generation in future iterations.
    """
    basic = analyze_audio(file_path)

    try:
        y, sr = librosa.load(file_path, sr=None)

        # Spectral centroid (brightness)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        avg_centroid = float(np.mean(spectral_centroids))

        # RMS energy
        rms = librosa.feature.rms(y=y)
        avg_energy = float(np.mean(rms))

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        avg_zcr = float(np.mean(zcr))

        basic["spectral"] = {
            "avg_centroid": round(avg_centroid, 2),
            "avg_energy": round(avg_energy, 6),
            "avg_zero_crossing_rate": round(avg_zcr, 6),
        }

    except Exception as e:
        logger.warning("⚠️ Extended analysis failed (continuing with basic): {}", e)

    return basic


# ---------------------------------------------------------------------------
# Chart generation — constants and helpers
# ---------------------------------------------------------------------------
# Clone Hero lane constants
# Standard fretted notes: Green=0, Red=1, Yellow=2, Blue=3, Orange=4
# Open note (purple bar): 7  — used heavily in metal (palm mutes, chugs)
LANE_GREEN = 0
LANE_RED = 1
LANE_YELLOW = 2
LANE_BLUE = 3
LANE_ORANGE = 4
LANE_OPEN = 7  # Purple bar — open string

NOTE_LANES = 6  # Green(0), Red(1), Yellow(2), Blue(3), Orange(4), Open(7)
RESOLUTION = 192  # Standard ticks per quarter note

# Difficulty settings: controls note density, max lane, chord probability, etc.
# allow_open: whether open notes (lane 7 / purple bar) are enabled for this
#             difficulty.  Open notes are essential for metal charting.
DIFFICULTY_PROFILES = {
    "easy": {
        "section_name": "EasySingle",
        "note_skip": 4,  # use every Nth onset
        "max_lane": 2,  # only green/red/yellow
        "chord_chance": 0.0,  # no chords
        "hopo_chance": 0.0,  # no HOPOs
        "sustain_chance": 0.05,
        "min_note_gap_ticks": 192,  # at least 1 beat apart
        "allow_open": False,  # no open notes on easy
    },
    "medium": {
        "section_name": "MediumSingle",
        "note_skip": 3,
        "max_lane": 3,  # up to blue
        "chord_chance": 0.05,
        "hopo_chance": 0.05,
        "sustain_chance": 0.10,
        "min_note_gap_ticks": 96,  # half beat
        "allow_open": False,  # no open notes on medium
    },
    "hard": {
        "section_name": "HardSingle",
        "note_skip": 2,
        "max_lane": 4,  # all 5 fretted lanes
        "chord_chance": 0.10,
        "hopo_chance": 0.15,
        "sustain_chance": 0.15,
        "min_note_gap_ticks": 48,  # quarter beat
        "allow_open": True,  # open notes on hard+
    },
    "expert": {
        "section_name": "ExpertSingle",
        "note_skip": 1,  # use every onset
        "max_lane": 4,  # all 5 fretted lanes
        "chord_chance": 0.15,
        "hopo_chance": 0.25,
        "sustain_chance": 0.20,
        "min_note_gap_ticks": 24,  # eighth beat
        "allow_open": True,  # open notes on hard+
    },
}


def _seconds_to_ticks(
    time_s: float,
    tempo: float,
    resolution: int = RESOLUTION,
    tempo_map: Optional[List[Tuple[int, int]]] = None,
) -> int:
    """Convert a time in seconds to chart ticks.

    When *tempo_map* is ``None`` (or has a single entry) this uses the
    simple constant-tempo formula.

    When a multi-entry tempo map is supplied the conversion walks the
    piecewise-constant tempo segments so that ticks stay aligned with
    what Clone Hero actually does at playback time.

    Parameters
    ----------
    time_s : float
        Wall-clock time in seconds.
    tempo : float
        Initial (or only) BPM – used as the base for the constant-tempo
        path and as the reference BPM that was used to *build* the
        tempo map tick positions.
    resolution : int
        Ticks per quarter note (default 192).
    tempo_map : list of (tick, milli_bpm), optional
        The sync-track tempo map produced by
        ``_compute_stable_tempo_map``.  Each entry is
        ``(tick_position, bpm * 1000)``.
    """
    if time_s <= 0:
        return 0

    # --- fast path: constant tempo (most common) ---
    if not tempo_map or len(tempo_map) <= 1:
        beats = time_s * (tempo / 60.0)
        return int(round(beats * resolution))

    # --- piecewise conversion ---
    # Walk through tempo segments, accumulating elapsed seconds until we
    # reach *time_s*, then return the corresponding tick.
    remaining = time_s
    prev_tick = 0
    prev_bpm = tempo_map[0][1] / 1000.0  # milli-BPM → BPM

    for idx in range(1, len(tempo_map)):
        seg_tick, seg_milli_bpm = tempo_map[idx]
        seg_bpm = seg_milli_bpm / 1000.0

        # How many seconds does the segment [prev_tick .. seg_tick) span
        # at prev_bpm?
        seg_ticks = seg_tick - prev_tick
        if prev_bpm > 0:
            seg_seconds = seg_ticks / (prev_bpm / 60.0 * resolution)
        else:
            seg_seconds = 0.0

        if remaining <= seg_seconds:
            # Target time falls inside this segment
            extra_ticks = remaining * (prev_bpm / 60.0) * resolution
            return int(round(prev_tick + extra_ticks))

        remaining -= seg_seconds
        prev_tick = seg_tick
        prev_bpm = seg_bpm

    # Past the last tempo marker – continue at the final BPM
    extra_ticks = remaining * (prev_bpm / 60.0) * resolution
    return int(round(prev_tick + extra_ticks))


def _snap_to_nearest_beat(
    time_s: float,
    beat_times: List[float],
    max_snap_window: float = 0.15,
) -> float:
    """Snap a timestamp to the nearest detected beat if close enough.

    Parameters
    ----------
    time_s : float
        The raw timestamp to snap (seconds).
    beat_times : list of float
        Sorted beat positions in seconds.
    max_snap_window : float
        Maximum distance (seconds) to snap.  If no beat is within this
        window the original time is returned unchanged.

    Returns
    -------
    float
        The snapped timestamp.
    """
    if not beat_times:
        return time_s

    # Binary-search-like scan for nearest beat
    best = time_s
    best_dist = max_snap_window + 1.0
    for bt in beat_times:
        d = abs(bt - time_s)
        if d < best_dist:
            best_dist = d
            best = bt
        # Once beats are past our window we can stop
        if bt > time_s + max_snap_window:
            break

    return best if best_dist <= max_snap_window else time_s


def _bpm_to_chart_value(bpm: float) -> int:
    """
    Convert a BPM value to .chart format.

    The .chart format stores tempo as **milli-BPM** (thousandths of a BPM).
    For example, 180 BPM is stored as ``B 180000``.

    This is NOT the same as MIDI, which uses microseconds-per-beat.
    """
    return int(round(bpm * 1000))


def _compute_stable_tempo_map(
    tempo: float,
    beat_times: List[float],
    resolution: int = RESOLUTION,
    change_threshold: float = 0.15,
    min_stable_beats: int = 8,
) -> List[Tuple[int, int]]:
    """
    Compute a stable tempo map from beat times.

    Instead of creating a tempo marker for every beat-to-beat variation,
    this function groups consecutive beats with similar inter-beat intervals
    and only emits a tempo change when the tempo genuinely shifts (e.g., at
    a section boundary or intentional tempo change).

    Returns a list of (tick, milli_bpm) tuples.
    """
    if len(beat_times) < 4:
        return [(0, _bpm_to_chart_value(tempo))]

    markers: List[Tuple[int, int]] = []
    markers.append((0, _bpm_to_chart_value(tempo)))

    # Compute inter-beat intervals
    intervals = [
        beat_times[i] - beat_times[i - 1]
        for i in range(1, len(beat_times))
        if beat_times[i] - beat_times[i - 1] > 0
    ]

    if not intervals:
        return markers

    # Use a sliding window to detect genuine tempo changes
    current_bpm = tempo
    stable_count = 0

    for i in range(len(intervals)):
        local_bpm = 60.0 / intervals[i]

        # Clamp to reasonable range
        if local_bpm < 30 or local_bpm > 500:
            continue

        pct_diff = abs(local_bpm - current_bpm) / current_bpm

        if pct_diff < change_threshold:
            stable_count += 1
        else:
            # Potential tempo change — check if the new tempo is sustained
            if i + min_stable_beats < len(intervals):
                upcoming = intervals[i : i + min_stable_beats]
                upcoming_bpms = [60.0 / iv for iv in upcoming if iv > 0]
                if upcoming_bpms:
                    avg_upcoming = sum(upcoming_bpms) / len(upcoming_bpms)
                    # If the upcoming beats are stable around the new tempo
                    spread = max(upcoming_bpms) - min(upcoming_bpms)
                    if spread / avg_upcoming < change_threshold:
                        # Genuine tempo change
                        # NOTE: this intentionally uses the initial tempo for
                        # tick placement because the tempo map itself is being
                        # *built* here — there is no existing map to reference.
                        tick = _seconds_to_ticks(beat_times[i + 1], tempo, resolution)
                        new_bpm = round(avg_upcoming, 3)
                        markers.append((tick, _bpm_to_chart_value(new_bpm)))
                        current_bpm = new_bpm
                        stable_count = 0

    return markers


# ---------------------------------------------------------------------------
# Note pattern generation
# ---------------------------------------------------------------------------

# Pre-defined note patterns that feel natural to play.
# Each pattern is a list of lane indices (0-4, 7=open).
#
# Metal-aware patterns derived from real tab analysis
# (e.g. Paleface Swiss "Please End Me"):
#   0M-0M-0M       → Open-Open-Open (chugs)
#   1M-0M-0M       → Green-Open-Open (gallop)
#   3M-2M-3M-4M    → Yellow-Red-Yellow-Blue (chromatic run)
#   4---3---0M-0M   → Blue-Yellow-Open-Open (riff into chug)

PATTERNS_EASY = [
    [7, 7, 0, 0],  # open chugs → green
    [0, 0, 7, 7],  # green → open chugs
    [7, 7, 7, 0],  # mostly open
    [0, 1, 0, 7],  # green-red-green-open
    [7, 0, 7, 0],  # alternating open-green
    [7, 7, 7, 7],  # straight open chugs
]

PATTERNS_MEDIUM = [
    [7, 7, 0, 7, 7],  # open-open-green-open-open
    [0, 7, 7, 1, 7],  # green-open-open-red-open
    [7, 0, 1, 0, 7],  # open-green-red-green-open
    [2, 1, 7, 7, 0],  # yellow-red-open-open-green
    [7, 7, 7, 0, 1],  # chugs into movement
    [0, 7, 7, 2, 1],  # gallop pattern
]

PATTERNS_HARD = [
    [0, 7, 7, 1, 2, 1],  # green-open-open-red-yellow-red (gallop + riff)
    [7, 7, 0, 2, 4, 2],  # open chugs → chromatic
    [2, 1, 2, 3, 7, 7],  # chromatic run → chugs
    [3, 2, 0, 7, 7, 0],  # descending → open
    [7, 0, 1, 2, 3, 7],  # open → ascending → open
    [4, 3, 2, 1, 0, 7],  # full descend to open
]

PATTERNS_EXPERT = [
    [0, 7, 7, 0, 7, 7, 0, 7],  # gallop: green-open-open repeating (1M-0M-0M)
    [7, 7, 7, 7, 0, 1, 2, 3],  # open chugs → chromatic ascent
    [2, 1, 2, 3, 7, 7, 7, 7],  # chromatic run → open chugs (3M-2M-3M-4M → 0M)
    [7, 0, 7, 0, 2, 1, 2, 3],  # alternating open-green → chromatic
    [3, 2, 1, 0, 7, 7, 7, 7],  # descending frets → open chugs
    [0, 7, 7, 3, 2, 3, 4, 7],  # riff pattern (fret-open-open-riff)
    [7, 7, 0, 7, 7, 0, 7, 0],  # breakdown pattern
    [1, 7, 7, 4, 7, 7, 1, 7],  # power chord jumps with open between
]

DIFFICULTY_PATTERNS = {
    "easy": PATTERNS_EASY,
    "medium": PATTERNS_MEDIUM,
    "hard": PATTERNS_HARD,
    "expert": PATTERNS_EXPERT,
}

# Chord shapes (pairs/triples of simultaneous notes) — for hard/expert
# Open note (7) CANNOT be combined with fretted notes in standard Clone Hero
CHORD_SHAPES = [
    (0, 1),  # GR — power chord
    (1, 2),  # RY
    (2, 3),  # YB
    (3, 4),  # BO
    (0, 2),  # GY — wide interval
    (1, 3),  # RB
    (0, 1, 2),  # GRY (triple for expert)
    (2, 3, 4),  # YBO (triple for expert)
]


def _select_note(
    index: int,
    onset_strength: float,
    difficulty: str,
    section_index: int,
    rng: random.Random,
    profile: Dict[str, Any],
    pitch_lane: Optional[int] = None,
) -> List[int]:
    """
    Select which lane(s) a note should be on.

    Returns a list of lane indices. A single-element list is a single note;
    multiple elements means a chord.

    Uses the onset strength, position in the song, and difficulty profile
    to decide note placement.

    Supports all 6 Clone Hero lanes:
        Green(0), Red(1), Yellow(2), Blue(3), Orange(4), Open(7)

    Parameters
    ----------
    pitch_lane : int, optional
        If provided (from instrument-specific pitch analysis), use this as
        the base lane instead of the pattern-based selection.  This makes
        notes follow the actual melodic contour of the isolated instrument.
        Can be 7 for open notes.
    """
    max_lane = profile["max_lane"]
    chord_chance = profile["chord_chance"]
    allow_open = profile.get("allow_open", True)

    if pitch_lane is not None:
        if pitch_lane == 7:
            # Open note from pitch detection — always allow on hard/expert,
            # map to Green on easy/medium if open notes aren't enabled
            if allow_open:
                base_lane = 7
            else:
                base_lane = 0  # downgrade open to Green on easy difficulties
        else:
            # Fretted note — clamp to difficulty's max lane
            base_lane = min(pitch_lane, max_lane)
    else:
        # Pattern-based selection (includes open notes in patterns)
        patterns = DIFFICULTY_PATTERNS.get(difficulty, PATTERNS_EXPERT)
        pattern = patterns[(section_index + index // 16) % len(patterns)]
        base_lane = pattern[index % len(pattern)]

        # Handle open notes from patterns
        if base_lane == 7:
            if not allow_open:
                base_lane = 0  # downgrade to Green
        else:
            # Clamp fretted notes to max lane for this difficulty
            base_lane = min(base_lane, max_lane)

    # Open notes cannot be combined with fretted notes in standard CH
    if base_lane == 7:
        return [7]

    # Possibly make a chord on strong onsets (hard/expert only)
    if chord_chance > 0 and onset_strength > 0.7 and rng.random() < chord_chance:
        # Only use fretted chord shapes (no open note in chords)
        shapes = [
            s for s in CHORD_SHAPES if all(ln <= max_lane for ln in s) and 7 not in s
        ]
        if shapes:
            chord = list(rng.choice(shapes))
            # Bias toward shapes that include our base_lane
            near = [s for s in shapes if base_lane in s]
            if near:
                chord = list(rng.choice(near))
            return sorted(set(chord))

    return [base_lane]


def _should_hopo(
    tick: int,
    prev_tick: int,
    prev_lanes: List[int],
    lanes: List[int],
    profile: Dict[str, Any],
    rng: random.Random,
) -> bool:
    """
    Decide whether a note should be a HOPO (hammer-on / pull-off).

    HOPOs fire when the note is close to the previous one and on a different
    lane. In .chart format this is indicated by ``N 5 0`` alongside the note.

    Open notes (lane 7) can be HOPOs — this mirrors real guitar hammer-ons
    from open string (e.g. 0h-1h-0 patterns in metal tabs).
    """
    if profile["hopo_chance"] <= 0:
        return False
    if len(lanes) > 1:
        return False  # no HOPO on chords
    if lanes == prev_lanes:
        return False  # same lane, no HOPO
    # Open notes CAN be HOPOs (open→fretted or fretted→open)
    gap = tick - prev_tick
    if gap > RESOLUTION // 2:
        return False  # too far apart
    return rng.random() < profile["hopo_chance"]


def _compute_sustain(
    tick: int,
    next_tick: Optional[int],
    onset_strength: float,
    profile: Dict[str, Any],
    rng: random.Random,
) -> int:
    """
    Compute sustain duration for a note in ticks.

    Returns 0 for a normal tap note, or a positive tick count for a held note.
    Sustains only happen on strong onsets and when there's enough space
    before the next note.
    """
    if profile["sustain_chance"] <= 0:
        return 0
    if onset_strength < 0.6:
        return 0
    if rng.random() > profile["sustain_chance"]:
        return 0

    if next_tick is None:
        # Last note — give it a full beat sustain
        return RESOLUTION

    available = next_tick - tick - 24  # leave a small gap before next note
    if available < RESOLUTION // 2:
        return 0  # not enough room

    # Sustain for 1-2 beats, capped by available space
    desired = RESOLUTION * rng.randint(1, 2)
    return min(desired, available)


# ---------------------------------------------------------------------------
# Star power placement
# ---------------------------------------------------------------------------
def _compute_star_power_sections(
    onset_ticks: List[int],
    total_ticks: int,
) -> List[Tuple[int, int]]:
    """
    Place star power (SP) sections throughout the chart.

    Returns a list of (start_tick, duration_ticks) tuples.
    SP sections are roughly every 30-50 notes, lasting about 8-16 notes.
    """
    if len(onset_ticks) < 20:
        return []

    sp_sections = []
    sp_interval = max(30, len(onset_ticks) // 6)  # ~6 SP sections per song
    sp_length = max(8, sp_interval // 4)

    i = sp_interval // 2  # start the first SP a bit into the song
    while i + sp_length < len(onset_ticks):
        start = onset_ticks[i]
        end = onset_ticks[min(i + sp_length, len(onset_ticks) - 1)]
        sp_sections.append((start, end - start))
        i += sp_interval

    return sp_sections


# ---------------------------------------------------------------------------
# Main chart generation
# ---------------------------------------------------------------------------
def generate_notes_chart(
    song_name: str,
    artist: str,
    album: str,
    year: str,
    genre: str,
    tempo: float,
    beat_times: List[float],
    onset_times: List[float],
    onset_strengths: List[float],
    duration: float,
    output_path: Path,
    audio_filename: str = "song.ogg",
    segments: Optional[List[Dict[str, Any]]] = None,
    difficulties: Optional[List[str]] = None,
    seed: Optional[int] = None,
    enable_lyrics: bool = True,
    charter: str = "nuniesmith",
    instrument: str = "guitar",
    stem_analysis: Optional[StemAnalysis] = None,
) -> bool:
    """
    Generate a Clone Hero compatible notes.chart file.

    The chart includes:
        - [Song] section with full metadata and MusicStream
        - [SyncTrack] with stable tempo markers (milli-BPM)
        - [Events] with section markers and optional lyrics
        - Note sections for each requested difficulty
        - HOPO markers, chords, sustains, and star power

    Parameters
    ----------
    instrument : str
        Target instrument: ``guitar``, ``bass``, ``drums``, ``vocals``,
        or ``full_mix``.  Determines chart section names and note mapping.
    stem_analysis : StemAnalysis, optional
        Pre-computed instrument-specific analysis from the stem separator.
        When provided, onset data and pitch contour come from the isolated
        stem rather than the full mix.

    Returns True on success, False on failure.
    """
    try:
        if difficulties is None:
            difficulties = ["easy", "medium", "hard", "expert"]

        rng = random.Random(seed or hash(song_name + artist))

        # Pad onset_strengths if shorter than onset_times
        while len(onset_strengths) < len(onset_times):
            onset_strengths.append(0.5)

        lines: List[str] = []

        # --- [Song] section ---
        lines.append("[Song]")
        lines.append("{")
        lines.append(f'  Name = "{song_name}"')
        lines.append(f'  Artist = "{artist}"')
        lines.append(f'  Album = "{album}"')
        lines.append(f'  Year = ", {year}"')
        lines.append(f'  Charter = "{charter}"')
        lines.append(f"  Offset = 0")
        lines.append(f"  Resolution = {RESOLUTION}")
        lines.append("  Player2 = bass")
        lines.append("  Difficulty = 0")
        lines.append("  PreviewStart = 0")
        lines.append("  PreviewEnd = 0")
        lines.append(f'  Genre = "{genre}"')
        lines.append('  MediaType = "cd"')
        lines.append(f'  MusicStream = "{audio_filename}"')
        lines.append("}")
        lines.append("")

        # --- [SyncTrack] section ---
        lines.append("[SyncTrack]")
        lines.append("{")

        tempo_map = _compute_stable_tempo_map(tempo, beat_times, RESOLUTION)
        for tick, milli_bpm in tempo_map:
            if tick == 0:
                lines.append("  0 = TS 4")
            lines.append(f"  {tick} = B {milli_bpm}")

        lines.append("}")
        lines.append("")

        # --- [Events] section ---
        # Collect ALL event lines first, then sort by tick to ensure
        # ascending order.  Clone Hero rejects charts with out-of-order
        # events as "corrupt".
        lines.append("[Events]")
        lines.append("{")

        used_segments = (
            segments
            if segments and len(segments) >= 2
            else _fallback_segments(duration)
        )

        # Gather section markers as (tick, line) tuples
        event_entries: List[Tuple[int, str]] = []
        for seg in used_segments:
            tick = _seconds_to_ticks(seg["time"], tempo, tempo_map=tempo_map)
            label = seg["label"]
            event_entries.append((tick, f'  {tick} = E "section {label}"'))

        # --- Lyric events (optional) ---
        if enable_lyrics:
            try:
                lyric_lines = generate_lyrics_for_chart(
                    tempo=tempo,
                    beat_times=beat_times,
                    onset_times=onset_times,
                    onset_strengths=onset_strengths,
                    duration=duration,
                    segments=used_segments,
                    song_name=song_name,
                    artist=artist,
                    genre=genre,
                    seed=seed or hash(song_name + artist),
                    tempo_map=tempo_map,
                )
                if lyric_lines:
                    # Parse the tick from each lyric line for sorting
                    for ll in lyric_lines:
                        stripped = ll.strip()
                        try:
                            tick_str = stripped.split("=")[0].strip()
                            tick_val = int(tick_str)
                        except (ValueError, IndexError):
                            tick_val = 0
                        event_entries.append((tick_val, ll))
                    logger.info("🎤 Added {} lyric events to chart", len(lyric_lines))
            except Exception as e:
                logger.warning("⚠️ Lyrics generation failed (continuing without): {}", e)

        # Sort all events by tick (stable sort keeps relative order for same tick)
        event_entries.sort(key=lambda e: e[0])
        for _, event_line in event_entries:
            lines.append(event_line)

        lines.append("}")
        lines.append("")

        # --- Determine effective onset data ---
        # If we have instrument-specific stem analysis, prefer its onset data
        # over the full-mix onset data passed in as arguments.
        effective_onsets = onset_times
        effective_strengths = onset_strengths
        effective_pitch_contour: Optional[List[float]] = None
        effective_drum_lanes: Optional[List[int]] = None

        if stem_analysis is not None and stem_analysis.onset_times:
            effective_onsets = stem_analysis.onset_times
            effective_strengths = stem_analysis.onset_strengths
            if stem_analysis.pitch_contour:
                effective_pitch_contour = stem_analysis.pitch_contour
            if stem_analysis.drum_lanes:
                effective_drum_lanes = stem_analysis.drum_lanes
            logger.info(
                "🎛️ Using {} stem data: {} onsets (full-mix had {})",
                instrument,
                len(effective_onsets),
                len(onset_times),
            )

        # --- Note sections for each difficulty ---
        for diff in difficulties:
            diff_lower = diff.lower()

            # Use instrument-aware profile if we have stem data,
            # otherwise fall back to the legacy guitar profiles
            if instrument != "full_mix":
                profile = get_difficulty_profile_for_instrument(instrument, diff_lower)
            elif diff_lower in DIFFICULTY_PROFILES:
                profile = DIFFICULTY_PROFILES[diff_lower]
            else:
                logger.warning("⚠️ Unknown difficulty '{}', skipping", diff)
                continue

            _generate_difficulty_section(
                lines=lines,
                profile=profile,
                difficulty=diff_lower,
                tempo=tempo,
                onset_times=effective_onsets,
                onset_strengths=effective_strengths,
                beat_times=beat_times,
                duration=duration,
                segments=used_segments,
                rng=rng,
                instrument=instrument,
                pitch_contour=effective_pitch_contour,
                drum_lanes=effective_drum_lanes,
                tempo_map=tempo_map,
                stem_analysis=stem_analysis,
            )

        # Write the file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8-sig") as f:
            content = "\n".join(lines)
            if not content.endswith("\n"):
                content += "\n"
            f.write(content)

        total_notes = len(onset_times)
        logger.info(
            "✅ Generated chart: {} ({} onsets, {:.1f} BPM, difficulties: {})",
            output_path,
            total_notes,
            tempo,
            ", ".join(difficulties),
        )
        return True

    except Exception as e:
        logger.error("❌ Error generating chart for '{}': {}", song_name, e)
        return False


def _generate_difficulty_section(
    lines: List[str],
    profile: Dict[str, Any],
    difficulty: str,
    tempo: float,
    onset_times: List[float],
    onset_strengths: List[float],
    beat_times: List[float],
    duration: float,
    segments: List[Dict[str, Any]],
    rng: random.Random,
    instrument: str = "guitar",
    pitch_contour: Optional[List[float]] = None,
    drum_lanes: Optional[List[int]] = None,
    tempo_map: Optional[List[Tuple[int, int]]] = None,
    stem_analysis: Optional["StemAnalysis"] = None,
) -> None:
    """Generate a single difficulty section and append it to *lines*.

    Parameters
    ----------
    instrument : str
        Target instrument (affects note selection logic).
    pitch_contour : list of float, optional
        Per-onset normalised pitch values (0.0–1.0) from stem analysis.
        Used for pitch-to-lane mapping on melodic instruments.
    drum_lanes : list of int, optional
        Per-onset drum lane assignments from drum stem analysis.
    tempo_map : list of (tick, milli_bpm), optional
        Piecewise tempo map for accurate seconds-to-ticks conversion.
    stem_analysis : StemAnalysis, optional
        Full stem analysis data including fundamental_freqs and
        is_open_note for semitone-based lane mapping.
    """
    section_name = profile["section_name"]
    note_skip = profile["note_skip"]
    min_gap = profile["min_note_gap_ticks"]

    lines.append(f"[{section_name}]")
    lines.append("{")

    # Select which onsets to use based on difficulty
    if note_skip > 1:
        selected_indices = list(range(0, len(onset_times), note_skip))
    else:
        selected_indices = list(range(len(onset_times)))

    if not selected_indices:
        lines.append("}")
        lines.append("")
        return

    # Convert to ticks and enforce minimum gap
    note_events: List[Tuple[int, float]] = []  # (tick, strength)
    prev_tick = -min_gap - 1
    for idx in selected_indices:
        t = onset_times[idx]
        strength = onset_strengths[idx] if idx < len(onset_strengths) else 0.5
        tick = _seconds_to_ticks(t, tempo, tempo_map=tempo_map)
        if tick - prev_tick >= min_gap:
            note_events.append((tick, strength))
            prev_tick = tick

    if not note_events:
        lines.append("}")
        lines.append("")
        return

    # Determine which section each note belongs to (for pattern variation)
    seg_times = [s["time"] for s in segments]

    def _section_for_time(t_seconds: float) -> int:
        for si in range(len(seg_times) - 1, -1, -1):
            if t_seconds >= seg_times[si]:
                return si
        return 0

    # Compute star power sections
    all_ticks = [ne[0] for ne in note_events]
    total_ticks = all_ticks[-1] if all_ticks else 0
    sp_sections = _compute_star_power_sections(all_ticks, total_ticks)

    def _in_star_power(tick: int) -> bool:
        for sp_start, sp_dur in sp_sections:
            if sp_start <= tick <= sp_start + sp_dur:
                return True
        return False

    # Generate notes
    prev_tick = -999
    prev_lanes: List[int] = []

    # Pre-compute pitch-derived lane assignments if available
    pitch_lanes: Optional[List[int]] = None
    if instrument == "drums" and drum_lanes is not None:
        # For drums, map the pre-analysed drum lane assignments to our
        # selected subset of onsets
        pitch_lanes = _map_drum_lanes_to_selected(
            drum_lanes, onset_times, note_events, profile
        )
    elif instrument in ("guitar", "bass", "vocals"):
        # For melodic instruments, use semitone-based lane mapping
        # when pYIN data is available (fund_freqs + is_open_note).
        # This gives musically correct fret→lane mapping including
        # open notes (lane 7) for metal charting.
        if (
            hasattr(stem_analysis, "fundamental_freqs")
            and stem_analysis is not None
            and stem_analysis.fundamental_freqs
        ):
            from src.services.stem_separator import _semitone_contour_to_lanes

            # Map the full onset fund_freqs to our selected note subset
            all_freqs = stem_analysis.fundamental_freqs
            all_open = stem_analysis.is_open_note
            n_orig = len(all_freqs)
            n_sel = len(note_events)
            sel_freqs: List[float] = []
            sel_open: List[bool] = []
            for si in range(n_sel):
                src_idx = int(si * n_orig / n_sel) if n_sel > 0 else 0
                src_idx = min(src_idx, n_orig - 1)
                sel_freqs.append(all_freqs[src_idx])
                sel_open.append(all_open[src_idx] if src_idx < len(all_open) else False)
            pitch_lanes = _semitone_contour_to_lanes(sel_freqs, sel_open, smoothing=2)
        elif pitch_contour is not None:
            # Legacy fallback: use normalised pitch contour
            selected_pitches = _map_pitch_to_selected(
                pitch_contour, onset_times, note_events
            )
            if selected_pitches:
                is_open = None
                if hasattr(stem_analysis, "is_open_note") and stem_analysis is not None:
                    # Map open note flags to selected subset
                    all_open = stem_analysis.is_open_note
                    n_orig = len(all_open)
                    n_sel = len(note_events)
                    is_open = []
                    if n_orig > 0:
                        for si in range(n_sel):
                            src_idx = int(si * n_orig / n_sel) if n_sel > 0 else 0
                            src_idx = min(src_idx, n_orig - 1)
                            is_open.append(
                                all_open[src_idx] if src_idx < len(all_open) else False
                            )
                    else:
                        # No open-note data available — default all to non-open
                        is_open = [False] * n_sel
                pitch_lanes = pitch_contour_to_lanes(
                    selected_pitches,
                    max_lane=profile["max_lane"],
                    smoothing=3,
                    is_open_note=is_open,
                )

    for i, (tick, strength) in enumerate(note_events):
        t_seconds = tick / (RESOLUTION * tempo / 60.0) if tempo > 0 else 0
        section_idx = _section_for_time(t_seconds)

        # Determine the pitch-derived lane for this note (if available)
        p_lane: Optional[int] = None
        if pitch_lanes is not None and i < len(pitch_lanes):
            p_lane = pitch_lanes[i]

        if instrument == "drums" and p_lane is not None:
            # Drums: use the detected drum lane directly
            lanes = [p_lane]
        else:
            lanes = _select_note(
                i,
                strength,
                difficulty,
                section_idx,
                rng,
                profile,
                pitch_lane=p_lane,
            )

        # HOPO marker (not used for drums)
        is_hopo = False
        if instrument != "drums":
            is_hopo = _should_hopo(tick, prev_tick, prev_lanes, lanes, profile, rng)

        # Sustain (drums never sustain)
        if instrument == "drums":
            sustain = 0
        else:
            next_tick = note_events[i + 1][0] if i + 1 < len(note_events) else None
            sustain = _compute_sustain(tick, next_tick, strength, profile, rng)

        # Write note(s)
        for lane in lanes:
            lines.append(f"  {tick} = N {lane} {sustain}")

        # HOPO / forced strum marker
        if is_hopo:
            lines.append(f"  {tick} = N 5 0")

        # Star power marker (only at the start of an SP section)
        if _in_star_power(tick):
            # Only emit the S marker once per SP section
            for sp_start, sp_dur in sp_sections:
                if tick == sp_start:
                    lines.append(f"  {tick} = S 2 {sp_dur}")
                    break

        prev_tick = tick
        prev_lanes = lanes

    lines.append("}")
    lines.append("")


# ---------------------------------------------------------------------------
# Song file processing pipeline
# ---------------------------------------------------------------------------
def _map_pitch_to_selected(
    pitch_contour: List[float],
    all_onset_times: List[float],
    note_events: List[Tuple[int, float]],
) -> List[float]:
    """Map full pitch contour to the subset of onsets used in note_events.

    The pitch_contour has one value per onset in the stem analysis, but
    note_events may be a sparse subset (due to note_skip / min_gap
    filtering).  We use proportional index mapping: the i-th selected
    note maps to the proportionally corresponding position in the
    original pitch contour.
    """
    if not pitch_contour or not note_events:
        return []

    n_original = len(pitch_contour)
    n_selected = len(note_events)
    result: List[float] = []

    for i in range(n_selected):
        # Proportional index into the original contour
        src_idx = int(i * n_original / n_selected) if n_selected > 0 else 0
        src_idx = min(src_idx, n_original - 1)
        result.append(pitch_contour[src_idx])

    return result


def _map_drum_lanes_to_selected(
    drum_lanes: List[int],
    all_onset_times: List[float],
    note_events: List[Tuple[int, float]],
    profile: Dict[str, Any],
) -> List[int]:
    """Map drum lane assignments to the subset of onsets in note_events.

    Uses proportional index mapping since note_events is a sparse subset
    of the original drum onsets.
    """
    if not drum_lanes:
        return [1] * len(note_events)  # default to snare

    max_lane = profile.get("max_lane", 4)
    result = []
    n_original = len(drum_lanes)
    n_selected = len(note_events)

    for i in range(n_selected):
        # Proportional mapping
        src_idx = int(i * n_original / n_selected) if n_selected > 0 else 0
        src_idx = min(src_idx, n_original - 1)
        lane = min(drum_lanes[src_idx], max_lane)
        result.append(lane)

    return result


def process_song_file(
    file_path: str,
    song_name: Optional[str] = None,
    artist: Optional[str] = None,
    difficulty: str = "expert",
    enable_lyrics: bool = True,
    enable_album_art: bool = True,
    charter: str = "nuniesmith",
    album: str = "Generated",
    year: str = "",
    genre: str = "Generated",
    cover_art_path: Optional[str] = None,
    instrument: str = "guitar",
    auto_validate: bool = True,
) -> Dict[str, Any]:
    """
    Full pipeline: analyse audio -> generate chart -> stage locally.

    1. Analyses the audio file for tempo, beats, and onsets
    2. Optionally separates stems for instrument-specific charting
    3. Generates a notes.chart file in a temp staging directory
       (all four difficulty levels are always generated)
    4. Creates a song.ini file
    5. Copies the source audio into the staging folder
    6. Optionally generates procedural lyrics (timed to beats)
    7. Optionally generates procedural album art (album.png)
       If ``cover_art_path`` is provided, that image is used instead
       of generating one procedurally.

    The caller is responsible for uploading the staging directory to
    Nextcloud and registering the song in the database (this is done
    in the async API layer).

    Parameters
    ----------
    file_path : str
        Path to the source audio file.
    song_name : str, optional
        Song title (defaults to the filename stem).
    artist : str, optional
        Artist name (defaults to "Unknown Artist").
    difficulty : str
        Primary difficulty hint (all four levels are always generated).
    enable_lyrics : bool
        If True, generate procedural lyrics in the [Events] section.
    enable_album_art : bool
        If True, generate a procedural album.png cover image.
    charter : str
        Name to use in the Charter field (default "nuniesmith").
    album : str
        Album name for chart metadata (default "Generated").
    year : str
        Year string for chart metadata.
    genre : str
        Genre string for chart metadata (default "Generated").
    cover_art_path : str, optional
        Path to an existing cover art image.  If provided and valid,
        this image is copied as album.png instead of generating one.
    instrument : str
        Target instrument for note charting: ``guitar``, ``bass``,
        ``drums``, ``vocals``, or ``full_mix`` (default ``guitar``).

    Returns a result dict with staging paths and metadata,
    or an error dict on failure.
    """
    source_path = Path(file_path)
    if not source_path.exists():
        return {"error": f"File not found: {file_path}"}

    # Default song name from filename
    if not song_name:
        song_name = source_path.stem
    if not artist:
        artist = "Unknown Artist"

    try:
        # Step 1: Analyse audio (full mix for tempo/beats/segments)
        logger.info(
            "🎵 Processing song file: {} (instrument={})", file_path, instrument
        )
        analysis = analyze_audio(file_path)

        tempo = analysis["tempo"]
        beat_times = analysis["beat_times"]
        onset_times = analysis["onset_times"]
        onset_strengths = analysis.get("onset_strengths", [])
        duration = analysis["duration"]
        segments = analysis.get("segments", [])

        # Step 1b: Instrument-specific stem analysis
        stem_analysis: Optional[StemAnalysis] = None
        if instrument and instrument != "full_mix":
            try:
                logger.info("🎛️ Separating {} stem...", instrument)
                stem_analysis, _ = analyze_instrument(
                    file_path,
                    instrument=instrument,
                    sensitivity=0.5,
                    beat_times=beat_times,
                )
                logger.info(
                    "✅ {} stem analysis: {} onsets (full-mix had {})",
                    instrument,
                    len(stem_analysis.onset_times),
                    len(onset_times),
                )
            except Exception as e:
                logger.warning(
                    "⚠️ Stem separation failed for '{}' (falling back to full mix): {}",
                    instrument,
                    e,
                )

        # Step 2: Create a temporary staging directory
        unique_id = uuid.uuid4().hex[:8]
        safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in song_name)
        safe_name = safe_name.strip()
        staging_dir = TEMP_DIR / f"{safe_name}_{unique_id}"
        staging_dir.mkdir(parents=True, exist_ok=True)

        # Step 3: Copy (and optionally convert) the audio file
        audio_ext = source_path.suffix.lower()

        if audio_ext in CLONE_HERO_SAFE_EXTENSIONS:
            # Already a safe format — just copy
            audio_filename = f"song{audio_ext}"
            audio_dest = staging_dir / audio_filename
            try:
                shutil.copy2(file_path, str(audio_dest))
                logger.info("🎶 Copied audio to {}", audio_dest)
            except Exception as e:
                logger.warning("⚠️ Could not copy audio to staging folder: {}", e)
        else:
            # Unsupported / unreliable format (FLAC, WAV, etc.)
            # Convert to OGG Vorbis for maximum Clone Hero compatibility
            logger.info(
                "🔄 Audio format '{}' is not reliably supported by Clone Hero — converting to OGG",
                audio_ext,
            )
            converted = convert_audio_to_ogg(source_path, staging_dir)
            if converted:
                audio_filename = converted.name  # "song.ogg"
            else:
                # Fallback: copy the original file as-is and hope for the best
                logger.warning(
                    "⚠️ Conversion failed — falling back to raw {} copy",
                    audio_ext,
                )
                audio_filename = f"song{audio_ext}"
                audio_dest = staging_dir / audio_filename
                try:
                    shutil.copy2(file_path, str(audio_dest))
                except Exception as e:
                    logger.warning("⚠️ Could not copy audio to staging folder: {}", e)

        # Step 4: Generate the notes.chart with ALL difficulties
        chart_path = staging_dir / "notes.chart"
        all_difficulties = ["easy", "medium", "hard", "expert"]

        chart_ok = generate_notes_chart(
            song_name=song_name,
            artist=artist,
            album=album,
            year=year,
            genre=genre,
            tempo=tempo,
            beat_times=beat_times,
            onset_times=onset_times,
            onset_strengths=onset_strengths,
            duration=duration,
            output_path=chart_path,
            audio_filename=audio_filename,
            segments=segments,
            difficulties=all_difficulties,
            enable_lyrics=enable_lyrics,
            instrument=instrument,
            stem_analysis=stem_analysis,
        )

        if not chart_ok:
            return {"error": "Failed to generate notes.chart"}

        # Step 4b: Validate and auto-fix the generated chart
        validation_result = None
        if auto_validate:
            try:
                from src.services.chart_validator import validate_and_fix_chart

                validation_result = validate_and_fix_chart(chart_path, staging_dir)
                if validation_result.fixes_applied:
                    logger.info(
                        "🔧 Applied {} fix(es) to generated chart for '{}'",
                        len(validation_result.fixes_applied),
                        song_name,
                    )
                    for fix_msg in validation_result.fixes_applied:
                        logger.debug("  ✅ {}", fix_msg)

                if validation_result.is_valid:
                    logger.info("✅ Generated chart passed validation")
                else:
                    logger.warning(
                        "⚠️ Generated chart has {} critical issue(s) after fixes",
                        validation_result.critical_count,
                    )
            except Exception as e:
                logger.warning("⚠️ Chart validation failed (continuing without): {}", e)

        # Step 5: Album art — use provided cover or generate procedurally
        has_album_art = False

        # 5a: If caller provided external cover art, use it directly
        if cover_art_path and Path(cover_art_path).exists():
            try:
                album_art_dest = staging_dir / "album.png"
                ext = Path(cover_art_path).suffix.lower()
                if ext in (".jpg", ".jpeg"):
                    # Convert JPEG to PNG for Clone Hero compatibility
                    try:
                        from PIL import Image

                        img = Image.open(cover_art_path)
                        img.save(str(album_art_dest), "PNG")
                        has_album_art = True
                        logger.info("🎨 Saved looked-up cover art as album.png")
                    except ImportError:
                        shutil.copy2(cover_art_path, str(album_art_dest))
                        has_album_art = True
                        logger.info(
                            "🎨 Copied looked-up cover art (no PIL for convert)"
                        )
                else:
                    shutil.copy2(cover_art_path, str(album_art_dest))
                    has_album_art = True
                    logger.info("🎨 Copied looked-up cover art as album.png")
            except Exception as e:
                logger.warning("⚠️ Failed to use provided cover art: {}", e)

        # 5b: Fall back to procedural generation if no external art
        if not has_album_art and enable_album_art and album_art_available():
            try:
                # Compute average RMS energy for palette generation
                rms_values = analysis.get("rms_values", [])
                avg_energy = sum(rms_values) / len(rms_values) if rms_values else 0.5

                album_art_path = staging_dir / "album.png"
                has_album_art = generate_album_art(
                    output_path=album_art_path,
                    song_name=song_name,
                    artist=artist,
                    tempo=tempo,
                    duration=duration,
                    onset_strengths=onset_strengths,
                    beat_times=beat_times,
                    genre="Generated",
                    energy=avg_energy,
                    seed=hash(song_name + artist),
                )
                if has_album_art:
                    logger.info("🎨 Generated album art: {}", album_art_path)
                else:
                    logger.warning("⚠️ Album art generation returned False")
            except Exception as e:
                logger.warning(
                    "⚠️ Album art generation failed (continuing without): {}", e
                )
        elif enable_album_art:
            logger.info("ℹ️ Album art generation skipped (Pillow not installed)")

        # Step 6: Create song.ini
        song_length_ms = str(int(duration * 1000))
        # Pick a sensible preview start (about 1/4 into the song)
        preview_start = str(int(duration * 250))

        song_data = {
            "title": song_name,
            "artist": artist,
            "album": album,
            "metadata": {
                "charter": charter,
                "song_length": song_length_ms,
                "genre": genre,
                "preview_start_time": preview_start,
                "diff_guitar": "3",
                "diff_bass": "-1",
                "diff_rhythm": "-1",
                "diff_drums": "-1",
                "diff_keys": "-1",
                "diff_guitarghl": "-1",
                "diff_bassghl": "-1",
                "delay": "0",
                "loading_phrase": f"Auto-generated chart at {tempo:.0f} BPM",
            },
        }
        ini_path = staging_dir / "song.ini"
        write_song_ini(ini_path, song_data)

        stem_note_count = (
            len(stem_analysis.onset_times) if stem_analysis else len(onset_times)
        )
        result = {
            "message": "Song generated successfully",
            "song_name": song_name,
            "artist": artist,
            "tempo": tempo,
            "duration": duration,
            "total_beats": len(beat_times),
            "total_notes": stem_note_count,
            "difficulty": difficulty,
            "difficulties_generated": all_difficulties,
            "has_lyrics": enable_lyrics,
            "has_album_art": has_album_art,
            "staging_dir": str(staging_dir),
            "unique_id": unique_id,
            "instrument": instrument,
        }

        # Include validation results if available
        if validation_result is not None:
            result["validation"] = {
                "valid": validation_result.is_valid,
                "critical": validation_result.critical_count,
                "warnings": validation_result.warning_count,
                "fixes_applied": validation_result.fixes_applied,
                "issues": [
                    iss.to_dict()
                    for iss in validation_result.issues
                    if iss.severity != "info"
                ],
            }

        return result

    except Exception as e:
        logger.exception("❌ Error processing song '{}': {}", song_name, e)
        return {"error": str(e)}


def generate_all_difficulties(file_path: str, **kwargs) -> Dict[str, Any]:
    """
    Generate charts for all four difficulty levels from a single audio file.

    This is now the default behaviour of :func:`process_song_file`, so this
    function simply delegates to it for backwards compatibility.
    """
    return process_song_file(file_path, **kwargs)


async def process_and_upload_song(
    file_path: str,
    song_name: Optional[str] = None,
    artist: Optional[str] = None,
    difficulty: str = "expert",
    enable_lyrics: bool = True,
    enable_album_art: bool = True,
    charter: str = "nuniesmith",
    album: str = "Generated",
    year: str = "",
    genre: str = "Generated",
    cover_art_path: Optional[str] = None,
    instrument: str = "guitar",
) -> Dict[str, Any]:
    """
    High-level async pipeline: generate chart -> upload to Nextcloud -> register in DB.

    Wraps :func:`process_song_file` (sync, CPU-bound) and then handles the
    async Nextcloud upload and database registration.

    Parameters
    ----------
    file_path : str
        Path to the source audio file.
    song_name : str, optional
        Song title (defaults to the filename stem).
    artist : str, optional
        Artist name (defaults to "Unknown Artist").
    difficulty : str
        Primary difficulty hint (all four levels are always generated).
    enable_lyrics : bool
        If True, generate procedural lyrics in the [Events] section.
    enable_album_art : bool
        If True, generate a procedural album.png cover image.
    charter : str
        Name for the Charter field (default "nuniesmith").
    album : str
        Album name for chart metadata.
    year : str
        Year string for chart metadata.
    genre : str
        Genre string for chart metadata.
    cover_art_path : str, optional
        Path to an existing cover art image to use instead of generating one.
    instrument : str
        Target instrument for charting (default ``guitar``).
    """
    import asyncio

    from src.database import upsert_song
    from src.webdav import is_configured, upload_song_folder

    if not is_configured():
        return {
            "error": "Nextcloud WebDAV is not configured. "
            "Set NEXTCLOUD_URL, NEXTCLOUD_USERNAME, and NEXTCLOUD_PASSWORD."
        }

    # Run the CPU-bound audio analysis + chart generation in a thread
    # (includes automatic chart validation and fixes)
    result = await asyncio.to_thread(
        process_song_file,
        file_path,
        song_name=song_name,
        artist=artist,
        difficulty=difficulty,
        enable_lyrics=enable_lyrics,
        enable_album_art=enable_album_art,
        charter=charter,
        album=album,
        year=year,
        genre=genre,
        cover_art_path=cover_art_path,
        instrument=instrument,
        auto_validate=True,
    )

    if "error" in result:
        return result

    staging_dir = result.get("staging_dir")
    if not staging_dir:
        return {"error": "No staging directory produced"}

    try:
        # Upload to Nextcloud
        remote_path = await upload_song_folder(
            local_dir=staging_dir,
            artist=result.get("artist", "Unknown Artist"),
            title=result.get("song_name", "Unknown"),
            suffix=result.get("unique_id", uuid.uuid4().hex[:8]),
        )

        if not remote_path:
            return {"error": "Failed to upload generated song to Nextcloud"}

        # Register in database
        metadata = {
            "charter": charter,
            "song_length": str(int(result.get("duration", 0) * 1000)),
            "genre": genre,
            "diff_guitar": "3",
        }
        if year:
            metadata["year"] = year
        metadata_json = json.dumps(metadata)
        song_id = await upsert_song(
            title=result.get("song_name", "Unknown"),
            artist=result.get("artist", "Unknown Artist"),
            album=album or "Generated",
            remote_path=remote_path,
            metadata=metadata_json,
        )

        result["id"] = song_id
        result["remote_path"] = remote_path
        result["message"] = "Song generated and uploaded to Nextcloud"

        return result

    finally:
        # Clean up the staging directory
        try:
            shutil.rmtree(staging_dir, ignore_errors=True)
        except Exception:
            pass
