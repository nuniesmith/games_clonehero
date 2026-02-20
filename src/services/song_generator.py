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
NOTE_LANES = 5  # Green(0), Red(1), Yellow(2), Blue(3), Orange(4)
RESOLUTION = 192  # Standard ticks per quarter note

# Difficulty settings: controls note density, max lane, chord probability, etc.
DIFFICULTY_PROFILES = {
    "easy": {
        "section_name": "EasySingle",
        "note_skip": 4,  # use every Nth onset
        "max_lane": 2,  # only green/red/yellow
        "chord_chance": 0.0,  # no chords
        "hopo_chance": 0.0,  # no HOPOs
        "sustain_chance": 0.05,
        "min_note_gap_ticks": 192,  # at least 1 beat apart
    },
    "medium": {
        "section_name": "MediumSingle",
        "note_skip": 3,
        "max_lane": 3,  # up to blue
        "chord_chance": 0.05,
        "hopo_chance": 0.05,
        "sustain_chance": 0.10,
        "min_note_gap_ticks": 96,  # half beat
    },
    "hard": {
        "section_name": "HardSingle",
        "note_skip": 2,
        "max_lane": 4,  # all 5 lanes
        "chord_chance": 0.10,
        "hopo_chance": 0.15,
        "sustain_chance": 0.15,
        "min_note_gap_ticks": 48,  # quarter beat
    },
    "expert": {
        "section_name": "ExpertSingle",
        "note_skip": 1,  # use every onset
        "max_lane": 4,  # all 5 lanes
        "chord_chance": 0.15,
        "hopo_chance": 0.25,
        "sustain_chance": 0.20,
        "min_note_gap_ticks": 24,  # eighth beat
    },
}


def _seconds_to_ticks(time_s: float, tempo: float, resolution: int = RESOLUTION) -> int:
    """Convert a time in seconds to chart ticks based on tempo and resolution."""
    beats = time_s * (tempo / 60.0)
    return int(round(beats * resolution))


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
# Each pattern is a list of lane indices (0-4).
PATTERNS_EASY = [
    [0, 0, 1, 1],
    [0, 1, 0, 1],
    [1, 1, 0, 0],
    [0, 1, 2, 1],
    [2, 1, 0, 1],
    [0, 0, 0, 1],
]

PATTERNS_MEDIUM = [
    [0, 1, 2, 1, 0],
    [0, 2, 1, 3, 1],
    [1, 0, 2, 0, 1],
    [3, 2, 1, 0, 1],
    [0, 1, 2, 3, 2],
    [2, 2, 1, 0, 0],
]

PATTERNS_HARD = [
    [0, 1, 2, 3, 2, 1],
    [0, 2, 4, 2, 0, 1],
    [1, 3, 2, 4, 3, 1],
    [4, 3, 2, 1, 0, 1],
    [0, 1, 3, 2, 4, 3],
    [2, 0, 3, 1, 4, 2],
]

PATTERNS_EXPERT = [
    [0, 1, 2, 3, 4, 3, 2, 1],
    [0, 2, 4, 3, 1, 0, 2, 4],
    [4, 3, 2, 1, 0, 1, 2, 3],
    [0, 1, 3, 4, 2, 0, 3, 1],
    [1, 3, 0, 4, 2, 1, 3, 0],
    [2, 4, 1, 3, 0, 2, 4, 1],
    [0, 0, 2, 2, 4, 4, 3, 1],
    [3, 1, 4, 2, 0, 3, 1, 4],
]

DIFFICULTY_PATTERNS = {
    "easy": PATTERNS_EASY,
    "medium": PATTERNS_MEDIUM,
    "hard": PATTERNS_HARD,
    "expert": PATTERNS_EXPERT,
}

# Chord shapes (pairs of simultaneous notes) — for hard/expert
CHORD_SHAPES = [
    (0, 1),  # GR
    (1, 2),  # RY
    (2, 3),  # YB
    (3, 4),  # BO
    (0, 2),  # GY
    (1, 3),  # RB
    (0, 1, 2),  # GRY (triple for expert)
]


def _select_note(
    index: int,
    onset_strength: float,
    difficulty: str,
    section_index: int,
    rng: random.Random,
    profile: Dict[str, Any],
) -> List[int]:
    """
    Select which lane(s) a note should be on.

    Returns a list of lane indices. A single-element list is a single note;
    multiple elements means a chord.

    Uses the onset strength, position in the song, and difficulty profile
    to decide note placement.
    """
    max_lane = profile["max_lane"]
    chord_chance = profile["chord_chance"]

    # Pick pattern based on section to create variation
    patterns = DIFFICULTY_PATTERNS.get(difficulty, PATTERNS_EXPERT)
    pattern = patterns[(section_index + index // 16) % len(patterns)]
    base_lane = pattern[index % len(pattern)]

    # Clamp to max lane for this difficulty
    base_lane = min(base_lane, max_lane)

    # Possibly make a chord on strong onsets (hard/expert only)
    if chord_chance > 0 and onset_strength > 0.7 and rng.random() < chord_chance:
        shapes = [s for s in CHORD_SHAPES if all(l <= max_lane for l in s)]
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
    """
    if profile["hopo_chance"] <= 0:
        return False
    if len(lanes) > 1:
        return False  # no HOPO on chords
    if lanes == prev_lanes:
        return False  # same lane, no HOPO
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
) -> bool:
    """
    Generate a Clone Hero compatible notes.chart file.

    The chart includes:
        - [Song] section with full metadata and MusicStream
        - [SyncTrack] with stable tempo markers (milli-BPM)
        - [Events] with section markers and optional lyrics
        - Note sections for each requested difficulty
        - HOPO markers, chords, sustains, and star power

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
        lines.append("[Events]")
        lines.append("{")

        used_segments = (
            segments
            if segments and len(segments) >= 2
            else _fallback_segments(duration)
        )
        for seg in used_segments:
            tick = _seconds_to_ticks(seg["time"], tempo)
            label = seg["label"]
            lines.append(f'  {tick} = E "section {label}"')

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
                    genre=genre,
                    seed=seed or hash(song_name + artist),
                )
                if lyric_lines:
                    lines.extend(lyric_lines)
                    logger.info("🎤 Added {} lyric events to chart", len(lyric_lines))
            except Exception as e:
                logger.warning("⚠️ Lyrics generation failed (continuing without): {}", e)

        lines.append("}")
        lines.append("")

        # --- Note sections for each difficulty ---
        for diff in difficulties:
            diff_lower = diff.lower()
            if diff_lower not in DIFFICULTY_PROFILES:
                logger.warning("⚠️ Unknown difficulty '{}', skipping", diff)
                continue
            profile = DIFFICULTY_PROFILES[diff_lower]
            _generate_difficulty_section(
                lines=lines,
                profile=profile,
                difficulty=diff_lower,
                tempo=tempo,
                onset_times=onset_times,
                onset_strengths=onset_strengths,
                beat_times=beat_times,
                duration=duration,
                segments=used_segments,
                rng=rng,
            )

        # Write the file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8-sig") as f:
            f.write("\n".join(lines))

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
) -> None:
    """Generate a single difficulty section and append it to *lines*."""
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
        tick = _seconds_to_ticks(t, tempo)
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

    for i, (tick, strength) in enumerate(note_events):
        t_seconds = tick / (RESOLUTION * tempo / 60.0) if tempo > 0 else 0
        section_idx = _section_for_time(t_seconds)

        lanes = _select_note(i, strength, difficulty, section_idx, rng, profile)

        # HOPO marker
        is_hopo = _should_hopo(tick, prev_tick, prev_lanes, lanes, profile, rng)

        # Sustain
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
) -> Dict[str, Any]:
    """
    Full pipeline: analyse audio -> generate chart -> stage locally.

    1. Analyses the audio file for tempo, beats, and onsets
    2. Generates a notes.chart file in a temp staging directory
       (all four difficulty levels are always generated)
    3. Creates a song.ini file
    4. Copies the source audio into the staging folder
    5. Optionally generates procedural lyrics (timed to beats)
    6. Optionally generates procedural album art (album.png)
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
        # Step 1: Analyse audio
        logger.info("🎵 Processing song file: {}", file_path)
        analysis = analyze_audio(file_path)

        tempo = analysis["tempo"]
        beat_times = analysis["beat_times"]
        onset_times = analysis["onset_times"]
        onset_strengths = analysis.get("onset_strengths", [])
        duration = analysis["duration"]
        segments = analysis.get("segments", [])

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
        )

        if not chart_ok:
            return {"error": "Failed to generate notes.chart"}

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

        return {
            "message": "Song generated successfully",
            "song_name": song_name,
            "artist": artist,
            "tempo": tempo,
            "duration": duration,
            "total_beats": len(beat_times),
            "total_notes": len(onset_times),
            "difficulty": difficulty,
            "difficulties_generated": all_difficulties,
            "has_lyrics": enable_lyrics,
            "has_album_art": has_album_art,
            "staging_dir": str(staging_dir),
            "unique_id": unique_id,
        }

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
