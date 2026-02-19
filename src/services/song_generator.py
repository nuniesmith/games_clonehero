"""
Clone Hero Content Manager - Song Generator Service

Analyses audio files using librosa to detect tempo, beats, and rhythm patterns,
then generates Clone Hero compatible notes.chart files.

This is the core music-analysis engine that powers the "Generate Songs" feature.
"""

import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import librosa
import numpy as np
from loguru import logger

from src.config import CONTENT_DIR
from src.database import insert_song_sync
from src.services.content_manager import get_content_directory, write_song_ini

# Output directory for generated charts
GENERATOR_DIR = CONTENT_DIR / "generator"


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
        - chroma (optional): chroma feature summary
    """
    try:
        logger.info(f"🎵 Analysing audio: {file_path}")

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
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr).tolist()

        logger.info(
            f"✅ Audio analysis complete: tempo={tempo:.1f} BPM, "
            f"{len(beat_times)} beats, {len(onset_times)} onsets, "
            f"duration={duration:.1f}s"
        )

        return {
            "tempo": round(tempo, 2),
            "beat_times": beat_times,
            "onset_times": onset_times,
            "duration": round(duration, 2),
        }

    except Exception as e:
        logger.error(f"❌ Error analysing audio {file_path}: {e}")
        raise


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
        logger.warning(f"⚠️ Extended analysis failed (continuing with basic): {e}")

    return basic


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------
NOTE_LANES = 5  # Green, Red, Yellow, Blue, Orange (standard Clone Hero)
RESOLUTION = 192  # Standard ticks per quarter note


def _seconds_to_ticks(time_s: float, tempo: float, resolution: int = RESOLUTION) -> int:
    """Convert a time in seconds to chart ticks based on tempo and resolution."""
    beats = time_s * (tempo / 60.0)
    return int(round(beats * resolution))


def _assign_note_lane(index: int, onset_time: float, total_onsets: int) -> int:
    """
    Assign a note to a lane (0-4) based on position and pattern.

    Uses a combination of index position and timing to create varied
    but somewhat predictable patterns. This is a basic heuristic;
    more sophisticated approaches could use spectral analysis.
    """
    # Create patterns that feel natural to play
    patterns = [
        [0, 1, 2, 1, 0],  # Wave pattern
        [0, 2, 4, 2, 0],  # Wide wave
        [0, 1, 2, 3, 4],  # Ascending
        [4, 3, 2, 1, 0],  # Descending
        [0, 0, 2, 2, 4],  # Pairs
        [1, 3, 1, 3, 2],  # Alternating
        [0, 2, 1, 3, 2],  # Mixed
        [2, 2, 2, 2, 2],  # Centre focus
    ]

    # Select pattern based on position in the song
    section = int((index / max(total_onsets, 1)) * len(patterns))
    section = min(section, len(patterns) - 1)
    pattern = patterns[section]

    return pattern[index % len(pattern)]


def generate_notes_chart(
    song_name: str,
    artist: str,
    tempo: float,
    beat_times: List[float],
    onset_times: List[float],
    duration: float,
    output_path: Path,
    difficulty: str = "expert",
) -> bool:
    """
    Generate a Clone Hero compatible notes.chart file.

    The chart includes:
        - [Song] section with metadata
        - [SyncTrack] with tempo markers
        - [Events] section markers
        - Note sections for the selected difficulty

    Returns True on success, False on failure.
    """
    try:
        difficulty_map = {
            "easy": "EasySingle",
            "medium": "MediumSingle",
            "hard": "HardSingle",
            "expert": "ExpertSingle",
        }

        section_name = difficulty_map.get(difficulty.lower(), "ExpertSingle")

        lines: List[str] = []

        # --- [Song] section ---
        lines.append("[Song]")
        lines.append("{")
        lines.append(f'  Name = "{song_name}"')
        lines.append(f'  Artist = "{artist}"')
        lines.append('  Charter = "Clone Hero Manager"')
        lines.append(f"  Resolution = {RESOLUTION}")
        lines.append("  Player2 = bass")
        lines.append("  Difficulty = 0")
        lines.append("  PreviewStart = 0")
        lines.append('  MediaType = "cd"')
        lines.append("}")
        lines.append("")

        # --- [SyncTrack] section ---
        lines.append("[SyncTrack]")
        lines.append("{")

        # Initial tempo and time signature
        tempo_us = int(round(60_000_000 / tempo))  # microseconds per beat
        lines.append("  0 = TS 4")  # 4/4 time signature
        lines.append(f"  0 = B {tempo_us}")  # Tempo in microseconds per beat

        # Add tempo changes if there are significant beat timing variations
        if len(beat_times) > 4:
            prev_tempo = tempo
            for i in range(2, len(beat_times) - 1):
                interval = beat_times[i] - beat_times[i - 1]
                if interval > 0:
                    local_tempo = 60.0 / interval
                    # Only add a tempo change if it differs significantly (> 5%)
                    if abs(local_tempo - prev_tempo) / prev_tempo > 0.05:
                        tick = _seconds_to_ticks(beat_times[i], tempo)
                        local_tempo_us = int(round(60_000_000 / local_tempo))
                        lines.append(f"  {tick} = B {local_tempo_us}")
                        prev_tempo = local_tempo

        lines.append("}")
        lines.append("")

        # --- [Events] section ---
        lines.append("[Events]")
        lines.append("{")

        # Add section markers at regular intervals
        if beat_times:
            section_interval = max(1, len(beat_times) // 8)
            section_names = [
                "Intro",
                "Verse 1",
                "Pre-Chorus",
                "Chorus",
                "Verse 2",
                "Bridge",
                "Solo",
                "Outro",
            ]
            for i, s_name in enumerate(section_names):
                beat_idx = i * section_interval
                if beat_idx < len(beat_times):
                    tick = _seconds_to_ticks(beat_times[beat_idx], tempo)
                    lines.append(f'  {tick} = E "section {s_name}"')

        lines.append("}")
        lines.append("")

        # --- Note section ---
        lines.append(f"[{section_name}]")
        lines.append("{")

        # Use onset times for note placement
        notes_to_use = onset_times if onset_times else beat_times
        total = len(notes_to_use)

        for i, t in enumerate(notes_to_use):
            tick = _seconds_to_ticks(t, tempo)
            lane = _assign_note_lane(i, t, total)
            # Format: tick = N lane duration
            # Duration 0 means a normal tap note
            lines.append(f"  {tick} = N {lane} 0")

        lines.append("}")
        lines.append("")

        # Write the file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(
            f"✅ Generated chart: {output_path} "
            f"({total} notes, {tempo:.1f} BPM, {difficulty})"
        )
        return True

    except Exception as e:
        logger.error(f"❌ Error generating chart for '{song_name}': {e}")
        return False


# ---------------------------------------------------------------------------
# Full processing pipeline
# ---------------------------------------------------------------------------
def process_song_file(
    file_path: str,
    song_name: Optional[str] = None,
    artist: Optional[str] = None,
    difficulty: str = "expert",
) -> Dict[str, Any]:
    """
    Full pipeline: analyse audio -> generate chart -> create song folder.

    1. Analyses the audio file for tempo, beats, and onsets
    2. Generates a notes.chart file
    3. Creates a song.ini file
    4. Optionally copies the source audio into the song folder
    5. Registers the song in the database

    Returns a result dict with generated file paths and metadata,
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
        logger.info(f"🎵 Processing song file: {file_path}")
        analysis = analyze_audio(file_path)

        tempo = analysis["tempo"]
        beat_times = analysis["beat_times"]
        onset_times = analysis["onset_times"]
        duration = analysis["duration"]

        # Step 2: Create output directory
        unique_id = uuid.uuid4().hex[:8]
        safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in song_name)
        safe_name = safe_name.strip()
        output_dir = get_content_directory("generator") / f"{safe_name}_{unique_id}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 3: Generate the notes.chart
        chart_path = output_dir / "notes.chart"
        chart_ok = generate_notes_chart(
            song_name=song_name,
            artist=artist,
            tempo=tempo,
            beat_times=beat_times,
            onset_times=onset_times,
            duration=duration,
            output_path=chart_path,
            difficulty=difficulty,
        )

        if not chart_ok:
            return {"error": "Failed to generate notes.chart"}

        # Step 4: Create song.ini
        song_data = {
            "title": song_name,
            "artist": artist,
            "album": "Generated",
            "metadata": {
                "charter": "Clone Hero Manager (AI)",
                "song_length": str(int(duration * 1000)),
                "genre": "Generated",
            },
        }
        ini_path = output_dir / "song.ini"
        write_song_ini(ini_path, song_data)

        # Step 5: Copy the audio file into the song folder
        audio_dest = output_dir / f"song{source_path.suffix}"
        try:
            shutil.copy2(file_path, str(audio_dest))
            logger.info(f"🎶 Copied audio to {audio_dest}")
        except Exception as e:
            logger.warning(f"⚠️ Could not copy audio to song folder: {e}")

        # Step 6: Register in database
        metadata_json = json.dumps(song_data.get("metadata", {}))
        song_id = insert_song_sync(
            title=song_name,
            artist=artist,
            album="Generated",
            file_path=str(output_dir),
            metadata=metadata_json,
        )

        return {
            "message": "Song processed successfully",
            "id": song_id,
            "song_name": song_name,
            "artist": artist,
            "tempo": tempo,
            "duration": duration,
            "total_beats": len(beat_times),
            "total_notes": len(onset_times) if onset_times else len(beat_times),
            "difficulty": difficulty,
            "notes_chart": str(chart_path),
            "song_ini": str(ini_path),
            "output_dir": str(output_dir),
        }

    except Exception as e:
        logger.exception(f"❌ Error processing song '{song_name}': {e}")
        return {"error": str(e)}


def generate_all_difficulties(file_path: str, **kwargs) -> Dict[str, Any]:
    """
    Generate charts for all four difficulty levels from a single audio file.

    Returns a combined result dict or error.
    """
    difficulties = ["easy", "medium", "hard", "expert"]
    results = {}
    first_result = None

    for diff in difficulties:
        result = process_song_file(file_path, difficulty=diff, **kwargs)
        if "error" in result:
            return result
        results[diff] = result
        if first_result is None:
            first_result = result

    if first_result:
        first_result["all_difficulties"] = True
        first_result["difficulties_generated"] = difficulties

    return first_result or {"error": "No charts generated"}
