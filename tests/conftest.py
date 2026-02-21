"""
Clone Hero Content Manager - Pytest Configuration & Shared Fixtures

Provides reusable fixtures for:
- Temporary directories for staging/output
- Sample chart file content (valid and invalid)
- Mock audio analysis results (mimicking librosa output)
- Sample song.ini data
- Helpers for creating test song folder structures
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Temporary directory fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Provide a clean temporary directory for each test."""
    return tmp_path


@pytest.fixture
def staging_dir(tmp_path: Path) -> Path:
    """Provide a staging directory mimicking the song generation output folder."""
    d = tmp_path / "staging"
    d.mkdir()
    return d


@pytest.fixture
def song_folder(tmp_path: Path) -> Path:
    """
    Create a minimal valid Clone Hero song folder with:
    - notes.chart  (valid, minimal)
    - song.ini     (valid, minimal)
    - song.ogg     (empty placeholder)
    """
    folder = tmp_path / "Test Artist - Test Song_abc12345"
    folder.mkdir()

    # notes.chart
    chart_content = SAMPLE_CHART_VALID
    (folder / "notes.chart").write_text(chart_content, encoding="utf-8-sig")

    # song.ini
    ini_content = SAMPLE_SONG_INI
    (folder / "song.ini").write_text(ini_content, encoding="utf-8")

    # Empty audio placeholder
    (folder / "song.ogg").write_bytes(b"\x00" * 64)

    return folder


@pytest.fixture
def song_folder_flac(tmp_path: Path) -> Path:
    """
    Create a song folder that uses FLAC audio (the problematic format).
    """
    folder = tmp_path / "Flac Artist - Flac Song_def67890"
    folder.mkdir()

    chart = SAMPLE_CHART_VALID.replace("song.ogg", "song.flac")
    (folder / "notes.chart").write_text(chart, encoding="utf-8-sig")
    (folder / "song.ini").write_text(SAMPLE_SONG_INI, encoding="utf-8")
    (folder / "song.flac").write_bytes(b"\x00" * 64)

    return folder


@pytest.fixture
def empty_chart_file(tmp_path: Path) -> Path:
    """Create an empty .chart file for error-handling tests."""
    p = tmp_path / "empty.chart"
    p.write_text("", encoding="utf-8")
    return p


@pytest.fixture
def malformed_chart_file(tmp_path: Path) -> Path:
    """Create a .chart file with unbalanced braces."""
    p = tmp_path / "malformed.chart"
    p.write_text(SAMPLE_CHART_MALFORMED, encoding="utf-8")
    return p


@pytest.fixture
def chart_missing_music_stream(tmp_path: Path) -> Path:
    """Create a .chart file missing the MusicStream field."""
    p = tmp_path / "no_music_stream.chart"
    p.write_text(SAMPLE_CHART_NO_MUSIC_STREAM, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Mock audio analysis fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_audio_analysis() -> Dict[str, Any]:
    """
    Return a dict mimicking the output of song_generator.analyze_audio().

    Contains realistic-looking but deterministic data for a ~180-second
    song at 120 BPM.
    """
    duration = 180.0
    tempo = 120.0
    beat_interval = 60.0 / tempo  # 0.5 seconds per beat
    num_beats = int(duration / beat_interval)

    beat_times = [i * beat_interval for i in range(num_beats)]

    # Onsets: roughly 2x the beat count (some on-beat, some off-beat)
    rng = np.random.RandomState(42)
    onset_times = sorted(rng.uniform(0.5, duration - 0.5, num_beats * 2).tolist())
    onset_strengths = rng.uniform(0.3, 1.0, len(onset_times)).tolist()

    # RMS energy curve
    rms_values = (np.sin(np.linspace(0, 4 * np.pi, 100)) * 0.3 + 0.5).tolist()

    # Segments (verse/chorus/bridge style)
    # The generator expects {"time": <float>, "label": <str>} format
    segments = [
        {"time": 0.0, "label": "Intro", "energy": 0.3},
        {"time": 30.0, "label": "Verse 1", "energy": 0.5},
        {"time": 75.0, "label": "Chorus", "energy": 0.8},
        {"time": 105.0, "label": "Verse 2", "energy": 0.5},
        {"time": 135.0, "label": "Chorus", "energy": 0.8},
        {"time": 165.0, "label": "Outro", "energy": 0.3},
    ]

    return {
        "tempo": tempo,
        "beat_times": beat_times,
        "onset_times": onset_times,
        "onset_strengths": onset_strengths,
        "duration": duration,
        "rms_values": rms_values,
        "segments": segments,
    }


@pytest.fixture
def mock_audio_analysis_short() -> Dict[str, Any]:
    """Short 30-second analysis for quick tests."""
    duration = 30.0
    tempo = 140.0
    beat_interval = 60.0 / tempo
    num_beats = int(duration / beat_interval)

    beat_times = [i * beat_interval for i in range(num_beats)]

    rng = np.random.RandomState(99)
    onset_times = sorted(rng.uniform(0.2, duration - 0.2, num_beats).tolist())
    onset_strengths = rng.uniform(0.4, 0.9, len(onset_times)).tolist()

    return {
        "tempo": tempo,
        "beat_times": beat_times,
        "onset_times": onset_times,
        "onset_strengths": onset_strengths,
        "duration": duration,
        "rms_values": [0.5] * 20,
        "segments": [
            {"time": 0.0, "label": "Intro", "energy": 0.4},
            {"time": 15.0, "label": "Main", "energy": 0.7},
        ],
    }


# ---------------------------------------------------------------------------
# Sample song.ini data fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_song_data() -> Dict[str, Any]:
    """Return a dict suitable for write_song_ini()."""
    return {
        "title": "Test Song",
        "artist": "Test Artist",
        "album": "Test Album",
        "metadata": {
            "charter": "testcharter",
            "song_length": "180000",
            "genre": "Rock",
            "preview_start_time": "45000",
            "diff_guitar": "3",
            "diff_bass": "-1",
            "diff_rhythm": "-1",
            "diff_drums": "-1",
            "diff_keys": "-1",
            "diff_guitarghl": "-1",
            "diff_bassghl": "-1",
            "year": "2024",
            "delay": "0",
            "loading_phrase": "Auto-generated chart at 120 BPM",
        },
    }


# ---------------------------------------------------------------------------
# Filename test cases fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def filename_test_cases() -> List[Dict[str, str]]:
    """
    Return a list of filename → expected parse result mappings.

    Each entry has:
      - filename: the raw filename string
      - expected_song: the expected cleaned song name
      - expected_artist: the expected cleaned artist (empty string if none)
    """
    return [
        {
            "filename": "Metallica - Enter Sandman.mp3",
            "expected_song": "Enter Sandman",
            "expected_artist": "Metallica",
        },
        {
            "filename": "artist_-_song_title.ogg",
            "expected_song": "Song Title",
            "expected_artist": "Artist",
        },
        {
            "filename": "Led Zeppelin – Stairway to Heaven.flac",
            "expected_song": "Stairway to Heaven",
            "expected_artist": "Led Zeppelin",
        },
        {
            "filename": "Bohemian Rhapsody.mp3",
            "expected_song": "Bohemian Rhapsody",
            "expected_artist": "",
        },
        {
            "filename": "01 - Welcome to the Jungle.mp3",
            "expected_song": "Welcome to the Jungle",
            "expected_artist": "",
        },
        {
            "filename": "03. Back in Black.ogg",
            "expected_song": "Back in Black",
            "expected_artist": "",
        },
        {
            "filename": "Nirvana - Smells Like Teen Spirit (Official Music Video).mp3",
            "expected_song": "Smells Like Teen Spirit",
            "expected_artist": "Nirvana",
        },
        {
            "filename": "ACDC - Thunderstruck [Official Audio].flac",
            "expected_song": "Thunderstruck",
            "expected_artist": "ACDC",
        },
        {
            "filename": "Drake - God's Plan (feat. Someone).wav",
            "expected_song": "God's Plan",
            "expected_artist": "Drake",
        },
        {
            "filename": "simple_song_name.opus",
            "expected_song": "Simple Song Name",
            "expected_artist": "",
        },
    ]


# ---------------------------------------------------------------------------
# Sample chart content constants
# ---------------------------------------------------------------------------

SAMPLE_CHART_VALID = """\
[Song]
{
  Name = "Test Song"
  Artist = "Test Artist"
  Album = "Test Album"
  Year = ", 2024"
  Charter = "testcharter"
  Offset = 0
  Resolution = 192
  Player2 = bass
  Difficulty = 0
  PreviewStart = 0
  PreviewEnd = 0
  Genre = "rock"
  MediaType = "cd"
  MusicStream = "song.ogg"
}
[SyncTrack]
{
  0 = TS 4
  0 = B 120000
}
[Events]
{
  768 = E "section Intro"
  5760 = E "section Verse 1"
  11520 = E "section Chorus"
}
[ExpertSingle]
{
  768 = N 0 0
  960 = N 1 0
  1152 = N 2 0
  1344 = N 3 0
  1536 = N 4 0
  1728 = N 0 0
  1920 = N 1 0
  2112 = N 2 96
  2304 = N 0 0
  2496 = N 3 0
  2688 = S 2 768
}
[HardSingle]
{
  768 = N 0 0
  1152 = N 1 0
  1536 = N 2 0
  1920 = N 0 0
  2304 = N 1 0
}
[MediumSingle]
{
  768 = N 0 0
  1536 = N 1 0
  2304 = N 0 0
}
[EasySingle]
{
  768 = N 0 0
  1536 = N 0 0
  2304 = N 1 0
}
"""

SAMPLE_CHART_MALFORMED = """\
[Song]
{
  Name = "Broken"
  Resolution = 192
  MusicStream = "song.ogg"

[SyncTrack]
{
  0 = TS 4
  0 = B 120000
}
[ExpertSingle]
{
  768 = N 0 0
}
"""

SAMPLE_CHART_NO_MUSIC_STREAM = """\
[Song]
{
  Name = "No Audio"
  Artist = "Nobody"
  Resolution = 192
  Offset = 0
}
[SyncTrack]
{
  0 = TS 4
  0 = B 120000
}
[ExpertSingle]
{
  768 = N 0 0
  960 = N 1 0
}
"""

SAMPLE_CHART_UNSTABLE_BPM = """\
[Song]
{
  Name = "Unstable BPM"
  Artist = "Jittery"
  Resolution = 192
  MusicStream = "song.ogg"
}
[SyncTrack]
{
  0 = TS 4
  0 = B 120000
  192 = B 121000
  384 = B 119500
  576 = B 120200
  768 = B 119800
  960 = B 120100
  1152 = B 119900
  1344 = B 120300
  1536 = B 119700
  1728 = B 120500
}
[ExpertSingle]
{
  768 = N 0 0
  960 = N 1 0
  1152 = N 2 0
}
"""

SAMPLE_SONG_INI = """\
[song]
name = Test Song
artist = Test Artist
album = Test Album
genre = rock
year = 2024
charter = testcharter
song_length = 180000
diff_guitar = 3
diff_bass = -1
diff_rhythm = -1
diff_drums = -1
diff_keys = -1
diff_guitarghl = -1
diff_bassghl = -1
preview_start_time = 45000
delay = 0
loading_phrase = Auto-generated chart at 120 BPM
"""


# ---------------------------------------------------------------------------
# MusicBrainz mock response fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_musicbrainz_response() -> Dict[str, Any]:
    """Return a mock MusicBrainz recording search response."""
    return {
        "recordings": [
            {
                "id": "abc12345-6789-0123-4567-890abcdef012",
                "title": "Enter Sandman",
                "artist-credit": [
                    {
                        "name": "Metallica",
                        "joinphrase": "",
                    }
                ],
                "releases": [
                    {
                        "id": "release-id-001",
                        "title": "Metallica (The Black Album)",
                        "date": "1991-08-12",
                        "release-group": {
                            "id": "rg-id-001",
                        },
                    }
                ],
                "tags": [
                    {"name": "metal", "count": 15},
                    {"name": "heavy metal", "count": 12},
                    {"name": "thrash metal", "count": 8},
                    {"name": "rock", "count": 5},
                ],
            }
        ]
    }


@pytest.fixture
def mock_musicbrainz_empty_response() -> Dict[str, Any]:
    """Return a mock MusicBrainz response with no results."""
    return {"recordings": []}


# ---------------------------------------------------------------------------
# Helper to create a complete song folder on disk
# ---------------------------------------------------------------------------


@pytest.fixture
def create_song_folder(tmp_path: Path):
    """
    Factory fixture: call with (name, artist, audio_ext) to create a
    full song folder on disk. Returns the folder Path.
    """

    def _factory(
        name: str = "Test Song",
        artist: str = "Test Artist",
        audio_ext: str = ".ogg",
        include_chart: bool = True,
        include_ini: bool = True,
        include_art: bool = False,
        charter: str = "testcharter",
    ) -> Path:
        safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in name)
        folder = tmp_path / f"{artist} - {safe}"
        folder.mkdir(parents=True, exist_ok=True)

        audio_file = f"song{audio_ext}"
        (folder / audio_file).write_bytes(b"\x00" * 128)

        if include_chart:
            chart = SAMPLE_CHART_VALID.replace("song.ogg", audio_file)
            chart = chart.replace("Test Song", name)
            chart = chart.replace("Test Artist", artist)
            chart = chart.replace("testcharter", charter)
            (folder / "notes.chart").write_text(chart, encoding="utf-8-sig")

        if include_ini:
            ini = SAMPLE_SONG_INI.replace("Test Song", name)
            ini = ini.replace("Test Artist", artist)
            ini = ini.replace("testcharter", charter)
            (folder / "song.ini").write_text(ini, encoding="utf-8")

        if include_art:
            # Create a tiny valid 1x1 PNG
            png_header = (
                b"\x89PNG\r\n\x1a\n"
                b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
                b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx"
                b"\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00"
                b"\x00\x00\x00IEND\xaeB`\x82"
            )
            (folder / "album.png").write_bytes(png_header)

        return folder

    return _factory
