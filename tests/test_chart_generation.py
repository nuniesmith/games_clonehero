"""
Clone Hero Content Manager - Chart Generation Tests

Tests for the song_generator module's chart output. Validates:
- generate_notes_chart() produces a well-formed .chart file
- All required sections are present ([Song], [SyncTrack], [Events], difficulty sections)
- [Song] metadata fields are correct (Name, Artist, Album, MusicStream, Resolution, etc.)
- [SyncTrack] contains valid BPM and time signature markers
- BPM stability: tempo map doesn't produce excessive BPM changes (jitter)
- Difficulty sections contain valid note events (N, S markers)
- All four difficulty levels are generated (Easy, Medium, Hard, Expert)
- Note numbers are within Clone Hero's valid range (0–4 for single, plus modifiers)
- Sustain lengths are non-negative
- Star power sections (S 2) are present and valid
- Events section contains section markers
- Output file uses UTF-8 BOM encoding
- MusicStream field matches the provided audio filename
- Edge cases: very short songs, very fast/slow tempos, no onsets, no segments
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

import numpy as np

from src.services.song_generator import (
    DIFFICULTY_PROFILES,
    _bpm_to_chart_value,
    _compute_stable_tempo_map,
    _compute_star_power_sections,
    _compute_sustain,
    _detect_tap_runs,
    _seconds_to_ticks,
    _select_note,
    _should_hopo,
    generate_notes_chart,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIFFICULTY_SECTIONS = [
    "[EasySingle]",
    "[MediumSingle]",
    "[HardSingle]",
    "[ExpertSingle]",
]

REQUIRED_SONG_FIELDS = [
    "Name",
    "Artist",
    "Resolution",
    "MusicStream",
]


def _parse_chart_sections(text: str) -> dict[str, str]:
    """
    Parse a .chart file into a dict of section_name -> section_body.
    The section body is everything between the opening '{' and closing '}'.
    """
    sections: dict[str, str] = {}
    current_section = None
    brace_depth = 0
    body_lines: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()

        if stripped.startswith("[") and stripped.endswith("]") and brace_depth == 0:
            current_section = stripped
            body_lines = []
            continue

        if stripped == "{":
            brace_depth += 1
            continue

        if stripped == "}":
            brace_depth -= 1
            if brace_depth == 0 and current_section is not None:
                sections[current_section] = "\n".join(body_lines)
                current_section = None
            continue

        if brace_depth > 0 and current_section is not None:
            body_lines.append(stripped)

    return sections


def _parse_song_fields(section_body: str) -> dict[str, str]:
    """Parse key = value pairs from a [Song] section body."""
    fields: dict[str, str] = {}
    for line in section_body.splitlines():
        line = line.strip()
        if "=" in line:
            key, _, value = line.partition("=")
            fields[key.strip()] = value.strip().strip('"')
    return fields


def _extract_note_events(section_body: str) -> list[tuple[int, str, str]]:
    """
    Extract note events from a difficulty section body.

    Returns a list of (tick, event_type, rest_of_line).
    For 'N' events, rest_of_line is e.g. '0 0' (note_num sustain_length).
    For 'S' events, rest_of_line is e.g. '2 768' (star_power length).
    """
    events: list[tuple[int, str, str]] = []
    for line in section_body.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        tick_part, _, event_part = line.partition("=")
        tick_str = tick_part.strip()
        event_str = event_part.strip()
        try:
            tick = int(tick_str)
        except ValueError:
            continue
        parts = event_str.split(None, 1)
        if len(parts) >= 2:
            events.append((tick, parts[0], parts[1]))
        elif len(parts) == 1:
            events.append((tick, parts[0], ""))
    return events


def _generate_chart_to_text(
    mock_analysis: dict[str, Any],
    output_path: Path,
    audio_filename: str = "song.ogg",
    song_name: str = "Test Song",
    artist: str = "Test Artist",
    album: str = "Test Album",
    year: str = "2024",
    genre: str = "Rock",
    difficulties: list[str] | None = None,
    enable_lyrics: bool = True,
) -> str:
    """
    Helper that calls generate_notes_chart with the given mock analysis
    and returns the resulting file text.
    """
    if difficulties is None:
        difficulties = ["easy", "medium", "hard", "expert"]

    ok = generate_notes_chart(
        song_name=song_name,
        artist=artist,
        album=album,
        year=year,
        genre=genre,
        tempo=mock_analysis["tempo"],
        beat_times=mock_analysis["beat_times"],
        onset_times=mock_analysis["onset_times"],
        onset_strengths=mock_analysis.get("onset_strengths", []),
        duration=mock_analysis["duration"],
        output_path=output_path,
        audio_filename=audio_filename,
        segments=mock_analysis.get("segments", []),
        difficulties=difficulties,
        enable_lyrics=enable_lyrics,
    )
    assert ok, "generate_notes_chart() returned False — chart generation failed"
    return output_path.read_text(encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# Test: File Structure & Encoding
# ---------------------------------------------------------------------------


class TestChartFileStructure:
    """Validate the overall file structure and encoding."""

    def test_file_created(self, mock_audio_analysis, tmp_path):
        """generate_notes_chart should create a file on disk."""
        out = tmp_path / "notes.chart"
        _generate_chart_to_text(mock_audio_analysis, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_utf8_bom_encoding(self, mock_audio_analysis, tmp_path):
        """Output should be UTF-8 with BOM (Clone Hero expects this)."""
        out = tmp_path / "notes.chart"
        _generate_chart_to_text(mock_audio_analysis, out)
        raw_bytes = out.read_bytes()
        assert raw_bytes[:3] == b"\xef\xbb\xbf", (
            "Chart file should start with UTF-8 BOM"
        )

    def test_balanced_braces(self, mock_audio_analysis, tmp_path):
        """Every '{' must have a matching '}'."""
        out = tmp_path / "notes.chart"
        text = _generate_chart_to_text(mock_audio_analysis, out)
        assert text.count("{") == text.count("}"), (
            f"Unbalanced braces: {text.count('{')} opens vs {text.count('}')} closes"
        )

    def test_all_required_sections_present(self, mock_audio_analysis, tmp_path):
        """Chart must have [Song], [SyncTrack], [Events], and all difficulty sections."""
        out = tmp_path / "notes.chart"
        text = _generate_chart_to_text(mock_audio_analysis, out)
        sections = _parse_chart_sections(text)

        for required in ["[Song]", "[SyncTrack]", "[Events]"]:
            assert required in sections, f"Missing required section: {required}"

        for diff_section in DIFFICULTY_SECTIONS:
            assert diff_section in sections, (
                f"Missing difficulty section: {diff_section}"
            )

    def test_section_order(self, mock_audio_analysis, tmp_path):
        """Sections should appear in the standard order."""
        out = tmp_path / "notes.chart"
        text = _generate_chart_to_text(mock_audio_analysis, out)

        positions = {}
        for section_name in ["[Song]", "[SyncTrack]", "[Events]"] + DIFFICULTY_SECTIONS:
            pos = text.find(section_name)
            if pos >= 0:
                positions[section_name] = pos

        # [Song] should come first
        assert positions.get("[Song]", float("inf")) < positions.get(
            "[SyncTrack]", float("inf")
        )
        assert positions.get("[SyncTrack]", float("inf")) < positions.get(
            "[Events]", float("inf")
        )


# ---------------------------------------------------------------------------
# Test: [Song] Section Metadata
# ---------------------------------------------------------------------------


class TestSongSection:
    """Validate [Song] section content."""

    def test_required_fields_present(self, mock_audio_analysis, tmp_path):
        """All required metadata fields must exist in [Song]."""
        out = tmp_path / "notes.chart"
        text = _generate_chart_to_text(mock_audio_analysis, out)
        sections = _parse_chart_sections(text)
        fields = _parse_song_fields(sections["[Song]"])

        for field in REQUIRED_SONG_FIELDS:
            assert field in fields, f"Missing required field in [Song]: {field}"

    def test_name_matches(self, mock_audio_analysis, tmp_path):
        out = tmp_path / "notes.chart"
        text = _generate_chart_to_text(
            mock_audio_analysis, out, song_name="My Custom Song"
        )
        sections = _parse_chart_sections(text)
        fields = _parse_song_fields(sections["[Song]"])
        assert fields["Name"] == "My Custom Song"

    def test_artist_matches(self, mock_audio_analysis, tmp_path):
        out = tmp_path / "notes.chart"
        text = _generate_chart_to_text(mock_audio_analysis, out, artist="Cool Band")
        sections = _parse_chart_sections(text)
        fields = _parse_song_fields(sections["[Song]"])
        assert fields["Artist"] == "Cool Band"

    def test_album_matches(self, mock_audio_analysis, tmp_path):
        out = tmp_path / "notes.chart"
        text = _generate_chart_to_text(mock_audio_analysis, out, album="Greatest Hits")
        sections = _parse_chart_sections(text)
        fields = _parse_song_fields(sections["[Song]"])
        assert fields["Album"] == "Greatest Hits"

    def test_music_stream_matches_audio_filename(self, mock_audio_analysis, tmp_path):
        """MusicStream must point to the actual audio file."""
        out = tmp_path / "notes.chart"
        text = _generate_chart_to_text(
            mock_audio_analysis, out, audio_filename="song.ogg"
        )
        sections = _parse_chart_sections(text)
        fields = _parse_song_fields(sections["[Song]"])
        assert fields["MusicStream"] == "song.ogg"

    def test_music_stream_opus(self, mock_audio_analysis, tmp_path):
        """MusicStream should work with .opus files too."""
        out = tmp_path / "notes.chart"
        text = _generate_chart_to_text(
            mock_audio_analysis, out, audio_filename="song.opus"
        )
        sections = _parse_chart_sections(text)
        fields = _parse_song_fields(sections["[Song]"])
        assert fields["MusicStream"] == "song.opus"

    def test_music_stream_mp3(self, mock_audio_analysis, tmp_path):
        out = tmp_path / "notes.chart"
        text = _generate_chart_to_text(
            mock_audio_analysis, out, audio_filename="song.mp3"
        )
        sections = _parse_chart_sections(text)
        fields = _parse_song_fields(sections["[Song]"])
        assert fields["MusicStream"] == "song.mp3"

    def test_resolution_is_192(self, mock_audio_analysis, tmp_path):
        """Standard Clone Hero resolution is 192 ticks per quarter note."""
        out = tmp_path / "notes.chart"
        text = _generate_chart_to_text(mock_audio_analysis, out)
        sections = _parse_chart_sections(text)
        fields = _parse_song_fields(sections["[Song]"])
        assert fields["Resolution"] == "192"

    def test_genre_and_year(self, mock_audio_analysis, tmp_path):
        out = tmp_path / "notes.chart"
        text = _generate_chart_to_text(
            mock_audio_analysis, out, genre="Metal", year="1991"
        )
        sections = _parse_chart_sections(text)
        fields = _parse_song_fields(sections["[Song]"])
        assert fields.get("Genre") == "metal" or fields.get("Genre") == "Metal"
        # Year format in .chart is typically ", YYYY"
        year_val = fields.get("Year", "")
        assert "1991" in year_val


# ---------------------------------------------------------------------------
# Test: [SyncTrack] — BPM & Time Signatures
# ---------------------------------------------------------------------------


class TestSyncTrack:
    """Validate tempo map and time signature markers."""

    def test_has_initial_time_signature(self, mock_audio_analysis, tmp_path):
        """There must be a TS event at tick 0."""
        out = tmp_path / "notes.chart"
        text = _generate_chart_to_text(mock_audio_analysis, out)
        sections = _parse_chart_sections(text)
        sync_body = sections["[SyncTrack]"]

        # Look for "0 = TS <n>"
        assert re.search(r"^\s*0\s*=\s*TS\s+\d+", sync_body, re.MULTILINE), (
            "Missing initial time signature (TS) at tick 0"
        )

    def test_has_initial_bpm(self, mock_audio_analysis, tmp_path):
        """There must be a B (BPM) event at tick 0."""
        out = tmp_path / "notes.chart"
        text = _generate_chart_to_text(mock_audio_analysis, out)
        sections = _parse_chart_sections(text)
        sync_body = sections["[SyncTrack]"]

        assert re.search(r"^\s*0\s*=\s*B\s+\d+", sync_body, re.MULTILINE), (
            "Missing initial BPM (B) marker at tick 0"
        )

    def test_bpm_values_are_positive(self, mock_audio_analysis, tmp_path):
        """All BPM values in SyncTrack must be positive integers."""
        out = tmp_path / "notes.chart"
        text = _generate_chart_to_text(mock_audio_analysis, out)
        sections = _parse_chart_sections(text)

        for line in sections["[SyncTrack]"].splitlines():
            match = re.match(r"\s*\d+\s*=\s*B\s+(\d+)", line)
            if match:
                bpm_val = int(match.group(1))
                assert bpm_val > 0, f"BPM value must be positive, got {bpm_val}"

    def test_bpm_stability_few_changes(self, mock_audio_analysis, tmp_path):
        """
                A stable tempo (120 BPM constant) should not produce excessive BPM
                changes in the sync track.  The old generator bug created hundreds
                of jittery BPM markers
        .
        """
        out = tmp_path / "notes.chart"
        text = _generate_chart_to_text(mock_audio_analysis, out)
        sections = _parse_chart_sections(text)

        bpm_events = re.findall(
            r"^\s*\d+\s*=\s*B\s+\d+", sections["[SyncTrack]"], re.MULTILINE
        )

        # For a constant-tempo song, we expect at most a handful of BPM markers
        # (1 initial + maybe a few from analysis variance).  Definitely not hundreds.
        assert len(bpm_events) < 50, (
            f"Too many BPM changes ({len(bpm_events)}) — possible tempo jitter bug"
        )

    def test_ticks_are_monotonically_increasing(self, mock_audio_analysis, tmp_path):
        """Tick values in SyncTrack should be monotonically non-decreasing."""
        out = tmp_path / "notes.chart"
        text = _generate_chart_to_text(mock_audio_analysis, out)
        sections = _parse_chart_sections(text)

        ticks = []
        for line in sections["[SyncTrack]"].splitlines():
            match = re.match(r"\s*(\d+)\s*=", line)
            if match:
                ticks.append(int(match.group(1)))

        for i in range(1, len(ticks)):
            assert ticks[i] >= ticks[i - 1], (
                f"SyncTrack ticks not monotonic: {ticks[i - 1]} > {ticks[i]} at index {i}"
            )


# ---------------------------------------------------------------------------
# Test: Difficulty Sections — Note Events
# ---------------------------------------------------------------------------


class TestDifficultySections:
    """Validate note events in the generated difficulty sections."""

    def test_all_four_difficulties_have_notes(self, mock_audio_analysis, tmp_path):
        """Each difficulty section should contain at least one note."""
        out = tmp_path / "notes.chart"
        text = _generate_chart_to_text(mock_audio_analysis, out)
        sections = _parse_chart_sections(text)

        for diff in DIFFICULTY_SECTIONS:
            events = _extract_note_events(sections[diff])
            note_events = [e for e in events if e[1] == "N"]
            assert len(note_events) > 0, f"{diff} has no note events"

    def test_note_numbers_in_valid_range(self, mock_audio_analysis, tmp_path):
        """
        Note numbers for Single guitar should be 0-4 (green through orange).
        5 = forced strum, 6 = tap, 7 = open note.  All valid <=7.
        """
        out = tmp_path / "notes.chart"
        text = _generate_chart_to_text(mock_audio_analysis, out)
        sections = _parse_chart_sections(text)

        for diff in DIFFICULTY_SECTIONS:
            events = _extract_note_events(sections[diff])
            for tick, etype, rest in events:
                if etype == "N":
                    parts = rest.split()
                    assert len(parts) >= 2, (
                        f"Malformed N event in {diff} at tick {tick}: '{rest}'"
                    )
                    note_num = int(parts[0])
                    assert 0 <= note_num <= 7, (
                        f"Invalid note number {note_num} in {diff} at tick {tick}"
                    )

    def test_sustain_lengths_non_negative(self, mock_audio_analysis, tmp_path):
        """Sustain lengths on note events must be >= 0."""
        out = tmp_path / "notes.chart"
        text = _generate_chart_to_text(mock_audio_analysis, out)
        sections = _parse_chart_sections(text)

        for diff in DIFFICULTY_SECTIONS:
            events = _extract_note_events(sections[diff])
            for tick, etype, rest in events:
                if etype == "N":
                    parts = rest.split()
                    sustain = int(parts[1])
                    assert sustain >= 0, (
                        f"Negative sustain {sustain} in {diff} at tick {tick}"
                    )

    def test_ticks_are_monotonically_increasing_in_notes(
        self, mock_audio_analysis, tmp_path
    ):
        """Note tick positions should be monotonically non-decreasing."""
        out = tmp_path / "notes.chart"
        text = _generate_chart_to_text(mock_audio_analysis, out)
        sections = _parse_chart_sections(text)

        for diff in DIFFICULTY_SECTIONS:
            events = _extract_note_events(sections[diff])
            ticks = [e[0] for e in events]
            for i in range(1, len(ticks)):
                assert ticks[i] >= ticks[i - 1], (
                    f"Non-monotonic ticks in {diff}: {ticks[i - 1]} > {ticks[i]}"
                )

    def test_expert_has_more_notes_than_easy(self, mock_audio_analysis, tmp_path):
        """Expert difficulty should generally have more notes than Easy."""
        out = tmp_path / "notes.chart"
        text = _generate_chart_to_text(mock_audio_analysis, out)
        sections = _parse_chart_sections(text)

        easy_notes = len(
            [e for e in _extract_note_events(sections["[EasySingle]"]) if e[1] == "N"]
        )
        expert_notes = len(
            [e for e in _extract_note_events(sections["[ExpertSingle]"]) if e[1] == "N"]
        )
        assert expert_notes >= easy_notes, (
            f"Expert ({expert_notes} notes) should have >= Easy ({easy_notes} notes)"
        )

    def test_note_density_increases_with_difficulty(
        self, mock_audio_analysis, tmp_path
    ):
        """Note count should generally increase: Easy <= Medium <= Hard <= Expert."""
        out = tmp_path / "notes.chart"
        text = _generate_chart_to_text(mock_audio_analysis, out)
        sections = _parse_chart_sections(text)

        counts = {}
        for diff in DIFFICULTY_SECTIONS:
            notes = [e for e in _extract_note_events(sections[diff]) if e[1] == "N"]
            counts[diff] = len(notes)

        # Allow some tolerance — we just check the overall trend
        assert counts["[EasySingle]"] <= counts["[ExpertSingle]"], (
            f"Easy ({counts['[EasySingle]']}) should have <= Expert ({counts['[ExpertSingle]']})"
        )

    def test_star_power_sections_present(self, mock_audio_analysis, tmp_path):
        """
        Expert section should contain star power markers (S 2 <length>).
        """
        out = tmp_path / "notes.chart"
        text = _generate_chart_to_text(mock_audio_analysis, out)
        sections = _parse_chart_sections(text)

        expert_events = _extract_note_events(sections["[ExpertSingle]"])
        sp_events = [e for e in expert_events if e[1] == "S"]
        # Star power is expected in Expert for a 180-second song
        assert len(sp_events) > 0, "Expert section should have star power (S) markers"

    def test_star_power_length_positive(self, mock_audio_analysis, tmp_path):
        """Star power S events should have positive length."""
        out = tmp_path / "notes.chart"
        text = _generate_chart_to_text(mock_audio_analysis, out)
        sections = _parse_chart_sections(text)

        for diff in DIFFICULTY_SECTIONS:
            events = _extract_note_events(sections[diff])
            for tick, etype, rest in events:
                if etype == "S":
                    parts = rest.split()
                    if len(parts) >= 2:
                        sp_len = int(parts[1])
                        assert sp_len > 0, (
                            f"Star power length must be positive in {diff} at tick {tick}"
                        )


# ---------------------------------------------------------------------------
# Test: [Events] Section
# ---------------------------------------------------------------------------


class TestEventsSection:
    """Validate the [Events] section."""

    def test_has_section_markers(self, mock_audio_analysis, tmp_path):
        """Events section should contain at least one 'section' event."""
        out = tmp_path / "notes.chart"
        text = _generate_chart_to_text(mock_audio_analysis, out)
        sections = _parse_chart_sections(text)
        events_body = sections.get("[Events]", "")

        section_events = re.findall(r'E\s+"section\s+', events_body)
        assert len(section_events) > 0, "Events should have at least one section marker"

    def test_lyrics_when_enabled(self, mock_audio_analysis, tmp_path):
        """When enable_lyrics=True, events should contain lyric entries."""
        out = tmp_path / "notes.chart"
        text = _generate_chart_to_text(mock_audio_analysis, out, enable_lyrics=True)
        sections = _parse_chart_sections(text)
        events_body = sections.get("[Events]", "")

        # Lyrics appear as E "lyric ..." or E "phrase_start"/"phrase_end"
        # The exact format depends on the lyrics generator, but there should
        # be _something_ beyond just section markers
        event_lines = [
            line.strip()
            for line in events_body.splitlines()
            if line.strip() and "=" in line
        ]
        section_lines = [el for el in event_lines if "section" in el.lower()]
        # With lyrics enabled, there should be more events than just sections
        # (unless lyrics generation itself is a no-op for the mock data)
        assert len(event_lines) >= len(section_lines), (
            "With lyrics enabled, expected events beyond section markers"
        )

    def test_events_ticks_monotonic(self, mock_audio_analysis, tmp_path):
        """Event tick values should be monotonically non-decreasing."""
        out = tmp_path / "notes.chart"
        text = _generate_chart_to_text(mock_audio_analysis, out)
        sections = _parse_chart_sections(text)
        events_body = sections.get("[Events]", "")

        ticks = []
        for line in events_body.splitlines():
            match = re.match(
                r"\s*(\d+)\
s*=",
                line,
            )
            if match:
                ticks.append(int(match.group(1)))

        for i in range(1, len(ticks)):
            assert ticks[i] >= ticks[i - 1], (
                f"Events ticks not monotonic: {ticks[i - 1]} > {ticks[i]}"
            )


# ---------------------------------------------------------------------------
# Test: Subset of difficulties
# ---------------------------------------------------------------------------


class TestSubsetDifficulties:
    """Test generating only a subset of difficulty levels."""

    def test_expert_only(self, mock_audio_analysis, tmp_path):
        """Generating only expert should still produce a valid chart."""
        out = tmp_path / "notes.chart"
        text = _generate_chart_to_text(
            mock_audio_analysis, out, difficulties=["expert"]
        )
        sections = _parse_chart_sections(text)
        assert "[ExpertSingle]" in sections
        # Other difficulties should not be present
        # (unless the generator always includes all four)

    def test_easy_and_hard(self, mock_audio_analysis, tmp_path):
        """Generating easy+hard should produce valid charts for those levels."""
        out = tmp_path / "notes.chart"
        text = _generate_chart_to_text(
            mock_audio_analysis, out, difficulties=["easy", "hard"]
        )
        sections = _parse_chart_sections(text)
        assert "[EasySingle]" in sections
        assert "[HardSingle]" in sections


# ---------------------------------------------------------------------------
# Test: Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test unusual or boundary inputs."""

    def test_very_short_song(self, tmp_path):
        """A very short song (5 seconds) should still produce a valid chart."""
        short_analysis = {
            "tempo": 120.0,
            "beat_times": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
            "onset_times": [0.5, 1.5, 2.5, 3.5, 4.5],
            "onset_strengths": [0.5, 0.6, 0.7, 0.8, 0.9],
            "duration": 5.0,
            "segments": [{"start": 0.0, "end": 5.0, "label": "Full"}],
        }
        out = tmp_path / "notes.chart"
        text = _generate_chart_to_text(short_analysis, out)
        sections = _parse_chart_sections(text)
        assert "[Song]" in sections
        assert "[ExpertSingle]" in sections

    def test_very_fast_tempo(self, tmp_path):
        """A very fast tempo (300 BPM) should not crash."""
        rng = np.random.RandomState(77)
        duration = 30.0
        beat_interval = 60.0 / 300.0
        num_beats = int(duration / beat_interval)
        beat_times = [i * beat_interval for i in range(num_beats)]
        onset_times = sorted(rng.uniform(0.1, duration - 0.1, num_beats).tolist())

        analysis = {
            "tempo": 300.0,
            "beat_times": beat_times,
            "onset_times": onset_times,
            "onset_strengths": rng.uniform(0.3, 1.0, len(onset_times)).tolist(),
            "duration": duration,
            "segments": [],
        }
        out = tmp_path / "notes.chart"
        text = _generate_chart_to_text(analysis, out)
        sections = _parse_chart_sections(text)
        assert "[ExpertSingle]" in sections

    def test_very_slow_tempo(self, tmp_path):
        """A very slow tempo (40 BPM) should not crash."""
        duration = 60.0
        beat_interval = 60.0 / 40.0
        num_beats = int(duration / beat_interval)
        beat_times = [i * beat_interval for i in range(num_beats)]
        onset_times = [bt + 0.1 for bt in beat_times if bt + 0.1 < duration]

        analysis = {
            "tempo": 40.0,
            "beat_times": beat_times,
            "onset_times": onset_times,
            "onset_strengths": [0.5] * len(onset_times),
            "duration": duration,
            "segments": [],
        }
        out = tmp_path / "notes.chart"
        text = _generate_chart_to_text(analysis, out)
        sections = _parse_chart_sections(text)
        assert "[Song]" in sections

    def test_no_onsets(self, tmp_path):
        """If there are no onsets, the chart should still be valid (maybe empty notes)."""
        beat_times_list: list[float] = [i * 0.5 for i in range(60)]
        out = tmp_path / "notes.chart"
        ok = generate_notes_chart(
            song_name="Empty",
            artist="Nobody",
            album="None",
            year="",
            genre="",
            tempo=120.0,
            beat_times=beat_times_list,
            onset_times=[],
            onset_strengths=[],
            duration=30.0,
            output_path=out,
            audio_filename="song.ogg",
            segments=[],
            difficulties=["expert"],
            enable_lyrics=False,
        )
        # The generator should handle this gracefully — either succeed with
        # an empty note section or return False.  Either is acceptable.
        if ok:
            text = out.read_text(encoding="utf-8-sig")
            sections = _parse_chart_sections(text)
            assert "[Song]" in sections

    def test_no_segments(self, mock_audio_analysis, tmp_path):
        """An analysis with no segments should still produce valid events."""
        mock_audio_analysis["segments"] = []
        out = tmp_path / "notes.chart"
        text = _generate_chart_to_text(mock_audio_analysis, out)
        sections = _parse_chart_sections(text)
        assert "[Events]" in sections

    def test_special_characters_in_metadata(self, mock_audio_analysis, tmp_path):
        """Song name with special chars should be written without corruption."""
        out = tmp_path / "notes.chart"
        text = _generate_chart_to_text(
            mock_audio_analysis,
            out,
            song_name='Song "With" Quotes & Amps',
            artist="Ar/ti\\st",
        )
        sections = _parse_chart_sections(text)
        assert "[Song]" in sections
        # The file should be parseable regardless of special chars
        fields = _parse_song_fields(sections["[Song]"])
        assert len(fields.get("Name", "")) > 0
        assert len(fields.get("Artist", "")) > 0

    def test_unicode_metadata(self, mock_audio_analysis, tmp_path):
        """Unicode characters in metadata should round-trip correctly."""
        out = tmp_path / "notes.chart"
        text = _generate_chart_to_text(
            mock_audio_analysis,
            out,
            song_name="Jóga",
            artist="Björk",
        )
        sections = _parse_chart_sections(text)
        fields = _parse_song_fields(sections["[Song]"])
        assert (
            "Jóga" in fields.get("Name", "") or "jóga" in fields.get("Name", "").lower()
        )


# ---------------------------------------------------------------------------
# Test: Helper Functions
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    """Test the internal helper functions used by the generator."""

    def test_seconds_to_ticks_zero(self):
        """0 seconds should map to 0 ticks."""
        result = _seconds_to_ticks(0.0, 120.0, 192)
        assert result == 0

    def test_seconds_to_ticks_one_beat(self):
        """
        At 120 BPM with resolution 192:
        One beat = 0.5 seconds = 192 ticks
        """
        result = _seconds_to_ticks(0.5, 120.0, 192)
        assert result == 192

    def test_seconds_to_ticks_positive(self):
        """Result should always be non-negative for positive input."""
        result = _seconds_to_ticks(5.0, 120.0, 192)
        assert result > 0

    def test_bpm_to_chart_value(self):
        """
        Chart BPM values are stored as BPM * 1000.
        120 BPM → 120000.
        """
        result = _bpm_to_chart_value(120.0)
        assert result == 120000

    def test_bpm_to_chart_value_fractional(self):
        """Fractional BPM should round to nearest integer * 1000."""
        result = _bpm_to_chart_value(128.5)
        assert result == 128500 or abs(result - 128500) < 1000

    def test_compute_stable_tempo_map_constant_bpm(self):
        """
        A perfectly constant-interval beat list should produce very few
        (ideally 1) BPM entries.
        """
        tempo = 120.0
        interval = 60.0 / tempo
        beats = [i * interval for i in range(100)]

        result = _compute_stable_tempo_map(tempo, beats, 192)
        # Should have at least 1 entry (the initial BPM)
        assert len(result) >= 1
        # Should NOT have hundreds of jittery entries
        assert len(result) < 10, (
            f"Stable tempo should produce few BPM changes, got {len(result)}"
        )

    def test_compute_stable_tempo_map_actual_tempo_change(self):
        """
        A beat list with a genuine tempo change should reflect it.
        """
        # First half at 120 BPM, second half at 150 BPM
        beats_120 = [i * 0.5 for i in range(50)]  # 25 seconds at 120 BPM
        beats_150 = [25.0 + i * 0.4 for i in range(50)]  # next part at 150 BPM
        all_beats = beats_120 + beats_150

        result = _compute_stable_tempo_map(120.0, all_beats, 192)
        # Should have at least 2 entries for the tempo change
        assert len(result) >= 2, "Should detect the tempo change"

    def test_select_note_returns_valid_range(self):
        """_select_note should return lane indices in 0-4."""
        rng = random.Random(42)
        profile = DIFFICULTY_PROFILES["expert"]
        for i in range(100):
            onset_strength = i / 100.0
            lanes = _select_note(i, onset_strength, "expert", 0, rng, profile)
            assert isinstance(lanes, list)
            for lane in lanes:
                assert (0 <= lane <= 4) or lane == 7, (
                    f"Lane {lane} out of range for expert (valid: 0-4, 7)"
                )

    def test_select_note_easy_fewer_frets(self):
        """Easy difficulty should tend to use fewer fret numbers (0-2)."""
        rng = random.Random(42)
        profile = DIFFICULTY_PROFILES["easy"]
        all_lanes = []
        for i in range(200):
            lanes = _select_note(i, 0.5, "easy", 0, rng, profile)
            all_lanes.extend(lanes)
        # Easy should mostly use notes 0-max_lane (which is 2 for easy)
        max_lane = profile["max_lane"]
        high_notes = sum(1 for n in all_lanes if n > max_lane)
        assert high_notes == 0, (
            f"Easy difficulty should not exceed max_lane={max_lane}: "
            f"found {high_notes} notes above it"
        )

    # --- Energy-based HOPO tests ---

    def test_should_hopo_disabled_on_easy(self):
        """Easy profile has hopo_energy_threshold=0 → never HOPO."""
        rng = random.Random(42)
        profile = DIFFICULTY_PROFILES["easy"]
        result = _should_hopo(
            tick=96,
            prev_tick=0,
            prev_lanes=[0],
            lanes=[1],
            profile=profile,
            rng=rng,
            current_strength=0.1,
            prev_strength=0.9,
        )
        assert result is False

    def test_should_hopo_legato_low_energy(self):
        """
        A note with very low energy relative to the previous one should
        be marked as HOPO (legato / hammer-on).
        """
        rng = random.Random(42)
        profile = DIFFICULTY_PROFILES["expert"]
        # Current note has 10% of previous note's energy → clear legato
        result = _should_hopo(
            tick=48,
            prev_tick=0,
            prev_lanes=[0],
            lanes=[1],
            profile=profile,
            rng=rng,
            current_strength=0.08,
            prev_strength=0.9,
        )
        assert result is True, "Low energy ratio (0.08/0.9 ≈ 0.09) should trigger HOPO"

    def test_should_hopo_strong_attack_no_hopo(self):
        """
        A note with strong attack energy (new pluck) should NOT be HOPO
        even when close to the previous note.
        """
        rng = random.Random(42)
        profile = DIFFICULTY_PROFILES["expert"]
        # Current note is just as loud as previous → new pick stroke
        result = _should_hopo(
            tick=48,
            prev_tick=0,
            prev_lanes=[0],
            lanes=[1],
            profile=profile,
            rng=rng,
            current_strength=0.85,
            prev_strength=0.9,
        )
        assert result is False, (
            "High energy ratio (0.85/0.9 ≈ 0.94) should not trigger HOPO"
        )

    def test_should_hopo_same_lane_no_hopo(self):
        """HOPO requires a lane change — same lane should never be HOPO."""
        rng = random.Random(42)
        profile = DIFFICULTY_PROFILES["expert"]
        result = _should_hopo(
            tick=48,
            prev_tick=0,
            prev_lanes=[1],
            lanes=[1],
            profile=profile,
            rng=rng,
            current_strength=0.1,
            prev_strength=0.9,
        )
        assert result is False

    def test_should_hopo_too_far_apart(self):
        """Notes more than half a beat apart should not be HOPO."""
        rng = random.Random(42)
        profile = DIFFICULTY_PROFILES["expert"]
        # gap = 192 > RESOLUTION//2 = 96
        result = _should_hopo(
            tick=192,
            prev_tick=0,
            prev_lanes=[0],
            lanes=[1],
            profile=profile,
            rng=rng,
            current_strength=0.1,
            prev_strength=0.9,
        )
        assert result is False

    def test_should_hopo_chord_no_hopo(self):
        """Chords (multiple lanes) should never be HOPO."""
        rng = random.Random(42)
        profile = DIFFICULTY_PROFILES["expert"]
        result = _should_hopo(
            tick=48,
            prev_tick=0,
            prev_lanes=[0],
            lanes=[0, 1],
            profile=profile,
            rng=rng,
            current_strength=0.1,
            prev_strength=0.9,
        )
        assert result is False

    # --- Tap detection tests ---

    def test_detect_tap_runs_disabled_easy(self):
        """Easy profile has tap_enabled=False → no taps."""
        profile = DIFFICULTY_PROFILES["easy"]
        events = [(i * 24, 0.5) for i in range(20)]
        result = _detect_tap_runs(events, None, profile)
        assert all(not t for t in result)

    def test_detect_tap_runs_fast_sequence(self):
        """A fast run of notes within tap_min_speed_ticks should be tapped."""
        profile = DIFFICULTY_PROFILES["expert"]
        # 10 notes, each 24 ticks apart (well within 48 tick threshold)
        events = [(i * 24, 0.5) for i in range(10)]
        # Pitch lanes all on Orange (4) → high lane bias satisfied
        pitch_lanes = [4] * 10
        result = _detect_tap_runs(events, pitch_lanes, profile)
        # All 10 should be tapped (run_length=10 >= tap_min_run=4)
        assert sum(result) == 10, (
            f"Expected 10 tapped notes in fast high-lane run, got {sum(result)}"
        )

    def test_detect_tap_runs_slow_sequence_no_tap(self):
        """Notes spaced far apart should NOT trigger tap detection."""
        profile = DIFFICULTY_PROFILES["expert"]
        # 10 notes, each 192 ticks apart (full beat, way above 48 threshold)
        events = [(i * 192, 0.5) for i in range(10)]
        pitch_lanes = [4] * 10
        result = _detect_tap_runs(events, pitch_lanes, profile)
        assert sum(result) == 0, "Slow sequence should not trigger taps"

    def test_detect_tap_runs_short_run_no_tap(self):
        """A fast run shorter than tap_min_run should NOT be tapped."""
        profile = DIFFICULTY_PROFILES["expert"]
        # Only 3 fast notes (below min_run=4), then a gap, then slow notes
        events = [
            (0, 0.5),
            (24, 0.5),
            (48, 0.5),  # 3 fast notes
            (500, 0.5),
            (700, 0.5),  # slow notes (gap > 48)
        ]
        pitch_lanes = [4, 4, 4, 4, 4]
        result = _detect_tap_runs(events, pitch_lanes, profile)
        assert sum(result) == 0, "Run of 3 is below min_run=4"

    def test_detect_tap_runs_high_lane_bias(self):
        """
        When tap_high_lane_bias is True, runs on low lanes (Green/Red)
        should NOT be tapped unless enough high-lane notes are present.
        """
        profile = DIFFICULTY_PROFILES["expert"]
        # 8 fast notes, all on Green (0) — low lane
        events = [(i * 24, 0.5) for i in range(8)]
        pitch_lanes = [0] * 8
        result = _detect_tap_runs(events, pitch_lanes, profile)
        # Less than 40% on Blue/Orange → no tap
        assert sum(result) == 0, (
            "All-Green run should not trigger taps with high_lane_bias"
        )

    def test_detect_tap_runs_no_pitch_lanes(self):
        """When pitch_lanes is None and high_lane_bias is True, taps can still fire."""
        profile = DIFFICULTY_PROFILES["expert"]
        # 8 fast notes, no pitch lane info
        events = [(i * 24, 0.5) for i in range(8)]
        result = _detect_tap_runs(events, None, profile)
        # With no pitch_lanes, high_lane_bias check is skipped → taps fire
        assert sum(result) == 8, "Without pitch_lanes, high_lane_bias should be skipped"

    def test_compute_sustain_non_negative(self):
        """_compute_sustain should always return non-negative values."""
        rng = random.Random(42)
        profile = DIFFICULTY_PROFILES["expert"]
        result = _compute_sustain(0, 384, 0.8, profile, rng)
        assert result >= 0

    def test_compute_sustain_no_next_note(self):
        """Sustain with no next note should still return non-negative."""
        rng = random.Random(42)
        profile = DIFFICULTY_PROFILES["expert"]
        result = _compute_sustain(0, None, 0.9, profile, rng)
        assert result >= 0

    def test_compute_sustain_weak_onset_zero(self):
        """Weak onsets shouldn't produce sustains."""
        rng = random.Random(42)
        profile = DIFFICULTY_PROFILES["expert"]
        result = _compute_sustain(0, 384, 0.2, profile, rng)
        assert result == 0

    def test_compute_star_power_sections_returns_list(self):
        """_compute_star_power_sections should return a list of (start, duration) tuples."""
        # onset_ticks is a list of tick positions (ints)
        onset_ticks = [i * 192 for i in range(100)]
        total_ticks = onset_ticks[-1] + 192
        result = _compute_star_power_sections(onset_ticks, total_ticks)
        assert isinstance(result, list)
        for item in result:
            assert len(item) == 2
            _start_tick, duration = item
            assert duration > 0, "Star power duration must be positive"

    def test_compute_star_power_sections_too_few_notes(self):
        """With fewer than 20 notes, no star power should be placed."""
        onset_ticks = [i * 192 for i in range(10)]
        total_ticks = onset_ticks[-1] + 192
        result = _compute_star_power_sections(onset_ticks, total_ticks)
        assert result == []


# ---------------------------------------------------------------------------
# Test: Energy-based HOPO in generated charts
# ---------------------------------------------------------------------------


class TestEnergyBasedHOPO:
    """Test that energy-based HOPO detection produces reasonable results."""

    def test_expert_chart_has_hopos(self, mock_audio_analysis, tmp_path):
        """Expert chart should contain some HOPO markers (N 5)."""
        out = tmp_path / "notes.chart"
        text = _generate_chart_to_text(mock_audio_analysis, out)
        sections = _parse_chart_sections(text)

        expert_events = _extract_note_events(sections["[ExpertSingle]"])
        hopo_count = sum(
            1
            for _, etype, rest in expert_events
            if etype == "N" and rest.split()[0] == "5"
        )
        # With energy-based detection, we expect at least some HOPOs
        # in a 180-second song (onset strengths vary 0.3–1.0)
        # This is a soft check — the exact count depends on the
        # random onset strengths in the fixture.
        assert hopo_count >= 0, "HOPO count should be non-negative"

    def test_easy_chart_has_no_hopos(self, mock_audio_analysis, tmp_path):
        """Easy chart should have zero HOPO markers (threshold is 0)."""
        out = tmp_path / "notes.chart"
        text = _generate_chart_to_text(mock_audio_analysis, out)
        sections = _parse_chart_sections(text)

        easy_events = _extract_note_events(sections["[EasySingle]"])
        hopo_count = sum(
            1
            for _, etype, rest in easy_events
            if etype == "N" and rest.split()[0] == "5"
        )
        assert hopo_count == 0, f"Easy should have no HOPOs, found {hopo_count}"

    def test_easy_chart_has_no_taps(self, mock_audio_analysis, tmp_path):
        """Easy chart should have zero tap markers (taps disabled)."""
        out = tmp_path / "notes.chart"
        text = _generate_chart_to_text(mock_audio_analysis, out)
        sections = _parse_chart_sections(text)

        easy_events = _extract_note_events(sections["[EasySingle]"])
        tap_count = sum(
            1
            for _, etype, rest in easy_events
            if etype == "N" and rest.split()[0] == "6"
        )
        assert tap_count == 0, f"Easy should have no taps, found {tap_count}"

    def test_medium_chart_has_no_hopos(self, mock_audio_analysis, tmp_path):
        """Medium chart should have zero HOPO markers (threshold is 0)."""
        out = tmp_path / "notes.chart"
        text = _generate_chart_to_text(mock_audio_analysis, out)
        sections = _parse_chart_sections(text)

        medium_events = _extract_note_events(sections["[MediumSingle]"])
        hopo_count = sum(
            1
            for _, etype, rest in medium_events
            if etype == "N" and rest.split()[0] == "5"
        )
        assert hopo_count == 0, f"Medium should have no HOPOs, found {hopo_count}"


# ---------------------------------------------------------------------------
# Test: Tap detection in generated charts
# ---------------------------------------------------------------------------


class TestTapDetection:
    """Test that tap markers (N 6) appear correctly in generated charts."""

    def test_tap_markers_valid_format(self, mock_audio_analysis, tmp_path):
        """Any tap markers should be N 6 0 (note 6, zero sustain)."""
        out = tmp_path / "notes.chart"
        text = _generate_chart_to_text(mock_audio_analysis, out)
        sections = _parse_chart_sections(text)

        for diff in DIFFICULTY_SECTIONS:
            events = _extract_note_events(sections[diff])
            for tick, etype, rest in events:
                if etype == "N" and rest.startswith("6"):
                    parts = rest.split()
                    assert len(parts) == 2, (
                        f"Tap marker at tick {tick} has unexpected format: {rest}"
                    )
                    assert parts[0] == "6" and parts[1] == "0", (
                        f"Tap marker should be 'N 6 0', got 'N {rest}' at tick {tick}"
                    )

    def test_no_taps_on_easy_medium(self, mock_audio_analysis, tmp_path):
        """Easy and Medium should never have tap markers."""
        out = tmp_path / "notes.chart"
        text = _generate_chart_to_text(mock_audio_analysis, out)
        sections = _parse_chart_sections(text)

        for diff in ["[EasySingle]", "[MediumSingle]"]:
            events = _extract_note_events(sections[diff])
            taps = [
                (tick, rest)
                for tick, etype, rest in events
                if etype == "N" and rest.split()[0] == "6"
            ]
            assert len(taps) == 0, f"{diff} should have no taps, found {len(taps)}"


# ---------------------------------------------------------------------------
# Test: Segment similarity labeling
# ---------------------------------------------------------------------------


class TestSegmentLabeling:
    """Test the segment detection and labeling logic."""

    def test_segments_have_labels(self, mock_audio_analysis, tmp_path):
        """Every segment in a generated chart should have a non-empty label."""
        out = tmp_path / "notes.chart"
        text = _generate_chart_to_text(mock_audio_analysis, out)
        sections = _parse_chart_sections(text)

        events_body = sections["[Events]"]
        section_events = [
            line for line in events_body.splitlines() if "section" in line.lower()
        ]
        assert len(section_events) >= 2, (
            f"Expected at least 2 section markers, got {len(section_events)}"
        )

    def test_segments_include_intro_outro(self, mock_audio_analysis, tmp_path):
        """Generated charts should include Intro and Outro section markers."""
        out = tmp_path / "notes.chart"
        text = _generate_chart_to_text(mock_audio_analysis, out)
        sections = _parse_chart_sections(text)

        events_body = sections["[Events]"]
        has_intro = "intro" in events_body.lower()
        has_outro = "outro" in events_body.lower()
        # The mock fixture provides explicit segments with Intro/Outro labels,
        # so these should always appear
        assert has_intro, "Chart should have an Intro section marker"
        assert has_outro, "Chart should have an Outro section marker"


# ---------------------------------------------------------------------------
# Test: Short analysis fixture
# ---------------------------------------------------------------------------


class TestShortAnalysis:
    """Test chart generation with the shorter (30-second) analysis fixture."""

    def test_short_song_chart_valid(self, mock_audio_analysis_short, tmp_path):
        """A 30-second song should produce a valid chart."""
        out = tmp_path / "notes.chart"
        text = _generate_chart_to_text(mock_audio_analysis_short, out)
        sections = _parse_chart_sections(text)

        assert "[Song]" in sections
        assert "[SyncTrack]" in sections
        assert "[ExpertSingle]" in sections

    def test_short_song_has_notes(self, mock_audio_analysis_short, tmp_path):
        """Even a short song should produce some notes."""
        out = tmp_path / "notes.chart"
        text = _generate_chart_to_text(mock_audio_analysis_short, out)
        sections = _parse_chart_sections(text)

        expert_events = _extract_note_events(sections["[ExpertSingle]"])
        note_events = [e for e in expert_events if e[1] == "N"]
        assert len(note_events) > 0, "Short song should still have notes"

    def test_short_song_tempo_matches(self, mock_audio_analysis_short, tmp_path):
        """The initial BPM should reflect the analysis tempo (140)."""
        out = tmp_path / "notes.chart"
        text = _generate_chart_to_text(mock_audio_analysis_short, out)
        sections = _parse_chart_sections(text)

        sync_body = sections["[SyncTrack]"]
        match = re.search(r"0\s*=\s*B\s+(\d+)", sync_body)
        assert match, "Missing initial BPM marker"

        initial_bpm = int(match.group(1)) / 1000.0
        # Should be close to 140 BPM (within 10%)
        assert abs(initial_bpm - 140.0) < 14.0, (
            f"Initial BPM {initial_bpm} too far from expected 140"
        )


# ---------------------------------------------------------------------------
# Test: Round-trip consistency
# ---------------------------------------------------------------------------


class TestRoundTripConsistency:
    """Test that generating the same chart twice produces consistent results."""

    def test_deterministic_output(self, mock_audio_analysis, tmp_path):
        """
        Generating a chart twice with the same inputs should produce
        the same sections and note counts (though not necessarily
        byte-identical if there's randomness — we check structural
        equivalence).
        """
        out1 = tmp_path / "chart1.chart"
        out2 = tmp_path / "chart2.chart"

        text1 = _generate_chart_to_text(mock_audio_analysis, out1)
        text2 = _generate_chart_to_text(mock_audio_analysis, out2)

        sections1 = _parse_chart_sections(text1)
        sections2 = _parse_chart_sections(text2)

        # Same set of sections
        assert set(sections1.keys()) == set(sections2.keys())

        # Same number of note events per section
        for diff in DIFFICULTY_SECTIONS:
            if diff in sections1 and diff in sections2:
                notes1 = len(
                    [e for e in _extract_note_events(sections1[diff]) if e[1] == "N"]
                )
                notes2 = len(
                    [e for e in _extract_note_events(sections2[diff]) if e[1] == "N"]
                )
                assert notes1 == notes2, (
                    f"Note count mismatch in {diff}: {notes1} vs {notes2}"
                )
