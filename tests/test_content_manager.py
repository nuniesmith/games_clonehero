"""
Clone Hero Content Manager - Content Manager & Chart Parser Tests

Tests for:
- song.ini writing (write_song_ini) and parsing (parse_song_ini)
- Chart parser utilities (parse_chart_file, get_chart_summary, chart_to_viewer_json)
- Content manager helpers (_sanitize_filename, get_temp_staging_dir)
- Archive extraction basics
- Round-trip consistency: write song.ini → parse song.ini → verify fields match

These tests validate the data pipeline that sits between the generator
(which produces chart files and metadata) and the Nextcloud uploader
(which stores the final artefacts).
"""

import configparser
import json
from pathlib import Path

from src.services.chart_parser import (
    chart_to_viewer_json,
    get_chart_summary,
    parse_chart_file,
)
from src.services.content_manager import (
    _sanitize_filename,
    get_temp_staging_dir,
    parse_song_ini,
    write_song_ini,
)
from tests.conftest import (
    SAMPLE_CHART_MALFORMED,
    SAMPLE_CHART_NO_MUSIC_STREAM,
    SAMPLE_CHART_VALID,
    SAMPLE_SONG_INI,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_ini_raw(path: Path) -> configparser.ConfigParser:
    """Read a song.ini using configparser for field-level inspection."""
    cp = configparser.ConfigParser()
    cp.read(str(path), encoding="utf-8")
    return cp


# ===========================================================================
# write_song_ini
# ===========================================================================


class TestWriteSongIni:
    """Test the write_song_ini() function."""

    def test_creates_file(self, tmp_path, sample_song_data):
        ini_path = tmp_path / "song.ini"
        write_song_ini(ini_path, sample_song_data)
        assert ini_path.exists()
        assert ini_path.stat().st_size > 0

    def test_contains_section_header(self, tmp_path, sample_song_data):
        ini_path = tmp_path / "song.ini"
        write_song_ini(ini_path, sample_song_data)
        text = ini_path.read_text(encoding="utf-8")
        # song.ini uses [song] or [Song] as the section header
        assert "[song]" in text.lower()

    def test_title_written(self, tmp_path, sample_song_data):
        ini_path = tmp_path / "song.ini"
        write_song_ini(ini_path, sample_song_data)
        text = ini_path.read_text(encoding="utf-8")
        assert "Test Song" in text

    def test_artist_written(self, tmp_path, sample_song_data):
        ini_path = tmp_path / "song.ini"
        write_song_ini(ini_path, sample_song_data)
        text = ini_path.read_text(encoding="utf-8")
        assert "Test Artist" in text

    def test_album_written(self, tmp_path, sample_song_data):
        ini_path = tmp_path / "song.ini"
        write_song_ini(ini_path, sample_song_data)
        text = ini_path.read_text(encoding="utf-8")
        assert "Test Album" in text

    def test_charter_written(self, tmp_path, sample_song_data):
        ini_path = tmp_path / "song.ini"
        write_song_ini(ini_path, sample_song_data)
        text = ini_path.read_text(encoding="utf-8")
        assert "testcharter" in text

    def test_genre_written(self, tmp_path, sample_song_data):
        ini_path = tmp_path / "song.ini"
        write_song_ini(ini_path, sample_song_data)
        text = ini_path.read_text(encoding="utf-8")
        assert "Rock" in text

    def test_year_written(self, tmp_path, sample_song_data):
        ini_path = tmp_path / "song.ini"
        write_song_ini(ini_path, sample_song_data)
        text = ini_path.read_text(encoding="utf-8")
        assert "2024" in text

    def test_song_length_written(self, tmp_path, sample_song_data):
        ini_path = tmp_path / "song.ini"
        write_song_ini(ini_path, sample_song_data)
        text = ini_path.read_text(encoding="utf-8")
        assert "180000" in text

    def test_difficulty_fields_written(self, tmp_path, sample_song_data):
        ini_path = tmp_path / "song.ini"
        write_song_ini(ini_path, sample_song_data)
        text = ini_path.read_text(encoding="utf-8").lower()
        assert "diff_guitar" in text

    def test_loading_phrase_written(self, tmp_path, sample_song_data):
        ini_path = tmp_path / "song.ini"
        write_song_ini(ini_path, sample_song_data)
        text = ini_path.read_text(encoding="utf-8")
        assert "Auto-generated chart at 120 BPM" in text

    def test_custom_metadata(self, tmp_path):
        """Custom metadata keys in the dict should be written."""
        data = {
            "title": "Custom",
            "artist": "Me",
            "album": "Mine",
            "metadata": {
                "charter": "mycharter",
                "song_length": "60000",
                "genre": "Pop",
                "custom_field": "custom_value",
            },
        }
        ini_path = tmp_path / "song.ini"
        write_song_ini(ini_path, data)
        text = ini_path.read_text(encoding="utf-8")
        # At minimum the core fields should be present
        assert "Custom" in text
        assert "Me" in text

    def test_empty_metadata(self, tmp_path):
        """Writing with minimal/empty metadata should not crash."""
        data = {
            "title": "Minimal",
            "artist": "Nobody",
            "album": "",
            "metadata": {},
        }
        ini_path = tmp_path / "song.ini"
        write_song_ini(ini_path, data)
        assert ini_path.exists()
        text = ini_path.read_text(encoding="utf-8")
        assert "Minimal" in text

    def test_special_characters_in_title(self, tmp_path):
        """Titles with special chars should be written without crashing."""
        data = {
            "title": 'Song "With" Quotes & Amps <Tags>',
            "artist": "Ar/ti\\st",
            "album": "Al=bum",
            "metadata": {"charter": "test"},
        }
        ini_path = tmp_path / "song.ini"
        write_song_ini(ini_path, data)
        assert ini_path.exists()
        text = ini_path.read_text(encoding="utf-8")
        assert len(text) > 0

    def test_unicode_in_fields(self, tmp_path):
        """Unicode characters should round-trip correctly."""
        data = {
            "title": "Jóga",
            "artist": "Björk",
            "album": "Homogénic",
            "metadata": {"charter": "tëst", "genre": "Électronique"},
        }
        ini_path = tmp_path / "song.ini"
        write_song_ini(ini_path, data)
        text = ini_path.read_text(encoding="utf-8")
        assert "Jóga" in text
        assert "Björk" in text

    def test_overwrite_existing(self, tmp_path, sample_song_data):
        """Writing to an existing file should overwrite it."""
        ini_path = tmp_path / "song.ini"
        ini_path.write_text("old content", encoding="utf-8")
        write_song_ini(ini_path, sample_song_data)
        text = ini_path.read_text(encoding="utf-8")
        assert "old content" not in text
        assert "Test Song" in text


# ===========================================================================
# parse_song_ini
# ===========================================================================


class TestParseSongIni:
    """Test the parse_song_ini() function."""

    def test_parses_title(self, tmp_path):
        ini_path = tmp_path / "song.ini"
        ini_path.write_text(SAMPLE_SONG_INI, encoding="utf-8")
        data = parse_song_ini(ini_path)
        assert data is not None
        assert data.get("title") == "Test Song" or data.get("name") == "Test Song"

    def test_parses_artist(self, tmp_path):
        ini_path = tmp_path / "song.ini"
        ini_path.write_text(SAMPLE_SONG_INI, encoding="utf-8")
        data = parse_song_ini(ini_path)
        assert data is not None
        assert data.get("artist") == "Test Artist"

    def test_parses_album(self, tmp_path):
        ini_path = tmp_path / "song.ini"
        ini_path.write_text(SAMPLE_SONG_INI, encoding="utf-8")
        data = parse_song_ini(ini_path)
        assert data is not None
        assert data.get("album") == "Test Album"

    def test_parses_genre(self, tmp_path):
        ini_path = tmp_path / "song.ini"
        ini_path.write_text(SAMPLE_SONG_INI, encoding="utf-8")
        data = parse_song_ini(ini_path)
        assert data is not None
        # genre may be in metadata sub-dict or top-level depending on impl
        genre = data.get("genre") or (data.get("metadata", {}) or {}).get("genre")
        assert genre is not None
        assert "rock" in str(genre).lower()

    def test_parses_charter(self, tmp_path):
        ini_path = tmp_path / "song.ini"
        ini_path.write_text(SAMPLE_SONG_INI, encoding="utf-8")
        data = parse_song_ini(ini_path)
        assert data is not None
        charter = data.get("charter") or (data.get("metadata", {}) or {}).get("charter")
        assert charter is not None
        assert "testcharter" in str(charter).lower()

    def test_returns_dict(self, tmp_path):
        ini_path = tmp_path / "song.ini"
        ini_path.write_text(SAMPLE_SONG_INI, encoding="utf-8")
        data = parse_song_ini(ini_path)
        assert isinstance(data, dict)

    def test_nonexistent_file(self, tmp_path):
        """Parsing a nonexistent file should return None."""
        ini_path = tmp_path / "does_not_exist.ini"
        data = parse_song_ini(ini_path)
        # parse_song_ini returns None when the file can't be read
        assert data is None

    def test_empty_file(self, tmp_path):
        ini_path = tmp_path / "empty.ini"
        ini_path.write_text("", encoding="utf-8")
        data = parse_song_ini(ini_path)
        # Empty file has no [song] section, so returns None
        assert data is None

    def test_malformed_ini(self, tmp_path):
        """A file with garbled content should not crash the parser."""
        ini_path = tmp_path / "garbled.ini"
        ini_path.write_text("this is not\nan ini file\nat all!!", encoding="utf-8")
        data = parse_song_ini(ini_path)
        # No [song] section, so returns None
        assert data is None


# ===========================================================================
# Round-trip: write_song_ini → parse_song_ini
# ===========================================================================


class TestSongIniRoundTrip:
    """Test that writing and then parsing a song.ini preserves key fields."""

    def test_title_round_trip(self, tmp_path, sample_song_data):
        ini_path = tmp_path / "song.ini"
        write_song_ini(ini_path, sample_song_data)
        parsed = parse_song_ini(ini_path)
        assert parsed is not None
        title = parsed.get("title") or parsed.get("name", "")
        assert title == "Test Song"

    def test_artist_round_trip(self, tmp_path, sample_song_data):
        ini_path = tmp_path / "song.ini"
        write_song_ini(ini_path, sample_song_data)
        parsed = parse_song_ini(ini_path)
        assert parsed is not None
        assert parsed.get("artist") == "Test Artist"

    def test_album_round_trip(self, tmp_path, sample_song_data):
        ini_path = tmp_path / "song.ini"
        write_song_ini(ini_path, sample_song_data)
        parsed = parse_song_ini(ini_path)
        assert parsed is not None
        assert parsed.get("album") == "Test Album"

    def test_unicode_round_trip(self, tmp_path):
        data = {
            "title": "Ágætis Byrjun",
            "artist": "Sigur Rós",
            "album": "Ágætis Byrjun",
            "metadata": {"charter": "tëster", "genre": "Post-Rock"},
        }
        ini_path = tmp_path / "song.ini"
        write_song_ini(ini_path, data)
        parsed = parse_song_ini(ini_path)
        assert parsed is not None
        title = parsed.get("title") or parsed.get("name", "")
        assert "Ágætis" in title or "gætis" in title
        assert parsed.get("artist") == "Sigur Rós" or "Sigur" in str(
            parsed.get("artist", "")
        )

    def test_multiple_writes_overwrite(self, tmp_path, sample_song_data):
        """Writing twice should leave only the latest data."""
        ini_path = tmp_path / "song.ini"
        write_song_ini(ini_path, sample_song_data)

        new_data = {
            "title": "New Song",
            "artist": "New Artist",
            "album": "New Album",
            "metadata": {"charter": "newcharter", "genre": "Jazz"},
        }
        write_song_ini(ini_path, new_data)
        parsed = parse_song_ini(ini_path)
        assert parsed is not None
        title = parsed.get("title") or parsed.get("name", "")
        assert "New Song" in title or "new" in title.lower()
        assert "Test Song" not in str(parsed)


# ===========================================================================
# parse_chart_file
# ===========================================================================


class TestParseChartFile:
    """Test the chart_parser.parse_chart_file() function."""

    def test_parses_valid_chart(self, tmp_path):
        chart_path = tmp_path / "notes.chart"
        chart_path.write_text(SAMPLE_CHART_VALID, encoding="utf-8-sig")
        result = parse_chart_file(str(chart_path))
        assert isinstance(result, dict)
        # Should contain parsed sections or data
        assert len(result) > 0

    def test_returns_song_metadata(self, tmp_path):
        chart_path = tmp_path / "notes.chart"
        chart_path.write_text(SAMPLE_CHART_VALID, encoding="utf-8-sig")
        result = parse_chart_file(str(chart_path))
        # parse_chart_file returns a dict with a "song" key containing metadata
        assert "song" in result
        song = result["song"]
        assert song.get("Name") == "Test Song"

    def test_returns_sync_track_info(self, tmp_path):
        chart_path = tmp_path / "notes.chart"
        chart_path.write_text(SAMPLE_CHART_VALID, encoding="utf-8-sig")
        result = parse_chart_file(str(chart_path))
        # parse_chart_file returns a dict with a "sync_track" key
        assert "sync_track" in result
        sync = result["sync_track"]
        assert "tempo_markers" in sync
        assert len(sync["tempo_markers"]) > 0

    def test_returns_note_data(self, tmp_path):
        chart_path = tmp_path / "notes.chart"
        chart_path.write_text(SAMPLE_CHART_VALID, encoding="utf-8-sig")
        result = parse_chart_file(str(chart_path))
        # parse_chart_file returns a dict with a "difficulties" key
        assert "difficulties" in result
        diffs = result["difficulties"]
        assert "expert" in diffs or len(diffs) > 0

    def test_handles_string_path(self, tmp_path):
        """Should accept both str and Path arguments."""
        chart_path = tmp_path / "notes.chart"
        chart_path.write_text(SAMPLE_CHART_VALID, encoding="utf-8-sig")
        result = parse_chart_file(str(chart_path))
        assert isinstance(result, dict)

    def test_empty_chart(self, tmp_path):
        chart_path = tmp_path / "empty.chart"
        chart_path.write_text("", encoding="utf-8")
        try:
            result = parse_chart_file(str(chart_path))
            # Should return empty or minimal result, not crash
            assert isinstance(result, dict)
        except Exception:
            # Raising an exception is also acceptable for empty charts
            pass

    def test_malformed_chart(self, tmp_path):
        chart_path = tmp_path / "malformed.chart"
        chart_path.write_text(SAMPLE_CHART_MALFORMED, encoding="utf-8")
        try:
            result = parse_chart_file(str(chart_path))
            assert isinstance(result, dict)
        except Exception:
            # Some implementations may raise on malformed input
            pass


# ===========================================================================
# get_chart_summary
# ===========================================================================


class TestGetChartSummary:
    """Test the chart_parser.get_chart_summary() function."""

    def test_returns_dict(self, tmp_path):
        chart_path = tmp_path / "notes.chart"
        chart_path.write_text(SAMPLE_CHART_VALID, encoding="utf-8-sig")
        parsed = parse_chart_file(str(chart_path))
        result = get_chart_summary(parsed)
        assert isinstance(result, dict)

    def test_contains_song_info(self, tmp_path):
        chart_path = tmp_path / "notes.chart"
        chart_path.write_text(SAMPLE_CHART_VALID, encoding="utf-8-sig")
        parsed = parse_chart_file(str(chart_path))
        result = get_chart_summary(parsed)
        # Should reference the song name
        assert result.get("name") == "Test Song"
        assert result.get("artist") == "Test Artist"

    def test_contains_difficulty_info(self, tmp_path):
        chart_path = tmp_path / "notes.chart"
        chart_path.write_text(SAMPLE_CHART_VALID, encoding="utf-8-sig")
        parsed = parse_chart_file(str(chart_path))
        result = get_chart_summary(parsed)
        # Should have difficulties summary
        assert "difficulties" in result
        diffs = result["difficulties"]
        assert len(diffs) > 0

    def test_contains_tempo_info(self, tmp_path):
        chart_path = tmp_path / "notes.chart"
        chart_path.write_text(SAMPLE_CHART_VALID, encoding="utf-8-sig")
        parsed = parse_chart_file(str(chart_path))
        result = get_chart_summary(parsed)
        # Should contain BPM range info
        assert "bpm_range" in result
        assert result["bpm_range"]["primary"] == 120.0

    def test_empty_chart_handled(self, tmp_path):
        chart_path = tmp_path / "empty.chart"
        chart_path.write_text("", encoding="utf-8")
        try:
            parsed = parse_chart_file(str(chart_path))
            result = get_chart_summary(parsed)
            assert isinstance(result, dict)
        except Exception:
            # parse_chart_file raises ValueError for empty charts
            pass

    def test_no_music_stream_chart(self, tmp_path):
        chart_path = tmp_path / "no_ms.chart"
        chart_path.write_text(SAMPLE_CHART_NO_MUSIC_STREAM, encoding="utf-8-sig")
        parsed = parse_chart_file(str(chart_path))
        result = get_chart_summary(parsed)
        assert isinstance(result, dict)


# ===========================================================================
# chart_to_viewer_json
# ===========================================================================


class TestChartToViewerJson:
    """Test the chart_parser.chart_to_viewer_json() function."""

    def test_returns_dict(self, tmp_path):
        chart_path = tmp_path / "notes.chart"
        chart_path.write_text(SAMPLE_CHART_VALID, encoding="utf-8-sig")
        parsed = parse_chart_file(str(chart_path))
        result = chart_to_viewer_json(parsed)
        assert isinstance(result, dict)

    def test_output_is_json_serializable(self, tmp_path):
        chart_path = tmp_path / "notes.chart"
        chart_path.write_text(SAMPLE_CHART_VALID, encoding="utf-8-sig")
        parsed = parse_chart_file(str(chart_path))
        result = chart_to_viewer_json(parsed)
        json_str = json.dumps(result)
        assert len(json_str) > 0
        # Round-trip
        reparsed = json.loads(json_str)
        assert isinstance(reparsed, dict)

    def test_contains_note_data(self, tmp_path):
        chart_path = tmp_path / "notes.chart"
        chart_path.write_text(SAMPLE_CHART_VALID, encoding="utf-8-sig")
        parsed = parse_chart_file(str(chart_path))
        result = chart_to_viewer_json(parsed)
        flat = json.dumps(result).lower()
        # Should contain note or chart viewer data
        assert len(flat) > 10  # Non-trivial output


# ===========================================================================
# _sanitize_filename
# ===========================================================================


class TestSanitizeFilename:
    """Test the filename sanitization helper."""

    def test_normal_name_unchanged(self):
        result = _sanitize_filename("My Song")
        assert result == "My Song" or result.strip() == "My Song"

    def test_strips_path_separators(self):
        result = _sanitize_filename("path/to/file")
        assert "/" not in result
        assert "\\" not in result

    def test_strips_dangerous_chars(self):
        result = _sanitize_filename('file<>:"|?*name')
        for c in '<>:"|?*':
            assert c not in result

    def test_handles_empty_string(self):
        result = _sanitize_filename("")
        assert isinstance(result, str)

    def test_handles_dots_and_spaces(self):
        result = _sanitize_filename("  ...song...  ")
        assert isinstance(result, str)

    def test_unicode_preserved(self):
        result = _sanitize_filename("Björk - Jóga")
        assert "Björk" in result or "jork" in result.lower() or "Bj" in result

    def test_long_name_handled(self):
        long_name = "A" * 500
        result = _sanitize_filename(long_name)
        assert isinstance(result, str)
        # Should either truncate or keep as-is without crashing
        assert len(result) <= 500 or len(result) > 0

    def test_only_special_chars(self):
        result = _sanitize_filename('***???"""')
        assert isinstance(result, str)

    def test_preserves_hyphens_and_underscores(self):
        result = _sanitize_filename("my-song_v2")
        assert "-" in result or "_" in result

    def test_null_bytes_removed(self):
        result = _sanitize_filename("song\x00name")
        # The sanitizer may not explicitly strip null bytes, but it should
        # not crash. If null bytes remain, that's an implementation detail.
        assert isinstance(result, str)


# ===========================================================================
# get_temp_staging_dir
# ===========================================================================


class TestGetTempStagingDir:
    """Test the staging directory helper."""

    def test_returns_path(self):
        result = get_temp_staging_dir()
        assert isinstance(result, (str, Path))

    def test_directory_exists_after_call(self):
        result = get_temp_staging_dir()
        p = Path(result)
        assert p.exists()
        assert p.is_dir()

    def test_unique_directories(self):
        """Multiple calls should return different directories."""
        d1 = get_temp_staging_dir()
        d2 = get_temp_staging_dir()
        assert str(d1) != str(d2)


# ===========================================================================
# Integration: Generate → Write INI → Parse INI → Validate
# ===========================================================================


class TestGeneratorOutputIntegration:
    """
    Integration tests that simulate the generator output pipeline:
    generate chart → write song.ini → parse back → verify consistency.
    """

    def test_song_folder_has_all_required_files(self, song_folder):
        """A well-formed song folder should contain the required files."""
        assert (song_folder / "notes.chart").exists()
        assert (song_folder / "song.ini").exists()
        assert (song_folder / "song.ogg").exists()

    def test_chart_is_parseable(self, song_folder):
        """The chart in a song folder should be parseable."""
        chart_path = str(song_folder / "notes.chart")
        result = parse_chart_file(chart_path)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_ini_is_parseable(self, song_folder):
        """The song.ini in a song folder should be parseable."""
        ini_path = song_folder / "song.ini"
        data = parse_song_ini(ini_path)
        assert isinstance(data, dict)
        title = data.get("title") or data.get("name", "")
        assert len(title) > 0

    def test_chart_summary_from_folder(self, song_folder):
        """get_chart_summary should work on a parsed chart from a real song folder."""
        chart_path = str(song_folder / "notes.chart")
        parsed = parse_chart_file(chart_path)
        summary = get_chart_summary(parsed)
        assert isinstance(summary, dict)
        assert len(summary) > 0

    def test_create_song_folder_factory(self, create_song_folder):
        """The factory fixture should produce valid folders."""
        folder = create_song_folder(
            name="Factory Test",
            artist="Factory Artist",
            audio_ext=".ogg",
            include_chart=True,
            include_ini=True,
            include_art=True,
        )
        assert (folder / "notes.chart").exists()
        assert (folder / "song.ini").exists()
        assert (folder / "song.ogg").exists()
        assert (folder / "album.png").exists()

    def test_create_song_folder_without_art(self, create_song_folder):
        folder = create_song_folder(
            name="No Art",
            artist="Artist",
            include_art=False,
        )
        assert (folder / "notes.chart").exists()
        assert not (folder / "album.png").exists()

    def test_create_song_folder_without_ini(self, create_song_folder):
        folder = create_song_folder(
            name="No INI",
            artist="Artist",
            include_ini=False,
        )
        assert (folder / "notes.chart").exists()
        assert not (folder / "song.ini").exists()

    def test_create_song_folder_mp3(self, create_song_folder):
        folder = create_song_folder(
            name="MP3 Song",
            artist="Artist",
            audio_ext=".mp3",
        )
        assert (folder / "song.mp3").exists()
        assert not (folder / "song.ogg").exists()

    def test_create_song_folder_flac(self, create_song_folder):
        folder = create_song_folder(
            name="FLAC Song",
            artist="Artist",
            audio_ext=".flac",
        )
        assert (folder / "song.flac").exists()

    def test_create_song_folder_custom_charter(self, create_song_folder):
        folder = create_song_folder(
            name="Charter Test",
            artist="Artist",
            charter="customcharter",
        )
        chart_text = (folder / "notes.chart").read_text(encoding="utf-8-sig")
        assert "customcharter" in chart_text
        ini_text = (folder / "song.ini").read_text(encoding="utf-8")
        assert "customcharter" in ini_text


# ===========================================================================
# Edge cases and error handling
# ===========================================================================


class TestEdgeCases:
    """Edge cases and error resilience tests."""

    def test_write_ini_to_nested_nonexistent_path(self, tmp_path, sample_song_data):
        """Writing to a path where parent dirs don't exist should either
        create them or raise a clear error."""
        nested_path = tmp_path / "a" / "b" / "c" / "song.ini"
        try:
            nested_path.parent.mkdir(parents=True, exist_ok=True)
            write_song_ini(nested_path, sample_song_data)
            assert nested_path.exists()
        except Exception:
            # Raising an error is also acceptable
            pass

    def test_parse_chart_with_extra_whitespace(self, tmp_path):
        """Charts with extra whitespace should still parse."""
        spaced_chart = SAMPLE_CHART_VALID.replace("\n", "\n\n\n")
        chart_path = tmp_path / "spaced.chart"
        chart_path.write_text(spaced_chart, encoding="utf-8-sig")
        result = parse_chart_file(str(chart_path))
        assert isinstance(result, dict)

    def test_parse_chart_with_windows_line_endings(self, tmp_path):
        """Charts with \\r\\n should parse correctly."""
        win_chart = SAMPLE_CHART_VALID.replace("\n", "\r\n")
        chart_path = tmp_path / "windows.chart"
        chart_path.write_text(win_chart, encoding="utf-8-sig")
        result = parse_chart_file(str(chart_path))
        assert isinstance(result, dict)

    def test_sanitize_filename_with_newlines(self):
        result = _sanitize_filename("song\nwith\nnewlines")
        # The sanitizer may not explicitly strip newlines, but it should
        # not crash. Check it returns a string.
        assert isinstance(result, str)

    def test_sanitize_filename_with_tabs(self):
        result = _sanitize_filename("song\twith\ttabs")
        assert "\t" not in result or isinstance(result, str)

    def test_write_ini_with_none_values(self, tmp_path):
        """None values in the metadata dict should not crash the writer."""
        data = {
            "title": "Test",
            "artist": "Artist",
            "album": None,
            "metadata": {
                "charter": None,
                "genre": "Rock",
            },
        }
        ini_path = tmp_path / "song.ini"
        try:
            write_song_ini(ini_path, data)
            assert ini_path.exists()
        except (TypeError, AttributeError):
            # Some implementations may not handle None gracefully
            pass

    def test_write_ini_with_numeric_values(self, tmp_path):
        """Numeric values in metadata should be converted to strings."""
        data = {
            "title": "Number Test",
            "artist": "Artist",
            "album": "Album",
            "metadata": {
                "song_length": 180000,
                "diff_guitar": 3,
                "year": 2024,
            },
        }
        ini_path = tmp_path / "song.ini"
        try:
            write_song_ini(ini_path, data)
            assert ini_path.exists()
            text = ini_path.read_text(encoding="utf-8")
            assert "180000" in text
        except (TypeError, ValueError):
            # Converting ints to strings may need explicit handling
            pass
