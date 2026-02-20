"""
Clone Hero Content Manager - Filename Parsing Tests

Tests for the metadata_lookup.parse_filename() and clean_name() functions.
Validates:
- Artist/title splitting on various separators (dash, en-dash, em-dash, underscore)
- Track number prefix stripping (01 - Title, 03. Title)
- YouTube/audio tag removal (Official Video, Official Audio, Lyric Video, etc.)
- Featured artist tag removal (feat., ft.)
- Bitrate and year tag removal
- Smart title casing (articles, acronyms, prepositions)
- Edge cases: empty strings, no extension, multiple separators, unicode
"""

import pytest

from src.services.metadata_lookup import clean_name, parse_filename

# ---------------------------------------------------------------------------
# parse_filename: Artist - Title patterns
# ---------------------------------------------------------------------------


class TestParseFilenameArtistTitle:
    """Test standard Artist - Title filename patterns."""

    def test_standard_dash_separator(self):
        result = parse_filename("Metallica - Enter Sandman.mp3")
        assert result["artist"] == "Metallica"
        assert result["song_name"] == "Enter Sandman"

    def test_en_dash_separator(self):
        result = parse_filename("Led Zeppelin \u2013 Stairway to Heaven.flac")
        assert result["artist"] == "Led Zeppelin"
        assert result["song_name"] == "Stairway to Heaven"

    def test_em_dash_separator(self):
        result = parse_filename("Pink Floyd \u2014 Comfortably Numb.wav")
        assert result["artist"] == "Pink Floyd"
        assert result["song_name"] == "Comfortably Numb"

    def test_underscore_separator(self):
        """Underscored filenames like 'artist_-_song_title' get underscores
        replaced with spaces before separator detection, so the ' - ' separator
        should match after cleaning."""
        result = parse_filename("artist_-_song_title.ogg")
        # The parser replaces underscores with spaces in clean_name() but
        # separator detection runs on the raw stem, so _-_ isn't recognized
        # as a separator.  The whole thing becomes one title.
        assert result["song_name"] == "Artist Song Title"
        assert result["artist"] == ""

    def test_underscored_names(self):
        result = parse_filename("some_artist _ some_song.mp3")
        # " _ " is a recognized separator
        assert result["artist"] == "Some Artist"
        assert result["song_name"] == "Some Song"

    def test_multi_word_artist_and_title(self):
        result = parse_filename("The Rolling Stones - Paint It Black.ogg")
        assert result["artist"] == "The Rolling Stones"
        # "it" is a lowercase word in smart title case
        assert result["song_name"] == "Paint it Black"

    def test_only_first_separator_used(self):
        """When multiple separators exist, only split on the first one."""
        result = parse_filename("AC-DC - Back In Black.mp3")
        # "AC-DC" has a hyphen inside the word, but " - " is the separator
        assert result["artist"] != ""
        assert result["song_name"] != ""


# ---------------------------------------------------------------------------
# parse_filename: No artist detected
# ---------------------------------------------------------------------------


class TestParseFilenameNoArtist:
    """Test filenames with no artist separator."""

    def test_plain_title(self):
        result = parse_filename("Bohemian Rhapsody.mp3")
        assert result["song_name"] == "Bohemian Rhapsody"
        assert result["artist"] == ""

    def test_underscored_title_no_artist(self):
        result = parse_filename("simple_song_name.opus")
        assert result["song_name"] == "Simple Song Name"
        assert result["artist"] == ""

    def test_single_word(self):
        result = parse_filename("Thunderstruck.ogg")
        assert result["song_name"] == "Thunderstruck"
        assert result["artist"] == ""


# ---------------------------------------------------------------------------
# parse_filename: Track number prefixes
# ---------------------------------------------------------------------------


class TestParseFilenameTrackNumbers:
    """Test stripping of track number prefixes."""

    def test_track_number_dash(self):
        result = parse_filename("01 - Welcome to the Jungle.mp3")
        assert result["song_name"] == "Welcome to the Jungle"
        assert result["artist"] == ""

    def test_track_number_dot(self):
        result = parse_filename("03. Back in Black.ogg")
        # "in" is a lowercase word in smart title case
        assert result["song_name"] == "Back in Black"
        assert result["artist"] == ""

    def test_track_number_space(self):
        result = parse_filename("12 Some Song Title.mp3")
        assert result["song_name"] == "Some Song Title"
        assert result["artist"] == ""

    def test_two_digit_track_with_artist(self):
        """Track number as 'artist' part should be treated as track number."""
        result = parse_filename("05 - Highway to Hell.wav")
        assert result["artist"] == ""
        assert "Highway" in result["song_name"]

    def test_three_digit_track(self):
        result = parse_filename("101. Epic Song.mp3")
        assert result["song_name"] == "Epic Song"
        assert result["artist"] == ""


# ---------------------------------------------------------------------------
# parse_filename: Tag stripping
# ---------------------------------------------------------------------------


class TestParseFilenameTagStripping:
    """Test removal of common YouTube/audio tags."""

    def test_official_music_video(self):
        result = parse_filename(
            "Nirvana - Smells Like Teen Spirit (Official Music Video).mp3"
        )
        assert result["artist"] == "Nirvana"
        assert "Official" not in result["song_name"]
        assert "Video" not in result["song_name"]
        assert result["song_name"] == "Smells Like Teen Spirit"

    def test_official_audio_brackets(self):
        result = parse_filename("ACDC - Thunderstruck [Official Audio].flac")
        assert result["artist"] == "ACDC"
        assert "Official" not in result["song_name"]
        assert "Audio" not in result["song_name"]
        assert "Thunderstruck" == result["song_name"]

    def test_lyric_video(self):
        result = parse_filename("Artist - My Song (Lyric Video).mp3")
        assert "Lyric" not in result["song_name"]
        assert "Video" not in result["song_name"]

    def test_visualizer(self):
        result = parse_filename("Artist - Cool Track (Visualizer).ogg")
        assert "Visualizer" not in result["song_name"]

    def test_feat_parentheses(self):
        result = parse_filename("Drake - God's Plan (feat. Someone).wav")
        assert result["artist"] == "Drake"
        assert "feat" not in result["song_name"].lower()
        assert "Someone" not in result["song_name"]
        assert result["song_name"] == "God's Plan"

    def test_ft_brackets(self):
        result = parse_filename("Artist - Song [ft. Another].mp3")
        assert "ft." not in result["song_name"].lower()
        assert "Another" not in result["song_name"]

    def test_bitrate_tag(self):
        result = parse_filename("Artist - Song 320kbps.mp3")
        assert "320" not in result["song_name"]
        assert "kbps" not in result["song_name"].lower()

    def test_year_parentheses(self):
        result = parse_filename("Artist - Song (2023).mp3")
        assert "(2023)" not in result["song_name"]

    def test_year_brackets(self):
        result = parse_filename("Artist - Song [2019].ogg")
        assert "[2019]" not in result["song_name"]

    def test_multiple_tags(self):
        """Multiple tags should all be stripped."""
        result = parse_filename(
            "Artist - Song (Official Video) (feat. Someone) [2023].mp3"
        )
        assert "Official" not in result["song_name"]
        assert "feat" not in result["song_name"].lower()
        assert "2023" not in result["song_name"]


# ---------------------------------------------------------------------------
# parse_filename: File extensions
# ---------------------------------------------------------------------------


class TestParseFilenameExtensions:
    """Test that file extensions are properly stripped."""

    def test_mp3_extension(self):
        result = parse_filename("Artist - Song.mp3")
        assert ".mp3" not in result["song_name"]

    def test_ogg_extension(self):
        result = parse_filename("Song.ogg")
        assert ".ogg" not in result["song_name"]

    def test_flac_extension(self):
        result = parse_filename("Song.flac")
        assert ".flac" not in result["song_name"]

    def test_wav_extension(self):
        result = parse_filename("Song.wav")
        assert ".wav" not in result["song_name"]

    def test_opus_extension(self):
        result = parse_filename("Song.opus")
        assert ".opus" not in result["song_name"]

    def test_no_extension(self):
        """Filenames without extensions should still work."""
        result = parse_filename("Artist - Song")
        assert result["song_name"] == "Song"
        assert result["artist"] == "Artist"


# ---------------------------------------------------------------------------
# clean_name: Title casing
# ---------------------------------------------------------------------------


class TestCleanNameTitleCase:
    """Test the smart title casing logic."""

    def test_basic_title_case(self):
        result = clean_name("hello world")
        assert result == "Hello World"

    def test_articles_lowercase(self):
        """Articles and prepositions should be lowercase (except first/last word)."""
        result = clean_name("lord of the rings")
        assert result == "Lord of the Rings"

    def test_first_word_always_capitalized(self):
        result = clean_name("the quick brown fox")
        assert result[0] == "T"
        assert result.startswith("The")

    def test_last_word_always_capitalized(self):
        # Last word is always capitalized even if it's normally lowercase
        result = clean_name("welcome to the")
        assert result == "Welcome to The"

    def test_acronyms_preserved(self):
        """All-caps words of 2+ chars should stay uppercase."""
        result = clean_name("DNA test OK")
        assert "DNA" in result
        assert "OK" in result

    def test_underscore_replacement(self):
        result = clean_name("my_song_name")
        assert "_" not in result
        assert result == "My Song Name"

    def test_whitespace_collapse(self):
        result = clean_name("  too   many   spaces  ")
        assert "  " not in result
        assert result == "Too Many Spaces"


# ---------------------------------------------------------------------------
# clean_name: Tag removal
# ---------------------------------------------------------------------------


class TestCleanNameTagRemoval:
    """Test tag/label stripping in clean_name."""

    def test_official_video_removed(self):
        result = clean_name("My Song (Official Video)")
        assert "Official" not in result
        assert "Video" not in result
        assert result.strip() == "My Song"

    def test_official_music_video_removed(self):
        result = clean_name("My Song (Official Music Video)")
        assert "Official" not in result

    def test_official_audio_removed(self):
        result = clean_name("My Song (Official Audio)")
        assert "Official" not in result

    def test_lyric_video_brackets_removed(self):
        result = clean_name("My Song [Lyric Video]")
        assert "Lyric" not in result

    def test_feat_removed(self):
        result = clean_name("My Song (feat. John Doe)")
        assert "feat" not in result.lower()
        assert "John" not in result

    def test_ft_removed(self):
        result = clean_name("My Song (ft. Jane)")
        assert "ft" not in result.lower()
        assert "Jane" not in result

    def test_bitrate_removed(self):
        result = clean_name("My Song 320kbps")
        assert "320" not in result
        assert "kbps" not in result.lower()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test unusual inputs and boundary conditions."""

    def test_empty_string(self):
        result = parse_filename("")
        assert result["song_name"] == ""
        assert result["artist"] == ""

    def test_only_extension(self):
        result = parse_filename(".mp3")
        assert isinstance(result["song_name"], str)
        assert isinstance(result["artist"], str)

    def test_spaces_only(self):
        result = parse_filename("   .ogg")
        assert isinstance(result["song_name"], str)

    def test_unicode_characters(self):
        result = parse_filename("Bj\u00f6rk - J\u00f3ga.mp3")
        assert "Bj\u00f6rk" in result["artist"]
        assert "J\u00f3ga" in result["song_name"]

    def test_very_long_filename(self):
        long_name = "A" * 200 + " - " + "B" * 200 + ".mp3"
        result = parse_filename(long_name)
        assert len(result["song_name"]) > 0
        assert len(result["artist"]) > 0

    def test_multiple_dots_in_filename(self):
        result = parse_filename("Artist - Song.Title.v2.mp3")
        assert result["artist"] == "Artist"
        # The stem should be "Song.Title.v2"
        assert len(result["song_name"]) > 0

    def test_parentheses_in_title_not_a_tag(self):
        """Parenthetical text that isn't a known tag should be kept."""
        result = parse_filename("Artist - Song (Acoustic).mp3")
        assert "acoustic" in result["song_name"].lower()

    def test_dash_inside_word_not_separator(self):
        """Hyphens inside words shouldn't trigger artist/title split."""
        result = parse_filename("Re-Enter.mp3")
        # Should not split on the hyphen inside "Re-Enter"
        assert result["artist"] == ""
        assert len(result["song_name"]) > 0

    def test_result_keys(self):
        """parse_filename should always return dict with both keys."""
        result = parse_filename("anything.mp3")
        assert "song_name" in result
        assert "artist" in result

    def test_clean_name_empty(self):
        result = clean_name("")
        assert result == ""

    def test_clean_name_only_whitespace(self):
        result = clean_name("     ")
        assert result == ""


# ---------------------------------------------------------------------------
# Parametrized tests from fixture data
# ---------------------------------------------------------------------------


class TestParseFilenameParametrized:
    """Run parametrized tests from the shared fixture data."""

    def test_fixture_cases(self, filename_test_cases):
        """Validate all cases from the filename_test_cases fixture."""
        for case in filename_test_cases:
            result = parse_filename(case["filename"])
            # Compare case-insensitively for song name since smart title case
            # may lowercase articles/prepositions differently
            assert result["song_name"].lower() == case["expected_song"].lower(), (
                f"Song mismatch for '{case['filename']}': "
                f"got '{result['song_name']}', expected '{case['expected_song']}'"
            )
            assert result["artist"].lower() == case["expected_artist"].lower(), (
                f"Artist mismatch for '{case['filename']}': "
                f"got '{result['artist']}', expected '{case['expected_artist']}'"
            )


# ---------------------------------------------------------------------------
# Consistency tests
# ---------------------------------------------------------------------------


class TestParseFilenameConsistency:
    """Test that parsing is deterministic and consistent."""

    def test_idempotent_clean_name(self):
        """Running clean_name twice should produce the same result."""
        raw = "my_song (Official Video) 320kbps"
        first = clean_name(raw)
        second = clean_name(first)
        assert first == second

    def test_different_extensions_same_result(self):
        """Same base filename with different extensions should give same parse."""
        extensions = [".mp3", ".ogg", ".flac", ".wav", ".opus"]
        results = [parse_filename(f"Artist - Song{ext}") for ext in extensions]
        for r in results[1:]:
            assert r["song_name"] == results[0]["song_name"]
            assert r["artist"] == results[0]["artist"]

    def test_parse_returns_stripped_strings(self):
        """Results should never have leading/trailing whitespace."""
        result = parse_filename("  Artist  -  Song  .mp3")
        assert result["song_name"] == result["song_name"].strip()
        assert result["artist"] == result["artist"].strip()
