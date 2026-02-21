"""
Tests for the stem separator service.

Tests cover:
    - Instrument enum and constants
    - Band-pass filtering
    - Spectral centroid computation
    - Pitch contour normalisation and lane mapping
    - Section name helpers
    - Difficulty profile generation per instrument
    - Per-instrument stem analysis (guitar, bass, drums, vocals, full_mix)
    - High-level analyze_instrument dispatcher
    - Proportional index mapping helpers used in song_generator
    - Demucs availability check
"""

import math
from unittest.mock import patch

import numpy as np
import pytest

from src.services.stem_separator import (
    DRUM_BANDS,
    DRUM_LANE_MAP,
    FREQ_BANDS,
    INSTRUMENT_SECTION_NAMES,
    Instrument,
    SeparationResult,
    StemAnalysis,
    _centroids_to_pitch_contour,
    _compute_spectral_centroids_at_onsets,
    analyze_bass_stem,
    analyze_drums_stem,
    analyze_guitar_stem,
    analyze_instrument,
    analyze_vocals_stem,
    bandpass_filter,
    demucs_available,
    get_difficulty_profile_for_instrument,
    get_section_name,
    pitch_contour_to_lanes,
    pitch_to_lane,
    separate_stems,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_RATE = 22050
DURATION = 5.0  # seconds


@pytest.fixture
def sine_signal():
    """Generate a simple 440 Hz sine wave (guitar A4)."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return signal


@pytest.fixture
def multi_tone_signal():
    """Generate a signal with low (bass), mid (guitar), and high (hihat) components."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    bass = 0.4 * np.sin(2 * np.pi * 80 * t)
    guitar = 0.3 * np.sin(2 * np.pi * 800 * t)
    hihat = 0.15 * np.sin(2 * np.pi * 8000 * t)
    # Add a few percussive clicks
    click_positions = [int(SAMPLE_RATE * i) for i in range(1, int(DURATION))]
    clicks = np.zeros_like(t)
    for pos in click_positions:
        if pos + 100 < len(clicks):
            clicks[pos : pos + 100] = 0.8 * np.random.default_rng(42).standard_normal(
                100
            )
    signal = (bass + guitar + hihat + clicks).astype(np.float32)
    return signal


@pytest.fixture
def separation_result(multi_tone_signal):
    """Create a SeparationResult from the multi-tone signal using real HPSS."""
    import librosa

    harmonic, percussive = librosa.effects.hpss(multi_tone_signal, margin=3.0)
    return SeparationResult(
        harmonic=harmonic,
        percussive=percussive,
        full_signal=multi_tone_signal,
        sample_rate=SAMPLE_RATE,
        duration=DURATION,
    )


@pytest.fixture
def mock_audio_file(tmp_path, multi_tone_signal):
    """Write a multi-tone signal to a WAV file for integration tests."""
    import soundfile as sf

    wav_path = tmp_path / "test_song.wav"
    sf.write(str(wav_path), multi_tone_signal, SAMPLE_RATE)
    return str(wav_path)


# ---------------------------------------------------------------------------
# Instrument enum
# ---------------------------------------------------------------------------


class TestInstrumentEnum:
    def test_values(self):
        assert Instrument.GUITAR == "guitar"
        assert Instrument.BASS == "bass"
        assert Instrument.DRUMS == "drums"
        assert Instrument.VOCALS == "vocals"
        assert Instrument.FULL_MIX == "full_mix"

    def test_is_string(self):
        assert isinstance(Instrument.GUITAR, str)
        assert isinstance(Instrument.DRUMS, str)

    def test_all_members(self):
        members = {m.value for m in Instrument}
        assert members == {"guitar", "bass", "drums", "vocals", "full_mix"}


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_freq_bands_have_expected_instruments(self):
        assert "bass" in FREQ_BANDS
        assert "guitar" in FREQ_BANDS
        assert "vocals" in FREQ_BANDS

    def test_freq_bands_are_positive_ranges(self):
        for name, (low, high) in FREQ_BANDS.items():
            assert low > 0, f"{name} low freq must be positive"
            assert high > low, f"{name} high must exceed low"

    def test_drum_bands_have_expected_components(self):
        assert "kick" in DRUM_BANDS
        assert "snare" in DRUM_BANDS
        assert "hihat" in DRUM_BANDS
        assert "tom" in DRUM_BANDS

    def test_drum_bands_are_positive_ranges(self):
        for name, (low, high) in DRUM_BANDS.items():
            assert low > 0, f"{name} low freq must be positive"
            assert high > low, f"{name} high must exceed low"

    def test_drum_lane_map_values(self):
        assert DRUM_LANE_MAP["kick"] == 0
        assert DRUM_LANE_MAP["snare"] == 1
        assert DRUM_LANE_MAP["hihat"] == 2
        assert DRUM_LANE_MAP["tom"] == 3

    def test_instrument_section_names_all_instruments(self):
        for inst in ("guitar", "bass", "drums", "vocals", "full_mix"):
            assert inst in INSTRUMENT_SECTION_NAMES

    def test_instrument_section_names_all_difficulties(self):
        for inst, sections in INSTRUMENT_SECTION_NAMES.items():
            for diff in ("easy", "medium", "hard", "expert"):
                assert diff in sections, f"{inst} missing {diff}"

    def test_drums_section_names_are_drums(self):
        for diff, name in INSTRUMENT_SECTION_NAMES["drums"].items():
            assert "Drums" in name

    def test_guitar_section_names_are_single(self):
        for diff, name in INSTRUMENT_SECTION_NAMES["guitar"].items():
            assert "Single" in name

    def test_bass_section_names_are_double_bass(self):
        for diff, name in INSTRUMENT_SECTION_NAMES["bass"].items():
            assert "DoubleBass" in name


# ---------------------------------------------------------------------------
# Band-pass filter
# ---------------------------------------------------------------------------


class TestBandpassFilter:
    def test_output_same_length(self, sine_signal):
        filtered = bandpass_filter(sine_signal, SAMPLE_RATE, 200.0, 600.0)
        assert len(filtered) == len(sine_signal)

    def test_passband_preserves_energy(self, sine_signal):
        """440 Hz sine is within 200-600 Hz — energy should be mostly preserved."""
        filtered = bandpass_filter(sine_signal, SAMPLE_RATE, 200.0, 600.0)
        original_rms = np.sqrt(np.mean(sine_signal**2))
        filtered_rms = np.sqrt(np.mean(filtered**2))
        # Should retain at least 50% of energy
        assert filtered_rms > original_rms * 0.5

    def test_stopband_attenuates(self, sine_signal):
        """440 Hz sine is outside 2000-5000 Hz — should be strongly attenuated."""
        filtered = bandpass_filter(sine_signal, SAMPLE_RATE, 2000.0, 5000.0)
        original_rms = np.sqrt(np.mean(sine_signal**2))
        filtered_rms = np.sqrt(np.mean(filtered**2))
        # Should attenuate to less than 20% of original energy
        assert filtered_rms < original_rms * 0.2

    def test_returns_ndarray(self, sine_signal):
        filtered = bandpass_filter(sine_signal, SAMPLE_RATE, 100.0, 1000.0)
        assert isinstance(filtered, np.ndarray)

    def test_empty_signal(self):
        empty = np.zeros(0, dtype=np.float32)
        # Should not crash on empty input
        filtered = bandpass_filter(empty, SAMPLE_RATE, 100.0, 1000.0)
        assert len(filtered) == 0

    def test_narrow_band(self, sine_signal):
        """Very narrow band around 440 Hz should still pass some signal."""
        filtered = bandpass_filter(sine_signal, SAMPLE_RATE, 430.0, 450.0)
        assert np.sqrt(np.mean(filtered**2)) > 0

    def test_full_spectrum_band(self, sine_signal):
        """Full spectrum should preserve nearly all energy."""
        filtered = bandpass_filter(sine_signal, SAMPLE_RATE, 1.0, 11025.0)
        original_rms = np.sqrt(np.mean(sine_signal**2))
        filtered_rms = np.sqrt(np.mean(filtered**2))
        assert filtered_rms > original_rms * 0.9


# ---------------------------------------------------------------------------
# Spectral centroid computation
# ---------------------------------------------------------------------------


class TestSpectralCentroids:
    def test_empty_onset_frames(self, sine_signal):
        result = _compute_spectral_centroids_at_onsets(
            sine_signal, SAMPLE_RATE, np.array([], dtype=int)
        )
        assert result == []

    def test_returns_list_of_floats(self, sine_signal):
        frames = np.array([10, 50, 100])
        result = _compute_spectral_centroids_at_onsets(sine_signal, SAMPLE_RATE, frames)
        assert isinstance(result, list)
        assert len(result) == 3
        for val in result:
            assert isinstance(val, float)

    def test_centroids_are_positive(self, sine_signal):
        frames = np.array([10, 50, 100])
        result = _compute_spectral_centroids_at_onsets(sine_signal, SAMPLE_RATE, frames)
        for val in result:
            assert val > 0

    def test_sine_centroid_near_frequency(self, sine_signal):
        """For a pure 440 Hz sine, centroids should be near 440 Hz."""
        frames = np.array([20, 50, 80])
        result = _compute_spectral_centroids_at_onsets(sine_signal, SAMPLE_RATE, frames)
        for val in result:
            # Allow generous tolerance (spectral centroid isn't exactly the fundamental)
            assert 200 < val < 1200

    def test_out_of_range_frames_handled(self, sine_signal):
        """Frames beyond the signal length should not crash."""
        frames = np.array([0, 999999])
        result = _compute_spectral_centroids_at_onsets(sine_signal, SAMPLE_RATE, frames)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Pitch contour normalisation
# ---------------------------------------------------------------------------


class TestPitchContour:
    def test_empty_centroids(self):
        assert _centroids_to_pitch_contour([], 100.0, 5000.0) == []

    def test_low_freq_maps_to_zero(self):
        result = _centroids_to_pitch_contour([100.0], 100.0, 5000.0)
        assert len(result) == 1
        assert result[0] == pytest.approx(0.0, abs=0.05)

    def test_high_freq_maps_to_one(self):
        result = _centroids_to_pitch_contour([5000.0], 100.0, 5000.0)
        assert len(result) == 1
        assert result[0] == pytest.approx(1.0, abs=0.05)

    def test_mid_freq_maps_to_middle(self):
        # Geometric midpoint of 100 and 5000 on log2 scale
        geometric_mid = math.sqrt(100.0 * 5000.0)
        result = _centroids_to_pitch_contour([geometric_mid], 100.0, 5000.0)
        assert len(result) == 1
        assert 0.3 < result[0] < 0.7

    def test_values_clamped_to_zero_one(self):
        result = _centroids_to_pitch_contour([10.0, 50000.0], 100.0, 5000.0)
        assert all(0.0 <= v <= 1.0 for v in result)

    def test_zero_centroid_maps_to_zero(self):
        result = _centroids_to_pitch_contour([0.0], 100.0, 5000.0)
        assert result[0] == 0.0

    def test_monotonic_input_gives_monotonic_output(self):
        centroids = [100.0, 500.0, 1000.0, 2000.0, 5000.0]
        result = _centroids_to_pitch_contour(centroids, 100.0, 5000.0)
        for i in range(1, len(result)):
            assert result[i] >= result[i - 1]


# ---------------------------------------------------------------------------
# Pitch-to-lane mapping
# ---------------------------------------------------------------------------


class TestPitchToLane:
    def test_zero_pitch_is_lane_zero(self):
        assert pitch_to_lane(0.0, max_lane=4) == 0

    def test_one_pitch_is_max_lane(self):
        assert pitch_to_lane(1.0, max_lane=4) == 4

    def test_mid_pitch_is_middle_lane(self):
        lane = pitch_to_lane(0.5, max_lane=4)
        assert lane in (2, 3)  # with 5 lanes, 0.5 maps to ~2.5

    def test_clamped_below_zero(self):
        assert pitch_to_lane(-0.5, max_lane=4) == 0

    def test_clamped_above_one(self):
        assert pitch_to_lane(1.5, max_lane=4) == 4

    def test_max_lane_two(self):
        assert pitch_to_lane(1.0, max_lane=2) == 2
        assert pitch_to_lane(0.0, max_lane=2) == 0


class TestPitchContourToLanes:
    def test_empty_contour(self):
        assert pitch_contour_to_lanes([]) == []

    def test_single_value(self):
        result = pitch_contour_to_lanes([0.5], max_lane=4, smoothing=1)
        assert len(result) == 1
        assert 0 <= result[0] <= 4

    def test_length_preserved(self):
        contour = [0.0, 0.25, 0.5, 0.75, 1.0]
        result = pitch_contour_to_lanes(contour, max_lane=4, smoothing=1)
        assert len(result) == len(contour)

    def test_all_values_in_range(self):
        contour = [0.1 * i for i in range(11)]
        result = pitch_contour_to_lanes(contour, max_lane=4, smoothing=1)
        for lane in result:
            assert 0 <= lane <= 4

    def test_smoothing_reduces_jitter(self):
        # Alternating high/low should be smoothed
        contour = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        no_smooth = pitch_contour_to_lanes(contour, max_lane=4, smoothing=1)
        smoothed = pitch_contour_to_lanes(contour, max_lane=4, smoothing=5)

        # Count lane changes
        no_smooth_changes = sum(
            1 for i in range(1, len(no_smooth)) if no_smooth[i] != no_smooth[i - 1]
        )
        smooth_changes = sum(
            1 for i in range(1, len(smoothed)) if smoothed[i] != smoothed[i - 1]
        )
        # Smoothing should reduce the number of lane changes
        assert smooth_changes <= no_smooth_changes

    def test_ascending_contour_gives_ascending_lanes(self):
        contour = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        result = pitch_contour_to_lanes(contour, max_lane=4, smoothing=1)
        for i in range(1, len(result)):
            assert result[i] >= result[i - 1]


# ---------------------------------------------------------------------------
# Section name helpers
# ---------------------------------------------------------------------------


class TestGetSectionName:
    def test_guitar_expert(self):
        assert get_section_name("guitar", "expert") == "ExpertSingle"

    def test_drums_hard(self):
        assert get_section_name("drums", "hard") == "HardDrums"

    def test_bass_easy(self):
        assert get_section_name("bass", "easy") == "EasyDoubleBass"

    def test_vocals_medium(self):
        assert get_section_name("vocals", "medium") == "MediumSingle"

    def test_full_mix_expert(self):
        assert get_section_name("full_mix", "expert") == "ExpertSingle"

    def test_case_insensitive(self):
        assert get_section_name("Guitar", "Expert") == "ExpertSingle"
        assert get_section_name("DRUMS", "EASY") == "EasyDrums"

    def test_unknown_instrument_defaults_to_guitar(self):
        result = get_section_name("kazoo", "expert")
        assert result == "ExpertSingle"


# ---------------------------------------------------------------------------
# Difficulty profile per instrument
# ---------------------------------------------------------------------------


class TestDifficultyProfile:
    def test_guitar_has_required_keys(self):
        profile = get_difficulty_profile_for_instrument("guitar", "expert")
        required = {
            "section_name",
            "note_skip",
            "max_lane",
            "chord_chance",
            "hopo_chance",
            "sustain_chance",
            "min_note_gap_ticks",
        }
        assert required.issubset(profile.keys())

    def test_guitar_expert_section_name(self):
        profile = get_difficulty_profile_for_instrument("guitar", "expert")
        assert profile["section_name"] == "ExpertSingle"

    def test_drums_no_hopo(self):
        for diff in ("easy", "medium", "hard", "expert"):
            profile = get_difficulty_profile_for_instrument("drums", diff)
            assert profile["hopo_chance"] == 0.0
            assert profile["sustain_chance"] == 0.0

    def test_drums_expert_section_name(self):
        profile = get_difficulty_profile_for_instrument("drums", "expert")
        assert profile["section_name"] == "ExpertDrums"

    def test_bass_more_sustain_than_guitar(self):
        bass_profile = get_difficulty_profile_for_instrument("bass", "expert")
        guitar_profile = get_difficulty_profile_for_instrument("guitar", "expert")
        assert bass_profile["sustain_chance"] >= guitar_profile["sustain_chance"]

    def test_bass_fewer_chords_than_guitar(self):
        bass_profile = get_difficulty_profile_for_instrument("bass", "expert")
        guitar_profile = get_difficulty_profile_for_instrument("guitar", "expert")
        assert bass_profile["chord_chance"] <= guitar_profile["chord_chance"]

    def test_easy_has_fewer_lanes_than_expert(self):
        for inst in ("guitar", "bass"):
            easy = get_difficulty_profile_for_instrument(inst, "easy")
            expert = get_difficulty_profile_for_instrument(inst, "expert")
            assert easy["max_lane"] <= expert["max_lane"]

    def test_easy_has_larger_gap_than_expert(self):
        for inst in ("guitar", "bass", "drums"):
            easy = get_difficulty_profile_for_instrument(inst, "easy")
            expert = get_difficulty_profile_for_instrument(inst, "expert")
            assert easy["min_note_gap_ticks"] >= expert["min_note_gap_ticks"]

    def test_vocals_no_chords(self):
        for diff in ("easy", "medium", "hard", "expert"):
            profile = get_difficulty_profile_for_instrument("vocals", diff)
            assert profile["chord_chance"] == 0.0

    def test_unknown_difficulty_uses_expert_base(self):
        profile = get_difficulty_profile_for_instrument("guitar", "impossible")
        # Should still return a valid profile (falls back to expert)
        assert "section_name" in profile
        assert profile["max_lane"] > 0


# ---------------------------------------------------------------------------
# StemAnalysis dataclass
# ---------------------------------------------------------------------------


class TestStemAnalysis:
    def test_default_values(self):
        sa = StemAnalysis(instrument="guitar")
        assert sa.instrument == "guitar"
        assert sa.onset_times == []
        assert sa.onset_strengths == []
        assert sa.pitch_contour == []
        assert sa.spectral_centroids == []
        assert sa.drum_lanes == []
        assert sa.drum_band_onsets == {}
        assert sa.stem_signal is None
        assert sa.sample_rate == 22050

    def test_custom_values(self):
        sa = StemAnalysis(
            instrument="drums",
            onset_times=[0.5, 1.0, 1.5],
            onset_strengths=[0.8, 0.6, 0.9],
            drum_lanes=[0, 1, 2],
            sample_rate=44100,
        )
        assert sa.instrument == "drums"
        assert len(sa.onset_times) == 3
        assert len(sa.drum_lanes) == 3
        assert sa.sample_rate == 44100


# ---------------------------------------------------------------------------
# SeparationResult dataclass
# ---------------------------------------------------------------------------


class TestSeparationResult:
    def test_default_values(self):
        sr = SeparationResult()
        assert sr.harmonic is None
        assert sr.percussive is None
        assert sr.full_signal is None
        assert sr.sample_rate == 22050
        assert sr.duration == 0.0

    def test_custom_values(self, multi_tone_signal):
        sr = SeparationResult(
            full_signal=multi_tone_signal,
            sample_rate=44100,
            duration=5.0,
        )
        assert sr.full_signal is not None
        assert sr.sample_rate == 44100
        assert sr.duration == 5.0


# ---------------------------------------------------------------------------
# Stem separation (HPSS)
# ---------------------------------------------------------------------------


class TestSeparateStems:
    @pytest.fixture(autouse=True)
    def _needs_soundfile(self):
        """Skip if soundfile is not available (needed for WAV writing)."""
        pytest.importorskip("soundfile")

    def test_returns_separation_result(self, mock_audio_file):
        result = separate_stems(mock_audio_file)
        assert isinstance(result, SeparationResult)

    def test_harmonic_and_percussive_not_none(self, mock_audio_file):
        result = separate_stems(mock_audio_file)
        assert result.harmonic is not None
        assert result.percussive is not None

    def test_output_same_length_as_input(self, mock_audio_file):
        result = separate_stems(mock_audio_file)
        assert len(result.harmonic) == len(result.full_signal)
        assert len(result.percussive) == len(result.full_signal)

    def test_duration_positive(self, mock_audio_file):
        result = separate_stems(mock_audio_file)
        assert result.duration > 0

    def test_sample_rate_set(self, mock_audio_file):
        result = separate_stems(mock_audio_file)
        assert result.sample_rate > 0


# ---------------------------------------------------------------------------
# Guitar stem analysis
# ---------------------------------------------------------------------------


class TestAnalyzeGuitarStem:
    def test_returns_stem_analysis(self, separation_result):
        result = analyze_guitar_stem(separation_result)
        assert isinstance(result, StemAnalysis)
        assert result.instrument == "guitar"

    def test_has_onsets(self, separation_result):
        result = analyze_guitar_stem(separation_result, sensitivity=0.8)
        # Multi-tone signal should produce some onsets in guitar range
        assert isinstance(result.onset_times, list)
        assert isinstance(result.onset_strengths, list)

    def test_onset_times_and_strengths_same_length(self, separation_result):
        result = analyze_guitar_stem(separation_result)
        assert len(result.onset_times) == len(result.onset_strengths)

    def test_has_pitch_contour(self, separation_result):
        result = analyze_guitar_stem(separation_result, sensitivity=0.8)
        if result.onset_times:
            assert len(result.pitch_contour) == len(result.onset_times)

    def test_pitch_contour_in_range(self, separation_result):
        result = analyze_guitar_stem(separation_result, sensitivity=0.8)
        for val in result.pitch_contour:
            assert 0.0 <= val <= 1.0

    def test_has_spectral_centroids(self, separation_result):
        result = analyze_guitar_stem(separation_result, sensitivity=0.8)
        if result.onset_times:
            assert len(result.spectral_centroids) == len(result.onset_times)
            for val in result.spectral_centroids:
                assert val >= 0

    def test_no_harmonic_returns_empty(self):
        empty_sep = SeparationResult(harmonic=None, sample_rate=SAMPLE_RATE)
        result = analyze_guitar_stem(empty_sep)
        assert result.onset_times == []

    def test_stem_signal_set(self, separation_result):
        result = analyze_guitar_stem(separation_result)
        assert result.stem_signal is not None

    def test_sensitivity_affects_onset_count(self, separation_result):
        low_sens = analyze_guitar_stem(separation_result, sensitivity=0.1)
        high_sens = analyze_guitar_stem(separation_result, sensitivity=0.9)
        # Higher sensitivity should produce more or equal onsets
        assert len(high_sens.onset_times) >= len(low_sens.onset_times)


# ---------------------------------------------------------------------------
# Bass stem analysis
# ---------------------------------------------------------------------------


class TestAnalyzeBassStem:
    def test_returns_stem_analysis(self, separation_result):
        result = analyze_bass_stem(separation_result)
        assert isinstance(result, StemAnalysis)
        assert result.instrument == "bass"

    def test_onset_times_and_strengths_same_length(self, separation_result):
        result = analyze_bass_stem(separation_result)
        assert len(result.onset_times) == len(result.onset_strengths)

    def test_has_pitch_contour(self, separation_result):
        result = analyze_bass_stem(separation_result, sensitivity=0.8)
        if result.onset_times:
            assert len(result.pitch_contour) == len(result.onset_times)

    def test_no_harmonic_returns_empty(self):
        empty_sep = SeparationResult(harmonic=None, sample_rate=SAMPLE_RATE)
        result = analyze_bass_stem(empty_sep)
        assert result.onset_times == []


# ---------------------------------------------------------------------------
# Drums stem analysis
# ---------------------------------------------------------------------------


class TestAnalyzeDrumsStem:
    def test_returns_stem_analysis(self, separation_result):
        result = analyze_drums_stem(separation_result)
        assert isinstance(result, StemAnalysis)
        assert result.instrument == "drums"

    def test_has_drum_lanes(self, separation_result):
        result = analyze_drums_stem(separation_result, sensitivity=0.8)
        if result.onset_times:
            assert len(result.drum_lanes) == len(result.onset_times)

    def test_drum_lanes_in_valid_range(self, separation_result):
        result = analyze_drums_stem(separation_result, sensitivity=0.8)
        for lane in result.drum_lanes:
            assert lane in (0, 1, 2, 3)

    def test_has_band_onset_data(self, separation_result):
        result = analyze_drums_stem(separation_result)
        assert isinstance(result.drum_band_onsets, dict)
        for band_name in ("kick", "snare", "hihat", "tom"):
            assert band_name in result.drum_band_onsets
            times, strengths = result.drum_band_onsets[band_name]
            assert isinstance(times, list)
            assert isinstance(strengths, list)
            assert len(times) == len(strengths)

    def test_onsets_sorted_by_time(self, separation_result):
        result = analyze_drums_stem(separation_result, sensitivity=0.8)
        for i in range(1, len(result.onset_times)):
            assert result.onset_times[i] >= result.onset_times[i - 1]

    def test_no_percussive_returns_empty(self):
        empty_sep = SeparationResult(percussive=None, sample_rate=SAMPLE_RATE)
        result = analyze_drums_stem(empty_sep)
        assert result.onset_times == []

    def test_onset_strengths_normalized(self, separation_result):
        result = analyze_drums_stem(separation_result, sensitivity=0.8)
        for strength in result.onset_strengths:
            assert 0.0 <= strength <= 1.0


# ---------------------------------------------------------------------------
# Vocals stem analysis
# ---------------------------------------------------------------------------


class TestAnalyzeVocalsStem:
    def test_returns_stem_analysis(self, separation_result):
        result = analyze_vocals_stem(separation_result)
        assert isinstance(result, StemAnalysis)
        assert result.instrument == "vocals"

    def test_onset_times_and_strengths_same_length(self, separation_result):
        result = analyze_vocals_stem(separation_result)
        assert len(result.onset_times) == len(result.onset_strengths)

    def test_no_harmonic_returns_empty(self):
        empty_sep = SeparationResult(harmonic=None, sample_rate=SAMPLE_RATE)
        result = analyze_vocals_stem(empty_sep)
        assert result.onset_times == []


# ---------------------------------------------------------------------------
# High-level analyze_instrument dispatcher
# ---------------------------------------------------------------------------


class TestAnalyzeInstrument:
    @pytest.fixture(autouse=True)
    def _needs_soundfile(self):
        pytest.importorskip("soundfile")

    def test_guitar_returns_stem_analysis(self, mock_audio_file):
        stem, sep = analyze_instrument(mock_audio_file, instrument="guitar")
        assert isinstance(stem, StemAnalysis)
        assert stem.instrument == "guitar"
        assert isinstance(sep, SeparationResult)

    def test_bass_returns_stem_analysis(self, mock_audio_file):
        stem, sep = analyze_instrument(mock_audio_file, instrument="bass")
        assert stem.instrument == "bass"

    def test_drums_returns_stem_analysis(self, mock_audio_file):
        stem, sep = analyze_instrument(mock_audio_file, instrument="drums")
        assert stem.instrument == "drums"

    def test_vocals_returns_stem_analysis(self, mock_audio_file):
        stem, sep = analyze_instrument(mock_audio_file, instrument="vocals")
        assert stem.instrument == "vocals"

    def test_full_mix_returns_stem_analysis(self, mock_audio_file):
        stem, sep = analyze_instrument(mock_audio_file, instrument="full_mix")
        assert stem.instrument == "full_mix"
        # Full mix doesn't do HPSS, so separation should be empty
        assert sep.harmonic is None

    def test_case_insensitive_instrument(self, mock_audio_file):
        stem, _ = analyze_instrument(mock_audio_file, instrument="GUITAR")
        assert stem.instrument == "guitar"

    def test_unknown_instrument_raises(self, mock_audio_file):
        with pytest.raises(ValueError, match="Unknown instrument"):
            analyze_instrument(mock_audio_file, instrument="banjo")

    def test_sensitivity_parameter_accepted(self, mock_audio_file):
        # Should not raise
        stem, _ = analyze_instrument(
            mock_audio_file, instrument="guitar", sensitivity=0.5
        )
        assert isinstance(stem, StemAnalysis)


# ---------------------------------------------------------------------------
# Proportional index mapping helpers (used in song_generator)
# ---------------------------------------------------------------------------


class TestProportionalMapping:
    """Test the _map_pitch_to_selected and _map_drum_lanes_to_selected helpers."""

    def test_map_pitch_to_selected_basic(self):
        from src.services.song_generator import _map_pitch_to_selected

        pitch = [0.0, 0.25, 0.5, 0.75, 1.0]
        all_onsets = [0.0, 1.0, 2.0, 3.0, 4.0]
        # note_events is a subset (3 out of 5)
        note_events = [(0, 0.5), (384, 0.5), (768, 0.5)]

        result = _map_pitch_to_selected(pitch, all_onsets, note_events)
        assert len(result) == 3
        for val in result:
            assert 0.0 <= val <= 1.0

    def test_map_pitch_to_selected_empty(self):
        from src.services.song_generator import _map_pitch_to_selected

        assert _map_pitch_to_selected([], [], []) == []
        assert _map_pitch_to_selected([0.5], [1.0], []) == []
        assert _map_pitch_to_selected([], [1.0], [(192, 0.5)]) == []

    def test_map_drum_lanes_to_selected_basic(self):
        from src.services.song_generator import _map_drum_lanes_to_selected

        drum_lanes = [0, 1, 2, 3, 0, 1, 2, 3]
        all_onsets = list(range(8))
        note_events = [(0, 0.5), (192, 0.5), (384, 0.5), (576, 0.5)]
        profile = {"max_lane": 4}

        result = _map_drum_lanes_to_selected(
            drum_lanes, all_onsets, note_events, profile
        )
        assert len(result) == 4
        for lane in result:
            assert 0 <= lane <= 4

    def test_map_drum_lanes_empty_returns_snare(self):
        from src.services.song_generator import _map_drum_lanes_to_selected

        result = _map_drum_lanes_to_selected(
            [], [], [(192, 0.5), (384, 0.5)], {"max_lane": 4}
        )
        # Should default to snare (lane 1)
        assert result == [1, 1]

    def test_map_drum_lanes_respects_max_lane(self):
        from src.services.song_generator import _map_drum_lanes_to_selected

        drum_lanes = [0, 1, 2, 3, 4]
        note_events = [(i * 192, 0.5) for i in range(5)]
        profile = {"max_lane": 2}

        result = _map_drum_lanes_to_selected(
            drum_lanes, list(range(5)), note_events, profile
        )
        for lane in result:
            assert lane <= 2


# ---------------------------------------------------------------------------
# Demucs availability
# ---------------------------------------------------------------------------


class TestDemucsAvailable:
    def test_returns_bool(self):
        result = demucs_available()
        assert isinstance(result, bool)

    def test_false_when_not_installed(self):
        with patch.dict("sys.modules", {"demucs": None}):
            # Force an ImportError
            assert demucs_available() is False


# ---------------------------------------------------------------------------
# Integration: instrument-aware chart generation
# ---------------------------------------------------------------------------


class TestInstrumentAwareChartGeneration:
    """Integration tests verifying that generate_notes_chart produces valid
    output when given instrument-specific stem analysis data."""

    def _make_stem_analysis(self, instrument, n_onsets=50, duration=30.0):
        """Create a synthetic StemAnalysis for testing."""
        rng = np.random.default_rng(42)
        times = sorted(rng.uniform(0.5, duration - 0.5, n_onsets).tolist())
        strengths = rng.uniform(0.3, 1.0, n_onsets).tolist()

        if instrument == "drums":
            lanes = rng.integers(0, 4, n_onsets).tolist()
            return StemAnalysis(
                instrument="drums",
                onset_times=times,
                onset_strengths=strengths,
                drum_lanes=lanes,
                sample_rate=SAMPLE_RATE,
            )
        else:
            pitch = np.linspace(0.0, 1.0, n_onsets).tolist()
            return StemAnalysis(
                instrument=instrument,
                onset_times=times,
                onset_strengths=strengths,
                pitch_contour=pitch,
                spectral_centroids=[500.0 + 2000.0 * p for p in pitch],
                sample_rate=SAMPLE_RATE,
            )

    @pytest.fixture
    def chart_args(self):
        """Common args for generate_notes_chart."""
        return {
            "song_name": "Test Song",
            "artist": "Test Artist",
            "album": "Test Album",
            "year": "2024",
            "genre": "Rock",
            "tempo": 120.0,
            "beat_times": [i * 0.5 for i in range(60)],
            "onset_times": [i * 0.3 for i in range(100)],
            "onset_strengths": [0.5] * 100,
            "duration": 30.0,
            "audio_filename": "song.ogg",
            "difficulties": ["easy", "medium", "hard", "expert"],
            "enable_lyrics": False,
        }

    def test_guitar_chart_has_single_sections(self, tmp_path, chart_args):
        from src.services.song_generator import generate_notes_chart

        out = tmp_path / "notes.chart"
        stem = self._make_stem_analysis("guitar")
        chart_args["output_path"] = out
        chart_args["instrument"] = "guitar"
        chart_args["stem_analysis"] = stem

        ok = generate_notes_chart(**chart_args)
        assert ok is True
        text = out.read_text(encoding="utf-8-sig")
        assert "[ExpertSingle]" in text
        assert "[EasySingle]" in text

    def test_drums_chart_has_drums_sections(self, tmp_path, chart_args):
        from src.services.song_generator import generate_notes_chart

        out = tmp_path / "notes.chart"
        stem = self._make_stem_analysis("drums")
        chart_args["output_path"] = out
        chart_args["instrument"] = "drums"
        chart_args["stem_analysis"] = stem

        ok = generate_notes_chart(**chart_args)
        assert ok is True
        text = out.read_text(encoding="utf-8-sig")
        assert "[ExpertDrums]" in text
        assert "[EasyDrums]" in text

    def test_bass_chart_has_double_bass_sections(self, tmp_path, chart_args):
        from src.services.song_generator import generate_notes_chart

        out = tmp_path / "notes.chart"
        stem = self._make_stem_analysis("bass")
        chart_args["output_path"] = out
        chart_args["instrument"] = "bass"
        chart_args["stem_analysis"] = stem

        ok = generate_notes_chart(**chart_args)
        assert ok is True
        text = out.read_text(encoding="utf-8-sig")
        assert "[ExpertDoubleBass]" in text

    def test_drums_chart_no_sustain(self, tmp_path, chart_args):
        from src.services.song_generator import generate_notes_chart

        out = tmp_path / "notes.chart"
        stem = self._make_stem_analysis("drums", n_onsets=30)
        chart_args["output_path"] = out
        chart_args["instrument"] = "drums"
        chart_args["stem_analysis"] = stem

        ok = generate_notes_chart(**chart_args)
        assert ok is True
        text = out.read_text(encoding="utf-8-sig")

        # In drum sections, all sustains should be 0
        import re

        in_drums = False
        for line in text.splitlines():
            stripped = line.strip()
            if "Drums" in stripped and stripped.startswith("["):
                in_drums = True
                continue
            if in_drums and stripped == "}":
                in_drums = False
                continue
            if in_drums:
                m = re.match(r"\s*\d+\s*=\s*N\s+(\d+)\s+(\d+)", line)
                if m:
                    sustain = int(m.group(2))
                    assert sustain == 0, f"Drum note has sustain {sustain}: {line}"

    def test_drums_chart_no_hopo(self, tmp_path, chart_args):
        from src.services.song_generator import generate_notes_chart

        out = tmp_path / "notes.chart"
        stem = self._make_stem_analysis("drums", n_onsets=30)
        chart_args["output_path"] = out
        chart_args["instrument"] = "drums"
        chart_args["stem_analysis"] = stem

        ok = generate_notes_chart(**chart_args)
        assert ok is True
        text = out.read_text(encoding="utf-8-sig")

        # HOPO markers (N 5 0) should never appear in drum sections
        in_drums = False
        for line in text.splitlines():
            stripped = line.strip()
            if "Drums" in stripped and stripped.startswith("["):
                in_drums = True
                continue
            if in_drums and stripped == "}":
                in_drums = False
                continue
            if in_drums and "= N 5 0" in line:
                pytest.fail(f"HOPO marker found in drum section: {line}")

    def test_guitar_pitch_lanes_affect_output(self, tmp_path, chart_args):
        """When pitch contour is all low, notes should favour low lanes."""
        from src.services.song_generator import generate_notes_chart

        out = tmp_path / "notes.chart"
        stem = StemAnalysis(
            instrument="guitar",
            onset_times=[i * 0.5 for i in range(20)],
            onset_strengths=[0.7] * 20,
            pitch_contour=[0.05] * 20,  # all very low pitch
            sample_rate=SAMPLE_RATE,
        )
        chart_args["output_path"] = out
        chart_args["instrument"] = "guitar"
        chart_args["stem_analysis"] = stem
        chart_args["difficulties"] = ["expert"]

        ok = generate_notes_chart(**chart_args)
        assert ok is True
        text = out.read_text(encoding="utf-8-sig")

        # Count lane 0 (green) notes vs lane 4 (orange) notes
        import re

        lane_0_count = 0
        lane_4_count = 0
        in_section = False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped == "[ExpertSingle]":
                in_section = True
                continue
            if in_section and stripped == "}":
                break
            if in_section:
                m = re.match(r"\s*\d+\s*=\s*N\s+(\d+)\s+\d+", line)
                if m:
                    lane = int(m.group(1))
                    if lane == 0:
                        lane_0_count += 1
                    elif lane == 4:
                        lane_4_count += 1

        # With all-low pitch, lane 0 should dominate
        assert lane_0_count > lane_4_count

    def test_full_mix_still_works(self, tmp_path, chart_args):
        """full_mix instrument should fall back to legacy pattern-based generation."""
        from src.services.song_generator import generate_notes_chart

        out = tmp_path / "notes.chart"
        chart_args["output_path"] = out
        chart_args["instrument"] = "full_mix"
        chart_args["stem_analysis"] = None

        ok = generate_notes_chart(**chart_args)
        assert ok is True
        text = out.read_text(encoding="utf-8-sig")
        assert "[ExpertSingle]" in text

    def test_chart_events_still_sorted(self, tmp_path, chart_args):
        """Events in [Events] should remain sorted even with stem analysis."""
        import re

        from src.services.song_generator import generate_notes_chart

        out = tmp_path / "notes.chart"
        stem = self._make_stem_analysis("guitar")
        chart_args["output_path"] = out
        chart_args["instrument"] = "guitar"
        chart_args["stem_analysis"] = stem
        chart_args["enable_lyrics"] = True

        ok = generate_notes_chart(**chart_args)
        assert ok is True
        text = out.read_text(encoding="utf-8-sig")

        in_events = False
        ticks = []
        for line in text.splitlines():
            if line.strip() == "[Events]":
                in_events = True
                continue
            if in_events and line.strip() == "}":
                break
            if in_events:
                m = re.match(r"\s*(\d+)\s*=", line)
                if m:
                    ticks.append(int(m.group(1)))

        for i in range(1, len(ticks)):
            assert ticks[i] >= ticks[i - 1], (
                f"Events out of order: tick {ticks[i - 1]} followed by {ticks[i]}"
            )
