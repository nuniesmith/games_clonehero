"""
Clone Hero Content Manager - Stem Separator Service

Isolates individual instrument stems (guitar, bass, drums, vocals) from a
full audio mix using librosa's built-in signal processing:

    - **Harmonic-Percussive Source Separation (HPSS)** splits audio into
      harmonic (pitched) and percussive (transient) components.
    - **Band-pass filtering** further isolates frequency ranges associated
      with each instrument.
    - **Beat-aligned hybrid onset detection** supplements stem-detected
      onsets with the full mix's beat grid, weighted by per-stem energy.
      This ensures adequate note density even when HPSS + band-pass
      aggressively strips energy (common with distorted instruments).
    - **pYIN pitch detection** extracts actual fundamental frequencies
      for accurate pitch-to-lane mapping (replaces spectral centroid
      which is unreliable for distorted instruments).
    - **Open note detection** identifies the lowest pitches in a song
      and maps them to Clone Hero open notes (N 7 / purple bar),
      essential for metal charting (palm mutes, chugs, drop tuning).

This approach requires no additional dependencies beyond librosa and numpy
(already in requirements.txt).  For higher-quality separation, an optional
Demucs integration is provided but not required.

Instrument frequency ranges (approximate):
    Bass guitar  :   40 Hz –  400 Hz
    Guitar       :   80 Hz – 5000 Hz (fundamentals + harmonics)
    Vocals       :  100 Hz – 8000 Hz
    Drums/kick   :   30 Hz –  200 Hz
    Drums/snare  :  150 Hz – 5000 Hz
    Drums/hi-hat : 3000 Hz – 16000 Hz
    Drums/toms   :   80 Hz – 3000 Hz
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import librosa
import numpy as np
from loguru import logger

# ---------------------------------------------------------------------------
# Instrument definitions
# ---------------------------------------------------------------------------


class Instrument(str, Enum):
    """Supported instrument targets for stem separation."""

    GUITAR = "guitar"
    BASS = "bass"
    DRUMS = "drums"
    VOCALS = "vocals"
    FULL_MIX = "full_mix"  # no separation (legacy behaviour)


# Clone Hero chart section names per instrument and difficulty
INSTRUMENT_SECTION_NAMES: dict[str, dict[str, str]] = {
    "guitar": {
        "easy": "EasySingle",
        "medium": "MediumSingle",
        "hard": "HardSingle",
        "expert": "ExpertSingle",
    },
    "bass": {
        "easy": "EasyDoubleBass",
        "medium": "MediumDoubleBass",
        "hard": "HardDoubleBass",
        "expert": "ExpertDoubleBass",
    },
    "drums": {
        "easy": "EasyDrums",
        "medium": "MediumDrums",
        "hard": "HardDrums",
        "expert": "ExpertDrums",
    },
    "vocals": {
        "easy": "EasySingle",
        "medium": "MediumSingle",
        "hard": "HardSingle",
        "expert": "ExpertSingle",
    },
    "full_mix": {
        "easy": "EasySingle",
        "medium": "MediumSingle",
        "hard": "HardSingle",
        "expert": "ExpertSingle",
    },
}


# ---------------------------------------------------------------------------
# Frequency bands for instrument isolation
# ---------------------------------------------------------------------------

# (low_hz, high_hz) — used for band-pass filtering after HPSS
FREQ_BANDS: dict[str, tuple[float, float]] = {
    "bass": (30.0, 400.0),
    "guitar": (80.0, 5000.0),
    "vocals": (100.0, 8000.0),
}

# Drum sub-bands for per-pad onset detection
DRUM_BANDS: dict[str, tuple[float, float]] = {
    "kick": (30.0, 200.0),
    "snare": (150.0, 5000.0),
    "hihat": (3000.0, 16000.0),
    "tom": (80.0, 3000.0),
}

# Map drum sub-bands to Clone Hero drum note lanes
# .chart drum notes: 0=kick, 1=red(snare), 2=yellow(hihat), 3=blue(tom), 4=green(cymbal/floor tom)
DRUM_LANE_MAP: dict[str, int] = {
    "kick": 0,
    "snare": 1,
    "hihat": 2,
    "tom": 3,
}

# Clone Hero lane constants for guitar/bass
# Standard fret notes: Green=0, Red=1, Yellow=2, Blue=3, Orange=4
# Open note (purple bar): 7
LANE_GREEN = 0
LANE_RED = 1
LANE_YELLOW = 2
LANE_BLUE = 3
LANE_ORANGE = 4
LANE_OPEN = 7  # Purple bar — open string, used heavily in metal

# Number of fretted lanes (0-4) plus open (7)
FRETTED_LANES = 5  # 0,1,2,3,4
TOTAL_LANES = 6  # 0,1,2,3,4 + 7(open)


# ---------------------------------------------------------------------------
# Data classes for separation results
# ---------------------------------------------------------------------------


@dataclass
class StemAnalysis:
    """Results from analysing a single instrument stem."""

    instrument: str
    onset_times: list[float] = field(default_factory=list)
    onset_strengths: list[float] = field(default_factory=list)

    # Pitch information (for melodic instruments: guitar, bass, vocals)
    # Values 0.0–1.0 representing relative pitch height at each onset
    pitch_contour: list[float] = field(default_factory=list)

    # Detected fundamental frequencies at each onset (Hz) via pYIN
    # NaN/0.0 indicates unvoiced/unpitched frames
    fundamental_freqs: list[float] = field(default_factory=list)

    # Per-onset flag: True if this onset should be an open note (lane 7)
    # Determined by pitch being in the lowest region of the song's range
    is_open_note: list[bool] = field(default_factory=list)

    # Spectral centroid at each onset (Hz) — secondary pitch indicator
    spectral_centroids: list[float] = field(default_factory=list)

    # For drums: per-onset lane assignments based on frequency band detection
    drum_lanes: list[int] = field(default_factory=list)

    # Sub-band onset data (drums only): dict of band_name -> (times, strengths)
    drum_band_onsets: dict[str, tuple[list[float], list[float]]] = field(
        default_factory=dict
    )

    # The separated audio signal and sample rate (for optional playback/debug)
    stem_signal: np.ndarray | None = field(default=None, repr=False)
    sample_rate: int = 22050

    # Whether beat-aligned supplementation was applied
    beat_aligned: bool = False


@dataclass
class SeparationResult:
    """Container for all separated stems from a single audio file."""

    harmonic: np.ndarray | None = field(default=None, repr=False)
    percussive: np.ndarray | None = field(default=None, repr=False)
    full_signal: np.ndarray | None = field(default=None, repr=False)
    sample_rate: int = 22050
    duration: float = 0.0


# ---------------------------------------------------------------------------
# Core separation functions
# ---------------------------------------------------------------------------


def separate_stems(
    file_path: str,
    sr: int | None = None,
) -> SeparationResult:
    """
    Perform Harmonic-Percussive Source Separation on an audio file.

    Parameters
    ----------
    file_path : str
        Path to the audio file.
    sr : int, optional
        Target sample rate.  ``None`` preserves the original rate.

    Returns
    -------
    SeparationResult
        Contains the harmonic and percussive components plus metadata.
    """
    logger.info("🎛️ Separating stems: {}", file_path)

    y, sample_rate = librosa.load(file_path, sr=sr, mono=True)
    sample_rate = int(sample_rate)
    duration = float(librosa.get_duration(y=y, sr=sample_rate))

    # Two-pass HPSS: first pass with wide margin, second pass to refine
    # margin=3 gives a clean harmonic/percussive split
    harmonic, percussive = librosa.effects.hpss(y, margin=3.0)

    logger.info(
        "✅ HPSS complete: sr={}, duration={:.1f}s, "
        "harmonic_rms={:.4f}, percussive_rms={:.4f}",
        sample_rate,
        duration,
        float(np.sqrt(np.mean(harmonic**2))),
        float(np.sqrt(np.mean(percussive**2))),
    )

    return SeparationResult(
        harmonic=harmonic,
        percussive=percussive,
        full_signal=y,
        sample_rate=sample_rate,
        duration=duration,
    )


def bandpass_filter(
    y: np.ndarray,
    sr: int,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    """
    Apply a band-pass filter using STFT masking.

    This is more robust than a simple Butterworth filter for music signals
    because it operates in the frequency domain and avoids phase artefacts.

    Parameters
    ----------
    y : np.ndarray
        Audio signal.
    sr : int
        Sample rate.
    low_hz : float
        Lower cutoff frequency in Hz.
    high_hz : float
        Upper cutoff frequency in Hz.

    Returns
    -------
    np.ndarray
        Filtered audio signal.
    """
    # Compute STFT
    n_fft = 2048
    stft = librosa.stft(y, n_fft=n_fft)

    # Build frequency mask
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    mask = np.zeros_like(freqs)

    # Smooth roll-off (raised cosine) to avoid artefacts
    for i, f in enumerate(freqs):
        if f < low_hz:
            # Below band — taper off
            rolloff = low_hz * 0.2  # 20% rolloff width
            if f > low_hz - rolloff and rolloff > 0:
                mask[i] = 0.5 * (1 + np.cos(np.pi * (low_hz - f) / rolloff))
        elif f > high_hz:
            # Above band — taper off
            rolloff = high_hz * 0.2
            if f < high_hz + rolloff and rolloff > 0:
                mask[i] = 0.5 * (1 + np.cos(np.pi * (f - high_hz) / rolloff))
        else:
            mask[i] = 1.0

    # Apply mask to STFT magnitude
    mask_2d = mask[:, np.newaxis]
    filtered_stft = stft * mask_2d

    # Inverse STFT
    return librosa.istft(filtered_stft, length=len(y))


# ---------------------------------------------------------------------------
# Instrument-specific analysis
# ---------------------------------------------------------------------------


def _compute_rms_at_times(
    y: np.ndarray,
    sr: int,
    times: list[float],
    window_sec: float = 0.05,
) -> list[float]:
    """
    Compute RMS energy of a signal at specific time points.

    Used to measure how much energy the isolated stem has at each beat,
    so we can decide whether a beat should become a note.

    Parameters
    ----------
    y : np.ndarray
        Audio signal (stem).
    sr : int
        Sample rate.
    times : list of float
        Time points in seconds.
    window_sec : float
        Window size in seconds around each time point.

    Returns
    -------
    list of float
        RMS energy values at each time point.
    """
    half_win = int(sr * window_sec / 2)
    result: list[float] = []

    for t in times:
        center = int(t * sr)
        start = max(0, center - half_win)
        end = min(len(y), center + half_win)
        if end <= start:
            result.append(0.0)
        else:
            segment = y[start:end]
            rms = float(np.sqrt(np.mean(segment**2)))
            result.append(rms)

    return result


def _beat_aligned_supplement(
    stem_onset_times: list[float],
    stem_onset_strengths: list[float],
    stem_signal: np.ndarray,
    sr: int,
    beat_times: list[float],
    energy_threshold: float = 0.15,
    merge_window: float = 0.05,
) -> tuple[list[float], list[float]]:
    """
    Supplement stem-detected onsets with the full mix's beat grid.

    HPSS + band-pass filtering can aggressively strip energy, especially
    for distorted instruments (metal guitar, overdriven bass).  This
    function adds beat positions where the stem signal has significant
    energy but no onset was detected.

    This ensures the chart has adequate note density — notes land on
    musical beats where the instrument is actually playing.

    Parameters
    ----------
    stem_onset_times : list of float
        Onset times detected from the isolated stem.
    stem_onset_strengths : list of float
        Corresponding onset strengths.
    stem_signal : np.ndarray
        The isolated stem audio signal.
    sr : int
        Sample rate.
    beat_times : list of float
        Full-mix beat positions in seconds.
    energy_threshold : float
        Minimum normalised RMS energy at a beat for it to be included.
        Range 0.0–1.0.  Lower = more permissive (more notes).
    merge_window : float
        If a beat is within this many seconds of an existing onset,
        skip it (avoid duplicates).

    Returns
    -------
    tuple of (list of float, list of float)
        Merged (onset_times, onset_strengths) sorted by time.
    """
    if stem_signal is None or len(beat_times) == 0:
        return stem_onset_times, stem_onset_strengths

    # Compute stem energy at each beat position
    beat_energies = _compute_rms_at_times(stem_signal, sr, beat_times)

    # Normalise energies
    max_energy = max(beat_energies) if beat_energies else 1.0
    if max_energy <= 0:
        return stem_onset_times, stem_onset_strengths

    norm_energies = [e / max_energy for e in beat_energies]

    # Build set of existing onset times for quick lookup
    existing_times = set()
    for t in stem_onset_times:
        existing_times.add(round(t, 3))

    # Supplement: add beats where the stem has energy but no onset was detected
    new_times: list[float] = list(stem_onset_times)
    new_strengths: list[float] = list(stem_onset_strengths)

    added = 0
    for bt, energy in zip(beat_times, norm_energies):
        if energy < energy_threshold:
            continue

        # Check if an existing onset is already close to this beat
        too_close = False
        for et in stem_onset_times:
            if abs(et - bt) < merge_window:
                too_close = True
                break
        if too_close:
            continue

        new_times.append(bt)
        new_strengths.append(energy * 0.8)  # slightly lower weight than detected onsets
        added += 1

    if added > 0:
        # Sort by time
        paired = sorted(zip(new_times, new_strengths), key=lambda x: x[0])
        new_times = [p[0] for p in paired]
        new_strengths = [p[1] for p in paired]
        logger.info(
            "🎯 Beat-aligned supplement: added {} beat-onsets to {} stem-onsets "
            "(total: {})",
            added,
            len(stem_onset_times),
            len(new_times),
        )

    return new_times, new_strengths


def analyze_guitar_stem(
    separation: SeparationResult,
    sensitivity: float = 0.5,
    beat_times: list[float] | None = None,
) -> StemAnalysis:
    """
    Analyse the guitar component from a harmonic stem.

    Uses band-pass filtering to isolate guitar frequencies, then detects
    onsets and extracts pitch contour for lane mapping.  For distorted
    guitar (metal, hard rock), also includes the percussive component
    (palm mutes, chugs are percussive transients).

    Beat-aligned supplementation adds notes at beat positions where the
    guitar stem has energy, ensuring adequate note density.

    Parameters
    ----------
    separation : SeparationResult
        Pre-computed HPSS result.
    sensitivity : float
        Onset detection sensitivity (0.0–1.0).  Higher values detect more
        onsets (busier charts).  Default 0.5.
    beat_times : list of float, optional
        Full-mix beat positions for beat-aligned supplementation.

    Returns
    -------
    StemAnalysis
        Guitar-specific onset and pitch data.
    """
    sr = separation.sample_rate
    harmonic = separation.harmonic

    if harmonic is None:
        logger.warning("⚠️ No harmonic stem available for guitar analysis")
        return StemAnalysis(instrument="guitar", sample_rate=sr)

    # Band-pass to guitar range
    low_hz, high_hz = FREQ_BANDS["guitar"]
    guitar_signal = bandpass_filter(harmonic, sr, low_hz, high_hz)

    # For heavy/distorted guitar, also include percussive transients
    # (palm mutes, chugs, pick attack) filtered to guitar range
    if separation.percussive is not None:
        perc_guitar = bandpass_filter(separation.percussive, sr, low_hz, high_hz)
        # Mix: 70% harmonic, 30% percussive
        guitar_signal = 0.7 * guitar_signal + 0.3 * perc_guitar

    # Onset detection on the isolated guitar signal
    onset_env = librosa.onset.onset_strength(y=guitar_signal, sr=sr)
    # Dynamic threshold based on sensitivity
    # Lower delta = more onsets detected
    delta = 0.05 + (1.0 - sensitivity) * 0.25
    onset_frames = librosa.onset.onset_detect(
        y=guitar_signal,
        sr=sr,
        onset_envelope=onset_env,
        delta=delta,
        wait=3,  # minimum frames between onsets (reduced from 4)
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr).tolist()

    # Normalise onset strengths
    if len(onset_frames) > 0:
        raw_strengths = onset_env[onset_frames]
        max_s = float(np.max(raw_strengths)) if np.max(raw_strengths) > 0 else 1.0
        onset_strengths = (raw_strengths / max_s).tolist()
    else:
        onset_strengths = []

    # Beat-aligned supplementation for better note density
    used_beat_align = False
    if beat_times:
        orig_count = len(onset_times)
        onset_times, onset_strengths = _beat_aligned_supplement(
            stem_onset_times=onset_times,
            stem_onset_strengths=onset_strengths,
            stem_signal=guitar_signal,
            sr=sr,
            beat_times=beat_times,
            energy_threshold=0.10,
        )
        used_beat_align = len(onset_times) > orig_count

    # --- pYIN pitch detection (replaces spectral centroid) ---
    # pYIN gives actual fundamental frequencies, which are far more
    # reliable than spectral centroids for distorted instruments.
    fund_freqs, is_open, pitch_contour = _detect_pitch_pyin(
        guitar_signal,
        sr,
        onset_times,
        low_hz,
        high_hz,
        open_threshold_semitones=0.8,  # within ~1 semitone of lowest = open
    )

    # Also compute spectral centroids as a secondary/fallback indicator
    onset_frames_for_spectral = librosa.time_to_frames(np.array(onset_times), sr=sr)
    spectral_centroids = _compute_spectral_centroids_at_onsets(
        guitar_signal, sr, onset_frames_for_spectral
    )

    n_open = sum(1 for o in is_open if o)
    logger.info(
        "🎸 Guitar analysis: {} onsets (beat-aligned={}, open_notes={})",
        len(onset_times),
        used_beat_align,
        n_open,
    )

    return StemAnalysis(
        instrument="guitar",
        onset_times=onset_times,
        onset_strengths=onset_strengths,
        pitch_contour=pitch_contour,
        fundamental_freqs=fund_freqs,
        is_open_note=is_open,
        spectral_centroids=spectral_centroids,
        stem_signal=guitar_signal,
        sample_rate=sr,
        beat_aligned=used_beat_align,
    )


def analyze_bass_stem(
    separation: SeparationResult,
    sensitivity: float = 0.5,
    beat_times: list[float] | None = None,
) -> StemAnalysis:
    """
    Analyse the bass guitar component from a harmonic stem.

    Similar to guitar but focused on lower frequencies, with wider onset
    detection windows (bass notes tend to be more sustained).

    Beat-aligned supplementation ensures adequate note density.

    Parameters
    ----------
    separation : SeparationResult
        Pre-computed HPSS result.
    sensitivity : float
        Onset detection sensitivity (0.0–1.0).  Default 0.5.
    beat_times : list of float, optional
        Full-mix beat positions for beat-aligned supplementation.

    Returns
    -------
    StemAnalysis
        Bass-specific onset and pitch data.
    """
    sr = separation.sample_rate
    harmonic = separation.harmonic

    if harmonic is None:
        logger.warning("⚠️ No harmonic stem available for bass analysis")
        return StemAnalysis(instrument="bass", sample_rate=sr)

    # Band-pass to bass range
    low_hz, high_hz = FREQ_BANDS["bass"]
    bass_signal = bandpass_filter(harmonic, sr, low_hz, high_hz)

    # Also include percussive bass transients (slap, pick attack)
    if separation.percussive is not None:
        perc_bass = bandpass_filter(separation.percussive, sr, low_hz, high_hz)
        bass_signal = 0.75 * bass_signal + 0.25 * perc_bass

    # Onset detection — bass needs wider windows and higher wait
    onset_env = librosa.onset.onset_strength(
        y=bass_signal,
        sr=sr,
        aggregate=np.median,  # median aggregation reduces sensitivity to noise
    )
    delta = 0.08 + (1.0 - sensitivity) * 0.25
    onset_frames = librosa.onset.onset_detect(
        y=bass_signal,
        sr=sr,
        onset_envelope=onset_env,
        delta=delta,
        wait=6,  # bass notes are more spaced out (reduced from 8)
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr).tolist()

    # Normalise onset strengths
    if len(onset_frames) > 0:
        raw_strengths = onset_env[onset_frames]
        max_s = float(np.max(raw_strengths)) if np.max(raw_strengths) > 0 else 1.0
        onset_strengths = (raw_strengths / max_s).tolist()
    else:
        onset_strengths = []

    # Beat-aligned supplementation for better note density
    used_beat_align = False
    if beat_times:
        orig_count = len(onset_times)
        onset_times, onset_strengths = _beat_aligned_supplement(
            stem_onset_times=onset_times,
            stem_onset_strengths=onset_strengths,
            stem_signal=bass_signal,
            sr=sr,
            beat_times=beat_times,
            energy_threshold=0.12,
        )
        used_beat_align = len(onset_times) > orig_count

    # --- pYIN pitch detection for bass ---
    # Bass uses even more open notes (drop tuning is standard in metal bass)
    fund_freqs, is_open, pitch_contour = _detect_pitch_pyin(
        bass_signal,
        sr,
        onset_times,
        low_hz,
        high_hz,
        open_threshold_semitones=1.0,  # within 1 semitone of lowest = open (bass uses more open)
    )

    # Secondary spectral centroids
    onset_frames_for_spectral = librosa.time_to_frames(np.array(onset_times), sr=sr)
    spectral_centroids = _compute_spectral_centroids_at_onsets(
        bass_signal, sr, onset_frames_for_spectral
    )

    n_open = sum(1 for o in is_open if o)
    logger.info(
        "🎸 Bass analysis: {} onsets (beat-aligned={}, open_notes={})",
        len(onset_times),
        used_beat_align,
        n_open,
    )

    return StemAnalysis(
        instrument="bass",
        onset_times=onset_times,
        onset_strengths=onset_strengths,
        pitch_contour=pitch_contour,
        fundamental_freqs=fund_freqs,
        is_open_note=is_open,
        spectral_centroids=spectral_centroids,
        stem_signal=bass_signal,
        sample_rate=sr,
        beat_aligned=used_beat_align,
    )


def analyze_drums_stem(
    separation: SeparationResult,
    sensitivity: float = 0.5,
    beat_times: list[float] | None = None,
) -> StemAnalysis:
    """
    Analyse the drum component from the percussive stem.

    Splits the percussive signal into sub-bands (kick, snare, hi-hat, toms)
    and detects onsets independently in each band.  Each onset is assigned
    a drum lane for charting.

    Beat-aligned supplementation ensures adequate note density.

    Parameters
    ----------
    separation : SeparationResult
        Pre-computed HPSS result.
    sensitivity : float
        Onset detection sensitivity (0.0–1.0).  Default 0.5.
    beat_times : list of float, optional
        Full-mix beat positions for beat-aligned supplementation.

    Returns
    -------
    StemAnalysis
        Drum-specific onset data with per-onset lane assignments.
    """
    sr = separation.sample_rate
    percussive = separation.percussive

    if percussive is None:
        logger.warning("⚠️ No percussive stem available for drum analysis")
        return StemAnalysis(instrument="drums", sample_rate=sr)

    # Detect onsets in each drum frequency band
    all_onsets: list[tuple[float, float, int, str]] = []  # (time, strength, lane, band)
    band_onset_data: dict[str, tuple[list[float], list[float]]] = {}

    for band_name, (low_hz, high_hz) in DRUM_BANDS.items():
        band_signal = bandpass_filter(percussive, sr, low_hz, high_hz)

        # Drum onsets need to be sharp — use a spectral flux onset function
        onset_env = librosa.onset.onset_strength(y=band_signal, sr=sr)

        # Different sensitivity per band — lowered deltas for more detections
        if band_name == "kick":
            delta = 0.10 + (1.0 - sensitivity) * 0.20
            wait = 4  # kick drum has minimum spacing (reduced from 6)
        elif band_name == "snare":
            delta = 0.08 + (1.0 - sensitivity) * 0.25
            wait = 3  # reduced from 4
        elif band_name == "hihat":
            delta = 0.15 + (1.0 - sensitivity) * 0.20
            wait = 2  # hi-hats can be rapid
        else:  # tom
            delta = 0.12 + (1.0 - sensitivity) * 0.25
            wait = 4  # reduced from 5

        onset_frames = librosa.onset.onset_detect(
            y=band_signal,
            sr=sr,
            onset_envelope=onset_env,
            delta=delta,
            wait=wait,
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr).tolist()

        # Normalise strengths within this band
        if len(onset_frames) > 0:
            raw_strengths = onset_env[onset_frames]
            max_s = float(np.max(raw_strengths)) if np.max(raw_strengths) > 0 else 1.0
            strengths = (raw_strengths / max_s).tolist()
        else:
            strengths = []

        band_onset_data[band_name] = (onset_times, strengths)
        lane = DRUM_LANE_MAP[band_name]

        for t, s in zip(onset_times, strengths):
            all_onsets.append((t, s, lane, band_name))

    # Sort all drum onsets by time
    all_onsets.sort(key=lambda x: x[0])

    # Merge simultaneous hits (within ~20ms tolerance)
    merged_times: list[float] = []
    merged_strengths: list[float] = []
    merged_lanes: list[int] = []
    merge_window = 0.020  # 20ms

    i = 0
    while i < len(all_onsets):
        t, s, lane, _ = all_onsets[i]

        # Collect all simultaneous hits
        simultaneous_lanes = [lane]
        max_strength = s

        j = i + 1
        while j < len(all_onsets) and all_onsets[j][0] - t < merge_window:
            simultaneous_lanes.append(all_onsets[j][2])
            max_strength = max(max_strength, all_onsets[j][1])
            j += 1

        # For charting, we pick the strongest/most prominent lane
        # but we could also emit chords (multiple simultaneous drum hits)
        # For now, pick the most "interesting" lane (prefer snare > kick > tom > hihat)
        lane_priority = {1: 4, 0: 3, 3: 2, 2: 1}  # snare > kick > tom > hihat
        best_lane = max(
            set(simultaneous_lanes), key=lambda ln: lane_priority.get(ln, 0)
        )

        merged_times.append(t)
        merged_strengths.append(max_strength)
        merged_lanes.append(best_lane)

        i = j

    # Beat-aligned supplementation for drums
    used_beat_align = False
    if beat_times and percussive is not None:
        orig_count = len(merged_times)
        merged_times, merged_strengths = _beat_aligned_supplement(
            stem_onset_times=merged_times,
            stem_onset_strengths=merged_strengths,
            stem_signal=percussive,
            sr=sr,
            beat_times=beat_times,
            energy_threshold=0.12,
        )
        # Assign default drum lane (snare=1) for supplemented beats
        while len(merged_lanes) < len(merged_times):
            merged_lanes.append(1)  # default to snare for beat-aligned additions
        used_beat_align = len(merged_times) > orig_count

    logger.info(
        "🥁 Drum analysis: {} total onsets (beat-aligned={}, kick={}, snare={}, hihat={}, tom={})",
        len(merged_times),
        used_beat_align,
        len(band_onset_data.get("kick", ([], []))[0]),
        len(band_onset_data.get("snare", ([], []))[0]),
        len(band_onset_data.get("hihat", ([], []))[0]),
        len(band_onset_data.get("tom", ([], []))[0]),
    )

    return StemAnalysis(
        instrument="drums",
        onset_times=merged_times,
        onset_strengths=merged_strengths,
        drum_lanes=merged_lanes,
        drum_band_onsets=band_onset_data,
        stem_signal=percussive,
        sample_rate=sr,
        beat_aligned=used_beat_align,
    )


def analyze_vocals_stem(
    separation: SeparationResult,
    sensitivity: float = 0.5,
    beat_times: list[float] | None = None,
) -> StemAnalysis:
    """
    Analyse the vocal component from the harmonic stem.

    Isolates the vocal frequency range and detects note onsets.  The pitch
    contour can be used for vocal-driven guitar charting or future vocal
    chart support.

    Beat-aligned supplementation ensures adequate note density.

    Parameters
    ----------
    separation : SeparationResult
        Pre-computed HPSS result.
    sensitivity : float
        Onset detection sensitivity (0.0–1.0).  Default 0.5.
    beat_times : list of float, optional
        Full-mix beat positions for beat-aligned supplementation.

    Returns
    -------
    StemAnalysis
        Vocal-specific onset and pitch data.
    """
    sr = separation.sample_rate
    harmonic = separation.harmonic

    if harmonic is None:
        logger.warning("⚠️ No harmonic stem available for vocal analysis")
        return StemAnalysis(instrument="vocals", sample_rate=sr)

    # Band-pass to vocal range
    low_hz, high_hz = FREQ_BANDS["vocals"]
    vocal_signal = bandpass_filter(harmonic, sr, low_hz, high_hz)

    # Onset detection — vocals need sensitivity to melodic changes
    onset_env = librosa.onset.onset_strength(y=vocal_signal, sr=sr)
    delta = 0.05 + (1.0 - sensitivity) * 0.30
    onset_frames = librosa.onset.onset_detect(
        y=vocal_signal,
        sr=sr,
        onset_envelope=onset_env,
        delta=delta,
        wait=4,  # vocal phrases (reduced from 6)
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr).tolist()

    # Normalise onset strengths
    if len(onset_frames) > 0:
        raw_strengths = onset_env[onset_frames]
        max_s = float(np.max(raw_strengths)) if np.max(raw_strengths) > 0 else 1.0
        onset_strengths = (raw_strengths / max_s).tolist()
    else:
        onset_strengths = []

    # Beat-aligned supplementation for better note density
    used_beat_align = False
    if beat_times:
        orig_count = len(onset_times)
        onset_times, onset_strengths = _beat_aligned_supplement(
            stem_onset_times=onset_times,
            stem_onset_strengths=onset_strengths,
            stem_signal=vocal_signal,
            sr=sr,
            beat_times=beat_times,
            energy_threshold=0.12,
        )
        used_beat_align = len(onset_times) > orig_count

    # Recompute onset frames for spectral analysis after supplementation
    onset_frames_for_spectral = librosa.time_to_frames(np.array(onset_times), sr=sr)

    # Spectral centroids for pitch mapping
    spectral_centroids = _compute_spectral_centroids_at_onsets(
        vocal_signal, sr, onset_frames_for_spectral
    )
    pitch_contour = _centroids_to_pitch_contour(spectral_centroids, low_hz, high_hz)

    logger.info(
        "🎤 Vocal analysis: {} onsets (beat-aligned={}), avg_centroid={:.0f} Hz",
        len(onset_times),
        used_beat_align,
        np.mean(spectral_centroids) if spectral_centroids else 0,
    )

    return StemAnalysis(
        instrument="vocals",
        onset_times=onset_times,
        onset_strengths=onset_strengths,
        pitch_contour=pitch_contour,
        spectral_centroids=spectral_centroids,
        stem_signal=vocal_signal,
        sample_rate=sr,
        beat_aligned=used_beat_align,
    )


# ---------------------------------------------------------------------------
# High-level analysis dispatcher
# ---------------------------------------------------------------------------


def analyze_instrument(
    file_path: str,
    instrument: str = "guitar",
    sensitivity: float = 0.5,
    sr: int | None = None,
    beat_times: list[float] | None = None,
    use_demucs: bool | None = None,
) -> tuple[StemAnalysis, SeparationResult]:
    """
    Full pipeline: load audio → separate → analyse the target instrument.

    This is the main entry point for instrument-specific analysis.

    **Demucs-first strategy**: When Demucs is installed and ``use_demucs``
    is not explicitly ``False``, this function uses Demucs for stem
    separation (far superior to HPSS for isolating individual instruments).
    The isolated stem is then passed through the same instrument-specific
    analyser (onset detection, pYIN pitch, etc.) for note-level detail.

    Falls back to librosa HPSS if Demucs is unavailable or fails.

    Parameters
    ----------
    file_path : str
        Path to the audio file.
    instrument : str
        One of ``guitar``, ``bass``, ``drums``, ``vocals``, ``full_mix``.
    sensitivity : float
        Onset detection sensitivity (0.0–1.0).  Higher = more onsets.
        Default raised to 0.5 for better note density.
    sr : int, optional
        Target sample rate (None = auto).
    beat_times : list of float, optional
        Full-mix beat positions in seconds.  When provided, these are
        used for beat-aligned onset supplementation — stem-detected
        onsets are supplemented with beat positions where the stem has
        energy but no onset was detected.  This greatly improves note
        density for distorted / heavy instruments.
    use_demucs : bool, optional
        If ``True``, force Demucs usage (error if unavailable).
        If ``False``, skip Demucs and use HPSS only.
        If ``None`` (default), auto-detect: use Demucs if installed,
        otherwise fall back to HPSS.

    Returns
    -------
    tuple of (StemAnalysis, SeparationResult)
        The instrument-specific analysis and the raw separation data.

    Raises
    ------
    ValueError
        If the instrument is not recognised.
    """
    instrument = instrument.lower().strip()

    if instrument == "full_mix":
        # Legacy behaviour: return full-mix onset analysis (no separation)
        return _analyze_full_mix(file_path, sr), SeparationResult()

    analyzers = {
        "guitar": analyze_guitar_stem,
        "bass": analyze_bass_stem,
        "drums": analyze_drums_stem,
        "vocals": analyze_vocals_stem,
    }

    if instrument not in analyzers:
        raise ValueError(
            f"Unknown instrument '{instrument}'. "
            f"Supported: {', '.join(analyzers.keys())}, full_mix"
        )

    # --- Demucs-first strategy ---
    # Try Demucs for high-quality stem isolation when available.
    # Demucs produces far cleaner instrument stems than HPSS + bandpass,
    # especially for distorted guitar, dense metal mixes, and polyphonic
    # instruments.  The isolated stem is then analysed with the same
    # onset/pitch pipeline used for HPSS stems.
    demucs_stems: dict[str, Path] | None = None
    should_try_demucs = (use_demucs is True) or (
        use_demucs is None and demucs_available()
    )

    if should_try_demucs:
        demucs_stem_name = INSTRUMENT_TO_DEMUCS_STEM.get(instrument)
        if demucs_stem_name:
            logger.info(
                "🎛️ Attempting Demucs separation for '{}' (stem='{}')...",
                instrument,
                demucs_stem_name,
            )
            demucs_stems = separate_with_demucs(file_path)

            if demucs_stems and demucs_stem_name in demucs_stems:
                stem_path = demucs_stems[demucs_stem_name]
                logger.info(
                    "✅ Using Demucs '{}' stem for {} analysis",
                    demucs_stem_name,
                    instrument,
                )
                # Load the Demucs stem and run light HPSS on it
                separation = _load_demucs_stem_as_separation(stem_path, sr=sr)

                analysis = analyzers[instrument](
                    separation,
                    sensitivity=sensitivity,
                    beat_times=beat_times,
                )

                # Clean up Demucs temp files
                try:
                    import shutil

                    # stem_path is inside a temp dir tree: .../demucs_xxx/model/songname/stem.wav
                    # Walk up to the demucs_xxx root to clean everything
                    demucs_root = stem_path.parent.parent.parent
                    if demucs_root.name.startswith("demucs_"):
                        shutil.rmtree(demucs_root, ignore_errors=True)
                except Exception:
                    pass

                return analysis, separation
            else:
                if use_demucs is True:
                    logger.error(
                        "❌ Demucs was required but failed for instrument '{}'",
                        instrument,
                    )
                else:
                    logger.warning(
                        "⚠️ Demucs separation failed — falling back to HPSS for '{}'",
                        instrument,
                    )

    # --- Fallback: HPSS-based separation ---
    separation = separate_stems(file_path, sr=sr)

    analysis = analyzers[instrument](
        separation,
        sensitivity=sensitivity,
        beat_times=beat_times,
    )
    return analysis, separation


def _analyze_full_mix(
    file_path: str,
    sr: int | None = None,
) -> StemAnalysis:
    """
    Analyse the full mix without separation (legacy behaviour).

    Returns a StemAnalysis with onset data from the unseparated signal.
    """
    y, sample_rate = librosa.load(file_path, sr=sr, mono=True)
    sample_rate = int(sample_rate)

    onset_env = librosa.onset.onset_strength(y=y, sr=sample_rate)
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sample_rate, onset_envelope=onset_env
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate).tolist()

    if len(onset_frames) > 0:
        raw_strengths = onset_env[onset_frames]
        max_s = float(np.max(raw_strengths)) if np.max(raw_strengths) > 0 else 1.0
        onset_strengths = (raw_strengths / max_s).tolist()
    else:
        onset_strengths = []

    logger.info("🎵 Full-mix analysis: {} onsets", len(onset_times))

    return StemAnalysis(
        instrument="full_mix",
        onset_times=onset_times,
        onset_strengths=onset_strengths,
        stem_signal=y,
        sample_rate=sample_rate,
    )


# ---------------------------------------------------------------------------
# Semitone-based lane mapping (derived from real guitar tab analysis)
# ---------------------------------------------------------------------------
#
# Real metal guitar tabs (e.g. Paleface Swiss "Please End Me" in Drop G#)
# show that notes cluster tightly around the lowest few frets:
#
#   Fret 0 (open string palm mutes / chugs) → Open  (N 7, purple bar)
#   Fret 1                                  → Green (N 0)
#   Fret 2                                  → Red   (N 1)
#   Fret 3                                  → Yellow(N 2)
#   Fret 4                                  → Blue  (N 3)
#   Fret 5+                                 → Orange(N 4)
#
# One guitar fret = one semitone, so we detect the *semitone distance*
# of each onset's fundamental frequency from the song's lowest detected
# pitch and use that as the lane selector.  This gives musically correct
# charting that mirrors how a real guitarist reads tabs.
#
# Common metal patterns this captures correctly:
#   0M-0M-0M          → Open-Open-Open  (chugs)
#   1M-0M-0M          → Green-Open-Open (gallop)
#   3M-2M-3M-4M       → Yellow-Red-Yellow-Blue (chromatic run)
#   4---3---0M-0M-0M   → Blue-Yellow-Open-Open-Open (riff into chug)
#   0h-1h-0-3h-4h-3   → Open-Green-Open-Yellow-Blue-Yellow (HOPOs)

# Semitone → Clone Hero lane mapping table
# Index = semitones above lowest pitch; value = CH lane number
SEMITONE_TO_LANE: list[int] = [
    LANE_OPEN,  # 0 semitones = open string
    LANE_GREEN,  # 1 semitone  = fret 1
    LANE_RED,  # 2 semitones = fret 2
    LANE_YELLOW,  # 3 semitones = fret 3
    LANE_BLUE,  # 4 semitones = fret 4
    LANE_ORANGE,  # 5 semitones = fret 5
    # Everything above 5 semitones also maps to Orange
]


def _hz_to_semitones(freq: float, ref_freq: float) -> float:
    """Convert a frequency to semitones above a reference frequency.

    Parameters
    ----------
    freq : float
        The frequency to convert (Hz).  Must be > 0.
    ref_freq : float
        The reference (lowest) frequency (Hz).  Must be > 0.

    Returns
    -------
    float
        Number of semitones above the reference.  Can be fractional.
    """
    if freq <= 0 or ref_freq <= 0:
        return 0.0
    return 12.0 * np.log2(freq / ref_freq)


def _semitones_to_lane(semitones: float) -> int:
    """Map a semitone distance to a Clone Hero lane.

    Uses the ``SEMITONE_TO_LANE`` table: 0 st → Open (7),
    1 → Green (0), 2 → Red (1), 3 → Yellow (2), 4 → Blue (3),
    5+ → Orange (4).

    Fractional semitones are rounded to the nearest integer.

    Parameters
    ----------
    semitones : float
        Distance in semitones from the song's lowest pitch.

    Returns
    -------
    int
        Clone Hero lane number (0–4 or 7).
    """
    idx = max(0, round(semitones))
    if idx < len(SEMITONE_TO_LANE):
        return SEMITONE_TO_LANE[idx]
    return LANE_ORANGE  # anything ≥ 5 semitones → Orange


# ---------------------------------------------------------------------------
# pYIN pitch detection
# ---------------------------------------------------------------------------


def _detect_pitch_pyin(
    y: np.ndarray,
    sr: int,
    onset_times: list[float],
    low_hz: float,
    high_hz: float,
    open_threshold_semitones: float = 0.8,
    hop_length: int = 512,
) -> tuple[list[float], list[bool], list[float]]:
    """
    Detect fundamental frequencies at onset positions using pYIN and map
    each onset to a Clone Hero lane via semitone distance from the lowest
    detected pitch.

    pYIN is a probabilistic variant of YIN that gives reliable fundamental
    frequency estimates even for distorted instruments (unlike spectral
    centroid which just measures brightness).

    The mapping mirrors real guitar tab → Clone Hero charting:

        0 semitones from lowest pitch  → Open  (N 7, purple bar)
        1 semitone                     → Green (N 0)
        2 semitones                    → Red   (N 1)
        3 semitones                    → Yellow(N 2)
        4 semitones                    → Blue  (N 3)
        5+ semitones                   → Orange(N 4)

    Unvoiced / unpitched onsets (where pYIN can't detect a fundamental)
    are treated as open notes — in metal these are typically percussive
    palm mutes where the pitch is damped.

    Parameters
    ----------
    y : np.ndarray
        Audio signal (isolated stem).
    sr : int
        Sample rate.
    onset_times : list of float
        Onset positions in seconds.
    low_hz : float
        Lower frequency bound for the instrument.
    high_hz : float
        Upper frequency bound for the instrument.
    open_threshold_semitones : float
        Onsets within this many semitones of the lowest detected pitch
        are classified as open notes.  Default 0.8 (anything less than
        ~a semitone above the lowest note = open).
    hop_length : int
        Hop length for pYIN analysis.

    Returns
    -------
    tuple of (fund_freqs, is_open_note, pitch_contour)
        fund_freqs : list of float
            Fundamental frequency (Hz) at each onset. 0.0 = unvoiced.
        is_open_note : list of bool
            True if the onset should be an open note (lane 7).
        pitch_contour : list of float
            Normalised pitch values 0.0–1.0 for compatibility with
            the legacy ``pitch_contour_to_lanes`` code path.
            Open notes get 0.0.
    """
    if not onset_times or len(y) == 0:
        return [], [], []

    # Run pYIN across the whole signal
    fmin = max(low_hz, 30.0)
    fmax = min(high_hz, sr / 2.0 - 1.0)

    try:
        f0, _voiced_flag, _voiced_prob = librosa.pyin(
            y,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            hop_length=hop_length,
            fill_na=0.0,
        )
    except Exception as e:
        logger.warning("⚠️ pYIN pitch detection failed: {} — falling back", e)
        n = len(onset_times)
        return [0.0] * n, [False] * n, [0.5] * n

    if f0 is None or len(f0) == 0:
        n = len(onset_times)
        return [0.0] * n, [False] * n, [0.5] * n

    # Sample f0 at each onset time
    onset_f0: list[float] = []
    for t in onset_times:
        frame = int(t * sr / hop_length)
        frame = max(0, min(frame, len(f0) - 1))
        onset_f0.append(float(f0[frame]))

    # Find the lowest voiced pitch — this is our "open string" reference
    voiced_pitches = [p for p in onset_f0 if p > 0]
    if not voiced_pitches:
        # No pitched content detected — treat everything as open
        n = len(onset_times)
        return onset_f0, [True] * n, [0.0] * n

    lowest_pitch = min(voiced_pitches)
    highest_pitch = max(voiced_pitches)
    total_semitone_range = _hz_to_semitones(highest_pitch, lowest_pitch)

    # Classify each onset using semitone distance
    fund_freqs: list[float] = []
    is_open: list[bool] = []
    pitch_contour: list[float] = []

    for freq in onset_f0:
        fund_freqs.append(freq)

        if freq <= 0:
            # Unvoiced / unpitched — treat as open (percussive palm mute)
            is_open.append(True)
            pitch_contour.append(0.0)
            continue

        semitones = _hz_to_semitones(freq, lowest_pitch)

        if semitones <= open_threshold_semitones:
            # Within ~1 semitone of lowest pitch = open string
            is_open.append(True)
            pitch_contour.append(0.0)
        else:
            is_open.append(False)
            # Normalise to 0.0–1.0 for legacy compatibility
            if total_semitone_range > 0:
                normalised = semitones / total_semitone_range
            else:
                normalised = 0.5
            pitch_contour.append(float(np.clip(normalised, 0.0, 1.0)))

    n_open = sum(1 for o in is_open if o)
    n_voiced = len(voiced_pitches)
    logger.info(
        "🎸 pYIN pitch detection: {} onsets, {} voiced, {} open, "
        "lowest={:.1f} Hz, range={:.1f} semitones",
        len(onset_times),
        n_voiced,
        n_open,
        lowest_pitch,
        total_semitone_range,
    )

    return fund_freqs, is_open, pitch_contour


def _semitone_contour_to_lanes(
    fund_freqs: list[float],
    is_open_note: list[bool],
    smoothing: int = 2,
) -> list[int]:
    """
    Convert fundamental frequencies to Clone Hero lane assignments
    using semitone-distance mapping.

    This is the primary lane-mapping function for guitar/bass and
    replaces the legacy ``pitch_contour_to_lanes`` path when pYIN
    data is available.

    The mapping mirrors real guitar tab → Clone Hero:
        0 semitones → Open (7)    1 semitone → Green (0)
        2 semitones → Red (1)     3 semitones → Yellow (2)
        4 semitones → Blue (3)    5+ semitones → Orange (4)

    A light smoothing pass avoids single-frame jitter while preserving
    genuine lane changes (riff movement).

    Parameters
    ----------
    fund_freqs : list of float
        Fundamental frequency (Hz) at each onset.  0.0 = unvoiced.
    is_open_note : list of bool
        Per-onset flag from ``_detect_pitch_pyin``.
    smoothing : int
        Window size for median smoothing of semitone values before
        lane quantisation.  Set to 1 for no smoothing.

    Returns
    -------
    list of int
        Lane assignments — values in {0,1,2,3,4,7}.
    """
    if not fund_freqs:
        return []

    # Find the lowest voiced frequency as the reference
    voiced = [f for f in fund_freqs if f > 0]
    if not voiced:
        return [LANE_OPEN] * len(fund_freqs)
    lowest = min(voiced)

    # Compute raw semitone distances
    raw_semitones: list[float] = []
    for freq in fund_freqs:
        if freq <= 0:
            raw_semitones.append(0.0)  # will be open anyway
        else:
            raw_semitones.append(_hz_to_semitones(freq, lowest))

    # Light median smoothing to reduce single-frame jitter
    # (but NOT on open notes — those are percussive and deliberate)
    semitones_arr = np.array(raw_semitones, dtype=float)
    if smoothing > 1 and len(semitones_arr) >= smoothing:
        from scipy.ndimage import median_filter

        try:
            smoothed = median_filter(semitones_arr, size=smoothing, mode="nearest")
        except ImportError:
            # scipy not available — skip smoothing
            smoothed = semitones_arr
    else:
        smoothed = semitones_arr

    # Map to lanes
    result: list[int] = []
    for i in range(len(fund_freqs)):
        if is_open_note[i]:
            result.append(LANE_OPEN)
        else:
            result.append(_semitones_to_lane(float(smoothed[i])))

    return result


# ---------------------------------------------------------------------------
# Pitch / spectral helpers
# ---------------------------------------------------------------------------


def _compute_spectral_centroids_at_onsets(
    y: np.ndarray,
    sr: int,
    onset_frames: np.ndarray,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> list[float]:
    """
    Compute the spectral centroid at each onset frame.

    The spectral centroid is the "centre of mass" of the spectrum and serves
    as a proxy for perceived pitch / brightness.

    Parameters
    ----------
    y : np.ndarray
        Audio signal.
    sr : int
        Sample rate.
    onset_frames : np.ndarray
        Frame indices of detected onsets.
    n_fft : int
        FFT window size.
    hop_length : int
        Hop length for STFT (must match librosa's default).

    Returns
    -------
    list of float
        Spectral centroid in Hz at each onset.
    """
    if len(onset_frames) == 0:
        return []

    # Compute full spectrogram centroid
    centroids = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )[0]

    result = []
    for frame in onset_frames:
        if frame < len(centroids):
            result.append(float(centroids[frame]))
        else:
            result.append(float(centroids[-1]) if len(centroids) > 0 else 0.0)

    return result


def _centroids_to_pitch_contour(
    centroids: list[float],
    low_hz: float,
    high_hz: float,
) -> list[float]:
    """
    Normalise spectral centroids to a 0.0–1.0 pitch contour.

    Values are clamped and scaled so that ``low_hz`` maps to 0.0 and
    ``high_hz`` maps to 1.0.  Uses logarithmic scaling since human pitch
    perception is approximately logarithmic.

    Parameters
    ----------
    centroids : list of float
        Spectral centroid values in Hz.
    low_hz : float
        Lower bound of the instrument's frequency range.
    high_hz : float
        Upper bound of the instrument's frequency range.

    Returns
    -------
    list of float
        Normalised pitch values in [0.0, 1.0].
    """
    if not centroids:
        return []

    # Use log scale for perceptual linearity
    log_low = np.log2(max(low_hz, 1.0))
    log_high = np.log2(max(high_hz, 2.0))
    log_range = log_high - log_low

    if log_range <= 0:
        return [0.5] * len(centroids)

    result = []
    for c in centroids:
        if c <= 0:
            result.append(0.0)
            continue
        log_c = np.log2(c)
        normalised = (log_c - log_low) / log_range
        result.append(float(np.clip(normalised, 0.0, 1.0)))

    return result


def pitch_to_lane(
    pitch: float,
    max_lane: int = 4,
) -> int:
    """
    Map a normalised pitch value (0.0–1.0) to a fretted guitar/bass lane.

    Low pitch → Green(0), high pitch → Orange(4).
    This is a **legacy / fallback** function used when pYIN data is not
    available.  When pYIN data IS available, prefer
    ``_semitone_contour_to_lanes`` which gives musically correct mapping.

    Does NOT handle open notes — those are flagged separately via
    ``is_open_note`` in StemAnalysis.

    Parameters
    ----------
    pitch : float
        Normalised pitch value in [0.0, 1.0].
    max_lane : int
        Maximum fretted lane index (0-based).  Usually 4 (Orange).

    Returns
    -------
    int
        Lane index in [0, max_lane].  Never returns 7 (open).
    """
    lane = int(pitch * (max_lane + 0.99))
    return max(0, min(lane, max_lane))


def pitch_contour_to_lanes(
    pitch_contour: list[float],
    max_lane: int = 4,
    smoothing: int = 3,
    is_open_note: list[bool] | None = None,
) -> list[int]:
    """
    Convert a pitch contour to lane assignments with optional smoothing.

    Applies a moving-average smoothing pass to avoid excessively jumpy
    lane assignments, then quantises to integer lane values.

    Open notes (flagged via ``is_open_note``) are assigned lane 7
    (Clone Hero purple bar) regardless of pitch contour value.

    Parameters
    ----------
    pitch_contour : list of float
        Normalised pitch values in [0.0, 1.0].
    max_lane : int
        Maximum fretted lane index (0–4).
    smoothing : int
        Window size for moving average smoothing.  Set to 1 for no smoothing.
    is_open_note : list of bool, optional
        Per-onset open-note flags.  When True, that onset gets lane 7.

    Returns
    -------
    list of int
        Lane assignments, one per onset.  Values are 0–4 (fretted) or
        7 (open / purple bar).
    """
    if not pitch_contour:
        return []

    contour = np.array(pitch_contour, dtype=float)

    # Apply moving average smoothing (only to non-open notes to avoid
    # smoothing pulling open notes toward fretted range)
    if smoothing > 1 and len(contour) >= smoothing:
        kernel = np.ones(smoothing) / smoothing
        # Pad to preserve length
        padded = np.pad(
            contour, (smoothing // 2, smoothing - 1 - smoothing // 2), mode="edge"
        )
        contour = np.convolve(padded, kernel, mode="valid")

    result: list[int] = []
    for i, p in enumerate(contour):
        # Check if this onset is an open note
        if is_open_note is not None and i < len(is_open_note) and is_open_note[i]:
            result.append(LANE_OPEN)
        else:
            result.append(pitch_to_lane(float(p), max_lane))

    return result


# ---------------------------------------------------------------------------
# Section name helpers
# ---------------------------------------------------------------------------


def get_section_name(instrument: str, difficulty: str) -> str:
    """
    Get the Clone Hero chart section name for an instrument + difficulty.

    Parameters
    ----------
    instrument : str
        One of ``guitar``, ``bass``, ``drums``, ``vocals``, ``full_mix``.
    difficulty : str
        One of ``easy``, ``medium``, ``hard``, ``expert``.

    Returns
    -------
    str
        Chart section name (e.g. ``ExpertSingle``, ``HardDrums``).
    """
    inst_map = INSTRUMENT_SECTION_NAMES.get(
        instrument.lower(), INSTRUMENT_SECTION_NAMES["guitar"]
    )
    return inst_map.get(difficulty.lower(), inst_map.get("expert", "ExpertSingle"))


def get_difficulty_profile_for_instrument(
    instrument: str,
    difficulty: str,
) -> dict[str, Any]:
    """
    Return a difficulty profile tuned for a specific instrument.

    Drums and bass need different density / gap settings compared to
    guitar.  This function returns a profile dict compatible with
    the existing ``DIFFICULTY_PROFILES`` structure in ``song_generator.py``
    but with instrument-specific overrides.

    Parameters
    ----------
    instrument : str
        Target instrument.
    difficulty : str
        Difficulty level.

    Returns
    -------
    dict
        Difficulty profile with section_name, note_skip, etc.
    """
    section_name = get_section_name(instrument, difficulty)

    # Base profiles per difficulty
    # allow_open: whether open notes (lane 7 / purple bar) are enabled.
    # Open notes are essential for metal charting (palm mutes, chugs,
    # drop-tuned open-string riffs).  Enabled on hard/expert only so
    # easier difficulties stay approachable.
    base_profiles = {
        "easy": {
            "note_skip": 4,
            "max_lane": 2,
            "chord_chance": 0.0,
            "hopo_energy_threshold": 0.0,  # no HOPOs on easy
            "sustain_chance": 0.05,
            "min_note_gap_ticks": 192,
            "allow_open": False,
            "tap_enabled": False,
        },
        "medium": {
            "note_skip": 3,
            "max_lane": 3,
            "chord_chance": 0.05,
            "hopo_energy_threshold": 0.0,  # no HOPOs on medium
            "sustain_chance": 0.10,
            "min_note_gap_ticks": 96,
            "allow_open": False,
            "tap_enabled": False,
        },
        "hard": {
            "note_skip": 2,
            "max_lane": 4,
            "chord_chance": 0.10,
            "hopo_energy_threshold": 0.45,  # moderate legato detection
            "sustain_chance": 0.15,
            "min_note_gap_ticks": 48,
            "allow_open": True,
            "tap_enabled": True,
            "tap_min_speed_ticks": 48,
            "tap_min_run": 5,
            "tap_high_lane_bias": True,
        },
        "expert": {
            "note_skip": 1,
            "max_lane": 4,
            "chord_chance": 0.15,
            "hopo_energy_threshold": 0.55,  # more permissive legato detection
            "sustain_chance": 0.20,
            "min_note_gap_ticks": 24,
            "allow_open": True,
            "tap_enabled": True,
            "tap_min_speed_ticks": 48,
            "tap_min_run": 4,
            "tap_high_lane_bias": True,
        },
    }

    profile: dict[str, Any] = dict(
        base_profiles.get(difficulty.lower(), base_profiles["expert"])
    )
    profile["section_name"] = section_name

    # Instrument-specific overrides
    inst = instrument.lower()

    if inst == "drums":
        # Drums: no HOPOs, no taps, no sustains, no open notes, tighter gaps
        profile["hopo_energy_threshold"] = 0.0
        profile["sustain_chance"] = 0.0
        profile["chord_chance"] = 0.0  # drum "chords" handled via simultaneous hits
        profile["allow_open"] = False  # drums don't use open notes
        profile["tap_enabled"] = False
        # Drums use lanes 0-4 even on easy (kick is always lane 0)
        if difficulty.lower() == "easy":
            profile["max_lane"] = 3  # kick + snare + hihat + tom
            profile["note_skip"] = 3
            profile["min_note_gap_ticks"] = 96
        elif difficulty.lower() == "medium":
            profile["max_lane"] = 3
            profile["note_skip"] = 2
            profile["min_note_gap_ticks"] = 48

    elif inst == "bass":
        # Bass: fewer notes, more sustain, less movement
        # Open notes are critical for bass in metal (drop-tuned root notes)
        # Bass HOPOs are less common; tighten the energy threshold
        if profile["hopo_energy_threshold"] > 0:
            profile["hopo_energy_threshold"] *= 0.7
        profile["sustain_chance"] *= 2.0
        profile["chord_chance"] *= 0.3  # bass rarely plays chords
        # Bass tapping is rare — disable or raise the bar
        profile["tap_enabled"] = False
        if difficulty.lower() in ("easy", "medium"):
            profile["max_lane"] = 2  # bass uses fewer frets on easy
            profile["allow_open"] = False  # no open notes on easy/medium bass
        profile["min_note_gap_ticks"] = max(
            profile["min_note_gap_ticks"],
            48,  # bass notes need more spacing
        )

    elif inst == "vocals":
        # Vocals: moderate density, no chords, more sustain, no open notes
        profile["chord_chance"] = 0.0
        profile["sustain_chance"] *= 1.5
        # Vocals don't use HOPOs or taps
        profile["hopo_energy_threshold"] = 0.0
        profile["allow_open"] = False  # vocals don't use open notes
        profile["tap_enabled"] = False

    return profile


# ---------------------------------------------------------------------------
# Demucs integration (optional, higher quality)
# ---------------------------------------------------------------------------


def demucs_available() -> bool:
    """Check if Demucs is installed and usable."""
    try:
        import demucs  # noqa: F401  # pyright: ignore[reportMissingImports]

        return True
    except ImportError:
        return False


def separate_with_demucs(
    file_path: str,
    model: str = "htdemucs",
) -> dict[str, Path] | None:
    """
    Separate audio into **all four stems** using Demucs (if installed).

    Demucs (Hybrid Transformer Demucs) provides state-of-the-art source
    separation, producing four high-quality stems: ``vocals``, ``drums``,
    ``bass``, and ``other`` (guitar/keys/synths).

    This function performs **full 4-stem separation** (not ``--two-stems``)
    so that every instrument can be analysed independently.  The ``other``
    stem is mapped to ``guitar`` by the caller.

    Parameters
    ----------
    file_path : str
        Path to the audio file.
    model : str
        Demucs model name.  ``htdemucs`` (default) is the best general-
        purpose model; ``htdemucs_ft`` is fine-tuned for higher quality
        but slower.

    Returns
    -------
    dict or None
        Mapping of stem name → Path to separated WAV file.
        Keys: ``vocals``, ``drums``, ``bass``, ``other`` (guitar).
        Returns None if Demucs is not available or separation fails.
    """
    if not demucs_available():
        logger.info("ℹ️ Demucs not installed — using librosa HPSS instead")
        return None

    import shutil
    import subprocess

    try:
        output_dir = Path(tempfile.mkdtemp(prefix="demucs_"))
        # Full 4-stem separation (no --two-stems flag)
        cmd = [
            "python",
            "-m",
            "demucs",
            "-n",
            model,
            "-o",
            str(output_dir),
            file_path,
        ]

        logger.info("🎛️ Running Demucs 4-stem separation (model={})...", model)
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=600,  # 10 minute timeout
        )

        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace")[-500:]
            logger.error("❌ Demucs failed (rc={}): {}", result.returncode, stderr)
            shutil.rmtree(output_dir, ignore_errors=True)
            return None

        # Find the output stems
        stem_dir = output_dir / model / Path(file_path).stem
        stems: dict[str, Path] = {}
        for stem_name in ("vocals", "drums", "bass", "other"):
            stem_file = stem_dir / f"{stem_name}.wav"
            if stem_file.exists():
                stems[stem_name] = stem_file

        if stems:
            logger.info(
                "✅ Demucs 4-stem separation complete: {} stems ({})",
                len(stems),
                ", ".join(stems.keys()),
            )
        else:
            logger.warning("⚠️ Demucs produced no output stems")
            shutil.rmtree(output_dir, ignore_errors=True)
            return None

        return stems

    except subprocess.TimeoutExpired:
        logger.error("❌ Demucs timed out after 10 minutes")
        return None
    except Exception as e:
        logger.error("❌ Demucs error: {}", e)
        return None


# Demucs stem name → our instrument name
DEMUCS_STEM_MAP: dict[str, str] = {
    "other": "guitar",  # Demucs "other" contains guitar, keys, synths
    "bass": "bass",
    "drums": "drums",
    "vocals": "vocals",
}

# Reverse: our instrument name → which Demucs stem to load
INSTRUMENT_TO_DEMUCS_STEM: dict[str, str] = {
    "guitar": "other",
    "bass": "bass",
    "drums": "drums",
    "vocals": "vocals",
}


def _load_demucs_stem_as_separation(
    stem_path: Path,
    sr: int | None = None,
) -> SeparationResult:
    """
    Load a Demucs-separated stem WAV and wrap it in a SeparationResult.

    Since Demucs already provides a clean instrument isolation, we run
    a light HPSS on the isolated stem to split it into harmonic and
    percussive components (useful for guitar: palm mutes are percussive
    transients even within the isolated guitar stem).

    Parameters
    ----------
    stem_path : Path
        Path to the Demucs-output WAV file.
    sr : int, optional
        Target sample rate (None = preserve original).

    Returns
    -------
    SeparationResult
        Contains harmonic/percussive split of the isolated stem.
    """
    y, sample_rate = librosa.load(str(stem_path), sr=sr, mono=True)
    sample_rate = int(sample_rate)
    duration = float(librosa.get_duration(y=y, sr=sample_rate))

    # Light HPSS on the already-isolated stem (margin=2 for gentler split)
    harmonic, percussive = librosa.effects.hpss(y, margin=2.0)

    logger.info(
        "✅ Loaded Demucs stem '{}': sr={}, duration={:.1f}s",
        stem_path.stem,
        sample_rate,
        duration,
    )

    return SeparationResult(
        harmonic=harmonic,
        percussive=percussive,
        full_signal=y,
        sample_rate=sample_rate,
        duration=duration,
    )
