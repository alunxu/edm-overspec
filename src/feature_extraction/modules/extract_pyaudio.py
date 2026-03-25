"""
extract_pyaudio.py — Librosa reimplementation of the 92 pyAudioAnalysis + Essentia features.
============================================================================================
Ports all 92 features originally extracted with pyAudioAnalysis and Essentia
(replicating the feature extraction pipeline methodology of Caparrini et al., 2020)
into pure librosa/scipy so the entire pipeline uses a single library stack.

Features (92 total):
  pyAudioAnalysis block (70):
    Short-term features computed on 50ms frames, summarized as mean + std:
      1-8   (×2):  ZCR, Energy, EnergyEntropy, SpectralCentroid,
                    SpectralSpread, SpectralEntropy, SpectralFlux, SpectralRolloff
      9-21  (×2):  MFCCs 1-13
      22-33 (×2):  ChromaVector 1-12
      34    (×2):  ChromaDeviation
    Total: 34 features × 2 stats = 68
    Plus: BPM (69) + BPMconf (70)

  Essentia block (22):
      71: bpm (librosa tempo)
      72: bpm_histogram_first_peak_bpm
      73: bpm_histogram_first_peak_weight
      74: bpm_histogram_second_peak_bpm
      75: bpm_histogram_second_peak_spread
      76: bpm_histogram_second_peak_weight
      77: danceability
      78-79: beats_loudness mean/stdev
      80: onset_rate
      81-86: beats_loudness_band_ratio mean (6 bands)
      87-92: beats_loudness_band_ratio stdev (6 bands)

Column names match the original CSV exactly for drop-in compatibility.
"""

import librosa
import numpy as np
import scipy.stats
import scipy.signal


def _energy_entropy(frame, n_sub=10):
    """
    Compute energy entropy of a frame (pyAudioAnalysis definition).
    Splits the frame into n_sub sub-frames and computes the entropy
    of the normalised energy distribution.
    """
    frame_energy = np.sum(frame ** 2)
    sub_len = len(frame) // n_sub
    if sub_len == 0:
        return 0.0
    sub_energies = np.array([
        np.sum(frame[i * sub_len:(i + 1) * sub_len] ** 2)
        for i in range(n_sub)
    ])
    sub_energies = sub_energies / (frame_energy + 1e-10)
    sub_energies = sub_energies[sub_energies > 0]
    return -np.sum(sub_energies * np.log2(sub_energies + 1e-10))


def _spectral_entropy(magnitude_spectrum, n_sub=10):
    """
    Compute spectral entropy (pyAudioAnalysis definition).
    Normalises the magnitude spectrum as a probability distribution
    and computes the entropy of n_sub spectral sub-bands.
    """
    total = np.sum(magnitude_spectrum)
    if total == 0:
        return 0.0
    prob = magnitude_spectrum / total
    sub_len = len(prob) // n_sub
    if sub_len == 0:
        return 0.0
    sub_energies = np.array([
        np.sum(prob[i * sub_len:(i + 1) * sub_len])
        for i in range(n_sub)
    ])
    sub_energies = sub_energies[sub_energies > 0]
    return -np.sum(sub_energies * np.log2(sub_energies + 1e-10))


def _spectral_flux(current_fft, previous_fft):
    """Spectral flux: L2 norm of the difference between successive normalised spectra."""
    current_norm = current_fft / (np.sum(current_fft) + 1e-10)
    previous_norm = previous_fft / (np.sum(previous_fft) + 1e-10)
    return np.sum((current_norm - previous_norm) ** 2)


def _compute_short_term_features(y, sr, st_win, st_step):
    """
    Compute 34 short-term features per frame, matching pyAudioAnalysis output.

    Parameters
    ----------
    y : np.ndarray
        Audio time series.
    sr : int
        Sample rate.
    st_win : float
        Short-term window size in seconds.
    st_step : float
        Short-term step size in seconds.

    Returns
    -------
    features : np.ndarray
        Array of shape (34, n_frames).
    """
    win_samples = int(round(st_win * sr))
    step_samples = int(round(st_step * sr))
    n_frames = max(1, (len(y) - win_samples) // step_samples + 1)

    # Pre-allocate
    feat_matrix = np.zeros((34, n_frames))
    prev_fft = np.zeros(win_samples // 2 + 1)

    # Hamming window
    hamming = np.hamming(win_samples)

    for i in range(n_frames):
        start = i * step_samples
        end = start + win_samples
        if end > len(y):
            break
        frame = y[start:end]
        frame_windowed = frame * hamming

        # FFT
        fft_magnitude = np.abs(np.fft.rfft(frame_windowed))
        fft_magnitude = fft_magnitude / (len(fft_magnitude) + 1e-10)

        # Frequency axis
        freqs = np.fft.rfftfreq(win_samples, d=1.0 / sr)

        # === 1. Zero Crossing Rate ===
        zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))
        feat_matrix[0, i] = zcr

        # === 2. Energy ===
        energy = np.sum(frame ** 2) / len(frame)
        feat_matrix[1, i] = energy

        # === 3. Energy Entropy ===
        feat_matrix[2, i] = _energy_entropy(frame)

        # === 4. Spectral Centroid ===
        total_fft = np.sum(fft_magnitude)
        if total_fft > 0:
            centroid = np.sum(freqs * fft_magnitude) / total_fft
        else:
            centroid = 0.0
        # Normalize to [0, 1] by dividing by Nyquist
        feat_matrix[3, i] = centroid / (sr / 2.0)

        # === 5. Spectral Spread ===
        if total_fft > 0:
            spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * fft_magnitude) / total_fft)
        else:
            spread = 0.0
        feat_matrix[4, i] = spread / (sr / 2.0)

        # === 6. Spectral Entropy ===
        feat_matrix[5, i] = _spectral_entropy(fft_magnitude)

        # === 7. Spectral Flux ===
        feat_matrix[6, i] = _spectral_flux(fft_magnitude, prev_fft)
        prev_fft = fft_magnitude.copy()

        # === 8. Spectral Rolloff ===
        cumsum = np.cumsum(fft_magnitude)
        rolloff_threshold = 0.90 * cumsum[-1] if len(cumsum) > 0 else 0
        rolloff_idx = np.where(cumsum >= rolloff_threshold)[0]
        if len(rolloff_idx) > 0:
            feat_matrix[7, i] = float(rolloff_idx[0]) / len(fft_magnitude)
        else:
            feat_matrix[7, i] = 0.0

        # === 9-21. MFCCs 1-13 ===
        # Use librosa for MFCCs on this frame
        S = librosa.feature.melspectrogram(
            y=frame, sr=sr, n_fft=win_samples, hop_length=win_samples + 1,
            n_mels=26, fmin=0, fmax=sr / 2
        )
        log_S = librosa.power_to_db(S, ref=np.max)
        mfccs = librosa.feature.mfcc(S=log_S, n_mfcc=13)
        feat_matrix[8:21, i] = mfccs[:, 0]

        # === 22-33. Chroma Vector (12 bins) ===
        chroma = librosa.feature.chroma_stft(
            y=frame, sr=sr, n_fft=win_samples, hop_length=win_samples + 1
        )
        feat_matrix[21:33, i] = chroma[:, 0]

        # === 34. Chroma Deviation ===
        feat_matrix[33, i] = np.std(chroma[:, 0])

    return feat_matrix[:, :n_frames]


def _compute_bpm_histogram(onset_env, sr, hop_length):
    """
    Compute BPM histogram and extract first/second peaks.
    Mimics Essentia's bpm_histogram features.

    Returns
    -------
    dict with keys: first_peak_bpm, first_peak_weight,
                    second_peak_bpm, second_peak_spread, second_peak_weight
    """
    # Compute autocorrelation of onset envelope
    ac = librosa.autocorrelate(onset_env, max_size=len(onset_env))
    ac = ac / (ac[0] + 1e-10)

    # Convert lag to BPM
    # lag in frames -> lag in seconds = lag * hop_length / sr
    # BPM = 60 / seconds
    min_bpm, max_bpm = 40, 250
    min_lag = int(round(60.0 / max_bpm * sr / hop_length))
    max_lag = int(round(60.0 / min_bpm * sr / hop_length))
    max_lag = min(max_lag, len(ac) - 1)
    min_lag = max(1, min_lag)

    if max_lag <= min_lag:
        return {
            'first_peak_bpm': 0.0, 'first_peak_weight': 0.0,
            'second_peak_bpm': 0.0, 'second_peak_spread': 0.0,
            'second_peak_weight': 0.0
        }

    ac_segment = ac[min_lag:max_lag + 1]
    lags = np.arange(min_lag, max_lag + 1)
    bpms = 60.0 / (lags * hop_length / sr)

    # Find peaks
    peaks, properties = scipy.signal.find_peaks(ac_segment, height=0)
    if len(peaks) == 0:
        best_idx = np.argmax(ac_segment)
        return {
            'first_peak_bpm': bpms[best_idx],
            'first_peak_weight': float(ac_segment[best_idx]),
            'second_peak_bpm': 0.0, 'second_peak_spread': 0.0,
            'second_peak_weight': 0.0
        }

    peak_heights = properties['peak_heights']
    sorted_peaks = np.argsort(peak_heights)[::-1]

    first_idx = peaks[sorted_peaks[0]]
    result = {
        'first_peak_bpm': bpms[first_idx],
        'first_peak_weight': float(ac_segment[first_idx]),
    }

    if len(sorted_peaks) > 1:
        second_idx = peaks[sorted_peaks[1]]
        result['second_peak_bpm'] = bpms[second_idx]
        result['second_peak_weight'] = float(ac_segment[second_idx])
        # Spread: width of the peak at half-height
        half_height = ac_segment[second_idx] / 2
        left = second_idx
        while left > 0 and ac_segment[left] > half_height:
            left -= 1
        right = second_idx
        while right < len(ac_segment) - 1 and ac_segment[right] > half_height:
            right += 1
        result['second_peak_spread'] = float(bpms[left] - bpms[right]) if right > left else 0.0
    else:
        result['second_peak_bpm'] = 0.0
        result['second_peak_spread'] = 0.0
        result['second_peak_weight'] = 0.0

    return result


def _compute_danceability(onset_env, sr, hop_length):
    """
    Compute danceability score.
    Based on regularity of onset pattern (autocorrelation peak strength).
    """
    ac = librosa.autocorrelate(onset_env, max_size=len(onset_env) // 2)
    ac = ac / (ac[0] + 1e-10)
    # Danceability: mean of top autocorrelation peaks in dance tempo range
    min_lag = int(round(60.0 / 200 * sr / hop_length))  # 200 BPM
    max_lag = int(round(60.0 / 60 * sr / hop_length))    # 60 BPM
    max_lag = min(max_lag, len(ac) - 1)
    min_lag = max(1, min_lag)
    if max_lag <= min_lag:
        return 0.0
    ac_dance = ac[min_lag:max_lag + 1]
    peaks, _ = scipy.signal.find_peaks(ac_dance, height=0)
    if len(peaks) > 0:
        return float(np.mean(ac_dance[peaks]))
    return float(np.max(ac_dance)) if len(ac_dance) > 0 else 0.0


def _compute_beats_loudness_band_ratios(y, sr, beat_frames, hop_length):
    """
    Compute beats loudness and band ratios (Essentia-style).

    6 bands: [20-150], [150-400], [400-1k], [1k-2.5k], [2.5k-6.3k], [6.3k-22k] Hz
    """
    # Band edges (Essentia defaults)
    band_edges = [20, 150, 400, 1000, 2500, 6300, min(22050, sr // 2)]
    n_bands = len(band_edges) - 1

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    # Compute total loudness and band ratios at each beat frame
    beat_loudness = []
    band_ratios = [[] for _ in range(n_bands)]

    for bf in beat_frames:
        if bf < S.shape[1]:
            frame_power = S[:, bf]
            total = np.sum(frame_power) + 1e-10
            beat_loudness.append(np.sqrt(total))
            for b in range(n_bands):
                mask = (freqs >= band_edges[b]) & (freqs < band_edges[b + 1])
                band_ratios[b].append(np.sum(frame_power[mask]) / total)

    if len(beat_loudness) == 0:
        return {
            'beats_loudness_mean': 0.0,
            'beats_loudness_stdev': 0.0,
            **{f'band_ratio_mean{i+1}': 0.0 for i in range(n_bands)},
            **{f'band_ratio_stdev{i+1}': 0.0 for i in range(n_bands)},
        }

    result = {
        'beats_loudness_mean': float(np.mean(beat_loudness)),
        'beats_loudness_stdev': float(np.std(beat_loudness)),
    }
    for b in range(n_bands):
        result[f'band_ratio_mean{b+1}'] = float(np.mean(band_ratios[b]))
        result[f'band_ratio_stdev{b+1}'] = float(np.std(band_ratios[b]))

    return result


def extract_pyaudio_features(y, sr, onset_env, hop_length):
    """
    Extract 92 features matching the original pyAudioAnalysis + Essentia pipeline.

    Parameters
    ----------
    y : np.ndarray
        Audio time series (mono, resampled to target sr).
    sr : int
        Sample rate.
    onset_env : np.ndarray
        Pre-computed onset strength envelope.
    hop_length : int
        Hop length in samples.

    Returns
    -------
    features : dict
        92 features with column names matching original CSV.
    """
    features = {}

    # ===================================================================
    # PART 1: pyAudioAnalysis features (70)
    # Short-term: 50ms window, 50ms step (non-overlapping)
    # Mid-term: averaged over entire track (1s window, 1s step)
    # ===================================================================
    st_win = 0.05
    st_step = 0.05

    st_features = _compute_short_term_features(y, sr, st_win, st_step)
    # st_features: (34, n_frames)

    # Feature names (matching original CSV column format)
    base_names = [
        "ZCR", "Energy", "EnergyEntropy", "SpectralCentroid",
        "SpectralSpread", "SpectralEntropy", "SpectralFlux", "SpectralRolloff",
    ]
    # Add MFCC names
    for i in range(1, 14):
        base_names.append(f"MFCCs{i}")
    # Add Chroma names
    for i in range(1, 13):
        base_names.append(f"ChromaVector{i}")
    base_names.append("ChromaDeviation")
    # Total: 8 + 13 + 12 + 1 = 34

    # Mean features (1-34)
    for i, name in enumerate(base_names):
        features[f"{i+1}-{name}m"] = float(np.mean(st_features[i]))

    # Std features (35-68)
    for i, name in enumerate(base_names):
        features[f"{i+35}-{name}std"] = float(np.std(st_features[i]))

    # BPM (69) and BPMconf (70)
    tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    features["69-BPM"] = float(tempo[0]) if len(tempo) > 0 else 0.0

    # BPM confidence: use the autocorrelation peak strength at the beat period
    ac_onset = librosa.autocorrelate(onset_env, max_size=len(onset_env))
    ac_onset = ac_onset / (ac_onset[0] + 1e-10)
    if features["69-BPM"] > 0:
        beat_lag = int(round(60.0 / features["69-BPM"] * sr / hop_length))
        if 0 < beat_lag < len(ac_onset):
            features["70-BPMconf"] = float(ac_onset[beat_lag])
        else:
            features["70-BPMconf"] = 0.0
    else:
        features["70-BPMconf"] = 0.0

    # ===================================================================
    # PART 2: Essentia rhythm features (22)
    # ===================================================================

    # 71: bpm (librosa tempo — may differ slightly from pyAudioAnalysis BPM)
    features["71-bpm"] = features["69-BPM"]  # Same estimator

    # 72-76: BPM histogram peaks
    bpm_hist = _compute_bpm_histogram(onset_env, sr, hop_length)
    features["72-bpm_histogram_first_peak_bpm"] = bpm_hist['first_peak_bpm']
    features["73-bpm_histogram_first_peak_weight"] = bpm_hist['first_peak_weight']
    features["74-bpm_histogram_second_peak_bpm"] = bpm_hist['second_peak_bpm']
    features["75-bpm_histogram_second_peak_spread"] = bpm_hist['second_peak_spread']
    features["76-bpm_histogram_second_peak_weight"] = bpm_hist['second_peak_weight']

    # 77: danceability
    features["77-danceability"] = _compute_danceability(onset_env, sr, hop_length)

    # 78-92: beats_loudness and band ratios
    _, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    bl = _compute_beats_loudness_band_ratios(y, sr, beat_frames, hop_length)
    features["78-beats_loudness.mean"] = bl['beats_loudness_mean']
    features["79-beats_loudness.stdev"] = bl['beats_loudness_stdev']

    # 80: onset_rate
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    duration = len(y) / sr
    features["80-onset_rate"] = len(onsets) / (duration + 1e-10)

    # 81-86: band ratio means
    for i in range(1, 7):
        features[f"{80+i}-beats_loudness_band_ratio.mean{i}"] = bl[f'band_ratio_mean{i}']

    # 87-92: band ratio stdevs
    for i in range(1, 7):
        features[f"{86+i}-beats_loudness_band_ratio.stdev{i}"] = bl[f'band_ratio_stdev{i}']

    return features
