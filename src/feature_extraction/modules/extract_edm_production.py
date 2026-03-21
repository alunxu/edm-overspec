"""
EDM-specific production feature extraction (15 features).

Features:
    Sub-bass energy ratio (20-60 Hz)                         (1)
    Beat regularity (autocorrelation at beat period)         (1)
    Energy variance over 8-bar windows                       (1)
    Spectral flux modulation: dominant freq + strength       (2)
    Low/mid/high energy ratios                               (3)
    Percussive onset density (onsets/sec on perc component)  (1)
    Kick prominence (perc energy 40-100Hz / total perc)      (1)
    Sidechain pumping proxy: mod depth + mod rate            (2)
    RMS contour stats: mean, std, skew                       (3)
"""

import librosa
import numpy as np
import scipy.stats


def extract_edm_features(y, sr, hop_length, onset_env, S_power, freqs, y_percussive, rms_full):
    """
    Extract EDM-specific production features.

    Parameters
    ----------
    y : np.ndarray
        Audio time series.
    sr : int
        Sample rate.
    hop_length : int
        Hop length in samples.
    onset_env : np.ndarray
        Pre-computed onset strength envelope.
    S_power : np.ndarray
        Pre-computed power spectrogram (|STFT|²).
    freqs : np.ndarray
        Frequency bin centers from librosa.fft_frequencies.
    y_percussive : np.ndarray
        Pre-computed percussive component from HPSS.
    rms_full : np.ndarray
        Pre-computed RMS energy.

    Returns
    -------
    features : dict
        15 EDM-specific production features.
    """
    features = {}
    total_energy = np.sum(S_power) + 1e-10

    # --- 1. Sub-bass energy ratio (20-60 Hz) ---
    # Captures "drop" intensity; high for Dubstep, Bass House
    sub_bass_mask = (freqs >= 20) & (freqs <= 60)
    features["sub_bass_ratio"] = np.sum(S_power[sub_bass_mask]) / total_energy

    # --- 2. Beat regularity ---
    # High = four-on-the-floor (House/Techno), Low = syncopated (D&B/Breaks)
    tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    beat_tempo = tempo[0] if len(tempo) > 0 else 120.0
    if beat_tempo > 0:
        beat_period_frames = int(round(60.0 / beat_tempo * sr / hop_length))
        if 0 < beat_period_frames < len(onset_env):
            ac = np.correlate(onset_env, onset_env, mode='full')
            ac = ac[len(ac)//2:]
            ac = ac / (ac[0] + 1e-10)
            features["beat_regularity"] = float(ac[beat_period_frames])
        else:
            features["beat_regularity"] = 0.0
    else:
        features["beat_regularity"] = 0.0

    # --- 3. Energy variance over 8-bar windows ---
    # Captures macro-structural dynamics: build-ups, drops, breakdowns
    if beat_tempo > 0:
        window_frames = int(round(60.0 / beat_tempo * 4 * 8 * sr / hop_length))
        window_frames = max(1, window_frames)
        n_windows = max(1, len(rms_full) // window_frames)
        window_energies = []
        for w in range(n_windows):
            start = w * window_frames
            end = min(start + window_frames, len(rms_full))
            window_energies.append(np.mean(rms_full[start:end]))
        features["energy_variance_8bar"] = float(np.var(window_energies)) if len(window_energies) > 1 else 0.0
    else:
        features["energy_variance_8bar"] = 0.0

    # --- 4. Spectral flux modulation rate (wobble bass detection) ---
    # FFT of spectral centroid time series → dominant LFO modulation frequency
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    centroid_detrended = centroid - np.mean(centroid)
    if len(centroid_detrended) > 1:
        mod_fft = np.abs(np.fft.rfft(centroid_detrended))
        mod_freqs = np.fft.rfftfreq(len(centroid_detrended), d=hop_length / sr)
        lfo_mask = (mod_freqs >= 0.5) & (mod_freqs <= 20.0)
        if np.any(lfo_mask):
            mod_fft_lfo = mod_fft[lfo_mask]
            mod_freqs_lfo = mod_freqs[lfo_mask]
            peak_idx = np.argmax(mod_fft_lfo)
            features["spectral_mod_rate"] = float(mod_freqs_lfo[peak_idx])
            features["spectral_mod_strength"] = float(mod_fft_lfo[peak_idx] / (np.sum(mod_fft) + 1e-10))
        else:
            features["spectral_mod_rate"] = 0.0
            features["spectral_mod_strength"] = 0.0
    else:
        features["spectral_mod_rate"] = 0.0
        features["spectral_mod_strength"] = 0.0

    # --- 5. Low/mid/high energy ratios ---
    low_mask = (freqs >= 20) & (freqs < 250)
    mid_mask = (freqs >= 250) & (freqs < 4000)
    high_mask = (freqs >= 4000) & (freqs <= 20000)
    features["energy_ratio_low"] = np.sum(S_power[low_mask]) / total_energy
    features["energy_ratio_mid"] = np.sum(S_power[mid_mask]) / total_energy
    features["energy_ratio_high"] = np.sum(S_power[high_mask]) / total_energy

    # --- 6. Percussive onset density ---
    # D&B/Breaks = high density, Downtempo/Ambient = low
    onset_env_perc = librosa.onset.onset_strength(y=y_percussive, sr=sr, hop_length=hop_length)
    onsets_perc = librosa.onset.onset_detect(onset_envelope=onset_env_perc, sr=sr,
                                              hop_length=hop_length, units='time')
    duration = len(y) / sr
    features["percussive_onset_density"] = len(onsets_perc) / (duration + 1e-10)

    # --- 7. Kick prominence ---
    # Ratio of percussive energy in kick range (40-100Hz) to total percussive
    S_perc = np.abs(librosa.stft(y_percussive, n_fft=2048, hop_length=hop_length)) ** 2
    kick_mask = (freqs >= 40) & (freqs <= 100)
    total_perc_energy = np.sum(S_perc) + 1e-10
    features["kick_prominence"] = np.sum(S_perc[kick_mask]) / total_perc_energy

    # --- 8. Sidechain pumping proxy ---
    # Measures rhythmic ducking effect at beat frequency
    rms_detrended = rms_full - np.mean(rms_full)
    if len(rms_detrended) > 1 and beat_tempo > 0:
        rms_ac = np.correlate(rms_detrended, rms_detrended, mode='full')
        rms_ac = rms_ac[len(rms_ac)//2:]
        rms_ac = rms_ac / (rms_ac[0] + 1e-10)
        beat_period = int(round(60.0 / beat_tempo * sr / hop_length))
        if 0 < beat_period < len(rms_ac):
            features["sidechain_mod_depth"] = float(rms_ac[beat_period])
        else:
            features["sidechain_mod_depth"] = 0.0
        features["sidechain_mod_rate"] = float(np.std(rms_full) / (np.mean(rms_full) + 1e-10))
    else:
        features["sidechain_mod_depth"] = 0.0
        features["sidechain_mod_rate"] = 0.0

    # --- 9. RMS contour stats ---
    features["rms_mean"] = float(np.mean(rms_full))
    features["rms_std"] = float(np.std(rms_full))
    features["rms_skew"] = float(scipy.stats.skew(rms_full))

    return features
