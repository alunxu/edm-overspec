"""
Harmonic-percussive and dynamic feature extraction (5 features).

Features:
    Harmonic-to-percussive energy ratio: mean + std   (2)
    Dynamic range (peak-to-RMS in dB)                 (1)
    RMS kurtosis                                      (1)
    Crest factor                                      (1)
"""

import librosa
import numpy as np
import scipy.stats


def extract_hp_dynamic_features(y, sr, hop_length, y_harmonic, y_percussive, rms_full):
    """
    Extract harmonic-percussive ratio and dynamic range features.

    Parameters
    ----------
    y : np.ndarray
        Audio time series.
    sr : int
        Sample rate.
    hop_length : int
        Hop length in samples.
    y_harmonic : np.ndarray
        Pre-computed harmonic component from HPSS.
    y_percussive : np.ndarray
        Pre-computed percussive component from HPSS.
    rms_full : np.ndarray
        Pre-computed RMS energy.

    Returns
    -------
    features : dict
        5 harmonic-percussive and dynamic features.
    """
    features = {}

    # Harmonic-percussive ratio (frame-wise)
    rms_h = librosa.feature.rms(y=y_harmonic, hop_length=hop_length)[0]
    rms_p = librosa.feature.rms(y=y_percussive, hop_length=hop_length)[0]
    frame_ratio = rms_h / (rms_p + 1e-10)
    features["hp_ratio_mean"] = float(np.mean(frame_ratio))
    features["hp_ratio_std"] = float(np.std(frame_ratio))

    # Dynamic range: peak-to-RMS in dB
    rms_val = np.sqrt(np.mean(y ** 2))
    peak_val = np.max(np.abs(y))
    if rms_val > 0:
        features["dynamic_range_db"] = float(20 * np.log10(peak_val / rms_val))
    else:
        features["dynamic_range_db"] = 0.0

    # RMS kurtosis
    features["rms_kurtosis"] = float(scipy.stats.kurtosis(rms_full))

    # Crest factor (peak / RMS)
    features["crest_factor"] = float(peak_val / (rms_val + 1e-10))

    return features
