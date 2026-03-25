"""
Spectral texture feature extraction (18 features).

Features not present in pyAudioAnalysis:
    Spectral contrast 7 sub-bands: mean + std   (14)
    Spectral flatness: mean + std               (2)
    Spectral bandwidth: mean + std              (2)
"""

import librosa
import numpy as np


def extract_spectral_texture_features(S, sr):
    """
    Extract spectral texture features.

    Parameters
    ----------
    S : np.ndarray
        Pre-computed magnitude STFT.
    sr : int
        Sample rate.

    Returns
    -------
    features : dict
        18 spectral texture features.
    """
    features = {}

    # Spectral contrast: 7 sub-bands, mean + std = 14
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr, n_bands=6, fmin=50)
    for i in range(contrast.shape[0]):
        features[f"spectral_contrast_band{i+1}_mean"] = np.mean(contrast[i])
        features[f"spectral_contrast_band{i+1}_std"] = np.std(contrast[i])

    # Spectral flatness: mean + std = 2
    flatness = librosa.feature.spectral_flatness(S=S)
    features["spectral_flatness_mean"] = np.mean(flatness)
    features["spectral_flatness_std"] = np.std(flatness)

    # Spectral bandwidth: mean + std = 2
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)
    features["spectral_bandwidth_mean"] = np.mean(bandwidth)
    features["spectral_bandwidth_std"] = np.std(bandwidth)

    return features
