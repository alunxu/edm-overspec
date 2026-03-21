"""
Tempogram feature extraction (64 features).

Features:
    Fourier tempogram top-5 peaks: BPM, mean, std           (15)
    Autocorrelation tempogram top-5 peaks: BPM, mean, std   (15)
    Cyclic Fourier top-5 peaks: s, mean, std + entropy + ratio  (17)
    Cyclic Auto top-5 peaks: s, mean, std + entropy + ratio     (17)
"""

import librosa
import numpy as np
import scipy.stats

from .utils import compute_cyclic_tempogram


def extract_tempogram_features(onset_env, y, sr, hop_length):
    """
    Extract 64 tempogram features.

    Parameters
    ----------
    onset_env : np.ndarray
        Pre-computed onset strength envelope.
    y : np.ndarray
        Audio time series.
    sr : int
        Sample rate.
    hop_length : int
        Hop length in samples.

    Returns
    -------
    features : dict
        64 tempogram features.
    """
    features = {}

    tempogram_fourier = librosa.feature.fourier_tempogram(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length
    )
    tempogram_auto = librosa.feature.tempogram(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length
    )
    ac_bpm = librosa.tempo_frequencies(tempogram_auto.shape[0], sr=sr, hop_length=hop_length)
    ft_bpm = librosa.tempo_frequencies(tempogram_fourier.shape[0], sr=sr, hop_length=hop_length)
    cyclic_ft = compute_cyclic_tempogram(np.abs(tempogram_fourier), ft_bpm)
    cyclic_ac = compute_cyclic_tempogram(tempogram_auto, ac_bpm)

    # Fourier Tempogram: top-5 peaks
    fourier_mag = np.abs(tempogram_fourier)
    fourier_mean_profile = np.mean(fourier_mag, axis=1)
    valid_idx = np.isfinite(ft_bpm)
    sorted_valid_idx = np.argsort(fourier_mean_profile[valid_idx])[-5:][::-1]
    top_idx_ft = np.where(valid_idx)[0][sorted_valid_idx]
    for i in range(5):
        features[f"fourier_peak{i+1}_bpm"] = ft_bpm[top_idx_ft[i]]
        features[f"fourier_peak{i+1}_mean"] = fourier_mean_profile[top_idx_ft[i]]
        features[f"fourier_peak{i+1}_std"] = np.std(fourier_mag[top_idx_ft[i]])

    # Autocorrelation Tempogram: top-5 peaks
    auto_mean_profile = np.mean(tempogram_auto, axis=1)
    valid_idx_ac = np.isfinite(ac_bpm)
    sorted_valid_idx_ac = np.argsort(auto_mean_profile[valid_idx_ac])[-5:][::-1]
    top_idx_ac = np.where(valid_idx_ac)[0][sorted_valid_idx_ac]
    for i in range(5):
        features[f"auto_peak{i+1}_bpm"] = ac_bpm[top_idx_ac[i]]
        features[f"auto_peak{i+1}_mean"] = auto_mean_profile[top_idx_ac[i]]
        features[f"auto_peak{i+1}_std"] = np.std(tempogram_auto[top_idx_ac[i]])

    # Cyclic tempograms
    def _summarize_cyclic(cyclic, prefix):
        cyclic_profile = np.mean(cyclic, axis=1)
        norm_profile = cyclic_profile / np.sum(cyclic_profile + 1e-10)
        top5_idx = np.argsort(norm_profile)[-5:][::-1]
        top5_vals = norm_profile[top5_idx]
        entropy = scipy.stats.entropy(norm_profile)
        peak_ratio = np.max(norm_profile) / np.sum(norm_profile)
        for i in range(5):
            s = 1 + top5_idx[i] / 120.0
            features[f"{prefix}_peak{i+1}_s"] = s
            features[f"{prefix}_peak{i+1}_mean"] = top5_vals[i]
            features[f"{prefix}_peak{i+1}_std"] = np.std(cyclic[top5_idx[i]])
        features[f"{prefix}_entropy"] = entropy
        features[f"{prefix}_peak_ratio"] = peak_ratio

    _summarize_cyclic(cyclic_ft, "cyclic_fourier")
    _summarize_cyclic(cyclic_ac, "cyclic_auto")
    return features
