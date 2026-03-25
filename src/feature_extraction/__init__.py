"""
Feature extraction package for EDM Phantom Genre project.

Extracts 102 audio features per track across 4 groups:
    extract_tempogram_features     - Fourier/autocorrelation/cyclic tempogram (64)
    extract_spectral_texture       - Spectral contrast, flatness, bandwidth (18)
    extract_edm_production         - EDM-specific production features (15)
    extract_harmonic_percussive    - H/P ratio + dynamic range features (5)
    utils                          - Shared utilities (I/O, cyclic tempogram)
"""

from .extract_pyaudio import extract_pyaudio_features
from .extract_tempogram_features import extract_tempogram_features
from .extract_spectral_texture import extract_spectral_texture_features
from .extract_edm_production import extract_edm_features
from .extract_harmonic_percussive import extract_hp_dynamic_features
from .utils import walk_audio_directory, load_metadata

__all__ = [
    'extract_pyaudio_features',
    'extract_tempogram_features',
    'extract_spectral_texture_features',
    'extract_edm_features',
    'extract_hp_dynamic_features',
    'walk_audio_directory',
    'load_metadata',
]
