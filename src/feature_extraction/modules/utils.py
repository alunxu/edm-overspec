"""
Shared utilities for feature extraction.

Functions:
    walk_audio_directory  - Discover all audio tracks with robust path handling
    load_metadata         - Load BPM/Key/Length from metadata.json files
    compute_cyclic_tempogram - Cyclic tempogram computation (used by tempogram module)
"""

import os
import json
import numpy as np


def compute_cyclic_tempogram(tempogram, bpm_axis, base_bpm=60, M=120):
    """
    Compute cyclic tempogram by pooling tempi related by powers of two.

    C_rho(t, s) = sum_{lambda in [s*rho]} T(t, lambda), s in [1, 2)

    Parameters
    ----------
    tempogram : np.ndarray
        Input tempogram (n_bins, n_frames).
    bpm_axis : np.ndarray
        BPM values for each bin.
    base_bpm : float
        Base BPM for cyclic computation (default 60).
    M : int
        Number of cyclic bins (default 120).

    Returns
    -------
    cyclic : np.ndarray
        Cyclic tempogram (M, n_frames).
    """
    cyclic = np.zeros((M, tempogram.shape[1]))
    bpm_axis = np.asarray(bpm_axis)
    valid_mask = np.isfinite(bpm_axis) & (bpm_axis > 0)
    if not np.any(valid_mask):
        return cyclic
    valid_bpm = bpm_axis[valid_mask]
    log_ratio = np.log2(valid_bpm / base_bpm)
    cyclic_idx = (np.mod(log_ratio, 1.0) * M).astype(int)
    valid_tempogram = tempogram[valid_mask]
    for i, idx in enumerate(cyclic_idx):
        if 0 <= idx < M:
            cyclic[idx] += valid_tempogram[i]
    return cyclic


def walk_audio_directory(audio_dir):
    """
    Walk audio directory and collect all track entries.

    Uses os.scandir for robust path handling (no string matching).
    Handles both flat layout (track_dir/preview.mp3) and nested layout
    (track_dir/actual_name/preview.mp3) for tracks where Beatport used
    a different actual name than the short directory name.

    Parameters
    ----------
    audio_dir : str
        Root directory containing genre subdirectories.

    Returns
    -------
    entries : list of (genre, song_name, mp3_path)
    """
    entries = []
    for genre_entry in sorted(os.scandir(audio_dir), key=lambda e: e.name):
        if not genre_entry.is_dir() or genre_entry.name.endswith('.json'):
            continue
        genre_name = genre_entry.name.replace('-Top100-19_03_2025', '')
        for track_entry in sorted(os.scandir(genre_entry.path), key=lambda e: e.name):
            if not track_entry.is_dir():
                continue
            mp3_path = os.path.join(track_entry.path, 'preview.mp3')
            if not os.path.isfile(mp3_path):
                # Check one level deeper: some tracks have an extra subdirectory
                found = False
                try:
                    for sub_entry in os.scandir(track_entry.path):
                        if sub_entry.is_dir():
                            nested_mp3 = os.path.join(sub_entry.path, 'preview.mp3')
                            if os.path.isfile(nested_mp3):
                                mp3_path = nested_mp3
                                found = True
                                break
                except OSError:
                    pass
                if not found:
                    print(f"  ⚠️ No preview.mp3 in {track_entry.path}")
                    continue
            parts = track_entry.name.split(' - ', 1)
            song_name = parts[1].strip() + '.mp3' if len(parts) > 1 else track_entry.name.strip() + '.mp3'
            entries.append((genre_name, song_name, mp3_path))
    return entries


def load_metadata(audio_dir):
    """
    Load metadata (BPM, Key, Length) from metadata.json files.

    Parameters
    ----------
    audio_dir : str
        Root directory containing genre subdirectories.

    Returns
    -------
    metadata : dict
        Mapping of (genre, song_name) -> {meta.Bpm, meta.Key, meta.Length_ms}.
    """
    metadata = {}
    for genre_entry in sorted(os.scandir(audio_dir), key=lambda e: e.name):
        if not genre_entry.is_dir() or genre_entry.name.endswith('.json'):
            continue
        genre_name = genre_entry.name.replace('-Top100-19_03_2025', '')
        for track_entry in sorted(os.scandir(genre_entry.path), key=lambda e: e.name):
            if not track_entry.is_dir():
                continue

            # Find metadata.json (same nesting logic as walk_audio_directory)
            json_path = os.path.join(track_entry.path, 'metadata.json')
            if not os.path.isfile(json_path):
                try:
                    for sub_entry in os.scandir(track_entry.path):
                        if sub_entry.is_dir():
                            nested_json = os.path.join(sub_entry.path, 'metadata.json')
                            if os.path.isfile(nested_json):
                                json_path = nested_json
                                break
                except OSError:
                    continue

            if not os.path.isfile(json_path):
                continue

            parts = track_entry.name.split(' - ', 1)
            song_name = parts[1].strip() + '.mp3' if len(parts) > 1 else track_entry.name.strip() + '.mp3'
            try:
                with open(json_path, 'r') as f:
                    meta = json.load(f)
                metadata[(genre_name, song_name)] = {
                    'meta.Bpm': meta.get('bpm', None),
                    'meta.Key': meta.get('key', None),
                    'meta.Length_ms': meta.get('length_ms', meta.get('duration_ms', None)),
                }
            except Exception as e:
                print(f"  ⚠️ Metadata error for {genre_name}/{song_name}: {e}")
    return metadata
