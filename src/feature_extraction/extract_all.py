"""
extract_all.py — Main feature extraction orchestrator.
=====================================================
Extracts all audio features for the EDM Phantom Genre project.
Delegates to modular extractors for each feature group.

Usage:
    python src/feature_extraction/extract_all.py [--workers N]

Feature groups (194 total, extracted per track):
    extract_pyaudio.py                → 92 features  (Original pyAudioAnalysis + Essentia)
    extract_tempogram_features.py     → 64 features  (Fourier/auto/cyclic tempogram)
    extract_spectral_texture.py       → 18 features  (contrast, flatness, bandwidth)
    extract_edm_production.py         → 15 features  (bass, rhythm, modulation, dynamics)
    extract_harmonic_percussive.py    →  5 features  (H/P ratio, dynamic range, crest)
    utils.py                          → shared I/O   (audio walk, metadata, cyclic tempogram)
"""

import os
import sys

import librosa
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add parent dir to path so we can import the feature_extraction package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_extraction.modules.extract_pyaudio import PyAudioFeaturesExtractor
from feature_extraction.modules.extract_edm_production import EDMProductionExtractor
from feature_extraction.modules.extract_tempogram_features import TempogramExtractor
from feature_extraction.modules.extract_spectral_texture import SpectralTextureExtractor
from feature_extraction.modules.extract_harmonic_percussive import HPDynamicsExtractor
from feature_extraction.modules.utils import process_in_parallel, save_checkpoint, walk_audio_directory, load_metadata


def extract_all_features(filename):
    """
    Extract all 194 features from a single audio file.

    Computes expensive operations (STFT, HPSS, RMS, onset envelope) once
    and shares them across all four librosa feature extractors to avoid redundant
    computation. The pyAudioAnalysis/Essentia wrapper handles its own inputs.

    Parameters
    ----------
    filename : str
        Path to audio file (MP3/WAV).

    Returns
    -------
    features : dict
        Dictionary of feature_name → float (194 entries).
    """
    y, sr = librosa.load(filename, sr=None)
    if sr != 44100:
        y = librosa.resample(y, orig_sr=sr, target_sr=44100)
        sr = 44100
    hop_length = 512

    # === Shared expensive computations (computed ONCE) ===
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length))      # magnitude STFT
    S_power = S ** 2                                                     # power STFT
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    y_harmonic, y_percussive = librosa.effects.hpss(y)                   # HPSS (slowest op)
    rms_full = librosa.feature.rms(y=y, hop_length=hop_length)[0]

    # === Extract all feature groups ===
    features = {}
    features.update(extract_pyaudio_features(y, sr, onset_env, hop_length))               # 92
    features.update(extract_tempogram_features(onset_env, y, sr, hop_length))             # 64
    features.update(extract_spectral_texture_features(S, sr))                             # 18
    features.update(extract_edm_features(y, sr, hop_length, onset_env, S_power, freqs, y_percussive, rms_full))  # 15
    features.update(extract_hp_dynamic_features(y, sr, hop_length, y_harmonic, y_percussive, rms_full))          # 5

    return features


def process_track(entry):
    """Process a single track. Returns features dict or None on failure."""
    genre, song_name, mp3_path = entry
    try:
        feats = extract_all_features(mp3_path)
        feats['genre'] = genre
        feats['song'] = song_name
        return feats
    except Exception as e:
        print(f"  ❌ [{genre}] {song_name}: {e}")
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Extract tempogram + spectral + EDM-specific features'
    )
    parser.add_argument('--workers', type=int, default=max(1, cpu_count() - 1),
                        help='Number of parallel workers (default: CPU count - 1)')
    parser.add_argument('--audio-dir', type=str, default=None,
                        help='Override audio directory path')
    parser.add_argument('--output', type=str, default=None,
                        help='Override output CSV path')
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    audio_dir = args.audio_dir or os.path.join(project_root, 'dataset', 'BeatportTop100-19_03_2025')
    output_csv = args.output or os.path.join(project_root, 'dataset', 'tempogram_features.csv')

    print("=" * 60)
    print("FEATURE EXTRACTION (Tempogram + Spectral + EDM-Specific)")
    print(f"Workers: {args.workers}")
    print("=" * 60)

    entries = walk_audio_directory(audio_dir)
    print(f"\nFound {len(entries)} audio files across {len(set(e[0] for e in entries))} genres")

    print("\nExtracting features...")
    results = []
    failed = []

    if args.workers > 1:
        with Pool(processes=args.workers) as pool:
            for result in tqdm(pool.imap_unordered(process_track, entries),
                              total=len(entries), desc="Processing"):
                if result is not None:
                    results.append(result)
                else:
                    failed.append(None)
    else:
        for entry in tqdm(entries, desc="Processing"):
            result = process_track(entry)
            if result is not None:
                results.append(result)
            else:
                failed.append(entry)

    print(f"\n✅ Succeeded: {len(results)}")
    print(f"❌ Failed: {len(failed)}")
    if failed and args.workers == 1:
        print("\nFailed tracks:")
        for item in failed:
            if item is not None:
                genre, song, path = item
                print(f"  [{genre}] {song}")

    df = pd.DataFrame(results)

    # Load metadata
    print("\nLoading metadata...")
    metadata = load_metadata(audio_dir)
    print(f"  Found metadata for {len(metadata)} tracks")

    meta_rows = []
    for _, row in df.iterrows():
        key = (row['genre'], row['song'])
        if key in metadata:
            meta_rows.append(metadata[key])
        else:
            meta_rows.append({'meta.Bpm': None, 'meta.Key': None, 'meta.Length_ms': None})

    df_meta = pd.DataFrame(meta_rows)
    df_final = pd.concat([df, df_meta], axis=1)

    print(f"\nFinal dataset: {len(df_final)} tracks, {df_final.shape[1]} columns")
    missing_meta = df_final['meta.Bpm'].isna().sum()
    print(f"Tracks missing metadata: {missing_meta}")

    df_final.to_csv(output_csv, index=False)
    print(f"\n✅ Saved to {output_csv}")


if __name__ == "__main__":
    main()
