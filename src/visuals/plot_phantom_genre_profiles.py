"""
plot_phantom_genre_profiles.py
==============================
Create a 2×2 radar figure comparing *acoustically convergent* genre
pairs — genres that frequently co-cluster AND whose percentile-ranked
acoustic profiles differ by less than 15 points on average.

Reuses the same feature-mapping and percentile-profile logic as
plot_musical_profile.py, but computes profiles per *genre* rather than
per *cluster*, then overlays two genres per subplot.

Output: paper/SMC/figs/paper/figa4_phantom_genre_profiles.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from math import pi
import pickle
import os
import re
import warnings
warnings.filterwarnings('ignore')

# --- Publication-quality defaults (match Fig 4) ---
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


# ────────────────────────────────────────────────────────────────
#  Explicit feature → dimension mapping  (shared with plot_musical_profile.py)
#  Every numeric feature is assigned to exactly one radar dimension.
# ────────────────────────────────────────────────────────────────

from plot_musical_profile import (
    _ENERGY_FEATURES, _DANCEABILITY_FEATURES, _TEMPO_FEATURES,
    _HARMONIC_FEATURES, _RHYTHMIC_FEATURES, _ELECTRONIC_FEATURES,
    create_feature_mapping,
)


# ────────────────────────────────────────────────────────────────
#  Genre-level profile computation  (z-score + p5/p95 percentile)
# ────────────────────────────────────────────────────────────────

def compute_genre_profiles(df, y_true, feature_mapping):
    """Compute p5/p95-scaled profiles per commercial genre label.

    For each dimension:
      1. Z-score normalise every feature so all contribute equally.
      2. Average the normalised features per track → per-dimension score.
      3. Scale each genre's mean score to 0–100 via p5/p95 linear
         interpolation and clip to [0, 100].
    """
    profiles = {}
    for genre in np.unique(y_true):
        mask = y_true == genre
        profile = {}
        for dim, features in feature_mapping.items():
            if not features:
                profile[dim] = 50.0
                continue
            try:
                raw = df[features]
                mu = raw.mean()
                sigma = raw.std().replace(0, 1)
                normed = (raw - mu) / sigma
                dim_vals = normed[mask].mean(axis=1)
                all_vals = normed.mean(axis=1)
            except Exception:
                profile[dim] = 50.0
                continue

            if len(all_vals) > 0:
                lo, hi = np.percentile(all_vals, 5), np.percentile(all_vals, 95)
                if hi > lo:
                    profile[dim] = float(np.clip(
                        (dim_vals.mean() - lo) / (hi - lo) * 100, 0, 100))
                else:
                    profile[dim] = 50.0
            else:
                profile[dim] = 50.0
        profiles[genre] = profile
    return profiles


# ────────────────────────────────────────────────────────────────
#  Identify acoustically convergent genre pairs
#  (dual filter: high co-clustering + low profile difference)
# ────────────────────────────────────────────────────────────────

def _profile_delta(profiles, g1, g2):
    """Mean absolute difference across six radar dimensions."""
    dims = list(profiles[g1].keys())
    return np.mean([abs(profiles[g1][d] - profiles[g2][d]) for d in dims])


def find_convergent_pairs(cluster_labels, y_true, profiles,
                          max_delta=15.0, top_n=4):
    """Find genre pairs with high co-clustering AND Δ < max_delta."""
    from itertools import combinations
    unique_genres = np.unique(y_true)
    pair_scores = []

    for g1, g2 in combinations(unique_genres, 2):
        # Skip if either genre not in profiles
        if g1 not in profiles or g2 not in profiles:
            continue

        mask1 = y_true == g1
        mask2 = y_true == g2
        cl1 = cluster_labels[mask1]
        cl2 = cluster_labels[mask2]

        # Co-clustering rate
        n_same = 0
        total = len(cl1) * len(cl2)
        for c in np.unique(cluster_labels):
            n1_in_c = (cl1 == c).sum()
            n2_in_c = (cl2 == c).sum()
            n_same += n1_in_c * n2_in_c

        co_rate = n_same / total if total > 0 else 0
        delta = _profile_delta(profiles, g1, g2)
        pair_scores.append((g1, g2, co_rate, delta))

    # Apply dual filter: Δ < max_delta, then sort by co-clustering
    filtered = [(g1, g2, cr, d) for g1, g2, cr, d in pair_scores if d < max_delta]
    filtered.sort(key=lambda x: x[2], reverse=True)

    print(f"\nDual filter: {len(filtered)}/{len(pair_scores)} pairs with Δ < {max_delta}")
    return filtered[:top_n]


# ────────────────────────────────────────────────────────────────
#  Plot
# ────────────────────────────────────────────────────────────────

def create_convergent_genre_figure(
        df, cluster_labels, y_true,
        save_path='paper/SMC/figs/paper/figa4_phantom_genre_profiles.png'):
    """Create 2×2 radar figure for acoustically convergent genre pairs."""

    # Feature mapping & genre profiles
    fmap = create_feature_mapping(df)
    profiles = compute_genre_profiles(df, y_true, fmap)

    # Convergent pairs (dual filter: co-clustering + Δ < 15)
    pairs = find_convergent_pairs(cluster_labels, y_true, profiles,
                                   max_delta=15.0, top_n=4)
    print("\nTop 4 acoustically convergent genre pairs:")
    for g1, g2, rate, delta in pairs:
        print(f"  {g1}  ↔  {g2}  :  co-cluster={rate:.3f}  Δ={delta:.1f}")

    # ── Radar setup ──
    categories = ['Energy', 'Dance.', 'Tempo', 'Harmonic', 'Rhythmic', 'Electronic']
    dim_keys = ['energy', 'danceability', 'tempo',
                'harmonic_complexity', 'rhythmic_density', 'electronic_texture']
    num_vars = len(categories)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]

    # ── Colour pairs (contrasting duo per subplot) ──
    pair_colors = [
        ('#E74C3C', '#3498DB'),   # red  / blue
        ('#8E44AD', '#E67E22'),   # purple / orange
        ('#16A085', '#C0392B'),   # teal / crimson
        ('#2980B9', '#27AE60'),   # blue / green
    ]
    bg_tints = ['#FFF5F5', '#FBF5FF', '#F0FDFB', '#F0F6FF']

    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.15)

    for idx, (g1, g2, co_rate, delta) in enumerate(pairs):
        ax = fig.add_subplot(gs[idx // 2, idx % 2], projection='polar')
        c1, c2 = pair_colors[idx]

        # Background & grid
        ax.set_facecolor(bg_tints[idx])
        ax.grid(True, linestyle='-', alpha=0.15, color='#888888', linewidth=0.6)
        for ring_val in [20, 40, 60, 80]:
            ring_angles = np.linspace(0, 2 * pi, 100)
            ax.plot(ring_angles, [ring_val] * 100,
                    linestyle=':', color='#AAAAAA', linewidth=0.5, alpha=0.6)

        # Genre 1
        vals1 = [profiles[g1].get(d, 50) for d in dim_keys] + \
                [profiles[g1].get(dim_keys[0], 50)]
        ax.plot(angles, vals1, linewidth=5, color=c1, alpha=0.15)          # glow
        ax.plot(angles, vals1, 'o-', linewidth=2.2, color=c1,
                markersize=5, markeredgecolor='white', markeredgewidth=0.8,
                label=g1[:22], zorder=3)
        ax.fill(angles, vals1, alpha=0.12, color=c1)

        # Genre 2
        vals2 = [profiles[g2].get(d, 50) for d in dim_keys] + \
                [profiles[g2].get(dim_keys[0], 50)]
        ax.plot(angles, vals2, linewidth=5, color=c2, alpha=0.15)          # glow
        ax.plot(angles, vals2, 's--', linewidth=2.2, color=c2,
                markersize=5, markeredgecolor='white', markeredgewidth=0.8,
                label=g2[:22], zorder=3)
        ax.fill(angles, vals2, alpha=0.12, color=c2)

        # Axis styling (match Fig 4)
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=18, fontweight='bold', color='#444444')
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80])
        ax.set_yticklabels(['20', '40', '60', '80'], size=7, color='#999999')
        ax.spines['polar'].set_color('#CCCCCC')
        ax.spines['polar'].set_linewidth(0.8)

        # Title: short genre names + co-clustering metric
        short1 = g1.split(' - ')[0][:16]
        short2 = g2.split(' - ')[0][:16]
        ax.set_title(f"{short1} vs {short2}\n"
                      f"Co-cluster: {co_rate:.2f}  |  Δ: {delta:.1f}",
                      fontsize=15, fontweight='bold', pad=12, color='black')

        ax.legend(loc='upper left', bbox_to_anchor=(-0.15, 1.06),
                  fontsize=15, frameon=True, fancybox=True,
                  edgecolor='#CCCCCC', framealpha=0.9)

    fig.suptitle('Acoustically Convergent Genre Pairs',
                 fontsize=27, fontweight='bold', color='black', y=0.98)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()


# ────────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from sklearn.cluster import KMeans

    # Load pre-computed features (same as plot_musical_profile.py)
    pkl_path = 'results/pkl/feature_matrix.pkl'
    print("Loading pre-computed features from saved pkl...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    X_selected = pd.DataFrame(
        data['X_selected'],
        columns=data['feature_names'] if data.get('feature_names') else None
    )
    y_true = pd.Series(data['genre_labels'], name='genre')

    csv_path = 'dataset/all_features_194.csv'
    df = pd.read_csv(csv_path)
    print(f"Loaded: {len(df)} songs × {df.shape[1]} features")

    # Data is already StandardScaler-transformed by the pipeline
    # Cluster with KMeans (k=35 to match paper)
    kmeans = KMeans(n_clusters=35, random_state=42, n_init=20, max_iter=500)
    cluster_labels = kmeans.fit_predict(X_selected)

    create_convergent_genre_figure(df, cluster_labels, y_true)
