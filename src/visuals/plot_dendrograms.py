"""
plot_dendrograms.py
===================
Figure 5 — Hierarchical clustering dendrogram of EDM genres.

Uses Ward's method on a co-clustering affinity matrix (K-means,
50 restarts, k=35) converted to a distance matrix.  This captures
*behavioural* genre relationships (how often genres co-cluster)
rather than raw centroid distance.

Vertical orientation (genres on y-axis) for single-column layout.

Usage:
    python -m visuals.plot_dendrograms          # from src/
    python src/visuals/plot_dendrograms.py      # from project root
"""

import os, sys, pickle, warnings
import numpy as np
import pandas as pd
from collections import Counter
from itertools import combinations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')

# ── Publication-quality font defaults (match other figures) ─────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# ── Genre family colour map ─────────────────────────────────────────
FAMILY_COLOURS = {
    'UK Bass Pole':              '#1a5276',   # deep navy
    'House Cluster':             '#27ae60',   # green
    'Trance Cluster':            '#8e44ad',   # purple
    'Other':                     '#95a5a6',   # muted grey
}

GENRE_TO_FAMILY = {
    # UK Continuum
    'Drum & Bass':                       'UK Bass Pole',
    'Dubstep':                           'UK Bass Pole',
    'UK Garage - Bassline':              'UK Bass Pole',
    '140 - Deep Dubstep - Grime':        'UK Bass Pole',
    'Breaks - Breakbeat - UK Bass':      'UK Bass Pole',
    'Bass House':                        'UK Bass Pole',
    'Bass - Club':                       'UK Bass Pole',
    'Hard Techno':                       'UK Bass Pole',
    'Hard Dance - Hardcore - Neo Rave':  'UK Bass Pole',
    'Trap - Future Bass':                'UK Bass Pole',

    # House Archipelago
    'Deep House':                        'House Cluster',
    'Funky House':                       'House Cluster',
    'House':                             'House Cluster',
    'Jackin House':                      'House Cluster',
    'Tech House':                        'House Cluster',
    'Afro House':                        'House Cluster',
    'Organic House':                     'House Cluster',
    'Indie Dance':                       'House Cluster',
    'Nu Disco - Disco':                  'House Cluster',
    'Amapiano':                          'House Cluster',

    # Trance–Progressive Axis
    'Trance (Main Floor)':               'Trance Cluster',
    'Trance (Raw - Deep - Hypnotic)':    'Trance Cluster',
    'Progressive House':                 'Trance Cluster',
    'Psy-Trance':                        'Trance Cluster',
    'Melodic House & Techno':            'Trance Cluster',
    'Mainstage':                         'Trance Cluster',
}

# Short names for readability
SHORT_NAMES = {
    'Drum & Bass': 'D&B',
    'Breaks - Breakbeat - UK Bass': 'Breaks',
    '140 - Deep Dubstep - Grime': '140/Grime',
    'Techno (Peak Time - Driving)': 'Peak Techno',
    'Techno (Raw - Deep - Hypnotic)': 'Deep Techno',
    'Trance (Main Floor)': 'Trance',
    'Trance (Raw - Deep - Hypnotic)': 'Deep Trance',
    'Melodic House & Techno': 'Melodic H&T',
    'Hard Dance - Hardcore - Neo Rave': 'Hardcore',
    'Dance - Pop': 'Dance Pop',
    'Electro (Classic - Detroit - Modern)': 'Electro',
    'Ambient - Experimental': 'Ambient',
    'Trap - Future Bass': 'Trap/Future',
    'Minimal - Deep Tech': 'Minimal',
    'Bass - Club': 'Bass/Club',
    'Nu Disco - Disco': 'Nu Disco',
    'UK Garage - Bassline': 'UK Garage',
    'Hard Techno': 'Hard Techno',
    'Progressive House': 'Prog House',
}


def _short(g):
    return SHORT_NAMES.get(g, g)


def _family(g):
    return GENRE_TO_FAMILY.get(g, 'Other')


def _colour(g):
    return FAMILY_COLOURS[_family(g)]


# ═══════════════════════════════════════════════════════════════════════
# Co-clustering affinity  (same method as plot_genre_network.py)
# ═══════════════════════════════════════════════════════════════════════

def compute_affinity_matrix(X, y, n_runs=50, k=35, random_state=42):
    """
    Compute co-clustering affinity between every genre pair.

    For each of `n_runs` K-means runs with different seeds, count how many
    tracks from genre-i and genre-j land in the same cluster, normalised
    by the geometric mean of genre sizes.  Average across runs.
    """
    genres = np.unique(y)
    n_genres = len(genres)
    genre_idx = {g: i for i, g in enumerate(genres)}
    affinity = np.zeros((n_genres, n_genres))

    for run in range(n_runs):
        km = KMeans(n_clusters=k, n_init=3, max_iter=200,
                    random_state=random_state + run)
        labels = km.fit_predict(X)

        for c in range(k):
            mask = labels == c
            genres_in_cluster = y[mask]
            counts = Counter(genres_in_cluster)
            for (g1, c1), (g2, c2) in combinations(counts.items(), 2):
                i1, i2 = genre_idx[g1], genre_idx[g2]
                norm = np.sqrt(c1 * c2)
                affinity[i1, i2] += norm
                affinity[i2, i1] += norm

    affinity /= n_runs
    np.fill_diagonal(affinity, 0)
    return affinity, genres


# ═══════════════════════════════════════════════════════════════════════
# Main plotting function
# ═══════════════════════════════════════════════════════════════════════

def plot_dendrograms(
    X, y,
    save_path='paper/SMC/figs/paper/fig5_dendrograms.png',
    n_runs=50,
):
    """
    Build a dendrogram from co-clustering affinity, vertical layout.
    """
    # ── Compute affinity → distance ────────────────────────────────
    affinity, genres = compute_affinity_matrix(X, y, n_runs=n_runs)

    # Normalise affinity to [0, 1]
    aff_max = affinity.max()
    if aff_max > 0:
        affinity_norm = affinity / aff_max
    else:
        affinity_norm = affinity

    # Distance = 1 - normalised affinity
    distance = 1.0 - affinity_norm
    np.fill_diagonal(distance, 0)

    # Ensure perfect symmetry
    distance = (distance + distance.T) / 2.0

    # Convert to condensed form for linkage
    dist_condensed = squareform(distance, checks=False)

    # ── Average linkage (best for affinity-derived distances) ──────
    Z = linkage(dist_condensed, method='average')

    # ── Short labels ───────────────────────────────────────────────
    labels_short = [_short(g) for g in genres]

    # Custom link colouring by genre family
    n = len(genres)

    def _leaf_family(idx):
        if idx < n:
            return _family(genres[idx])
        ll = _leaf_family(int(Z[idx - n, 0]))
        rr = _leaf_family(int(Z[idx - n, 1]))
        return ll if ll == rr else 'Mixed'

    def _link_colour(link_id):
        left_id = int(Z[link_id - n, 0])
        right_id = int(Z[link_id - n, 1])
        fl = _leaf_family(left_id)
        fr = _leaf_family(right_id)
        if fl == fr and fl != 'Mixed':
            return FAMILY_COLOURS.get(fl, '#95a5a6')
        return '#bdc3c7'

    # ── Figure (vertical: genres on y-axis) ────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(5, 10))

    dn = dendrogram(
        Z,
        labels=labels_short,
        ax=ax,
        orientation='left',        # vertical = genres on y-axis
        leaf_font_size=10,
        link_color_func=_link_colour,   # correct per-link colouring
        leaf_rotation=0,
    )

    # Re-colour y-tick labels by genre family
    for lbl in ax.get_yticklabels():
        text = lbl.get_text()
        genre_full = next(
            (g for g in genres if _short(g) == text), None
        )
        if genre_full:
            lbl.set_color(_colour(genre_full))
            lbl.set_fontweight('bold')
        lbl.set_fontsize(10)

    ax.set_xlabel('Co-clustering Distance', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=10)

    # Legend
    patches = [
        mpatches.Patch(color=c, label=fam)
        for fam, c in FAMILY_COLOURS.items()
    ]
    ax.legend(
        handles=patches, loc='upper left', fontsize=9,
        frameon=True, framealpha=0.95, edgecolor='#cccccc',
        title='Acoustic Affinity',
        title_fontproperties={'weight': 'bold', 'size': 10},
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Saved: {save_path}')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(script_dir) == 'visuals':
        project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    else:
        project_root = os.getcwd()
    os.chdir(project_root)

    feat_path = 'results/pkl/feature_matrix.pkl'
    if not os.path.exists(feat_path):
        print(f'ERROR: {feat_path} not found. Run the pipeline first.')
        sys.exit(1)

    with open(feat_path, 'rb') as f:
        feat_data = pickle.load(f)

    X = StandardScaler().fit_transform(feat_data['X_selected'])
    y = np.array(feat_data['genre_labels'])

    plot_dendrograms(X, y)
