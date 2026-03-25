"""
plot_dendrograms.py
===================
Figure 5 — Hierarchical clustering dendrogram of EDM genres based purely on 
acoustic structure, without enforcing cultural lineages.

Loads the pre-computed co-clustering convergence matrix to ensure consistency.
"""

import os, sys, warnings
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

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 9,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# ── Data-Derived Acoustic Families ──────────────────────────────────
# Based purely on the internal linkage structure of the genre_convergence_matrix.csv

FAMILY_COLOURS = {
    'High-Energy Bass & Hardcore': '#4d84b2',   # softer blue
    'Groove & Breaks Cluster':     '#5eba7d',   # muted green
    'Melodic & Atmospheric':       '#a879cc',   # soft purple
    'Commercial & Mainstage':      '#e08b46',   # muted orange
    'Unaffiliated':                '#9ca3a4',   # light gray
}

GENRE_TO_FAMILY = {
    # High-Energy Bass & Hardcore
    '140 - Deep Dubstep - Grime':        'High-Energy Bass & Hardcore',
    'Drum & Bass':                       'High-Energy Bass & Hardcore',
    'Dubstep':                           'High-Energy Bass & Hardcore',
    'Hard Dance - Hardcore - Neo Rave':  'High-Energy Bass & Hardcore',
    'Hard Techno':                       'High-Energy Bass & Hardcore',
    'Trap - Future Bass':                'High-Energy Bass & Hardcore',

    # Groove & Breaks Cluster
    'Bass - Club':                       'Groove & Breaks Cluster',
    'Bass House':                        'Groove & Breaks Cluster',
    'Breaks - Breakbeat - UK Bass':      'Groove & Breaks Cluster',
    'Deep House':                        'Groove & Breaks Cluster',
    'Electro (Classic - Detroit - Modern)': 'Groove & Breaks Cluster',
    'House':                             'Groove & Breaks Cluster',
    'Indie Dance':                       'Groove & Breaks Cluster',
    'Jackin House':                      'Groove & Breaks Cluster',
    'Minimal - Deep Tech':               'Groove & Breaks Cluster',
    'Tech House':                        'Groove & Breaks Cluster',
    'UK Garage - Bassline':              'Groove & Breaks Cluster',

    # Melodic & Atmospheric
    'Afro House':                        'Melodic & Atmospheric',
    'Amapiano':                          'Melodic & Atmospheric',
    'Downtempo':                         'Melodic & Atmospheric',
    'Melodic House & Techno':            'Melodic & Atmospheric',
    'Organic House':                     'Melodic & Atmospheric',
    'Progressive House':                 'Melodic & Atmospheric',
    'Psy-Trance':                        'Melodic & Atmospheric',
    'Techno (Peak Time - Driving)':      'Melodic & Atmospheric',
    'Techno (Raw - Deep - Hypnotic)':    'Melodic & Atmospheric',
    'Trance (Raw - Deep - Hypnotic)':    'Melodic & Atmospheric',

    # Commercial & Mainstage
    'Dance - Pop':                       'Commercial & Mainstage',
    'Funky House':                       'Commercial & Mainstage',
    'Mainstage':                         'Commercial & Mainstage',
    'Nu Disco - Disco':                  'Commercial & Mainstage',
    'Trance (Main Floor)':               'Commercial & Mainstage',

    # Unaffiliated (Isolates)
    'Ambient - Experimental':            'Unaffiliated',
    'DJ Tools':                          'Unaffiliated',
    'Electronica':                       'Unaffiliated',
}

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
    return GENRE_TO_FAMILY.get(g, 'Unaffiliated')

def _colour(g):
    return FAMILY_COLOURS[_family(g)]

def plot_dendrograms(csv_path='results/csv/genre_convergence_matrix.csv',
                     save_path='paper/SMC/figs/paper/fig5_dendrograms.png'):
    
    # ── Load Affinity Matrix ───────────────────────────────────────
    df = pd.read_csv(csv_path, index_col=0)
    genres = df.index.values
    affinity = df.values
    
    # Normalise affinity to [0, 1]
    np.fill_diagonal(affinity, 1.0)
    aff_max = affinity.max()
    affinity_norm = affinity / aff_max if aff_max > 0 else affinity
    
    # Distance = 1 - normalised affinity
    distance = 1.0 - affinity_norm
    np.fill_diagonal(distance, 0)
    
    # Ensure perfect symmetry
    distance = (distance + distance.T) / 2.0
    dist_condensed = squareform(distance, checks=False)
    
    # Average linkage
    Z = linkage(dist_condensed, method='average')
    
    labels_short = [_short(g) for g in genres]
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
            return FAMILY_COLOURS.get(fl, '#7f8c8d')
        return '#bdc3c7' # Grey for mixed branches

    # ── Figure (vertical: genres on y-axis) ────────────────────────
    # Width: 5.5 inches for legend breathing room
    # Height: enough to fit 35 larger labels without overlap (9.0 inches)
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 9.0))

    dn = dendrogram(
        Z,
        labels=labels_short,
        ax=ax,
        orientation='left',
        leaf_font_size=12,
        link_color_func=_link_colour,
        leaf_rotation=0,
    )

    # Re-colour y-tick labels by genre family
    for lbl in ax.get_yticklabels():
        text = lbl.get_text()
        genre_full = next((g for g in genres if _short(g) == text), None)
        if genre_full:
            lbl.set_color(_colour(genre_full))
            lbl.set_fontsize(12)
            lbl.set_fontweight('bold')

    ax.set_xlabel('Co-clustering Distance', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='x', labelsize=12)
    
    # Legend
    # Exclude 'Unaffiliated' if you don't want it in the legend, or include for clarity
    patches = [
        mpatches.Patch(color=c, label=fam)
        for fam, c in FAMILY_COLOURS.items()
    ]
    ax.legend(
        handles=patches, loc='lower center', bbox_to_anchor=(0.65, 1.00), 
        fontsize=10.5, frameon=True, framealpha=0.95, edgecolor='#cccccc',
        title='Acoustic Families',
        title_fontproperties={'weight': 'bold', 'size': 11.5},
        ncol=2
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Saved: {save_path}')
    plt.close(fig)

if __name__ == '__main__':
    plot_dendrograms()
