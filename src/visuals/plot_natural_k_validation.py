#!/usr/bin/env python3
"""
Generate natural-k validation figure showing multiple clustering metrics
across k values, with consensus optimal k marked.

Replaces Table 2 in the manuscript.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Paths
PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PKL_PATH = os.path.join(PROJECT, 'results', 'pkl', 'feature_matrix.pkl')
CACHE_PATH = os.path.join(PROJECT, 'results', 'pkl', 'natural_k_sweep.pkl')
OUT_DIR = os.path.join(PROJECT, 'paper', 'SMC', 'figs', 'paper')
os.makedirs(OUT_DIR, exist_ok=True)


def run_sweep(X, min_k=10, max_k=30):
    """Sweep K-means across k values and collect metrics."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

    k_values = list(range(min_k, max_k + 1))
    metrics = {'silhouette': [], 'calinski': [], 'davies_bouldin': [], 'inertia': []}

    for k in k_values:
        print(f"  k={k}...", end=" ", flush=True)
        km = KMeans(n_clusters=k, init='k-means++', n_init=50,
                    max_iter=300, random_state=42)
        labels = km.fit_predict(X)
        metrics['silhouette'].append(silhouette_score(X, labels))
        metrics['calinski'].append(calinski_harabasz_score(X, labels))
        metrics['davies_bouldin'].append(davies_bouldin_score(X, labels))
        metrics['inertia'].append(km.inertia_)
        print(f"Sil={metrics['silhouette'][-1]:.4f}")

    for key in metrics:
        metrics[key] = np.array(metrics[key])

    return k_values, metrics


def find_elbow(k_values, values):
    """Find elbow point using the maximum distance from the line connecting endpoints."""
    k_arr = np.array(k_values, dtype=float)
    v_arr = np.array(values, dtype=float)

    # Normalize
    k_norm = (k_arr - k_arr[0]) / (k_arr[-1] - k_arr[0])
    v_norm = (v_arr - v_arr[0]) / (v_arr[-1] - v_arr[0] + 1e-12)

    # Distance from line connecting first and last point
    line_vec = np.array([k_norm[-1] - k_norm[0], v_norm[-1] - v_norm[0]])
    line_len = np.linalg.norm(line_vec)
    line_unit = line_vec / (line_len + 1e-12)

    distances = []
    for i in range(len(k_arr)):
        point_vec = np.array([k_norm[i] - k_norm[0], v_norm[i] - v_norm[0]])
        proj = np.dot(point_vec, line_unit)
        proj_point = line_unit * proj
        dist = np.linalg.norm(point_vec - proj_point)
        distances.append(dist)

    return k_values[np.argmax(distances)]


def plot_validation_figure(k_values, metrics, consensus_k, method_ks, out_path):
    """Create a single-panel figure with all four metrics normalized and overlaid."""
    BASE = 9
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times'],
        'font.size': BASE,
        'axes.labelsize': int(BASE * 1.4),
        'axes.titlesize': 11,
        'xtick.labelsize': BASE,
        'ytick.labelsize': BASE,
    })

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    fig.subplots_adjust(top=0.95, bottom=0.18, left=0.13, right=0.97)

    # Normalize each metric to [0, 1]; invert "lower is better" metrics
    # so all curves read as "higher = better"
    line_specs = [
        ('silhouette',    'Silhouette ↑',         True,  '#2563EB', 'o'),
        ('inertia',       'Inertia (elbow) ↓',    False, '#059669', 's'),
        ('davies_bouldin','Davies-Bouldin ↓',      False, '#D97706', '^'),
        ('calinski',      'Calinski-Harabasz ↑',   True,  '#7C3AED', 'D'),
    ]

    for key, label, higher_better, color, marker in line_specs:
        raw = metrics[key]
        # Min-max normalize
        lo, hi = raw.min(), raw.max()
        normed = (raw - lo) / (hi - lo + 1e-12)
        if not higher_better:
            normed = 1.0 - normed  # invert so "up = good" for all
        ax.plot(k_values, normed, marker=marker, linestyle='-', color=color,
                linewidth=1.4, markersize=4.5, markerfacecolor='white',
                markeredgewidth=1.2, markeredgecolor=color, label=label,
                zorder=3)

    # Consensus k line
    ax.axvline(x=consensus_k, color='#DC2626', linestyle='--',
               linewidth=1.8, alpha=0.85, zorder=2,
               label=f'Consensus $k$={consensus_k}')

    # Shade the consensus region lightly
    ax.axvspan(consensus_k - 0.5, consensus_k + 0.5,
               color='#FEE2E2', alpha=0.5, zorder=1)

    ax.set_xlabel('Number of Clusters ($k$)', fontweight='bold')
    ax.set_ylabel('Normalized Score', fontweight='bold')
    ax.set_xlim(k_values[0] - 0.5, k_values[-1] + 0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(loc='lower left', fontsize=7, frameon=True,
              fancybox=True, edgecolor='#E5E7EB', ncol=1)

    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {out_path}")
    plt.close()


def load_or_compute_sweep():
    """Load cached sweep results, or compute and cache them."""
    if os.path.exists(CACHE_PATH):
        print(f"Loading cached sweep from {CACHE_PATH}")
        with open(CACHE_PATH, 'rb') as f:
            cache = pickle.load(f)
        return cache['k_values'], cache['metrics']

    print("Loading feature matrix...")
    fm = pickle.load(open(PKL_PATH, 'rb'))
    X = pd.DataFrame(fm['X_selected'], columns=fm['feature_names'])

    print("Running K-means sweep (k=10..30)...")
    k_values, metrics = run_sweep(X, min_k=10, max_k=30)

    # Cache results
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump({'k_values': k_values, 'metrics': metrics}, f)
    print(f"Cached sweep to {CACHE_PATH}")

    return k_values, metrics


def main():
    k_values, metrics = load_or_compute_sweep()

    # Find optimal k per method (matching pipeline logic)
    method_ks = {
        'elbow': find_elbow(k_values, metrics['inertia']),
        'silhouette': find_elbow(k_values, [-s for s in metrics['silhouette']]),
        'calinski': find_elbow(k_values, [-c for c in metrics['calinski']]),
        'davies_bouldin': find_elbow(k_values, metrics['davies_bouldin']),
    }

    print(f"\nOptimal k by method:")
    for m, k in method_ks.items():
        print(f"  {m:15s}: {k}")

    # Consensus: mode voting, ties → lower k
    from collections import Counter
    votes = Counter(method_ks.values())
    max_votes = max(votes.values())
    consensus_k = min(k for k, v in votes.items() if v == max_votes)
    print(f"  {'CONSENSUS':15s}: {consensus_k}")

    out_path = os.path.join(OUT_DIR, 'figa2_natural_k_validation.png')
    plot_validation_figure(k_values, metrics, consensus_k, method_ks, out_path)


if __name__ == '__main__':
    main()
