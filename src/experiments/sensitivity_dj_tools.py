"""
Sensitivity analysis: re-run natural-k discovery after excluding DJ Tools tracks.
Compares the consensus optimal k with and without DJ Tools to verify robustness.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analysis.modules.clustering_algorithms import NaturalClusterFinder

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
})

# ── Load the saved feature matrix ──
with open('results/pkl/feature_matrix.pkl', 'rb') as f:
    data = pickle.load(f)

X = data['X_selected']
genre_labels = data['genre_labels']

print(f"Full dataset: {X.shape[0]} tracks, {X.shape[1]} features")
print(f"Unique genres: {len(np.unique(genre_labels))}")

# ── Run WITH all genres (baseline) ──
print("\n" + "="*60)
print("BASELINE: All 35 genres (3500 tracks)")
print("="*60)
finder_all = NaturalClusterFinder(min_clusters=15, max_clusters=30, random_state=42)
k_all, results_all = finder_all.find_optimal_k(X)
print(f"\n>>> Consensus optimal k (all genres): {k_all}")

# ── Exclude DJ Tools ──
dj_tools_mask = genre_labels != 'DJ Tools'
X_no_dj = X[dj_tools_mask]
labels_no_dj = genre_labels[dj_tools_mask]
n_excluded = (~dj_tools_mask).sum()

print(f"\n" + "="*60)
print(f"SENSITIVITY: Excluding DJ Tools ({n_excluded} tracks removed)")
print(f"Remaining: {X_no_dj.shape[0]} tracks, {len(np.unique(labels_no_dj))} genres")
print("="*60)
finder_no_dj = NaturalClusterFinder(min_clusters=15, max_clusters=30, random_state=42)
k_no_dj, results_no_dj = finder_no_dj.find_optimal_k(X_no_dj)
print(f"\n>>> Consensus optimal k (without DJ Tools): {k_no_dj}")

# ── Summary ──
print("\n" + "="*60)
print("SENSITIVITY ANALYSIS SUMMARY")
print("="*60)
print(f"  With DJ Tools:    k = {k_all}")
print(f"  Without DJ Tools: k = {k_no_dj}")
print(f"  Difference:       {abs(k_all - k_no_dj)}")
if k_all == k_no_dj:
    print("  ✓ Result is ROBUST — DJ Tools does not affect natural k")
else:
    print(f"  △ k shifted by {abs(k_all - k_no_dj)} — consider investigating further")

# ── Comparison figure ──
k_values = results_all['k_values']
metrics_all = results_all['metrics']
metrics_no_dj = results_no_dj['metrics']

fig, axes = plt.subplots(2, 2, figsize=(10, 7))

metric_info = [
    ('silhouette', 'Silhouette Score', axes[0, 0]),
    ('inertia', 'Inertia', axes[0, 1]),
    ('calinski', 'Calinski-Harabasz Index', axes[1, 0]),
    ('davies_bouldin', 'Davies-Bouldin Index', axes[1, 1]),
]

for metric_key, metric_name, ax in metric_info:
    vals_all = metrics_all[metric_key]
    vals_no_dj = metrics_no_dj[metric_key]
    
    ax.plot(k_values, vals_all, 'o-', color='#2b5ea7', linewidth=2,
            markersize=5, label='All 35 genres', alpha=0.85)
    ax.plot(k_values, vals_no_dj, 's--', color='#c44e2e', linewidth=2,
            markersize=5, label='Excl. DJ Tools (34)', alpha=0.85)
    
    ax.axvline(x=k_all, color='#2b5ea7', linestyle=':', alpha=0.5, linewidth=1.5,
               label=f'Consensus $k$={k_all}')
    
    ax.set_xlabel('$k$', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
    ax.set_title(metric_name, fontsize=13, fontweight='bold')
    ax.legend(fontsize=8.5, loc='best')
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle('Sensitivity Analysis: DJ Tools Exclusion',
             fontsize=15, fontweight='bold', y=1.02)
fig.tight_layout()

os.makedirs('results/figs/paper', exist_ok=True)
outpath = 'results/figs/figa3_sensitivity_dj_tools.png'
plt.savefig(outpath, dpi=200, bbox_inches='tight')
print(f"\nSaved comparison figure: {outpath}")
