"""
Regenerate the confusion-matrix figure (Fig. 3) with publication styling.

Data source: the local pipeline's clustering results CSV produced by
main_analysis.py on the top100 dataset (consistent with Table 1).

Style adjustments requested:
  - Title: concise and informative
  - Axis labels: 6× original (~7pt → 42pt)
  - Y-axis genre ticks: 5× original (~6pt → 30pt)
  - Consistent Times New Roman font
  - Colorbar: closer, thinner, 4× label (~7pt → 28pt)
"""
import pathlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ── Global font ────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
})

# ── Paths ──────────────────────────────────────────────────
ROOT    = pathlib.Path(__file__).resolve().parents[2]          # repo root
CSV     = ROOT / 'results' / 'csv' / 'edm_clustering_summary.csv'
OUTFILE = ROOT / 'paper' / 'SMC' / 'figs' / 'paper' / 'confusion_matrix.png'
OUTFILE.parent.mkdir(parents=True, exist_ok=True)

# ── Load pre-computed labels ───────────────────────────────
df = pd.read_csv(CSV)
y_true = df['true_genre'].values
y_pred = df['kmeans_cluster_35'].values

# ── Build confusion matrix (cross-tab) ────────────────────
unique_true = sorted(np.unique(y_true))
unique_pred = sorted(np.unique(y_pred))

cm = np.zeros((len(unique_true), len(unique_pred)), dtype=int)
for i, g in enumerate(unique_true):
    for j, c in enumerate(unique_pred):
        cm[i, j] = ((y_true == g) & (y_pred == c)).sum()

# ── Custom blue colormap (matching original) ──────────────
colors = ['#FFFFFF', '#F0F8FF', '#E6F3FF', '#D6EBFF', '#C6E3FF',
          '#B3D9FF', '#99CCFF', '#80BFFF', '#66B2FF', '#4DA6FF',
          '#3399FF', '#1A8CFF', '#0080FF', '#0066CC', '#004C99']
cmap = LinearSegmentedColormap.from_list('custom_blues', colors, N=256)

# ── Font sizes (multiplied from original base) ────────────
BASE_AX_LABEL  = 7    # original axis label pt
BASE_TICK      = 6    # original genre-tick pt
BASE_CBAR      = 7    # original cbar-label pt

AX_LABEL  = BASE_AX_LABEL * 6   # 42 pt
GENRE_TICK = BASE_TICK * 5       # 30 pt
CBAR_LABEL = BASE_CBAR * 4      # 28 pt
CBAR_TICK  = 24
CELL_FONT  = 18
TITLE_SIZE = 42

# ── Plot ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(24, 18))

im = ax.imshow(cm, cmap=cmap, aspect='auto', interpolation='nearest')

# Cell annotations
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        val = cm[i, j]
        if val > 0:
            text_color = 'white' if val > cm.max() * 0.5 else 'black'
            ax.text(j, i, str(val), ha='center', va='center',
                    fontsize=CELL_FONT, fontweight='bold', color=text_color)

# Minor-tick grid
ax.set_xticks(np.arange(len(unique_pred) + 1) - 0.5, minor=True)
ax.set_yticks(np.arange(len(unique_true) + 1) - 0.5, minor=True)
ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
ax.tick_params(which='minor', size=0)

# Major ticks / labels
ax.set_xticks(range(len(unique_pred)))
ax.set_xticklabels(unique_pred, fontsize=GENRE_TICK)
ax.set_yticks(range(len(unique_true)))
ax.set_yticklabels(unique_true, fontsize=GENRE_TICK, fontweight='demibold')

# Rotate x ticks
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

# Axis labels
ax.set_xlabel('Acoustic Cluster',       fontsize=AX_LABEL, fontweight='bold')
ax.set_ylabel('Commercial Taxonomy',    fontsize=AX_LABEL, fontweight='bold')

# Title
ax.set_title('Correspondence Between Genre Labels and Acoustic Clusters',
             fontsize=TITLE_SIZE, fontweight='bold', color='black', pad=36)

# Colorbar — thinner & closer
cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.015)
cbar.set_label('Number of Songs', fontsize=CBAR_LABEL * 1.3, fontweight='bold',
               rotation=270, labelpad=36)
cbar.ax.tick_params(labelsize=CBAR_TICK)

plt.tight_layout()
plt.savefig(OUTFILE, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {OUTFILE}")
plt.close()
