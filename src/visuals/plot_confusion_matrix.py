"""
Fig 3: Genre–Cluster Correspondence with Purity Analysis

Layout (GridSpec, 3 rows × 1 col):
  ┌─────────────────────────────────────────┐
  │  Purity bar-chart  (top)                │
  ├─────────────────────────────────────────┤
  │  Main heatmap (genres × clusters)       │
  │  Y-axis has inline concentration bars   │
  │  normalised per-genre (row → %)         │
  ├─────────────────────────────────────────┤
  │  Cluster-size bar-chart (bottom)        │
  └─────────────────────────────────────────┘

Clusters  sorted by purity (high → low, left → right).
Genres    sorted by concentration (high → low, top → bottom).
"""
import pathlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

# ── Global font ────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
})

# ── Paths ──────────────────────────────────────────────────
ROOT    = pathlib.Path(__file__).resolve().parents[2]
CSV     = ROOT / 'results' / 'csv' / 'edm_clustering_summary.csv'
OUTFILE = ROOT / 'paper' / 'SMC' / 'figs' / 'paper' / 'fig3_confusion_matrix.png'
OUTFILE.parent.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────
df = pd.read_csv(CSV)
y_true = df['true_genre'].values
y_pred = df['kmeans_cluster_35'].values

# ── Build confusion matrix ─────────────────────────────────
genres  = sorted(np.unique(y_true))
clusters = sorted(np.unique(y_pred))
n_genres  = len(genres)
n_clusters = len(clusters)

cm = np.zeros((n_genres, n_clusters), dtype=int)
for i, g in enumerate(genres):
    for j, c in enumerate(clusters):
        cm[i, j] = ((y_true == g) & (y_pred == c)).sum()

# Row-normalised (per-genre %)
row_sums = cm.sum(axis=1, keepdims=True)
cm_pct = (cm / row_sums * 100)

# ── Cluster purity (dominant-genre share) ──────────────────
col_sums = cm.sum(axis=0)
cluster_purity = cm.max(axis=0) / col_sums * 100  # % of dominant genre
cluster_dominant = np.array([genres[i] for i in cm.argmax(axis=0)])

# ── Genre concentration (% of genre in its top cluster) ────
genre_concentration = cm.max(axis=1) / cm.sum(axis=1) * 100

# ── Sort: clusters by purity descending, genres by concentration descending
cluster_order = np.argsort(-cluster_purity)
genre_order   = np.argsort(-genre_concentration)

cm_sorted     = cm[np.ix_(genre_order, cluster_order)]
cm_pct_sorted = cm_pct[np.ix_(genre_order, cluster_order)]
purity_sorted = cluster_purity[cluster_order]
conc_sorted   = genre_concentration[genre_order]
size_sorted   = col_sums[cluster_order]

genres_sorted   = [genres[i] for i in genre_order]
clusters_sorted = [clusters[i] for i in cluster_order]

# ── Colour palette (blue/white, like reference) ────────────
heat_colors = ['#FFFFFF', '#DEEBF7', '#C6DBEF', '#9ECAE1',
               '#6BAED6', '#4292C6', '#2171B5', '#084594']
cmap = LinearSegmentedColormap.from_list('blue_white', heat_colors, N=256)

def purity_color(p):
    """Green-yellow-orange-red gradient for purity bars."""
    if p >= 70: return '#2ecc71'
    if p >= 50: return '#82c91e'
    if p >= 35: return '#f0c040'
    if p >= 20: return '#e67e22'
    return '#e74c3c'

# ── Figure layout (RECTANGLE) ──────────────────────────────
fig = plt.figure(figsize=(20, 16))
gs = gridspec.GridSpec(
    3, 1,
    height_ratios=[1, 10, 1],
    hspace=0.03,
)

ax_top   = fig.add_subplot(gs[0, 0])   # purity bars
ax_main  = fig.add_subplot(gs[1, 0])   # heatmap
ax_bot   = fig.add_subplot(gs[2, 0])   # cluster sizes

# ═══════════════════════════════════════════════════════════
# MAIN HEATMAP
# ═══════════════════════════════════════════════════════════
im = ax_main.imshow(cm_pct_sorted, cmap=cmap, aspect='auto',
                     interpolation='nearest', vmin=0, vmax=100)

# Cell annotations — ALL cells with raw count (numbers in every cell)
for i in range(n_genres):
    for j in range(n_clusters):
        raw = cm_sorted[i, j]
        pct = cm_pct_sorted[i, j]
        if raw > 0:
            color = 'white' if pct > 50 else 'black'
            # Increase fontsize slightly so numbers are easier to read
            fontsize = 12 if raw < 10 else 10
            ax_main.text(j, i, str(raw), ha='center', va='center',
                        fontsize=fontsize, fontweight='bold', color=color)

# Grid
ax_main.set_xticks(np.arange(n_clusters + 1) - 0.5, minor=True)
ax_main.set_yticks(np.arange(n_genres + 1)   - 0.5, minor=True)
ax_main.grid(which='minor', color='white', linewidth=1.2, alpha=0.8)
ax_main.tick_params(which='minor', size=0)

# ── Y-axis: genre labels with inline concentration bars ────
ax_main.set_yticks(range(n_genres))
ax_main.set_yticklabels([])  # clear default labels
ax_main.tick_params(left=False)  # Remove left ticks

# Draw a box *only* around the heatmap
# Move the left spine to x=-0.5 instead of xmin_limit
ax_main.spines['left'].set_visible(True)
ax_main.spines['left'].set_position(('data', -0.5))
ax_main.spines['right'].set_visible(False)

# Label removed from the top (will be moved below)

# Draw inline concentration bars directly on/beside the y-axis
# We use ax_main's coordinate system via transform tricks
max_conc = max(conc_sorted)
bar_width_data = 12.0  # halved to save horizontal space

for i, (genre, conc) in enumerate(zip(genres_sorted, conc_sorted)):
    # Inline concentration bar (drawn in data coordinates, left of column 0)
    bar_len = (conc / max_conc) * bar_width_data
    bar_color = purity_color(conc)
    ax_main.barh(i, -bar_len, left=-0.5, height=0.75,
                color=bar_color, edgecolor='none', alpha=0.35,
                clip_on=False, zorder=1)

    # Genre name label, placed inside the bar, right-aligned to the y-axis
    ax_main.text(-0.8, i, genre, ha='right', va='center',
                fontsize=16, fontweight='demibold', zorder=5)

# Adjust x-limits so the bars fit perfectly
xmax_limit = n_clusters - 0.5
xmin_limit = -0.5 - bar_width_data - 0.5
ax_main.set_xlim(xmin_limit, xmax_limit)

# Restrict top and bottom spines to only cover the heatmap width
ax_main.spines['top'].set_bounds(-0.5, xmax_limit)
ax_main.spines['bottom'].set_bounds(-0.5, xmax_limit)

# Add concentration axis below the bars
import math
max_c = max(conc_sorted)
tick_step = 20 if max_c > 60 else 10
conc_ticks = np.arange(0, math.ceil(max_c/tick_step)*tick_step + 1, tick_step)
y_line = n_genres - 0.5
x_start = -0.5
x_max_drawn = -0.5 - (conc_ticks[-1] / max_c) * bar_width_data
ax_main.plot([x_start, x_max_drawn], [y_line, y_line], color='black', lw=1.0, clip_on=False)
for t in conc_ticks:
    x_t = -0.5 - (t / max_c) * bar_width_data
    # Draw ticks only going ABOVE the axis line (y_line - 0.4 since y is inverted)
    ax_main.plot([x_t, x_t], [y_line, y_line - 0.4], color='black', lw=1.2, clip_on=False)
    
    # The text was hidden behind ax_bot. We render it on ax_bot instead!
    # Shift '0' slightly to the left so it is not bisected by the left spine of ax_bot at x=-0.5
    text_x = x_t - 0.4 if t == 0 else x_t
    ax_bot.text(text_x, 0.98, f"{int(t)}", ha='center', va='top', fontsize=12, 
                transform=ax_bot.get_xaxis_transform(), clip_on=False)

# Genre Concentration Label below the axis (drawn on ax_bot to prevent occlusion)
center_x_conc = (x_start + x_max_drawn) / 2.0

# PREVENT EMPTY ROWS: Lock the y-limits of the main axes so adding bottom plots won't stretch the box!
ax_main.set_ylim(n_genres - 0.5, -0.5)

# X-axis cluster labels (labels moved to bottom axis to avoid overlap)
ax_main.set_xticks(range(n_clusters))
ax_main.set_xticklabels([])

# ═══════════════════════════════════════════════════════════
# TOP: Cluster Purity bars
# ═══════════════════════════════════════════════════════════
bar_colors_top = [purity_color(p) for p in purity_sorted]
ax_top.bar(range(n_clusters), purity_sorted, color=bar_colors_top,
           edgecolor='white', linewidth=0.5, width=0.85)

ax_top.set_xlim(xmin_limit, xmax_limit)
ax_top.set_ylim(0, 105)
ax_top.yaxis.tick_left()
ax_top.yaxis.set_label_position("left")
ax_top.set_ylabel('Cluster\nPurity', fontsize=18, fontweight='bold', rotation=90, labelpad=10)
ax_top.set_xticks([])
ax_top.spines['bottom'].set_visible(False)
ax_top.spines['right'].set_visible(False)
ax_top.spines['top'].set_visible(False)
ax_top.spines['left'].set_visible(True)
ax_top.spines['left'].set_position(('data', -0.5))
ax_top.tick_params(axis='y', labelsize=12)

# ═══════════════════════════════════════════════════════════
# BOTTOM: Cluster size bars
# ═══════════════════════════════════════════════════════════
ax_bot.bar(range(n_clusters), size_sorted, color='#5b9bd5',
           edgecolor='white', linewidth=0.5, width=0.85, alpha=0.8)

ax_bot.set_xlim(xmin_limit, xmax_limit)
# Setup Cluster Size axis (ax_bot)
max_size = size_sorted.max()
# Increase ylim to create vertical separation between the 200 tick and Genre Concentration's 0 (which is at y=0.95 ax_bot)
ax_bot.set_ylim(0, 250)

ax_bot.yaxis.tick_left()
ax_bot.yaxis.set_label_position("left")
ax_bot.set_ylabel("Cluster\nSize", fontsize=18, fontweight='bold', labelpad=15, rotation=90)

# Explicitly add 0 and 200 without overlapping by using the padded ylim
ax_bot.set_yticks([0, 200])
ax_bot.set_yticklabels(["0", "200"], fontsize=12)

# X-axis cluster labels (moved here)
ax_bot.set_xticks(range(n_clusters))
ax_bot.set_xticklabels([f"C{c}" for c in clusters_sorted], rotation=45, ha='right', fontsize=12, rotation_mode='anchor')
# Center the x-axis label under the bars (not the whole figure which includes the left margin)
bar_center_x = (n_clusters - 1) / 2.0
ax_bot.text(bar_center_x, -0.4, 'Acoustic Cluster (sorted by purity)',
            fontsize=24, fontweight='bold', ha='center', va='top',
            transform=ax_bot.get_xaxis_transform(), clip_on=False)

# Add the Genre Concentration label underneath# Genre Concentration Label below the axis (drawn on ax_bot to prevent occlusion)
# The vertical tick values (60, 50...) were also drawn on ax_bot earlier

x_pos_conc = center_x_conc - 3.5  # Push further left to avoid Cluster Size vertical label
ax_bot.text(x_pos_conc, 0.50, "Genre\nConcentration", ha='center', va='center',
            fontsize=18, fontweight='bold', transform=ax_bot.get_xaxis_transform(), clip_on=False)

ax_bot.spines['top'].set_visible(False)
ax_bot.spines['bottom'].set_visible(False)
ax_bot.spines['left'].set_visible(True)
ax_bot.spines['right'].set_visible(False)

# Adjust the right spine position to match the right edge of the plot securely
ax_bot.spines['left'].set_position(('data', -0.5))

# Draw baseline for the bars
ax_bot.plot([-0.5, xmax_limit - 0.5], [0, 0], color='black', lw=1.2)

# ═══════════════════════════════════════════════════════════
# Summary stats (small, bottom — no title)
# ═══════════════════════════════════════════════════════════
avg_purity = purity_sorted.mean()
avg_conc   = conc_sorted.mean()
pure_clusters = (purity_sorted >= 50).sum()

plt.savefig(OUTFILE, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {OUTFILE}")
print(f"  Avg purity: {avg_purity:.1f}%, Avg concentration: {avg_conc:.1f}%")
print(f"  Pure clusters (≥50%): {pure_clusters}/{n_clusters}")
plt.close()
