"""
Figure 2 – Feature Importance
Panel (a): Stacked horizontal bar chart of the top-25 features (normalised
           so the total bar length equals the ensemble score).
Panel (b): Horizontal box-plots with strip overlay showing ensemble-score
           distribution per feature category.

Saves to paper/SMC/figs/paper/fig2_feature_importance.png
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Patch

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
})

# ──────────────────────────────────────────────────────────────
# Human-readable feature name mapping (comprehensive)
# ──────────────────────────────────────────────────────────────
READABLE_NAMES = {
    # ── Standard MIR: Spectral ──
    '1-ZCRm':                  'Zero-Crossing Rate',
    '2-Energym':               'Energy (Mean)',
    '2-Energym_log':           'Energy (log)',
    '3-EnergyEntropym':        'Energy Entropy',
    '4-SpectralCentroidm':     'Spectral Centroid',
    '5-SpectralSpreadm':       'Spectral Spread',
    '6-SpectralEntropym':      'Spectral Entropy',
    '7-SpectralFluxm':         'Spectral Flux',
    '8-SpectralRolloffm':      'Spectral Roll-off',
    '34-ChromaDeviationm':     'Chroma Deviation',
    '35-ZCRstd':               'Zero-Crossing Rate (σ)',
    '36-Energystd':            'Energy (σ)',
    '37-EnergyEntropystd':     'Energy Entropy (σ)',
    '38-SpectralCentroidstd':  'Spectral Centroid (σ)',
    '38-SpectralCentroidstd_log': 'Spectral Centroid σ (log)',
    '39-SpectralSpreadstd':    'Spectral Spread (σ)',
    '40-SpectralEntropystd':   'Spectral Entropy (σ)',
    '40-SpectralEntropystd_log': 'Spectral Entropy σ (log)',
    '41-SpectralFluxstd':      'Spectral Flux (σ)',
    # ── Standard MIR: MFCCs ──
    '9-MFCCs1m':               'MFCC 1 (Mean)',
    '10-MFCCs2m':              'MFCC 2 (Mean)',
    '11-MFCCs3m':              'MFCC 3 (Mean)',
    '18-MFCCs10m':             'MFCC 10 (Mean)',
    '43-MFCCs1std':            'MFCC 1 (σ)',
    '44-MFCCs2std':            'MFCC 2 (σ)',
    '44-MFCCs2std_log':        'MFCC 2 σ (log)',
    '47-MFCCs5std':            'MFCC 5 (σ)',
    '49-MFCCs7std':            'MFCC 7 (σ)',
    # ── Standard MIR: Chroma ──
    '25-ChromaVector4m':       'Chroma Vector 4',
    '32-ChromaVector11m':      'Chroma Vector 11',
    '57-ChromaVector2std':     'Chroma Vector 2 (σ)',
    '60-ChromaVector5std':     'Chroma Vector 5 (σ)',
    '63-ChromaVector8std':     'Chroma Vector 8 (σ)',
    '64-ChromaVector9std':     'Chroma Vector 9 (σ)',
    # ── Standard MIR: Basic rhythm / onset ──
    '69-BPM':                  'Tempo (Librosa)',
    '69-BPM_log':              'Tempo (Librosa, log)',
    '69-BPM_x_3-EnergyEntropym': 'Tempo × Energy Entropy',
    '69-BPM_x_2-Energym':     'Tempo × Energy',
    '70-BPMconf':              'Tempo Confidence',
    '70-BPMconf_log':          'Tempo Confidence (log)',
    '80-onset_rate':           'Onset Rate',
    # ── Standard MIR: Beat-level loudness ──
    '78-beats_loudness.mean':  'Beat Loudness (Mean)',
    '79-beats_loudness.stdev': 'Beat Loudness (σ)',
    '81-beats_loudness_band_ratio.mean1': 'Beat Loudness Band 1',
    '85-beats_loudness_band_ratio.mean5': 'Beat Loudness Band 5',
    '86-beats_loudness_band_ratio.mean6': 'Beat Loudness Band 6',
    '87-beats_loudness_band_ratio.stdev1': 'Beat Loudness Band 1 (σ)',
    '91-beats_loudness_band_ratio.stdev5': 'Beat Loudness Band 5 (σ)',
    # ── Standard MIR: Spectral contrast / dynamics ──
    'spectral_contrast_band4_mean': 'Spectral Contrast Band 4',
    'spectral_contrast_band5_mean': 'Spectral Contrast Band 5',
    'spectral_contrast_band6_mean': 'Spectral Contrast Band 6',
    'spectral_contrast_band7_mean': 'Spectral Contrast Band 7',
    'spectral_contrast_band7_std':  'Spectral Contrast Band 7 (σ)',
    'spectral_flatness_std':   'Spectral Flatness (σ)',
    'rms_std':                 'RMS Energy (σ)',
    'rms_skew':                'RMS Energy (Skew)',
    'dynamic_range_db':        'Dynamic Range (dB)',

    # ── EDM Production & Texture ──
    'sidechain_mod_depth':     'Sidechain Mod Depth',
    'sidechain_mod_rate':      'Sidechain Mod Rate',
    'sub_bass_ratio':          'Sub-Bass Ratio',
    'kick_prominence':         'Kick Prominence',
    'energy_sub_ratio':        'Energy Sub-Band Ratio',
    'energy_ratio_low':        'Energy Low-Band Ratio',
    'energy_ratio_high':       'Energy High-Band Ratio',
    'spectral_mod_rate':       'Spectral Modulation Rate',
    'spectral_mod_strength':   'Spectral Modulation Strength',
    'groove_pump_ratio':       'Groove Pump Ratio',

    # ── Tempogram: Fourier ──
    'fourier_peak1_mean':      'Fourier Tempo Peak 1',
    'fourier_peak1_bpm':       'Fourier Peak 1 BPM',
    'fourier_peak1_std':       'Fourier Peak 1 (σ)',
    'fourier_peak2_mean':      'Fourier Tempo Peak 2',
    'fourier_peak2_mean_log':  'Fourier Peak 2 (log)',
    'fourier_peak2_bpm':       'Fourier Peak 2 BPM',
    'fourier_peak2_bpm_log':   'Fourier Peak 2 BPM (log)',
    'fourier_peak2_std':       'Fourier Peak 2 (σ)',
    'fourier_peak3_bpm':       'Fourier Peak 3 BPM',
    'fourier_peak3_mean':      'Fourier Peak 3 Strength',
    'fourier_peak3_mean_log':  'Fourier Peak 3 (log)',
    'fourier_peak3_std':       'Fourier Peak 3 (σ)',
    'fourier_peak4_bpm':       'Fourier Peak 4 BPM',
    'fourier_peak4_mean_log':  'Fourier Peak 4 (log)',
    'fourier_peak4_std':       'Fourier Peak 4 (σ)',
    'fourier_peak5_bpm':       'Fourier Peak 5 BPM',
    'fourier_peak5_mean_log':  'Fourier Peak 5 (log)',
    'fourier_peak5_std':       'Fourier Peak 5 (σ)',

    # ── Tempogram: Autocorrelation ──
    'auto_peak1_bpm':          'Autocorr. Peak 1 BPM',
    'auto_peak1_bpm_log':      'Autocorr. Peak 1 BPM (log)',
    'auto_peak1_mean':         'Autocorr. Peak 1 Strength',
    'auto_peak2_bpm':          'Autocorr. Peak 2 BPM',
    'auto_peak2_mean':         'Autocorr. Peak 2 Strength',
    'auto_peak3_bpm':          'Autocorr. Peak 3 BPM',
    'auto_peak3_mean':         'Autocorr. Peak 3 Strength',
    'auto_peak4_bpm':          'Autocorr. Peak 4 BPM',
    'auto_peak4_std':          'Autocorr. Peak 4 (σ)',
    'auto_peak5_bpm':          'Autocorr. Peak 5 BPM',

    # ── Tempogram: Cyclic Fourier ──
    'cyclic_fourier_peak1_s':      'Cyclic Fourier Peak 1',
    'cyclic_fourier_peak1_s_log':  'Cyclic Fourier Peak 1 (log)',
    'cyclic_fourier_peak1_mean':   'Cyclic Fourier Peak 1 Strength',
    'cyclic_fourier_peak1_std':    'Cyclic Fourier Peak 1 (σ)',
    'cyclic_fourier_peak2_s':      'Cyclic Fourier Peak 2',
    'cyclic_fourier_peak2_mean':   'Cyclic Fourier Peak 2 Strength',
    'cyclic_fourier_peak2_mean_log': 'Cyclic Fourier Peak 2 (log)',
    'cyclic_fourier_peak3_s':      'Cyclic Fourier Peak 3',
    'cyclic_fourier_peak3_std':    'Cyclic Fourier Peak 3 (σ)',
    'cyclic_fourier_peak4_s':      'Cyclic Fourier Peak 4',
    'cyclic_fourier_peak4_mean':   'Cyclic Fourier Peak 4 Strength',
    'cyclic_fourier_peak4_std':    'Cyclic Fourier Peak 4 (σ)',
    'cyclic_fourier_peak5_std':    'Cyclic Fourier Peak 5 (σ)',
    'cyclic_fourier_entropy':      'Cyclic Fourier Entropy',

    # ── Tempogram: Cyclic Autocorrelation ──
    'cyclic_auto_peak1_s':         'Cyclic Auto Peak 1',
    'cyclic_auto_peak1_mean':      'Cyclic Auto Peak 1 Strength',
    'cyclic_auto_peak2_s':         'Cyclic Auto Peak 2',
    'cyclic_auto_peak2_mean':      'Cyclic Auto Peak 2 Strength',
    'cyclic_auto_peak2_std':       'Cyclic Auto Peak 2 (σ)',
    'cyclic_auto_peak3_s':         'Cyclic Auto Peak 3',
    'cyclic_auto_peak3_mean':      'Cyclic Auto Peak 3 Strength',
    'cyclic_auto_peak4_s':         'Cyclic Auto Peak 4',
    'cyclic_auto_peak5_s':         'Cyclic Auto Peak 5',
    'cyclic_auto_peak5_mean':      'Cyclic Auto Peak 5 Strength',
    'cyclic_auto_entropy':         'Cyclic Auto Entropy',

    # ── Tempogram: Histogram / summary ──
    '71-bpm':                      'Tempo (Autocorrelation)',
    '72-bpm_histogram_first_peak_bpm':  'Tempo Histogram Peak',
    '73-bpm_histogram_first_peak_weight': 'Tempo Hist. Peak Weight',
    '74-bpm_histogram_second_peak_bpm':  'Tempo Hist. 2nd Peak',
    '75-bpm_histogram_second_peak_spread': 'Tempo Hist. 2nd Spread',
    '76-bpm_histogram_second_peak_weight': 'Tempo Hist. 2nd Weight',
    'tempo_features_mean':         'Tempo Features (Mean)',

    # ── Metadata ──
    'meta.Bpm':                    'BPM (Metadata)',
    'meta.Bpm_log':                'BPM (Metadata, log)',
    'meta.Length_ms':              'Track Length',
}


# ──────────────────────────────────────────────────────────────
# Feature category classification – matches paper Section 3.2
# ──────────────────────────────────────────────────────────────
EDM_PRODUCTION = {
    'sidechain_mod_depth', 'sidechain_mod_rate', 'sub_bass_ratio',
    'kick_prominence', 'energy_sub_ratio', 'energy_ratio_low',
    'energy_ratio_high', 'spectral_mod_rate', 'spectral_mod_strength',
    'groove_pump_ratio',
}

TEMPOGRAM_PREFIXES = (
    'fourier_peak', 'auto_peak',
    'cyclic_fourier_', 'cyclic_auto_',
    'tempo_features',
    '71-', '72-', '73-', '74-', '75-', '76-',
)


def classify_category(name):
    """Classify feature into one of the paper's 3 categories."""
    if name in EDM_PRODUCTION:
        return 'EDM Production & Texture'
    if any(name.startswith(p) for p in TEMPOGRAM_PREFIXES):
        return 'Tempogram-Based'
    if name.startswith('meta.'):
        return 'Metadata'
    return 'Conventional MIR'


def readable(name):
    """Get human-readable name, with fallback cleanup."""
    if name in READABLE_NAMES:
        return READABLE_NAMES[name]
    clean = name
    if clean[0].isdigit() and '-' in clean:
        clean = clean.split('-', 1)[1]
    return clean.replace('_', ' ').title()


# ──────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────
df = pd.read_csv('results/csv/feature_importance_scores.csv')
df['category'] = df['feature'].apply(classify_category)
df['display_name'] = df['feature'].apply(readable)

score_cols = ['anova', 'mutual_info', 'random_forest', 'extra_trees',
              'variance', 'cluster_separation']
labels = ['ANOVA', 'MI', 'RF', 'ET', 'VARIANCE', 'SEPARATION']
colors = ['#e07a5f', '#81b29a', '#6fa8dc', '#b6d7a8', '#f4c542', '#c4a8d4']

# Top 25 by total stacked importance
df['total_importance'] = df[score_cols].sum(axis=1)
top = df.sort_values('total_importance', ascending=False).head(25)
top = top.iloc[::-1]

# Category colours
cat_colors = {
    'Conventional MIR':        '#2b5ea7',
    'EDM Production & Texture': '#c44e2e',
    'Tempogram-Based':         '#1a6b3c',
    'Metadata':                '#7a7a7a',
}

# ──────────────────────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────────────────────
fig, (ax_bar, ax_box) = plt.subplots(
    2, 1, figsize=(10, 12),
    gridspec_kw={'height_ratios': [4, 1.1], 'hspace': 0.28}
)

# ── Panel (a): Stacked importance bars ──────────────────────
y = np.arange(len(top))
left = np.zeros(len(top))

n_methods = len(score_cols)
for col, label, color in zip(score_cols, labels, colors):
    vals = top[col].values / n_methods
    ax_bar.barh(y, vals, left=left, color=color, edgecolor='white',
                linewidth=0.3, label=label)
    left += vals

ax_bar.set_yticks(y)
ylabels = top['display_name'].values
ycolors = [cat_colors.get(c, '#333') for c in top['category'].values]
ax_bar.set_yticklabels(ylabels, fontsize=21, fontweight='demibold')
for tick_label, color in zip(ax_bar.get_yticklabels(), ycolors):
    tick_label.set_color(color)

ax_bar.set_xlabel('Ensemble Score', fontsize=22, fontweight='bold')
ax_bar.set_xlim(0, 0.6)
ax_bar.spines['top'].set_visible(False)
ax_bar.spines['right'].set_visible(False)
ax_bar.tick_params(axis='x', labelsize=18)
ax_bar.grid(axis='x', alpha=0.3)
ax_bar.set_title('(a) Top 25 Important Features', fontsize=28, fontweight='bold',
                 loc='left', pad=12)

# Category legend patches
cat_patches = [
    Patch(facecolor=cat_colors['Conventional MIR'], label='Conventional MIR'),
    Patch(facecolor=cat_colors['EDM Production & Texture'], label='EDM Production & Texture'),
    Patch(facecolor=cat_colors['Tempogram-Based'], label='Tempogram-Based'),
]
# Scoring method legend
ax_bar.legend(fontsize=13, loc='lower right')

# Feature Category legend — pinned to far-left edge using figure coords
cat_legend = fig.legend(handles=cat_patches, fontsize=15.5,
                        loc='upper left', bbox_to_anchor=(-0.10, 0.35),
                        ncol=1, framealpha=0.95, edgecolor='#ccc',
                        title='Feature Category', title_fontsize=15.5)
cat_legend.get_title().set_fontweight('bold')
ax_bar.legend(fontsize=13, loc='lower right')

# ── Panel (b): Box plot + strip ─────────────────────────────
df_compare = df[df['category'] != 'Metadata'].copy()
cat_order = ['Tempogram-Based', 'EDM Production & Texture', 'Conventional MIR']

# Build data lists
box_data = []
for cat in cat_order:
    box_data.append(
        df_compare[df_compare['category'] == cat]['ensemble_score'].values
    )

# Draw box plots (horizontal) — hide fliers since we overlay strip dots
bp = ax_box.boxplot(box_data, positions=range(len(cat_order)), vert=False,
                    patch_artist=True, widths=0.55, showfliers=False,
                    boxprops=dict(linewidth=1.2),
                    whiskerprops=dict(linewidth=1.2, color='#555'),
                    capprops=dict(linewidth=1.2, color='#555'),
                    medianprops=dict(linewidth=2.0, color='#222'))

# Colour each box by category
for patch, cat in zip(bp['boxes'], cat_order):
    patch.set_facecolor(cat_colors[cat])
    patch.set_alpha(0.5)
    patch.set_edgecolor(cat_colors[cat])

# Overlay individual feature dots (strip plot)
np.random.seed(42)
for i, cat in enumerate(cat_order):
    vals = df_compare[df_compare['category'] == cat]['ensemble_score'].values
    jitter = np.random.normal(0, 0.08, size=len(vals))
    ax_box.scatter(vals, i + jitter, color=cat_colors[cat],
                   alpha=0.65, s=28, edgecolor='white', linewidth=0.4, zorder=5)

ax_box.set_yticks(range(len(cat_order)))
ax_box.set_yticklabels(cat_order, fontsize=20, fontweight='bold')
for tick_label in ax_box.get_yticklabels():
    tick_label.set_color(cat_colors.get(tick_label.get_text(), '#333'))

ax_box.set_xlabel('Ensemble Score', fontsize=22, fontweight='bold')
ax_box.spines['top'].set_visible(False)
ax_box.spines['right'].set_visible(False)
ax_box.tick_params(axis='x', labelsize=18)
ax_box.grid(axis='x', alpha=0.3)
ax_box.set_title('(b) Importance by Feature Category', fontsize=26,
                 fontweight='bold', loc='left', pad=12)
ax_box.set_ylim(-0.45, 2.45)
ax_box.set_xlim(0, 0.6)

fig.subplots_adjust(left=0.25, right=0.95, top=0.96, bottom=0.06)

# Align both axes so y-axes are vertically consistent
fig.align_ylabels([ax_bar, ax_box])

os.makedirs('paper/SMC/figs/paper', exist_ok=True)
outpath = 'paper/SMC/figs/paper/fig2_feature_importance.png'
plt.savefig(outpath, dpi=300, bbox_inches='tight')
print(f"Saved: {outpath}")
