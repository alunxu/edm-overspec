# Acoustic Overspecification in Electronic Dance Music Taxonomy

Code repository for the paper *"Acoustic Overspecification in Electronic Dance Music Taxonomy,"* under review at SMC 2026. The supplementary material (appendix) is available as [`Supplementary Material.pdf`](Supplementary%20Material.pdf) in this repository.

## Overview

We extract 194 continuous, audio-derived features from 3,500 tracks across 35 Beatport genres, organized into three groups: conventional MIR descriptors (92 dims: spectral, timbral, harmonic, rhythmic), EDM-specific production features (38 dims: H/P ratio, sidechain proxy, sub-bass ratio, spectral contrast), and tempogram-based features (64 dims: Fourier, autocorrelation, and cyclic tempograms). To validate that findings reflect inherent acoustic structure rather than feature engineering artifacts, we additionally benchmark against pre-trained audio embeddings (MERT-95M and CLAP-tiny). Across all representations, unsupervised clustering consistently compresses the 35 commercial labels into 17–20 acoustically distinct families, exposing systematic overspecification in industry taxonomies.

## Repository Structure

```
EDMPhantomGenre/
├── Supplementary Material.pdf   # Appendix (feature taxonomy, validation curves, etc.)
├── dataset/                     # Extracted feature matrices and metadata
├── results/
│   ├── csv/                     # Clustering summaries, convergence matrices, fragmentation scores
│   ├── figs/                    # Pre-generated figures
│   └── reports/                 # Analysis reports
├── src/
│   ├── feature_extraction/      # 194-feature extraction from raw audio
│   ├── feature_processing/      # Yeo-Johnson normalisation, scaling, greedy selection
│   ├── analysis/                # Agglomerative and divisive clustering pipeline
│   ├── experiments/             # Sensitivity analyses (DJ Tools, feature count sweeps)
│   └── visuals/                 # Reproduction scripts for all manuscript figures
├── requirements.txt
└── LICENSE
```

## Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Feature Extraction
Raw audio is not included due to copyright. To extract the 194-feature set from your own tracks:
```bash
python src/feature_extraction/extract_all.py --audio-dir /path/to/mp3s --output dataset/features.csv
```

### 3. Run the Analysis Pipeline
```bash
python src/analysis/run_pipeline.py
```

### 4. Reproduce Manuscript Figures
```bash
python src/visuals/plot_tsne_projection.py         # Fig. 1: t-SNE projection
python src/visuals/plot_feature_importance.py       # Fig. 2: Feature importance
python src/visuals/plot_confusion_matrix.py         # Fig. 3: Genre-to-cluster confusion
python src/visuals/plot_musical_profile.py          # Fig. 4: Musical radar profiles
python src/visuals/plot_dendrograms.py              # Fig. 5: Hierarchical dendrogram

python src/visuals/plot_natural_k_validation.py     # Appendix Fig. 1: Validation curves
python src/experiments/sensitivity_dj_tools.py      # Appendix Fig. 2: DJ Tools sensitivity
python src/visuals/plot_phantom_genre_profiles.py   # Appendix Fig. 3: Convergent genre pairs
```

## Supplementary Material

The appendix is provided as [`Supplementary Material.pdf`](Supplementary%20Material.pdf) in this repository. It contains:
- **Appendix A**: Full taxonomy of the 194 acoustic features
- **Appendix B**: Cluster validation curves for natural-$k$ discovery
- **Appendix C**: DJ Tools sensitivity analysis
- **Appendix D**: Acoustically convergent genre pairs

## License

See [LICENSE](LICENSE) for details.