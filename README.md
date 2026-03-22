# EDMPhantomGenre (SMC 2026)

Code repository for the analysis of "phantom genres" in Electronic Dance Music (EDM) --- genre labels that exist in industry taxonomies (e.g., Beatport) but lack distinct, separable acoustic characteristics when analyzed through computational audio features.

## Repository Structure

```
EDMPhantomGenre/
├── dataset/                     # Extracted feature matrices and metadata definitions
├── paper/SMC/                   # SMC 2026 LaTeX manuscript and compiled figures
├── src/                         # Source code (Python 3.10+)
│   ├── feature_extraction/      # Scripts to extract 194 features from Beatport audio
│   ├── feature_processing/      # Yeo-Johnson normalization, scaling, and greedy selection
│   ├── analysis/                # Main clustering pipeline (Divisive, K-means, metrics)
│   ├── experiments/             # Parameter sweeps and robustness/sensitivity analyses
│   └── visuals/                 # Reproduction scripts for all SMC manuscript figures
└── requirements.txt             # Dependency list
```

## Getting Started

### 1. Requirements
Install the required Python packages into your environment:
```bash
pip install -r requirements.txt
```

### 2. Feature Extraction
Raw audio files are not included in this repository due to copyright restrictions. To extract the exhaustive 194-feature set from your own local track collection:
```bash
python src/feature_extraction/extract_all.py --audio-dir /path/to/mp3s --output dataset/features.csv
```

### 3. Running the Analysis
To cleanly execute the core clustering pipeline, evaluating K-means and Divisive Hierarchical Clustering across internal, external, and distribution metrics:
```bash
python src/analysis/run_pipeline.py
```

### 4. Reproducing Manuscript Figures
All figures presented in the SMC submission can be identically reproduced using the tightly-coupled scripts in `src/visuals/`. The scripts will output high-resolution files directly to `paper/SMC/figs/paper/`.

```bash
# Main Text Figures
python src/visuals/plot_tsne_projection.py         # Figure 1: t-SNE Acoustic Space
python src/visuals/plot_feature_importance.py      # Figure 2: Feature Importance
python src/visuals/plot_confusion_matrix.py        # Figure 3: Genre-to-Cluster Confusion Matrix
python src/visuals/plot_musical_profile.py         # Figure 4: Radar Profiles (Crystallized vs. Hybrid)
python src/visuals/plot_dendrograms.py             # Figure 5: Hierarchical Co-clustering Dendrogram

# Appendix Figures
python src/visuals/plot_natural_k_validation.py    # Figure A.1: Validation Curves (k=10-30)
python src/visuals/plot_phantom_genre_profiles.py  # Figure A.3: Convergent Phantom Genre Pairs
```

## License

See [LICENSE](LICENSE) for details.