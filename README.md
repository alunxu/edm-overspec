# EDMPhantomGenre

Analysis of "phantom genres" in Electronic Dance Music (EDM) — genre labels that exist in industry taxonomies but lack distinct musical characteristics when analyzed through audio features.

## Repository Structure

```
EDMPhantomGenre/
├── dataset/                           # Raw audio dataset and metadata
│   ├── BeatportTop100-19_03_2025/     # Beatport Top-100 audio files by genre
│   └── top100_with_tempogram_nmf_meta.csv  # Extracted features + metadata
├── src/                               # Source code (analysis pipeline)
│   ├── common/                        # Shared utilities & feature pipeline
│   ├── natural_vs_taxonomy/           # Clustering: natural vs. industry genres
│   ├── phantom_genre/                 # Phantom genre detection & diagnostics
│   ├── hierarchical_genre_model/      # Hierarchical taxonomy visualizations
│   ├── deviation/                     # Centroid divergence & genre flow
│   └── timeline_analysis/            # Temporal evolution of genres
├── results/                           # Computed outputs (CSV, PKL, reports)
├── phantom_analysis_results/          # Phantom genre analysis outputs
├── ICASSP2026_unsupervised_edm/       # Paper submission (LaTeX)
├── EDM_overspec.tex                   # Main manuscript (LaTeX)
├── refs.bib                           # Bibliography
└── requirements.txt                   # Python dependencies
```

## Getting Started

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Analysis Pipeline

```bash
# 1. Natural vs. taxonomy clustering
cd src/natural_vs_taxonomy
python main_analysis.py

# 2. Phantom genre detection
cd ../phantom_genre
python phantom_genre_detector.py

# 3. Hierarchical taxonomy visualizations
cd ../hierarchical_genre_model
python hierarchical_edm_taxonomy.py
```

## Key Modules

| Module | Description |
|---|---|
| `natural_vs_taxonomy` | Compares data-driven clusters to industry genre labels using multi-criteria feature selection and ensemble clustering |
| `phantom_genre` | Detects genre pairs that are musically indistinguishable but taxonomically distinct |
| `hierarchical_genre_model` | Builds and visualizes a feature-based hierarchical taxonomy |
| `deviation` | Analyzes how natural cluster centroids diverge from genre label centroids |
| `timeline_analysis` | Explores temporal evolution of genre similarity |
| `common` | Shared feature engineering pipeline used across modules |

## License

See [LICENSE](LICENSE) for details.