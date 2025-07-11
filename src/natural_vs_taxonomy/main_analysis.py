"""
main_analysis.py
================
Main script to run complete EDM genre analysis with modular components
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# Import all modules
from feature_engineering import EDMFeatureEngineer
from feature_selection import EDMFeatureSelector
from clustering_algorithms import EDMClusterer
from experiments import NaturalClusterExperiment, GenreConvergenceExperiment
from visualization import (
    plot_validation_metrics, plot_tsne_comparison,
    plot_genre_convergence_heatmap
)
from utils import ensure_directories, get_output_paths, save_dataframe_with_info, create_analysis_summary


def load_data(filepath='dataset/top100_with_tempogram_nmf_meta.csv'):
    """Load the EDM dataset."""
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    df = pd.read_csv(filepath)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[numeric_cols]
    y = df['genre']
    
    print(f"Dataset: {len(df)} songs")
    print(f"Features: {len(numeric_cols)}")
    print(f"Genres: {len(y.unique())}")
    
    return df, X, y


def run_complete_analysis():
    """Run the complete EDM genre analysis pipeline."""
    print("\n" + "="*70)
    print("EDM GENRE ANALYSIS - COMPLETE PIPELINE")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Ensure output directories exist
    ensure_directories()
    
    # 1. Load data
    df, X, y = load_data()
    
    # 2. Feature Engineering
    engineer = EDMFeatureEngineer(random_state=42)
    X_engineered = engineer.fit_transform(X)
    
    # 3. Feature Selection
    selector = EDMFeatureSelector(n_features=100, random_state=42)
    X_selected = selector.fit_select(X_engineered, y, n_clusters=35)
    
    # 4. Clustering
    clusterer = EDMClusterer(random_state=42)
    
    # Forced 35 clusters
    print("\n" + "="*60)
    print("FORCED CLUSTERING (35 CLUSTERS)")
    print("="*60)
    forced_labels = clusterer.fit_forced_clusters(X_selected, n_clusters=35)
    forced_metrics = clusterer.evaluate_clustering(X_selected, forced_labels, y)
    
    print("\nForced clustering metrics:")
    for metric, value in forced_metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    # Natural clusters
    print("\n" + "="*60)
    print("NATURAL CLUSTERING")
    print("="*60)
    natural_labels, natural_k = clusterer.find_natural_clusters(X_selected, min_k=5, max_k=35)
    natural_metrics = clusterer.evaluate_clustering(X_selected, natural_labels, y)
    
    print("\nNatural clustering metrics:")
    for metric, value in natural_metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    # 5. Experiments
    # Experiment 1: Natural Cluster Discovery
    exp1 = NaturalClusterExperiment()
    exp1_results = exp1.run(X_selected.values, y.values, forced_labels, 
                           natural_labels, natural_k)
    
    # Experiment 2: Genre Convergence
    exp2 = GenreConvergenceExperiment()
    exp2_results = exp2.run(X_selected.values, y.values, forced_labels)
    
    # 6. Visualizations
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    # Plot validation metrics with genre-based dendrogram
    if clusterer.natural_clusterer:
        plot_validation_metrics(clusterer.natural_clusterer.results, 
                              features_scaled=X_selected.values,
                              y_true=y.values)
    
    # Plot t-SNE comparison
    plot_tsne_comparison(X_selected.values, natural_labels, forced_labels, y.values)
    
    # Plot genre convergence
    plot_genre_convergence_heatmap(exp2_results['convergence_matrix'])
    
    # 7. Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    # Get organized output paths
    paths = get_output_paths()
    
    # Save all results
    all_results = {
        'data_info': {
            'n_songs': len(df),
            'n_genres': len(y.unique()),
            'n_features_original': X.shape[1],
            'n_features_engineered': X_engineered.shape[1],
            'n_features_selected': X_selected.shape[1]
        },
        'feature_engineering': engineer.get_feature_info(),
        'feature_selection': selector.get_selection_info(),
        'clustering': {
            'forced_labels': forced_labels,
            'forced_metrics': forced_metrics,
            'natural_labels': natural_labels,
            'natural_k': natural_k,
            'natural_metrics': natural_metrics,
            'natural_results': clusterer.natural_clusterer.results
        },
        'experiments': {
            'exp1_natural_discovery': exp1_results,
            'exp2_genre_convergence': exp2_results
        }
    }
    
    # Save pickle files
    with open(paths['complete_results'], 'wb') as f:
        pickle.dump(all_results, f)
    print(f"Saved: {paths['complete_results']}")
    
    # Save clustering summary
    summary_df = pd.DataFrame({
        'song': df['song'],
        'artist': df['artist'] if 'artist' in df.columns else 'Unknown',
        'true_genre': y,
        'forced_cluster_35': forced_labels,
        'natural_cluster': natural_labels
    })
    save_dataframe_with_info(summary_df, paths['clustering_summary'], 
                            "Complete clustering assignments for all songs")
    
    # Save convergence matrix
    exp2_results['convergence_matrix'].to_csv(paths['convergence_matrix'])
    print(f"Saved: {paths['convergence_matrix']}")
    
    # Save top converging pairs
    save_dataframe_with_info(exp2_results['convergence_pairs'].head(50), 
                            paths['converging_pairs'],
                            "Top 50 most acoustically similar genre pairs")
    
    # Save phantom distinctions
    if exp2_results['phantom_distinctions'] is not None:
        save_dataframe_with_info(exp2_results['phantom_distinctions'], 
                                paths['phantom_distinctions'],
                                "Genres that sound similar but are marketed differently")
    
    # Save genre fragmentation
    if 'genre_entropy' in exp2_results:
        save_dataframe_with_info(exp2_results['genre_entropy'], 
                                paths['genre_fragmentation'],
                                "How fragmented each genre is across clusters")
    
    # Save feature importance scores
    if hasattr(selector, 'feature_scores'):
        save_dataframe_with_info(selector.feature_scores.head(100), 
                                paths['feature_scores'],
                                "Top 100 features by importance score")
    
    # 8. Final Report
    print_final_report(all_results)
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return all_results


def print_final_report(results):
    """Print final analysis report."""
    print("\n" + "="*70)
    print("FINAL REPORT")
    print("="*70)
    
    # Get paths for saving
    paths = get_output_paths()
    
    data_info = results['data_info']
    clustering = results['clustering']
    exp1 = results['experiments']['exp1_natural_discovery']
    exp2 = results['experiments']['exp2_genre_convergence']
    
    natural_k = clustering['natural_k']
    forced_k = 35
    
    report = f"""
EDM GENRE ANALYSIS - FINAL REPORT
{'='*60}

DATA SUMMARY:
- Songs analyzed: {data_info['n_songs']}
- Original genres: {data_info['n_genres']}
- Original features: {data_info['n_features_original']}
- Engineered features: {data_info['n_features_engineered']}
- Selected features: {data_info['n_features_selected']}

KEY FINDINGS:
- Natural acoustic families: {natural_k}
- Industry genre categories: {forced_k}
- Over-segmentation: {forced_k - natural_k} excess categories

CLUSTERING QUALITY:
- Natural vs Genres NMI: {exp1['metrics']['natural_vs_genres_nmi']:.3f}
- Forced vs Genres NMI: {exp1['metrics']['forced_vs_genres_nmi']:.3f}
- Natural vs Forced NMI: {exp1['metrics']['natural_vs_forced_nmi']:.3f}

INTERPRETATION:
The EDM industry's {forced_k}-genre taxonomy appears to be over-segmented
by approximately {forced_k - natural_k} categories. The data naturally forms {natural_k}
distinct acoustic families, suggesting significant marketing-driven
categorization beyond actual musical differences.

RECOMMENDATIONS:
1. Consolidate to {natural_k} core acoustic families
2. Use marketing variations as sub-categories
3. Merge acoustically identical genres identified in convergence analysis
4. Base future genre decisions on acoustic similarity metrics

OUTPUT FILES:
results/
├── csv/
│   ├── edm_clustering_summary.csv : All cluster assignments
│   ├── genre_convergence_matrix.csv : Genre similarity matrix
│   ├── top_converging_pairs.csv : Most similar genre pairs
│   ├── phantom_distinctions.csv : Marketing vs acoustic
│   ├── genre_fragmentation.csv : Genre cluster distribution
│   └── feature_importance_scores.csv : Selected features
├── pkl/
│   ├── edm_complete_analysis_results.pkl : Complete results
│   └── genre_merge_analysis.pkl : Hierarchical merge analysis
└── reports/
    └── edm_analysis_report.txt : This report

plots/
├── validation_metrics_combined.png : Cluster validation with genre dendrogram
├── tsne_comparison_interpretable.png : Clustering comparison
└── genre_convergence_heatmap.png : Genre similarity matrix

{'='*60}
    """
    
    print(report)
    
    # Save text report
    with open(paths['text_report'], 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {paths['text_report']}")
    
    # Also save timestamped version
    with open(paths['timestamped_report'], 'w') as f:
        f.write(report)
    print(f"Timestamped copy: {paths['timestamped_report']}")


if __name__ == "__main__":
    # Check if we're in the right directory
    if not os.path.exists('dataset/top100_with_tempogram_nmf_meta.csv'):
        print("ERROR: Cannot find dataset file.")
        print("Please run from project root directory.")
        print("\nExpected structure:")
        print("  project_root/")
        print("    ├── dataset/")
        print("    │   └── top100_with_tempogram_nmf_meta.csv")
        print("    ├── feature_engineering.py")
        print("    ├── feature_selection.py")
        print("    ├── clustering_algorithms.py")
        print("    ├── experiments.py")
        print("    ├── visualization.py")
        print("    └── main_analysis.py")
        sys.exit(1)
    
    # Run complete analysis
    results = run_complete_analysis()