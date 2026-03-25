"""
run_pipeline.py
===============
Main script to run complete EDM genre analysis with modular components
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# Import all modules
from feature_processing.modules.feature_engineering import EDMFeatureEngineer
from feature_processing.modules.feature_selection import EDMFeatureSelector
from analysis.modules.clustering_algorithms import EDMClusterer
from analysis.modules.evaluation_metrics import NaturalClusterExperiment, GenreConvergenceExperiment
from analysis.modules.utils import ensure_directories, get_output_paths, save_dataframe_with_info, create_analysis_summary


def load_data(filepath='dataset/all_features_194.csv'):
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
    
    # 3. Correlation Deduplication
    X_dedup = EDMFeatureSelector.remove_correlated(X_engineered, threshold=0.95)
    
    # 4. Ensemble Feature Selection (top 20 by label-supervised ensemble score)
    from sklearn.preprocessing import StandardScaler
    selector = EDMFeatureSelector(n_features=20, random_state=42)
    X_selected = selector.fit_select(X_dedup, y, n_clusters=35)
    
    # Scale features for distance-based clustering
    print("\nScaling feature matrix for distance-based clustering...")
    scaler = StandardScaler()
    X_selected = pd.DataFrame(
        scaler.fit_transform(X_selected), 
        columns=X_selected.columns, 
        index=X_selected.index
    )
    
    # Save scaled feature matrix for visualization scripts
    print("Saving scaled feature matrix for future analysis...")
    feature_matrix_path = os.path.join('results', 'pkl', 'feature_matrix.pkl')
    with open(feature_matrix_path, 'wb') as f:
        pickle.dump({
            'X_selected': X_selected.values,
            'feature_names': X_selected.columns.tolist(),
            'genre_labels': y.values,
            'song_names': df['song'].values
        }, f)
    print(f"Saved feature matrix: {feature_matrix_path}")
    
    # 5. Clustering
    clusterer = EDMClusterer(random_state=42)
    
    # Forced 35 clusters — K-means
    print("\n" + "="*60)
    print("FORCED CLUSTERING (K-MEANS, k=35)")
    print("="*60)
    forced_labels = clusterer.fit_forced_clusters(X_selected, n_clusters=35, method='kmeans')
    forced_metrics = clusterer.evaluate_clustering(X_selected, forced_labels, y)
    
    print("\nK-means (k=35) metrics:")
    for metric, value in forced_metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    # Forced 35 clusters — Divisive (as described in manuscript)
    print("\n" + "="*60)
    print("FORCED CLUSTERING (DIVISIVE, k=35)")
    print("="*60)
    divisive_labels = clusterer.fit_forced_clusters(X_selected, n_clusters=35, method='divisive')
    divisive_metrics = clusterer.evaluate_clustering(X_selected, divisive_labels, y)
    
    print("\nDivisive (k=35) metrics:")
    for metric, value in divisive_metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    # Natural clusters
    print("\n" + "="*60)
    print("NATURAL CLUSTERING")
    print("="*60)
    natural_labels, natural_k = clusterer.find_natural_clusters(X_selected, min_k=15, max_k=30)
    natural_metrics = clusterer.evaluate_clustering(X_selected, natural_labels, y)
    
    print("\nNatural clustering metrics:")
    for metric, value in natural_metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    # 6. Experiments
    # Experiment 1: Natural Cluster Discovery
    exp1 = NaturalClusterExperiment()
    exp1_results = exp1.run(X_selected.values, y.values, forced_labels, 
                           natural_labels, natural_k)
    
    # Experiment 2: Genre Convergence
    exp2 = GenreConvergenceExperiment()
    exp2_results = exp2.run(X_selected.values, y.values, forced_labels)
    
    # Visualization is handled by separate scripts in src/visuals/
    
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
            'n_features_deduped': X_dedup.shape[1],
            'n_features_selected': X_selected.shape[1]
        },
        'feature_engineering': engineer.get_feature_info(),
        'feature_selection': selector.get_selection_info(),
        'feature_matrix': X_selected.values,
        'clustering': {
            'forced_labels': forced_labels,
            'forced_metrics': forced_metrics,
            'divisive_labels': divisive_labels,
            'divisive_metrics': divisive_metrics,
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
        'true_genre': y,
        'kmeans_cluster_35': forced_labels,
        'divisive_cluster_35': divisive_labels,
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
        save_dataframe_with_info(selector.feature_scores, 
                                paths['feature_scores'],
                                "All features by importance score")
    
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
├── genre_convergence_heatmap.png : Genre similarity matrix
├── cluster_genre_mapping.png : Natural clusters to genres mapping
└── cluster_genre_summary.png : Simplified cluster composition

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
    if not os.path.exists('dataset/all_features_194.csv'):
        print("ERROR: Cannot find dataset file.")
        print("Please run from project root directory.")
        print("Expected: dataset/all_features_194.csv")
        print("  project_root/")
        print("    ├── dataset/")
        print("    │   └── all_features_194.csv")
        print("    └── src/")
        print("        ├── feature_extraction/  (raw features)")
        print("        ├── feature_processing/  (scaling & selection)")
        print("        ├── analysis/")
        print("        │   ├── modules/         (core logic)")
        print("        │   └── run_pipeline.py  (execute here!)")
        print("        └── visuals/             (plotting scripts)")
        sys.exit(1)
    
    # Run complete analysis
    results = run_complete_analysis()