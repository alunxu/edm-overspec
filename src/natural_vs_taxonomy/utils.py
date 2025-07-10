"""
utils.py
========
Utility functions for EDM analysis including file management
"""

import os
from datetime import datetime


def ensure_directories():
    """Ensure all output directories exist."""
    directories = [
        'results',
        'results/csv',
        'results/pkl',
        'results/reports',
        'plots'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
    
    return directories


def get_output_paths():
    """Get organized output paths for all files."""
    ensure_directories()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    paths = {
        # CSV files
        'clustering_summary': 'results/csv/edm_clustering_summary.csv',
        'convergence_matrix': 'results/csv/genre_convergence_matrix.csv',
        'converging_pairs': 'results/csv/top_converging_pairs.csv',
        'phantom_distinctions': 'results/csv/phantom_distinctions.csv',
        'genre_fragmentation': 'results/csv/genre_fragmentation.csv',
        'feature_scores': 'results/csv/feature_importance_scores.csv',
        
        # Pickle files
        'complete_results': 'results/pkl/edm_complete_analysis_results.pkl',
        'feature_engineering': 'results/pkl/feature_engineering_info.pkl',
        'clustering_results': 'results/pkl/clustering_results.pkl',
        
        # Reports
        'text_report': 'results/reports/edm_analysis_report.txt',
        'timestamped_report': f'results/reports/edm_analysis_report_{timestamp}.txt',
        
        # Plots
        'validation_metrics': 'plots/validation_metrics.png',
        'tsne_comparison': 'plots/tsne_comparison.png',
        'convergence_heatmap': 'plots/genre_convergence_heatmap.png',
        'cluster_comparison': 'plots/cluster_comparison.png',
        'summary_report': 'plots/summary_report.png',
        'dendrogram': 'plots/hierarchical_dendrogram.png',
        'merge_distances': 'plots/merge_distances.png',
        'method_comparison': 'plots/method_comparison.png'
    }
    
    return paths


def save_dataframe_with_info(df, filepath, description=""):
    """Save DataFrame with metadata."""
    df.to_csv(filepath, index=False)
    print(f"Saved: {filepath}")
    if description:
        print(f"  Description: {description}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}")


def create_analysis_summary(results):
    """Create a summary dictionary from results."""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'data': {
            'n_songs': results['data_info']['n_songs'],
            'n_genres': results['data_info']['n_genres'],
            'n_features_original': results['data_info']['n_features_original'],
            'n_features_selected': results['data_info']['n_features_selected']
        },
        'clustering': {
            'natural_clusters': results['clustering']['natural_k'],
            'forced_clusters': 35,
            'over_segmentation': 35 - results['clustering']['natural_k']
        },
        'quality_metrics': {
            'natural_silhouette': results['clustering']['natural_metrics'].get('silhouette', 'N/A'),
            'forced_silhouette': results['clustering']['forced_metrics'].get('silhouette', 'N/A'),
            'natural_vs_genres_nmi': results['experiments']['exp1_natural_discovery']['metrics']['natural_vs_genres_nmi'],
            'forced_vs_genres_nmi': results['experiments']['exp1_natural_discovery']['metrics']['forced_vs_genres_nmi']
        }
    }
    
    return summary