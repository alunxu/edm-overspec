"""
visualization.py
================
Publication-quality visualizations for EDM analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import os
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'sans-serif',
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.5,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'lines.linewidth': 2,
    'lines.markersize': 8
})


def ensure_plot_dir():
    """Ensure plot directory exists."""
    if not os.path.exists('plots'):
        os.makedirs('plots')


def plot_validation_metrics(results, save=True):
    """
    Plot clustering validation metrics.
    
    Parameters:
    -----------
    results : dict
        Results from natural cluster finding
    save : bool
        Whether to save the plot
    """
    ensure_plot_dir()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    k_values = results['k_values']
    metrics = results['metrics']
    optimal_k = results['optimal_k']
    
    # Silhouette scores
    ax1.plot(k_values, metrics['silhouette'], 'go-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Silhouette Score')
    ax1.set_title('Silhouette Score Analysis')
    ax1.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax1.axvline(x=35, color='orange', linestyle=':', alpha=0.7, linewidth=2)
    ax1.legend(['Score', f'Natural ({optimal_k})', 'Industry (35)'])
    
    # Elbow plot
    ax2.plot(k_values, metrics['inertia'], 'bo-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Within-Cluster Sum of Squares')
    ax2.set_title('Elbow Method')
    ax2.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax2.axvline(x=35, color='orange', linestyle=':', alpha=0.7, linewidth=2)
    
    # Calinski-Harabasz
    ax3.plot(k_values, metrics['calinski'], 'mo-', linewidth=2, markersize=8)
    ax3.set_xlabel('Number of Clusters')
    ax3.set_ylabel('Calinski-Harabasz Score')
    ax3.set_title('Calinski-Harabasz Score')
    ax3.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax3.axvline(x=35, color='orange', linestyle=':', alpha=0.7, linewidth=2)
    
    # Davies-Bouldin
    ax4.plot(k_values, metrics['davies_bouldin'], 'co-', linewidth=2, markersize=8)
    ax4.set_xlabel('Number of Clusters')
    ax4.set_ylabel('Davies-Bouldin Index')
    ax4.set_title('Davies-Bouldin Index (lower is better)')
    ax4.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax4.axvline(x=35, color='orange', linestyle=':', alpha=0.7, linewidth=2)
    
    plt.suptitle('Cluster Validation Metrics', fontsize=18)
    plt.tight_layout()
    
    if save:
        plt.savefig('plots/validation_metrics.png', dpi=300, bbox_inches='tight')
        print("Saved: plots/validation_metrics.png")
    plt.show()


def plot_tsne_comparison(X, natural_labels, forced_labels, y_true, save=True):
    """
    Create t-SNE comparison plot.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    natural_labels : array-like
        Natural cluster labels
    forced_labels : array-like
        Forced cluster labels
    y_true : array-like
        True genre labels
    save : bool
        Whether to save the plot
    """
    ensure_plot_dir()
    
    print("Computing t-SNE projection...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
    X_tsne = tsne.fit_transform(X)
    
    # Convert genre labels to numeric if needed
    if isinstance(y_true[0], str):
        unique_genres = np.unique(y_true)
        genre_to_num = {genre: i for i, genre in enumerate(unique_genres)}
        y_numeric = np.array([genre_to_num[genre] for genre in y_true])
    else:
        y_numeric = y_true
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Natural clusters
    scatter1 = ax1.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                          c=natural_labels, cmap='tab20', 
                          alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
    ax1.set_title(f'Natural Clusters ({len(np.unique(natural_labels))})')
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Forced clusters
    scatter2 = ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                          c=forced_labels, cmap='rainbow', 
                          alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
    ax2.set_title(f'Industry Clusters ({len(np.unique(forced_labels))})')
    ax2.set_xlabel('t-SNE 1')
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # True genres
    scatter3 = ax3.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                          c=y_numeric, cmap='rainbow', 
                          alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
    ax3.set_title(f'True Genres ({len(np.unique(y_numeric))})')
    ax3.set_xlabel('t-SNE 1')
    ax3.set_xticks([])
    ax3.set_yticks([])
    
    plt.suptitle('Clustering Comparison: Natural vs Industry vs True Genres', fontsize=18)
    plt.tight_layout()
    
    if save:
        plt.savefig('plots/tsne_comparison.png', dpi=300, bbox_inches='tight')
        print("Saved: plots/tsne_comparison.png")
    plt.show()
    
    return X_tsne


def plot_genre_convergence_heatmap(convergence_matrix, save=True):
    """
    Plot genre convergence heatmap.
    
    Parameters:
    -----------
    convergence_matrix : DataFrame
        Genre similarity matrix
    save : bool
        Whether to save the plot
    """
    ensure_plot_dir()
    
    plt.figure(figsize=(12, 10))
    
    # Create mask for diagonal
    mask = np.zeros_like(convergence_matrix, dtype=bool)
    np.fill_diagonal(mask, True)
    
    # Create heatmap
    sns.heatmap(convergence_matrix, 
                mask=mask,
                cmap='RdBu_r',
                center=0.5,
                vmin=0, vmax=1,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": .8, "label": "Acoustic Similarity"},
                annot=False)
    
    plt.title('Genre Acoustic Similarity Matrix', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save:
        plt.savefig('plots/genre_convergence_heatmap.png', dpi=300, bbox_inches='tight')
        print("Saved: plots/genre_convergence_heatmap.png")
    plt.show()


def plot_cluster_comparison_bars(natural_k, forced_k, save=True):
    """
    Simple bar chart comparing natural vs forced clusters.
    
    Parameters:
    -----------
    natural_k : int
        Number of natural clusters
    forced_k : int
        Number of forced clusters
    save : bool
        Whether to save the plot
    """
    ensure_plot_dir()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    categories = ['Natural', 'Industry']
    values = [natural_k, forced_k]
    colors = ['#2ca02c', '#ff7f0e']
    
    bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(val), ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    ax.set_ylabel('Number of Clusters', fontsize=14)
    ax.set_title('Natural vs Industry Clustering', fontsize=16)
    ax.set_ylim(0, max(values) * 1.2)
    
    # Add difference annotation
    diff = forced_k - natural_k
    ax.text(0.5, max(values) * 1.1, f'Over-segmentation: {diff} categories',
            ha='center', transform=ax.transData, fontsize=12,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    
    if save:
        plt.savefig('plots/cluster_comparison.png', dpi=300, bbox_inches='tight')
        print("Saved: plots/cluster_comparison.png")
    plt.show()


def create_summary_report_plot(results_dict, save=True):
    """
    Create a summary report visualization.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary containing all results
    save : bool
        Whether to save the plot
    """
    ensure_plot_dir()
    
    fig = plt.figure(figsize=(16, 10))
    
    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Extract data
    natural_k = results_dict.get('natural_clusters', 10)
    forced_k = results_dict.get('forced_clusters', 35)
    
    # Panel 1: Cluster comparison
    ax1 = fig.add_subplot(gs[0, 0])
    categories = ['Natural', 'Industry']
    values = [natural_k, forced_k]
    colors = ['#2ca02c', '#ff7f0e']
    
    bars = ax1.bar(categories, values, color=colors, alpha=0.7)
    ax1.set_ylabel('Number of Clusters')
    ax1.set_title('Cluster Count')
    
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(val), ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Panel 2: Metrics comparison
    ax2 = fig.add_subplot(gs[0, 1])
    if 'metrics' in results_dict:
        metrics = results_dict['metrics']
        metric_names = ['NMI\nvs Genres', 'ARI\nvs Genres']
        natural_vals = [
            metrics.get('natural_vs_genres_nmi', 0),
            metrics.get('natural_vs_genres_ari', 0)
        ]
        forced_vals = [
            metrics.get('forced_vs_genres_nmi', 0),
            metrics.get('forced_vs_genres_ari', 0)
        ]
        
        x = np.arange(len(metric_names))
        width = 0.35
        
        ax2.bar(x - width/2, natural_vals, width, label='Natural', color='#2ca02c')
        ax2.bar(x + width/2, forced_vals, width, label='Industry', color='#ff7f0e')
        
        ax2.set_ylabel('Score')
        ax2.set_title('Clustering Quality')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metric_names)
        ax2.legend()
        ax2.set_ylim(0, 1)
    
    # Panel 3: Method results
    ax3 = fig.add_subplot(gs[0, 2])
    if 'method_results' in results_dict:
        methods = list(results_dict['method_results'].keys())
        values = list(results_dict['method_results'].values())
        
        ax3.barh(methods, values, color='skyblue')
        ax3.set_xlabel('Optimal Clusters')
        ax3.set_title('By Method')
        ax3.axvline(x=natural_k, color='red', linestyle='--', alpha=0.7)
    
    # Panel 4: Text summary
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis('off')
    
    summary_text = f"""
    KEY FINDINGS:
    • Natural acoustic families: {natural_k}
    • Industry taxonomy: {forced_k} genres
    • Over-segmentation: {forced_k - natural_k} excess categories
    
    INTERPRETATION:
    The EDM industry has created {forced_k - natural_k} additional genre categories beyond
    what the acoustic features naturally support. This suggests significant
    marketing-driven categorization rather than purely musical differences.
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.3))
    
    plt.suptitle('EDM Genre Analysis Summary', fontsize=18, fontweight='bold')
    
    if save:
        plt.savefig('plots/summary_report.png', dpi=300, bbox_inches='tight')
        print("Saved: plots/summary_report.png")
    plt.show()