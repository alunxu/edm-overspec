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
from scipy.cluster.hierarchy import dendrogram, linkage
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
    'lines.markersize': 6
})


def ensure_plot_dir():
    """Ensure plot directory exists."""
    if not os.path.exists('plots'):
        os.makedirs('plots')


def plot_validation_metrics(results, features_scaled=None, y_true=None, save=True):
    """
    Plot clustering validation metrics with genre-based hierarchical dendrogram.
    
    Parameters:
    -----------
    results : dict
        Results from natural cluster finding
    features_scaled : array-like, optional
        Scaled features for hierarchical clustering
    y_true : array-like, optional
        True genre labels
    save : bool
        Whether to save the plot
    """
    ensure_plot_dir()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    k_values = results['k_values']
    metrics = results['metrics']
    optimal_k = results['optimal_k']
    
    # Left panel: All metrics on one plot
    # Normalize metrics to [0, 1] for comparison
    silhouette_norm = (metrics['silhouette'] - np.min(metrics['silhouette'])) / \
                      (np.max(metrics['silhouette']) - np.min(metrics['silhouette']))
    
    # Invert and normalize inertia (lower is better)
    inertia_norm = 1 - (metrics['inertia'] - np.min(metrics['inertia'])) / \
                   (np.max(metrics['inertia']) - np.min(metrics['inertia']))
    
    # Normalize Calinski-Harabasz
    calinski_norm = (metrics['calinski'] - np.min(metrics['calinski'])) / \
                    (np.max(metrics['calinski']) - np.min(metrics['calinski']))
    
    # Invert and normalize Davies-Bouldin (lower is better)
    davies_norm = 1 - (metrics['davies_bouldin'] - np.min(metrics['davies_bouldin'])) / \
                  (np.max(metrics['davies_bouldin']) - np.min(metrics['davies_bouldin']))
    
    # Plot all metrics
    ax1.plot(k_values, silhouette_norm, 'o-', linewidth=2.5, markersize=6, 
             label='Silhouette Score', color='#2ecc71')
    ax1.plot(k_values, inertia_norm, 's-', linewidth=2.5, markersize=6, 
             label='Elbow Method (inverted)', color='#3498db')
    ax1.plot(k_values, calinski_norm, '^-', linewidth=2.5, markersize=6, 
             label='Calinski-Harabasz', color='#9b59b6')
    ax1.plot(k_values, davies_norm, 'd-', linewidth=2.5, markersize=6, 
             label='Davies-Bouldin (inverted)', color='#e74c3c')
    
    # Add vertical lines
    ax1.axvline(x=optimal_k, color='darkred', linestyle='--', alpha=0.8, linewidth=2.5,
                label=f'Optimal k={optimal_k}')
    ax1.axvline(x=35, color='orange', linestyle=':', alpha=0.8, linewidth=2.5,
                label='Industry (35)')
    
    # Highlight optimal region
    ax1.axvspan(optimal_k-1, optimal_k+1, alpha=0.2, color='red')
    
    ax1.set_xlabel('Number of Clusters', fontsize=14)
    ax1.set_ylabel('Normalized Score (higher is better)', fontsize=14)
    ax1.set_title('Clustering Validation Metrics Comparison', fontsize=16, weight='bold')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(k_values[0]-0.5, k_values[-1]+0.5)
    ax1.set_ylim(-0.05, 1.05)
    
    # Right panel: Genre-based hierarchical clustering dendrogram
    if features_scaled is not None and y_true is not None:
        # Create genre-level features by averaging
        unique_genres = np.unique(y_true)
        genre_features = []
        genre_labels = []
        
        for genre in unique_genres:
            genre_mask = y_true == genre
            genre_mean = np.mean(features_scaled[genre_mask], axis=0)
            genre_features.append(genre_mean)
            genre_labels.append(genre)
        
        genre_features = np.array(genre_features)
        
        # Compute linkage for genres
        genre_linkage = linkage(genre_features, method='ward')
        
        # Create dendrogram with genre labels
        dend = dendrogram(genre_linkage, 
                         labels=genre_labels,
                         color_threshold=None,
                         ax=ax2,
                         leaf_rotation=90,
                         leaf_font_size=10)
        
        # Find appropriate cut heights
        max_height = max(genre_linkage[:, 2])
        
        # Find height that gives optimal_k clusters
        if optimal_k <= len(unique_genres):
            sorted_heights = np.sort(genre_linkage[:, 2])
            if optimal_k > 1:
                optimal_height = sorted_heights[-(optimal_k-1)] - 0.01
            else:
                optimal_height = sorted_heights[-1] + 0.01
            
            ax2.axhline(y=optimal_height, color='darkred', linestyle='--', 
                       linewidth=2.5, label=f'Cut for {optimal_k} clusters')
        
        # Add annotation for industry standard (35 genres)
        # ax2.text(0.02, 0.02, 
        #         f'Note: Industry uses {len(unique_genres)} genres\nHierarchy shows natural groupings',
        #         transform=ax2.transAxes, 
        #         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8),
        #         fontsize=10)
        
        ax2.set_xlabel('Genre', fontsize=14)
        ax2.set_ylabel('Distance', fontsize=14)
        ax2.set_title('Genre Hierarchical Clustering', fontsize=16, weight='bold')
        ax2.legend(loc='best', fontsize=11)
        
    else:
        ax2.text(0.5, 0.5, 'Hierarchical clustering data not provided', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        ax2.axis('off')
    
    plt.suptitle('Cluster Validation Analysis', fontsize=18, weight='bold')
    plt.tight_layout()
    
    if save:
        plt.savefig('plots/validation_metrics_combined.png', dpi=300, bbox_inches='tight')
        print("Saved: plots/validation_metrics_combined.png")
    plt.show()


def plot_tsne_comparison(X, natural_labels, forced_labels, y_true, save=True):
    """
    Create t-SNE comparison plot with interpretable titles.
    
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
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Natural clusters
    scatter1 = ax1.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                          c=natural_labels, cmap='tab10', 
                          alpha=0.7, s=30, edgecolors='white', linewidth=0.5)
    ax1.set_title(f'What the Music Shows:\n{len(np.unique(natural_labels))} Natural Acoustic Families', 
                  fontsize=14, weight='bold')
    ax1.set_xlabel('t-SNE Dimension 1')
    ax1.set_ylabel('t-SNE Dimension 2')
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Add text annotation
    ax1.text(0.02, 0.02, 'Data-driven clustering\nbased on acoustic features', 
            transform=ax1.transAxes, 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
            fontsize=10)
    
    # Forced clusters
    scatter2 = ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                          c=forced_labels, cmap='rainbow', 
                          alpha=0.7, s=30, edgecolors='white', linewidth=0.5)
    ax2.set_title(f'What Industry Imposes:\n{len(np.unique(forced_labels))} Marketing Categories', 
                  fontsize=14, weight='bold')
    ax2.set_xlabel('t-SNE Dimension 1')
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # Add text annotation
    ax2.text(0.02, 0.02, 'Forced to match\n35 industry genres', 
            transform=ax2.transAxes, 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.7),
            fontsize=10)
    
    # True genres
    scatter3 = ax3.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                          c=y_numeric, cmap='rainbow', 
                          alpha=0.7, s=30, edgecolors='white', linewidth=0.5)
    ax3.set_title(f'Market Reality:\n{len(np.unique(y_numeric))} Genre Labels', 
                  fontsize=14, weight='bold')
    ax3.set_xlabel('t-SNE Dimension 1')
    ax3.set_xticks([])
    ax3.set_yticks([])
    
    # Add text annotation
    ax3.text(0.02, 0.02, 'How tracks are\nactually labeled', 
            transform=ax3.transAxes, 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.7),
            fontsize=10)
    
    # Main title with key insight
    plt.suptitle('The Phantom Genre Problem: Natural Clustering vs Commercial Categorization', 
                fontsize=18, weight='bold')
    
    # Add bottom annotation
    fig.text(0.5, 0.02, 
            f'Key Finding: {len(np.unique(forced_labels)) - len(np.unique(natural_labels))} ' +
            f'excess genre categories exist beyond natural acoustic boundaries',
            ha='center', fontsize=12, style='italic', weight='bold', color='darkred')
    
    plt.tight_layout()
    
    if save:
        plt.savefig('plots/tsne_comparison_interpretable.png', dpi=300, bbox_inches='tight')
        print("Saved: plots/tsne_comparison_interpretable.png")
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