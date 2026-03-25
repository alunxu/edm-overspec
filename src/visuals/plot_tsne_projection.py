"""
Script to generate t-SNE visualization with consistent genre colors across panels
Both panels are colored by genre labels for direct comparison
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')
import pickle
import os

# Import your original modules (only needed if pkl files don't exist)
# from feature_engineering import EDMFeatureEngineer
# from feature_selection import EDMFeatureSelector

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 22,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.titlesize': 28,
    'axes.labelsize': 25,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 17,
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


def load_and_process_data():
    """
    Load pre-computed features and labels from saved pkl files.
    Uses results from run_pipeline.py (the canonical pipeline run).
    """
    feature_matrix_path = 'results/pkl/feature_matrix.pkl'
    results_path = 'results/pkl/edm_complete_analysis_results.pkl'
    
    if not os.path.exists(feature_matrix_path):
        raise FileNotFoundError(
            f"Feature matrix not found at {feature_matrix_path}. "
            "Run run_pipeline.py first to generate it."
        )
    
    print("Loading pre-computed features from saved pkl...")
    with open(feature_matrix_path, 'rb') as f:
        fm = pickle.load(f)
    
    X_selected = pd.DataFrame(
        fm['X_selected'],
        columns=fm['feature_names'] if fm.get('feature_names') else None
    )
    y = pd.Series(fm['genre_labels'], name='genre')
    
    # Load the new dataset for metadata
    df = pd.read_csv('dataset/all_features_194.csv')
    
    print(f"Loaded: {X_selected.shape[0]} songs × {X_selected.shape[1]} features")
    print(f"Genres: {y.nunique()}")
    
    return X_selected, y, df


def get_natural_clusters(X_selected, n_clusters):
    """
    Generate natural clusters using K-means.
    Features are already StandardScaler-scaled from the pipeline.
    """
    print(f"\nComputing {n_clusters} natural clusters...")
    
    # No extra scaling needed — data is already StandardScaler-transformed in the pipeline
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=50,
        max_iter=300
    )
    natural_labels = kmeans.fit_predict(X_selected)
    
    print(f"Created {n_clusters} natural clusters")
    
    return natural_labels


def create_genre_colormap(genres):
    """
    Create a consistent colormap for genres.
    Returns a dictionary mapping each genre to a specific color.
    """
    import matplotlib.cm as cm
    
    unique_genres = np.unique(genres)
    n_genres = len(unique_genres)
    
    # Use a combination of colormaps to get 35 distinct colors
    if n_genres <= 20:
        colors = cm.get_cmap('tab20')(np.linspace(0, 1, n_genres))
    else:
        # Combine multiple colormaps for more colors
        colors1 = cm.get_cmap('tab20')(np.linspace(0, 1, 20))
        colors2 = cm.get_cmap('tab20b')(np.linspace(0, 1, min(20, n_genres - 20)))
        colors3 = cm.get_cmap('tab20c')(np.linspace(0, 1, max(0, n_genres - 40)))
        colors = np.vstack([colors1, colors2[:n_genres-20]])
        if n_genres > 40:
            colors = np.vstack([colors, colors3[:n_genres-40]])
    
    # Create genre to color mapping
    genre_colors = {}
    for i, genre in enumerate(unique_genres):
        genre_colors[genre] = colors[i]
    
    return genre_colors


def plot_tsne_consistent_colors(X_selected, natural_labels, y_true, save_path='tsne_comparison.png'):
    """
    Create the two-panel t-SNE visualization with consistent genre colors.
    Left panel: Shows data colored by genre but annotated with cluster boundaries
    Right panel: Shows data colored by genre (same colors)
    """
    print("\n" + "="*60)
    print("CREATING t-SNE VISUALIZATION WITH CONSISTENT COLORS")
    print("="*60)
    
    print("Computing t-SNE projection...")
    print("This may take a few minutes...")
    
    # Convert to numpy array if it's a DataFrame
    if hasattr(X_selected, 'values'):
        X_array = X_selected.values
    else:
        X_array = X_selected
    
    # Convert genre labels to numpy array if needed
    if hasattr(y_true, 'values'):
        y_array = y_true.values
    else:
        y_array = y_true
    
    # Compute t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=min(30, len(X_array)-1),
        random_state=42
    )
    X_tsne = tsne.fit_transform(X_array)
    
    print("t-SNE projection complete!")
    
    # Create consistent genre colormap
    genre_colors = create_genre_colormap(y_array)
    unique_genres = np.unique(y_array)
    
    # Create the figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # LEFT PANEL: Natural clusters structure but colored by genre
    # This shows how genres naturally group into clusters
    for genre in unique_genres:
        mask = y_array == genre
        ax1.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                   c=[genre_colors[genre]], label=genre,
                   alpha=0.7, s=30, edgecolors='white', linewidth=0.5)
    
    # Optionally, add cluster boundaries or centroids
    # Calculate cluster centroids for visualization
    for cluster_id in range(len(np.unique(natural_labels))):
        cluster_mask = natural_labels == cluster_id
        if np.any(cluster_mask):
            centroid = X_tsne[cluster_mask].mean(axis=0)
            ax1.scatter(centroid[0], centroid[1], 
                       c='black', s=100, marker='x', linewidths=2, alpha=0.5)
            ax1.text(centroid[0], centroid[1], str(cluster_id), 
                    fontsize=8, ha='center', va='center', weight='bold',
                    bbox=dict(boxstyle='circle,pad=0.1', facecolor='white', alpha=0.5))
    
    n_unique = len(np.unique(natural_labels))
    ax1.set_title(f'Natural Clustering Structure\n({n_unique} clusters marked with X)', 
                  fontsize=14, weight='bold')
    ax1.set_xlabel('t-SNE Dimension 1')
    ax1.set_ylabel('t-SNE Dimension 2')
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Add annotation box
    ax1.text(0.02, 0.02, 'Genres colored consistently\nCluster centers marked', 
            transform=ax1.transAxes, 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
            fontsize=10)
    
    # RIGHT PANEL: Same data, same colors, showing genre distribution
    for genre in unique_genres:
        mask = y_array == genre
        ax2.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                   c=[genre_colors[genre]], label=genre,
                   alpha=0.7, s=30, edgecolors='white', linewidth=0.5)
    
    ax2.set_title(f'Market Reality:\n{len(unique_genres)} Genre Labels', 
                  fontsize=14, weight='bold')
    ax2.set_xlabel('t-SNE Dimension 1')
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # Add annotation box
    ax2.text(0.02, 0.02, 'Same genre colors\nas left panel', 
            transform=ax2.transAxes, 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.7),
            fontsize=10)
    
    # Main title
    plt.suptitle('The Phantom Genre Problem: Natural Clustering vs Commercial Categorization', 
                fontsize=18, weight='bold')
    
    # Bottom annotation with key finding
    fig.text(0.5, 0.02, 
            f'Key Finding: Music naturally forms {len(np.unique(natural_labels))} acoustic families, but industry uses {len(unique_genres)} genre labels',
            ha='center', fontsize=12, style='italic', weight='bold', color='darkred')
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved as: {save_path}")
    
    plt.close()
    
    return X_tsne, genre_colors


def plot_tsne_alternative(X_selected, natural_labels, y_true, n_clusters=23, save_path='tsne_alternative.png'):
    """
    Alternative visualization: Left shows clusters, Right shows the SAME data with genre overlay
    This better shows the mismatch between natural clusters and genre labels
    """
    print("\nCreating alternative visualization...")
    
    # Convert to arrays
    if hasattr(X_selected, 'values'):
        X_array = X_selected.values
    else:
        X_array = X_selected
    
    if hasattr(y_true, 'values'):
        y_array = y_true.values
    else:
        y_array = y_true
    
    # Compute t-SNE
    tsne = TSNE(n_components=2, perplexity=min(30, len(X_array)-1), random_state=42)
    X_tsne = tsne.fit_transform(X_array)
    
    # Create consistent colors for clusters (handle any n_clusters)
    import matplotlib.cm as cm
    n_unique_clusters = len(np.unique(natural_labels))
    if n_unique_clusters <= 20:
        cluster_colors = cm.get_cmap('tab20')(np.linspace(0, 1, max(n_unique_clusters, 1)))
    else:
        colors1 = cm.get_cmap('tab20')(np.linspace(0, 1, 20))
        colors2 = cm.get_cmap('Set3')(np.linspace(0, 1, n_unique_clusters - 20))
        cluster_colors = np.vstack([colors1, colors2])
    
    # Create genre colors
    genre_colors = create_genre_colormap(y_array)
    unique_genres = np.unique(y_array)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    
    # LEFT: Natural clusters
    for idx, cid in enumerate(np.unique(natural_labels)):
        mask = natural_labels == cid
        ax1.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                   c=[cluster_colors[idx % len(cluster_colors)]], label=f'C{cid}',
                   alpha=0.7, s=30, edgecolors='white', linewidth=0.5)
    
    ax1.set_title(f'Acoustic Reality\n{n_unique_clusters} Natural Clusters', 
                  fontsize=34, weight='bold', color='black', pad=20)
    ax1.set_xlabel('t-SNE Dimension 1', fontsize=25, fontweight='bold')
    ax1.set_ylabel('t-SNE Dimension 2', fontsize=25, fontweight='bold')
    ax1.set_xticks([])
    ax1.set_yticks([])
    # Grey border
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('#9e9e9e')
        spine.set_linewidth(2.0)
    
    # RIGHT: Same points colored by genre
    for genre in unique_genres:
        mask = y_array == genre
        ax2.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                   c=[genre_colors[genre]], label=genre,
                   alpha=0.7, s=30, edgecolors='white', linewidth=0.5)
    
    ax2.set_title(f'Commercial Taxonomy\n{len(unique_genres)} Industry Labels', 
                  fontsize=34, weight='bold', color='black', pad=20)
    ax2.set_xlabel('t-SNE Dimension 1', fontsize=25, fontweight='bold')
    ax2.set_xticks([])
    ax2.set_yticks([])
    # Grey border
    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('#9e9e9e')
        spine.set_linewidth(2.0)
    

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Alternative figure saved as: {save_path}")
    plt.close()
    
    return X_tsne


def main():
    """
    Main function to generate t-SNE visualizations.
    Loads natural_k dynamically from pipeline results.
    """
    
    print("\n" + "="*70)
    print("t-SNE VISUALIZATION GENERATOR")
    print("Using Pipeline Feature Space")
    print("="*70 + "\n")
    
    # Step 1: Load pre-computed features
    X_selected, y, df = load_and_process_data()
    
    # Step 2: Load natural_k from pipeline results, or default to 23
    results_path = 'results/pkl/edm_complete_analysis_results.pkl'
    if os.path.exists(results_path):
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        natural_k = results['clustering']['natural_k']
        print(f"Loaded natural_k = {natural_k} from pipeline results")
    else:
        natural_k = 20
        print(f"Pipeline results not found, using default natural_k = {natural_k}")
    
    # Step 3: Get natural clusters
    natural_labels = get_natural_clusters(X_selected, n_clusters=natural_k)
    
    # Step 4: Create visualization
    plot_tsne_alternative(
        X_selected, natural_labels, y,
        n_clusters=natural_k,
        save_path='paper/SMC/figs/paper/fig1_tsne.png'
    )
    
    print("\n" + "="*70)
    print("FIGURE GENERATION COMPLETE!")
    print("="*70)
    
    # Print summary
    print(f"\nSummary:")
    print(f"- Songs analyzed: {len(df)}")
    print(f"- Natural clusters: {natural_k}")
    print(f"- Industry genres: {len(y.unique())}")
    print(f"- Over-segmentation: {len(y.unique()) - natural_k} excess categories")
    
    return natural_labels, y


if __name__ == "__main__":
    try:
        # Run the main analysis
        natural_labels, y = main()
        
    except ImportError as e:
        print(f"\nERROR: Could not import required modules: {e}")
        print("\nPlease ensure these files are in the same directory:")
        print("  - feature_engineering.py")
        print("  - feature_selection.py")
        print("  - This script")
        print("  - dataset/all_features_194.csv")
        
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nPlease ensure the dataset file exists at:")
        print("  dataset/all_features_194.csv")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()