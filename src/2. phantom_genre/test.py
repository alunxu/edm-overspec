"""
phantom_2d_simple.py
====================
Simplified 2D visualization using the exact feature matrix from main analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import pickle
import os


def load_feature_matrix():
    """Load the exact feature matrix used in main analysis."""
    
    # First try the dedicated feature matrix file
    feature_matrix_path = 'results/pkl/feature_matrix.pkl'
    if os.path.exists(feature_matrix_path):
        print(f"Loading features from {feature_matrix_path}")
        with open(feature_matrix_path, 'rb') as f:
            data = pickle.load(f)
        return data['X_selected'], data['genre_labels']
    
    # Otherwise load from main results
    results_path = 'results/pkl/edm_complete_analysis_results.pkl'
    if os.path.exists(results_path):
        print(f"Loading features from {results_path}")
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        
        # Get clustering data
        clustering_df = pd.read_csv('results/csv/edm_clustering_summary.csv')
        
        # The feature matrix should be in the results
        if 'feature_matrix' in results:
            X = results['feature_matrix']
        else:
            # Reconstruct t-SNE input from the visualization
            # The main analysis uses X_selected.values for t-SNE
            # We need to match that exactly
            print("Feature matrix not found in results. Reconstructing...")
            
            # Load original data and apply same transformations
            df = pd.read_csv('dataset/top100_with_tempogram_nmf_meta.csv')
            
            # This is a simplified reconstruction - ideally run main_analysis.py 
            # with the feature matrix saving code added
            raise ValueError("Please run main_analysis.py with feature matrix saving enabled")
        
        y = clustering_df['true_genre'].values
        return X, y
    
    raise FileNotFoundError("Could not find saved feature matrix. Please run main_analysis.py first.")


def create_phantom_2d_visualization():
    """Create 2D visualization of phantom genres using saved features."""
    
    print("="*70)
    print("PHANTOM GENRES IN 2D SPACE - USING MAIN ANALYSIS FEATURES")
    print("="*70)
    
    # Load the exact feature matrix from main analysis
    X, y = load_feature_matrix()
    print(f"\nLoaded feature matrix: {X.shape}")
    print(f"Unique genres: {len(np.unique(y))}")
    
    # Calculate genre centroids in feature space
    unique_genres = np.unique(y)
    genre_centroids = {}
    genre_features = []
    
    for genre in unique_genres:
        mask = y == genre
        centroid = X[mask].mean(axis=0)
        genre_centroids[genre] = centroid
        genre_features.append(centroid)
    
    genre_features = np.array(genre_features)
    
    # Calculate pairwise distances
    distances = cdist(genre_features, genre_features, metric='euclidean')
    
    # Find close pairs (phantom candidates)
    phantom_threshold = np.percentile(distances[distances > 0], 20)
    phantom_pairs = []
    
    for i in range(len(unique_genres)):
        for j in range(i+1, len(unique_genres)):
            if distances[i, j] < phantom_threshold:
                phantom_pairs.append({
                    'genre1': unique_genres[i],
                    'genre2': unique_genres[j],
                    'distance': distances[i, j]
                })
    
    phantom_pairs_df = pd.DataFrame(phantom_pairs).sort_values('distance')
    print(f"\nFound {len(phantom_pairs_df)} close genre pairs")
    
    # Create visualization
    fig = plt.figure(figsize=(20, 10))
    
    # 1. t-SNE projection (matching main analysis)
    ax1 = plt.subplot(1, 2, 1)
    
    print("\nComputing t-SNE (this matches the main analysis)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
    X_tsne = tsne.fit_transform(X)
    
    # Plot with genre colors
    unique_genres = np.unique(y)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_genres)))
    genre_color_map = {genre: colors[i] for i, genre in enumerate(unique_genres)}
    
    # Plot all points
    for genre in unique_genres:
        mask = y == genre
        ax1.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                   c=[genre_color_map[genre]], 
                   alpha=0.6, s=50, 
                   label=genre if len(genre) < 20 else genre[:17]+'...',
                   edgecolors='white', linewidth=0.5)
    
    # Calculate and plot centroids in t-SNE space
    genre_centroids_tsne = {}
    for genre in unique_genres:
        mask = y == genre
        if mask.any():
            centroid_tsne = X_tsne[mask].mean(axis=0)
            genre_centroids_tsne[genre] = centroid_tsne
            
            # Plot centroid as star
            ax1.scatter(centroid_tsne[0], centroid_tsne[1], 
                       c=[genre_color_map[genre]], 
                       s=300, marker='*', 
                       edgecolors='black', linewidth=2)
    
    # Draw connections for phantom pairs
    for _, pair in phantom_pairs_df.head(10).iterrows():
        g1, g2 = pair['genre1'], pair['genre2']
        if g1 in genre_centroids_tsne and g2 in genre_centroids_tsne:
            x1, y1 = genre_centroids_tsne[g1]
            x2, y2 = genre_centroids_tsne[g2]
            
            ax1.plot([x1, x2], [y1, y2], 'k-', alpha=0.5, linewidth=2, zorder=1)
            
            # Add distance label
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax1.text(mid_x, mid_y, f"{pair['distance']:.1f}", 
                    fontsize=9, ha='center', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax1.set_xlabel('t-SNE Dimension 1')
    ax1.set_ylabel('t-SNE Dimension 2')
    ax1.set_title('Phantom Genres in t-SNE Space\n(Using 100 selected features from main analysis)', 
                 fontsize=14, weight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add legend
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
              fontsize=8, ncol=1)
    
    # 2. PCA projection for comparison
    ax2 = plt.subplot(1, 2, 2)
    
    print("Computing PCA projection...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    
    # Plot all points
    for genre in unique_genres:
        mask = y == genre
        ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=[genre_color_map[genre]], 
                   alpha=0.6, s=50,
                   edgecolors='white', linewidth=0.5)
    
    # Plot centroids and connections in PCA space
    genre_centroids_pca = {}
    for genre in unique_genres:
        mask = y == genre
        if mask.any():
            centroid_pca = X_pca[mask].mean(axis=0)
            genre_centroids_pca[genre] = centroid_pca
            
            ax2.scatter(centroid_pca[0], centroid_pca[1], 
                       c=[genre_color_map[genre]], 
                       s=300, marker='*', 
                       edgecolors='black', linewidth=2)
    
    # Draw phantom connections
    for _, pair in phantom_pairs_df.head(10).iterrows():
        g1, g2 = pair['genre1'], pair['genre2']
        if g1 in genre_centroids_pca and g2 in genre_centroids_pca:
            x1, y1 = genre_centroids_pca[g1]
            x2, y2 = genre_centroids_pca[g2]
            
            ax2.plot([x1, x2], [y1, y2], 'k-', alpha=0.5, linewidth=2, zorder=1)
    
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax2.set_title(f'Phantom Genres in PCA Space\n(Total variance explained: {sum(pca.explained_variance_ratio_):.1%})',
                 fontsize=14, weight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/phantom_genres_2d_simple.png', dpi=300, bbox_inches='tight')
    print("\nSaved: plots/phantom_genres_2d_simple.png")
    plt.show()
    
    # Print top phantom pairs
    print("\n" + "="*70)
    print("TOP PHANTOM PAIRS (Closest in 100-D feature space):")
    print("="*70)
    for _, pair in phantom_pairs_df.head(15).iterrows():
        print(f"{pair['genre1']:30} ↔ {pair['genre2']:30} : distance = {pair['distance']:.3f}")
    
    # Also load and compare with co-clustering results
    try:
        coclustering_df = pd.read_csv('results/phantom_analysis_final/phantom_genre_pairs.csv')
        print("\n" + "="*70)
        print("COMPARING WITH CO-CLUSTERING RESULTS:")
        print("="*70)
        
        # Check which co-clustering pairs are also close in feature space
        for _, row in coclustering_df.head(10).iterrows():
            g1, g2 = row['genre1'], row['genre2']
            
            # Find feature space distance
            if g1 in unique_genres and g2 in unique_genres:
                i1 = np.where(unique_genres == g1)[0][0]
                i2 = np.where(unique_genres == g2)[0][0]
                feat_dist = distances[i1, i2]
                
                print(f"{g1:25} ↔ {g2:25} : "
                      f"co-cluster = {row['similarity']:.2f}, "
                      f"feature dist = {feat_dist:.2f}")
    except:
        pass
    
    return phantom_pairs_df


if __name__ == "__main__":
    try:
        phantom_pairs = create_phantom_2d_visualization()
        print("\n" + "="*70)
        print("VISUALIZATION COMPLETE!")
        print("="*70)
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have run main_analysis.py first.")
        print("If the feature matrix is not saved, add the saving code to main_analysis.py")
        import traceback
        traceback.print_exc()