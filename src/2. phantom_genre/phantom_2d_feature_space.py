"""
phantom_2d_feature_space.py
===========================
Visualize phantom genre pairs in 2D feature space using various dimensionality reduction techniques
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import matplotlib.patches as mpatches
from matplotlib.patches import ConnectionPatch
from adjustText import adjust_text


def visualize_phantom_genres_2d():
    """Create 2D visualizations showing phantom genre relationships."""
    
    print("="*70)
    print("PHANTOM GENRES IN 2D FEATURE SPACE")
    print("="*70)
    
    # Load data
    df = pd.read_csv('dataset/top100_with_tempogram_nmf_meta.csv')
    
    # Get numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[numeric_cols].values
    y = df['genre'].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Calculate genre centroids
    print("\nCalculating genre centroids...")
    unique_genres = np.unique(y)
    genre_centroids = {}
    genre_features_scaled = []
    genre_labels = []
    
    for genre in unique_genres:
        mask = y == genre
        centroid = X_scaled[mask].mean(axis=0)
        genre_centroids[genre] = centroid
        genre_features_scaled.append(centroid)
        genre_labels.append(genre)
    
    genre_features_scaled = np.array(genre_features_scaled)
    
    # Calculate pairwise distances between genres
    distances = cdist(genre_features_scaled, genre_features_scaled, metric='euclidean')
    
    # Find phantom pairs (very close in feature space)
    phantom_threshold = np.percentile(distances[distances > 0], 15)  # Bottom 15% of distances
    phantom_pairs = []
    
    for i in range(len(genre_labels)):
        for j in range(i+1, len(genre_labels)):
            if distances[i, j] < phantom_threshold:
                phantom_pairs.append({
                    'genre1': genre_labels[i],
                    'genre2': genre_labels[j],
                    'distance': distances[i, j]
                })
    
    phantom_pairs_df = pd.DataFrame(phantom_pairs).sort_values('distance')
    
    print(f"\nFound {len(phantom_pairs_df)} close genre pairs (distance < {phantom_threshold:.2f})")
    
    # Create figure with multiple projections
    fig = plt.figure(figsize=(20, 15))
    
    # 1. PCA projection
    ax1 = plt.subplot(2, 2, 1)
    create_2d_projection(X_scaled, y, genre_centroids, phantom_pairs_df, 
                        'PCA', ax1, genre_labels)
    
    # 2. t-SNE projection
    ax2 = plt.subplot(2, 2, 2)
    create_2d_projection(X_scaled, y, genre_centroids, phantom_pairs_df, 
                        'TSNE', ax2, genre_labels)
    
    # 3. MDS projection
    ax3 = plt.subplot(2, 2, 3)
    create_2d_projection(X_scaled, y, genre_centroids, phantom_pairs_df, 
                        'MDS', ax3, genre_labels)
    
    # 4. Distance matrix visualization
    ax4 = plt.subplot(2, 2, 4)
    create_distance_heatmap(distances, genre_labels, phantom_pairs_df, ax4)
    
    plt.suptitle('Phantom Genres in 2D Feature Space', fontsize=18, weight='bold')
    plt.tight_layout()
    plt.savefig('plots/phantom_genres_2d_space.png', dpi=300, bbox_inches='tight')
    print("\nSaved: plots/phantom_genres_2d_space.png")
    plt.show()
    
    # Create detailed view of top phantom pairs
    create_phantom_pairs_detail(X_scaled, y, phantom_pairs_df.head(6))
    
    return phantom_pairs_df


def create_2d_projection(X, y, genre_centroids, phantom_pairs, method, ax, genre_labels):
    """Create a 2D projection using specified method."""
    
    print(f"\nCreating {method} projection...")
    
    # Apply dimensionality reduction
    if method == 'PCA':
        reducer = PCA(n_components=2, random_state=42)
        X_2d = reducer.fit_transform(X)
        explained_var = reducer.explained_variance_ratio_
        ax.set_xlabel(f'PC1 ({explained_var[0]:.1%} var)')
        ax.set_ylabel(f'PC2 ({explained_var[1]:.1%} var)')
    elif method == 'TSNE':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        X_2d = reducer.fit_transform(X)
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
    elif method == 'MDS':
        reducer = MDS(n_components=2, random_state=42)
        X_2d = reducer.fit_transform(X)
        ax.set_xlabel('MDS 1')
        ax.set_ylabel('MDS 2')
    
    # Create color map for genres
    unique_genres = np.unique(y)
    n_genres = len(unique_genres)
    colors = plt.cm.tab20(np.linspace(0, 1, n_genres))
    genre_color_map = {genre: colors[i] for i, genre in enumerate(unique_genres)}
    
    # Plot all points with transparency
    for genre in unique_genres:
        mask = y == genre
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                  c=[genre_color_map[genre]], 
                  alpha=0.3, s=30, 
                  label=genre if len(genre) < 20 else genre[:17]+'...')
    
    # Calculate and plot genre centroids
    genre_centroids_2d = {}
    for genre in unique_genres:
        mask = y == genre
        if mask.any():
            centroid_2d = X_2d[mask].mean(axis=0)
            genre_centroids_2d[genre] = centroid_2d
            
            # Plot centroid
            ax.scatter(centroid_2d[0], centroid_2d[1], 
                      c=[genre_color_map[genre]], 
                      s=200, marker='*', 
                      edgecolors='black', linewidth=2)
    
    # Draw connections between phantom pairs
    for _, pair in phantom_pairs.iterrows():
        g1, g2 = pair['genre1'], pair['genre2']
        if g1 in genre_centroids_2d and g2 in genre_centroids_2d:
            x1, y1 = genre_centroids_2d[g1]
            x2, y2 = genre_centroids_2d[g2]
            
            # Draw line
            ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.5, linewidth=2, zorder=1)
            
            # Add distance label at midpoint
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y, f"{pair['distance']:.1f}", 
                   fontsize=8, ha='center', 
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
    
    ax.set_title(f'{method} Projection', fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add legend outside plot
    if method == 'PCA':
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                 fontsize=8, ncol=2)


def create_distance_heatmap(distances, genre_labels, phantom_pairs, ax):
    """Create a heatmap of genre distances."""
    
    # Create DataFrame for easier handling
    dist_df = pd.DataFrame(distances, index=genre_labels, columns=genre_labels)
    
    # Mask upper triangle
    mask = np.triu(np.ones_like(distances), k=1)
    
    # Create custom colormap
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    
    # Plot heatmap
    sns.heatmap(dist_df, mask=mask, cmap=cmap, center=np.median(distances),
                square=True, linewidths=0.5, 
                cbar_kws={"shrink": .8, "label": "Feature Distance"},
                ax=ax)
    
    # Highlight phantom pairs
    for _, pair in phantom_pairs.iterrows():
        g1_idx = genre_labels.index(pair['genre1'])
        g2_idx = genre_labels.index(pair['genre2'])
        
        # Draw rectangle around phantom pair
        rect = plt.Rectangle((min(g1_idx, g2_idx), max(g1_idx, g2_idx)), 
                           1, 1, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
    
    ax.set_title('Genre Distance Matrix\n(Red boxes = phantom pairs)', 
                fontsize=14, weight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)


def create_phantom_pairs_detail(X, y, top_phantom_pairs):
    """Create detailed view of top phantom pairs."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Use t-SNE for all detailed views
    print("\nCreating detailed t-SNE projections for top phantom pairs...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X)
    
    for idx, (_, pair) in enumerate(top_phantom_pairs.iterrows()):
        if idx >= 6:
            break
            
        ax = axes[idx]
        g1, g2 = pair['genre1'], pair['genre2']
        
        # Create mask for these two genres
        mask = (y == g1) | (y == g2)
        other_mask = ~mask
        
        # Plot other genres in gray
        ax.scatter(X_tsne[other_mask, 0], X_tsne[other_mask, 1], 
                  c='lightgray', alpha=0.2, s=20)
        
        # Plot the two phantom genres
        mask_g1 = y == g1
        mask_g2 = y == g2
        
        scatter1 = ax.scatter(X_tsne[mask_g1, 0], X_tsne[mask_g1, 1], 
                            c='#FF6B6B', alpha=0.8, s=60, label=g1[:20])
        scatter2 = ax.scatter(X_tsne[mask_g2, 0], X_tsne[mask_g2, 1], 
                            c='#4ECDC4', alpha=0.8, s=60, label=g2[:20])
        
        # Calculate and plot centroids
        if mask_g1.any() and mask_g2.any():
            centroid1 = X_tsne[mask_g1].mean(axis=0)
            centroid2 = X_tsne[mask_g2].mean(axis=0)
            
            ax.scatter(*centroid1, c='#FF6B6B', s=300, marker='*', 
                      edgecolors='black', linewidth=2)
            ax.scatter(*centroid2, c='#4ECDC4', s=300, marker='*', 
                      edgecolors='black', linewidth=2)
            
            # Draw connection
            ax.plot([centroid1[0], centroid2[0]], 
                   [centroid1[1], centroid2[1]], 
                   'k--', alpha=0.5, linewidth=2)
            
            # Calculate overlap
            from matplotlib.patches import Ellipse
            from sklearn.covariance import EllipticEnvelope
            
            # Fit ellipses to show distribution
            try:
                for data, color in [(X_tsne[mask_g1], '#FF6B6B'), 
                                   (X_tsne[mask_g2], '#4ECDC4')]:
                    if len(data) > 2:
                        mean = data.mean(axis=0)
                        cov = np.cov(data.T)
                        eigenvalues, eigenvectors = np.linalg.eig(cov)
                        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
                        
                        ellipse = Ellipse(mean, 2*np.sqrt(eigenvalues[0]), 
                                        2*np.sqrt(eigenvalues[1]), 
                                        angle=angle, facecolor=color, 
                                        alpha=0.2, edgecolor=color, linewidth=2)
                        ax.add_patch(ellipse)
            except:
                pass
        
        ax.set_title(f'{g1[:15]} vs {g2[:15]}\nDistance: {pair["distance"]:.2f}', 
                    fontsize=10, weight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.suptitle('Detailed View: Top Phantom Genre Pairs in t-SNE Space', 
                fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig('plots/phantom_pairs_detail_2d.png', dpi=300, bbox_inches='tight')
    print("Saved: plots/phantom_pairs_detail_2d.png")
    plt.show()


def main():
    """Run the 2D phantom genre visualization."""
    
    try:
        phantom_pairs = visualize_phantom_genres_2d()
        
        print("\n" + "="*70)
        print("TOP PHANTOM GENRE PAIRS (Closest in Feature Space):")
        print("="*70)
        
        for _, pair in phantom_pairs.head(10).iterrows():
            print(f"{pair['genre1']} ↔ {pair['genre2']}: distance = {pair['distance']:.3f}")
        
        print("\n" + "="*70)
        print("2D VISUALIZATION COMPLETE!")
        print("="*70)
        print("\nCreated visualizations:")
        print("1. phantom_genres_2d_space.png - Multiple 2D projections")
        print("2. phantom_pairs_detail_2d.png - Detailed view of top pairs")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()