import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.path import Path
import matplotlib.patches as mpatches
from scipy.interpolate import interp1d
from feature_processing.feature_pipeline import FeaturePipeline
import warnings
warnings.filterwarnings('ignore')

def engineer_features(df):
    """Apply feature engineering using the shared pipeline."""
    pipeline = FeaturePipeline()
    feature_df, _ = pipeline.extract_features(df)
    feature_df = pipeline.engineer(feature_df)
    return feature_df, list(feature_df.columns)


def create_centroid_divergence(data_path):
    """Create visualization showing natural centroids diverging to genre centroids"""
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} tracks across {df['genre'].nunique()} genres")
    
    # Engineer features
    features_df, numeric_cols = engineer_features(df)
    
    # Apply ensemble scaling
    robust_scaler = RobustScaler()
    features_robust = robust_scaler.fit_transform(features_df)
    
    standard_scaler = StandardScaler()
    features_standard = standard_scaler.fit_transform(features_df)
    
    minmax_scaler = MinMaxScaler()
    features_minmax = minmax_scaler.fit_transform(features_df)
    
    features_scaled = (features_robust + features_standard + features_minmax) / 3
    
    # Find natural clusters
    kmeans = KMeans(n_clusters=8, random_state=42)
    natural_labels = kmeans.fit_predict(features_scaled)
    
    # Apply t-SNE for visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(features_scaled)
    
    # Add to dataframe
    df['tsne_x'] = tsne_result[:, 0]
    df['tsne_y'] = tsne_result[:, 1]
    df['natural_cluster'] = natural_labels
    
    # Calculate natural cluster centroids in t-SNE space
    natural_centroids = {}
    for cluster in range(8):
        mask = df['natural_cluster'] == cluster
        natural_centroids[cluster] = {
            'x': df[mask]['tsne_x'].mean(),
            'y': df[mask]['tsne_y'].mean()
        }
    
    # Calculate genre centroids
    genre_centroids = {}
    genre_cluster_mapping = {}
    
    for genre in df['genre'].unique():
        genre_data = df[df['genre'] == genre]
        
        # Find dominant natural cluster for this genre
        dominant_cluster = genre_data['natural_cluster'].mode()[0]
        genre_cluster_mapping[genre] = dominant_cluster
        
        # Calculate genre centroid
        genre_centroids[genre] = {
            'x': genre_data['tsne_x'].mean(),
            'y': genre_data['tsne_y'].mean(),
            'cluster': dominant_cluster,
            'count': len(genre_data)
        }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_facecolor('#f5f5f5')
    fig.patch.set_facecolor('white')
    
    # Define colors for natural clusters
    cluster_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', 
                     '#9b59b6', '#1abc9c', '#34495e', '#e67e22']
    
    # Plot natural cluster centroids (stars)
    for cluster, centroid in natural_centroids.items():
        ax.scatter(centroid['x'], centroid['y'], 
                  s=800, marker='*', c=cluster_colors[cluster],
                  edgecolors='black', linewidth=3, zorder=20,
                  label=f'Natural Cluster {cluster+1}')
    
    # Plot genre centroids and connections
    for genre, genre_cent in genre_centroids.items():
        natural_cluster = genre_cent['cluster']
        nat_cent = natural_centroids[natural_cluster]
        
        # Draw curved connection from natural centroid to genre centroid
        # Create control points for Bezier curve
        dx = genre_cent['x'] - nat_cent['x']
        dy = genre_cent['y'] - nat_cent['y']
        
        # Control point offset for curve
        ctrl_offset_x = dy * 0.3  # Perpendicular offset
        ctrl_offset_y = -dx * 0.3
        
        # Mid point with offset
        mid_x = (nat_cent['x'] + genre_cent['x']) / 2 + ctrl_offset_x
        mid_y = (nat_cent['y'] + genre_cent['y']) / 2 + ctrl_offset_y
        
        # Create path for curved line
        verts = [
            (nat_cent['x'], nat_cent['y']),
            (mid_x, mid_y),
            (genre_cent['x'], genre_cent['y'])
        ]
        codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
        
        path = Path(verts, codes)
        from matplotlib.patches import PathPatch
        patch = PathPatch(path, facecolor='none', 
                         edgecolor=cluster_colors[natural_cluster],
                         linewidth=2, alpha=0.5)
        ax.add_patch(patch)
        
        # Plot genre centroid
        ax.scatter(genre_cent['x'], genre_cent['y'], 
                  s=150, c='white',
                  edgecolors=cluster_colors[natural_cluster], 
                  linewidth=2, zorder=10)
        
        # Add genre label for major genres
        if genre_cent['count'] > 50:  # Only label genres with many tracks
            display_name = genre.replace(' (Peak Time - Driving)', ' (PTD)')\
                               .replace(' (Raw - Deep - Hypnotic)', ' (RDH)')\
                               .replace(' (Main Floor)', ' (MF)')\
                               .replace('Minimal - Deep Tech', 'Min-DT')\
                               .replace('Melodic House & Techno', 'Mel H&T')
            
            # Position label away from centroid
            angle = np.arctan2(genre_cent['y'] - nat_cent['y'], 
                             genre_cent['x'] - nat_cent['x'])
            offset_x = 3 * np.cos(angle)
            offset_y = 3 * np.sin(angle)
            
            ax.annotate(display_name, 
                       (genre_cent['x'], genre_cent['y']),
                       xytext=(genre_cent['x'] + offset_x, genre_cent['y'] + offset_y),
                       fontsize=9, ha='center',
                       bbox=dict(boxstyle='round,pad=0.3', 
                               facecolor='white', alpha=0.7))
    
    # Add grid
    ax.grid(True, alpha=0.3, color='gray', linestyle='--')
    
    # Labels and title
    ax.set_xlabel('t-SNE Dimension 1', fontsize=14)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=14)
    ax.set_title('Natural Cluster Centroids Diverge into 35 Genre Centroids', 
                fontsize=18, weight='bold', pad=20)
    
    # Add legend
    legend_elements = [
        mpatches.Circle((0, 0), 0.5, fc=cluster_colors[0], label='Cluster 1 Family'),
        mpatches.Circle((0, 0), 0.5, fc=cluster_colors[1], label='Cluster 2 Family'),
        mpatches.Circle((0, 0), 0.5, fc=cluster_colors[2], label='Cluster 3 Family'),
        mpatches.Circle((0, 0), 0.5, fc=cluster_colors[3], label='Cluster 4 Family'),
        mpatches.Circle((0, 0), 0.5, fc='black', label='Natural Centroid'),
        mpatches.Circle((0, 0), 0.5, fc='white', ec='black', label='Genre Centroid')
    ]
    ax.legend(handles=legend_elements[:6], loc='upper right', fontsize=10)
    
    # Add explanation
    ax.text(0.02, 0.98, 
           'Stars (★) = Natural cluster centroids (8 total)\n' +
           'Circles (○) = Commercial genre centroids (35 total)\n' +
           'Curved lines show how natural clusters fragment',
           transform=ax.transAxes, va='top', fontsize=11,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('centroid_divergence.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_simplified_divergence(data_path):
    """Create a cleaner, more stylized version"""
    
    # Load data and process (same as above)
    df = pd.read_csv(data_path)
    features_df, numeric_cols = engineer_features(df)
    
    # Apply scaling
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df)
    
    # Apply PCA for cleaner 2D representation
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(features_scaled)
    
    # Find natural clusters
    kmeans = KMeans(n_clusters=8, random_state=42)
    natural_labels = kmeans.fit_predict(features_scaled)
    
    df['pca_x'] = pca_result[:, 0]
    df['pca_y'] = pca_result[:, 1]
    df['natural_cluster'] = natural_labels
    
    # Calculate centroids
    natural_centroids = {}
    for cluster in range(8):
        mask = df['natural_cluster'] == cluster
        natural_centroids[cluster] = {
            'x': df[mask]['pca_x'].mean(),
            'y': df[mask]['pca_y'].mean()
        }
    
    # Calculate genre centroids
    genre_centroids = {}
    for genre in df['genre'].unique():
        genre_data = df[df['genre'] == genre]
        dominant_cluster = genre_data['natural_cluster'].mode()[0]
        
        genre_centroids[genre] = {
            'x': genre_data['pca_x'].mean(),
            'y': genre_data['pca_y'].mean(),
            'cluster': dominant_cluster
        }
    
    # Create cleaner visualization
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor('#fafafa')
    
    # Softer colors
    soft_colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', 
                   '#dda0dd', '#f4a460', '#98d8c8', '#ffcc5c']
    
    # Plot connections first (behind everything)
    for genre, genre_cent in genre_centroids.items():
        nat_cluster = genre_cent['cluster']
        nat_cent = natural_centroids[nat_cluster]
        
        # Simple curved line
        t = np.linspace(0, 1, 100)
        
        # Bezier curve
        ctrl_x = (nat_cent['x'] + genre_cent['x']) / 2
        ctrl_y = (nat_cent['y'] + genre_cent['y']) / 2 + 2  # Add upward curve
        
        x_curve = (1-t)**2 * nat_cent['x'] + 2*(1-t)*t * ctrl_x + t**2 * genre_cent['x']
        y_curve = (1-t)**2 * nat_cent['y'] + 2*(1-t)*t * ctrl_y + t**2 * genre_cent['y']
        
        ax.plot(x_curve, y_curve, color=soft_colors[nat_cluster], 
               alpha=0.3, linewidth=1.5)
    
    # Plot natural centroids
    for cluster, centroid in natural_centroids.items():
        ax.scatter(centroid['x'], centroid['y'], 
                  s=1000, marker='*', c=soft_colors[cluster],
                  edgecolors='darkgray', linewidth=2, zorder=20)
        
        # Add cluster number
        ax.text(centroid['x'], centroid['y'], str(cluster+1), 
               ha='center', va='center', fontsize=12, weight='bold',
               color='white')
    
    # Plot genre centroids
    for genre, genre_cent in genre_centroids.items():
        ax.scatter(genre_cent['x'], genre_cent['y'], 
                  s=120, c='white',
                  edgecolors=soft_colors[genre_cent['cluster']], 
                  linewidth=2, zorder=10)
    
    # Clean styling
    ax.set_xlabel('Principal Component 1', fontsize=14)
    ax.set_ylabel('Principal Component 2', fontsize=14)
    ax.set_title('How 8 Natural Sound Clusters Fragment into 35 Marketing Categories', 
                fontsize=16, weight='bold')
    
    # Minimal grid
    ax.grid(True, alpha=0.2, linestyle=':')
    
    # Remove spines for cleaner look
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')
    
    plt.tight_layout()
    plt.savefig('centroid_divergence_clean.png', dpi=300, bbox_inches='tight')
    plt.close()

# Main execution
if __name__ == "__main__":
    data_path = 'dataset/top100_with_tempogram_nmf_meta.csv'
    
    print("Creating centroid divergence visualization...")
    create_centroid_divergence(data_path)
    
    print("Creating simplified version...")
    create_simplified_divergence(data_path)
    
    print("\nCreated:")
    print("- centroid_divergence.png (detailed version)")
    print("- centroid_divergence_clean.png (simplified version)")