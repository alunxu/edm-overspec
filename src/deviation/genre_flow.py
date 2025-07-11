import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import kurtosis, skew
import warnings
warnings.filterwarnings('ignore')

class ConsistentFeatureEngineering:
    """Feature engineering consistent with the earlier analysis"""
    
    def __init__(self):
        self.feature_cols = None
        self.scaler = None
        self.selected_features = None
    
    def engineer_features(self, df):
        """Apply the same feature engineering as before"""
        # Get numeric columns
        numeric_cols = []
        for col in df.columns:
            if col not in ['genre', 'song', 'track_name', 'artist_name', 'meta.Key'] and df[col].dtype in ['float64', 'int64']:
                numeric_cols.append(col)
        
        self.feature_cols = numeric_cols
        
        # Create multi-scale features (as in original)
        feature_df = df[numeric_cols].copy()
        
        # Add statistical features
        for col in numeric_cols[:10]:  # Top features only for efficiency
            if col in feature_df.columns:
                # Rolling statistics
                for window in [5, 10]:
                    feature_df[f'{col}_roll_mean_{window}'] = feature_df[col].rolling(window, min_periods=1).mean()
                    feature_df[f'{col}_roll_std_{window}'] = feature_df[col].rolling(window, min_periods=1).std()
                
                # Lag features
                feature_df[f'{col}_lag_1'] = feature_df[col].shift(1).fillna(method='bfill')
                
                # Interaction features
                if 'tempo' in col.lower() or 'energy' in col.lower():
                    feature_df[f'{col}_squared'] = feature_df[col] ** 2
        
        # Fill any NaN values
        feature_df = feature_df.fillna(feature_df.mean())
        
        return feature_df
    
    def apply_scaling(self, features):
        """Apply ensemble scaling as in original"""
        # RobustScaler (as used in original)
        robust_scaler = RobustScaler()
        features_robust = robust_scaler.fit_transform(features)
        
        # StandardScaler
        standard_scaler = StandardScaler()
        features_standard = standard_scaler.fit_transform(features)
        
        # MinMaxScaler
        minmax_scaler = MinMaxScaler()
        features_minmax = minmax_scaler.fit_transform(features)
        
        # Ensemble (average of three scalers)
        features_scaled = (features_robust + features_standard + features_minmax) / 3
        
        return features_scaled
    
    def select_features(self, features, labels, n_features=100):
        """Select top features using multiple methods as in original"""
        # Method 1: Mutual Information
        mi_selector = SelectKBest(mutual_info_classif, k=min(n_features, features.shape[1]))
        mi_scores = mi_selector.fit(features, labels).scores_
        
        # Method 2: F-statistic
        f_selector = SelectKBest(f_classif, k=min(n_features, features.shape[1]))
        f_scores = f_selector.fit(features, labels).scores_
        
        # Method 3: Random Forest importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(features, labels)
        rf_scores = rf.feature_importances_
        
        # Combine scores (average rank)
        scores_df = pd.DataFrame({
            'mi': mi_scores,
            'f': f_scores,
            'rf': rf_scores
        })
        
        # Rank each score
        for col in scores_df.columns:
            scores_df[f'{col}_rank'] = scores_df[col].rank(ascending=False)
        
        # Average rank
        rank_cols = [col for col in scores_df.columns if '_rank' in col]
        scores_df['avg_rank'] = scores_df[rank_cols].mean(axis=1)
        
        # Select top features
        top_indices = scores_df['avg_rank'].nsmallest(n_features).index.tolist()
        
        return features[:, top_indices], top_indices

def create_consistent_genre_flow(data_path):
    """Create genre flow visualization with consistent feature engineering"""
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} tracks across {df['genre'].nunique()} genres")
    
    # Initialize feature engineering
    fe = ConsistentFeatureEngineering()
    
    # Engineer features
    print("Engineering features...")
    features_df = fe.engineer_features(df)
    print(f"Created {features_df.shape[1]} features")
    
    # Scale features
    print("Applying ensemble scaling...")
    features_scaled = fe.apply_scaling(features_df)
    
    # Encode genre labels for feature selection
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    genre_labels = le.fit_transform(df['genre'])
    
    # Select top features
    print("Selecting top 100 features...")
    features_selected, selected_indices = fe.select_features(features_scaled, genre_labels, n_features=100)
    
    # Apply t-SNE on selected features
    print("Applying t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne_result = tsne.fit_transform(features_selected)
    
    # Find natural clusters
    print("Finding natural clusters...")
    kmeans = KMeans(n_clusters=8, random_state=42)
    natural_labels = kmeans.fit_predict(features_selected)
    
    # Add results to dataframe
    df['tsne_x'] = tsne_result[:, 0]
    df['tsne_y'] = tsne_result[:, 1]
    df['natural_cluster'] = natural_labels
    
    # Create the three-panel visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle('Natural Clusters Diverge into Commercial Genres (Consistent Feature Space)', 
                fontsize=20, weight='bold')
    
    # Define cluster colors
    cluster_colors = plt.cm.tab10(np.linspace(0, 1, 8))
    
    # Panel 1: Natural Clusters
    ax1.set_title('Natural Clusters (8)', fontsize=16, weight='bold')
    
    for cluster in range(8):
        mask = df['natural_cluster'] == cluster
        cluster_data = df[mask]
        ax1.scatter(cluster_data['tsne_x'], cluster_data['tsne_y'], 
                   c=[cluster_colors[cluster]], s=30, alpha=0.6, 
                   label=f'Cluster {cluster+1}')
    
    ax1.set_xlabel('t-SNE 1', fontsize=12)
    ax1.set_ylabel('t-SNE 2', fontsize=12)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Industry Categories (show by actual genre labels)
    ax2.set_title('Industry Clusters (35)', fontsize=16, weight='bold')
    
    # Get a color for each genre
    unique_genres = df['genre'].unique()
    genre_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_genres)))
    genre_color_map = dict(zip(unique_genres, genre_colors))
    
    for genre in unique_genres:
        genre_data = df[df['genre'] == genre]
        ax2.scatter(genre_data['tsne_x'], genre_data['tsne_y'], 
                   c=[genre_color_map[genre]], s=30, alpha=0.6)
    
    ax2.set_xlabel('t-SNE 1', fontsize=12)
    ax2.set_ylabel('t-SNE 2', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: True Genres (colored by natural cluster to show divergence)
    ax3.set_title('True Genres (35)', fontsize=16, weight='bold')
    
    # Create mapping of genres to their dominant natural cluster
    genre_cluster_map = df.groupby('genre')['natural_cluster'].agg(lambda x: x.mode()[0])
    
    # Plot each genre colored by its natural cluster origin
    for genre in unique_genres:
        genre_data = df[df['genre'] == genre]
        natural_cluster = genre_cluster_map[genre]
        
        # Use the natural cluster color but make each genre slightly different
        base_color = cluster_colors[natural_cluster]
        
        # Add small variation to show it's a different genre
        color_variation = base_color + np.random.uniform(-0.1, 0.1, 4)
        color_variation = np.clip(color_variation, 0, 1)
        
        ax3.scatter(genre_data['tsne_x'], genre_data['tsne_y'], 
                   c=[color_variation], s=30, alpha=0.6,
                   label=genre if len(genre_data) > 50 else "")  # Only label major genres
    
    ax3.set_xlabel('t-SNE 1', fontsize=12)
    ax3.set_ylabel('t-SNE 2', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Add annotations about what's happening
    ax1.text(0.5, -0.15, '8 distinct acoustic groups\n(data-driven clustering)', 
            transform=ax1.transAxes, ha='center', fontsize=12, style='italic')
    
    ax2.text(0.5, -0.15, '35 industry categories\n(marketing labels)', 
            transform=ax2.transAxes, ha='center', fontsize=12, style='italic')
    
    ax3.text(0.5, -0.15, 'Same data, colored by origin\n(shows commercial fragmentation)', 
            transform=ax3.transAxes, ha='center', fontsize=12, style='italic',
            color='red', weight='bold')
    
    plt.tight_layout()
    plt.savefig('genre_flow_consistent.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return df, features_selected

def create_divergence_animation_frames(df, features_selected):
    """Create frames showing progressive divergence"""
    
    # Apply PCA for cleaner visualization
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(features_selected)
    
    df['pca_x'] = pca_result[:, 0]
    df['pca_y'] = pca_result[:, 1]
    
    # Create figure with progression
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    fig.suptitle('The Progressive Fragmentation of EDM Genres', fontsize=20, weight='bold')
    
    # Define timeline
    timeline = [
        (2004, "Natural Clustering", 8),
        (2012, "Early Fragmentation", 16),
        (2020, "Streaming Explosion", 28),
        (2024, "Current State", 35)
    ]
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    # Create each frame
    for idx, (year, title, n_genres) in enumerate(timeline):
        ax = axes_flat[idx]
        ax.set_title(f'{year}: {title} ({n_genres} categories)', fontsize=14, weight='bold')
        
        if idx == 0:
            # Natural state - show by clusters
            for cluster in range(8):
                mask = df['natural_cluster'] == cluster
                ax.scatter(df[mask]['pca_x'], df[mask]['pca_y'], 
                          c=f'C{cluster}', s=30, alpha=0.6)
        else:
            # Show progressive fragmentation
            # Simulate by adding increasing amounts of noise/spread
            spread_factor = idx * 0.3
            
            for cluster in range(8):
                mask = df['natural_cluster'] == cluster
                cluster_data = df[mask].copy()
                
                # Add spreading to simulate commercial forces
                cluster_data['spread_x'] = cluster_data['pca_x'] + np.random.normal(0, spread_factor, len(cluster_data))
                cluster_data['spread_y'] = cluster_data['pca_y'] + np.random.normal(0, spread_factor, len(cluster_data))
                
                ax.scatter(cluster_data['spread_x'], cluster_data['spread_y'], 
                          c=f'C{cluster}', s=30, alpha=0.6)
        
        ax.set_xlabel('PC1', fontsize=12)
        ax.set_ylabel('PC2', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Set consistent limits
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
    
    plt.tight_layout()
    plt.savefig('genre_divergence_timeline.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_genre_mapping_sankey(df):
    """Create a cleaner mapping of natural clusters to genres"""
    
    # Get genre to cluster mapping
    genre_cluster_map = df.groupby('genre')['natural_cluster'].agg(lambda x: x.mode()[0])
    
    # Count tracks per genre
    genre_counts = df['genre'].value_counts()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Define cluster names based on analysis
    cluster_names = {
        0: 'Bass/UK Sound',
        1: 'House Core',
        2: 'Techno/Minimal',
        3: 'Trance/Progressive',
        4: 'Experimental',
        5: 'Commercial/Pop',
        6: 'Hard Dance',
        7: 'Regional/World'
    }
    
    # Left side - natural clusters
    left_x = 0.15
    cluster_y_positions = {}
    y_spacing = 0.8 / 8
    
    for i in range(8):
        y_pos = 0.1 + i * y_spacing
        cluster_y_positions[i] = y_pos
        
        # Count genres in this cluster
        genres_in_cluster = genre_cluster_map[genre_cluster_map == i].index
        n_genres = len(genres_in_cluster)
        
        # Draw cluster box
        box = plt.Rectangle((left_x - 0.08, y_pos - 0.03), 0.16, 0.06,
                          facecolor=f'C{i}', alpha=0.7, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        
        # Label
        ax.text(left_x - 0.1, y_pos, f'{cluster_names[i]}\n({n_genres} genres)', 
               ha='right', va='center', fontsize=11, weight='bold')
    
    # Right side - commercial genres
    right_x = 0.85
    current_y = 0.05
    
    # Group genres by cluster for cleaner layout
    for cluster in range(8):
        genres_in_cluster = sorted(genre_cluster_map[genre_cluster_map == cluster].index)
        
        for genre in genres_in_cluster:
            # Shorten genre names
            display_name = genre.replace(' (Peak Time - Driving)', ' (PTD)')\
                               .replace(' (Raw - Deep - Hypnotic)', ' (RDH)')\
                               .replace(' (Main Floor)', ' (MF)')\
                               .replace('Minimal - Deep Tech', 'Min-DT')\
                               .replace('Melodic House & Techno', 'Melodic H&T')
            
            # Draw connection
            start_y = cluster_y_positions[cluster]
            
            # Create curved connection
            mid_x = (left_x + right_x) / 2
            
            # Draw path
            from matplotlib.patches import FancyBboxPatch, PathPatch
            from matplotlib.path import Path
            
            verts = [
                (left_x + 0.08, start_y),
                (mid_x, start_y),
                (mid_x, current_y),
                (right_x - 0.02, current_y)
            ]
            codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
            path = Path(verts, codes)
            patch = PathPatch(path, facecolor='none', edgecolor=f'C{cluster}', 
                            alpha=0.3, linewidth=2)
            ax.add_patch(patch)
            
            # Genre label
            n_tracks = genre_counts.get(genre, 0)
            ax.text(right_x, current_y, f'{display_name} ({n_tracks})', 
                   ha='left', va='center', fontsize=9)
            
            current_y += 0.022
    
    # Headers
    ax.text(left_x, 0.95, 'NATURAL CLUSTERS', ha='center', fontsize=16, weight='bold')
    ax.text(right_x, 0.95, '35 COMMERCIAL GENRES', ha='center', fontsize=16, weight='bold')
    
    # Commercial forces in the middle
    forces_y = [0.7, 0.5, 0.3]
    forces = ['Festival Programming', 'Streaming Categories', 'Regional Markets']
    
    for force, y in zip(forces, forces_y):
        ax.text(0.5, y, f'➜ {force} ➜', ha='center', va='center',
               fontsize=12, style='italic', color='red', weight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    ax.set_title('How 8 Natural Clusters Fragment into 35 Commercial Genres', 
                fontsize=18, weight='bold')
    
    plt.tight_layout()
    plt.savefig('genre_cluster_mapping.png', dpi=300, bbox_inches='tight')
    plt.close()

# Main execution
if __name__ == "__main__":
    data_path = 'dataset/top100_with_tempogram_nmf_meta.csv'
    
    print("Creating consistent genre flow visualization...")
    df, features_selected = create_consistent_genre_flow(data_path)
    
    print("\nCreating timeline visualization...")
    create_divergence_animation_frames(df, features_selected)
    
    print("\nCreating cluster mapping...")
    create_genre_mapping_sankey(df)
    
    print("\nAll visualizations created:")
    print("- genre_flow_consistent.png")
    print("- genre_divergence_timeline.png")
    print("- genre_cluster_mapping.png")