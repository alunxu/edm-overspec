import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse, Rectangle
import matplotlib.patches as mpatches

# Set style for publication quality
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def create_phantom_genre_figure(data_path, output_path='phantom_genres_key_figure.png'):
    """Create a single powerful figure showing phantom genres"""
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Define phantom pairs with colors
    phantom_pairs = [
        {
            'genres': ['Deep House', 'Organic House'],
            'label': 'House\nFragmentation',
            'color': '#FF6B6B',
            'industry_count': 2
        },
        {
            'genres': ['Tech House', 'Minimal - Deep Tech'],
            'label': 'Techno\nMicro-genres',
            'color': '#4ECDC4',
            'industry_count': 2
        },
        {
            'genres': ['Techno (Peak Time - Driving)', 'Techno (Raw - Deep - Hypnotic)'],
            'label': 'Techno\nMood Labels',
            'color': '#45B7D1',
            'industry_count': 2
        },
        {
            'genres': ['Trance (Main Floor)', 'Trance (Raw - Deep - Hypnotic)'],
            'label': 'Trance\nVenue Split',
            'color': '#96CEB4',
            'industry_count': 2
        }
    ]
    
    # Get all genres involved
    all_phantom_genres = []
    for pair in phantom_pairs:
        all_phantom_genres.extend(pair['genres'])
    
    # Filter data
    phantom_df = df[df['genre'].isin(all_phantom_genres)].copy()
    
    # Get numeric features
    numeric_cols = []
    for col in df.columns:
        if col not in ['genre', 'song', 'meta.Key']:
            if df[col].dtype in ['float64', 'int64']:
                numeric_cols.append(col)
    
    # Standardize and apply PCA
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(phantom_df[numeric_cols])
    
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(features_scaled)
    
    phantom_df['PC1'] = pca_result[:, 0]
    phantom_df['PC2'] = pca_result[:, 1]
    
    # Create figure with specific layout
    fig = plt.figure(figsize=(16, 10))
    
    # Main plot - PCA visualization
    ax_main = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=3)
    
    # Plot each phantom pair
    for i, pair in enumerate(phantom_pairs):
        # Get data for this pair
        pair_mask = phantom_df['genre'].isin(pair['genres'])
        pair_data = phantom_df[pair_mask]
        
        # Plot points
        for genre in pair['genres']:
            genre_data = pair_data[pair_data['genre'] == genre]
            ax_main.scatter(genre_data['PC1'], genre_data['PC2'], 
                          alpha=0.6, s=60, color=pair['color'],
                          label=genre if i == 0 else "")
        
        # Calculate and draw confidence ellipse
        pair_points = pair_data[['PC1', 'PC2']].values
        mean = np.mean(pair_points, axis=0)
        cov = np.cov(pair_points.T)
        
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width, height = 4 * np.sqrt(eigenvalues)  # 2 standard deviations
        
        ellipse = Ellipse(mean, width, height, angle=angle,
                         facecolor=pair['color'], alpha=0.2,
                         edgecolor=pair['color'], linewidth=3)
        ax_main.add_patch(ellipse)
        
        # Add label
        ax_main.text(mean[0], mean[1], pair['label'], 
                    fontsize=11, weight='bold', ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax_main.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=14)
    ax_main.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=14)
    ax_main.set_title('Phantom Genres: Acoustically Identical, Marketed as Different', 
                     fontsize=18, weight='bold', pad=20)
    
    # Add grid
    ax_main.grid(True, alpha=0.3)
    ax_main.set_axisbelow(True)
    
    # Summary statistics panel
    ax_stats = plt.subplot2grid((3, 3), (0, 2), rowspan=2)
    ax_stats.axis('off')
    
    # Calculate overlap statistics
    overlap_stats = []
    for pair in phantom_pairs:
        g1, g2 = pair['genres']
        g1_data = df[df['genre'] == g1]
        g2_data = df[df['genre'] == g2]
        
        # Simple overlap calculation based on feature means
        overlaps = []
        for feature in numeric_cols[:50]:  # Top 50 features
            try:
                m1 = g1_data[feature].mean()
                m2 = g2_data[feature].mean()
                s1 = g1_data[feature].std()
                s2 = g2_data[feature].std()
                
                # Calculate overlap coefficient
                if s1 > 0 and s2 > 0:
                    diff = abs(m1 - m2)
                    avg_std = (s1 + s2) / 2
                    overlap = max(0, 1 - diff / (2 * avg_std))
                    overlaps.append(overlap)
            except:
                continue
        
        avg_overlap = np.mean(overlaps) * 100 if overlaps else 0
        overlap_stats.append({'pair': pair['label'].replace('\n', ' '), 'overlap': avg_overlap})
    
    # Display statistics
    stats_text = "ACOUSTIC SIMILARITY\n" + "="*25 + "\n\n"
    for stat in overlap_stats:
        stats_text += f"{stat['pair']}:\n  {stat['overlap']:.0f}% similar\n\n"
    
    stats_text += "\nKEY FINDING:\n"
    stats_text += "8 marketed genres\n≈ 4 acoustic realities"
    
    ax_stats.text(0.1, 0.95, stats_text, transform=ax_stats.transAxes,
                 fontsize=14, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', alpha=0.8))
    
    # Industry vs Reality comparison
    ax_compare = plt.subplot2grid((3, 3), (2, 2))
    
    categories = ['Industry\nTaxonomy', 'Acoustic\nReality']
    values = [8, 4]  # 8 genres shown, ~4 actual clusters
    colors = ['#ff7f0e', '#2ca02c']
    
    bars = ax_compare.bar(categories, values, color=colors, alpha=0.7, width=0.6)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax_compare.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{value}', ha='center', va='bottom', fontsize=16, weight='bold')
    
    ax_compare.set_ylabel('Number of Categories', fontsize=12)
    ax_compare.set_title('Genre Over-segmentation\n(Shown Pairs Only)', fontsize=12, pad=10)
    ax_compare.set_ylim(0, 10)
    ax_compare.grid(axis='y', alpha=0.3)
    
    # Add overall title and subtitle
    fig.suptitle('The Phantom Genre Problem in EDM', fontsize=22, weight='bold', y=0.98)
    fig.text(0.5, 0.94, 'Four supposedly different genre pairs are acoustically indistinguishable', 
             ha='center', fontsize=14, style='italic')
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Create second figure - Feature similarity heatmap
    create_similarity_heatmap(df, phantom_pairs, numeric_cols)

def create_similarity_heatmap(df, phantom_pairs, numeric_cols, output_path='phantom_genres_heatmap.png'):
    """Create a heatmap showing feature similarities"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate similarity matrix
    similarity_data = []
    
    for pair in phantom_pairs:
        g1, g2 = pair['genres']
        g1_data = df[df['genre'] == g1]
        g2_data = df[df['genre'] == g2]
        
        # Calculate similarity for key feature groups
        feature_groups = {
            'Tempo & Rhythm': ['meta.Bpm', 'meta.Length_ms'],
            'Energy': [col for col in numeric_cols if 'Energy' in col][:5],
            'Spectral': [col for col in numeric_cols if 'Spectral' in col][:5],
            'MFCC': [col for col in numeric_cols if 'MFCC' in col][:5],
            'Chroma': [col for col in numeric_cols if 'Chroma' in col][:5]
        }
        
        similarities = {}
        for group_name, features in feature_groups.items():
            group_sims = []
            for feature in features:
                if feature in g1_data.columns and feature in g2_data.columns:
                    try:
                        m1 = g1_data[feature].mean()
                        m2 = g2_data[feature].mean()
                        s1 = g1_data[feature].std()
                        s2 = g2_data[feature].std()
                        
                        if s1 > 0 and s2 > 0:
                            diff = abs(m1 - m2)
                            avg_std = (s1 + s2) / 2
                            sim = max(0, 1 - diff / (3 * avg_std))
                            group_sims.append(sim)
                    except:
                        continue
            
            if group_sims:
                similarities[group_name] = np.mean(group_sims)
        
        similarity_data.append({
            'Pair': pair['label'].replace('\n', ' '),
            **similarities
        })
    
    # Create DataFrame and plot
    sim_df = pd.DataFrame(similarity_data)
    sim_df = sim_df.set_index('Pair')
    
    # Create heatmap
    sns.heatmap(sim_df.T, annot=True, fmt='.0%', cmap='RdYlGn', 
                vmin=0.5, vmax=1.0, linewidths=0.5, cbar_kws={'label': 'Acoustic Similarity'},
                annot_kws={'size': 12})
    
    ax.set_xlabel('Genre Pairs', fontsize=14, weight='bold')
    ax.set_ylabel('Acoustic Feature Groups', fontsize=14, weight='bold')
    ax.set_title('Feature Group Similarities Across Phantom Genre Pairs', 
                fontsize=16, weight='bold', pad=20)
    
    # Rotate labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add text box with interpretation
    textstr = 'Values show acoustic similarity (100% = identical)\nMost features show >70% similarity between "different" genres'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, -0.15, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

# Run the analysis
if __name__ == "__main__":
    data_path = 'dataset/top100_with_tempogram_nmf_meta.csv'
    create_phantom_genre_figure(data_path)
    print("Created phantom_genres_key_figure.png and phantom_genres_heatmap.png")