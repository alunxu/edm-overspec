import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
from matplotlib.lines import Line2D

def create_phantom_pairs_focus(data_path):
    """Create a visualization focused specifically on the phantom pairs"""
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Get numeric features
    numeric_cols = []
    for col in df.columns:
        if col not in ['genre', 'song', 'meta.Key'] and df[col].dtype in ['float64', 'int64']:
            numeric_cols.append(col)
    
    # Define phantom pairs
    phantom_pairs = [
        {
            'genre1': 'Deep House',
            'genre2': 'Organic House',
            'color': '#e74c3c',
            'position': 0
        },
        {
            'genre1': 'Tech House',
            'genre2': 'Minimal - Deep Tech',
            'color': '#3498db',
            'position': 1
        },
        {
            'genre1': 'Techno (Peak Time - Driving)',
            'genre2': 'Techno (Raw - Deep - Hypnotic)',
            'color': '#2ecc71',
            'position': 2
        },
        {
            'genre1': 'Trance (Main Floor)',
            'genre2': 'Trance (Raw - Deep - Hypnotic)',
            'color': '#9b59b6',
            'position': 3
        }
    ]
    
    # Get genre features
    genre_features = df.groupby('genre')[numeric_cols].mean()
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(genre_features)
    
    # Apply PCA
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(features_scaled)
    
    # Calculate similarities
    similarity_matrix = cosine_similarity(features_scaled)
    
    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
    fig.suptitle('Phantom Genres: High Similarity Despite Different Names', 
                fontsize=20, weight='bold')
    
    # Left panel - PCA space with only phantom pairs
    ax1.set_title('Phantom Pairs in Acoustic Space', fontsize=16, weight='bold')
    
    # Plot all genres as background (gray)
    for i, genre in enumerate(genre_features.index):
        if not any(genre in [p['genre1'], p['genre2']] for p in phantom_pairs):
            ax1.scatter(pca_result[i, 0], pca_result[i, 1], 
                       s=100, c='lightgray', alpha=0.5, edgecolors='gray')
    
    # Plot phantom pairs prominently
    for pair in phantom_pairs:
        if pair['genre1'] in genre_features.index and pair['genre2'] in genre_features.index:
            idx1 = genre_features.index.get_loc(pair['genre1'])
            idx2 = genre_features.index.get_loc(pair['genre2'])
            
            # Plot points
            ax1.scatter(pca_result[idx1, 0], pca_result[idx1, 1], 
                       s=300, c=pair['color'], edgecolors='black', linewidth=2,
                       marker='o', label=pair['genre1'])
            ax1.scatter(pca_result[idx2, 0], pca_result[idx2, 1], 
                       s=300, c=pair['color'], edgecolors='black', linewidth=2,
                       marker='s', label=pair['genre2'])
            
            # Draw connecting line
            ax1.plot([pca_result[idx1, 0], pca_result[idx2, 0]], 
                    [pca_result[idx1, 1], pca_result[idx2, 1]], 
                    color=pair['color'], linewidth=3, alpha=0.8, linestyle='--')
            
            # Add similarity at midpoint
            sim = similarity_matrix[idx1, idx2]
            mid_x = (pca_result[idx1, 0] + pca_result[idx2, 0]) / 2
            mid_y = (pca_result[idx1, 1] + pca_result[idx2, 1]) / 2
            ax1.text(mid_x, mid_y, f'{sim:.0%}', 
                    ha='center', va='center', fontsize=12, weight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             edgecolor=pair['color'], linewidth=2))
            
            # Add labels
            for idx, label in [(idx1, pair['genre1']), (idx2, pair['genre2'])]:
                display_label = label.replace(' (Peak Time - Driving)', ' (PTD)')\
                                    .replace(' (Raw - Deep - Hypnotic)', ' (RDH)')\
                                    .replace(' (Main Floor)', ' (MF)')\
                                    .replace('Minimal - Deep Tech', 'Min-DT')
                ax1.annotate(display_label, 
                           (pca_result[idx, 0], pca_result[idx, 1]),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=11, weight='bold')
    
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=14)
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Right panel - Direct comparison bars
    ax2.set_title('Acoustic Similarity of "Different" Genres', fontsize=16, weight='bold')
    
    y_positions = []
    similarities = []
    colors = []
    labels = []
    
    for i, pair in enumerate(phantom_pairs):
        if pair['genre1'] in genre_features.index and pair['genre2'] in genre_features.index:
            idx1 = genre_features.index.get_loc(pair['genre1'])
            idx2 = genre_features.index.get_loc(pair['genre2'])
            sim = similarity_matrix[idx1, idx2] * 100
            
            y_positions.append(i)
            similarities.append(sim)
            colors.append(pair['color'])
            
            # Create label
            g1_short = pair['genre1'].replace(' (Peak Time - Driving)', ' (PTD)')\
                                    .replace(' (Raw - Deep - Hypnotic)', ' (RDH)')\
                                    .replace(' (Main Floor)', ' (MF)')
            g2_short = pair['genre2'].replace(' (Peak Time - Driving)', ' (PTD)')\
                                    .replace(' (Raw - Deep - Hypnotic)', ' (RDH)')\
                                    .replace(' (Main Floor)', ' (MF)')\
                                    .replace('Minimal - Deep Tech', 'Min-DT')
            labels.append(f"{g1_short}\nvs\n{g2_short}")
    
    # Create horizontal bars
    bars = ax2.barh(y_positions, similarities, color=colors, 
                    edgecolor='black', linewidth=2, alpha=0.8)
    
    # Add value labels
    for bar, sim in zip(bars, similarities):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{sim:.0f}%', ha='left', va='center',
                fontsize=14, weight='bold')
    
    # Add reference lines
    ax2.axvline(x=50, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    ax2.axvline(x=70, color='green', linestyle='--', alpha=0.5, linewidth=2)
    ax2.text(50, -0.5, '50%', ha='center', fontsize=10, color='gray')
    ax2.text(70, -0.5, '70%', ha='center', fontsize=10, color='green')
    
    # Formatting
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(labels, fontsize=12)
    ax2.set_xlabel('Acoustic Similarity (%)', fontsize=14, weight='bold')
    ax2.set_xlim(0, 100)
    ax2.grid(axis='x', alpha=0.3)
    
    # Add insight box
    ax2.text(0.98, 0.98, 
            'All phantom pairs show\n>60% acoustic similarity\n\n' +
            'If these were truly\ndifferent genres, we\'d\nexpect <30% similarity',
            transform=ax2.transAxes, ha='right', va='top',
            fontsize=11, style='italic',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', 
                     edgecolor='darkred', linewidth=2))
    
    plt.tight_layout()
    plt.savefig('phantom_pairs_focus.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_network_visualization(data_path):
    """Create a network-style visualization of genre relationships"""
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Get numeric features
    numeric_cols = []
    for col in df.columns:
        if col not in ['genre', 'song', 'meta.Key'] and df[col].dtype in ['float64', 'int64']:
            numeric_cols.append(col)
    
    # Get genre features
    genre_features = df.groupby('genre')[numeric_cols].mean()
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(genre_features)
    
    # Calculate similarities
    similarity_matrix = cosine_similarity(features_scaled)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 14))
    
    # Use force-directed layout simulation
    # Position genres in a circle first
    n_genres = len(genre_features)
    angles = np.linspace(0, 2*np.pi, n_genres, endpoint=False)
    positions = {}
    
    for i, genre in enumerate(genre_features.index):
        x = np.cos(angles[i]) * 5
        y = np.sin(angles[i]) * 5
        positions[genre] = (x, y)
    
    # Draw all genres as nodes
    genre_families = {
        'House': '#e74c3c',
        'Techno': '#3498db',
        'Trance': '#9b59b6',
        'Bass': '#2ecc71',
        'Other': '#95a5a6'
    }
    
    for i, genre in enumerate(genre_features.index):
        # Determine color
        color = '#95a5a6'  # default
        for family, fam_color in genre_families.items():
            if family.lower() in genre.lower():
                color = fam_color
                break
        
        x, y = positions[genre]
        ax.scatter(x, y, s=500, c=color, edgecolors='black', 
                  linewidth=2, alpha=0.8, zorder=10)
        
        # Add label
        display_name = genre.replace(' (Peak Time - Driving)', ' (PTD)')\
                           .replace(' (Raw - Deep - Hypnotic)', ' (RDH)')\
                           .replace(' (Main Floor)', ' (MF)')\
                           .replace('Minimal - Deep Tech', 'Min-DT')\
                           .replace('Breaks - Breakbeat - UK Bass', 'Breaks')\
                           .replace('140 - Deep Dubstep - Grime', '140/Grime')\
                           .replace('Electro (Classic - Detroit - Modern)', 'Electro')\
                           .replace('UK Garage - Bassline', 'UK Garage')\
                           .replace('Melodic House & Techno', 'Melodic H&T')
        
        # Adjust label position based on angle
        angle = angles[i]
        if -np.pi/4 <= angle <= np.pi/4:  # Right side
            ha = 'left'
            offset = (10, 0)
        elif 3*np.pi/4 <= angle <= 5*np.pi/4:  # Left side
            ha = 'right'
            offset = (-10, 0)
        elif np.pi/4 < angle < 3*np.pi/4:  # Top
            ha = 'center'
            offset = (0, 10)
        else:  # Bottom
            ha = 'center'
            offset = (0, -10)
        
        ax.annotate(display_name, (x, y), xytext=offset, 
                   textcoords='offset points', ha=ha, va='center',
                   fontsize=9, weight='bold')
    
    # Draw connections for high similarity (>70%)
    threshold = 0.7
    for i in range(n_genres):
        for j in range(i+1, n_genres):
            if similarity_matrix[i, j] > threshold:
                g1 = genre_features.index[i]
                g2 = genre_features.index[j]
                x1, y1 = positions[g1]
                x2, y2 = positions[g2]
                
                # Different line styles for phantom pairs
                phantom_pairs = [
                    ('Deep House', 'Organic House'),
                    ('Tech House', 'Minimal - Deep Tech'),
                    ('Techno (Peak Time - Driving)', 'Techno (Raw - Deep - Hypnotic)'),
                    ('Trance (Main Floor)', 'Trance (Raw - Deep - Hypnotic)')
                ]
                
                is_phantom = (g1, g2) in phantom_pairs or (g2, g1) in phantom_pairs
                
                if is_phantom:
                    ax.plot([x1, x2], [y1, y2], 'r-', linewidth=3, alpha=0.8, zorder=5)
                else:
                    ax.plot([x1, x2], [y1, y2], 'gray', linewidth=1, alpha=0.3, zorder=1)
    
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    
    ax.set_title('EDM Genre Network: Red Lines Show Phantom Pairs (>70% Similar)', 
                fontsize=18, weight='bold', pad=20)
    
    # Add legend
    legend_elements = [
        mpatches.Circle((0, 0), 0.5, fc='#e74c3c', label='House Family'),
        mpatches.Circle((0, 0), 0.5, fc='#3498db', label='Techno Family'),
        mpatches.Circle((0, 0), 0.5, fc='#9b59b6', label='Trance Family'),
        mpatches.Circle((0, 0), 0.5, fc='#2ecc71', label='Bass Family'),
        Line2D([0], [0], color='red', linewidth=3, label='Phantom Pairs'),
        Line2D([0], [0], color='gray', linewidth=1, label='Other High Similarity')
    ]
    ax.legend(handles=legend_elements, loc='center', fontsize=11,
             bbox_to_anchor=(0.5, -0.05), ncol=3)
    
    plt.tight_layout()
    plt.savefig('genre_network.png', dpi=300, bbox_inches='tight')
    plt.close()

# Main execution
if __name__ == "__main__":
    data_path = 'dataset/top100_with_tempogram_nmf_meta.csv'
    
    print("Creating phantom pairs focused visualization...")
    create_phantom_pairs_focus(data_path)
    
    print("Creating network visualization...")
    create_network_visualization(data_path)
    
    print("\nGenerated:")
    print("- phantom_pairs_focus.png")
    print("- genre_network.png")