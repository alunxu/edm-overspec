"""
final_phantom_visualization.py
==============================
Simplified two-panel visualization with color coding for genre families
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def create_genre_color_mapping(unique_genres):
    """Create color mapping for genre families."""
    
    # Define genre families and their colors
    genre_families = {
        'House': {
            'color': '#FF6B6B',  # Red shades
            'genres': ['House', 'Deep House', 'Tech House', 'Progressive House', 
                      'Funky House', 'Jackin House', 'Afro House', 'Bass House',
                      'Organic House', 'Melodic House & Techno']
        },
        'Techno': {
            'color': '#4ECDC4',  # Teal shades
            'genres': ['Techno (Peak Time - Driving)', 'Techno (Raw - Deep - Hypnotic)',
                      'Hard Techno', 'Melodic House & Techno']  # Note: MH&T bridges both
        },
        'Trance': {
            'color': '#45B7D1',  # Blue shades
            'genres': ['Trance (Main Floor)', 'Trance (Raw - Deep - Hypnotic)', 
                      'Psy-Trance']
        },
        'Bass Music': {
            'color': '#96CEB4',  # Green shades
            'genres': ['Drum & Bass', 'Dubstep', '140 - Deep Dubstep - Grime',
                      'UK Garage - Bassline', 'Bass - Club', 'Trap - Future Bass']
        },
        'Breaks': {
            'color': '#DDA0DD',  # Plum
            'genres': ['Breaks - Breakbeat - UK Bass']
        },
        'Dance/Pop': {
            'color': '#FFB6C1',  # Light pink
            'genres': ['Dance - Pop', 'Mainstage', 'Indie Dance']
        },
        'Underground': {
            'color': '#9370DB',  # Purple
            'genres': ['Minimal - Deep Tech', 'Downtempo', 'Nu Disco - Disco']
        },
        'Hard Styles': {
            'color': '#FF8C00',  # Dark orange
            'genres': ['Hard Dance - Hardcore - Neo Rave']
        },
        'Other': {
            'color': '#A9A9A9',  # Gray
            'genres': ['Ambient - Experimental', 'Electronica', 'DJ Tools',
                      'Electro (Classic - Detroit - Modern)', 'Amapiano']
        }
    }
    
    # Create color mapping
    color_map = {}
    family_map = {}
    
    for family, info in genre_families.items():
        base_color = info['color']
        for genre in info['genres']:
            color_map[genre] = base_color
            family_map[genre] = family
    
    # Handle any missing genres
    for genre in unique_genres:
        if genre not in color_map:
            color_map[genre] = genre_families['Other']['color']
            family_map[genre] = 'Other'
    
    return color_map, family_map, genre_families


def calculate_co_clustering_matrix(genres, clusters, unique_genres):
    """Calculate co-clustering frequency matrix."""
    
    n_genres = len(unique_genres)
    genre_to_idx = {genre: i for i, genre in enumerate(unique_genres)}
    
    # Initialize matrices
    co_cluster_counts = np.zeros((n_genres, n_genres))
    genre_counts = np.zeros(n_genres)
    
    # Count genre occurrences
    for genre in genres:
        genre_counts[genre_to_idx[genre]] += 1
    
    # Count co-occurrences by cluster
    unique_clusters = np.unique(clusters)
    
    for cluster in unique_clusters:
        cluster_mask = clusters == cluster
        cluster_genres = genres[cluster_mask]
        
        # Count each genre in this cluster
        cluster_genre_counts = Counter(cluster_genres)
        
        # Update co-occurrence matrix
        for g1, count1 in cluster_genre_counts.items():
            idx1 = genre_to_idx[g1]
            for g2, count2 in cluster_genre_counts.items():
                idx2 = genre_to_idx[g2]
                if g1 != g2:
                    co_cluster_counts[idx1, idx2] += min(count1, count2)
    
    # Convert to similarity matrix
    similarity_matrix = np.zeros((n_genres, n_genres))
    
    for i in range(n_genres):
        for j in range(n_genres):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                union = genre_counts[i] + genre_counts[j] - co_cluster_counts[i, j]
                if union > 0:
                    similarity_matrix[i, j] = co_cluster_counts[i, j] / union
    
    return pd.DataFrame(similarity_matrix, index=unique_genres, columns=unique_genres)


def create_final_visualization():
    """Create the final two-panel visualization."""
    
    print("="*70)
    print("PHANTOM GENRE ANALYSIS - FINAL VISUALIZATION")
    print("="*70)
    
    # Load data
    csv_path = 'results/csv/edm_clustering_summary.csv'
    clustering_df = pd.read_csv(csv_path)
    
    genres = clustering_df['true_genre'].values
    natural_clusters = clustering_df['natural_cluster'].values
    
    unique_genres = sorted(np.unique(genres))
    color_map, family_map, genre_families = create_genre_color_mapping(unique_genres)
    
    # Calculate co-clustering matrix
    print("\nCalculating genre relationships...")
    co_cluster_sim = calculate_co_clustering_matrix(genres, natural_clusters, unique_genres)
    
    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9), 
                                   gridspec_kw={'width_ratios': [1, 1.2]})
    
    # Panel 1: Genre Coherence Scatter Plot
    create_coherence_scatter(ax1, genres, natural_clusters, unique_genres, 
                           color_map, family_map, genre_families)
    
    # Panel 2: Co-clustering Matrix
    create_clustering_matrix(ax2, co_cluster_sim, unique_genres, 
                           color_map, family_map)
    
    # Add main title
    fig.suptitle('EDM Phantom Genre Analysis', fontsize=16, weight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig('plots/phantom_genre_final.png', dpi=300, bbox_inches='tight')
    print("\nSaved: plots/phantom_genre_final.png")
    plt.show()
    
    # Create summary report
    create_summary_report(co_cluster_sim, unique_genres, family_map)


def create_coherence_scatter(ax, genres, clusters, unique_genres, color_map, family_map, genre_families):
    """Create scatter plot showing genre coherence."""
    
    # Calculate fragmentation and concentration
    fragmentation_scores = []
    concentration_scores = []
    genre_sizes = []
    colors = []
    
    for genre in unique_genres:
        genre_mask = genres == genre
        genre_clusters = clusters[genre_mask]
        
        # Count occurrences
        cluster_counts = Counter(genre_clusters)
        n_clusters = len(cluster_counts)
        total = sum(cluster_counts.values())
        
        # Max concentration
        max_concentration = max(cluster_counts.values()) / total if total > 0 else 0
        
        fragmentation_scores.append(n_clusters)
        concentration_scores.append(max_concentration)
        genre_sizes.append(total)
        colors.append(color_map[genre])
    
    # Create scatter plot
    scatter = ax.scatter(fragmentation_scores, concentration_scores,
                        s=[s*1.5 for s in genre_sizes],  # Smaller circles
                        c=colors,
                        alpha=0.7,
                        edgecolors='black',
                        linewidth=0.8)
    
    # Add genre labels for important points with smart positioning
    for i, genre in enumerate(unique_genres):
        # Only label the most extreme cases to avoid clutter
        if (fragmentation_scores[i] >= 6 or 
            concentration_scores[i] <= 0.45 or
            (fragmentation_scores[i] <= 3 and concentration_scores[i] >= 0.88) or
            genre in ['Bass - Club', 'Amapiano', 'Ambient - Experimental']):  # Always show phantom examples
            
            # Determine label position based on location
            if fragmentation_scores[i] < 4:
                ha = 'right'
                offset_x = -8
            else:
                ha = 'left'
                offset_x = 8
            
            # Vertical offset based on density
            if concentration_scores[i] > 0.7:
                offset_y = -5
            elif concentration_scores[i] < 0.4:
                offset_y = 5
            else:
                offset_y = 0
            
            ax.annotate(genre, 
                       (fragmentation_scores[i], concentration_scores[i]),
                       xytext=(offset_x, offset_y), 
                       textcoords='offset points',
                       fontsize=9,
                       ha=ha,
                       bbox=dict(boxstyle='round,pad=0.3', 
                               facecolor=colors[i], 
                               alpha=0.3,
                               edgecolor='none'),
                       arrowprops=dict(arrowstyle='-', 
                                     color=colors[i], 
                                     alpha=0.5,
                                     lw=1))
    
    # Add quadrant lines
    ax.axvline(x=3, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax.axhline(y=0.6, color='red', linestyle='--', alpha=0.5, linewidth=2)
    
    # Add quadrant labels
    ax.text(1.5, 0.9, 'Coherent\nGenres', ha='center', va='center',
            fontsize=11, alpha=0.7, weight='bold')
    ax.text(6, 0.9, 'Broad\nCategories', ha='center', va='center',
            fontsize=11, alpha=0.7, weight='bold')
    ax.text(1.5, 0.4, 'Split\nGenres', ha='center', va='center',
            fontsize=11, alpha=0.7, weight='bold')
    ax.text(6, 0.4, 'Phantom\nGenres', ha='center', va='center',
            fontsize=11, alpha=0.7, weight='bold', color='red')
    
    ax.set_xlabel('Number of Clusters (Fragmentation)')
    ax.set_ylabel('Concentration in Primary Cluster')
    ax.set_title('Genre Coherence Analysis', fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(2.5, max(fragmentation_scores) + 0.8)
    ax.set_ylim(0.25, 0.92)
    
    # Create legend for genre families
    legend_elements = []
    for family, info in sorted(genre_families.items()):
        if family in ['House', 'Techno', 'Trance', 'Bass Music', 'Other']:  # Main families
            legend_elements.append(
                mpatches.Patch(color=info['color'], label=family, alpha=0.7)
            )
    
    ax.legend(handles=legend_elements, loc='lower left', 
             title='Genre Families', framealpha=0.9)


def create_clustering_matrix(ax, similarity_matrix, unique_genres, color_map, family_map):
    """Create co-clustering matrix with color-coded labels."""
    
    # Hierarchical clustering for ordering
    distance_matrix = 1 - similarity_matrix.values
    condensed_dist = squareform(distance_matrix)
    linkage_matrix = linkage(condensed_dist, method='average')
    dendro = dendrogram(linkage_matrix, no_plot=True)
    order = dendro['leaves']
    
    # Reorder everything
    ordered_genres = [unique_genres[i] for i in order]
    ordered_sim = similarity_matrix.iloc[order, order]
    
    # Create heatmap
    im = ax.imshow(ordered_sim, cmap='RdBu_r', aspect='auto', 
                   vmin=0, vmax=1, interpolation='nearest')
    
    # Set ticks and labels
    ax.set_xticks(range(len(ordered_genres)))
    ax.set_yticks(range(len(ordered_genres)))
    ax.set_xticklabels(ordered_genres, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(ordered_genres, fontsize=10)
    
    # Color the labels according to genre family
    for i, (xtick, ytick) in enumerate(zip(ax.get_xticklabels(), ax.get_yticklabels())):
        genre = ordered_genres[i]
        xtick.set_color(color_map[genre])
        xtick.set_weight('bold')
        ytick.set_color(color_map[genre])
        ytick.set_weight('bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Co-clustering Frequency', fontsize=11)
    
    ax.set_title('Genre Co-clustering Matrix', fontsize=14, weight='bold')


def create_summary_report(co_cluster_sim, unique_genres, family_map):
    """Create a summary report of phantom genres."""
    
    output_dir = 'results/phantom_analysis_final'
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Find strong phantom pairs
    phantom_pairs = []
    
    for i, g1 in enumerate(unique_genres):
        for j, g2 in enumerate(unique_genres):
            if i < j:
                similarity = co_cluster_sim.loc[g1, g2]
                if similarity > 0.7:  # High co-clustering threshold
                    same_family = family_map[g1] == family_map[g2]
                    phantom_pairs.append({
                        'genre1': g1,
                        'genre2': g2,
                        'similarity': similarity,
                        'same_family': same_family,
                        'family': family_map[g1] if same_family else f"{family_map[g1]}-{family_map[g2]}"
                    })
    
    phantom_df = pd.DataFrame(phantom_pairs)
    phantom_df = phantom_df.sort_values('similarity', ascending=False)
    
    # Save results
    phantom_df.to_csv(f'{output_dir}/phantom_genre_pairs.csv', index=False)
    
    # Create text summary
    with open(f'{output_dir}/phantom_genre_summary.txt', 'w') as f:
        f.write("PHANTOM GENRE ANALYSIS SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Total genre pairs with >70% co-clustering: {len(phantom_df)}\n")
        f.write(f"Within same family: {sum(phantom_df['same_family'])}\n")
        f.write(f"Cross-family phantoms: {sum(~phantom_df['same_family'])}\n\n")
        
        f.write("TOP PHANTOM GENRE PAIRS:\n")
        f.write("-"*60 + "\n")
        for _, row in phantom_df.head(15).iterrows():
            f.write(f"{row['genre1']} ↔ {row['genre2']}: {row['similarity']:.3f}")
            if row['same_family']:
                f.write(f" (both {row['family']})")
            f.write("\n")
        
        # Identify most problematic genres
        f.write("\n\nMOST FRAGMENTED GENRES:\n")
        f.write("-"*60 + "\n")
        fragmentation_count = {}
        for _, row in phantom_df.iterrows():
            for genre in [row['genre1'], row['genre2']]:
                fragmentation_count[genre] = fragmentation_count.get(genre, 0) + 1
        
        for genre, count in sorted(fragmentation_count.items(), 
                                  key=lambda x: x[1], reverse=True)[:10]:
            f.write(f"{genre}: appears in {count} phantom pairs\n")
    
    print(f"\nSummary saved to: {output_dir}/")


if __name__ == "__main__":
    create_final_visualization()