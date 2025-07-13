"""
edm_taxonomy_sunburst.py
========================
Create a sunburst diagram for EDM genre taxonomy with proportional sizing.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Wedge, Circle
import pickle
from scipy.cluster.hierarchy import linkage, fcluster
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import colorsys


def create_sunburst_visualization():
    """Create a sunburst diagram showing genre hierarchy."""
    
    print("="*70)
    print("EDM TAXONOMY SUNBURST VISUALIZATION")
    print("="*70)
    
    # Load data
    feature_matrix_path = 'results/pkl/feature_matrix.pkl'
    with open(feature_matrix_path, 'rb') as f:
        data = pickle.load(f)
    
    X = data['X_selected']
    y = data['genre_labels']
    
    # Get genre information
    unique_genres = np.unique(y)
    genre_counts = {}
    for genre in unique_genres:
        mask = y == genre
        genre_counts[genre] = mask.sum()
    
    # Perform clustering
    genre_centroids = {}
    for genre in unique_genres:
        mask = y == genre
        genre_centroids[genre] = X[mask].mean(axis=0)
    
    centroid_matrix = np.array([genre_centroids[g] for g in unique_genres])
    linkage_matrix = linkage(centroid_matrix, method='ward')
    
    # Get clusters
    clusters_8 = fcluster(linkage_matrix, 8, criterion='maxclust')
    clusters_15 = fcluster(linkage_matrix, 15, criterion='maxclust')
    
    # Load phantom information
    phantom_pairs_path = 'phantom_analysis_results/all_genre_pairs_analysis.csv'
    df_pairs = pd.read_csv(phantom_pairs_path)
    phantoms = df_pairs[(df_pairs['co_clustering'] > 0.8) & 
                       (df_pairs['centroid_distance'] < 10)]
    phantom_set = set()
    for _, row in phantoms.iterrows():
        phantom_set.add(row['genre1'])
        phantom_set.add(row['genre2'])
    
    # Create smart abbreviations
    genre_display = create_smart_genre_names(unique_genres)
    
    # Create the sunburst
    create_sunburst(unique_genres, clusters_8, clusters_15, 
                    genre_counts, phantom_set, genre_display)


def create_smart_genre_names(genres):
    """Create readable short versions of genre names."""
    display = {}
    
    for genre in genres:
        # Handle specific long genre names
        if genre == "Hard Dance - Hardcore - Neo Rave":
            display[genre] = "Hard Dance/HC"
        elif genre == "140 - Deep Dubstep - Grime":
            display[genre] = "140/Dubstep"
        elif genre == "Trap - Future Bass":
            display[genre] = "Trap/Future"
        elif genre == "Minimal - Deep Tech":
            display[genre] = "Minimal/Tech"
        elif genre == "Techno (Peak Time - Driving)":
            display[genre] = "Peak Techno"
        elif genre == "Techno (Melodic)":
            display[genre] = "Melodic Tech"
        elif genre == "Trance (Main Floor)":
            display[genre] = "Main Trance"
        elif genre == "Trance (Raw - Deep - Hypnotic)":
            display[genre] = "Deep Trance"
        elif genre == "Electro (Classic - Detroit - Modern)":
            display[genre] = "Electro"
        elif genre == "Bass - Club":
            display[genre] = "Bass Club"
        elif genre == "Breaks - Breakbeat - UK Bass":
            display[genre] = "Breaks/UK"
        elif genre == "Dance - Pop":
            display[genre] = "Dance Pop"
        elif genre == "Melodic House & Techno":
            display[genre] = "Melodic H&T"
        elif genre == "Progressive House":
            display[genre] = "Prog House"
        elif genre == "Organic House":
            display[genre] = "Organic Hse"
        elif genre == "UK Garage - Bassline":
            display[genre] = "UK Garage"
        elif genre == "Drum & Bass":
            display[genre] = "D&B"
        else:
            # Keep short names as is
            display[genre] = genre
    
    return display


def create_sunburst(genres, family_clusters, acoustic_clusters, 
                    genre_counts, phantom_set, genre_display):
    """Create a sunburst diagram with three rings."""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 16))
    ax.set_aspect('equal')
    
    # Title
    ax.text(0.5, 1.05, 'EDM Genre Taxonomy Sunburst', 
            transform=ax.transAxes, ha='center', fontsize=24, weight='bold')
    ax.text(0.5, 1.02, 'Size proportional to track counts | Inner: Families | Middle: Acoustic | Outer: Genres',
            transform=ax.transAxes, ha='center', fontsize=14, style='italic')
    
    # Color scheme for families
    family_colors = {
        1: '#FF6B6B',  # Red
        2: '#4ECDC4',  # Teal
        3: '#45B7D1',  # Blue
        4: '#96CEB4',  # Green
        5: '#FECA57',  # Yellow
        6: '#DDA0DD',  # Plum
        7: '#98D8C8',  # Mint
        8: '#F8B500'   # Orange
    }
    
    # Calculate total tracks
    total_tracks = sum(genre_counts.values())
    
    # Ring radii
    inner_radius = 0.2
    middle_radius = 0.5
    outer_radius = 0.9
    
    # Build hierarchy data structure
    hierarchy = build_hierarchy(genres, family_clusters, acoustic_clusters, genre_counts)
    
    # Draw inner ring (families)
    family_angles = {}
    current_angle = 0
    
    for family_id in sorted(hierarchy.keys()):
        family_data = hierarchy[family_id]
        family_tracks = family_data['tracks']
        
        # Calculate angle proportional to tracks
        angle_span = (family_tracks / total_tracks) * 360
        
        # Draw family wedge
        wedge = Wedge((0, 0), middle_radius, current_angle, current_angle + angle_span,
                     width=middle_radius - inner_radius,
                     facecolor=family_colors[family_id], 
                     edgecolor='white', linewidth=2)
        ax.add_patch(wedge)
        
        # Add family label
        mid_angle = current_angle + angle_span / 2
        label_radius = (inner_radius + middle_radius) / 2
        x = label_radius * np.cos(np.radians(mid_angle))
        y = label_radius * np.sin(np.radians(mid_angle))
        
        # Rotate text for readability
        rotation = mid_angle - 90 if 90 < mid_angle < 270 else mid_angle + 90
        ha = 'center'
        
        ax.text(x, y, f'Family {family_id}\n{family_tracks}', 
               ha=ha, va='center', rotation=rotation,
               fontsize=10, weight='bold')
        
        family_angles[family_id] = (current_angle, current_angle + angle_span)
        current_angle += angle_span
    
    # Draw middle ring (acoustic clusters)
    acoustic_angles = {}
    
    for family_id in sorted(hierarchy.keys()):
        family_start, family_end = family_angles[family_id]
        family_data = hierarchy[family_id]
        
        # Calculate positions for acoustic clusters within family
        family_angle_span = family_end - family_start
        acoustic_current = family_start
        
        for acoustic_id in sorted(family_data['clusters'].keys()):
            cluster_data = family_data['clusters'][acoustic_id]
            cluster_tracks = cluster_data['tracks']
            
            # Proportional angle within family
            cluster_proportion = cluster_tracks / family_data['tracks']
            cluster_angle_span = cluster_proportion * family_angle_span
            
            # Draw acoustic cluster wedge with lighter color
            cluster_color = lighten_color(family_colors[family_id], 0.3)
            wedge = Wedge((0, 0), outer_radius, acoustic_current, 
                         acoustic_current + cluster_angle_span,
                         width=outer_radius - middle_radius,
                         facecolor=cluster_color, 
                         edgecolor='white', linewidth=1.5)
            ax.add_patch(wedge)
            
            # Add acoustic cluster label (if large enough)
            if cluster_angle_span > 5:  # Only label if wedge is big enough
                mid_angle = acoustic_current + cluster_angle_span / 2
                label_radius = (middle_radius + outer_radius) / 2
                x = label_radius * np.cos(np.radians(mid_angle))
                y = label_radius * np.sin(np.radians(mid_angle))
                
                rotation = mid_angle - 90 if 90 < mid_angle < 270 else mid_angle + 90
                ax.text(x, y, f'A{acoustic_id}', 
                       ha='center', va='center', rotation=rotation,
                       fontsize=8, weight='bold')
            
            acoustic_angles[(family_id, acoustic_id)] = (acoustic_current, 
                                                         acoustic_current + cluster_angle_span)
            acoustic_current += cluster_angle_span
    
    # Draw outer ring (genres)
    for family_id in sorted(hierarchy.keys()):
        family_data = hierarchy[family_id]
        
        for acoustic_id in sorted(family_data['clusters'].keys()):
            cluster_data = family_data['clusters'][acoustic_id]
            cluster_start, cluster_end = acoustic_angles[(family_id, acoustic_id)]
            
            # Calculate positions for genres within cluster
            cluster_angle_span = cluster_end - cluster_start
            genre_current = cluster_start
            
            for genre_info in cluster_data['genres']:
                genre = genre_info['name']
                genre_tracks = genre_info['tracks']
                is_phantom = genre_info['is_phantom']
                
                # Proportional angle within cluster
                genre_proportion = genre_tracks / cluster_data['tracks']
                genre_angle_span = genre_proportion * cluster_angle_span
                
                # Draw genre wedge
                if is_phantom:
                    # Phantom genre - bright red with thick border
                    wedge = Wedge((0, 0), 1.05, genre_current, 
                                 genre_current + genre_angle_span,
                                 width=1.05 - outer_radius,
                                 facecolor='#FF1744', 
                                 edgecolor='darkred', linewidth=3,
                                 alpha=0.9)
                    ax.add_patch(wedge)
                    
                    # Add glow effect
                    glow = Wedge((0, 0), 1.06, genre_current, 
                                genre_current + genre_angle_span,
                                width=1.06 - outer_radius,
                                facecolor='red', 
                                alpha=0.3)
                    ax.add_patch(glow)
                else:
                    # Regular genre - even lighter color
                    genre_color = lighten_color(family_colors[family_id], 0.5)
                    wedge = Wedge((0, 0), 1.0, genre_current, 
                                 genre_current + genre_angle_span,
                                 width=1.0 - outer_radius,
                                 facecolor=genre_color, 
                                 edgecolor='white', linewidth=1)
                    ax.add_patch(wedge)
                
                # Add genre label (if large enough)
                if genre_angle_span > 3:  # Only label if wedge is big enough
                    mid_angle = genre_current + genre_angle_span / 2
                    label_radius = (outer_radius + 1.0) / 2
                    x = label_radius * np.cos(np.radians(mid_angle))
                    y = label_radius * np.sin(np.radians(mid_angle))
                    
                    # Text rotation and alignment
                    if 90 < mid_angle < 270:
                        rotation = mid_angle - 90
                        ha = 'right' if genre_angle_span < 10 else 'center'
                    else:
                        rotation = mid_angle + 90
                        ha = 'left' if genre_angle_span < 10 else 'center'
                    
                    display_name = genre_display[genre]
                    fontsize = 7 if genre_angle_span < 10 else 8
                    
                    ax.text(x, y, display_name, 
                           ha=ha, va='center', rotation=rotation,
                           fontsize=fontsize + (2 if is_phantom else 0), 
                           weight='bold',
                           color='white' if is_phantom else 'black',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='darkred', 
                                   alpha=0.8) if is_phantom else None)
                
                genre_current += genre_angle_span
    
    # Add center circle
    center = Circle((0, 0), inner_radius, facecolor='white', 
                   edgecolor='black', linewidth=2)
    ax.add_patch(center)
    ax.text(0, 0, f'Total\n{total_tracks}\ntracks', 
           ha='center', va='center', fontsize=14, weight='bold')
    
    # Add legend
    legend_elements = [
        patches.Patch(facecolor='#FF1744', edgecolor='darkred', linewidth=3,
                     alpha=0.9, label=f'PHANTOM GENRES ({len(phantom_set)})'),
        patches.Patch(facecolor='gray', alpha=0.7, label='Regular genres'),
        patches.Patch(facecolor='darkgray', alpha=0.5, label='Acoustic clusters'),
        patches.Patch(facecolor='darkgray', alpha=0.8, label='Genre families')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=12,
             bbox_to_anchor=(1.15, -0.05))
    
    # Add phantom genre callout
    ax.text(0.5, -0.05, 
           f'⚠️ PHANTOM GENRES HIGHLIGHTED IN RED ({len(phantom_set)} genres with high similarity)', 
           transform=ax.transAxes, ha='center', fontsize=14, weight='bold',
           color='darkred', bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFE0E0', 
                                     edgecolor='darkred', linewidth=2))
    
    # Set limits and remove axes
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('phantom_analysis_results/edm_taxonomy_sunburst.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    print("Saved: edm_taxonomy_sunburst.png")
    plt.show()


def build_hierarchy(genres, family_clusters, acoustic_clusters, genre_counts):
    """Build hierarchical data structure for sunburst."""
    
    hierarchy = {}
    
    for i, genre in enumerate(genres):
        family_id = family_clusters[i]
        acoustic_id = acoustic_clusters[i]
        
        # Initialize family if needed
        if family_id not in hierarchy:
            hierarchy[family_id] = {
                'tracks': 0,
                'clusters': {}
            }
        
        # Initialize acoustic cluster if needed
        if acoustic_id not in hierarchy[family_id]['clusters']:
            hierarchy[family_id]['clusters'][acoustic_id] = {
                'tracks': 0,
                'genres': []
            }
        
        # Add genre info
        genre_info = {
            'name': genre,
            'tracks': genre_counts[genre],
            'is_phantom': genre in phantom_set
        }
        
        hierarchy[family_id]['clusters'][acoustic_id]['genres'].append(genre_info)
        hierarchy[family_id]['clusters'][acoustic_id]['tracks'] += genre_counts[genre]
        hierarchy[family_id]['tracks'] += genre_counts[genre]
    
    return hierarchy


def lighten_color(color, amount=0.5):
    """Lighten a color by a given amount."""
    try:
        # Convert hex to RGB
        c = color.lstrip('#')
        r, g, b = tuple(int(c[i:i+2], 16)/255.0 for i in (0, 2, 4))
        
        # Convert to HLS
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        
        # Lighten
        l = min(1, l + (1 - l) * amount)
        
        # Convert back to RGB
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        
        # Convert to hex
        return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
    except:
        return color


if __name__ == "__main__":
    try:
        # Check if phantom_set is defined globally (it was referenced but not defined)
        phantom_set = set()  # Initialize empty if not loaded elsewhere
        
        create_sunburst_visualization()
        
        print("\n" + "="*70)
        print("SUNBURST VISUALIZATION COMPLETE!")
        print("="*70)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()