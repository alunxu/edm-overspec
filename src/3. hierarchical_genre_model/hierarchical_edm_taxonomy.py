"""
hierarchical_edm_taxonomy.py
============================
Create a hierarchical visualization of reconciled EDM taxonomy.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import pickle
import os
import warnings
warnings.filterwarnings('ignore')


def create_hierarchical_taxonomy():
    """Create a hierarchical visualization of EDM taxonomy."""
    
    print("="*70)
    print("HIERARCHICAL EDM TAXONOMY - RECONCILED MODEL")
    print("="*70)
    
    # Load analysis results
    all_pairs_path = 'phantom_analysis_results/all_genre_pairs_analysis.csv'
    df_pairs = pd.read_csv(all_pairs_path)
    
    # Get all unique genres
    all_genres = sorted(list(set(df_pairs['genre1'].unique()) | set(df_pairs['genre2'].unique())))
    print(f"Total genres in analysis: {len(all_genres)}")
    
    # Define the hierarchical structure based on analysis
    # Level 1: 8 fundamental families
    fundamental_families = {
        'House Family': {
            'color': '#E74C3C',
            'genres': ['House', 'Deep House', 'Tech House', 'Jackin House', 
                      'Minimal - Deep Tech', 'Funky House', 'Afro House']
        },
        'Techno Family': {
            'color': '#3498DB',
            'genres': ['Techno (Peak Time - Driving)', 'Techno (Melodic)', 
                      'Hard Techno', 'Mainstage']
        },
        'Trance Family': {
            'color': '#9B59B6',
            'genres': ['Trance (Main Floor)', 'Trance (Raw - Deep - Hypnotic)', 
                      'Psy-Trance', 'Progressive House', 'Organic House']
        },
        'Bass Music Family': {
            'color': '#2ECC71',
            'genres': ['Dubstep', 'Trap - Future Bass', '140 - Deep Dubstep - Grime',
                      'Bass House', 'Bass - Club']
        },
        'Breakbeat Family': {
            'color': '#F39C12',
            'genres': ['Drum & Bass', 'UK Garage', 'Breaks - Breakbeat']
        },
        'Electronic Pop Family': {
            'color': '#E91E63',
            'genres': ['Dance - Pop', 'Indie Dance', 'Electro (Classic - Detroit - Modern)',
                      'Melodic House']
        },
        'Disco Family': {
            'color': '#00BCD4',
            'genres': ['Disco', 'Nu Disco']
        },
        'Experimental Family': {
            'color': '#795548',
            'genres': ['Ambient', 'Downtempo', 'Experimental', 'Electronica']
        },
        'Hardcore Family': {
            'color': '#FF5722',
            'genres': ['Hard Dance - Hardcore - Neo Rave']
        }
    }
    
    # Identify phantom pairs for special marking
    phantoms = df_pairs[(df_pairs['co_clustering'] > 0.8) & 
                       (df_pairs['centroid_distance'] < 10)]
    phantom_pairs = [(row['genre1'], row['genre2']) for _, row in phantoms.iterrows()]
    
    # Create the visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))
    
    # Left panel: Hierarchical tree structure
    create_tree_view(ax1, fundamental_families, phantom_pairs)
    
    # Right panel: Circular/radial view
    create_radial_view(ax2, fundamental_families, phantom_pairs)
    
    # Main title
    fig.suptitle('Reconciled EDM Taxonomy: Bridging Acoustic and Cultural Classifications', 
                 fontsize=20, weight='bold')
    
    # Add legend for phantom genres
    phantom_patch = patches.Patch(color='none', edgecolor='red', 
                                 linewidth=2, linestyle='--', 
                                 label='Phantom genre pairs')
    ax2.legend(handles=[phantom_patch], loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = 'phantom_analysis_results'
    plt.savefig(f'{output_dir}/hierarchical_edm_taxonomy.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f'{output_dir}/hierarchical_edm_taxonomy.pdf', 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"\nSaved hierarchical taxonomy visualization")
    print(f"  - {output_dir}/hierarchical_edm_taxonomy.png")
    print(f"  - {output_dir}/hierarchical_edm_taxonomy.pdf")
    
    plt.show()
    
    # Create detailed statistics
    create_taxonomy_statistics(fundamental_families, phantom_pairs)


def create_tree_view(ax, families, phantom_pairs):
    """Create a tree-style hierarchical view."""
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'Hierarchical Tree View', ha='center', fontsize=16, weight='bold')
    
    # Level labels
    ax.text(0.5, 10.5, 'Level 1:\nFundamental\nFamilies', ha='center', fontsize=10, 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray'))
    ax.text(0.5, 6, 'Level 2:\nAcoustic\nGenres', ha='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray'))
    ax.text(0.5, 2, 'Level 3:\nCultural\nVariations', ha='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray'))
    
    # Calculate positions
    n_families = len(families)
    family_width = 8.0 / n_families
    start_x = 1.5
    
    # Draw families and their genres
    for i, (family, info) in enumerate(families.items()):
        # Level 1: Family box
        family_x = start_x + i * family_width + family_width/2
        family_box = FancyBboxPatch((family_x - family_width/3, 9.8), 
                                   family_width/1.5, 0.8,
                                   boxstyle="round,pad=0.05",
                                   facecolor=info['color'], 
                                   edgecolor='black',
                                   alpha=0.7)
        ax.add_patch(family_box)
        ax.text(family_x, 10.2, family.replace(' Family', ''), 
                ha='center', va='center', fontsize=9, weight='bold')
        
        # Level 2 & 3: Genre boxes
        genres = info['genres']
        n_genres = len(genres)
        
        if n_genres > 0:
            # Distribute genres across Level 2
            genre_spacing = family_width / (n_genres + 1)
            
            for j, genre in enumerate(genres):
                genre_x = family_x - family_width/2 + (j + 1) * genre_spacing
                
                # Check if this genre is part of a phantom pair
                is_phantom = any((genre in pair) for pair in phantom_pairs)
                
                # Level 2: Acoustic genre
                if 'Deep' in genre or 'Tech' in genre or 'Main' in genre:
                    level2_y = 6
                    # Draw connection from family to genre
                    ax.plot([family_x, genre_x], [9.8, level2_y + 0.4], 
                           'k-', alpha=0.3, linewidth=1)
                else:
                    level2_y = 5.5
                    ax.plot([family_x, genre_x], [9.8, level2_y + 0.4], 
                           'k--', alpha=0.2, linewidth=0.8)
                
                # Genre box
                genre_name = genre.split(' - ')[0][:10]  # Shorten names
                
                if is_phantom:
                    # Phantom genre - special styling
                    ax.text(genre_x, level2_y, genre_name, 
                           ha='center', va='center', fontsize=7,
                           bbox=dict(boxstyle='round,pad=0.2', 
                                   facecolor='white', 
                                   edgecolor='red',
                                   linewidth=2,
                                   linestyle='--'))
                else:
                    ax.text(genre_x, level2_y, genre_name, 
                           ha='center', va='center', fontsize=7,
                           bbox=dict(boxstyle='round,pad=0.2', 
                                   facecolor=info['color'], 
                                   alpha=0.3))
                
                # Level 3: Cultural variations (if genre has sub-parts)
                if ' - ' in genre:
                    variations = genre.split(' - ')[1:]
                    for k, var in enumerate(variations):
                        var_y = 2 + k * 0.5
                        ax.plot([genre_x, genre_x], [level2_y - 0.2, var_y + 0.2], 
                               'k:', alpha=0.2, linewidth=0.5)
                        ax.text(genre_x, var_y, var[:8], 
                               ha='center', va='center', fontsize=6,
                               style='italic', alpha=0.7)


def create_radial_view(ax, families, phantom_pairs):
    """Create a radial/circular view of the taxonomy."""
    
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.axis('off')
    ax.set_aspect('equal')
    
    # Title
    ax.text(0, 3.5, 'Radial Taxonomy View', ha='center', fontsize=16, weight='bold')
    
    # Center: Core
    center = plt.Circle((0, 0), 0.5, color='lightgray', alpha=0.3)
    ax.add_patch(center)
    ax.text(0, 0, 'EDM\nCore', ha='center', va='center', fontsize=10, weight='bold')
    
    # Draw families in a circle
    n_families = len(families)
    angle_step = 2 * np.pi / n_families
    
    for i, (family, info) in enumerate(families.items()):
        angle = i * angle_step - np.pi/2  # Start from top
        
        # Family position
        family_r = 1.5
        family_x = family_r * np.cos(angle)
        family_y = family_r * np.sin(angle)
        
        # Family circle
        family_circle = plt.Circle((family_x, family_y), 0.35, 
                                  color=info['color'], alpha=0.7)
        ax.add_patch(family_circle)
        
        # Family name
        ax.text(family_x, family_y, family.replace(' Family', ''), 
                ha='center', va='center', fontsize=8, weight='bold',
                rotation=angle * 180/np.pi + 90 if abs(angle) > np.pi/2 else angle * 180/np.pi)
        
        # Draw genres around family
        genres = info['genres']
        n_genres = len(genres)
        
        if n_genres > 0:
            genre_angle_step = np.pi / 3 / max(n_genres, 1)  # Spread within 60 degrees
            start_angle = angle - genre_angle_step * (n_genres - 1) / 2
            
            for j, genre in enumerate(genres):
                genre_angle = start_angle + j * genre_angle_step
                genre_r = 2.3
                genre_x = genre_r * np.cos(genre_angle)
                genre_y = genre_r * np.sin(genre_angle)
                
                # Check if phantom
                is_phantom = any((genre in pair) for pair in phantom_pairs)
                
                # Draw connection
                ax.plot([family_x, genre_x], [family_y, genre_y], 
                       'k-', alpha=0.3, linewidth=1)
                
                # Genre node
                if is_phantom:
                    genre_circle = plt.Circle((genre_x, genre_y), 0.15, 
                                            color='white', 
                                            edgecolor='red',
                                            linewidth=2,
                                            linestyle='--')
                else:
                    genre_circle = plt.Circle((genre_x, genre_y), 0.15, 
                                            color=info['color'], alpha=0.4)
                ax.add_patch(genre_circle)
                
                # Genre label
                genre_name = genre.split(' - ')[0][:8]
                label_r = 2.6
                label_x = label_r * np.cos(genre_angle)
                label_y = label_r * np.sin(genre_angle)
                
                rotation = genre_angle * 180/np.pi - 90
                if rotation > 90:
                    rotation -= 180
                elif rotation < -90:
                    rotation += 180
                    
                ax.text(label_x, label_y, genre_name, 
                       ha='center', va='center', fontsize=6,
                       rotation=rotation)
    
    # Add concentric circles for levels
    for r, label in [(0.5, 'Core'), (1.5, 'Families'), (2.3, 'Genres')]:
        circle = plt.Circle((0, 0), r, fill=False, 
                          linestyle=':', alpha=0.3)
        ax.add_patch(circle)


def create_taxonomy_statistics(families, phantom_pairs):
    """Create statistics about the taxonomy."""
    
    print("\n" + "="*70)
    print("TAXONOMY STATISTICS")
    print("="*70)
    
    # Count genres at each level
    total_genres = sum(len(info['genres']) for info in families.values())
    n_families = len(families)
    
    print(f"\nLevel 1 - Fundamental Families: {n_families}")
    for family, info in families.items():
        print(f"  {family}: {len(info['genres'])} genres")
    
    print(f"\nLevel 2 - Acoustic Genres: ~{int(total_genres * 0.5)} (estimated)")
    print(f"Level 3 - Cultural Variations: {total_genres}")
    
    print(f"\nPhantom genre pairs identified: {len(phantom_pairs)}")
    for g1, g2 in phantom_pairs[:5]:
        print(f"  {g1} ↔ {g2}")
    
    if len(phantom_pairs) > 5:
        print(f"  ... and {len(phantom_pairs) - 5} more")
    
    # Calculate consolidation potential
    consolidation = (total_genres - int(total_genres * 0.5)) / total_genres * 100
    print(f"\nPotential genre consolidation: {consolidation:.1f}%")
    print("(By merging phantom genres into their acoustic families)")


if __name__ == "__main__":
    try:
        create_hierarchical_taxonomy()
        
        print("\n" + "="*70)
        print("HIERARCHICAL TAXONOMY COMPLETE!")
        print("="*70)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()