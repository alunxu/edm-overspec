"""
tismir_clean_legend_figure.py
=============================
Create clean TISMIR figure with smaller markers and legend instead of labels.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 16


def create_clean_legend_figure():
    """Create clean figure with legend for phantom pairs."""
    
    # Load the actual genre pairs data
    all_pairs_path = 'phantom_analysis_results/all_genre_pairs_analysis.csv'
    
    try:
        df = pd.read_csv(all_pairs_path)
        print(f"Loaded {len(df)} genre pairs from analysis")
    except FileNotFoundError:
        print(f"Error: Could not find {all_pairs_path}")
        return
    
    # Get unique genres count
    all_genres = set(df['genre1'].unique()) | set(df['genre2'].unique())
    n_genres = len(all_genres)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    # Define thresholds
    co_cluster_threshold = 0.8
    distance_threshold = 10
    
    # Create phantom region
    phantom_region = Rectangle((co_cluster_threshold, 0), 
                              1-co_cluster_threshold, distance_threshold,
                              facecolor='#90EE90', alpha=0.2, 
                              edgecolor='darkgreen', linewidth=2,
                              linestyle='-', zorder=1)
    ax.add_patch(phantom_region)
    
    # Identify phantom pairs
    phantoms = df[(df['co_clustering'] > co_cluster_threshold) & 
                  (df['centroid_distance'] < distance_threshold)]
    non_phantoms = df[~((df['co_clustering'] > co_cluster_threshold) & 
                       (df['centroid_distance'] < distance_threshold))]
    
    # Plot non-phantom pairs with smaller markers
    ax.scatter(non_phantoms['co_clustering'], 
              non_phantoms['centroid_distance'],
              c='#D3D3D3', s=20, alpha=0.5,  # Reduced from 50
              edgecolors='none',
              label=f'Non-phantom genre pairs (n={len(non_phantoms)})', 
              zorder=2)
    
    # Plot phantom pairs with smaller markers
    # Assign numbers to phantom pairs for legend
    phantoms = phantoms.copy()
    phantoms['pair_id'] = range(1, len(phantoms) + 1)
    
    scatter = ax.scatter(phantoms['co_clustering'], 
                        phantoms['centroid_distance'],
                        c='#FF4500', s=80,  # Reduced from 200
                        edgecolors='black', linewidth=1,
                        label=f'Phantom genre pairs (n={len(phantoms)})', 
                        zorder=10, marker='o')
    
    # Add numbers to phantom points
    for _, row in phantoms.iterrows():
        ax.annotate(str(row['pair_id']), 
                   (row['co_clustering'], row['centroid_distance']),
                   ha='center', va='center', 
                   fontsize=8, weight='bold', color='white')
    
    # Add threshold lines
    ax.axhline(y=distance_threshold, color='#4169E1', linestyle='--', 
               linewidth=1.5, alpha=0.8, zorder=3)
    ax.axvline(x=co_cluster_threshold, color='#DC143C', linestyle='--', 
               linewidth=1.5, alpha=0.8, zorder=3)
    
    # Add threshold labels
    ax.text(co_cluster_threshold - 0.01, df['centroid_distance'].max() * 0.95, 
            'Co-clustering threshold = 0.8', 
            rotation=0, va='top', ha='right', color='#DC143C', 
            fontsize=10, weight='bold')
    ax.text(0.99, distance_threshold + 0.5, 'Distance threshold = 10', 
            ha='right', va='bottom', color='#4169E1', 
            fontsize=10, weight='bold')
    
    # Add "PHANTOM ZONE" text
    ax.text(0.9, 5, 'PHANTOM\nZONE', ha='center', va='center',
            fontsize=12, weight='bold', color='darkgreen', alpha=0.7)
    
    # Set axis labels
    ax.set_xlabel('Co-clustering Similarity Score\n' + 
                  r'(Frequency of joint cluster assignment across multiple $k$-means runs)',
                  fontsize=12)
    ax.set_ylabel('Feature Space Distance\n' + 
                  '(Euclidean distance between genre centroids)',
                  fontsize=12)
    
    # Set title
    ax.set_title('Phantom Genre Detection in Electronic Dance Music',
                 fontsize=14, weight='bold', pad=12)
    ax.text(0.5, 1.015, 'High co-clustering with low feature distance indicates redundant genre labels',
            transform=ax.transAxes, ha='center', fontsize=11, style='italic')
    
    # Configure axes
    ax.set_xlim(-0.02, 1.02)
    y_max = df['centroid_distance'].max() * 1.1
    ax.set_ylim(-1, max(45, y_max))
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Create main legend
    handles, labels = ax.get_legend_handles_labels()
    legend1 = ax.legend(handles, labels, loc='lower left', 
                       frameon=True, fancybox=True,
                       shadow=False, borderpad=1, framealpha=0.95,
                       bbox_to_anchor=(0.02, 0.02))
    legend1.get_frame().set_facecolor('white')
    legend1.get_frame().set_edgecolor('#CCCCCC')
    
    # Add data summary
    summary_text = f'Dataset: {n_genres} genres, {len(df)} unique pairs analyzed'
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='gray', alpha=0.9))
    
    # Create phantom pairs legend (as a table)
    if len(phantoms) > 0:
        # Sort by co-clustering score
        phantoms_sorted = phantoms.sort_values('co_clustering', ascending=False)
        
        # Create legend text
        phantom_legend_text = "Phantom Genre Pairs:\n" + "-" * 30 + "\n"
        
        for _, row in phantoms_sorted.iterrows():
            # Simplify genre names
            g1 = row['genre1'].split(' - ')[0][:20]
            g2 = row['genre2'].split(' - ')[0][:20]
            
            phantom_legend_text += f"{row['pair_id']:2d}. {g1} ↔ {g2}\n"
            phantom_legend_text += f"    Co-clustering: {row['co_clustering']:.3f}, Distance: {row['centroid_distance']:.1f}\n"
        
        # Add phantom pairs legend box
        props = dict(boxstyle='round,pad=0.5', facecolor='lightyellow', 
                    edgecolor='darkgoldenrod', alpha=0.9)
        ax.text(1.02, 0.5, phantom_legend_text, transform=ax.transAxes, 
                fontsize=8, verticalalignment='center',
                bbox=props, family='monospace')
    
    # Adjust layout to accommodate legend
    plt.subplots_adjust(right=0.75)
    
    # Save
    output_dir = 'phantom_analysis_results'
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    fig.savefig(f'{output_dir}/phantom_genre_clean_legend_tismir.pdf', 
                dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(f'{output_dir}/phantom_genre_clean_legend_tismir.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"\nFigure saved with clean design and legend")
    print(f"  - {output_dir}/phantom_genre_clean_legend_tismir.pdf")
    print(f"  - {output_dir}/phantom_genre_clean_legend_tismir.png")
    
    plt.show()


def create_minimal_version():
    """Create an even more minimal version without the side legend."""
    
    # Load data
    all_pairs_path = 'phantom_analysis_results/all_genre_pairs_analysis.csv'
    
    try:
        df = pd.read_csv(all_pairs_path)
    except FileNotFoundError:
        print(f"Error: Could not find {all_pairs_path}")
        return
    
    # Get unique genres count
    all_genres = set(df['genre1'].unique()) | set(df['genre2'].unique())
    n_genres = len(all_genres)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Define thresholds
    co_cluster_threshold = 0.8
    distance_threshold = 10
    
    # Create phantom region
    phantom_region = Rectangle((co_cluster_threshold, 0), 
                              1-co_cluster_threshold, distance_threshold,
                              facecolor='#90EE90', alpha=0.15, 
                              edgecolor='darkgreen', linewidth=2,
                              linestyle='-', zorder=1)
    ax.add_patch(phantom_region)
    
    # Identify pairs
    phantoms = df[(df['co_clustering'] > co_cluster_threshold) & 
                  (df['centroid_distance'] < distance_threshold)]
    non_phantoms = df[~((df['co_clustering'] > co_cluster_threshold) & 
                       (df['centroid_distance'] < distance_threshold))]
    
    # Plot with very small markers
    ax.scatter(non_phantoms['co_clustering'], 
              non_phantoms['centroid_distance'],
              c='silver', s=15, alpha=0.4,
              edgecolors='none', zorder=2)
    
    ax.scatter(phantoms['co_clustering'], 
              phantoms['centroid_distance'],
              c='#FF4500', s=60,
              edgecolors='black', linewidth=0.8,
              zorder=10, marker='o')
    
    # Add threshold lines
    ax.axhline(y=distance_threshold, color='#4169E1', linestyle='--', 
               linewidth=1.5, alpha=0.6, zorder=3)
    ax.axvline(x=co_cluster_threshold, color='#DC143C', linestyle='--', 
               linewidth=1.5, alpha=0.6, zorder=3)
    
    # Minimal labels
    ax.text(0.79, 1, '0.8', ha='right', va='bottom', 
            color='#DC143C', fontsize=10, weight='bold')
    ax.text(1, 10.5, '10', ha='right', va='bottom', 
            color='#4169E1', fontsize=10, weight='bold')
    
    # Set axis labels
    ax.set_xlabel('Co-clustering Similarity Score', fontsize=12)
    ax.set_ylabel('Feature Space Distance', fontsize=12)
    
    # Set title
    ax.set_title('Phantom Genre Detection in Electronic Dance Music',
                 fontsize=14, weight='bold')
    
    # Configure axes
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-1, 45)
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Simple legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='silver', 
               markersize=6, alpha=0.6, label=f'{len(non_phantoms)} genre pairs'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF4500',
               markersize=8, markeredgecolor='black', markeredgewidth=0.8,
               label=f'{len(phantoms)} phantom pairs'),
        patches.Patch(facecolor='#90EE90', alpha=0.3, edgecolor='darkgreen',
                     label='Phantom zone')
    ]
    
    ax.legend(handles=legend_elements, loc='lower left', 
             frameon=True, fancybox=True, framealpha=0.9)
    
    # Add summary
    ax.text(0.02, 0.98, f'{n_genres} genres analyzed', 
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='gray', alpha=0.9))
    
    plt.tight_layout()
    
    # Save minimal version
    output_dir = 'phantom_analysis_results'
    fig.savefig(f'{output_dir}/phantom_genre_minimal_tismir.pdf', 
                dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(f'{output_dir}/phantom_genre_minimal_tismir.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"\nMinimal figure saved")
    plt.show()


if __name__ == "__main__":
    # Create version with detailed legend
    create_clean_legend_figure()
    
    # Also create minimal version
    create_minimal_version()