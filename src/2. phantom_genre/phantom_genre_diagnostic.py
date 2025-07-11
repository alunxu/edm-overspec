"""
phantom_genre_diagnostic.py
==========================
Diagnostic script to understand why phantom genres aren't being detected
and to find appropriate thresholds.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp
import pickle
import os


def diagnose_phantom_detection():
    """Run diagnostics to understand the data and find appropriate thresholds."""
    
    print("="*70)
    print("PHANTOM GENRE DETECTION DIAGNOSTICS")
    print("="*70)
    
    # Load data
    feature_matrix_path = 'results/pkl/feature_matrix.pkl'
    convergence_path = 'results/csv/genre_convergence_matrix.csv'
    
    # Try alternative paths if main path doesn't exist
    if not os.path.exists(convergence_path):
        # Try loading from pairs file
        pairs_path = 'results/csv/top_converging_pairs.csv'
        if os.path.exists(pairs_path):
            print(f"Loading convergence data from {pairs_path}")
            pairs_df = pd.read_csv(pairs_path)
            # Create convergence lookup
            convergence_dict = {}
            for _, row in pairs_df.iterrows():
                key = tuple(sorted([row['genre1'], row['genre2']]))
                convergence_dict[key] = row['similarity']
        else:
            print("Warning: No convergence data found!")
            convergence_dict = {}
    else:
        convergence_df = pd.read_csv(convergence_path, index_col=0)
        convergence_dict = None
    
    # Load feature matrix
    with open(feature_matrix_path, 'rb') as f:
        data = pickle.load(f)
    X = data['X_selected']
    y = data['genre_labels']
    
    unique_genres = np.unique(y)
    print(f"\nAnalyzing {len(unique_genres)} genres with {len(X)} tracks")
    
    # Calculate all pairwise metrics
    results = []
    
    print("\nCalculating pairwise metrics...")
    for i, g1 in enumerate(unique_genres):
        for j, g2 in enumerate(unique_genres[i+1:], i+1):
            # Get masks
            mask1 = y == g1
            mask2 = y == g2
            
            if mask1.sum() < 3 or mask2.sum() < 3:
                continue
            
            X1 = X[mask1]
            X2 = X[mask2]
            
            # Calculate distances
            centroid_dist = np.linalg.norm(X1.mean(0) - X2.mean(0))
            min_dist = cdist(X1, X2).min()
            
            # Calculate simple overlap (fraction of X1 points closer to X2 than to other X1 points)
            if len(X1) > 1 and len(X2) > 1:
                nn_dist_to_other = cdist(X1, X2).min(axis=1).mean()
                nn_dist_to_same = cdist(X1, X1)
                np.fill_diagonal(nn_dist_to_same, np.inf)
                nn_dist_to_same = nn_dist_to_same.min(axis=1).mean()
                overlap = nn_dist_to_other / (nn_dist_to_same + 1e-10)
            else:
                overlap = 1.0
            
            # Get co-clustering score
            if convergence_dict is not None:
                key = tuple(sorted([g1, g2]))
                co_clustering = convergence_dict.get(key, 0.0)
            else:
                try:
                    if g1 in convergence_df.index and g2 in convergence_df.columns:
                        co_clustering = convergence_df.loc[g1, g2]
                    elif g2 in convergence_df.index and g1 in convergence_df.columns:
                        co_clustering = convergence_df.loc[g2, g1]
                    else:
                        co_clustering = 0.0
                except:
                    co_clustering = 0.0
            
            results.append({
                'genre1': g1,
                'genre2': g2,
                'co_clustering': co_clustering,
                'centroid_distance': centroid_dist,
                'min_distance': min_dist,
                'overlap_ratio': overlap,
                'n_tracks_g1': mask1.sum(),
                'n_tracks_g2': mask2.sum()
            })
    
    results_df = pd.DataFrame(results)
    
    # Print statistics
    print("\n" + "="*70)
    print("CO-CLUSTERING STATISTICS:")
    print(f"Mean: {results_df['co_clustering'].mean():.3f}")
    print(f"Std: {results_df['co_clustering'].std():.3f}")
    print(f"Min: {results_df['co_clustering'].min():.3f}")
    print(f"Max: {results_df['co_clustering'].max():.3f}")
    print(f"Percentiles: 25%={results_df['co_clustering'].quantile(0.25):.3f}, "
          f"50%={results_df['co_clustering'].quantile(0.50):.3f}, "
          f"75%={results_df['co_clustering'].quantile(0.75):.3f}, "
          f"90%={results_df['co_clustering'].quantile(0.90):.3f}")
    
    print("\n" + "="*70)
    print("DISTANCE STATISTICS:")
    print(f"Centroid Distance - Mean: {results_df['centroid_distance'].mean():.3f}")
    print(f"Centroid Distance - Std: {results_df['centroid_distance'].std():.3f}")
    print(f"Centroid Distance - Percentiles: "
          f"10%={results_df['centroid_distance'].quantile(0.10):.3f}, "
          f"25%={results_df['centroid_distance'].quantile(0.25):.3f}, "
          f"50%={results_df['centroid_distance'].quantile(0.50):.3f}")
    
    # Find pairs with high co-clustering
    high_co_clustering = results_df[results_df['co_clustering'] > 
                                   results_df['co_clustering'].quantile(0.75)]
    
    print("\n" + "="*70)
    print(f"PAIRS WITH HIGH CO-CLUSTERING (top 25%, n={len(high_co_clustering)}):")
    print("-"*70)
    
    # Sort by co-clustering
    high_co_clustering = high_co_clustering.sort_values('co_clustering', ascending=False)
    
    for idx, row in high_co_clustering.head(20).iterrows():
        print(f"{row['genre1']:25} ↔ {row['genre2']:25}")
        print(f"  Co-clustering: {row['co_clustering']:.3f}, "
              f"Distance: {row['centroid_distance']:.2f}, "
              f"Overlap: {row['overlap_ratio']:.3f}")
    
    # Create diagnostic plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Distribution of co-clustering scores
    ax1 = axes[0, 0]
    ax1.hist(results_df['co_clustering'], bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(results_df['co_clustering'].mean(), color='red', linestyle='--', 
                label=f'Mean: {results_df["co_clustering"].mean():.3f}')
    ax1.axvline(results_df['co_clustering'].quantile(0.75), color='green', linestyle='--',
                label=f'75th percentile: {results_df["co_clustering"].quantile(0.75):.3f}')
    ax1.set_xlabel('Co-clustering Score')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Co-clustering Scores')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Distribution of distances
    ax2 = axes[0, 1]
    ax2.hist(results_df['centroid_distance'], bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(results_df['centroid_distance'].quantile(0.25), color='green', linestyle='--',
                label=f'25th percentile: {results_df["centroid_distance"].quantile(0.25):.3f}')
    ax2.set_xlabel('Centroid Distance')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Feature Distances')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Scatter plot: co-clustering vs distance
    ax3 = axes[0, 2]
    scatter = ax3.scatter(results_df['co_clustering'], 
                         results_df['centroid_distance'],
                         c=results_df['overlap_ratio'],
                         s=50, alpha=0.6, cmap='RdYlBu_r')
    
    # Add potential phantom region
    ax3.axhspan(0, results_df['centroid_distance'].quantile(0.25), 
                results_df['co_clustering'].quantile(0.75), 1.0, 
                alpha=0.1, color='green', label='Potential phantoms')
    
    ax3.set_xlabel('Co-clustering Score')
    ax3.set_ylabel('Centroid Distance')
    ax3.set_title('All Genre Pairs: Co-clustering vs Distance')
    plt.colorbar(scatter, ax=ax3, label='Overlap Ratio')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Overlap ratio distribution
    ax4 = axes[1, 0]
    ax4.hist(results_df['overlap_ratio'], bins=50, edgecolor='black', alpha=0.7)
    ax4.set_xlabel('Overlap Ratio')
    ax4.set_ylabel('Count')
    ax4.set_title('Distribution of Overlap Ratios\n(lower = more overlap)')
    ax4.grid(True, alpha=0.3)
    
    # 5. Top candidates based on relaxed criteria
    ax5 = axes[1, 1]
    
    # Find candidates with relaxed thresholds
    candidates = results_df[
        (results_df['co_clustering'] > results_df['co_clustering'].quantile(0.70)) &
        (results_df['centroid_distance'] < results_df['centroid_distance'].quantile(0.30))
    ].sort_values('co_clustering', ascending=False)
    
    if len(candidates) > 0:
        y_pos = np.arange(min(10, len(candidates)))
        labels = [f"{row['genre1'][:12]}-{row['genre2'][:12]}" 
                 for _, row in candidates.head(10).iterrows()]
        scores = candidates.head(10)['co_clustering'].values
        
        ax5.barh(y_pos, scores)
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(labels, fontsize=9)
        ax5.set_xlabel('Co-clustering Score')
        ax5.set_title(f'Top Phantom Candidates\n(Found {len(candidates)} with relaxed criteria)')
        ax5.grid(True, alpha=0.3, axis='x')
    else:
        ax5.text(0.5, 0.5, 'No candidates found\neven with relaxed criteria', 
                ha='center', va='center', transform=ax5.transAxes, fontsize=14)
    
    # 6. Summary and recommendations
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Calculate recommended thresholds
    rec_co_clustering = results_df['co_clustering'].quantile(0.70)
    rec_distance = results_df['centroid_distance'].quantile(0.30)
    
    summary = f"""
DIAGNOSTIC SUMMARY
==================

Data Statistics:
- Total genre pairs: {len(results_df)}
- Pairs with co-clustering > 0: {len(results_df[results_df['co_clustering'] > 0])}
- Mean co-clustering: {results_df['co_clustering'].mean():.3f}
- Mean distance: {results_df['centroid_distance'].mean():.3f}

Recommended Thresholds:
- Min co-clustering: {rec_co_clustering:.3f}
  (70th percentile)
- Max distance: {rec_distance:.3f}
  (30th percentile)
  
Potential Phantoms Found: {len(candidates)}

Next Steps:
1. Use relaxed thresholds
2. Check convergence data quality
3. Consider alternative metrics
"""
    
    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
            fontsize=11, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray'))
    
    plt.suptitle('Phantom Genre Detection Diagnostics', fontsize=16, weight='bold')
    plt.tight_layout()
    
    # Save diagnostic plots
    output_dir = 'phantom_analysis_results'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/diagnostic_plots.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved diagnostic plots to {output_dir}/diagnostic_plots.png")
    plt.show()
    
    # Export diagnostic data
    results_df.to_csv(f'{output_dir}/all_genre_pairs_analysis.csv', index=False)
    print(f"Exported all pair data to {output_dir}/all_genre_pairs_analysis.csv")
    
    # Return recommendations
    return {
        'recommended_co_clustering_threshold': rec_co_clustering,
        'recommended_distance_threshold': rec_distance,
        'n_potential_phantoms': len(candidates),
        'candidates': candidates
    }


def run_adjusted_phantom_detection():
    """Run phantom detection with adjusted thresholds based on diagnostics."""
    
    # First run diagnostics
    recommendations = diagnose_phantom_detection()
    
    if recommendations['n_potential_phantoms'] > 0:
        print("\n" + "="*70)
        print("RUNNING PHANTOM DETECTION WITH ADJUSTED THRESHOLDS")
        print("="*70)
        
        # Import and run the main detector with new thresholds
        from phantom_genre_detector import PhantomGenreDetector
        
        detector = PhantomGenreDetector()
        
        # Use more relaxed thresholds
        phantom_candidates = detector.detect_phantom_genres(
            min_co_clustering=max(0.5, recommendations['recommended_co_clustering_threshold']),
            min_musical_sim=0.4  # Much more relaxed
        )
        
        if len(phantom_candidates) > 0:
            detector.visualize_phantom_landscape()
            detector.export_results()
            detector.generate_report()
    else:
        print("\nNo phantom candidates found even with relaxed criteria.")
        print("Possible issues:")
        print("1. Convergence/co-clustering data might be missing or incomplete")
        print("2. The genres in your dataset might be genuinely distinct")
        print("3. Feature extraction might need adjustment")


if __name__ == "__main__":
    run_adjusted_phantom_detection()