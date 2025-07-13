"""
phantom_genre_radar_profiles.py
===============================
Create radar plots showing music profiles of phantom genre pairs.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from math import pi
import pickle
import os
from sklearn.preprocessing import StandardScaler


def create_phantom_radar_profiles():
    """Create radar plots for top phantom genre pairs."""
    
    print("="*70)
    print("PHANTOM GENRE MUSIC PROFILES - RADAR VISUALIZATION")
    print("="*70)
    
    # Load feature matrix and labels
    feature_matrix_path = 'results/pkl/feature_matrix.pkl'
    with open(feature_matrix_path, 'rb') as f:
        data = pickle.load(f)
    
    X = data['X_selected']
    y = data['genre_labels']
    feature_names = data.get('feature_names', [f'Feature_{i}' for i in range(X.shape[1])])
    
    # Load phantom pairs from analysis
    phantom_pairs_path = 'phantom_analysis_results/all_genre_pairs_analysis.csv'
    df_pairs = pd.read_csv(phantom_pairs_path)
    
    # Get top 4 phantom pairs
    phantoms = df_pairs[(df_pairs['co_clustering'] > 0.8) & 
                       (df_pairs['centroid_distance'] < 10)]
    top_phantoms = phantoms.nlargest(4, 'co_clustering')
    
    print(f"\nTop 4 phantom genre pairs:")
    for _, row in top_phantoms.iterrows():
        print(f"  {row['genre1']} ↔ {row['genre2']}: "
              f"co_clustering={row['co_clustering']:.3f}, distance={row['centroid_distance']:.1f}")
    
    # Define feature mappings for radar dimensions
    feature_mappings = {
        'energy': {
            'keywords': ['energy', 'rms', 'power', 'loud', 'amplitude'],
            'features': []
        },
        'danceability': {
            'keywords': ['dance', 'groove', 'swing'],
            'features': []
        },
        'tempo': {
            'keywords': ['bpm', 'tempo'],
            'features': []
        },
        'bass_presence': {
            'keywords': ['spectralrolloff', 'spectralspread', 'low'],
            'features': []
        },
        'harmonic_content': {
            'keywords': ['chroma', 'tonnetz', 'harmonic'],
            'features': []
        },
        'rhythmic_complexity': {
            'keywords': ['onset', 'beat', 'percussive'],
            'features': []
        }
    }
    
    # First, let's check if we have actual feature names or need to load from original data
    if all(f.startswith('Feature_') for f in feature_names[:5]):
        # We need to load the original dataset to get proper feature names
        print("\nLoading original dataset to get feature names...")
        try:
            original_df = pd.read_csv('dataset/top100_with_tempogram_nmf_meta.csv')
            numeric_cols = original_df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Map features based on actual column names
            for col in numeric_cols:
                col_lower = col.lower()
                mapped = False
                
                for dim, config in feature_mappings.items():
                    if any(keyword in col_lower for keyword in config['keywords']):
                        config['features'].append(col)
                        mapped = True
                        break
                
                # Store the actual feature names for later use
                feature_names = numeric_cols
        except:
            print("Could not load original dataset, using index-based mapping...")
            # Fallback to index mapping
            for i, feat_name in enumerate(feature_names):
                feat_lower = str(feat_name).lower()
                
                for dim, config in feature_mappings.items():
                    if any(keyword in feat_lower for keyword in config['keywords']):
                        config['features'].append(i)
                        break
    else:
        # We have actual feature names
        for i, col in enumerate(feature_names):
            col_lower = col.lower()
            
            for dim, config in feature_mappings.items():
                if any(keyword in col_lower for keyword in config['keywords']):
                    config['features'].append(i)
                    break
    
    # Print mapping summary
    for dim, config in feature_mappings.items():
        print(f"  {dim}: {len(config['features'])} features")
    
    # Calculate genre profiles using the approach from the reference code
    print("\nCalculating genre profiles...")
    genre_profiles = {}
    unique_genres = np.unique(y)
    
    # First calculate mean features for each genre
    genre_features = {}
    for genre in unique_genres:
        mask = y == genre
        if mask.sum() > 0:
            genre_features[genre] = X[mask].mean(axis=0)
    
    # Convert to DataFrame for easier manipulation
    genre_features_df = pd.DataFrame(genre_features).T
    
    # Now calculate profiles with percentile ranking
    for genre in unique_genres:
        profile = {}
        
        for dim, config in feature_mappings.items():
            if config['features']:
                # Use actual feature names if available, otherwise use indices
                if isinstance(config['features'][0], str):
                    # We have feature names
                    features = config['features'][:3]  # Top 3 features for this dimension
                    dim_values = []
                    
                    for feat in features:
                        if feat in genre_features_df.columns:
                            # Get percentile rank across all genres
                            values = genre_features_df[feat]
                            genre_value = genre_features_df.loc[genre, feat]
                            percentile = (values <= genre_value).sum() / len(values) * 100
                            dim_values.append(percentile)
                else:
                    # We have indices
                    feature_indices = config['features'][:3]
                    dim_values = []
                    
                    for idx in feature_indices:
                        if idx < len(genre_features[genre]):
                            # Get all values for this feature across genres
                            all_values = [genre_features[g][idx] for g in unique_genres]
                            genre_value = genre_features[genre][idx]
                            # Calculate percentile
                            percentile = sum(1 for v in all_values if v <= genre_value) / len(all_values) * 100
                            dim_values.append(percentile)
                
                if dim_values:
                    profile[dim] = np.mean(dim_values)
                else:
                    profile[dim] = 50
            else:
                profile[dim] = 50
        
        genre_profiles[genre] = profile
    
    # Create radar plots
    fig = plt.figure(figsize=(20, 5))
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.3)
    
    categories = ['Energy', 'Dance', 'Tempo', 'Bass', 'Harmonic', 'Rhythm']
    num_vars = len(categories)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]
    
    # Define consistent colors
    color1 = '#E74C3C'  # Red
    color2 = '#3498DB'  # Blue
    
    for idx, (_, row) in enumerate(top_phantoms.iterrows()):
        ax = fig.add_subplot(gs[0, idx], projection='polar')
        
        genre1 = row['genre1']
        genre2 = row['genre2']
        
        if genre1 in genre_profiles and genre2 in genre_profiles:
            # Get values for both genres
            values1 = []
            values2 = []
            
            for dim in ['energy', 'danceability', 'tempo', 'bass_presence', 
                       'harmonic_content', 'rhythmic_complexity']:
                values1.append(genre_profiles[genre1].get(dim, 50))
                values2.append(genre_profiles[genre2].get(dim, 50))
            
            values1 += values1[:1]  # Complete the circle
            values2 += values2[:1]
            
            # Calculate profile difference
            profile_diff = np.mean([abs(v1 - v2) for v1, v2 in zip(values1[:-1], values2[:-1])])
            
            # Plot both profiles
            ax.plot(angles, values1, 'o-', linewidth=2.5, 
                   label=genre1[:20], color=color1, markersize=8, alpha=0.8)
            ax.fill(angles, values1, alpha=0.25, color=color1)
            
            ax.plot(angles, values2, 's--', linewidth=2.5, 
                   label=genre2[:20], color=color2, markersize=7, alpha=0.8)
            ax.fill(angles, values2, alpha=0.25, color=color2)
            
            # Customize the radar chart
            ax.set_theta_offset(pi / 2)
            ax.set_theta_direction(-1)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, size=11)
            ax.set_ylim(0, 100)
            ax.set_yticks([25, 50, 75])
            ax.set_yticklabels(['25', '50', '75'], size=9)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Title with metrics
            title = f"{genre1.split(' - ')[0][:15]} vs {genre2.split(' - ')[0][:15]}\n"
            title += f"Co-clustering: {row['co_clustering']:.2f} | "
            title += f"Distance: {row['centroid_distance']:.1f} | "
            title += f"Profile Δ: {profile_diff:.1f}"
            ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
            
            # Highlight very similar profiles
            if profile_diff < 15:
                ax.patch.set_facecolor('#F0F0F0')
                ax.patch.set_alpha(0.3)
            
            # Legend
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), 
                     ncol=1, fontsize=10, frameon=True)
    
    # Main title
    fig.suptitle('Phantom Genre Music Profiles:',
                 fontsize=18, fontweight='bold', y=0.96)
    
    # Add explanation
    fig.text(0.5, 0.02, 
             'Showing top 4 phantom genre pairs (co-clustering > 0.8, distance < 10). ' +
             'Gray background indicates highly similar acoustic profiles (Δ < 15).',
             ha='center', fontsize=12, style='italic')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = 'phantom_analysis_results'
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(f'{output_dir}/phantom_genre_radar_profiles.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f'{output_dir}/phantom_genre_radar_profiles.pdf', 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"\nSaved radar profile visualization")
    print(f"  - {output_dir}/phantom_genre_radar_profiles.png")
    print(f"  - {output_dir}/phantom_genre_radar_profiles.pdf")
    
    plt.show()
    
    # Print profile analysis
    print("\n" + "="*70)
    print("PHANTOM GENRE PROFILE ANALYSIS:")
    print("="*70)
    
    for _, row in top_phantoms.iterrows():
        genre1 = row['genre1']
        genre2 = row['genre2']
        
        if genre1 in genre_profiles and genre2 in genre_profiles:
            print(f"\n{genre1} ↔ {genre2}:")
            print(f"  Co-clustering: {row['co_clustering']:.3f}")
            print(f"  Feature distance: {row['centroid_distance']:.1f}")
            
            print("  Profile comparison:")
            for dim in ['energy', 'danceability', 'tempo', 'bass_presence', 
                       'harmonic_content', 'rhythmic_complexity']:
                v1 = genre_profiles[genre1].get(dim, 50)
                v2 = genre_profiles[genre2].get(dim, 50)
                diff = abs(v1 - v2)
                print(f"    {dim:20s}: {v1:5.1f} vs {v2:5.1f} (Δ={diff:4.1f})")


if __name__ == "__main__":
    try:
        create_phantom_radar_profiles()
        
        print("\n" + "="*70)
        print("RADAR PROFILE ANALYSIS COMPLETE!")
        print("="*70)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()