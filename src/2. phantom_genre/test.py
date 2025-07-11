"""
true_phantom_radar_analysis.py
==============================
Identify and visualize genres that are ACTUALLY acoustically similar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from math import pi
from scipy.spatial.distance import cosine
from sklearn.preprocessing import StandardScaler
import seaborn as sns


def find_acoustically_similar_genres():
    """Find genre pairs that are truly acoustically similar based on features."""
    
    print("="*70)
    print("TRUE PHANTOM GENRE ANALYSIS - BASED ON ACOUSTIC SIMILARITY")
    print("="*70)
    
    # Load original dataset
    df = pd.read_csv('dataset/top100_with_tempogram_nmf_meta.csv')
    
    # Get numeric features only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\nUsing {len(numeric_cols)} acoustic features")
    
    # Calculate mean features for each genre
    print("\nCalculating genre acoustic profiles...")
    genre_features = df.groupby('genre')[numeric_cols].mean()
    
    # Standardize features for fair comparison
    scaler = StandardScaler()
    genre_features_scaled = pd.DataFrame(
        scaler.fit_transform(genre_features),
        index=genre_features.index,
        columns=genre_features.columns
    )
    
    # Calculate pairwise acoustic similarity
    print("\nCalculating acoustic similarities between genres...")
    genres = genre_features_scaled.index
    n_genres = len(genres)
    
    similarity_matrix = np.zeros((n_genres, n_genres))
    
    for i, g1 in enumerate(genres):
        for j, g2 in enumerate(genres):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                # Cosine similarity
                feat1 = genre_features_scaled.loc[g1].values
                feat2 = genre_features_scaled.loc[g2].values
                similarity_matrix[i, j] = 1 - cosine(feat1, feat2)
    
    # Convert to DataFrame
    similarity_df = pd.DataFrame(similarity_matrix, index=genres, columns=genres)
    
    # Find truly similar pairs (high acoustic similarity)
    similar_pairs = []
    
    for i, g1 in enumerate(genres):
        for j, g2 in enumerate(genres):
            if i < j:  # Avoid duplicates
                sim = similarity_df.loc[g1, g2]
                if sim > 0.85:  # High similarity threshold
                    similar_pairs.append({
                        'genre1': g1,
                        'genre2': g2,
                        'acoustic_similarity': sim
                    })
    
    similar_pairs_df = pd.DataFrame(similar_pairs)
    if len(similar_pairs_df) > 0:
        similar_pairs_df = similar_pairs_df.sort_values('acoustic_similarity', ascending=False)
        print(f"\nFound {len(similar_pairs_df)} acoustically similar genre pairs (>0.85 similarity)")
    else:
        print("\nNo genre pairs found with >0.85 acoustic similarity. Lowering threshold...")
        # Try with lower threshold
        similar_pairs = []
        for i, g1 in enumerate(genres):
            for j, g2 in enumerate(genres):
                if i < j:  # Avoid duplicates
                    sim = similarity_df.loc[g1, g2]
                    if sim > 0.75:  # Lower threshold
                        similar_pairs.append({
                            'genre1': g1,
                            'genre2': g2,
                            'acoustic_similarity': sim
                        })
        
        similar_pairs_df = pd.DataFrame(similar_pairs)
        if len(similar_pairs_df) > 0:
            similar_pairs_df = similar_pairs_df.sort_values('acoustic_similarity', ascending=False)
            print(f"\nFound {len(similar_pairs_df)} acoustically similar genre pairs (>0.75 similarity)")
        else:
            print("\nNo similar pairs found even with lower threshold!")
    
    # Create heatmap of acoustic similarity
    create_acoustic_similarity_heatmap(similarity_df)
    
    # Create radar plots for truly similar genres
    create_true_phantom_radar_plots(df, genre_features, similar_pairs_df)
    
    return similarity_df, similar_pairs_df


def create_acoustic_similarity_heatmap(similarity_df):
    """Create a heatmap showing acoustic similarity between all genres."""
    
    plt.figure(figsize=(14, 12))
    
    # Hierarchical clustering for better ordering
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform
    
    # Convert similarity to distance
    distance_matrix = 1 - similarity_df.values
    condensed_dist = squareform(distance_matrix)
    linkage_matrix = linkage(condensed_dist, method='average')
    dendro = dendrogram(linkage_matrix, no_plot=True)
    order = dendro['leaves']
    
    # Reorder matrix
    ordered_sim = similarity_df.iloc[order, order]
    
    # Create heatmap
    mask = np.triu(np.ones_like(ordered_sim), k=1)
    
    sns.heatmap(ordered_sim, 
                mask=mask,
                cmap='RdBu_r',
                center=0.5,
                vmin=0, vmax=1,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": .8, "label": "Acoustic Similarity"},
                xticklabels=True,
                yticklabels=True,
                annot=False)
    
    plt.title('Acoustic Similarity Matrix - Based on Audio Features', fontsize=16, weight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    
    # Highlight high similarity regions
    plt.text(0.02, 0.98, 'Red = Acoustically similar\nBlue = Acoustically different', 
            transform=plt.gca().transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            va='top')
    
    plt.tight_layout()
    plt.savefig('plots/acoustic_similarity_matrix.png', dpi=300, bbox_inches='tight')
    print("\nSaved: plots/acoustic_similarity_matrix.png")
    plt.show()


def create_true_phantom_radar_plots(df, genre_features, similar_pairs_df):
    """Create radar plots for genres that are truly acoustically similar."""
    
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
    
    # Map features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        col_lower = col.lower()
        for dim, config in feature_mappings.items():
            if any(keyword in col_lower for keyword in config['keywords']):
                config['features'].append(col)
                break
    
    # Calculate normalized profiles
    print("\nCalculating normalized genre profiles...")
    genre_profiles = {}
    
    for genre in genre_features.index:
        profile = {}
        
        for dim, config in feature_mappings.items():
            if config['features']:
                # Get top 3 features for this dimension
                features = config['features'][:3]
                dim_values = []
                
                for feat in features:
                    if feat in genre_features.columns:
                        # Get percentile rank across all genres
                        values = genre_features[feat]
                        genre_value = genre_features.loc[genre, feat]
                        percentile = (values <= genre_value).sum() / len(values) * 100
                        dim_values.append(percentile)
                
                if dim_values:
                    profile[dim] = np.mean(dim_values)
                else:
                    profile[dim] = 50
            else:
                profile[dim] = 50
        
        genre_profiles[genre] = profile
    
    # Create radar plots
    fig = plt.figure(figsize=(16, 12))
    
    # Show top 8 most similar pairs
    n_pairs = min(8, len(similar_pairs_df))
    if n_pairs == 0:
        print("No highly similar genre pairs found!")
        return
    
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.3)
    
    categories = ['Energy', 'Dance', 'Tempo', 'Bass', 'Harmonic', 'Rhythm']
    num_vars = len(categories)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]
    
    for idx, (_, row) in enumerate(similar_pairs_df.head(n_pairs).iterrows()):
        ax = fig.add_subplot(gs[idx // 4, idx % 4], projection='polar')
        
        genre1 = row['genre1']
        genre2 = row['genre2']
        similarity = row['acoustic_similarity']
        
        # Get profiles
        if genre1 in genre_profiles and genre2 in genre_profiles:
            # Genre 1
            values1 = []
            for dim in ['energy', 'danceability', 'tempo', 'bass_presence', 
                       'harmonic_content', 'rhythmic_complexity']:
                values1.append(genre_profiles[genre1].get(dim, 50))
            values1 += values1[:1]
            
            # Genre 2
            values2 = []
            for dim in ['energy', 'danceability', 'tempo', 'bass_presence', 
                       'harmonic_content', 'rhythmic_complexity']:
                values2.append(genre_profiles[genre2].get(dim, 50))
            values2 += values2[:1]
            
            # Calculate profile difference
            profile_diff = np.mean([abs(v1 - v2) for v1, v2 in zip(values1[:-1], values2[:-1])])
            
            # Plot with transparency to show overlap
            ax.plot(angles, values1, 'o-', linewidth=2.5, 
                   label=genre1[:25], color='#FF6B6B', markersize=8, alpha=0.8)
            ax.fill(angles, values1, alpha=0.3, color='#FF6B6B')
            
            ax.plot(angles, values2, 's--', linewidth=2.5, 
                   label=genre2[:25], color='#4ECDC4', markersize=7, alpha=0.8)
            ax.fill(angles, values2, alpha=0.25, color='#4ECDC4')
            
            # Setup
            ax.set_theta_offset(pi / 2)
            ax.set_theta_direction(-1)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, size=9)
            ax.set_ylim(0, 100)
            ax.set_yticks([25, 50, 75])
            ax.set_yticklabels(['25', '50', '75'], size=8)
            ax.grid(True, linestyle='--', alpha=0.5)
            
            # Title shows acoustic similarity
            ax.set_title(f'Similarity: {similarity:.3f} | Diff: {profile_diff:.1f}', 
                        fontsize=10, fontweight='bold', pad=15)
            
            # Highlight if truly phantom (very similar)
            if profile_diff < 10:
                ax.set_facecolor('#FFE5E5')
            
            ax.legend(loc='upper left', bbox_to_anchor=(-0.2, 1.15), 
                     fontsize=8, frameon=True)
    
    fig.suptitle('True Phantom Genres: Based on Acoustic Feature Similarity', 
                 fontsize=18, fontweight='bold')
    
    fig.text(0.5, 0.02, 
             'Showing genre pairs with >85% acoustic similarity. Pink background indicates nearly identical profiles (diff < 10).',
             ha='center', fontsize=12, style='italic')
    
    plt.tight_layout()
    plt.savefig('plots/true_phantom_genres_radar.png', dpi=300, bbox_inches='tight')
    print("Saved: plots/true_phantom_genres_radar.png")
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("TRUE PHANTOM GENRE PAIRS (Acoustically Indistinguishable):")
    print("="*60)
    
    for _, row in similar_pairs_df.head(10).iterrows():
        print(f"{row['genre1']} ↔ {row['genre2']}: {row['acoustic_similarity']:.3f}")


if __name__ == "__main__":
    try:
        similarity_df, similar_pairs = find_acoustically_similar_genres()
        
        # Also check our co-clustering results against acoustic similarity
        print("\n" + "="*60)
        print("COMPARING CO-CLUSTERING VS ACOUSTIC SIMILARITY")
        print("="*60)
        
        # Load co-clustering results
        try:
            phantom_df = pd.read_csv('results/phantom_analysis_final/phantom_genre_pairs.csv')
            
            # Check acoustic similarity for top co-clustering pairs
            print("\nAcoustic similarity of top co-clustering pairs:")
            for _, row in phantom_df.head(10).iterrows():
                g1, g2 = row['genre1'], row['genre2']
                if g1 in similarity_df.index and g2 in similarity_df.index:
                    acoustic_sim = similarity_df.loc[g1, g2]
                    print(f"{g1} ↔ {g2}: Co-cluster={row['similarity']:.2f}, Acoustic={acoustic_sim:.3f}")
        except:
            print("Could not load co-clustering results for comparison")
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()