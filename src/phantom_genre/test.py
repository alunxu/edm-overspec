import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
# Set style
plt.style.use('seaborn-v0_8-whitegrid')

def create_distribution_comparison(data_path, output_path='phantom_genres_distributions.png'):
    """Create distribution plots showing overlap between phantom genres"""
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Define phantom pairs
    phantom_pairs = [
        {
            'genre1': 'Deep House',
            'genre2': 'Organic House',
            'title': 'Deep House vs Organic House',
            'color1': '#E74C3C',
            'color2': '#EC7063'
        },
        {
            'genre1': 'Tech House',
            'genre2': 'Minimal - Deep Tech',
            'title': 'Tech House vs Minimal-Deep Tech',
            'color1': '#3498DB',
            'color2': '#5DADE2'
        },
        {
            'genre1': 'Techno (Peak Time - Driving)',
            'genre2': 'Techno (Raw - Deep - Hypnotic)',
            'title': 'Techno (PTD) vs Techno (RDH)',
            'color1': '#2ECC71',
            'color2': '#58D68D'
        },
        {
            'genre1': 'Trance (Main Floor)',
            'genre2': 'Trance (Raw - Deep - Hypnotic)',
            'title': 'Trance (MF) vs Trance (RDH)',
            'color1': '#9B59B6',
            'color2': '#BB8FCE'
        }
    ]
    
    # Select key features to show
    key_features = [
        'meta.Bpm',
        '2-Energym',
        '4-SpectralCentroidm',
        '9-MFCCs1m'
    ]
    
    feature_names = {
        'meta.Bpm': 'Tempo (BPM)',
        '2-Energym': 'Energy',
        '4-SpectralCentroidm': 'Spectral Centroid',
        '9-MFCCs1m': 'MFCC 1'
    }
    
    # Create figure
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))
    
    for row, pair in enumerate(phantom_pairs):
        g1_data = df[df['genre'] == pair['genre1']]
        g2_data = df[df['genre'] == pair['genre2']]
        
        for col, feature in enumerate(key_features):
            ax = axes[row, col]
            
            # Get feature data
            feat1 = g1_data[feature].dropna()
            feat2 = g2_data[feature].dropna()
            
            # Create overlapping histograms
            # Determine common bins
            all_data = np.concatenate([feat1, feat2])
            bins = np.linspace(all_data.min(), all_data.max(), 25)
            
            # Plot histograms
            ax.hist(feat1, bins=bins, alpha=0.5, label=pair['genre1'].replace(' (Peak Time - Driving)', ' (PTD)')
                                                                    .replace(' (Raw - Deep - Hypnotic)', ' (RDH)')
                                                                    .replace(' (Main Floor)', ' (MF)'),
                    color=pair['color1'], density=True, edgecolor='black', linewidth=0.5)
            
            ax.hist(feat2, bins=bins, alpha=0.5, label=pair['genre2'].replace(' (Peak Time - Driving)', ' (PTD)')
                                                                    .replace(' (Raw - Deep - Hypnotic)', ' (RDH)')
                                                                    .replace(' (Main Floor)', ' (MF)')
                                                                    .replace('Minimal - Deep Tech', 'Min-DT'),
                    color=pair['color2'], density=True, edgecolor='black', linewidth=0.5)
            
            # Add KDE curves
            try:
                kde1 = stats.gaussian_kde(feat1)
                kde2 = stats.gaussian_kde(feat2)
                x_range = np.linspace(bins[0], bins[-1], 100)
                ax.plot(x_range, kde1(x_range), color=pair['color1'], linewidth=2)
                ax.plot(x_range, kde2(x_range), color=pair['color2'], linewidth=2)
            except:
                pass
            
            # Calculate overlap percentage
            hist1, _ = np.histogram(feat1, bins=bins, density=True)
            hist2, _ = np.histogram(feat2, bins=bins, density=True)
            overlap = np.minimum(hist1, hist2).sum() / np.maximum(hist1, hist2).sum() * 100
            
            # Formatting
            if row == 0:
                ax.set_title(feature_names[feature], fontsize=12, weight='bold')
            if col == 0:
                ax.set_ylabel(pair['title'], fontsize=11, weight='bold')
            
            # Add overlap percentage
            ax.text(0.95, 0.95, f'{overlap:.0f}%', transform=ax.transAxes,
                   ha='right', va='top', fontsize=10, weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            # Clean up axes
            ax.set_xlabel('')
            ax.set_ylabel('')
            if row < 3:
                ax.set_xticklabels([])
            
            # Add legend only to first plot of each row
            if col == 0 and row == 0:
                ax.legend(loc='upper left', fontsize=8)
    
    # Overall title
    fig.suptitle('Distribution Overlap: Phantom Genres Show Nearly Identical Acoustic Properties', 
                fontsize=18, weight='bold')
    
    # Add explanation
    fig.text(0.5, 0.01, 
            'Each plot shows overlapping distributions of acoustic features. Percentages indicate distribution overlap.\n' +
            'High overlap (>50%) demonstrates these "different" genres are acoustically similar.',
            ha='center', fontsize=12, style='italic')
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_single_pair_distribution(data_path, output_path='phantom_genre_single_dist.png'):
    """Create a detailed distribution comparison for one genre pair"""
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Focus on one pair for clarity
    genre1 = 'Deep House'
    genre2 = 'Organic House'
    
    g1_data = df[df['genre'] == genre1]
    g2_data = df[df['genre'] == genre2]
    
    # Select more features
    features = ['meta.Bpm', '2-Energym', '3-EnergyEntropym', '4-SpectralCentroidm', 
                '5-SpectralSpreadm', '8-SpectralRolloffm', '9-MFCCs1m', '10-MFCCs2m']
    
    feature_names = {
        'meta.Bpm': 'Tempo (BPM)',
        '2-Energym': 'Energy',
        '3-EnergyEntropym': 'Energy Entropy',
        '4-SpectralCentroidm': 'Spectral Centroid',
        '5-SpectralSpreadm': 'Spectral Spread',
        '8-SpectralRolloffm': 'Spectral Rolloff',
        '9-MFCCs1m': 'MFCC 1',
        '10-MFCCs2m': 'MFCC 2'
    }
    
    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    overlaps = []
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        
        # Get data
        feat1 = g1_data[feature].dropna()
        feat2 = g2_data[feature].dropna()
        
        # Normalize data for better comparison
        scaler = StandardScaler()
        feat1_norm = scaler.fit_transform(feat1.values.reshape(-1, 1)).flatten()
        feat2_norm = scaler.transform(feat2.values.reshape(-1, 1)).flatten()
        
        # Plot violin plots
        parts1 = ax.violinplot([feat1_norm], positions=[0], widths=0.4, 
                               showmeans=True, showmedians=True)
        parts2 = ax.violinplot([feat2_norm], positions=[0.5], widths=0.4,
                               showmeans=True, showmedians=True)
        
        # Color the violins
        for pc in parts1['bodies']:
            pc.set_facecolor('#E74C3C')
            pc.set_alpha(0.6)
        for pc in parts2['bodies']:
            pc.set_facecolor('#EC7063')
            pc.set_alpha(0.6)
        
        # Calculate overlap
        bins = np.linspace(min(feat1_norm.min(), feat2_norm.min()), 
                          max(feat1_norm.max(), feat2_norm.max()), 20)
        hist1, _ = np.histogram(feat1_norm, bins=bins, density=True)
        hist2, _ = np.histogram(feat2_norm, bins=bins, density=True)
        overlap = np.minimum(hist1, hist2).sum() / np.maximum(hist1, hist2).sum() * 100
        overlaps.append(overlap)
        
        # Labels and formatting
        ax.set_title(f'{feature_names[feature]}\n{overlap:.0f}% overlap', fontsize=11)
        ax.set_xticks([0, 0.5])
        ax.set_xticklabels(['Deep\nHouse', 'Organic\nHouse'], fontsize=10)
        ax.set_ylabel('Normalized Value', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    
    # Summary statistics
    avg_overlap = np.mean(overlaps)
    
    fig.suptitle(f'Deep House vs Organic House: Feature Distributions\n' + 
                f'Average Distribution Overlap: {avg_overlap:.0f}%', 
                fontsize=16, weight='bold')
    
    # Add interpretation
    fig.text(0.5, -0.02, 
            'Near-identical distributions across multiple acoustic features demonstrate these are phantom distinctions.\n' +
            'If these were truly different genres, we would expect to see clearly separated distributions.',
            ha='center', fontsize=12, style='italic')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return avg_overlap

# Run the analysis
if __name__ == "__main__":
    data_path = 'dataset/top100_with_tempogram_nmf_meta.csv'
    
    # Create both visualizations
    create_distribution_comparison(data_path)
    avg_overlap = create_single_pair_distribution(data_path)
    
    print(f"Created distribution comparison plots")
    print(f"Deep House vs Organic House average overlap: {avg_overlap:.0f}%")