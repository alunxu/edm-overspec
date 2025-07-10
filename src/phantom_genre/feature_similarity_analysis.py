import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def analyze_feature_similarities(data_path, genre1, genre2):
    """Deep dive into which features are most similar between two genres"""
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Get data for both genres
    g1_data = df[df['genre'] == genre1]
    g2_data = df[df['genre'] == genre2]
    
    print(f"\nAnalyzing: {genre1} ({len(g1_data)} tracks) vs {genre2} ({len(g2_data)} tracks)")
    print("="*70)
    
    # Get numeric features
    numeric_cols = []
    for col in df.columns:
        if col not in ['genre', 'song', 'meta.Key']:
            try:
                pd.to_numeric(df[col], errors='coerce')
                if df[col].dtype in ['float64', 'int64']:
                    numeric_cols.append(col)
            except:
                pass
    
    # Calculate similarities for all features
    results = []
    
    for feature in numeric_cols:
        try:
            # Get clean numeric data
            f1 = pd.to_numeric(g1_data[feature], errors='coerce').dropna()
            f2 = pd.to_numeric(g2_data[feature], errors='coerce').dropna()
            
            if len(f1) < 5 or len(f2) < 5:
                continue
            
            # Calculate statistics
            mean1, mean2 = f1.mean(), f2.mean()
            std1, std2 = f1.std(), f2.std()
            
            # T-test
            t_stat, p_value = stats.ttest_ind(f1, f2)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((std1**2 + std2**2) / 2)
            cohen_d = abs(mean1 - mean2) / pooled_std if pooled_std > 0 else 0
            
            # Relative difference
            rel_diff = abs(mean1 - mean2) / ((mean1 + mean2) / 2) * 100 if (mean1 + mean2) != 0 else 0
            
            results.append({
                'Feature': feature,
                'Mean_G1': mean1,
                'Mean_G2': mean2,
                'Relative_Diff_%': rel_diff,
                'P_Value': p_value,
                'Cohen_d': cohen_d,
                'Significant': p_value < 0.05
            })
            
        except Exception as e:
            continue
    
    results_df = pd.DataFrame(results)
    
    # Sort by similarity (smallest relative difference)
    results_df = results_df.sort_values('Relative_Diff_%')
    
    # Print most similar features
    print("\nMOST SIMILAR FEATURES (Top 20):")
    print("-"*70)
    print(f"{'Feature':<30} {'Rel Diff %':<12} {'P-Value':<10} {'Cohen d':<10} {'Sig?':<5}")
    print("-"*70)
    
    for _, row in results_df.head(20).iterrows():
        sig = "No" if row['P_Value'] > 0.05 else "Yes"
        print(f"{row['Feature']:<30} {row['Relative_Diff_%']:<12.1f} {row['P_Value']:<10.3f} {row['Cohen_d']:<10.3f} {sig:<5}")
    
    # Summary statistics
    non_sig = (results_df['P_Value'] > 0.05).sum()
    negligible = (results_df['Cohen_d'] < 0.2).sum()
    small = ((results_df['Cohen_d'] >= 0.2) & (results_df['Cohen_d'] < 0.5)).sum()
    
    print(f"\nSUMMARY STATISTICS:")
    print(f"- Total features analyzed: {len(results_df)}")
    print(f"- Non-significant differences (p > 0.05): {non_sig} ({non_sig/len(results_df)*100:.1f}%)")
    print(f"- Negligible effect sizes (d < 0.2): {negligible} ({negligible/len(results_df)*100:.1f}%)")
    print(f"- Small effect sizes (0.2 ≤ d < 0.5): {small} ({small/len(results_df)*100:.1f}%)")
    print(f"- Average relative difference: {results_df['Relative_Diff_%'].mean():.1f}%")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Feature Similarity Analysis: {genre1} vs {genre2}', fontsize=16)
    
    # 1. Relative difference distribution
    ax = axes[0, 0]
    ax.hist(results_df['Relative_Diff_%'], bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(results_df['Relative_Diff_%'].mean(), color='red', linestyle='--', 
               label=f'Mean: {results_df["Relative_Diff_%"].mean():.1f}%')
    ax.set_xlabel('Relative Difference (%)')
    ax.set_ylabel('Number of Features')
    ax.set_title('Distribution of Feature Differences')
    ax.legend()
    
    # 2. P-value distribution
    ax = axes[0, 1]
    ax.hist(results_df['P_Value'], bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(0.05, color='red', linestyle='--', label='p = 0.05')
    ax.set_xlabel('P-Value')
    ax.set_ylabel('Number of Features')
    ax.set_title('Statistical Significance Distribution')
    ax.legend()
    
    # 3. Effect size distribution
    ax = axes[1, 0]
    ax.hist(results_df['Cohen_d'], bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(0.2, color='red', linestyle='--', label='d = 0.2 (small)')
    ax.axvline(0.5, color='orange', linestyle='--', label='d = 0.5 (medium)')
    ax.set_xlabel("Cohen's d")
    ax.set_ylabel('Number of Features')
    ax.set_title('Effect Size Distribution')
    ax.legend()
    
    # 4. Top differing features
    ax = axes[1, 1]
    top_diff = results_df.nlargest(10, 'Relative_Diff_%')
    y_pos = np.arange(len(top_diff))
    ax.barh(y_pos, top_diff['Relative_Diff_%'])
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f[:25] for f in top_diff['Feature']])
    ax.set_xlabel('Relative Difference (%)')
    ax.set_title('Top 10 Most Different Features')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f'feature_analysis_{genre1.replace(" ", "_")}_vs_{genre2.replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return results_df

# Analyze the main phantom pairs
if __name__ == "__main__":
    data_path = 'dataset/top100_with_tempogram_nmf_meta.csv'
    
    pairs = [
        ('Deep House', 'Organic House'),
        ('Tech House', 'Minimal - Deep Tech'),
        ('Techno (Peak Time - Driving)', 'Techno (Raw - Deep - Hypnotic)'),
        ('Trance (Main Floor)', 'Trance (Raw - Deep - Hypnotic)')
    ]
    
    all_results = {}
    for g1, g2 in pairs:
        results = analyze_feature_similarities(data_path, g1, g2)
        all_results[f"{g1} vs {g2}"] = results
    
    # Create summary plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    summary_data = []
    for pair_name, results in all_results.items():
        summary_data.append({
            'Pair': pair_name.replace(' (Peak Time - Driving)', ' (PTD)')\
                              .replace(' (Raw - Deep - Hypnotic)', ' (RDH)')\
                              .replace(' (Main Floor)', ' (MF)')\
                              .replace('Minimal - Deep Tech', 'Min-DT'),
            'Non-Sig %': (results['P_Value'] > 0.05).sum() / len(results) * 100,
            'Avg Rel Diff %': results['Relative_Diff_%'].mean(),
            'Negligible Effect %': (results['Cohen_d'] < 0.2).sum() / len(results) * 100
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    x = np.arange(len(summary_df))
    width = 0.25
    
    ax.bar(x - width, summary_df['Non-Sig %'], width, label='Non-Significant Features %', alpha=0.8)
    ax.bar(x, summary_df['Avg Rel Diff %'], width, label='Avg Relative Difference %', alpha=0.8)
    ax.bar(x + width, summary_df['Negligible Effect %'], width, label='Negligible Effect Size %', alpha=0.8)
    
    ax.set_xlabel('Genre Pairs')
    ax.set_ylabel('Percentage')
    ax.set_title('Phantom Genre Evidence: Feature Similarity Summary')
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df['Pair'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('phantom_genre_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n\nALL PAIRS SUMMARY:")
    print("="*70)
    for _, row in summary_df.iterrows():
        print(f"\n{row['Pair']}:")
        print(f"  - Non-significant features: {row['Non-Sig %']:.1f}%")
        print(f"  - Average relative difference: {row['Avg Rel Diff %']:.1f}%")
        print(f"  - Negligible effect sizes: {row['Negligible Effect %']:.1f}%")