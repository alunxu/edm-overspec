"""
Sweep feature counts to find the optimal number for clustering.
Strategy: run ONE greedy selection with n_greedy=35, getting a ranked ordering.
Then evaluate subsets by taking the first 20, 25, 30, 35 greedy features
+ varying ensemble top-ups.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import pickle
import time
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score, normalized_mutual_info_score,
                             adjusted_rand_score)

from feature_processing.modules.feature_engineering import EDMFeatureEngineer
from feature_processing.modules.feature_selection import EDMFeatureSelector

CACHE_PATH = 'results/pkl/feature_sweep_cache.pkl'
GREEDY_CACHE = 'results/pkl/greedy35_cache.pkl'

# ── Load & preprocess (with caching) ──
if os.path.exists(CACHE_PATH):
    print("Loading cached preprocessed data...")
    with open(CACHE_PATH, 'rb') as f:
        cache = pickle.load(f)
    X_dedup = cache['X_dedup']
    y = cache['y']
else:
    df = pd.read_csv('dataset/all_features_194.csv')
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[numeric_cols]
    y = df['genre']
    engineer = EDMFeatureEngineer(random_state=42)
    X_engineered = engineer.fit_transform(X)
    X_dedup = EDMFeatureSelector.remove_correlated(X_engineered, threshold=0.95)
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump({'X_dedup': X_dedup, 'y': y}, f)

print(f"Feature pool: {X_dedup.shape[1]} features, {len(y)} samples, {y.nunique()} genres")

# ── ONE greedy selection with n=35 to get ranked ordering ──
if os.path.exists(GREEDY_CACHE):
    print("Loading cached greedy-35 results...")
    with open(GREEDY_CACHE, 'rb') as f:
        gcache = pickle.load(f)
    greedy_ordered = gcache['greedy_ordered']
    ensemble_ranked = gcache['ensemble_ranked']
else:
    print("\nRunning greedy silhouette selection with n=35 (this takes ~15 min)...")
    t0 = time.time()
    selector = EDMFeatureSelector(random_state=42)
    greedy_ordered = selector.greedy_silhouette_selection(
        X_dedup, n_greedy=35, n_clusters=35, candidate_pool=60
    )
    print(f"  Greedy selection took {time.time()-t0:.1f}s")
    
    # Get ensemble ranking
    print("Running ensemble scoring...")
    ens_selector = EDMFeatureSelector(n_features=50, random_state=42)
    ens_selector.fit_select(X_dedup, y, n_clusters=35)
    ensemble_ranked = ens_selector.feature_scores['feature'].tolist()
    
    with open(GREEDY_CACHE, 'wb') as f:
        pickle.dump({'greedy_ordered': greedy_ordered, 'ensemble_ranked': ensemble_ranked}, f)
    print(f"Cached greedy-35 results to {GREEDY_CACHE}")

print(f"\nGreedy-35 features (ordered): {greedy_ordered}")

# ── Evaluate subsets ──
# For each total target, take first N_greedy features + enough ensemble top-up
TOTALS = [20, 25, 30, 35, 40, 45]
results = []

for n_total in TOTALS:
    # Try different greedy/ensemble splits
    for n_greedy in range(min(n_total, 15), min(n_total + 1, 36)):
        n_extra = n_total - n_greedy
        if n_extra < 0 or n_extra > 15:
            continue
        # Only test a few sensible splits: all-greedy, mostly-greedy, balanced
        if n_extra not in [0, 5, 10]:
            continue
        if n_greedy > 35:
            continue
            
        # Take first n_greedy from the ranked greedy list
        greedy_feats = greedy_ordered[:n_greedy]
        
        # Add ensemble top-up features not in greedy set
        extra_feats = []
        for feat in ensemble_ranked:
            if feat not in greedy_feats:
                extra_feats.append(feat)
            if len(extra_feats) >= n_extra:
                break
        
        features = greedy_feats + extra_feats
        actual_total = len(features)
        X_arr = X_dedup[features].values
        
        for k in [20, 35]:
            km = KMeans(n_clusters=k, init='k-means++', n_init=50,
                        max_iter=300, random_state=42)
            labels = km.fit_predict(X_arr)
            
            sil = silhouette_score(X_arr, labels)
            db = davies_bouldin_score(X_arr, labels)
            ch = calinski_harabasz_score(X_arr, labels)
            nmi = normalized_mutual_info_score(y, labels)
            ari = adjusted_rand_score(y, labels)
            
            results.append({
                'n_greedy': n_greedy, 'n_extra': n_extra,
                'n_total': actual_total, 'k': k,
                'silhouette': sil, 'davies_bouldin': db,
                'calinski_harabasz': ch, 'nmi': nmi, 'ari': ari,
                'features': features,
            })
        
        print(f"  {actual_total} feat (g={n_greedy},e={n_extra}): "
              f"k=20 Sil={results[-2]['silhouette']:.4f} DB={results[-2]['davies_bouldin']:.4f} NMI={results[-2]['nmi']:.4f}  |  "
              f"k=35 Sil={results[-1]['silhouette']:.4f} NMI={results[-1]['nmi']:.4f}")

# ── Summary ──
df_res = pd.DataFrame([{k: v for k, v in r.items() if k != 'features'} for r in results])

for k_eval in [20, 35]:
    print(f"\n{'='*70}")
    print(f"RESULTS AT k={k_eval} (sorted by silhouette)")
    print(f"{'='*70}")
    print(f"{'n_feat':>6}  {'config':>12}  {'Sil':>7}  {'DB':>7}  {'CH':>8}  {'NMI':>7}  {'ARI':>7}")
    sub = df_res[df_res['k'] == k_eval].sort_values('silhouette', ascending=False)
    for _, row in sub.iterrows():
        print(f"  {int(row['n_total']):>4}  g={int(row['n_greedy']):>2},e={int(row['n_extra']):>2}  "
              f"{row['silhouette']:.4f}  {row['davies_bouldin']:.4f}  {row['calinski_harabasz']:>8.1f}  "
              f"{row['nmi']:.4f}  {row['ari']:.4f}")

# Best composite at k=20
print(f"\n{'='*70}")
print("BEST CONFIG (k=20, composite: Sil↑ + (1-DB_norm)↑ + NMI↑)")
print(f"{'='*70}")
k20 = df_res[df_res['k'] == 20].copy()
for col, higher_better in [('silhouette', True), ('davies_bouldin', False), ('nmi', True)]:
    lo, hi = k20[col].min(), k20[col].max()
    normed = (k20[col] - lo) / (hi - lo + 1e-12)
    if not higher_better:
        normed = 1.0 - normed
    k20[f'{col}_n'] = normed
k20['composite'] = k20['silhouette_n'] + k20['davies_bouldin_n'] + k20['nmi_n']
best_idx = k20['composite'].idxmax()
best = k20.loc[best_idx]
print(f"  Winner: {int(best['n_total'])} features (g={int(best['n_greedy'])}, e={int(best['n_extra'])})")
print(f"  Sil={best['silhouette']:.4f}  DB={best['davies_bouldin']:.4f}  NMI={best['nmi']:.4f}")

# Save the winner's features for potential pipeline update
winner_feats = [r for r in results if r['n_total'] == int(best['n_total']) 
                and r['n_greedy'] == int(best['n_greedy'])
                and r['n_extra'] == int(best['n_extra'])
                and r['k'] == 20][0]['features']
print(f"  Features: {winner_feats}")

os.makedirs('results/experiments', exist_ok=True)
with open('results/experiments/feature_count_sweep.pkl', 'wb') as f:
    pickle.dump({'results': results, 'df': df_res, 'winner_features': winner_feats}, f)
print(f"\nSaved: results/experiments/feature_count_sweep.pkl")
