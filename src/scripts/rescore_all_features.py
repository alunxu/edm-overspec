"""
Rescore ALL features after correlation removal and save to CSV.
This replaces the truncated (top-100) version with the full set.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from feature_processing.modules.feature_engineering import EDMFeatureEngineer
from feature_processing.modules.feature_selection import EDMFeatureSelector

# 1. Load raw data
print("Loading data...")
df = pd.read_csv('dataset/all_features_194.csv')
non_feat = {'genre', 'song', 'track_name', 'artist_name', 'meta.Key'}
feat_cols = [c for c in df.columns if c not in non_feat and pd.api.types.is_numeric_dtype(df[c])]
X = df[feat_cols].copy()
y = df['genre']
print(f"  Raw features: {X.shape[1]}")

# 2. Feature engineering (same as pipeline)
engineer = EDMFeatureEngineer(random_state=42)
X_eng = engineer.fit_transform(X)
print(f"  After engineering: {X_eng.shape[1]}")

# 3. Correlation removal (same threshold as pipeline)
X_dedup = EDMFeatureSelector.remove_correlated(X_eng, threshold=0.95)
print(f"  After correlation removal: {X_dedup.shape[1]}")

# 4. Score ALL features using ensemble
selector = EDMFeatureSelector(n_features=X_dedup.shape[1], random_state=42)
selector.fit_select(X_dedup, y, n_clusters=35)

# 5. Save full scores
outpath = 'results/csv/feature_importance_scores.csv'
selector.feature_scores.to_csv(outpath, index=False)
print(f"\nSaved {len(selector.feature_scores)} feature scores to {outpath}")
