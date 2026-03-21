"""
feature_pipeline.py
====================
Shared feature engineering pipeline for EDM genre analysis.

Consolidates duplicated feature engineering logic previously found in:
  - natural_vs_taxonomy/feature_engineering.py (EDMFeatureEngineer)
  - deviation/genre_flow.py (ConsistentFeatureEngineering)
  - deviation/centroid_divergence_viz.py (engineer_features)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from feature_processing.modules.feature_engineering import EDMFeatureEngineer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


# Columns that should never be treated as numeric features
NON_FEATURE_COLS = {'genre', 'song', 'track_name', 'artist_name', 'meta.Key'}


class FeaturePipeline:
    """
    Unified feature engineering pipeline for EDM genre analysis.

    Provides three stages:
      1. Feature extraction from raw DataFrame
      2. Multi-scale feature engineering (interactions, ratios, log transforms)
      3. Ensemble scaling

    Example usage::

        pipeline = FeaturePipeline()
        features, col_names = pipeline.extract_features(df)
        features_engineered = pipeline.engineer(features)
        features_scaled = pipeline.scale(features_engineered)
    """

    def __init__(self):
        self.feature_cols = None
        self.engineered_cols = None

    def extract_features(self, df):
        """
        Extract numeric feature columns from a raw DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Raw dataset with mixed columns.

        Returns
        -------
        feature_df : pd.DataFrame
            DataFrame with only numeric feature columns.
        feature_cols : list[str]
            List of selected column names.
        """
        numeric_cols = [
            col for col in df.columns
            if col not in NON_FEATURE_COLS
            and df[col].dtype in ('float64', 'int64')
        ]
        self.feature_cols = numeric_cols
        return df[numeric_cols].copy(), numeric_cols

    def engineer(self, feature_df, add_interactions=True):
        """
        Apply multi-scale feature engineering.

        Parameters
        ----------
        feature_df : pd.DataFrame
            Numeric feature matrix.
        add_interactions : bool
            Whether to add interaction/ratio features.

        Returns
        -------
        pd.DataFrame
            Engineered feature matrix.
        """
        result = feature_df.copy()

        if add_interactions:
            # Log transforms for strictly positive features to fix skewness
            for col in feature_df.columns:
                if feature_df[col].min() >= 0:
                    result[f'{col}_log'] = np.log1p(feature_df[col])

            # Key interaction features (Cross-domain products as mentioned in paper)
            interactions = [
                ('meta.Bpm', '80-onset_rate', 'bpm_onset_ratio'),
                ('2-Energym', 'sub_bass_ratio', 'energy_sub_interaction'),
                ('77-danceability', 'sidechain_mod_depth', 'groove_pump_interaction'),
                ('7-SpectralFluxm', '80-onset_rate', 'flux_onset_interaction')
            ]
            
            for feat1, feat2, name in interactions:
                if feat1 in feature_df.columns and feat2 in feature_df.columns:
                    # For ratios, divide (adding epsilon)
                    if 'ratio' in name:
                        result[name] = feature_df[feat1] / (feature_df[feat2] + 1e-6)
                    # For interactions, multiply
                    else:
                        result[name] = feature_df[feat1] * feature_df[feat2]
                        
            # Square and Sqrt transforms for major rhythm/energy anchors
            anchors = ['meta.Bpm', '77-danceability', '80-onset_rate', '2-Energym']
            for feat in anchors:
                if feat in feature_df.columns:
                    result[f'{feat}_squared'] = feature_df[feat] ** 2
                    result[f'{feat}_sqrt'] = np.sqrt(np.abs(feature_df[feat]))

        # Fill any sparse NaN values with dataset mean
        result = result.fillna(result.mean())

        self.engineered_cols = result.columns.tolist()
        return result

    def scale(self, features, method='power'):
        """
        Apply scaling to feature matrix.

        Parameters
        ----------
        features : pd.DataFrame or np.ndarray
            Feature matrix.
        method : str
            One of 'ensemble', 'robust', 'standard', 'power'.

        Returns
        -------
        np.ndarray
            Scaled feature matrix.
        """
        if isinstance(features, pd.DataFrame):
            X = features.values
            columns = features.columns
            index = features.index
        else:
            X = features
            columns = None
            index = None

        if method == 'ensemble':
            X_robust = RobustScaler().fit_transform(X)
            X_power = PowerTransformer(method='yeo-johnson').fit_transform(X)
            X_standard = StandardScaler().fit_transform(X)
            X_scaled = 0.5 * X_robust + 0.3 * X_power + 0.2 * X_standard
        elif method == 'robust':
            X_scaled = RobustScaler().fit_transform(X)
        elif method == 'power':
            X_scaled = PowerTransformer(method='yeo-johnson').fit_transform(X)
        else:
            X_scaled = StandardScaler().fit_transform(X)

        if columns is not None:
            return pd.DataFrame(X_scaled, columns=columns, index=index)
        return X_scaled

    def select_features(self, features, labels, n_features=100):
        """
        Select top features using an ensemble of scoring methods.

        Parameters
        ----------
        features : np.ndarray or pd.DataFrame
            Feature matrix.
        labels : array-like
            Target labels.
        n_features : int
            Number of features to select.

        Returns
        -------
        selected : np.ndarray
            Selected feature matrix.
        top_indices : list[int]
            Indices of selected features.
        """
        if isinstance(features, pd.DataFrame):
            X = features.values
        else:
            X = features

        def _normalize(scores):
            scores = np.array(scores, dtype=float)
            mn, mx = scores.min(), scores.max()
            if mx == mn:
                return np.ones_like(scores) * 0.5
            return (scores - mn) / (mx - mn)

        # 1. ANOVA F-statistic
        f_scores = _normalize(
            SelectKBest(f_classif, k='all').fit(X, labels).scores_
        )
        # 2. Mutual Information
        mi_scores = _normalize(
            mutual_info_classif(X, labels, random_state=42)
        )
        # 3. Random Forest importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, labels)
        rf_scores = _normalize(rf.feature_importances_)
        # 4. Extra Trees importance
        et = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        et.fit(X, labels)
        et_scores = _normalize(et.feature_importances_)
        # 5. Variance-based scoring
        variances = np.var(X, axis=0)
        variance_scores = _normalize(variances)
        # 6. Cluster separation score
        n_clusters = min(35, len(X) // 10)
        km_labels = KMeans(n_clusters=n_clusters, n_init=3, random_state=42).fit_predict(X)
        sep_scores = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            col = X[:, i]
            global_mean = col.mean()
            between = sum(
                (km_labels == c).sum() * (col[km_labels == c].mean() - global_mean) ** 2
                for c in range(n_clusters)
            )
            within = sum(
                np.sum((col[km_labels == c] - col[km_labels == c].mean()) ** 2)
                for c in range(n_clusters)
            )
            sep_scores[i] = between / (within + 1e-6)
        sep_scores = _normalize(sep_scores)

        # Weighted ensemble (matches paper Eq. 3)
        ensemble = (
            0.25 * f_scores
            + 0.20 * mi_scores
            + 0.20 * rf_scores
            + 0.15 * et_scores
            + 0.10 * variance_scores
            + 0.10 * sep_scores
        )

        top_indices = np.argsort(ensemble)[-n_features:].tolist()
        return X[:, top_indices], top_indices
