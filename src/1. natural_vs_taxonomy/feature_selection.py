"""
feature_selection.py
====================
Multi-criteria feature selection for EDM analysis
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


class EDMFeatureSelector:
    """
    Advanced feature selection using multiple criteria optimized for clustering.
    """
    
    def __init__(self, n_features=100, random_state=42):
        self.n_features = n_features
        self.random_state = random_state
        self.selected_features = None
        self.feature_scores = None
        self.selector_fitted = False
        
    def fit_select(self, X, y, n_clusters=35):
        """
        Select features using ensemble of methods.
        
        Parameters:
        -----------
        X : DataFrame or array-like
            Feature matrix
        y : array-like
            Genre labels
        n_clusters : int
            Target number of clusters
            
        Returns:
        --------
        X_selected : DataFrame
            Selected features
        """
        print("\n" + "="*60)
        print("FEATURE SELECTION")
        print("="*60)
        print(f"Selecting top {self.n_features} features from {X.shape[1]} total")
        
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Calculate scores using multiple methods
        scores = self._calculate_feature_scores(X, y, n_clusters)
        
        # Select top features
        self._select_top_features(X, scores)
        
        # Print summary
        print(f"\nTop 20 selected features:")
        for i, (feat, score) in enumerate(self.feature_scores.head(20).iterrows()):
            print(f"  {i+1:2d}. {score['feature']:40s} : {score['ensemble_score']:.4f}")
        
        return X[self.selected_features]
    
    def _calculate_feature_scores(self, X, y, n_clusters):
        """Calculate feature importance using multiple methods."""
        print("\nCalculating feature scores...")
        
        scores = {}
        
        # 1. ANOVA F-statistic
        print("  1. ANOVA F-statistic...")
        anova_selector = SelectKBest(f_classif, k='all')
        anova_selector.fit(X, y)
        scores['anova'] = self._normalize_scores(anova_selector.scores_)
        
        # 2. Mutual Information
        print("  2. Mutual Information...")
        mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
        scores['mutual_info'] = self._normalize_scores(mi_scores)
        
        # 3. Random Forest importance
        print("  3. Random Forest importance...")
        rf = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf.fit(X, y)
        scores['random_forest'] = self._normalize_scores(rf.feature_importances_)
        
        # 4. Extra Trees importance
        print("  4. Extra Trees importance...")
        et = ExtraTreesClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1
        )
        et.fit(X, y)
        scores['extra_trees'] = self._normalize_scores(et.feature_importances_)
        
        # 5. Variance-based score
        print("  5. Variance-based score...")
        variances = X.var()
        if variances.max() > 0 and variances.std() > 1e-10:
            # Prefer features with moderate variance
            variance_scores = 1 - np.abs(variances - variances.median()) / variances.max()
        else:
            variance_scores = np.ones(X.shape[1]) * 0.5
        scores['variance'] = self._normalize_scores(variance_scores)
        
        # 6. Cluster separation score
        print("  6. Cluster separation score...")
        separation_scores = self._compute_cluster_separation_scores(X, n_clusters)
        scores['cluster_separation'] = self._normalize_scores(separation_scores)
        
        return scores
    
    def _normalize_scores(self, scores):
        """Normalize scores to [0, 1] range."""
        scores = np.array(scores)
        if scores.max() == scores.min():
            return np.ones_like(scores) * 0.5
        return (scores - scores.min()) / (scores.max() - scores.min())
    
    def _compute_cluster_separation_scores(self, X, n_clusters):
        """Compute feature importance based on cluster separation potential."""
        # Quick K-means to estimate cluster structure
        kmeans = KMeans(n_clusters=min(n_clusters, len(X) // 10), 
                       n_init=3, 
                       random_state=self.random_state)
        labels = kmeans.fit_predict(X)
        
        scores = []
        for i in range(X.shape[1]):
            feature_values = X.iloc[:, i].values
            
            between_var = 0
            within_var = 0
            
            for c in range(n_clusters):
                mask = labels == c
                if mask.sum() > 0:
                    cluster_mean = feature_values[mask].mean()
                    global_mean = feature_values.mean()
                    
                    between_var += mask.sum() * (cluster_mean - global_mean) ** 2
                    within_var += np.sum((feature_values[mask] - cluster_mean) ** 2)
            
            # Higher ratio = better cluster separation
            score = between_var / (within_var + 1e-6)
            scores.append(score)
        
        return np.array(scores)
    
    def _select_top_features(self, X, scores):
        """Select top features based on ensemble scores."""
        # Define weights for each method
        weights = {
            'anova': 0.25,
            'mutual_info': 0.20,
            'random_forest': 0.20,
            'extra_trees': 0.15,
            'variance': 0.10,
            'cluster_separation': 0.10
        }
        
        # Calculate ensemble scores
        ensemble_scores = np.zeros(X.shape[1])
        for method, weight in weights.items():
            ensemble_scores += weight * scores[method]
        
        # Select top features
        top_indices = np.argsort(ensemble_scores)[-self.n_features:]
        self.selected_features = X.columns[top_indices].tolist()
        
        # Store detailed scores
        self.feature_scores = pd.DataFrame({
            'feature': X.columns,
            'ensemble_score': ensemble_scores,
            **scores
        }).sort_values('ensemble_score', ascending=False)
        
        self.selector_fitted = True
    
    def transform(self, X):
        """Transform data using selected features."""
        if not self.selector_fitted:
            raise ValueError("Selector not fitted. Call fit_select first.")
        
        if isinstance(X, pd.DataFrame):
            return X[self.selected_features]
        else:
            # Assume same column order
            feature_indices = [i for i, feat in enumerate(X.columns) 
                             if feat in self.selected_features]
            return X[:, feature_indices]
    
    def get_selection_info(self):
        """Get information about feature selection."""
        if not self.selector_fitted:
            return {"status": "Not fitted"}
        
        return {
            'n_features_selected': len(self.selected_features),
            'selected_features': self.selected_features,
            'top_features': self.feature_scores.head(10)['feature'].tolist(),
            'feature_scores': self.feature_scores
        }