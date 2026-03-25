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
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


class EDMFeatureSelector:
    """
    Multi-criteria feature selection optimized for genre-discriminative clustering.
    
    Primary method:
    - fit_select(): 6-method ensemble scoring (ANOVA, MI, RF, ET, variance,
      cluster separation), each label-supervised against 35 genre labels.
    
    Also provides (legacy / experimental):
    - greedy_silhouette_selection(): forward selection maximizing silhouette
    - hybrid_select(): greedy + ensemble top-up
    """
    
    def __init__(self, n_features=100, random_state=42):
        self.n_features = n_features
        self.random_state = random_state
        self.selected_features = None
        self.feature_scores = None
        self.selector_fitted = False
        self.hybrid_features = None  # tracks hybrid selection details
        
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
        top_to_print = min(20, self.n_features)
        print(f"\nTop {top_to_print} selected features:")
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
            n_jobs=1  # single-threaded for deterministic results
        )
        rf.fit(X, y)
        scores['random_forest'] = self._normalize_scores(rf.feature_importances_)
        
        # 4. Extra Trees importance
        print("  4. Extra Trees importance...")
        et = ExtraTreesClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=1  # single-threaded for deterministic results
        )
        et.fit(X, y)
        scores['extra_trees'] = self._normalize_scores(et.feature_importances_)
        
        # 5. Variance-based score
        print("  5. Variance-based score...")
        variances = X.var()
        if variances.max() > 0:
            # Higher variance = more discriminative for clustering
            variance_scores = variances / variances.max()
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
        actual_k = min(n_clusters, len(X) // 10)
        kmeans = KMeans(n_clusters=actual_k, 
                       n_init=3, 
                       random_state=self.random_state)
        labels = kmeans.fit_predict(X)
        unique_labels = np.unique(labels)
        
        scores = []
        for i in range(X.shape[1]):
            feature_values = X.iloc[:, i].values
            global_mean = feature_values.mean()
            
            between_var = 0
            within_var = 0
            
            for c in unique_labels:
                mask = labels == c
                if mask.sum() > 0:
                    cluster_mean = feature_values[mask].mean()
                    
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
    
    # ------------------------------------------------------------------
    # Correlation-based deduplication
    # ------------------------------------------------------------------
    @staticmethod
    def remove_correlated(X, threshold=0.95):
        """
        Remove features with Pearson |r| > threshold.
        
        Keeps the first feature of each correlated pair (column order).
        Standard preprocessing to avoid redundant features that inflate
        dimensionality without adding information.
        
        Parameters:
        -----------
        X : DataFrame
            Feature matrix (already scaled)
        threshold : float
            Correlation threshold (default 0.95)
            
        Returns:
        --------
        X_dedup : DataFrame
            Feature matrix with redundant features removed
        """
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
        X_dedup = X.drop(columns=to_drop)
        print(f"\nCorrelation deduplication (|r| > {threshold}):")
        print(f"  Removed {len(to_drop)} redundant features ({X.shape[1]} → {X_dedup.shape[1]})")
        return X_dedup
    
    # ------------------------------------------------------------------
    # Greedy silhouette-optimized forward selection
    # ------------------------------------------------------------------
    def greedy_silhouette_selection(self, X, n_greedy=20, n_clusters=35, 
                                    candidate_pool=80):
        """
        Greedy forward selection maximizing K-means silhouette coefficient.
        
        Starts with the single feature that gives the highest silhouette,
        then iteratively adds the feature that most improves silhouette.
        
        Parameters:
        -----------
        X : DataFrame
            Feature matrix (deduped, scaled)
        n_greedy : int
            Number of features to greedily select
        n_clusters : int
            Number of clusters for K-means evaluation
        candidate_pool : int
            Number of top individual-silhouette features to search from
            
        Returns:
        --------
        greedy_features : list of str
            Names of greedily selected features
        """
        print(f"\n  Greedy silhouette-optimized selection ({n_greedy} features)...")
        
        feature_names = X.columns.tolist()
        X_arr = X.values
        
        # Score each feature individually by silhouette
        individual_scores = []
        for i in range(len(feature_names)):
            km = KMeans(n_clusters=n_clusters, n_init=5, 
                       random_state=self.random_state, max_iter=50)
            labels = km.fit_predict(X_arr[:, i:i+1])
            sil = silhouette_score(X_arr[:, i:i+1], labels)
            individual_scores.append((feature_names[i], sil, i))
        individual_scores.sort(key=lambda x: -x[1])
        
        # Start with the best single feature
        selected_indices = [individual_scores[0][2]]
        print(f"    Seed: {individual_scores[0][0]} (sil={individual_scores[0][1]:.4f})")
        
        # Greedy forward selection from top candidates
        pool = individual_scores[:candidate_pool]
        for step in range(n_greedy - 1):
            best_sil = -1
            best_idx = -1
            for _, _, idx in pool:
                if idx in selected_indices:
                    continue
                trial = selected_indices + [idx]
                X_trial = X_arr[:, trial]
                km = KMeans(n_clusters=n_clusters, n_init=10, 
                           random_state=self.random_state, max_iter=100)
                labels = km.fit_predict(X_trial)
                sil = silhouette_score(X_trial, labels)
                if sil > best_sil:
                    best_sil = sil
                    best_idx = idx
            selected_indices.append(best_idx)
            n = len(selected_indices)
            if n in [5, 10, 15, 20] or n == n_greedy:
                print(f"    {n:2d} features: sil={best_sil:.4f}  (+{feature_names[best_idx]})")
        
        greedy_features = [feature_names[i] for i in selected_indices]
        return greedy_features
    
    # ------------------------------------------------------------------
    # Hybrid selection: greedy + ensemble
    # ------------------------------------------------------------------
    def hybrid_select(self, X, y, n_greedy=20, n_ensemble_extra=5, 
                     n_clusters=35, candidate_pool=80):
        """
        Two-stage hybrid feature selection combining silhouette-optimized
        greedy forward selection with ensemble-scored genre-discriminative 
        features.
        
        Stage 1: Greedy forward selection maximizing silhouette (cluster 
                 compactness, typically selects rhythmic/temporal features).
        Stage 2: Add top ensemble-scored features not already selected 
                 (genre discrimination, adds timbral/production diversity).
        
        Parameters:
        -----------
        X : DataFrame
            Feature matrix (deduped, scaled)
        y : array-like
            Genre labels
        n_greedy : int
            Number of features from greedy selection (default 20)
        n_ensemble_extra : int
            Number of additional ensemble features to add (default 5)
        n_clusters : int
            Number of clusters for K-means evaluation
        candidate_pool : int
            Search pool size for greedy selection
            
        Returns:
        --------
        X_hybrid : DataFrame
            Selected feature matrix
        """
        n_total = n_greedy + n_ensemble_extra
        
        print("\n" + "="*60)
        print(f"HYBRID FEATURE SELECTION ({n_total} features)")
        print("="*60)
        print(f"  Stage 1: Greedy silhouette optimization ({n_greedy} features)")
        print(f"  Stage 2: Ensemble genre-discriminative top-up (+{n_ensemble_extra})")
        
        # Stage 1: Greedy silhouette selection
        greedy_features = self.greedy_silhouette_selection(
            X, n_greedy=n_greedy, n_clusters=n_clusters,
            candidate_pool=candidate_pool
        )
        
        # Stage 2: Ensemble scoring to find genre-discriminative features
        print(f"\n  Ensemble scoring for genre-discriminative features...")
        ensemble_selector = EDMFeatureSelector(n_features=n_ensemble_extra, random_state=self.random_state)
        ensemble_selector.fit_select(X, y, n_clusters=n_clusters)
        ensemble_ranked = ensemble_selector.feature_scores['feature'].tolist()
        
        # Add top ensemble features not already in greedy set
        extra_features = []
        for feat in ensemble_ranked:
            if feat not in greedy_features:
                extra_features.append(feat)
            if len(extra_features) >= n_ensemble_extra:
                break
        
        # Combine
        hybrid_features = greedy_features + extra_features
        
        print(f"\n  Hybrid selection summary:")
        print(f"    Greedy ({n_greedy}): {', '.join(greedy_features[:5])}...")
        print(f"    Ensemble top-up ({len(extra_features)}): {', '.join(extra_features)}")
        print(f"    Total: {len(hybrid_features)} features")
        
        # Store selection info
        self.selected_features = hybrid_features
        self.hybrid_features = {
            'greedy': greedy_features,
            'ensemble_extra': extra_features,
            'n_greedy': n_greedy,
            'n_ensemble_extra': len(extra_features),
        }
        self.feature_scores = ensemble_selector.feature_scores
        self.selector_fitted = True
        
        return X[hybrid_features]
    
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
        
        info = {
            'n_features_selected': len(self.selected_features),
            'selected_features': self.selected_features,
            'top_features': self.feature_scores.head(10)['feature'].tolist() if self.feature_scores is not None else [],
            'feature_scores': self.feature_scores
        }
        if self.hybrid_features:
            info['hybrid'] = self.hybrid_features
        return info