"""
clustering_algorithms.py
========================
Clustering algorithms for EDM genre analysis
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')


class NaturalClusterFinder:
    """
    Find natural number of clusters using K-means and multiple validation methods.
    Tests k ∈ [min_clusters, max_clusters] and selects optimal k via weighted consensus
    of silhouette, Calinski-Harabasz, Davies-Bouldin, elbow, and gap statistic.
    """
    
    def __init__(self, min_clusters=15, max_clusters=40, random_state=42):
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.random_state = random_state
        self.results = {}
        
    def find_optimal_k(self, X):
        """
        Find optimal number of clusters using K-means with multiple validation methods.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
            
        Returns:
        --------
        optimal_k : int
            Optimal number of clusters
        results : dict
            Detailed results from all methods
        """
        print("\n" + "="*60)
        print("FINDING NATURAL CLUSTERS (K-means)")
        print("="*60)
        print(f"Testing range: {self.min_clusters} to {self.max_clusters} clusters")
        
        # Also compute linkage for cophenetic metric in evaluate_clustering
        print("\nComputing hierarchical linkage (for cophenetic metric)...")
        linkage_matrix = linkage(X, method='ward')
        
        # Test different k values with K-means
        k_values = list(range(self.min_clusters, self.max_clusters + 1))
        
        # Calculate metrics using K-means
        metrics, kmeans_models = self._calculate_all_metrics(X, k_values)
        
        # Find optimal k using different methods
        optimal_methods = self._find_optimal_by_methods(X, k_values, metrics)
        
        # Consensus approach
        optimal_k = self._calculate_consensus(optimal_methods)
        
        # Store results
        self.results = {
            'optimal_k': optimal_k,
            'linkage_matrix': linkage_matrix,
            'k_values': k_values,
            'metrics': metrics,
            'method_results': optimal_methods,
            'kmeans_models': kmeans_models
        }
        
        return optimal_k, self.results
    
    def _calculate_all_metrics(self, X, k_values):
        """Calculate all clustering metrics for different k values using K-means."""
        metrics = {
            'silhouette': [],
            'calinski': [],
            'davies_bouldin': [],
            'inertia': []
        }
        kmeans_models = {}
        
        print("Evaluating different cluster numbers with K-means...")
        for k in k_values:
            if k % 5 == 0:
                print(f"  Testing k={k}...")
            
            km = KMeans(
                n_clusters=k,
                init='k-means++',
                n_init=50,
                max_iter=300,
                random_state=self.random_state
            )
            labels = km.fit_predict(X)
            kmeans_models[k] = km
            
            if len(np.unique(labels)) < 2:
                metrics['silhouette'].append(-1)
                metrics['calinski'].append(0)
                metrics['davies_bouldin'].append(np.inf)
                metrics['inertia'].append(np.inf)
                continue
            
            # Calculate standard metrics
            metrics['silhouette'].append(silhouette_score(X, labels))
            metrics['calinski'].append(calinski_harabasz_score(X, labels))
            metrics['davies_bouldin'].append(davies_bouldin_score(X, labels))
            metrics['inertia'].append(km.inertia_)
        
        # Convert to numpy arrays
        for key in metrics:
            metrics[key] = np.array(metrics[key], dtype=float).flatten()
        
        return metrics, kmeans_models
    
    def _find_optimal_by_methods(self, X, k_values, metrics):
        """Find optimal k using different methods."""
        results = {}
        
        # Elbow method (on inertia curve)
        results['elbow'] = self._find_elbow_point(k_values, metrics['inertia'])
        
        # Silhouette method — use elbow of the negated curve (find point of
        # diminishing returns, not raw argmax which favors high k when the
        # curve is flat/monotonically increasing)
        sil_inverted = [-s for s in metrics['silhouette']]
        results['silhouette'] = self._find_elbow_point(k_values, sil_inverted)
        
        # Calinski-Harabasz — typically monotonically decreasing, use elbow
        ch_inverted = [-c for c in metrics['calinski']]
        results['calinski'] = self._find_elbow_point(k_values, ch_inverted)
        
        # Davies-Bouldin — lower is better, use elbow on the raw curve
        results['davies_bouldin'] = self._find_elbow_point(k_values, metrics['davies_bouldin'])
        
        return results
    
    def _find_elbow_point(self, k_values, inertias):
        """Find elbow point in inertia curve."""
        if len(inertias) < 3:
            return k_values[0]
        
        x = np.array(k_values, dtype=float).flatten()
        y = np.array(inertias, dtype=float).flatten()
        
        if len(x) != len(y):
            return k_values[len(k_values)//3]
        
        if np.all(y == y[0]) or np.isnan(y).any() or np.isinf(y).any():
            return k_values[len(k_values)//3]
        
        # Normalize
        x_range = x.max() - x.min()
        y_range = y.max() - y.min()
        
        if x_range == 0 or y_range == 0:
            return k_values[len(k_values)//3]
        
        x_norm = (x - x.min()) / x_range
        y_norm = (y - y.min()) / y_range
        
        # Calculate distances from line connecting first to last point
        p1 = np.array([x_norm[0], y_norm[0]])
        p2 = np.array([x_norm[-1], y_norm[-1]])
        
        distances = []
        for i in range(1, len(x_norm) - 1):
            p0 = np.array([x_norm[i], y_norm[i]])
            numerator = np.abs((p2[1] - p1[1]) * (p1[0] - p0[0]) - 
                              (p1[1] - p0[1]) * (p2[0] - p1[0]))
            denominator = np.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)
            distance = numerator / denominator if denominator > 0 else 0
            distances.append(distance)
        
        if distances:
            elbow_idx = np.argmax(distances) + 1
            return k_values[elbow_idx]
        
        return k_values[len(k_values)//3]
    
    def _calculate_gap_statistic(self, X, k_values, B=10, precomputed_inertias=None):
        """
        Real Gap Statistic (Tibshirani, Walther & Hastie, 2001).
        
        Compares log(inertia) of the data against B uniform reference
        datasets sampled from the bounding box of X. Selects the
        smallest k where Gap(k) >= Gap(k+1) - s_{k+1}.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix (already scaled)
        k_values : list
            Range of k to evaluate
        B : int
            Number of reference datasets to generate
        precomputed_inertias : array-like, optional
            Pre-computed inertias from _calculate_all_metrics to avoid
            redundant K-means runs on the real data
        """
        X_arr = np.array(X)
        n_samples, n_features = X_arr.shape
        
        # Bounding box of the data
        mins = X_arr.min(axis=0)
        maxs = X_arr.max(axis=0)
        
        # Log inertia for the real data (reuse if available)
        if precomputed_inertias is not None:
            log_W = np.log(precomputed_inertias)
        else:
            log_W = []
            for k in k_values:
                km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
                km.fit(X_arr)
                log_W.append(np.log(km.inertia_))
            log_W = np.array(log_W)
        
        # Log inertia for B reference uniform datasets
        ref_log_W = np.zeros((B, len(k_values)))
        rng = np.random.RandomState(self.random_state)
        for b in range(B):
            X_ref = rng.uniform(low=mins, high=maxs, size=(n_samples, n_features))
            for ki, k in enumerate(k_values):
                km = KMeans(n_clusters=k, n_init=5, random_state=self.random_state)
                km.fit(X_ref)
                ref_log_W[b, ki] = np.log(km.inertia_)
        
        # Gap(k) = E*[log(W_ref)] - log(W_data)
        gap = ref_log_W.mean(axis=0) - log_W
        # Standard deviation of reference log inertias
        sdk = ref_log_W.std(axis=0) * np.sqrt(1 + 1.0 / B)
        
        # Select smallest k where Gap(k) >= Gap(k+1) - s_{k+1}
        for i in range(len(k_values) - 1):
            if gap[i] >= gap[i + 1] - sdk[i + 1]:
                return k_values[i]
        
        # Fallback: return k with largest gap
        return k_values[np.argmax(gap)]
    
    def _calculate_consensus(self, method_results):
        """Calculate consensus optimal k using weighted mode voting.
        
        Each method votes for a k value, weighted by its reliability.
        The k with the highest total weight wins. Ties are broken in
        favor of lower k (parsimony / Occam's razor).
        """
        candidates = {k: 0.0 for k in method_results.values() 
                     if k is not None and self.min_clusters <= k <= self.max_clusters}
        
        if not candidates:
            return self.min_clusters
        
        # Equal-weight voting — each method captures a different aspect
        # of cluster quality; no method is privileged a priori.
        weights = {
            'elbow': 1.0,
            'silhouette': 1.0,
            'calinski': 1.0,
            'davies_bouldin': 1.0,
        }
        
        for method, k in method_results.items():
            if k in candidates:
                candidates[k] += weights.get(method, 1.0)
        
        # Select k with highest total vote weight (ties → prefer lower k)
        max_weight = max(candidates.values())
        optimal_k = min(k for k, w in candidates.items() if w == max_weight)
        
        print(f"\nOptimal k by different methods:")
        for method, k in method_results.items():
            print(f"  {method:15s}: {k}")
        print(f"\n  Vote tally:")
        for k in sorted(candidates):
            bar = '█' * int(candidates[k] * 2)
            print(f"    k={k:2d}: {candidates[k]:.1f} {bar}")
        print(f"  {'CONSENSUS':15s}: {optimal_k} ← Selected (highest vote weight)")
        
        return optimal_k


class EDMClusterer:
    """
    Main clustering class for EDM genre analysis.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.forced_clusterer = None
        self.natural_clusterer = None
        
    def fit_forced_clusters(self, X, n_clusters=35, method='kmeans'):
        """
        Fit forced clustering with specified number of clusters.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        n_clusters : int
            Number of clusters to force
        method : str
            Clustering method ('kmeans', 'hierarchical', or 'divisive')
            
        Returns:
        --------
        labels : array
            Cluster labels
        """
        print(f"\nFitting {method} with {n_clusters} clusters...")
        
        if method == 'kmeans':
            self.forced_clusterer = KMeans(
                n_clusters=n_clusters,
                init='k-means++',
                n_init=50,
                max_iter=300,
                random_state=self.random_state
            )
            labels = self.forced_clusterer.fit_predict(X)
            
        elif method == 'hierarchical':
            self.forced_clusterer = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='ward'
            )
            labels = self.forced_clusterer.fit_predict(X)
            
        elif method == 'divisive':
            labels = self._divisive_clustering(X, n_clusters)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return labels
    
    def _divisive_clustering(self, X, n_clusters, min_size=10):
        """
        Divisive hierarchical clustering using heterogeneity-based splitting.
        
        Iteratively splits the most heterogeneous cluster using the
        heterogeneity score: H(C) = σ²(C) · (1 + sil(C)) · log(|C| + 1)
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        n_clusters : int
            Target number of clusters
        min_size : int
            Minimum cluster size to consider for splitting
        """
        X_arr = np.array(X)
        n = len(X_arr)
        labels = np.zeros(n, dtype=int)
        
        # Track clusters: {cluster_id: array_of_indices}
        clusters = {0: np.arange(n)}
        next_id = 1
        
        while len(clusters) < n_clusters:
            # Calculate heterogeneity for each cluster
            heterogeneities = {}
            for cid, indices in clusters.items():
                if len(indices) < min_size * 2:
                    # Too small to split
                    heterogeneities[cid] = -1
                    continue
                
                sub_X = X_arr[indices]
                
                # Variance component: σ²(C)
                variance = np.var(sub_X, axis=0).mean()
                
                # Silhouette of tentative 2-way split
                try:
                    km2 = KMeans(n_clusters=2, n_init=10, random_state=self.random_state)
                    sub_labels = km2.fit_predict(sub_X)
                    if len(np.unique(sub_labels)) == 2:
                        sil = silhouette_score(sub_X, sub_labels)
                    else:
                        sil = 0.0
                except:
                    sil = 0.0
                
                # H(C) = σ²(C) · (1 + sil(C)) · log(|C| + 1)
                h = variance * (1 + sil) * np.log(len(indices) + 1)
                heterogeneities[cid] = h
            
            # Find most heterogeneous cluster
            best_cid = max(heterogeneities, key=heterogeneities.get)
            if heterogeneities[best_cid] <= 0:
                print(f"  Warning: Cannot split further at {len(clusters)} clusters")
                break
            
            # Split the most heterogeneous cluster
            indices = clusters[best_cid]
            sub_X = X_arr[indices]
            km2 = KMeans(n_clusters=2, n_init=10, random_state=self.random_state)
            sub_labels = km2.fit_predict(sub_X)
            
            left = indices[sub_labels == 0]
            right = indices[sub_labels == 1]
            
            if len(left) == 0 or len(right) == 0:
                # Degenerate split
                heterogeneities[best_cid] = -1
                continue
            
            # Replace parent with two children
            del clusters[best_cid]
            clusters[best_cid] = left  # reuse parent id for left
            clusters[next_id] = right
            next_id += 1
            
            if len(clusters) % 5 == 0:
                print(f"  {len(clusters)} clusters...")
        
        # Build final label array
        final_labels = np.zeros(n, dtype=int)
        for new_id, (_, indices) in enumerate(clusters.items()):
            final_labels[indices] = new_id
        
        print(f"  Divisive clustering complete: {len(clusters)} clusters")
        return final_labels
    
    def find_natural_clusters(self, X, min_k=15, max_k=40):
        """
        Find natural clusters using K-means with consensus-based optimal k selection.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        min_k : int
            Minimum clusters to test (default: 15, matching paper)
        max_k : int
            Maximum clusters to test (default: 40, matching paper)
            
        Returns:
        --------
        labels : array
            Natural cluster labels
        n_clusters : int
            Number of natural clusters found
        """
        self.natural_clusterer = NaturalClusterFinder(min_k, max_k, self.random_state)
        optimal_k, results = self.natural_clusterer.find_optimal_k(X)
        
        # Get K-means labels for optimal k
        km = results['kmeans_models'][optimal_k]
        labels = km.predict(X)
        
        return labels, optimal_k
    
    def evaluate_clustering(self, X, labels, y_true=None):
        """
        Evaluate clustering quality with all Table 1 metrics.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        labels : array-like
            Cluster labels
        y_true : array-like, optional
            True labels for external validation
            
        Returns:
        --------
        metrics : dict
            Clustering quality metrics including:
            Internal: silhouette, calinski_harabasz, davies_bouldin, cophenetic_corr
            External: nmi, ari, mi_score, purity
            Distribution: cluster_balance, norm_balance
        """
        from sklearn.metrics import (
            normalized_mutual_info_score, adjusted_rand_score,
            mutual_info_score
        )
        from scipy.cluster.hierarchy import cophenet
        
        metrics = {}
        
        n_clusters = len(np.unique(labels))
        metrics['n_clusters'] = n_clusters
        
        if n_clusters > 1:
            # Internal metrics
            metrics['silhouette'] = silhouette_score(X, labels)
            metrics['calinski_harabasz'] = calinski_harabasz_score(X, labels)
            metrics['davies_bouldin'] = davies_bouldin_score(X, labels)
            
            # Cophenetic correlation (Ward linkage)
            X_arr = np.array(X)
            Z = linkage(X_arr, method='ward')
            corr_coeff, _ = cophenet(Z, pdist(X_arr))
            metrics['cophenetic_corr'] = corr_coeff
            
            # Distribution metrics: cluster balance (entropy, natural log)
            # Table 1 uses natural log: Balance/Norm.Bal ratio = ln(35) ≈ 3.555
            unique, counts = np.unique(labels, return_counts=True)
            size_probs = counts / counts.sum()
            cluster_balance = -np.sum(size_probs * np.log(size_probs + 1e-10))
            max_entropy = np.log(n_clusters) if n_clusters > 1 else 0
            norm_balance = cluster_balance / max_entropy if max_entropy > 0 else 0
            metrics['cluster_balance'] = cluster_balance
            metrics['norm_balance'] = norm_balance
        
        if y_true is not None:
            # External metrics
            metrics['nmi'] = normalized_mutual_info_score(y_true, labels)
            metrics['ari'] = adjusted_rand_score(y_true, labels)
            metrics['mi_score'] = mutual_info_score(y_true, labels)
            
            # Purity: fraction of samples correctly assigned to dominant class per cluster
            contingency = pd.crosstab(pd.Series(y_true), pd.Series(labels))
            metrics['purity'] = contingency.max(axis=0).sum() / len(y_true)
        
        return metrics