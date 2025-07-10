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
    Find natural number of clusters using multiple validation methods.
    """
    
    def __init__(self, min_clusters=5, max_clusters=35, random_state=42):
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.random_state = random_state
        self.results = {}
        
    def find_optimal_k(self, X):
        """
        Find optimal number of clusters using multiple methods.
        
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
        print("FINDING NATURAL CLUSTERS")
        print("="*60)
        print(f"Testing range: {self.min_clusters} to {self.max_clusters} clusters")
        
        # Compute hierarchical clustering
        print("\nComputing hierarchical clustering...")
        linkage_matrix = linkage(X, method='ward')
        
        # Test different k values
        max_k = min(self.max_clusters + 1, len(X) // 10)
        k_values = list(range(self.min_clusters, max_k))
        
        # Calculate metrics
        metrics = self._calculate_all_metrics(X, linkage_matrix, k_values)
        
        # Find optimal k using different methods
        optimal_methods = self._find_optimal_by_methods(k_values, metrics)
        
        # Consensus approach
        optimal_k = self._calculate_consensus(optimal_methods)
        
        # Store results
        self.results = {
            'optimal_k': optimal_k,
            'linkage_matrix': linkage_matrix,
            'k_values': k_values,
            'metrics': metrics,
            'method_results': optimal_methods
        }
        
        return optimal_k, self.results
    
    def _calculate_all_metrics(self, X, linkage_matrix, k_values):
        """Calculate all clustering metrics for different k values."""
        metrics = {
            'silhouette': [],
            'calinski': [],
            'davies_bouldin': [],
            'inertia': []
        }
        
        print("Evaluating different cluster numbers...")
        for k in k_values:
            if k % 5 == 0:
                print(f"  Testing k={k}...")
            
            labels = fcluster(linkage_matrix, k, criterion='maxclust')
            
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
            
            # Calculate inertia (within-cluster sum of squares)
            inertia = 0.0
            for i in np.unique(labels):
                cluster_mask = labels == i
                cluster_points = X[cluster_mask]
                if len(cluster_points) > 0:
                    # Ensure we're working with numpy arrays
                    cluster_array = np.array(cluster_points)
                    center = cluster_array.mean(axis=0)
                    # Calculate squared distances
                    squared_distances = np.sum((cluster_array - center) ** 2)
                    inertia += float(squared_distances)
            
            metrics['inertia'].append(inertia)
        
        # Convert to numpy arrays and ensure they are 1D
        for key in metrics:
            metrics[key] = np.array(metrics[key], dtype=float).flatten()
        
        return metrics
    
    def _find_optimal_by_methods(self, k_values, metrics):
        """Find optimal k using different methods."""
        results = {}
        
        # Elbow method
        results['elbow'] = self._find_elbow_point(k_values, metrics['inertia'])
        
        # Silhouette method
        results['silhouette'] = k_values[np.argmax(metrics['silhouette'])]
        
        # Calinski-Harabasz
        results['calinski'] = k_values[np.argmax(metrics['calinski'])]
        
        # Davies-Bouldin
        results['davies_bouldin'] = k_values[np.argmin(metrics['davies_bouldin'])]
        
        # Gap statistic
        results['gap'] = self._calculate_gap_statistic(k_values, metrics['inertia'])
        
        return results
    
    def _find_elbow_point(self, k_values, inertias):
        """Find elbow point in inertia curve."""
        if len(inertias) < 3:
            return k_values[0]
        
        # Convert to numpy arrays and ensure 1D float arrays
        x = np.array(k_values, dtype=float).flatten()
        y = np.array(inertias, dtype=float).flatten()
        
        # Validate data
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
        
        # Calculate distances from line
        distances = []
        
        # Line from first to last point
        p1 = np.array([x_norm[0], y_norm[0]])
        p2 = np.array([x_norm[-1], y_norm[-1]])
        
        for i in range(1, len(x_norm) - 1):
            # Current point
            p0 = np.array([x_norm[i], y_norm[i]])
            
            # Calculate perpendicular distance from point to line
            # Using the formula: |det([p2-p1, p1-p0])| / |p2-p1|
            numerator = np.abs((p2[1] - p1[1]) * (p1[0] - p0[0]) - 
                              (p1[1] - p0[1]) * (p2[0] - p1[0]))
            denominator = np.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)
            
            if denominator > 0:
                distance = numerator / denominator
            else:
                distance = 0
            
            distances.append(distance)
        
        if distances:
            elbow_idx = np.argmax(distances) + 1
            return k_values[elbow_idx]
        
        return k_values[len(k_values)//3]
    
    def _calculate_gap_statistic(self, k_values, inertias):
        """Simplified gap statistic calculation."""
        # For large datasets, use a simple heuristic
        # Find where the rate of change decreases most
        if len(inertias) > 2:
            rates = -np.diff(inertias)
            rate_changes = -np.diff(rates)
            if len(rate_changes) > 0:
                best_idx = np.argmax(rate_changes) + 2
                if best_idx < len(k_values):
                    return k_values[best_idx]
        
        return k_values[len(k_values)//3]
    
    def _calculate_consensus(self, method_results):
        """Calculate consensus optimal k from multiple methods."""
        candidates = list(method_results.values())
        candidates = [k for k in candidates 
                     if k is not None and self.min_clusters <= k <= self.max_clusters]
        
        if not candidates:
            return self.min_clusters
        
        # Weighted voting
        weights = {
            'elbow': 2.0,
            'silhouette': 1.5,
            'calinski': 1.0,
            'davies_bouldin': 1.0,
            'gap': 1.5
        }
        
        weighted_sum = 0
        total_weight = 0
        
        for method, k in method_results.items():
            if k in candidates:
                weight = weights.get(method, 1.0)
                weighted_sum += k * weight
                total_weight += weight
        
        optimal_k = int(round(weighted_sum / total_weight))
        optimal_k = max(self.min_clusters, min(optimal_k, self.max_clusters))
        
        print(f"\nOptimal k by different methods:")
        for method, k in method_results.items():
            print(f"  {method:15s}: {k}")
        print(f"  {'CONSENSUS':15s}: {optimal_k} ← Selected")
        
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
            Clustering method ('kmeans' or 'hierarchical')
            
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
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return labels
    
    def find_natural_clusters(self, X, min_k=5, max_k=20):
        """
        Find natural clusters in the data.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        min_k : int
            Minimum clusters to test
        max_k : int
            Maximum clusters to test
            
        Returns:
        --------
        labels : array
            Natural cluster labels
        n_clusters : int
            Number of natural clusters found
        """
        self.natural_clusterer = NaturalClusterFinder(min_k, max_k, self.random_state)
        optimal_k, results = self.natural_clusterer.find_optimal_k(X)
        
        # Get labels for optimal k
        linkage_matrix = results['linkage_matrix']
        labels = fcluster(linkage_matrix, optimal_k, criterion='maxclust')
        
        return labels, optimal_k
    
    def evaluate_clustering(self, X, labels, y_true=None):
        """
        Evaluate clustering quality.
        
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
            Clustering quality metrics
        """
        metrics = {}
        
        n_clusters = len(np.unique(labels))
        metrics['n_clusters'] = n_clusters
        
        if n_clusters > 1:
            metrics['silhouette'] = silhouette_score(X, labels)
            metrics['calinski_harabasz'] = calinski_harabasz_score(X, labels)
            metrics['davies_bouldin'] = davies_bouldin_score(X, labels)
        
        if y_true is not None:
            from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
            metrics['nmi'] = normalized_mutual_info_score(y_true, labels)
            metrics['ari'] = adjusted_rand_score(y_true, labels)
        
        return metrics