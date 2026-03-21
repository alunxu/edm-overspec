"""
evaluation_metrics.py
=====================
Experiment modules for EDM genre analysis
"""

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances, normalized_mutual_info_score, adjusted_rand_score
import warnings
warnings.filterwarnings('ignore')


class NaturalClusterExperiment:
    """
    Experiment 1: Natural Cluster Discovery
    """
    
    def __init__(self):
        self.results = {}
        
    def run(self, X, y, forced_labels, natural_labels, natural_k):
        """
        Run natural cluster discovery experiment.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            True genre labels
        forced_labels : array-like
            Forced 35-cluster labels
        natural_labels : array-like
            Natural cluster labels
        natural_k : int
            Number of natural clusters
            
        Returns:
        --------
        results : dict
            Experiment results
        """
        print("\n" + "="*60)
        print("EXPERIMENT 1: NATURAL CLUSTER DISCOVERY")
        print("="*60)
        
        # Calculate comparison metrics
        metrics = {
            'natural_clusters': natural_k,
            'forced_clusters': 35,
            'difference': 35 - natural_k,
            'natural_vs_forced_nmi': normalized_mutual_info_score(natural_labels, forced_labels),
            'natural_vs_forced_ari': adjusted_rand_score(natural_labels, forced_labels),
            'natural_vs_genres_nmi': normalized_mutual_info_score(y, natural_labels),
            'natural_vs_genres_ari': adjusted_rand_score(y, natural_labels),
            'forced_vs_genres_nmi': normalized_mutual_info_score(y, forced_labels),
            'forced_vs_genres_ari': adjusted_rand_score(y, forced_labels),
        }
        
        # Cluster size analysis
        natural_sizes = np.bincount(natural_labels)
        forced_sizes = np.bincount(forced_labels)
        
        size_stats = {
            'natural_mean_size': natural_sizes.mean(),
            'natural_std_size': natural_sizes.std(),
            'natural_min_size': natural_sizes.min(),
            'natural_max_size': natural_sizes.max(),
            'forced_mean_size': forced_sizes.mean(),
            'forced_std_size': forced_sizes.std(),
            'forced_min_size': forced_sizes.min(),
            'forced_max_size': forced_sizes.max(),
        }
        
        # Print results
        print(f"\nResults:")
        print(f"Natural clusters: {natural_k}")
        print(f"Industry clusters: 35")
        print(f"Over-segmentation: {35 - natural_k} excess categories")
        
        print(f"\nComparison Metrics:")
        print(f"Natural vs Forced - NMI: {metrics['natural_vs_forced_nmi']:.4f}, "
              f"ARI: {metrics['natural_vs_forced_ari']:.4f}")
        print(f"Natural vs Genres - NMI: {metrics['natural_vs_genres_nmi']:.4f}, "
              f"ARI: {metrics['natural_vs_genres_ari']:.4f}")
        print(f"Forced vs Genres  - NMI: {metrics['forced_vs_genres_nmi']:.4f}, "
              f"ARI: {metrics['forced_vs_genres_ari']:.4f}")
        
        print(f"\nCluster Sizes:")
        print(f"Natural - Mean: {size_stats['natural_mean_size']:.1f}, "
              f"Std: {size_stats['natural_std_size']:.1f}")
        print(f"Forced  - Mean: {size_stats['forced_mean_size']:.1f}, "
              f"Std: {size_stats['forced_std_size']:.1f}")
        
        self.results = {
            'metrics': metrics,
            'size_stats': size_stats,
            'natural_labels': natural_labels,
            'natural_k': natural_k
        }
        
        return self.results


class GenreConvergenceExperiment:
    """
    Experiment 2: Genre Convergence Analysis
    """
    
    def __init__(self):
        self.results = {}
        
    def run(self, X, y, forced_labels):
        """
        Run genre convergence analysis experiment.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            True genre labels
        forced_labels : array-like
            Cluster labels
            
        Returns:
        --------
        results : dict
            Experiment results
        """
        print("\n" + "="*60)
        print("EXPERIMENT 2: GENRE CONVERGENCE ANALYSIS")
        print("="*60)
        
        # Calculate convergence matrix
        convergence_matrix = self._calculate_genre_convergence(X, y)
        
        # Find converging pairs
        convergence_pairs = self._find_converging_pairs(convergence_matrix)
        
        # Find phantom distinctions
        phantom_distinctions = self._find_phantom_distinctions(
            convergence_matrix, y, forced_labels
        )
        
        # Genre fragmentation analysis
        genre_entropy = self._analyze_genre_fragmentation(y, forced_labels)
        
        self.results = {
            'convergence_matrix': convergence_matrix,
            'convergence_pairs': convergence_pairs,
            'phantom_distinctions': phantom_distinctions,
            'genre_entropy': genre_entropy
        }
        
        return self.results
    
    def _calculate_genre_convergence(self, X, y):
        """Calculate genre convergence matrix."""
        print("\nCalculating genre convergence matrix...")
        
        unique_genres = sorted(np.unique(y))
        n_genres = len(unique_genres)
        convergence_matrix = np.zeros((n_genres, n_genres))
        
        # Data is already Yeo-Johnson scaled from the pipeline
        X_scaled = X
        
        print(f"Calculating similarities for {n_genres} genres...")
        
        for i, genre1 in enumerate(unique_genres):
            if i % 5 == 0:
                print(f"  Processing genre {i+1}/{n_genres}...")
            
            for j, genre2 in enumerate(unique_genres):
                if i <= j:
                    mask1 = y == genre1
                    mask2 = y == genre2
                    
                    X1 = X_scaled[mask1]
                    X2 = X_scaled[mask2]
                    
                    if len(X1) > 0 and len(X2) > 0:
                        if i == j:
                            # Within-genre similarity
                            if len(X1) > 1:
                                from scipy.spatial.distance import pdist
                                # Use cosine similarity instead of euclidean
                                dists = pdist(X1, metric='cosine')
                                if len(dists) > 0:
                                    avg_dist = dists.mean()
                                    # Convert distance to similarity
                                    similarity = 1 - avg_dist
                                else:
                                    similarity = 1.0
                            else:
                                similarity = 1.0
                        else:
                            # Between-genre similarity
                            # Calculate centroid-based similarity
                            centroid1 = X1.mean(axis=0)
                            centroid2 = X2.mean(axis=0)
                            
                            # Use cosine similarity
                            from sklearn.metrics.pairwise import cosine_similarity
                            similarity = cosine_similarity(
                                centroid1.reshape(1, -1), 
                                centroid2.reshape(1, -1)
                            )[0, 0]
                            
                            # Ensure similarity is in [0, 1]
                            similarity = max(0, min(1, similarity))
                        
                        convergence_matrix[i, j] = similarity
                        convergence_matrix[j, i] = similarity
        
        # Print some statistics
        off_diagonal = convergence_matrix[~np.eye(n_genres, dtype=bool)]
        print(f"\nSimilarity statistics:")
        print(f"  Min (off-diagonal): {off_diagonal.min():.3f}")
        print(f"  Max (off-diagonal): {off_diagonal.max():.3f}")
        print(f"  Mean (off-diagonal): {off_diagonal.mean():.3f}")
        print(f"  Std (off-diagonal): {off_diagonal.std():.3f}")
        
        return pd.DataFrame(convergence_matrix, 
                           index=unique_genres, 
                           columns=unique_genres)
    
    def _find_converging_pairs(self, convergence_matrix):
        """Find top converging genre pairs."""
        print("\nFinding converging genre pairs...")
        
        pairs = []
        genre_names = convergence_matrix.index
        
        for i in range(len(genre_names)):
            for j in range(i+1, len(genre_names)):
                pairs.append({
                    'genre1': genre_names[i],
                    'genre2': genre_names[j],
                    'similarity': convergence_matrix.iloc[i, j]
                })
        
        pairs_df = pd.DataFrame(pairs).sort_values('similarity', ascending=False)
        
        print("\nTop 10 Most Similar Genre Pairs:")
        for idx, row in pairs_df.head(10).iterrows():
            print(f"  {row['genre1']:30s} ↔ {row['genre2']:30s} : {row['similarity']:.3f}")
        
        return pairs_df
    
    def _find_phantom_distinctions(self, convergence_matrix, y, labels,
                                   sim_threshold=0.5, overlap_threshold=0.3):
        """Find phantom distinctions."""
        print("\nFinding phantom distinctions...")
        
        genre_names = convergence_matrix.index
        phantoms = []
        
        for i, genre1 in enumerate(genre_names):
            for j, genre2 in enumerate(genre_names):
                if i < j:
                    similarity = convergence_matrix.loc[genre1, genre2]
                    
                    mask1 = y == genre1
                    mask2 = y == genre2
                    
                    if mask1.sum() > 0 and mask2.sum() > 0:
                        clusters1 = set(labels[mask1])
                        clusters2 = set(labels[mask2])
                        
                        if len(clusters1.union(clusters2)) > 0:
                            overlap = len(clusters1.intersection(clusters2)) / \
                                     len(clusters1.union(clusters2))
                        else:
                            overlap = 0
                        
                        if similarity > sim_threshold and overlap < overlap_threshold:
                            phantoms.append({
                                'genre1': genre1,
                                'genre2': genre2,
                                'acoustic_similarity': similarity,
                                'clustering_overlap': overlap,
                                'phantom_score': similarity * (1 - overlap)
                            })
        
        if phantoms:
            phantom_df = pd.DataFrame(phantoms).sort_values('phantom_score', 
                                                           ascending=False)
            print(f"\nFound {len(phantom_df)} phantom distinctions")
            print("Top 5:")
            for idx, row in phantom_df.head(5).iterrows():
                print(f"  {row['genre1']} vs {row['genre2']}")
            
            return phantom_df
        else:
            print("No phantom distinctions found")
            return None
    
    def _analyze_genre_fragmentation(self, y, labels):
        """Analyze how fragmented genres are across clusters."""
        print("\nAnalyzing genre fragmentation...")
        
        genre_confusion = pd.crosstab(y, labels, normalize='index')
        
        genre_stats = []
        for genre in genre_confusion.index:
            probs = genre_confusion.loc[genre].values
            probs = probs[probs > 0]
            
            if len(probs) > 0:
                entropy = -np.sum(probs * np.log2(probs))
                n_clusters = (genre_confusion.loc[genre] > 0).sum()
                dominant_pct = probs.max() * 100
                
                genre_stats.append({
                    'genre': genre,
                    'entropy': entropy,
                    'n_clusters': n_clusters,
                    'dominant_cluster_pct': dominant_pct
                })
        
        genre_entropy_df = pd.DataFrame(genre_stats).sort_values('entropy', 
                                                                 ascending=False)
        
        print("\nMost fragmented genres:")
        for idx, row in genre_entropy_df.head(5).iterrows():
            print(f"  {row['genre']:30s} : {row['n_clusters']} clusters, "
                  f"entropy={row['entropy']:.2f}")
        
        return genre_entropy_df