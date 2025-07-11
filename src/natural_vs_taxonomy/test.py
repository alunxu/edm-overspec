"""
debug_hierarchical_clustering.py
================================
Debug script to identify why hierarchical clustering isn't showing
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

def debug_hierarchical_issue(X, results):
    """
    Debug why hierarchical clustering isn't showing in the plot.
    
    Parameters:
    -----------
    X : array-like
        Your feature matrix
    results : dict
        Your clustering results
    """
    print("=== DEBUGGING HIERARCHICAL CLUSTERING ISSUE ===\n")
    
    # Check 1: Feature matrix
    print("1. Checking feature matrix X:")
    if X is None:
        print("   ❌ ERROR: X is None!")
        return
    
    try:
        X_array = np.array(X)
        print(f"   ✓ Shape: {X_array.shape}")
        print(f"   ✓ Type: {type(X)}")
        print(f"   ✓ Contains NaN: {np.any(np.isnan(X_array))}")
        print(f"   ✓ Contains Inf: {np.any(np.isinf(X_array))}")
    except Exception as e:
        print(f"   ❌ ERROR converting to array: {e}")
        return
    
    # Check 2: Scaling
    print("\n2. Testing StandardScaler:")
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_array)
        print(f"   ✓ Scaling successful")
        print(f"   ✓ Scaled shape: {X_scaled.shape}")
        print(f"   ✓ Scaled mean: {np.mean(X_scaled):.6f} (should be ~0)")
        print(f"   ✓ Scaled std: {np.std(X_scaled):.6f} (should be ~1)")
    except Exception as e:
        print(f"   ❌ ERROR during scaling: {e}")
        return
    
    # Check 3: Results structure
    print("\n3. Checking results dictionary:")
    required_keys = ['k_values', 'metrics', 'optimal_k']
    for key in required_keys:
        if key in results:
            print(f"   ✓ '{key}' found")
        else:
            print(f"   ❌ '{key}' MISSING!")
    
    if 'metrics' in results:
        required_metrics = ['silhouette', 'inertia', 'calinski', 'davies_bouldin']
        for metric in required_metrics:
            if metric in results['metrics']:
                print(f"   ✓ metrics['{metric}'] found")
            else:
                print(f"   ❌ metrics['{metric}'] MISSING!")
    
    # Check 4: Test hierarchical clustering
    print("\n4. Testing hierarchical clustering:")
    try:
        from scipy.cluster.hierarchy import linkage
        # Test with a small subset to see if it works
        n_test = min(100, len(X_scaled))
        X_test = X_scaled[:n_test]
        linkage_matrix = linkage(X_test, method='ward')
        print(f"   ✓ Hierarchical clustering works!")
        print(f"   ✓ Linkage matrix shape: {linkage_matrix.shape}")
    except Exception as e:
        print(f"   ❌ ERROR in hierarchical clustering: {e}")
        return
    
    # Check 5: Import check
    print("\n5. Checking function imports:")
    try:
        from visualization import plot_validation_metrics, plot_with_hierarchical
        print("   ✓ Functions imported successfully")
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        print("   Make sure the visualization file is in the same directory")
    
    print("\n=== RECOMMENDED SOLUTIONS ===")
    print("\nOption 1 - Use the convenience function:")
    print("   from visualization import plot_with_hierarchical")
    print("   plot_with_hierarchical(results, X, save=True)")
    
    print("\nOption 2 - Call directly with scaled features:")
    print("   from sklearn.preprocessing import StandardScaler")
    print("   from visualization import plot_validation_metrics")
    print("   X_scaled = StandardScaler().fit_transform(X)")
    print("   plot_validation_metrics(results, features_scaled=X_scaled, save=True)")
    
    print("\nOption 3 - Test with minimal example:")
    print("   See the test_minimal_hierarchical() function below")
    
    return X_scaled


def test_minimal_hierarchical():
    """
    Test with minimal data to ensure the function works.
    """
    print("\n=== TESTING WITH MINIMAL EXAMPLE ===")
    
    # Create minimal test data
    n_samples = 200
    n_features = 10
    X = np.random.randn(n_samples, n_features)
    
    # Create test results
    k_values = list(range(5, 36))
    results = {
        'k_values': k_values,
        'metrics': {
            'silhouette': np.random.rand(len(k_values)),
            'inertia': np.random.rand(len(k_values)) * 1000,
            'calinski': np.random.rand(len(k_values)) * 100,
            'davies_bouldin': np.random.rand(len(k_values)) * 2
        },
        'optimal_k': 8
    }
    
    # Test both methods
    print("\nTesting method 1 - plot_with_hierarchical:")
    try:
        from visualization import plot_with_hierarchical
        plot_with_hierarchical(results, X, save=False)
        print("✓ Method 1 works!")
    except Exception as e:
        print(f"❌ Method 1 failed: {e}")
    
    print("\nTesting method 2 - direct call:")
    try:
        from visualization import plot_validation_metrics
        from sklearn.preprocessing import StandardScaler
        X_scaled = StandardScaler().fit_transform(X)
        plot_validation_metrics(results, features_scaled=X_scaled, save=False)
        print("✓ Method 2 works!")
    except Exception as e:
        print(f"❌ Method 2 failed: {e}")


def check_your_code():
    """
    Show common mistakes and their fixes.
    """
    print("\n=== COMMON MISTAKES AND FIXES ===\n")
    
    print("❌ WRONG (shows 'Hierarchical clustering data not provided'):")
    print("   plot_validation_metrics(results)")
    print("   plot_validation_metrics(results, save=True)")
    print("   plot_validation_metrics(results, features_scaled=None)")
    
    print("\n✓ CORRECT (shows hierarchical dendrogram):")
    print("   plot_validation_metrics(results, features_scaled=X_scaled)")
    print("   plot_with_hierarchical(results, X)")
    
    print("\n❌ COMMON MISTAKE - Wrong variable name:")
    print("   plot_validation_metrics(results, X)  # Wrong parameter name!")
    print("   plot_validation_metrics(results, features=X_scaled)  # Wrong parameter name!")
    
    print("\n✓ CORRECT parameter name:")
    print("   plot_validation_metrics(results, features_scaled=X_scaled)")


if __name__ == "__main__":
    # Run the debugging
    print("Run this script with your data:")
    print("python debug_hierarchical_clustering.py")
    print("\nOr import and use:")
    print("from debug_hierarchical_clustering import debug_hierarchical_issue")
    print("X_scaled = debug_hierarchical_issue(X, results)")
    
    # Show common mistakes
    check_your_code()
    
    # Test with minimal example
    test_minimal_hierarchical()