import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, 
    calinski_harabasz_score, 
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
import pandas as pd


def cluster_purity(labels_true, labels_pred):
    """
    Compute cluster purity
    
    Args:
        labels_true: Ground truth labels
        labels_pred: Predicted cluster labels
        
    Returns:
        purity: Cluster purity score
    """
    # Contingency matrix
    contingency_matrix = np.zeros((len(np.unique(labels_true)), 
                                   len(np.unique(labels_pred))))
    
    for i, true_label in enumerate(np.unique(labels_true)):
        for j, pred_label in enumerate(np.unique(labels_pred)):
            contingency_matrix[i, j] = np.sum(
                (labels_true == true_label) & (labels_pred == pred_label)
            )
    
    # Purity is the sum of max values in each column divided by total
    purity = np.sum(np.max(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
    
    return purity


def evaluate_clustering(features, labels_true, labels_pred, method_name="Method"):
    """
    Evaluate clustering quality with multiple metrics
    
    Args:
        features: Feature vectors
        labels_true: Ground truth genre labels
        labels_pred: Predicted cluster labels
        method_name: Name of the method
        
    Returns:
        results: Dictionary of evaluation metrics
    """
    results = {
        'method': method_name,
        'silhouette': silhouette_score(features, labels_pred),
        'calinski_harabasz': calinski_harabasz_score(features, labels_pred),
        'davies_bouldin': davies_bouldin_score(features, labels_pred),
    }
    
    # If we have ground truth labels, compute additional metrics
    if labels_true is not None:
        results['adjusted_rand_index'] = adjusted_rand_score(labels_true, labels_pred)
        results['normalized_mutual_info'] = normalized_mutual_info_score(labels_true, labels_pred)
        results['purity'] = cluster_purity(labels_true, labels_pred)
    
    return results


class ClusteringPipeline:
    """Pipeline for clustering experiments"""
    
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.results = []
        
    def run_kmeans(self, features, labels_true=None, random_state=42):
        """Run K-Means clustering"""
        print(f"Running K-Means (k={self.n_clusters})...")
        
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=random_state, n_init=10)
        labels_pred = kmeans.fit_predict(features)
        
        results = evaluate_clustering(features, labels_true, labels_pred, "K-Means")
        self.results.append(results)
        
        return labels_pred, results
    
    def run_agglomerative(self, features, labels_true=None):
        """Run Agglomerative Clustering"""
        print(f"Running Agglomerative Clustering (k={self.n_clusters})...")
        
        agg = AgglomerativeClustering(n_clusters=self.n_clusters)
        labels_pred = agg.fit_predict(features)
        
        results = evaluate_clustering(features, labels_true, labels_pred, "Agglomerative")
        self.results.append(results)
        
        return labels_pred, results
    
    def run_dbscan(self, features, labels_true=None, eps=0.5, min_samples=5):
        """Run DBSCAN clustering"""
        print(f"Running DBSCAN (eps={eps}, min_samples={min_samples})...")
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels_pred = dbscan.fit_predict(features)
        
        # DBSCAN can produce -1 for noise points
        n_clusters = len(set(labels_pred)) - (1 if -1 in labels_pred else 0)
        print(f"  Found {n_clusters} clusters")
        
        if n_clusters > 1:
            results = evaluate_clustering(features, labels_true, labels_pred, "DBSCAN")
            self.results.append(results)
        else:
            results = {'method': 'DBSCAN', 'error': 'Too few clusters'}
            
        return labels_pred, results
    
    def run_baseline_pca_kmeans(self, features, labels_true=None, n_components=32, random_state=42):
        """
        Baseline: PCA + K-Means
        
        Args:
            features: Original features
            labels_true: Ground truth labels
            n_components: Number of PCA components
            random_state: Random seed
            
        Returns:
            labels_pred: Predicted cluster labels
            results: Evaluation results
        """
        print(f"Running Baseline: PCA ({n_components} components) + K-Means...")
        
        # Apply PCA
        pca = PCA(n_components=n_components, random_state=random_state)
        features_pca = pca.fit_transform(features)
        
        print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
        
        # Apply K-Means
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=random_state, n_init=10)
        labels_pred = kmeans.fit_predict(features_pca)
        
        results = evaluate_clustering(features_pca, labels_true, labels_pred, "PCA+K-Means")
        self.results.append(results)
        
        return labels_pred, results, features_pca
    
    def print_results(self):
        """Print all results in a formatted table"""
        if not self.results:
            print("No results to display")
            return
        
        df = pd.DataFrame(self.results)
        
        print("\n" + "="*80)
        print("CLUSTERING EVALUATION RESULTS")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)
        
        # Print interpretation
        print("\nMetric Interpretation:")
        print("  Silhouette Score: Higher is better (range: -1 to 1)")
        print("  Calinski-Harabasz: Higher is better")
        print("  Davies-Bouldin: Lower is better")
        print("  Adjusted Rand Index: Higher is better (range: -1 to 1)")
        print("  Normalized Mutual Info: Higher is better (range: 0 to 1)")
        print("  Purity: Higher is better (range: 0 to 1)")
        
        return df
    
    def save_results(self, path='./results/clustering_metrics.csv'):
        """Save results to CSV"""
        if not self.results:
            print("No results to save")
            return
        
        df = pd.DataFrame(self.results)
        df.to_csv(path, index=False)
        print(f"\nResults saved to {path}")


def compare_methods(vae_features, pca_features, original_features, labels_true, n_clusters=5):
    """
    Compare VAE, PCA, and baseline clustering
    
    Args:
        vae_features: Latent features from VAE
        pca_features: PCA-reduced features
        original_features: Original features
        labels_true: Ground truth labels
        n_clusters: Number of clusters
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE COMPARISON: VAE vs PCA vs Original Features")
    print("="*80)
    
    all_results = []
    
    # 1. VAE + K-Means
    print("\n1. VAE Latent Features + K-Means")
    kmeans_vae = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_vae = kmeans_vae.fit_predict(vae_features)
    results_vae = evaluate_clustering(vae_features, labels_true, labels_vae, "VAE+K-Means")
    all_results.append(results_vae)
    
    # 2. PCA + K-Means (Baseline)
    print("\n2. PCA Features + K-Means (Baseline)")
    kmeans_pca = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_pca = kmeans_pca.fit_predict(pca_features)
    results_pca = evaluate_clustering(pca_features, labels_true, labels_pca, "PCA+K-Means")
    all_results.append(results_pca)
    
    # 3. Original Features + K-Means
    print("\n3. Original Features + K-Means")
    kmeans_orig = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_orig = kmeans_orig.fit_predict(original_features)
    results_orig = evaluate_clustering(original_features, labels_true, labels_orig, "Original+K-Means")
    all_results.append(results_orig)
    
    # Print comparison table
    df = pd.DataFrame(all_results)
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    return df, labels_vae, labels_pca, labels_orig


if __name__ == "__main__":
    # Example usage
    print("Clustering Evaluation Pipeline")
    print("="*80)
    
    # Load data
    latent_features = np.load('./data/latent_features.npy')
    labels = np.load('./data/labels.npy')
    
    print(f"Latent features shape: {latent_features.shape}")
    print(f"Number of samples: {len(latent_features)}")
    print(f"Number of unique labels: {len(np.unique(labels))}")
    
    # Create clustering pipeline
    pipeline = ClusteringPipeline(n_clusters=5)
    
    # Run different clustering methods
    kmeans_labels, kmeans_results = pipeline.run_kmeans(latent_features, labels)
    
    # Run baseline
    pca_labels, pca_results, pca_features = pipeline.run_baseline_pca_kmeans(
        latent_features, labels, n_components=32
    )
    
    # Print all results
    results_df = pipeline.print_results()
    
    # Save results
    pipeline.save_results('./results/clustering_metrics.csv')
    
    print("\n✓ Clustering evaluation complete!")