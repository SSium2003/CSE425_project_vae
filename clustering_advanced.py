"""
Advanced clustering comparison for Medium task
Compares multiple VAE architectures and clustering algorithms
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
import pickle

from clustering import evaluate_clustering, cluster_purity


def load_all_features():
    """Load all feature variants"""
    features_dict = {}
    
    print("="*80)
    print("LOADING ALL FEATURE VARIANTS")
    print("="*80)
    
    # 1. Basic VAE latent features
    if Path('./data/latent_features.npy').exists():
        features_dict['VAE_basic'] = {
            'features': np.load('./data/latent_features.npy'),
            'labels': np.load('./data/labels.npy')
        }
        print(f"✓ Basic VAE: {features_dict['VAE_basic']['features'].shape}")
    
    # 2. ConvVAE latent features
    if Path('./data/conv_latent_features.npy').exists():
        features_dict['VAE_conv'] = {
            'features': np.load('./data/conv_latent_features.npy'),
            'labels': np.load('./data/conv_labels.npy')
        }
        print(f"✓ ConvVAE: {features_dict['VAE_conv']['features'].shape}")
    
    # 3. Multimodal VAE latent features
    if Path('./data/multimodal_latent_features.npy').exists():
        features_dict['VAE_multimodal'] = {
            'features': np.load('./data/multimodal_latent_features.npy'),
            'labels': np.load('./data/multimodal_labels.npy')
        }
        print(f"✓ Multimodal VAE: {features_dict['VAE_multimodal']['features'].shape}")
    
    # 4. PCA baseline
    if Path('./data/processed_features.pkl').exists():
        with open('./data/processed_features.pkl', 'rb') as f:
            data = pickle.load(f)
        
        from dataset import normalize_features
        normalized, _, _ = normalize_features(data['features'])
        
        pca = PCA(n_components=32)
        pca_features = pca.fit_transform(normalized)
        
        features_dict['PCA_baseline'] = {
            'features': pca_features,
            'labels': data['labels']
        }
        print(f"✓ PCA Baseline: {pca_features.shape}")
    
    # 5. Raw audio features
    if Path('./data/processed_features.pkl').exists():
        with open('./data/processed_features.pkl', 'rb') as f:
            data = pickle.load(f)
        
        from dataset import normalize_features
        normalized, _, _ = normalize_features(data['features'])
        
        features_dict['Raw_audio'] = {
            'features': normalized,
            'labels': data['labels']
        }
        print(f"✓ Raw Audio: {normalized.shape}")
    
    # 6. Concatenated features (if available)
    if Path('./data/features_concatenated.pkl').exists():
        with open('./data/features_concatenated.pkl', 'rb') as f:
            data = pickle.load(f)
        
        from dataset import normalize_features
        normalized, _, _ = normalize_features(data['features'])
        
        features_dict['Audio+Text_concat'] = {
            'features': normalized,
            'labels': data['labels']
        }
        print(f"✓ Concatenated: {normalized.shape}")
    
    return features_dict


def run_comprehensive_clustering(features_dict, n_clusters=5):
    """
    Run all clustering methods on all feature types
    
    Returns DataFrame with all results
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE CLUSTERING EVALUATION")
    print("="*80)
    
    all_results = []
    
    for feature_name, data in features_dict.items():
        features = data['features']
        labels_true = data['labels']
        
        print(f"\n{'='*60}")
        print(f"Feature type: {feature_name}")
        print(f"{'='*60}")
        
        # 1. K-Means
        print(f"[1/3] K-Means...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels_kmeans = kmeans.fit_predict(features)
        
        results_kmeans = evaluate_clustering(
            features, labels_true, labels_kmeans, 
            f"{feature_name}+K-Means"
        )
        all_results.append(results_kmeans)
        
        # 2. Agglomerative Clustering
        print(f"[2/3] Agglomerative...")
        agg = AgglomerativeClustering(n_clusters=n_clusters)
        labels_agg = agg.fit_predict(features)
        
        results_agg = evaluate_clustering(
            features, labels_true, labels_agg,
            f"{feature_name}+Agglomerative"
        )
        all_results.append(results_agg)
        
        # 3. DBSCAN (try multiple eps values)
        print(f"[3/3] DBSCAN...")
        # Auto-tune eps based on feature scale
        from sklearn.neighbors import NearestNeighbors
        neighbors = NearestNeighbors(n_neighbors=5)
        neighbors.fit(features)
        distances, _ = neighbors.kneighbors(features)
        eps = np.percentile(distances[:, -1], 90)
        
        dbscan = DBSCAN(eps=eps, min_samples=5)
        labels_dbscan = dbscan.fit_predict(features)
        
        n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
        
        if n_clusters_dbscan > 1:
            # Remove noise points for evaluation
            mask = labels_dbscan != -1
            if mask.sum() > 0:
                results_dbscan = evaluate_clustering(
                    features[mask], labels_true[mask], labels_dbscan[mask],
                    f"{feature_name}+DBSCAN"
                )
                results_dbscan['n_clusters'] = n_clusters_dbscan
                all_results.append(results_dbscan)
        else:
            print(f"  ⚠ DBSCAN found {n_clusters_dbscan} clusters (skipping)")
    
    # Create DataFrame
    results_df = pd.DataFrame(all_results)
    
    return results_df


def analyze_results(results_df):
    """Analyze and print key findings"""
    print("\n" + "="*80)
    print("RESULTS ANALYSIS")
    print("="*80)
    
    # Sort by silhouette score
    results_sorted = results_df.sort_values('silhouette', ascending=False)
    
    print("\n📊 TOP 5 METHODS (by Silhouette Score):")
    print("-"*80)
    top5 = results_sorted.head(5)
    print(top5[['method', 'silhouette', 'calinski_harabasz', 'davies_bouldin']].to_string(index=False))
    
    # Best per feature type
    print("\n📊 BEST CLUSTERING ALGORITHM PER FEATURE TYPE:")
    print("-"*80)
    
    feature_types = results_df['method'].str.split('+').str[0].unique()
    
    for ftype in feature_types:
        ftype_results = results_df[results_df['method'].str.startswith(ftype)]
        if len(ftype_results) > 0:
            best = ftype_results.loc[ftype_results['silhouette'].idxmax()]
            print(f"{ftype:20s} → {best['method']:40s} (Sil: {best['silhouette']:.4f})")
    
    # VAE comparisons
    print("\n📊 VAE ARCHITECTURE COMPARISON (K-Means):")
    print("-"*80)
    
    vae_methods = results_df[results_df['method'].str.contains('VAE')]
    vae_kmeans = vae_methods[vae_methods['method'].str.contains('K-Means')]
    
    if len(vae_kmeans) > 0:
        print(vae_kmeans[['method', 'silhouette', 'adjusted_rand_index', 'normalized_mutual_info']].to_string(index=False))
    
    # Key findings
    print("\n🔍 KEY FINDINGS:")
    print("-"*80)
    
    best_overall = results_sorted.iloc[0]
    print(f"1. Best overall method: {best_overall['method']}")
    print(f"   Silhouette: {best_overall['silhouette']:.4f}")
    print(f"   ARI: {best_overall.get('adjusted_rand_index', 'N/A')}")
    
    # Compare VAE vs PCA
    vae_basic = results_df[results_df['method'] == 'VAE_basic+K-Means']
    pca_baseline = results_df[results_df['method'] == 'PCA_baseline+K-Means']
    
    if len(vae_basic) > 0 and len(pca_baseline) > 0:
        vae_sil = vae_basic['silhouette'].values[0]
        pca_sil = pca_baseline['silhouette'].values[0]
        improvement = ((vae_sil - pca_sil) / pca_sil) * 100
        
        print(f"\n2. VAE vs PCA Baseline:")
        print(f"   VAE: {vae_sil:.4f}")
        print(f"   PCA: {pca_sil:.4f}")
        print(f"   Improvement: {improvement:+.1f}%")
    
    # Multimodal benefit
    vae_multi = results_df[results_df['method'] == 'VAE_multimodal+K-Means']
    if len(vae_multi) > 0 and len(vae_basic) > 0:
        multi_sil = vae_multi['silhouette'].values[0]
        basic_sil = vae_basic['silhouette'].values[0]
        improvement = ((multi_sil - basic_sil) / basic_sil) * 100
        
        print(f"\n3. Multimodal VAE benefit:")
        print(f"   Multimodal: {multi_sil:.4f}")
        print(f"   Basic: {basic_sil:.4f}")
        print(f"   Improvement: {improvement:+.1f}%")


if __name__ == "__main__":
    print("="*80)
    print("ADVANCED CLUSTERING COMPARISON")
    print("="*80)
    
    # Load all features
    features_dict = load_all_features()
    
    if len(features_dict) == 0:
        print("\n⚠ No features found!")
        print("Make sure you've run:")
        print("  - train.py (basic VAE)")
        print("  - train_conv_vae.py (convolutional VAE)")
        print("  - train_multimodal_vae.py (multimodal VAE)")
        exit(1)
    
    # Run comprehensive clustering
    results_df = run_comprehensive_clustering(features_dict, n_clusters=5)
    
    # Save results
    results_df.to_csv('./results/clustering_metrics_all.csv', index=False)
    print(f"\n✓ Saved results to ./results/clustering_metrics_all.csv")
    
    # Analyze
    analyze_results(results_df)
    
    print("\n" + "="*80)
    print("✓ CLUSTERING EVALUATION COMPLETE")
    print("="*80)
    print("\nNext step: Run visualize_advanced.py for comprehensive visualizations")