"""
Comprehensive evaluation for Hard Task
Includes all Beta-VAE variants and complete metrics
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
import pickle

from clustering_easy import evaluate_clustering, cluster_purity


def load_all_features_hard():
    """Load all feature variants including Beta-VAEs"""
    features_dict = {}
    
    print("="*80)
    print("LOADING ALL FEATURE VARIANTS (INCLUDING BETA-VAE)")
    print("="*80)
    
    # Load labels
    labels = np.load('./data/labels.npy')
    
    # 1. Basic VAE
    if Path('./data/latent_features.npy').exists():
        features_dict['VAE_basic'] = {
            'features': np.load('./data/latent_features.npy'),
            'labels': labels
        }
        print(f"✓ Basic VAE: {features_dict['VAE_basic']['features'].shape}")
    
    # 2. ConvVAE
    if Path('./data/conv_latent_features.npy').exists():
        features_dict['VAE_conv'] = {
            'features': np.load('./data/conv_latent_features.npy'),
            'labels': np.load('./data/conv_labels.npy')
        }
        print(f"✓ ConvVAE: {features_dict['VAE_conv']['features'].shape}")
    
    # 3. Multimodal VAE
    if Path('./data/multimodal_latent_features.npy').exists():
        features_dict['VAE_multimodal'] = {
            'features': np.load('./data/multimodal_latent_features.npy'),
            'labels': np.load('./data/multimodal_labels.npy')
        }
        print(f"✓ Multimodal VAE: {features_dict['VAE_multimodal']['features'].shape}")
    
    # 4. Beta-VAEs (multiple beta values)
    beta_values = [0.5, 1.0, 4.0, 10.0]
    beta_labels = np.load('./data/beta_vae_labels.npy') if Path('./data/beta_vae_labels.npy').exists() else labels
    
    for beta in beta_values:
        beta_file = f'./data/beta_vae_latent_beta_{beta}.npy'
        if Path(beta_file).exists():
            features_dict[f'BetaVAE_beta_{beta}'] = {
                'features': np.load(beta_file),
                'labels': beta_labels
            }
            print(f"✓ Beta-VAE (β={beta}): {features_dict[f'BetaVAE_beta_{beta}']['features'].shape}")
    
    # 5. PCA baseline
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
    
    # 6. Raw features
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
    
    return features_dict


def run_comprehensive_hard_clustering(features_dict, n_clusters=5):
    """
    Run ALL clustering methods on ALL feature types
    Complete evaluation for Hard task
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE HARD TASK EVALUATION")
    print("="*80)
    
    all_results = []
    
    for feature_name, data in features_dict.items():
        features = data['features']
        labels_true = data['labels']
        
        print(f"\n{'='*60}")
        print(f"Evaluating: {feature_name}")
        print(f"{'='*60}")
        
        # K-Means
        print(f"[1/3] K-Means...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels_kmeans = kmeans.fit_predict(features)
        results_kmeans = evaluate_clustering(features, labels_true, labels_kmeans, 
                                            f"{feature_name}+K-Means")
        all_results.append(results_kmeans)
        
        # Agglomerative
        print(f"[2/3] Agglomerative...")
        agg = AgglomerativeClustering(n_clusters=n_clusters)
        labels_agg = agg.fit_predict(features)
        results_agg = evaluate_clustering(features, labels_true, labels_agg,
                                        f"{feature_name}+Agglomerative")
        all_results.append(results_agg)
        
        # DBSCAN
        print(f"[3/3] DBSCAN...")
        from sklearn.neighbors import NearestNeighbors
        neighbors = NearestNeighbors(n_neighbors=5)
        neighbors.fit(features)
        distances, _ = neighbors.kneighbors(features)
        eps = np.percentile(distances[:, -1], 90)
        
        dbscan = DBSCAN(eps=eps, min_samples=5)
        labels_dbscan = dbscan.fit_predict(features)
        
        n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
        
        if n_clusters_dbscan > 1:
            mask = labels_dbscan != -1
            if mask.sum() > 0:
                results_dbscan = evaluate_clustering(
                    features[mask], labels_true[mask], labels_dbscan[mask],
                    f"{feature_name}+DBSCAN"
                )
                results_dbscan['n_clusters'] = n_clusters_dbscan
                all_results.append(results_dbscan)
        
        print(f"✓ {feature_name} complete")
    
    results_df = pd.DataFrame(all_results)
    return results_df


def analyze_hard_task_results(results_df):
    """Detailed analysis for Hard task report"""
    print("\n" + "="*80)
    print("HARD TASK RESULTS ANALYSIS")
    print("="*80)
    
    # Overall best
    print("\n🏆 TOP 10 METHODS (Overall Performance):")
    print("-"*80)
    top10 = results_df.nlargest(10, 'silhouette')
    print(top10[['method', 'silhouette', 'calinski_harabasz', 'davies_bouldin', 
                 'adjusted_rand_index', 'normalized_mutual_info', 'purity']].to_string(index=False))
    
    # Beta-VAE comparison
    print("\n🔬 BETA-VAE ANALYSIS:")
    print("-"*80)
    beta_results = results_df[results_df['method'].str.contains('BetaVAE')]
    
    if len(beta_results) > 0:
        # Group by beta value
        beta_kmeans = beta_results[beta_results['method'].str.contains('K-Means')]
        
        if len(beta_kmeans) > 0:
            print("\nBeta-VAE + K-Means Performance:")
            print(beta_kmeans[['method', 'silhouette', 'adjusted_rand_index', 
                              'normalized_mutual_info', 'purity']].to_string(index=False))
            
            # Find best beta
            best_beta_row = beta_kmeans.loc[beta_kmeans['silhouette'].idxmax()]
            print(f"\n✨ Best Beta Value: {best_beta_row['method']}")
            print(f"   Silhouette: {best_beta_row['silhouette']:.4f}")
            print(f"   ARI: {best_beta_row['adjusted_rand_index']:.4f}")
            print(f"   NMI: {best_beta_row['normalized_mutual_info']:.4f}")
            print(f"   Purity: {best_beta_row['purity']:.4f}")
    
    # VAE architecture comparison
    print("\n🏗️ VAE ARCHITECTURE COMPARISON (K-Means only):")
    print("-"*80)
    vae_methods = ['VAE_basic+K-Means', 'VAE_conv+K-Means', 'VAE_multimodal+K-Means', 
                   'PCA_baseline+K-Means']
    
    comparison_data = []
    for method in vae_methods:
        if method in results_df['method'].values:
            row = results_df[results_df['method'] == method].iloc[0]
            comparison_data.append(row)
    
    # Add best Beta-VAE
    beta_kmeans_results = results_df[results_df['method'].str.contains('BetaVAE') & 
                                     results_df['method'].str.contains('K-Means')]
    if len(beta_kmeans_results) > 0:
        best_beta = beta_kmeans_results.loc[beta_kmeans_results['silhouette'].idxmax()]
        comparison_data.append(best_beta)
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df[['method', 'silhouette', 'calinski_harabasz', 
                           'adjusted_rand_index', 'normalized_mutual_info']].to_string(index=False))
    
    # Clustering algorithm comparison
    print("\n🔄 CLUSTERING ALGORITHM COMPARISON:")
    print("-"*80)
    
    # Extract algorithm names
    results_df['algorithm'] = results_df['method'].str.split('+').str[-1]
    
    for algo in ['K-Means', 'Agglomerative', 'DBSCAN']:
        algo_results = results_df[results_df['algorithm'] == algo]
        if len(algo_results) > 0:
            best = algo_results.loc[algo_results['silhouette'].idxmax()]
            avg_sil = algo_results['silhouette'].mean()
            print(f"{algo:15s} - Best: {best['silhouette']:.4f} ({best['method']})")
            print(f"{'':15s}   Avg:  {avg_sil:.4f}")
    
    # Multi-modal benefit analysis
    print("\n🎭 MULTI-MODAL BENEFIT ANALYSIS:")
    print("-"*80)
    
    audio_only = results_df[results_df['method'] == 'VAE_basic+K-Means']
    multimodal = results_df[results_df['method'] == 'VAE_multimodal+K-Means']
    
    if len(audio_only) > 0 and len(multimodal) > 0:
        audio_sil = audio_only['silhouette'].values[0]
        multi_sil = multimodal['silhouette'].values[0]
        improvement = ((multi_sil - audio_sil) / audio_sil) * 100
        
        print(f"Audio-only VAE:     {audio_sil:.4f}")
        print(f"Multimodal VAE:     {multi_sil:.4f}")
        print(f"Improvement:        {improvement:+.2f}%")
        
        if improvement > 0:
            print("\n✓ Multi-modal features IMPROVE clustering performance")
        else:
            print("\n⚠ Multi-modal features do not improve clustering")
            print("  Possible reasons: text features may be noisy, need better fusion")
    
    # Disentanglement benefit
    print("\n🧬 DISENTANGLEMENT ANALYSIS (Beta-VAE):")
    print("-"*80)
    
    if len(beta_kmeans) > 0:
        # Compare beta = 1.0 (standard) vs beta > 1.0 (disentangled)
        standard_beta = beta_kmeans[beta_kmeans['method'].str.contains('beta_1.0')]
        high_beta = beta_kmeans[beta_kmeans['method'].str.contains('beta_4.0') | 
                               beta_kmeans['method'].str.contains('beta_10.0')]
        
        if len(standard_beta) > 0 and len(high_beta) > 0:
            std_sil = standard_beta['silhouette'].mean()
            high_sil = high_beta['silhouette'].mean()
            
            print(f"Standard VAE (β=1.0):    {std_sil:.4f}")
            print(f"Disentangled VAE (β>1):  {high_sil:.4f}")
            
            if high_sil > std_sil:
                improvement = ((high_sil - std_sil) / std_sil) * 100
                print(f"Improvement:             {improvement:+.2f}%")
                print("\n✓ Disentanglement HELPS clustering!")
            else:
                print("\n⚠ Higher beta does not improve clustering")
                print("  Trade-off: better disentanglement but worse reconstruction")


def create_hard_task_summary_table(results_df):
    """Create summary table for report"""
    print("\n" + "="*80)
    print("SUMMARY TABLE FOR REPORT")
    print("="*80)
    
    # Select key methods
    key_methods = [
        'PCA_baseline+K-Means',
        'VAE_basic+K-Means',
        'VAE_conv+K-Means',
        'VAE_multimodal+K-Means'
    ]
    
    # Add best Beta-VAE
    beta_methods = results_df[results_df['method'].str.contains('BetaVAE') & 
                             results_df['method'].str.contains('K-Means')]
    if len(beta_methods) > 0:
        best_beta_method = beta_methods.loc[beta_methods['silhouette'].idxmax()]['method']
        key_methods.append(best_beta_method)
    
    # Filter and format
    summary = results_df[results_df['method'].isin(key_methods)]
    
    if len(summary) > 0:
        summary = summary[['method', 'silhouette', 'calinski_harabasz', 'davies_bouldin',
                          'adjusted_rand_index', 'normalized_mutual_info', 'purity']]
        
        print("\n" + summary.to_string(index=False))
        
        # Save as LaTeX table
        latex_table = summary.to_latex(index=False, float_format="%.4f")
        
        with open('./results/hard_task_summary_table.tex', 'w') as f:
            f.write(latex_table)
        
        print("\n✓ LaTeX table saved to ./results/hard_task_summary_table.tex")


if __name__ == "__main__":
    print("="*80)
    print("HARD TASK COMPREHENSIVE EVALUATION")
    print("="*80)
    
    # Load all features
    features_dict = load_all_features_hard()
    
    if len(features_dict) == 0:
        print("\n⚠ No features found!")
        print("Make sure you've run beta_vae.py first")
        exit(1)
    
    print(f"\nTotal feature variants: {len(features_dict)}")
    
    # Run comprehensive clustering
    results_df = run_comprehensive_hard_clustering(features_dict, n_clusters=5)
    
    # Save results
    results_df.to_csv('./results/clustering_metrics_hard_task.csv', index=False)
    print(f"\n✓ Results saved to ./results/clustering_metrics_hard_task.csv")
    
    # Detailed analysis
    analyze_hard_task_results(results_df)
    
    # Create summary table for report
    create_hard_task_summary_table(results_df)
    
    print("\n" + "="*80)
    print("✓ HARD TASK EVALUATION COMPLETE")
    print("="*80)
    print("\nNext step: Run visualize_hard.py for comprehensive visualizations")
