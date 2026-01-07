"""
Comprehensive visualizations for Hard Task
Includes Beta-VAE analysis and disentanglement plots
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import pickle

sns.set_style("whitegrid")


def plot_beta_vae_latent_comparison(genre_names, save_path='./results/beta_vae_latent_comparison.png'):
    """
    Compare latent spaces of different Beta-VAE values
    """
    print("Creating Beta-VAE latent space comparison...")
    
    beta_values = [0.5, 1.0, 4.0, 10.0]
    labels = np.load('./data/beta_vae_labels.npy')
    
    # Check which betas are available
    available_betas = []
    for beta in beta_values:
        if Path(f'./data/beta_vae_latent_beta_{beta}.npy').exists():
            available_betas.append(beta)
    
    if len(available_betas) == 0:
        print("  No Beta-VAE features found, skipping")
        return
    
    n_betas = len(available_betas)
    fig, axes = plt.subplots(2, (n_betas + 1) // 2, figsize=(18, 10))
    axes = axes.flatten() if n_betas > 1 else [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(genre_names)))
    
    for idx, beta in enumerate(available_betas):
        ax = axes[idx]
        
        # Load features
        features = np.load(f'./data/beta_vae_latent_beta_{beta}.npy')
        
        # Sample for speed
        if len(features) > 500:
            indices = np.random.choice(len(features), 500, replace=False)
            features = features[indices]
            labels_sample = labels[indices]
        else:
            labels_sample = labels
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
        features_2d = tsne.fit_transform(features)
        
        # Plot
        for i, genre_idx in enumerate(np.unique(labels_sample)):
            mask = labels_sample == genre_idx
            ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                      c=[colors[i]], alpha=0.6, s=30, 
                      edgecolors='black', linewidth=0.3,
                      label=genre_names[genre_idx] if idx == 0 else "")
        
        ax.set_title(f'Beta-VAE (β={beta})', fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplots
    for idx in range(n_betas, len(axes)):
        fig.delaxes(axes[idx])
    
    # Add legend
    if n_betas > 0:
        handles, labels_legend = axes[0].get_legend_handles_labels()
        fig.legend(handles, genre_names, loc='upper center', 
                  bbox_to_anchor=(0.5, 0.98), ncol=5, fontsize=11)
    
    plt.suptitle('Beta-VAE Latent Space Comparison (t-SNE)', 
                fontsize=16, fontweight='bold', y=0.96)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {save_path}")
    plt.close()


def plot_disentanglement_analysis(save_path='./results/disentanglement_analysis.png'):
    """
    Visualize disentanglement metrics across beta values
    """
    print("Creating disentanglement analysis plot...")
    
    beta_values = [0.5, 1.0, 4.0, 10.0]
    
    # Collect metrics
    metrics = {
        'beta': [],
        'avg_correlation': [],
        'variance_mean': [],
        'variance_std': [],
        'active_dims': []
    }
    
    for beta in beta_values:
        feature_file = f'./data/beta_vae_latent_beta_{beta}.npy'
        if not Path(feature_file).exists():
            continue
        
        latent = np.load(feature_file)
        
        # Correlation
        corr = np.corrcoef(latent.T)
        avg_abs_corr = np.mean(np.abs(corr - np.eye(latent.shape[1])))
        
        # Variance
        variances = np.var(latent, axis=0)
        
        metrics['beta'].append(beta)
        metrics['avg_correlation'].append(avg_abs_corr)
        metrics['variance_mean'].append(np.mean(variances))
        metrics['variance_std'].append(np.std(variances))
        metrics['active_dims'].append(np.sum(variances > 0.1))
    
    if len(metrics['beta']) == 0:
        print("  No data available, skipping")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Average Correlation
    ax = axes[0, 0]
    ax.plot(metrics['beta'], metrics['avg_correlation'], 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Beta Value', fontsize=12)
    ax.set_ylabel('Avg Absolute Correlation', fontsize=12)
    ax.set_title('Latent Dimension Independence\n(Lower = More Disentangled)', 
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Variance Statistics
    ax = axes[0, 1]
    ax.plot(metrics['beta'], metrics['variance_mean'], 'o-', linewidth=2, 
           markersize=8, label='Mean Variance')
    ax.fill_between(metrics['beta'], 
                    np.array(metrics['variance_mean']) - np.array(metrics['variance_std']),
                    np.array(metrics['variance_mean']) + np.array(metrics['variance_std']),
                    alpha=0.3)
    ax.set_xlabel('Beta Value', fontsize=12)
    ax.set_ylabel('Variance', fontsize=12)
    ax.set_title('Latent Dimension Variance', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Active Dimensions
    ax = axes[1, 0]
    ax.bar(range(len(metrics['beta'])), metrics['active_dims'], 
          color=plt.cm.viridis(np.linspace(0, 1, len(metrics['beta']))))
    ax.set_xticks(range(len(metrics['beta'])))
    ax.set_xticklabels([f'β={b}' for b in metrics['beta']])
    ax.set_xlabel('Beta Value', fontsize=12)
    ax.set_ylabel('Number of Active Dimensions', fontsize=12)
    ax.set_title('Active Latent Dimensions\n(Variance > 0.1)', 
                fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 4: Trade-off visualization
    ax = axes[1, 1]
    ax.scatter(metrics['avg_correlation'], metrics['variance_mean'], 
              s=200, c=metrics['beta'], cmap='viridis', 
              edgecolors='black', linewidth=2)
    
    for i, beta in enumerate(metrics['beta']):
        ax.annotate(f'β={beta}', 
                   (metrics['avg_correlation'][i], metrics['variance_mean'][i]),
                   xytext=(10, 10), textcoords='offset points', fontsize=10)
    
    ax.set_xlabel('Avg Absolute Correlation\n(Independence)', fontsize=12)
    ax.set_ylabel('Mean Variance\n(Information)', fontsize=12)
    ax.set_title('Disentanglement Trade-off', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Beta Value', fontsize=11)
    
    plt.suptitle('Beta-VAE Disentanglement Analysis', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {save_path}")
    plt.close()


def plot_hard_task_performance_summary(results_df, save_path='./results/hard_task_performance_summary.png'):
    """
    Create comprehensive performance summary for Hard task
    """
    print("Creating Hard task performance summary...")
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Top methods overall
    ax1 = fig.add_subplot(gs[0, :])
    top_methods = results_df.nlargest(10, 'silhouette')
    methods = [m.split('+')[0][:20] for m in top_methods['method']]
    
    bars = ax1.barh(range(len(methods)), top_methods['silhouette'].values,
                    color=plt.cm.viridis(np.linspace(0, 1, len(methods))))
    ax1.set_yticks(range(len(methods)))
    ax1.set_yticklabels(methods, fontsize=10)
    ax1.set_xlabel('Silhouette Score', fontsize=12)
    ax1.set_title('Top 10 Methods by Silhouette Score', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars, top_methods['silhouette'].values)):
        ax1.text(val, i, f' {val:.4f}', va='center', fontsize=9)
    
    # Plot 2: Beta-VAE comparison
    ax2 = fig.add_subplot(gs[1, 0])
    beta_results = results_df[results_df['method'].str.contains('BetaVAE') & 
                             results_df['method'].str.contains('K-Means')]
    
    if len(beta_results) > 0:
        beta_vals = beta_results['method'].str.extract(r'beta_(\d+\.?\d*)')[0].astype(float).values
        sil_scores = beta_results['silhouette'].values
        
        ax2.plot(beta_vals, sil_scores, 'o-', linewidth=2, markersize=10, color='steelblue')
        ax2.set_xlabel('Beta Value', fontsize=11)
        ax2.set_ylabel('Silhouette Score', fontsize=11)
        ax2.set_title('Beta-VAE Performance', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Highlight best
        best_idx = np.argmax(sil_scores)
        ax2.scatter(beta_vals[best_idx], sil_scores[best_idx], 
                   s=200, color='red', marker='*', zorder=5)
        ax2.annotate(f'Best: β={beta_vals[best_idx]}', 
                    (beta_vals[best_idx], sil_scores[best_idx]),
                    xytext=(10, -20), textcoords='offset points',
                    fontsize=9, bbox=dict(boxstyle='round', facecolor='yellow'))
    
    # Plot 3: Architecture comparison
    ax3 = fig.add_subplot(gs[1, 1])
    arch_methods = ['VAE_basic+K-Means', 'VAE_conv+K-Means', 'VAE_multimodal+K-Means']
    arch_data = results_df[results_df['method'].isin(arch_methods)]
    
    if len(arch_data) > 0:
        arch_names = [m.split('+')[0].replace('VAE_', '') for m in arch_data['method']]
        bars = ax3.bar(range(len(arch_names)), arch_data['silhouette'].values,
                      color=['skyblue', 'lightcoral', 'lightgreen'])
        ax3.set_xticks(range(len(arch_names)))
        ax3.set_xticklabels(arch_names, rotation=0, fontsize=10)
        ax3.set_ylabel('Silhouette Score', fontsize=11)
        ax3.set_title('VAE Architecture Comparison', fontsize=12, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, arch_data['silhouette'].values):
            ax3.text(bar.get_x() + bar.get_width()/2, val,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Clustering algorithm comparison
    ax4 = fig.add_subplot(gs[1, 2])
    results_df['algorithm'] = results_df['method'].str.split('+').str[-1]
    
    algo_means = results_df.groupby('algorithm')['silhouette'].agg(['mean', 'std'])
    algos = algo_means.index.tolist()
    means = algo_means['mean'].values
    stds = algo_means['std'].values
    
    bars = ax4.bar(range(len(algos)), means, yerr=stds, capsize=5,
                  color=['lightblue', 'lightgreen', 'lightyellow'])
    ax4.set_xticks(range(len(algos)))
    ax4.set_xticklabels(algos, rotation=0, fontsize=10)
    ax4.set_ylabel('Avg Silhouette Score', fontsize=11)
    ax4.set_title('Clustering Algorithm Performance', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # Plot 5: All metrics for top method
    ax5 = fig.add_subplot(gs[2, :2])
    best_method = results_df.loc[results_df['silhouette'].idxmax()]
    
    metrics = ['silhouette', 'calinski_harabasz', 'davies_bouldin', 
              'adjusted_rand_index', 'normalized_mutual_info', 'purity']
    available_metrics = [m for m in metrics if m in best_method.index and pd.notna(best_method[m])]
    
    if len(available_metrics) > 0:
        values = [best_method[m] for m in available_metrics]
        
        # Normalize for visualization
        normalized_values = []
        for i, m in enumerate(available_metrics):
            if m == 'davies_bouldin':
                # Lower is better, invert
                normalized_values.append(1 / (1 + values[i]))
            elif m == 'calinski_harabasz':
                # Scale down for visualization
                normalized_values.append(values[i] / results_df[m].max())
            else:
                normalized_values.append(values[i])
        
        bars = ax5.barh(range(len(available_metrics)), normalized_values,
                       color=plt.cm.Set3(np.linspace(0, 1, len(available_metrics))))
        ax5.set_yticks(range(len(available_metrics)))
        ax5.set_yticklabels(available_metrics, fontsize=10)
        ax5.set_xlabel('Normalized Score', fontsize=11)
        ax5.set_title(f'All Metrics for Best Method: {best_method["method"]}', 
                     fontsize=12, fontweight='bold')
        ax5.grid(axis='x', alpha=0.3)
        
        # Add actual values as text
        for i, (bar, val, norm_val) in enumerate(zip(bars, values, normalized_values)):
            ax5.text(norm_val, i, f' {val:.4f}', va='center', fontsize=9)
    
    # Plot 6: Summary statistics
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    
    stats_text = f"""
    SUMMARY STATISTICS
    
    Total Methods Evaluated: {len(results_df)}
    
    Best Overall:
    {results_df.loc[results_df['silhouette'].idxmax()]['method'][:30]}
    Silhouette: {results_df['silhouette'].max():.4f}
    
    Best Beta-VAE:
    """
    
    if len(beta_results) > 0:
        best_beta = beta_results.loc[beta_results['silhouette'].idxmax()]
        stats_text += f"{best_beta['method'][:30]}\n"
        stats_text += f"Silhouette: {best_beta['silhouette']:.4f}\n"
    
    stats_text += f"""
    
    Metric Ranges:
    Sil: [{results_df['silhouette'].min():.3f}, {results_df['silhouette'].max():.3f}]
    ARI: [{results_df['adjusted_rand_index'].min():.3f}, {results_df['adjusted_rand_index'].max():.3f}]
    NMI: [{results_df['normalized_mutual_info'].min():.3f}, {results_df['normalized_mutual_info'].max():.3f}]
    """
    
    ax6.text(0.1, 0.5, stats_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Hard Task: Comprehensive Performance Analysis', 
                fontsize=18, fontweight='bold', y=0.98)
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {save_path}")
    plt.close()


def create_all_hard_visualizations():
    """Create all visualizations for Hard task"""
    print("\n" + "="*80)
    print("CREATING ALL HARD TASK VISUALIZATIONS")
    print("="*80)
    
    genre_names = ['Hip-Hop', 'Pop', 'Folk', 'Experimental', 'Rock']
    
    # Load results (if available)
    results_path = Path('./results/clustering_metrics_hard_task.csv')
    if not results_path.exists():
        print(f"  Results file not found: {results_path}. Skipping performance summary.")
        results_df = None
    else:
        results_df = pd.read_csv(results_path)

    # Create visualizations
    plot_beta_vae_latent_comparison(genre_names)
    plot_disentanglement_analysis()
    if results_df is not None:
        plot_hard_task_performance_summary(results_df)
    else:
        print("  Performance summary skipped due to missing results CSV.")
    
    print("\n✓ All Hard task visualizations complete!")


if __name__ == "__main__":
    print("="*80)
    print("HARD TASK VISUALIZATION GENERATION")
    print("="*80)
    
    create_all_hard_visualizations()
    
    print("\n" + "="*80)
    print("✓ HARD TASK VISUALIZATIONS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - ./results/beta_vae_latent_comparison.png")
    print("  - ./results/disentanglement_analysis.png")
    print("  - ./results/hard_task_performance_summary.png")