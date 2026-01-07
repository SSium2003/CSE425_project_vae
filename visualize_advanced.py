"""
Advanced visualizations for Medium task
Creates comprehensive comparison plots
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import pickle

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def create_comparison_tsne(features_dict, genre_names, save_path='./results/tsne_comparison.png'):
    """
    Create side-by-side t-SNE plots for all feature types
    """
    print("Creating t-SNE comparison...")
    
    n_methods = len(features_dict)
    n_cols = 3
    n_rows = (n_methods + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    axes = axes.flatten() if n_methods > 1 else [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(genre_names)))
    
    for idx, (name, data) in enumerate(features_dict.items()):
        ax = axes[idx]
        
        features = data['features']
        labels = data['labels']
        
        # Compute t-SNE
        if features.shape[0] > 1000:
            # Sample for speed
            indices = np.random.choice(len(features), 1000, replace=False)
            features_sample = features[indices]
            labels_sample = labels[indices]
        else:
            features_sample = features
            labels_sample = labels
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features_sample)-1))
        features_2d = tsne.fit_transform(features_sample)
        
        # Plot
        for i, genre_idx in enumerate(np.unique(labels_sample)):
            mask = labels_sample == genre_idx
            ax.scatter(
                features_2d[mask, 0],
                features_2d[mask, 1],
                c=[colors[i]],
                label=genre_names[genre_idx] if idx == 0 else "",
                alpha=0.6,
                s=30,
                edgecolors='black',
                linewidth=0.3
            )
        
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplots
    for idx in range(n_methods, len(axes)):
        fig.delaxes(axes[idx])
    
    # Add legend
    if n_methods > 0:
        handles, labels_legend = axes[0].get_legend_handles_labels()
        fig.legend(handles, genre_names, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=5)
    
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {save_path}")
    plt.close()


def create_metrics_heatmap(results_df, save_path='./results/metrics_heatmap.png'):
    """
    Create heatmap of all metrics across methods
    """
    print("Creating metrics heatmap...")
    
    # Select numeric columns
    metric_cols = ['silhouette', 'calinski_harabasz', 'davies_bouldin', 
                   'adjusted_rand_index', 'normalized_mutual_info', 'purity']
    
    # Filter available columns
    available_cols = [col for col in metric_cols if col in results_df.columns]
    
    # Create pivot table
    pivot_data = results_df[['method'] + available_cols].set_index('method')
    
    # Normalize each metric to [0, 1] for better comparison
    pivot_norm = pivot_data.copy()
    for col in available_cols:
        if col == 'davies_bouldin':  # Lower is better
            pivot_norm[col] = 1 - (pivot_data[col] - pivot_data[col].min()) / (pivot_data[col].max() - pivot_data[col].min())
        else:  # Higher is better
            pivot_norm[col] = (pivot_data[col] - pivot_data[col].min()) / (pivot_data[col].max() - pivot_data[col].min())
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Raw values
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax1, cbar_kws={'label': 'Score'})
    ax1.set_title('Raw Metric Values', fontsize=14, fontweight='bold')
    ax1.set_xlabel('')
    ax1.set_ylabel('Method', fontsize=12)
    
    # Normalized values
    sns.heatmap(pivot_norm, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax2, vmin=0, vmax=1, cbar_kws={'label': 'Normalized Score'})
    ax2.set_title('Normalized Metrics (0-1 scale)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {save_path}")
    plt.close()


def create_architecture_comparison(results_df, save_path='./results/architecture_comparison.png'):
    """
    Compare VAE architectures specifically
    """
    print("Creating architecture comparison...")
    
    # Filter VAE methods with K-Means
    vae_results = results_df[results_df['method'].str.contains('VAE') & results_df['method'].str.contains('K-Means')]
    
    if len(vae_results) == 0:
        print("  No VAE results found, skipping")
        return
    
    # Metrics to compare
    metrics = ['silhouette', 'calinski_harabasz', 'davies_bouldin', 
               'adjusted_rand_index', 'normalized_mutual_info', 'purity']
    available_metrics = [m for m in metrics if m in vae_results.columns]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(available_metrics[:6]):
        ax = axes[idx]
        
        methods = vae_results['method'].str.split('+').str[0].values
        values = vae_results[metric].values
        
        # Create bar plot
        bars = ax.bar(range(len(methods)), values, color=plt.cm.Set3(np.linspace(0, 1, len(methods))))
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Remove empty subplots
    for idx in range(len(available_metrics), 6):
        fig.delaxes(axes[idx])
    
    plt.suptitle('VAE Architecture Comparison (K-Means)', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {save_path}")
    plt.close()


def create_clustering_algorithm_comparison(results_df, save_path='./results/clustering_algorithms_comparison.png'):
    """
    Compare clustering algorithms across feature types
    """
    print("Creating clustering algorithm comparison...")
    
    # Extract algorithm name
    results_df['algorithm'] = results_df['method'].str.split('+').str[1]
    results_df['feature'] = results_df['method'].str.split('+').str[0]
    
    # Group by algorithm
    algorithms = results_df['algorithm'].unique()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Create grouped bar chart
        data_pivot = results_df.pivot_table(
            values=metric,
            index='feature',
            columns='algorithm',
            aggfunc='first'
        )
        
        data_pivot.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.set_xlabel('Feature Type', fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.legend(title='Algorithm', loc='best')
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.suptitle('Clustering Algorithm Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {save_path}")
    plt.close()


def create_summary_figure(results_df, features_dict, genre_names, save_path='./results/summary_figure.png'):
    """
    Create a comprehensive summary figure
    """
    print("Creating summary figure...")
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Top: Best method comparison
    ax1 = fig.add_subplot(gs[0, :])
    
    top_methods = results_df.nlargest(8, 'silhouette')
    methods = top_methods['method'].values
    silhouette = top_methods['silhouette'].values
    
    bars = ax1.barh(range(len(methods)), silhouette, color=plt.cm.viridis(np.linspace(0, 1, len(methods))))
    ax1.set_yticks(range(len(methods)))
    ax1.set_yticklabels(methods, fontsize=10)
    ax1.set_xlabel('Silhouette Score', fontsize=12)
    ax1.set_title('Top 8 Methods by Silhouette Score', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, silhouette)):
        ax1.text(val, i, f' {val:.4f}', va='center', fontsize=9)
    
    # Middle row: t-SNE plots for top 3 methods
    top_3_methods = results_df.nlargest(3, 'silhouette')['method'].str.split('+').str[0].unique()[:3]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(genre_names)))
    
    for idx, method_name in enumerate(top_3_methods):
        if method_name in features_dict:
            ax = fig.add_subplot(gs[1, idx])
            
            features = features_dict[method_name]['features']
            labels = features_dict[method_name]['labels']
            
            # Sample if too many points
            if len(features) > 500:
                indices = np.random.choice(len(features), 500, replace=False)
                features = features[indices]
                labels = labels[indices]
            
            # t-SNE
            tsne = TSNE(n_components=2, random_state=42)
            features_2d = tsne.fit_transform(features)
            
            for i, genre_idx in enumerate(np.unique(labels)):
                mask = labels == genre_idx
                ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                          c=[colors[i]], alpha=0.6, s=20, edgecolors='black', linewidth=0.3)
            
            ax.set_title(method_name, fontsize=12, fontweight='bold')
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            ax.grid(True, alpha=0.3)
    
    # Bottom row: Metrics comparison
    ax4 = fig.add_subplot(gs[2, :2])
    
    metric_cols = ['silhouette', 'adjusted_rand_index', 'normalized_mutual_info']
    available_cols = [col for col in metric_cols if col in results_df.columns]
    
    top_5 = results_df.nlargest(5, 'silhouette')
    
    x = np.arange(len(top_5))
    width = 0.25
    
    for i, metric in enumerate(available_cols):
        offset = (i - len(available_cols)/2) * width
        ax4.bar(x + offset, top_5[metric].values, width, label=metric)
    
    ax4.set_xlabel('Method', fontsize=11)
    ax4.set_ylabel('Score', fontsize=11)
    ax4.set_title('Top 5 Methods - Multiple Metrics', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([m.split('+')[0][:15] for m in top_5['method']], rotation=45, ha='right', fontsize=9)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # Statistics box
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.axis('off')
    
    stats_text = f"""
    SUMMARY STATISTICS
    
    Total Methods: {len(results_df)}
    
    Best Method:
    {results_df.iloc[results_df['silhouette'].idxmax()]['method']}
    
    Silhouette: {results_df['silhouette'].max():.4f}
    
    Metrics Range:
    Sil: [{results_df['silhouette'].min():.3f}, {results_df['silhouette'].max():.3f}]
    CH: [{results_df['calinski_harabasz'].min():.1f}, {results_df['calinski_harabasz'].max():.1f}]
    """
    
    ax5.text(0.1, 0.5, stats_text, transform=ax5.transAxes, fontsize=10,
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Comprehensive Clustering Analysis Summary', fontsize=18, fontweight='bold', y=0.98)
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    print("="*80)
    print("ADVANCED VISUALIZATIONS")
    print("="*80)
    
    # Load results
    results_df = pd.read_csv('./results/clustering_metrics_all.csv')
    print(f"Loaded {len(results_df)} results")
    
    # Load features
    from clustering_advanced import load_all_features
    features_dict = load_all_features()
    
    # Genre names
    genre_names = ['Hip-Hop', 'Pop', 'Folk', 'Experimental', 'Rock']
    
    # Create all visualizations
    print("\nCreating visualizations...")
    
    create_comparison_tsne(features_dict, genre_names)
    create_metrics_heatmap(results_df)
    create_architecture_comparison(results_df)
    create_clustering_algorithm_comparison(results_df)
    create_summary_figure(results_df, features_dict, genre_names)
    
    print("\n" + "="*80)
    print("✓ ALL VISUALIZATIONS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - ./results/tsne_comparison.png")
    print("  - ./results/metrics_heatmap.png")
    print("  - ./results/architecture_comparison.png")
    print("  - ./results/clustering_algorithms_comparison.png")
    print("  - ./results/summary_figure.png")
    print("\nYou can now use these in your report!")