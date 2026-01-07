import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from pathlib import Path
import umap

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 8)


def plot_tsne(features, labels, title="t-SNE Visualization", 
              save_path='./results/tsne_plot.png', label_names=None):
    """
    Create t-SNE visualization of features
    
    Args:
        features: Feature vectors (n_samples, n_features)
        labels: Labels for coloring (n_samples,)
        title: Plot title
        save_path: Path to save figure
        label_names: Optional list of label names
    """
    print(f"Computing t-SNE projection...")
    
    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        label_name = label_names[label] if label_names else f"Cluster {label}"
        
        plt.scatter(
            features_2d[mask, 0], 
            features_2d[mask, 1],
            c=[colors[i]], 
            label=label_name,
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )
    
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    
    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved t-SNE plot to {save_path}")
    plt.close()


def plot_umap(features, labels, title="UMAP Visualization",
              save_path='./results/umap_plot.png', label_names=None):
    """
    Create UMAP visualization of features
    
    Args:
        features: Feature vectors
        labels: Labels for coloring
        title: Plot title
        save_path: Path to save figure
        label_names: Optional list of label names
    """
    print(f"Computing UMAP projection...")
    
    # Compute UMAP
    reducer = umap.UMAP(n_components=2, random_state=42)
    features_2d = reducer.fit_transform(features)
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        label_name = label_names[label] if label_names else f"Cluster {label}"
        
        plt.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[colors[i]],
            label=label_name,
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )
    
    plt.xlabel('UMAP Component 1', fontsize=12)
    plt.ylabel('UMAP Component 2', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    
    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved UMAP plot to {save_path}")
    plt.close()


def plot_cluster_distribution(labels_true, labels_pred, genre_names, 
                               save_path='./results/cluster_distribution.png'):
    """
    Plot distribution of true genres within predicted clusters
    
    Args:
        labels_true: True genre labels
        labels_pred: Predicted cluster labels
        genre_names: List of genre names
        save_path: Path to save figure
    """
    print("Creating cluster distribution plot...")
    
    # Create contingency matrix
    n_genres = len(np.unique(labels_true))
    n_clusters = len(np.unique(labels_pred))
    
    contingency = np.zeros((n_clusters, n_genres))
    
    for cluster in range(n_clusters):
        cluster_mask = labels_pred == cluster
        for genre in range(n_genres):
            genre_mask = labels_true == genre
            contingency[cluster, genre] = np.sum(cluster_mask & genre_mask)
    
    # Normalize by cluster size
    cluster_sizes = contingency.sum(axis=1, keepdims=True)
    contingency_normalized = contingency / (cluster_sizes + 1e-10)
    
    # Create heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw counts
    sns.heatmap(contingency, annot=True, fmt='.0f', cmap='Blues',
                xticklabels=genre_names, 
                yticklabels=[f'Cluster {i}' for i in range(n_clusters)],
                ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_title('Cluster-Genre Distribution (Counts)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('True Genre', fontsize=12)
    ax1.set_ylabel('Predicted Cluster', fontsize=12)
    
    # Normalized
    sns.heatmap(contingency_normalized, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=genre_names,
                yticklabels=[f'Cluster {i}' for i in range(n_clusters)],
                ax=ax2, cbar_kws={'label': 'Proportion'})
    ax2.set_title('Cluster-Genre Distribution (Proportions)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('True Genre', fontsize=12)
    ax2.set_ylabel('Predicted Cluster', fontsize=12)
    
    plt.tight_layout()
    
    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved cluster distribution to {save_path}")
    plt.close()


def plot_comparison(results_df, save_path='./results/metrics_comparison.png'):
    """
    Plot comparison of different methods across metrics
    
    Args:
        results_df: DataFrame with clustering results
        save_path: Path to save figure
    """
    print("Creating metrics comparison plot...")
    
    # Select numeric columns only
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) == 0:
        print("No numeric metrics to plot")
        return
    
    n_metrics = len(numeric_cols)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    methods = results_df['method'].tolist()
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
    
    for i, metric in enumerate(numeric_cols):
        if i >= len(axes):
            break
            
        ax = axes[i]
        values = results_df[metric].tolist()
        
        bars = ax.bar(range(len(methods)), values, color=colors, edgecolor='black')
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for j, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Remove empty subplots
    for i in range(n_metrics, len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle('Clustering Methods Comparison', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved metrics comparison to {save_path}")
    plt.close()


def create_all_visualizations(vae_features, pca_features, labels_true, 
                               labels_vae, labels_pca, genre_names, results_df):
    """
    Create all visualizations for the project
    
    Args:
        vae_features: VAE latent features
        pca_features: PCA features
        labels_true: True genre labels
        labels_vae: VAE clustering labels
        labels_pca: PCA clustering labels
        genre_names: List of genre names
        results_df: DataFrame with clustering results
    """
    print("\n" + "="*80)
    print("CREATING ALL VISUALIZATIONS")
    print("="*80)
    
    # 1. t-SNE with true labels (VAE features)
    plot_tsne(vae_features, labels_true, 
              title="VAE Latent Space - Colored by True Genre",
              save_path='./results/tsne_vae_true_labels.png',
              label_names=genre_names)
    
    # 2. t-SNE with predicted clusters (VAE features)
    plot_tsne(vae_features, labels_vae,
              title="VAE Latent Space - Colored by Predicted Clusters",
              save_path='./results/tsne_vae_clusters.png')
    
    # 3. t-SNE with true labels (PCA features)
    plot_tsne(pca_features, labels_true,
              title="PCA Feature Space - Colored by True Genre",
              save_path='./results/tsne_pca_true_labels.png',
              label_names=genre_names)
    
    # 4. UMAP with true labels (VAE features)
    plot_umap(vae_features, labels_true,
              title="VAE Latent Space (UMAP) - Colored by True Genre",
              save_path='./results/umap_vae_true_labels.png',
              label_names=genre_names)
    
    # 5. Cluster distribution
    plot_cluster_distribution(labels_true, labels_vae, genre_names,
                              save_path='./results/cluster_distribution_vae.png')
    
    plot_cluster_distribution(labels_true, labels_pca, genre_names,
                              save_path='./results/cluster_distribution_pca.png')
    
    # 6. Metrics comparison
    plot_comparison(results_df, save_path='./results/metrics_comparison.png')
    
    print("\n✓ All visualizations created successfully!")
    print("="*80)


if __name__ == "__main__":
    # Example usage
    print("Visualization Pipeline")
    print("="*80)
    
    # Load data
    latent_features = np.load('./data/latent_features.npy')
    labels = np.load('./data/labels.npy')
    
    genre_names = ['Hip-Hop', 'Pop', 'Folk', 'Experimental', 'Rock']
    
    # Create basic t-SNE plot
    plot_tsne(latent_features, labels, 
              title="VAE Latent Space",
              save_path='./results/tsne_visualization.png',
              label_names=genre_names)
    
    print("\n✓ Visualization complete!")