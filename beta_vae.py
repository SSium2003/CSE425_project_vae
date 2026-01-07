"""
Beta-VAE for disentangled latent representations
Beta controls the weight of KL divergence for better disentanglement
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt

from vae import VAE, vae_loss
from train import VAETrainer, prepare_dataloader


class BetaVAE(VAE):
    """
    Beta-VAE: Same architecture as VAE, but trains with different beta values
    
    Beta > 1: Encourages disentanglement (independent latent factors)
    Beta = 1: Standard VAE
    Beta < 1: Focuses more on reconstruction
    """
    
    def __init__(self, input_dim, hidden_dims=[512, 256], latent_dim=32, beta=1.0):
        super(BetaVAE, self).__init__(input_dim, hidden_dims, latent_dim)
        self.beta = beta
        print(f"Beta-VAE initialized with beta={beta}")


class BetaVAETrainer(VAETrainer):
    """Trainer for Beta-VAE with configurable beta"""
    
    def __init__(self, model, beta=1.0, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(model, device)
        self.beta = beta
        
    def train_epoch(self, train_loader, optimizer, beta=None):
        """Train one epoch with specified beta"""
        if beta is None:
            beta = self.beta
            
        self.model.train()
        total_loss = 0
        total_recon = 0
        total_kld = 0
        
        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(self.device)
            
            # Forward pass
            recon_batch, mu, logvar = self.model(data)
            
            # Compute loss with beta
            loss, recon_loss, kld_loss = vae_loss(recon_batch, data, mu, logvar, beta)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kld += kld_loss.item()
        
        n_samples = len(train_loader.dataset)
        return total_loss / n_samples, total_recon / n_samples, total_kld / n_samples
    
    def train(self, train_loader, epochs=50, lr=1e-3, beta=None):
        """Train Beta-VAE"""
        if beta is None:
            beta = self.beta
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        print(f"Training Beta-VAE on {self.device}")
        print(f"Beta: {beta}, Epochs: {epochs}, LR: {lr}")
        print("="*60)
        
        for epoch in range(epochs):
            avg_loss, avg_recon, avg_kld = self.train_epoch(train_loader, optimizer, beta)
            
            self.history['train_loss'].append(avg_loss)
            self.history['recon_loss'].append(avg_recon)
            self.history['kld_loss'].append(avg_kld)
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"Loss: {avg_loss:.4f} "
                      f"(Recon: {avg_recon:.4f}, KLD: {avg_kld:.4f})")
        
        print("="*60)
        print(f"Training complete! Beta={beta}")


def train_multiple_beta_vaes(features, labels, beta_values=[0.5, 1.0, 4.0, 10.0], 
                             epochs=30, latent_dim=32):
    """
    Train Beta-VAEs with different beta values
    
    Args:
        features: Input features
        labels: Genre labels
        beta_values: List of beta values to try
        epochs: Number of epochs per beta
        latent_dim: Latent dimension
        
    Returns:
        Dictionary of results for each beta
    """
    print("="*80)
    print("TRAINING MULTIPLE BETA-VAES")
    print("="*80)
    
    from dataset import normalize_features
    
    # Normalize features
    normalized_features, mean, std = normalize_features(features)
    
    # Create dataloader
    train_loader = prepare_dataloader(normalized_features, batch_size=32, shuffle=True)
    test_loader = prepare_dataloader(normalized_features, batch_size=32, shuffle=False)
    
    input_dim = features.shape[1]
    results = {}
    
    for beta in beta_values:
        print(f"\n{'='*80}")
        print(f"TRAINING BETA-VAE WITH BETA={beta}")
        print(f"{'='*80}")
        
        # Initialize model
        model = BetaVAE(
            input_dim=input_dim,
            hidden_dims=[512, 256],
            latent_dim=latent_dim,
            beta=beta
        )
        
        # Train
        trainer = BetaVAETrainer(model, beta=beta)
        trainer.train(train_loader, epochs=epochs, lr=1e-3, beta=beta)
        
        # Save model
        model_path = f'./models/beta_vae_beta_{beta}.pt'
        trainer.save_model(model_path)
        
        # Plot training history
        plot_path = f'./results/beta_vae_beta_{beta}_training.png'
        trainer.plot_training_history(plot_path)
        
        # Extract latent features
        latent_features = trainer.extract_latent_features(test_loader)
        
        # Save features
        np.save(f'./data/beta_vae_latent_beta_{beta}.npy', latent_features)
        
        # Store results
        results[f'beta_{beta}'] = {
            'beta': beta,
            'model': model,
            'trainer': trainer,
            'latent_features': latent_features,
            'final_loss': trainer.history['train_loss'][-1],
            'final_recon': trainer.history['recon_loss'][-1],
            'final_kld': trainer.history['kld_loss'][-1]
        }
        
        print(f"\n✓ Beta={beta} complete!")
        print(f"  Final Loss: {results[f'beta_{beta}']['final_loss']:.4f}")
        print(f"  Final Recon: {results[f'beta_{beta}']['final_recon']:.4f}")
        print(f"  Final KLD: {results[f'beta_{beta}']['final_kld']:.4f}")
    
    # Save all labels
    np.save('./data/beta_vae_labels.npy', labels)
    
    return results


def compare_beta_values(results):
    """
    Compare training dynamics across different beta values
    """
    print("\n" + "="*80)
    print("BETA-VAE COMPARISON")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    
    # Plot 1: Total Loss
    ax = axes[0, 0]
    for i, (name, data) in enumerate(results.items()):
        beta = data['beta']
        history = data['trainer'].history
        ax.plot(history['train_loss'], label=f'β={beta}', color=colors[i])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss vs Beta', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Reconstruction Loss
    ax = axes[0, 1]
    for i, (name, data) in enumerate(results.items()):
        beta = data['beta']
        history = data['trainer'].history
        ax.plot(history['recon_loss'], label=f'β={beta}', color=colors[i])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reconstruction Loss')
    ax.set_title('Reconstruction Loss vs Beta', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: KL Divergence
    ax = axes[1, 0]
    for i, (name, data) in enumerate(results.items()):
        beta = data['beta']
        history = data['trainer'].history
        ax.plot(history['kld_loss'], label=f'β={beta}', color=colors[i])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('KL Divergence')
    ax.set_title('KL Divergence vs Beta', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Final Metrics Comparison
    ax = axes[1, 1]
    beta_vals = [data['beta'] for data in results.values()]
    final_losses = [data['final_loss'] for data in results.values()]
    final_recons = [data['final_recon'] for data in results.values()]
    final_klds = [data['final_kld'] for data in results.values()]
    
    x = np.arange(len(beta_vals))
    width = 0.25
    
    ax.bar(x - width, final_recons, width, label='Reconstruction', color='orange')
    ax.bar(x, final_klds, width, label='KL Divergence', color='green')
    ax.bar(x + width, final_losses, width, label='Total Loss', color='blue')
    
    ax.set_xlabel('Beta Value')
    ax.set_ylabel('Loss')
    ax.set_title('Final Loss Components', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'β={b}' for b in beta_vals])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./results/beta_vae_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved comparison plot to ./results/beta_vae_comparison.png")
    plt.close()


def analyze_disentanglement(results, features, labels):
    """
    Analyze disentanglement quality
    
    Higher beta should lead to:
    1. More independent latent dimensions
    2. Better interpretability
    3. Potentially better clustering
    """
    print("\n" + "="*80)
    print("DISENTANGLEMENT ANALYSIS")
    print("="*80)
    
    for name, data in results.items():
        beta = data['beta']
        latent = data['latent_features']
        
        # Compute correlation between latent dimensions
        correlation = np.corrcoef(latent.T)
        avg_abs_corr = np.mean(np.abs(correlation - np.eye(latent.shape[1])))
        
        # Compute variance in each dimension
        variances = np.var(latent, axis=0)
        
        print(f"\nBeta = {beta}:")
        print(f"  Avg absolute correlation: {avg_abs_corr:.4f} (lower = more disentangled)")
        print(f"  Variance range: [{variances.min():.4f}, {variances.max():.4f}]")
        print(f"  Active dimensions (var > 0.1): {np.sum(variances > 0.1)}/{len(variances)}")


if __name__ == "__main__":
    print("="*80)
    print("BETA-VAE TRAINING FOR HARD TASK")
    print("="*80)
    
    # Load processed features
    with open('./data/processed_features.pkl', 'rb') as f:
        data = pickle.load(f)
    
    features = data['features']
    labels = data['labels']
    
    print(f"\nDataset: {features.shape}")
    print(f"Training Beta-VAEs with different beta values...")
    
    # Train with different beta values
    beta_values = [0.5, 1.0, 4.0, 10.0]
    epochs = 30  # Quick training for comparison
    
    print(f"\nBeta values to test: {beta_values}")
    print(f"Epochs per beta: {epochs}")
    print(f"Total training time: ~{len(beta_values) * epochs} epochs = ~{len(beta_values) * 30} minutes")
    
    # Train all
    results = train_multiple_beta_vaes(
        features, 
        labels, 
        beta_values=beta_values,
        epochs=epochs,
        latent_dim=32
    )
    
    # Compare results
    compare_beta_values(results)
    
    # Analyze disentanglement
    analyze_disentanglement(results, features, labels)
    
    print("\n" + "="*80)
    print("✓ BETA-VAE TRAINING COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    for beta in beta_values:
        print(f"  - ./models/beta_vae_beta_{beta}.pt")
        print(f"  - ./data/beta_vae_latent_beta_{beta}.npy")
    print(f"  - ./results/beta_vae_comparison.png")
    
    print("\nNext step: Run clustering_hard.py to evaluate all Beta-VAEs")