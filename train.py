import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

from vae import VAE, vae_loss


class VAETrainer:
    """Trainer class for VAE model"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.history = {
            'train_loss': [],
            'recon_loss': [],
            'kld_loss': []
        }
        
    def train_epoch(self, train_loader, optimizer, beta=1.0):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_recon = 0
        total_kld = 0
        
        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(self.device)
            
            # Forward pass
            recon_batch, mu, logvar = self.model(data)
            
            # Compute loss
            loss, recon_loss, kld_loss = vae_loss(recon_batch, data, mu, logvar, beta)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kld += kld_loss.item()
        
        # Average losses
        n_samples = len(train_loader.dataset)
        avg_loss = total_loss / n_samples
        avg_recon = total_recon / n_samples
        avg_kld = total_kld / n_samples
        
        return avg_loss, avg_recon, avg_kld
    
    def train(self, train_loader, epochs=50, lr=1e-3, beta=1.0):
        """
        Train VAE
        
        Args:
            train_loader: DataLoader for training data
            epochs: Number of training epochs
            lr: Learning rate
            beta: Weight for KL divergence
        """
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        print(f"Training on {self.device}")
        print(f"Epochs: {epochs}, Learning rate: {lr}, Beta: {beta}")
        print("="*60)
        
        for epoch in range(epochs):
            avg_loss, avg_recon, avg_kld = self.train_epoch(train_loader, optimizer, beta)
            
            # Save history
            self.history['train_loss'].append(avg_loss)
            self.history['recon_loss'].append(avg_recon)
            self.history['kld_loss'].append(avg_kld)
            
            # Print progress
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"Loss: {avg_loss:.4f} "
                      f"(Recon: {avg_recon:.4f}, KLD: {avg_kld:.4f})")
        
        print("="*60)
        print("Training complete!")
        
    def extract_latent_features(self, data_loader):
        """
        Extract latent features for all data
        
        Args:
            data_loader: DataLoader
            
        Returns:
            latent_features: Numpy array of latent representations
        """
        self.model.eval()
        latent_features = []
        
        with torch.no_grad():
            for (data,) in data_loader:
                data = data.to(self.device)
                mu = self.model.get_latent(data)
                latent_features.append(mu.cpu().numpy())
        
        return np.vstack(latent_features)
    
    def plot_training_history(self, save_path='./results/training_history.png'):
        """Plot training loss curves"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Total loss
        axes[0].plot(self.history['train_loss'])
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Total Loss')
        axes[0].grid(True)
        
        # Reconstruction loss
        axes[1].plot(self.history['recon_loss'], color='orange')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Reconstruction Loss')
        axes[1].grid(True)
        
        # KLD loss
        axes[2].plot(self.history['kld_loss'], color='green')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('KL Divergence')
        axes[2].grid(True)
        
        plt.tight_layout()
        
        # Save figure
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history to {save_path}")
        plt.close()
        
    def save_model(self, path='./models/vae_model.pt'):
        """Save model checkpoint"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }, path)
        print(f"Model saved to {path}")
        
    def load_model(self, path='./models/vae_model.pt'):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint['history']
        print(f"Model loaded from {path}")


def prepare_dataloader(features, batch_size=32, shuffle=True):
    """
    Prepare PyTorch DataLoader from numpy features
    
    Args:
        features: Numpy array of features
        batch_size: Batch size
        shuffle: Whether to shuffle data
        
    Returns:
        DataLoader
    """
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(features)
    
    # Create dataset
    dataset = TensorDataset(X_tensor)
    
    # Create dataloader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return loader


if __name__ == "__main__":
    # Example training pipeline
    print("VAE Training Pipeline")
    print("="*60)
    
    # 1. Load processed features
    from dataset import FMADataset, normalize_features
    
    print("Loading processed features...")
    data = FMADataset.load_processed('./data/processed_features.pkl')
    features = data['features']
    labels = data['labels']
    
    # 2. Normalize features
    normalized_features, mean, std = normalize_features(features)
    
    print(f"Feature shape: {normalized_features.shape}")
    print(f"Number of samples: {len(normalized_features)}")
    
    # 3. Create DataLoader
    train_loader = prepare_dataloader(normalized_features, batch_size=32, shuffle=True)
    
    # 4. Initialize model
    input_dim = normalized_features.shape[1]
    model = VAE(input_dim=input_dim, hidden_dims=[512, 256], latent_dim=32)
    
    print(f"\nModel architecture:")
    print(f"Input dim: {input_dim}")
    print(f"Latent dim: 32")
    
    # 5. Train
    trainer = VAETrainer(model)
    trainer.train(train_loader, epochs=50, lr=1e-3, beta=1.0)
    
    # 6. Plot training history
    trainer.plot_training_history('./results/training_history.png')
    
    # 7. Save model
    trainer.save_model('./models/vae_model.pt')
    
    # 8. Extract latent features
    print("\nExtracting latent features...")
    test_loader = prepare_dataloader(normalized_features, batch_size=32, shuffle=False)
    latent_features = trainer.extract_latent_features(test_loader)
    
    print(f"Latent features shape: {latent_features.shape}")
    
    # Save latent features
    np.save('./data/latent_features.npy', latent_features)
    np.save('./data/labels.npy', labels)
    
    print("\n✓ Training complete!")
    print("Latent features saved to ./data/latent_features.npy")