"""
Train Multimodal VAE for hybrid audio + text clustering
Uses separate encoders for audio and text, then fuses in latent space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import pickle

from vae import vae_loss
from train import VAETrainer


class MultimodalVAE(nn.Module):
    """
    Multimodal VAE with separate encoders for audio and text
    """
    
    def __init__(self, audio_dim, text_dim, hidden_dims=[256, 128], latent_dim=32, fusion='concat'):
        """
        Args:
            audio_dim: Dimension of audio features
            text_dim: Dimension of text features
            hidden_dims: Hidden layer dimensions
            latent_dim: Latent space dimension
            fusion: Fusion strategy ('concat', 'sum', 'product')
        """
        super(MultimodalVAE, self).__init__()
        
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.latent_dim = latent_dim
        self.fusion = fusion
        
        # Audio encoder
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.Dropout(0.2)
        )
        
        # Text encoder
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.Dropout(0.2)
        )
        
        # Fusion layer
        if fusion == 'concat':
            fusion_dim = hidden_dims[1] * 2
        else:  # sum or product
            fusion_dim = hidden_dims[1]
        
        # Latent space
        self.fc_mu = nn.Linear(fusion_dim, latent_dim)
        self.fc_logvar = nn.Linear(fusion_dim, latent_dim)
        
        # Decoders
        self.decoder_common = nn.Sequential(
            nn.Linear(latent_dim, fusion_dim),
            nn.ReLU(),
            nn.BatchNorm1d(fusion_dim),
            nn.Dropout(0.2)
        )
        
        # Audio decoder
        self.audio_decoder = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[0], audio_dim)
        )
        
        # Text decoder
        self.text_decoder = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[0], text_dim)
        )
    
    def encode(self, audio, text):
        """Encode audio and text to latent distribution"""
        h_audio = self.audio_encoder(audio)
        h_text = self.text_encoder(text)
        
        # Fusion
        if self.fusion == 'concat':
            h = torch.cat([h_audio, h_text], dim=1)
        elif self.fusion == 'sum':
            h = h_audio + h_text
        elif self.fusion == 'product':
            h = h_audio * h_text
        else:
            raise ValueError(f"Unknown fusion: {self.fusion}")
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent vector to audio and text"""
        h = self.decoder_common(z)
        audio_recon = self.audio_decoder(h)
        text_recon = self.text_decoder(h)
        return audio_recon, text_recon
    
    def forward(self, audio, text):
        """Forward pass"""
        mu, logvar = self.encode(audio, text)
        z = self.reparameterize(mu, logvar)
        audio_recon, text_recon = self.decode(z)
        return audio_recon, text_recon, mu, logvar
    
    def get_latent(self, audio, text):
        """Get latent representation"""
        mu, _ = self.encode(audio, text)
        return mu


def multimodal_vae_loss(audio_recon, text_recon, audio, text, mu, logvar, 
                        alpha=0.5, beta=1.0):
    """
    Multimodal VAE loss
    
    Args:
        audio_recon, text_recon: Reconstructions
        audio, text: Inputs
        mu, logvar: Latent distribution parameters
        alpha: Weight for audio reconstruction (1-alpha for text)
        beta: Weight for KL divergence
    """
    # Reconstruction losses
    audio_recon_loss = F.mse_loss(audio_recon, audio, reduction='sum')
    text_recon_loss = F.mse_loss(text_recon, text, reduction='sum')
    
    # Weighted combination
    recon_loss = alpha * audio_recon_loss + (1 - alpha) * text_recon_loss
    
    # KL divergence
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    loss = recon_loss + beta * kld_loss
    
    return loss, recon_loss, kld_loss


class MultimodalVAETrainer:
    """Trainer for multimodal VAE"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.history = {
            'train_loss': [],
            'recon_loss': [],
            'kld_loss': []
        }
    
    def train_epoch(self, train_loader, optimizer, alpha=0.5, beta=1.0):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        total_recon = 0
        total_kld = 0
        
        for audio, text in train_loader:
            audio = audio.to(self.device)
            text = text.to(self.device)
            
            # Forward
            audio_recon, text_recon, mu, logvar = self.model(audio, text)
            
            # Loss
            loss, recon_loss, kld_loss = multimodal_vae_loss(
                audio_recon, text_recon, audio, text, mu, logvar, alpha, beta
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kld += kld_loss.item()
        
        n_samples = len(train_loader.dataset)
        return total_loss / n_samples, total_recon / n_samples, total_kld / n_samples
    
    def train(self, train_loader, epochs=50, lr=1e-3, alpha=0.5, beta=1.0):
        """Train the model"""
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        print(f"Training on {self.device}")
        print(f"Epochs: {epochs}, LR: {lr}, Alpha: {alpha}, Beta: {beta}")
        print("="*60)
        
        for epoch in range(epochs):
            avg_loss, avg_recon, avg_kld = self.train_epoch(
                train_loader, optimizer, alpha, beta
            )
            
            self.history['train_loss'].append(avg_loss)
            self.history['recon_loss'].append(avg_recon)
            self.history['kld_loss'].append(avg_kld)
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"Loss: {avg_loss:.4f} "
                      f"(Recon: {avg_recon:.4f}, KLD: {avg_kld:.4f})")
        
        print("="*60)
        print("Training complete!")
    
    def extract_latent_features(self, data_loader):
        """Extract latent features"""
        self.model.eval()
        latent_features = []
        
        with torch.no_grad():
            for audio, text in data_loader:
                audio = audio.to(self.device)
                text = text.to(self.device)
                mu = self.model.get_latent(audio, text)
                latent_features.append(mu.cpu().numpy())
        
        return np.vstack(latent_features)
    
    def save_model(self, path='./models/multimodal_vae_model.pt'):
        """Save model"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }, path)
        print(f"Model saved to {path}")


if __name__ == "__main__":
    print("="*80)
    print("MULTIMODAL VAE TRAINING")
    print("="*80)
    
    # Load hybrid features (separate audio and text)
    print("\nLoading hybrid features...")
    with open('./data/features_separate.pkl', 'rb') as f:
        data = pickle.load(f)
    
    audio_features = data['audio_features']
    text_features = data['text_features']
    labels = data['labels']
    
    print(f"Audio features: {audio_features.shape}")
    print(f"Text features: {text_features.shape}")
    print(f"Labels: {labels.shape}")
    
    # Normalize
    from dataset import normalize_features
    audio_norm, _, _ = normalize_features(audio_features)
    text_norm, _, _ = normalize_features(text_features)
    
    # Create DataLoader
    audio_tensor = torch.FloatTensor(audio_norm)
    text_tensor = torch.FloatTensor(text_norm)
    dataset = TensorDataset(audio_tensor, text_tensor)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print(f"DataLoader created: {len(train_loader)} batches")
    
    # Initialize model
    model = MultimodalVAE(
        audio_dim=audio_features.shape[1],
        text_dim=text_features.shape[1],
        hidden_dims=[256, 128],
        latent_dim=32,
        fusion='concat'
    )
    
    print(f"\nMultimodal VAE Architecture:")
    print(f"  Audio dim: {audio_features.shape[1]}")
    print(f"  Text dim: {text_features.shape[1]}")
    print(f"  Latent dim: 32")
    print(f"  Fusion: concat")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train
    trainer = MultimodalVAETrainer(model)
    trainer.train(
        train_loader,
        epochs=50,
        lr=1e-3,
        alpha=0.5,  # Equal weight for audio and text
        beta=1.0
    )
    
    # Save model
    trainer.save_model('./models/multimodal_vae_model.pt')
    
    # Extract features
    print("\nExtracting latent features...")
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    latent_features = trainer.extract_latent_features(test_loader)
    
    print(f"Latent features shape: {latent_features.shape}")
    
    # Save
    np.save('./data/multimodal_latent_features.npy', latent_features)
    np.save('./data/multimodal_labels.npy', labels)
    
    print("\n" + "="*80)
    print("✓ MULTIMODAL VAE TRAINING COMPLETE")
    print("="*80)
    print(f"Model: ./models/multimodal_vae_model.pt")
    print(f"Features: ./data/multimodal_latent_features.npy")
    print("\nNext: Run clustering_advanced.py to compare all methods")