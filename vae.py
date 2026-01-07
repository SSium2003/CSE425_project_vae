import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """Variational Autoencoder for music feature extraction"""
    
    def __init__(self, input_dim, hidden_dims=[512, 256], latent_dim=32):
        """
        Args:
            input_dim: Input feature dimension (e.g., 40 for 20 MFCC mean+std)
            hidden_dims: List of hidden layer dimensions
            latent_dim: Latent space dimension
        """
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Build encoder
        encoder_layers = []
        in_dim = input_dim
        
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.BatchNorm1d(h_dim))
            encoder_layers.append(nn.Dropout(0.2))
            in_dim = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space layers
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Build decoder
        decoder_layers = []
        reversed_dims = [latent_dim] + hidden_dims[::-1]
        
        for i in range(len(reversed_dims) - 1):
            decoder_layers.append(nn.Linear(reversed_dims[i], reversed_dims[i+1]))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.BatchNorm1d(reversed_dims[i+1]))
            decoder_layers.append(nn.Dropout(0.2))
        
        self.decoder = nn.Sequential(*decoder_layers)
        self.final_layer = nn.Linear(hidden_dims[0], input_dim)
        
    def encode(self, x):
        """
        Encode input to latent distribution parameters
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            mu: Mean of latent distribution (batch_size, latent_dim)
            logvar: Log variance of latent distribution (batch_size, latent_dim)
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + std * epsilon
        
        Args:
            mu: Mean (batch_size, latent_dim)
            logvar: Log variance (batch_size, latent_dim)
            
        Returns:
            z: Sampled latent vector (batch_size, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """
        Decode latent vector to reconstruction
        
        Args:
            z: Latent vector (batch_size, latent_dim)
            
        Returns:
            reconstruction: Reconstructed input (batch_size, input_dim)
        """
        h = self.decoder(z)
        reconstruction = self.final_layer(h)
        return reconstruction
    
    def forward(self, x):
        """
        Forward pass through VAE
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            reconstruction: Reconstructed input
            mu: Latent distribution mean
            logvar: Latent distribution log variance
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    
    def get_latent(self, x):
        """
        Get latent representation (using mean, no sampling)
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            z: Latent representation (batch_size, latent_dim)
        """
        mu, _ = self.encode(x)
        return mu


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE loss function: Reconstruction + KL divergence
    
    Args:
        recon_x: Reconstructed input
        x: Original input
        mu: Latent distribution mean
        logvar: Latent distribution log variance
        beta: Weight for KL divergence term (default=1.0)
        
    Returns:
        loss: Total loss
        recon_loss: Reconstruction loss component
        kld_loss: KL divergence component
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence loss
    # KLD = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    loss = recon_loss + beta * kld_loss
    
    return loss, recon_loss, kld_loss


class ConvVAE(nn.Module):
    """
    Convolutional VAE for spectrogram/MFCC features (for Medium task)
    This is a bonus - you can use this for Medium task
    """
    
    def __init__(self, input_channels=1, latent_dim=32):
        super(ConvVAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        
        # Calculate flattened size (depends on input size)
        # For MFCC: (20, 128) -> after 3 conv layers -> (128, 2, 16)
        self.flatten_dim = 128 * 2 * 16
        
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
        )
        
    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, 128, 2, 16)
        reconstruction = self.decoder(h)
        return reconstruction
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    
    def get_latent(self, x):
        mu, _ = self.encode(x)
        return mu


if __name__ == "__main__":
    # Test VAE
    print("Testing VAE architecture...")
    
    input_dim = 40  # 20 MFCC mean + 20 MFCC std
    batch_size = 32
    
    model = VAE(input_dim=input_dim, hidden_dims=[512, 256], latent_dim=32)
    
    # Dummy input
    x = torch.randn(batch_size, input_dim)
    
    # Forward pass
    recon, mu, logvar = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Latent mu shape: {mu.shape}")
    print(f"Reconstruction shape: {recon.shape}")
    
    # Test loss
    loss, recon_loss, kld = vae_loss(recon, x, mu, logvar)
    print(f"\nLoss: {loss.item():.2f}")
    print(f"Reconstruction: {recon_loss.item():.2f}")
    print(f"KL Divergence: {kld.item():.2f}")
    
    # Test latent extraction
    latent = model.get_latent(x)
    print(f"\nLatent representation shape: {latent.shape}")
    
    print("\n✓ VAE model working correctly!")