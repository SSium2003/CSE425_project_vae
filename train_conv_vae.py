"""
Train Convolutional VAE on MFCC spectrograms
Better for capturing temporal structure in audio
"""

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import librosa
from pathlib import Path
from tqdm import tqdm
import pickle

from vae import ConvVAE, vae_loss
from train import VAETrainer


def extract_mfcc_spectrograms(data_path='./data', max_samples=None, 
                              n_mfcc=20, max_frames=128):
    """
    Extract 2D MFCC spectrograms for ConvVAE
    
    Args:
        data_path: Path to FMA data
        max_samples: Max samples to process
        n_mfcc: Number of MFCC coefficients
        max_frames: Fixed number of time frames
        
    Returns:
        spectrograms: Array (N, 1, n_mfcc, max_frames)
        labels: Genre labels
        track_ids: Track IDs
    """
    print("="*80)
    print("EXTRACTING 2D MFCC SPECTROGRAMS")
    print("="*80)
    
    from dataset import FMADataset
    
    # Check if already processed
    spectrogram_file = Path(data_path) / 'mfcc_spectrograms.pkl'
    if spectrogram_file.exists():
        print(f"Loading pre-computed spectrograms from {spectrogram_file}...")
        with open(spectrogram_file, 'rb') as f:
            data = pickle.load(f)
        return data['spectrograms'], data['labels'], data['track_ids']
    
    # Load dataset
    dataset = FMADataset(
        data_path=data_path,
        genres=['Hip-Hop', 'Pop', 'Folk', 'Experimental', 'Rock'],
        max_samples=max_samples or 600
    )
    dataset.load_metadata()
    
    spectrograms = []
    labels = []
    track_ids = []
    
    print(f"\nExtracting 2D MFCCs (shape: {n_mfcc} x {max_frames})...")
    
    for track_id in tqdm(dataset.tracks.index):
        audio_path = dataset.get_audio_path(track_id)
        
        if not audio_path.exists():
            continue
        
        try:
            # Load audio
            y, sr = librosa.load(audio_path, duration=30, sr=22050)
            
            # Extract MFCC
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            
            # Pad or truncate to fixed length
            if mfcc.shape[1] < max_frames:
                # Pad with zeros
                pad_width = max_frames - mfcc.shape[1]
                mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
            else:
                # Truncate
                mfcc = mfcc[:, :max_frames]
            
            # Add channel dimension: (n_mfcc, max_frames) -> (1, n_mfcc, max_frames)
            mfcc = mfcc[np.newaxis, :, :]
            
            spectrograms.append(mfcc)
            
            # Get label
            genre = dataset.tracks.loc[track_id, ('track', 'genre_top')]
            labels.append(dataset.genre_to_idx[genre])
            track_ids.append(track_id)
            
        except Exception as e:
            continue
    
    spectrograms = np.array(spectrograms)
    labels = np.array(labels)
    track_ids = np.array(track_ids)
    
    print(f"\n✓ Extracted {len(spectrograms)} spectrograms")
    print(f"Spectrogram shape: {spectrograms.shape}")
    
    # Save for future use
    save_data = {
        'spectrograms': spectrograms,
        'labels': labels,
        'track_ids': track_ids,
        'n_mfcc': n_mfcc,
        'max_frames': max_frames
    }
    
    with open(spectrogram_file, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"Saved spectrograms to {spectrogram_file}")
    
    return spectrograms, labels, track_ids


class ConvVAETrainer(VAETrainer):
    """Extended trainer for ConvVAE"""
    
    def extract_latent_features(self, data_loader):
        """Extract latent features for ConvVAE"""
        self.model.eval()
        latent_features = []
        
        with torch.no_grad():
            for (data,) in data_loader:
                data = data.to(self.device)
                mu = self.model.get_latent(data)
                latent_features.append(mu.cpu().numpy())
        
        return np.vstack(latent_features)


def prepare_conv_dataloader(spectrograms, batch_size=16, shuffle=True):
    """Prepare DataLoader for ConvVAE"""
    X_tensor = torch.FloatTensor(spectrograms)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


if __name__ == "__main__":
    print("="*80)
    print("CONVOLUTIONAL VAE TRAINING")
    print("="*80)
    
    # Extract spectrograms
    spectrograms, labels, track_ids = extract_mfcc_spectrograms(
        data_path='./data',
        max_samples=600,  # Per genre
        n_mfcc=20,
        max_frames=128
    )
    
    print(f"\nDataset ready:")
    print(f"  Shape: {spectrograms.shape}")
    print(f"  Samples: {len(spectrograms)}")
    
    # Normalize spectrograms
    print("\nNormalizing spectrograms...")
    mean = spectrograms.mean()
    std = spectrograms.std()
    spectrograms_norm = (spectrograms - mean) / (std + 1e-8)
    
    # Create DataLoader
    train_loader = prepare_conv_dataloader(spectrograms_norm, batch_size=16, shuffle=True)
    print(f"DataLoader created: {len(train_loader)} batches")
    
    # Initialize ConvVAE
    model = ConvVAE(input_channels=1, latent_dim=32)
    
    print(f"\nConvVAE Architecture:")
    print(f"  Input: (1, 20, 128)")
    print(f"  Latent dim: 32")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train
    trainer = ConvVAETrainer(model)
    
    print("\nStarting training...")
    trainer.train(
        train_loader,
        epochs=50,
        lr=1e-3,
        beta=1.0
    )
    
    # Plot training history
    trainer.plot_training_history('./results/conv_vae_training_history.png')
    
    # Save model
    trainer.save_model('./models/conv_vae_model.pt')
    
    # Extract latent features
    print("\nExtracting latent features...")
    test_loader = prepare_conv_dataloader(spectrograms_norm, batch_size=16, shuffle=False)
    latent_features = trainer.extract_latent_features(test_loader)
    
    print(f"Latent features shape: {latent_features.shape}")
    
    # Save features
    np.save('./data/conv_latent_features.npy', latent_features)
    np.save('./data/conv_labels.npy', labels)
    np.save('./data/conv_track_ids.npy', track_ids)
    
    print("\n" + "="*80)
    print("✓ CONVOLUTIONAL VAE TRAINING COMPLETE")
    print("="*80)
    print(f"Model saved to: ./models/conv_vae_model.pt")
    print(f"Features saved to: ./data/conv_latent_features.npy")
    print("\nNext step: Run train_multimodal_vae.py for hybrid features")