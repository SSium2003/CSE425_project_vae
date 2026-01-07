"""
Quick test script to verify all components work
Run this BEFORE starting the full pipeline!
"""

import numpy as np
import torch
import sys

print("="*80)
print("SYSTEM TEST - Verifying All Components")
print("="*80)

# Test 1: Imports
print("\n[1/6] Testing imports...")
try:
    from dataset import FMADataset, normalize_features
    from vae import VAE, vae_loss
    from train import VAETrainer, prepare_dataloader
    from clustering import ClusteringPipeline, compare_methods
    from visualize import plot_tsne
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test 2: PyTorch and GPU
print("\n[2/6] Testing PyTorch...")
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
print("✓ PyTorch working")

# Test 3: VAE Model
print("\n[3/6] Testing VAE model...")
try:
    input_dim = 40
    model = VAE(input_dim=input_dim, hidden_dims=[128, 64], latent_dim=16)
    
    # Test forward pass
    x = torch.randn(8, input_dim)
    recon, mu, logvar = model(x)
    
    # Test loss
    loss, recon_loss, kld = vae_loss(recon, x, mu, logvar)
    
    # Test latent extraction
    latent = model.get_latent(x)
    
    assert recon.shape == x.shape, "Reconstruction shape mismatch"
    assert mu.shape == (8, 16), "Mu shape mismatch"
    assert latent.shape == (8, 16), "Latent shape mismatch"
    
    print(f"✓ VAE model working (input:{input_dim} -> latent:{16})")
except Exception as e:
    print(f"✗ VAE error: {e}")
    sys.exit(1)

# Test 4: Training
print("\n[4/6] Testing training pipeline...")
try:
    # Create dummy data
    dummy_features = np.random.randn(100, 40)
    normalized, mean, std = normalize_features(dummy_features)
    
    # Create dataloader
    loader = prepare_dataloader(normalized, batch_size=16, shuffle=True)
    
    # Initialize trainer
    model = VAE(input_dim=40, hidden_dims=[128, 64], latent_dim=16)
    trainer = VAETrainer(model)
    
    # Quick training (1 epoch)
    trainer.train(loader, epochs=1, lr=1e-3)
    
    # Extract features
    test_loader = prepare_dataloader(normalized, batch_size=16, shuffle=False)
    latent = trainer.extract_latent_features(test_loader)
    
    assert latent.shape == (100, 16), "Latent extraction failed"
    
    print("✓ Training pipeline working")
except Exception as e:
    print(f"✗ Training error: {e}")
    sys.exit(1)

# Test 5: Clustering
print("\n[5/6] Testing clustering...")
try:
    # Use latent features from previous test
    dummy_labels = np.random.randint(0, 5, 100)
    
    pipeline = ClusteringPipeline(n_clusters=5)
    labels_pred, results = pipeline.run_kmeans(latent, dummy_labels)
    
    assert len(labels_pred) == 100, "Clustering output size mismatch"
    assert 'silhouette' in results, "Missing evaluation metrics"
    
    print(f"✓ Clustering working (Silhouette: {results['silhouette']:.3f})")
except Exception as e:
    print(f"✗ Clustering error: {e}")
    sys.exit(1)

# Test 6: Visualization
print("\n[6/6] Testing visualization...")
try:
    # Test t-SNE plot (don't save, just test)
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(latent)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=dummy_labels, cmap='tab10')
    plt.close()
    
    print("✓ Visualization working")
except Exception as e:
    print(f"✗ Visualization error: {e}")
    sys.exit(1)

# Final check: Data directory
print("\n[7/7] Checking data directory...")
from pathlib import Path

data_path = Path('./data')
if not data_path.exists():
    print("⚠ Warning: ./data directory not found")
    print("  Creating directory...")
    data_path.mkdir(parents=True)

# Check for FMA dataset
fma_small = data_path / 'fma_small'
fma_metadata = data_path / 'fma_metadata'

if not fma_small.exists() or not fma_metadata.exists():
    print("⚠ Warning: FMA dataset not found in ./data/")
    print("  Expected:")
    print(f"    - {fma_small}")
    print(f"    - {fma_metadata}")
    print("\n  Please ensure you've downloaded the FMA-small dataset!")
else:
    print("✓ FMA dataset found")
    
    # Count audio files
    audio_files = list(fma_small.glob('**/*.mp3'))
    print(f"  Found {len(audio_files)} audio files")

# Summary
print("\n" + "="*80)
print("SYSTEM TEST COMPLETE")
print("="*80)

print("\n✅ All core components are working!")
print("\n📋 Next steps:")
print("  1. Ensure FMA-small dataset is downloaded")
print("  2. Run: python dataset.py (to process audio features)")
print("  3. Run: python main.ipynb (complete pipeline)")
print("  4. Or use individual scripts: train.py, clustering.py, visualize.py")

print("\n💡 Tips:")
print("  - Start with small subset first (max_samples=100) to test")
print("  - Monitor GPU memory if using CUDA")
print("  - Training 50 epochs takes ~30-60 minutes")

print("\n" + "="*80)
print("Good luck with your project! 🚀")
print("="*80)