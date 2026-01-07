"""
Combine audio MFCC features with text embeddings
Creates multimodal feature representation
"""

import numpy as np
import pickle
from pathlib import Path
import pandas as pd

def load_audio_features(path='./data/processed_features.pkl'):
    """Load audio MFCC features"""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def load_text_features(path='./data/text_features.pkl'):
    """Load text embeddings"""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def match_and_combine_features(audio_data, text_data, strategy='concatenate'):
    """
    Match audio and text features by track_id and combine them
    
    Args:
        audio_data: Dict with audio features and track_ids
        text_data: Dict with text features and track_ids
        strategy: How to combine features
                 'concatenate': Simple concatenation [audio; text]
                 'weighted': Weighted combination
                 'separate': Keep separate for multi-encoder VAE
    
    Returns:
        combined_features: Combined feature array
        labels: Genre labels
        track_ids: Matched track IDs
        metadata: Info about the combination
    """
    print("="*80)
    print("MATCHING AND COMBINING FEATURES")
    print("="*80)
    
    # Extract data
    audio_features = audio_data['features']
    audio_track_ids = audio_data['track_ids']
    labels = audio_data['labels']
    
    text_features = text_data['features']
    text_track_ids = text_data['track_ids']
    
    print(f"\nAudio features: {audio_features.shape}")
    print(f"Text features: {text_features.shape}")
    
    # Create mapping for fast lookup
    text_id_to_idx = {tid: i for i, tid in enumerate(text_track_ids)}
    
    # Match features
    matched_audio = []
    matched_text = []
    matched_labels = []
    matched_track_ids = []
    
    print("\nMatching features by track_id...")
    for i, track_id in enumerate(audio_track_ids):
        if track_id in text_id_to_idx:
            text_idx = text_id_to_idx[track_id]
            
            matched_audio.append(audio_features[i])
            matched_text.append(text_features[text_idx])
            matched_labels.append(labels[i])
            matched_track_ids.append(track_id)
    
    matched_audio = np.array(matched_audio)
    matched_text = np.array(matched_text)
    matched_labels = np.array(matched_labels)
    matched_track_ids = np.array(matched_track_ids)
    
    print(f"✓ Matched {len(matched_track_ids)} tracks")
    print(f"  Match rate: {len(matched_track_ids)/len(audio_track_ids)*100:.1f}%")
    
    # Combine features based on strategy
    if strategy == 'concatenate':
        print("\nCombining features via concatenation...")
        combined_features = np.concatenate([matched_audio, matched_text], axis=1)
        print(f"✓ Combined features: {combined_features.shape}")
        
        return_data = {
            'features': combined_features,
            'labels': matched_labels,
            'track_ids': matched_track_ids,
            'genre_names': audio_data['genre_names'],
            'audio_dim': matched_audio.shape[1],
            'text_dim': matched_text.shape[1],
            'strategy': 'concatenate'
        }
    
    elif strategy == 'separate':
        print("\nKeeping features separate for multi-encoder VAE...")
        
        return_data = {
            'audio_features': matched_audio,
            'text_features': matched_text,
            'labels': matched_labels,
            'track_ids': matched_track_ids,
            'genre_names': audio_data['genre_names'],
            'audio_dim': matched_audio.shape[1],
            'text_dim': matched_text.shape[1],
            'strategy': 'separate'
        }
    
    elif strategy == 'weighted':
        print("\nCombining features with learned weights...")
        # Normalize first
        from sklearn.preprocessing import StandardScaler
        scaler_audio = StandardScaler()
        scaler_text = StandardScaler()
        
        audio_norm = scaler_audio.fit_transform(matched_audio)
        text_norm = scaler_text.fit_transform(matched_text)
        
        # Weighted combination (adjust weights as needed)
        alpha = 0.5  # Weight for audio
        beta = 0.5   # Weight for text
        
        # To combine different dimensions, use dimensionality matching
        from sklearn.decomposition import PCA
        target_dim = min(matched_audio.shape[1], matched_text.shape[1])
        
        pca_audio = PCA(n_components=target_dim)
        pca_text = PCA(n_components=target_dim)
        
        audio_reduced = pca_audio.fit_transform(audio_norm)
        text_reduced = pca_text.fit_transform(text_norm)
        
        combined_features = alpha * audio_reduced + beta * text_reduced
        
        print(f"✓ Weighted combination: {combined_features.shape}")
        
        return_data = {
            'features': combined_features,
            'labels': matched_labels,
            'track_ids': matched_track_ids,
            'genre_names': audio_data['genre_names'],
            'audio_dim': target_dim,
            'text_dim': target_dim,
            'strategy': 'weighted',
            'alpha': alpha,
            'beta': beta
        }
    
    return return_data


def create_all_feature_variants():
    """
    Create multiple feature variants for comparison
    
    Returns dict with:
    - audio_only
    - text_only
    - concatenated
    - separate (for multi-encoder)
    """
    print("="*80)
    print("CREATING ALL FEATURE VARIANTS")
    print("="*80)
    
    # Load features
    audio_data = load_audio_features()
    text_data = load_text_features()
    
    variants = {}
    
    # 1. Audio only (already have this)
    print("\n[1/4] Audio-only features...")
    variants['audio_only'] = audio_data
    print(f"✓ Audio shape: {audio_data['features'].shape}")
    
    # 2. Text only
    print("\n[2/4] Text-only features...")
    # Need to match text features with audio labels
    text_id_to_idx = {tid: i for i, tid in enumerate(text_data['track_ids'])}
    
    matched_text_features = []
    matched_labels = []
    matched_ids = []
    
    for i, track_id in enumerate(audio_data['track_ids']):
        if track_id in text_id_to_idx:
            text_idx = text_id_to_idx[track_id]
            matched_text_features.append(text_data['features'][text_idx])
            matched_labels.append(audio_data['labels'][i])
            matched_ids.append(track_id)
    
    variants['text_only'] = {
        'features': np.array(matched_text_features),
        'labels': np.array(matched_labels),
        'track_ids': np.array(matched_ids),
        'genre_names': audio_data['genre_names']
    }
    print(f"✓ Text shape: {variants['text_only']['features'].shape}")
    
    # 3. Concatenated
    print("\n[3/4] Concatenated features...")
    variants['concatenated'] = match_and_combine_features(
        audio_data, text_data, strategy='concatenate'
    )
    
    # 4. Separate (for multi-encoder VAE)
    print("\n[4/4] Separate features (multi-encoder)...")
    variants['separate'] = match_and_combine_features(
        audio_data, text_data, strategy='separate'
    )
    
    return variants


if __name__ == "__main__":
    print("="*80)
    print("HYBRID FEATURE CREATION")
    print("="*80)
    
    # Create all variants
    variants = create_all_feature_variants()
    
    # Save all variants
    print("\n" + "="*80)
    print("SAVING FEATURE VARIANTS")
    print("="*80)
    
    for name, data in variants.items():
        save_path = f'./data/features_{name}.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ Saved {name} to {save_path}")
    
    # Summary
    print("\n" + "="*80)
    print("FEATURE CREATION COMPLETE")
    print("="*80)
    
    print("\nFeature Variants Created:")
    for name, data in variants.items():
        if 'features' in data:
            print(f"  {name}: {data['features'].shape}")
        else:
            print(f"  {name}: audio={data['audio_features'].shape}, text={data['text_features'].shape}")
    
    print("\nNext steps:")
    print("1. Train basic VAE on audio_only (already done)")
    print("2. Train ConvVAE: run train_conv_vae.py")
    print("3. Train multimodal VAE: run train_multimodal_vae.py")
    print("4. Compare all methods: run clustering_advanced.py")