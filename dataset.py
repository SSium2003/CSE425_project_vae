import os
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm
import pickle

class FMADataset:
    """Dataset class for FMA-small audio processing"""
    
    def __init__(self, data_path='F:/CSE425', genres=None, n_mfcc=20, 
                 n_fft=2048, hop_length=512, max_samples=600):
        """
        Args:
            data_path: Path to FMA data directory
            genres: List of genres to use (default: 5 main genres)
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Hop length for MFCC
            max_samples: Max samples per genre
        """
        self.data_path = Path(data_path)
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_samples = max_samples
        
        # Default genres for clustering
        if genres is None:
            self.genres = ['Hip-Hop', 'Pop', 'Folk', 'Experimental', 'Rock']
        else:
            self.genres = genres
        
        self.genre_to_idx = {g: i for i, g in enumerate(self.genres)}
        
    def load_metadata(self):
        """Load and filter FMA metadata"""
        print("Loading metadata...")
        
        # Load tracks metadata
        tracks_file = self.data_path / 'fma_metadata' / 'tracks.csv'
        tracks = pd.read_csv(tracks_file, index_col=0, header=[0, 1])
        
        # Filter for small subset
        small = tracks['set', 'subset'] == 'small'
        tracks_small = tracks[small]
        
        # Filter by selected genres
        genre_mask = tracks_small['track', 'genre_top'].isin(self.genres)
        self.tracks = tracks_small[genre_mask].copy()
        
        print(f"Total tracks: {len(self.tracks)}")
        print(f"Genre distribution:")
        for genre in self.genres:
            count = (self.tracks['track', 'genre_top'] == genre).sum()
            print(f"  {genre}: {count}")
        
        return self.tracks
    
    def get_audio_path(self, track_id):
        """Get path to audio file given track ID"""
        tid_str = f'{track_id:06d}'
        return self.data_path / 'fma_small' / tid_str[:3] / f'{tid_str}.mp3'
    
    def extract_mfcc(self, audio_path, duration=30):
        """
        Extract MFCC features from audio file
        
        Args:
            audio_path: Path to audio file
            duration: Duration to load (seconds)
            
        Returns:
            mfcc: MFCC features (n_mfcc, n_frames)
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, duration=duration, sr=22050)
            
            # Extract MFCC
            mfcc = librosa.feature.mfcc(
                y=y, 
                sr=sr, 
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            return mfcc
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None
    
    def process_dataset(self, save_path='F:/CSE425/data/processed_features.pkl'):
        """
        Process all audio files and extract features
        
        Returns:
            features: Array of MFCC features (n_samples, feature_dim)
            labels: Genre labels (n_samples,)
            track_ids: Track IDs (n_samples,)
        """
        if not hasattr(self, 'tracks'):
            self.load_metadata()
        
        features_list = []
        labels_list = []
        track_ids_list = []
        
        # Sample tracks per genre
        sampled_tracks = []
        for genre in self.genres:
            genre_tracks = self.tracks[self.tracks['track', 'genre_top'] == genre]
            n_samples = min(self.max_samples, len(genre_tracks))
            sampled = genre_tracks.sample(n=n_samples, random_state=42)
            sampled_tracks.append(sampled)
        
        sampled_df = pd.concat(sampled_tracks)
        
        print(f"\nProcessing {len(sampled_df)} audio files...")
        
        for track_id in tqdm(sampled_df.index):
            audio_path = self.get_audio_path(track_id)
            
            if not audio_path.exists():
                continue
            
            # Extract MFCC
            mfcc = self.extract_mfcc(audio_path)
            
            if mfcc is not None:
                # Flatten MFCC to 1D vector (take mean over time)
                # Alternative: use all frames, or fixed number of frames
                mfcc_mean = np.mean(mfcc, axis=1)  # Shape: (n_mfcc,)
                
                # Can also use statistics: mean + std
                mfcc_std = np.std(mfcc, axis=1)
                feature_vector = np.concatenate([mfcc_mean, mfcc_std])
                
                features_list.append(feature_vector)
                
                # Get genre label
                genre = sampled_df.loc[track_id, ('track', 'genre_top')]
                labels_list.append(self.genre_to_idx[genre])
                track_ids_list.append(track_id)
        
        # Convert to arrays
        features = np.array(features_list)
        labels = np.array(labels_list)
        track_ids = np.array(track_ids_list)
        
        print(f"\nProcessed {len(features)} tracks successfully")
        print(f"Feature shape: {features.shape}")
        
        # Save processed features
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        data_dict = {
            'features': features,
            'labels': labels,
            'track_ids': track_ids,
            'genre_names': self.genres,
            'genre_to_idx': self.genre_to_idx
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(data_dict, f)
        
        print(f"Saved processed features to {save_path}")
        
        return features, labels, track_ids
    
    @staticmethod
    def load_processed(path='F:/CSE425/data/processed_features.pkl'):
        """Load pre-processed features"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data


def normalize_features(features):
    """Normalize features to zero mean and unit variance"""
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    std[std == 0] = 1  # Avoid division by zero
    
    normalized = (features - mean) / std
    return normalized, mean, std


if __name__ == "__main__":
    # Example usage
    print("="*60)
    print("FMA Dataset Processor")
    print("="*60)
    
    # Initialize dataset
    dataset = FMADataset(
        data_path='F:/CSE425',
        genres=['Hip-Hop', 'Pop', 'Folk', 'Experimental', 'Rock'],
        max_samples=600  # 600 per genre = 3000 total
    )
    
    # Process and save
    features, labels, track_ids = dataset.process_dataset()
    
    # Normalize
    normalized_features, mean, std = normalize_features(features)
    
    print("\n" + "="*60)
    print("Dataset processing complete!")
    print(f"Total samples: {len(features)}")
    print(f"Feature dimension: {features.shape[1]}")
    print("="*60)
