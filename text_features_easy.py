"""
Convert lyrics and metadata to text feature embeddings
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import pickle

class TextFeatureExtractor:
    """Extract text features from lyrics and metadata"""
    
    def __init__(self, method='sentence-transformer'):
        """
        Args:
            method: 'sentence-transformer' (best) or 'tfidf' (faster)
        """
        self.method = method
        
        if method == 'sentence-transformer':
            try:
                from sentence_transformers import SentenceTransformer
                print("Loading sentence-transformer model...")
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.feature_dim = 384
                print(f"✓ Model loaded (embedding dim: {self.feature_dim})")
            except ImportError:
                print("⚠ sentence-transformers not installed")
                print("Install: pip install sentence-transformers")
                print("Falling back to TF-IDF")
                self.method = 'tfidf'
        
        if method == 'tfidf' or self.method == 'tfidf':
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.model = TfidfVectorizer(
                max_features=384,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2
            )
            self.feature_dim = 384
            self.fitted = False
    
    def extract_from_lyrics(self, lyrics_dict):
        """
        Extract features from lyrics
        
        Args:
            lyrics_dict: Dict of track_id -> lyrics data
            
        Returns:
            features: Array (n_tracks, feature_dim)
            track_ids: List of track IDs
        """
        texts = []
        track_ids = []
        
        print("Processing lyrics...")
        for track_id, data in tqdm(lyrics_dict.items()):
            if data.get('success') and 'lyrics' in data:
                lyrics = data['lyrics']
                # Clean lyrics (remove annotations, newlines, etc.)
                lyrics = lyrics.replace('\\n', ' ').replace('[', '').replace(']', '')
                texts.append(lyrics)
                track_ids.append(int(track_id))
        
        print(f"Processing {len(texts)} lyrics texts...")
        
        if self.method == 'sentence-transformer':
            features = self.model.encode(
                texts,
                show_progress_bar=True,
                batch_size=32
            )
        else:  # tfidf
            if not self.fitted:
                features = self.model.fit_transform(texts).toarray()
                self.fitted = True
            else:
                features = self.model.transform(texts).toarray()
        
        print(f"✓ Extracted features: {features.shape}")
        
        return features, track_ids
    
    def extract_from_metadata(self, tracks_df):
        """
        Extract features from metadata (fallback for tracks without lyrics)
        
        Args:
            tracks_df: FMA tracks DataFrame
            
        Returns:
            features: Array (n_tracks, feature_dim)
            track_ids: List of track IDs
        """
        texts = []
        track_ids = []
        
        print("Processing metadata...")
        for idx, row in tqdm(tracks_df.iterrows(), total=len(tracks_df)):
            # Combine title, artist, tags, genres
            title = str(row.get(('track', 'title'), ''))
            artist = str(row.get(('artist', 'name'), ''))
            tags = str(row.get(('track', 'tags'), ''))
            genre = str(row.get(('track', 'genre_top'), ''))
            
            # Create text
            text = f"{title} by {artist}. Genre: {genre}. Tags: {tags}"
            text = text.replace('nan', '').strip()
            
            if len(text) > 10:  # Skip if too short
                texts.append(text)
                track_ids.append(idx)
        
        print(f"Processing {len(texts)} metadata texts...")
        
        if self.method == 'sentence-transformer':
            features = self.model.encode(
                texts,
                show_progress_bar=True,
                batch_size=32
            )
        else:  # tfidf
            if not self.fitted:
                features = self.model.fit_transform(texts).toarray()
                self.fitted = True
            else:
                features = self.model.transform(texts).toarray()
        
        print(f"✓ Extracted features: {features.shape}")
        
        return features, track_ids


def create_hybrid_text_features(lyrics_path='./data/lyrics.json',
                                tracks_path='F:/CSE425/fma_metadata/tracks.csv',
                                method='sentence-transformer'):
    """
    Create text features using both lyrics and metadata
    
    - Use lyrics when available
    - Use metadata as fallback
    
    Returns:
        features: Text features array
        track_ids: Corresponding track IDs
        metadata: Dict with info about sources
    """
    print("="*80)
    print("CREATING HYBRID TEXT FEATURES")
    print("="*80)
    
    # Initialize extractor
    extractor = TextFeatureExtractor(method=method)
    
    # Load lyrics
    lyrics_dict = {}
    if Path(lyrics_path).exists():
        print(f"\nLoading lyrics from {lyrics_path}...")
        with open(lyrics_path, 'r', encoding='utf-8') as f:
            lyrics_dict = json.load(f)
        print(f"✓ Loaded {len(lyrics_dict)} lyrics")
    else:
        print(f"⚠ No lyrics file found at {lyrics_path}")
    
    # Load metadata
    print(f"\nLoading metadata from {tracks_path}...")
    tracks = pd.read_csv(tracks_path, index_col=0, header=[0, 1])
    small = tracks['set', 'subset'] == 'small'
    tracks_small = tracks[small]
    
    # Filter for specific genres
    genres = ['Hip-Hop', 'Pop', 'Folk', 'Experimental', 'Rock']
    genre_mask = tracks_small['track', 'genre_top'].isin(genres)
    tracks_filtered = tracks_small[genre_mask]
    print(f"✓ Loaded metadata for {len(tracks_filtered)} tracks")
    
    # Extract features from lyrics
    lyrics_features = None
    lyrics_track_ids = []
    
    if lyrics_dict:
        print("\n[1/2] Extracting features from lyrics...")
        lyrics_features, lyrics_track_ids = extractor.extract_from_lyrics(lyrics_dict)
    
    # Extract features from metadata (for all tracks)
    print("\n[2/2] Extracting features from metadata...")
    metadata_features, metadata_track_ids = extractor.extract_from_metadata(tracks_filtered)
    
    # Combine: use lyrics if available, else use metadata
    print("\nCombining lyrics and metadata features...")
    
    all_features = []
    all_track_ids = []
    source_info = {}
    
    # Convert to sets for fast lookup
    lyrics_id_set = set(lyrics_track_ids)
    
    # For each track in metadata
    for i, track_id in enumerate(metadata_track_ids):
        if track_id in lyrics_id_set:
            # Use lyrics feature
            lyrics_idx = lyrics_track_ids.index(track_id)
            all_features.append(lyrics_features[lyrics_idx])
            source_info[track_id] = 'lyrics'
        else:
            # Use metadata feature
            all_features.append(metadata_features[i])
            source_info[track_id] = 'metadata'
        
        all_track_ids.append(track_id)
    
    all_features = np.array(all_features)
    
    # Statistics
    lyrics_count = sum(1 for s in source_info.values() if s == 'lyrics')
    metadata_count = sum(1 for s in source_info.values() if s == 'metadata')
    
    print(f"\n✓ Created hybrid text features: {all_features.shape}")
    print(f"  From lyrics: {lyrics_count} tracks")
    print(f"  From metadata: {metadata_count} tracks")
    print(f"  Total: {len(all_track_ids)} tracks")
    
    metadata = {
        'feature_dim': all_features.shape[1],
        'method': method,
        'source_info': source_info,
        'lyrics_count': lyrics_count,
        'metadata_count': metadata_count
    }
    
    return all_features, all_track_ids, metadata


if __name__ == "__main__":
    print("="*80)
    print("TEXT FEATURE EXTRACTION")
    print("="*80)
    
    # Choose method
    print("\nChoose feature extraction method:")
    print("1. Sentence Transformers (best quality, needs GPU)")
    print("2. TF-IDF (faster, works on CPU)")
    
    choice = input("Enter choice [1]: ").strip() or "1"
    method = 'sentence-transformer' if choice == '1' else 'tfidf'
    
    # Create features
    features, track_ids, metadata = create_hybrid_text_features(method=method)
    
    # Save
    print("\nSaving text features...")
    save_data = {
        'features': features,
        'track_ids': np.array(track_ids),
        'metadata': metadata
    }
    
    with open('./data/text_features.pkl', 'wb') as f:
        pickle.dump(save_data, f)
    
    print("✓ Saved to ./data/text_features.pkl")
    
    print("\n" + "="*80)
    print("TEXT FEATURE EXTRACTION COMPLETE")
    print("="*80)
    print(f"Feature shape: {features.shape}")
    print(f"Tracks with lyrics: {metadata['lyrics_count']}")
    print(f"Tracks with metadata: {metadata['metadata_count']}")
    print("\nNext step: Run hybrid_features.py to combine with audio features")