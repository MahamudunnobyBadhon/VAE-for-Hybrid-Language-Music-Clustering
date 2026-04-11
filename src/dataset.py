"""
Dataset module for the VAE Music Clustering project.

Handles:
  - Audio loading and feature extraction (MFCC, Mel-spectrograms)
  - Synthetic dataset generation for testing
  - PyTorch Dataset wrappers
  - Metadata management
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

from src.config import (
    SAMPLE_RATE, AUDIO_DURATION, N_MFCC, N_MELS, HOP_LENGTH, N_FFT,
    AUDIO_ENGLISH_DIR, AUDIO_BANGLA_DIR, MFCC_DIR, MEL_SPEC_DIR,
    METADATA_PATH, FEATURES_DIR, LYRICS_EMB_DIR, BATCH_SIZE, RANDOM_STATE, DATA_DIR,
)


# ============================================================
# Audio Feature Extraction
# ============================================================

def _load_fixed_duration(audio_path: str, sr: int, duration: float) -> np.ndarray:
    """Load audio and ensure exactly `duration` seconds via tiling or trimming."""
    import librosa
    y, _ = librosa.load(audio_path, sr=sr, duration=duration)
    target = int(duration * sr)
    if len(y) < target:
        # Tile to fill the required length (avoids silence bias from zero-padding)
        repeats = (target // len(y)) + 1
        y = np.tile(y, repeats)
    return y[:target]


def extract_mfcc(audio_path: str, sr: int = SAMPLE_RATE,
                 duration: float = AUDIO_DURATION,
                 n_mfcc: int = N_MFCC) -> np.ndarray:
    """
    Extract MFCC features from an audio file.
    Returns a fixed-length feature vector (mean + std of MFCCs).
    """
    y = _load_fixed_duration(audio_path, sr, duration)
    import librosa
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                 hop_length=HOP_LENGTH, n_fft=N_FFT)
    # Aggregate over time: mean and std -> (2 * n_mfcc,) vector
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    return np.concatenate([mfcc_mean, mfcc_std])  # shape: (2*n_mfcc,)


def extract_mel_spectrogram(audio_path: str, sr: int = SAMPLE_RATE,
                            duration: float = AUDIO_DURATION,
                            n_mels: int = N_MELS) -> np.ndarray:
    """
    Extract Mel-spectrogram features from an audio file.
    Returns a fixed-length feature vector (mean + std of Mel bands).
    """
    y = _load_fixed_duration(audio_path, sr, duration)
    import librosa
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                         hop_length=HOP_LENGTH, n_fft=N_FFT)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_mean = np.mean(mel_db, axis=1)
    mel_std = np.std(mel_db, axis=1)
    return np.concatenate([mel_mean, mel_std])  # shape: (2*n_mels,)


def extract_combined_audio_features(audio_path: str) -> np.ndarray:
    """
    Extract combined audio features: MFCC + Mel-spectrogram + Chroma + Spectral Contrast.
    Returns a comprehensive feature vector.
    """
    import librosa
    y = _load_fixed_duration(audio_path, SAMPLE_RATE, AUDIO_DURATION)
    sr = SAMPLE_RATE

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                 hop_length=HOP_LENGTH, n_fft=N_FFT)
    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr,
                                          hop_length=HOP_LENGTH, n_fft=N_FFT)
    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr,
                                                  hop_length=HOP_LENGTH, n_fft=N_FFT)
    # Tonnetz
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)

    features = []
    for feat in [mfcc, chroma, contrast, tonnetz]:
        features.append(np.mean(feat, axis=1))
        features.append(np.std(feat, axis=1))

    return np.concatenate(features)


# ============================================================
# Batch Feature Extraction from Audio Files
# ============================================================

def extract_features_from_directory(audio_dirs: dict,
                                     feature_type: str = "mfcc") -> tuple:
    """
    Extract features from audio files organized in language directories.

    Args:
        audio_dirs: dict mapping language -> directory path
                    e.g., {"english": Path(...), "bangla": Path(...)}
        feature_type: "mfcc", "mel", or "combined"

    Returns:
        features: np.ndarray of shape (n_samples, feature_dim)
        metadata: pd.DataFrame with columns: song_id, filename, language
    """
    extractor = {
        "mfcc": extract_mfcc,
        "mel": extract_mel_spectrogram,
        "combined": extract_combined_audio_features,
    }[feature_type]

    all_features = []
    metadata_rows = []
    song_id = 0

    for language, audio_dir in audio_dirs.items():
        audio_dir = Path(audio_dir)
        if not audio_dir.exists():
            print(f"Warning: {audio_dir} does not exist, skipping.")
            continue

        audio_files = sorted([
            f for f in audio_dir.iterdir()
            if f.suffix.lower() in {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
        ])

        print(f"Extracting {feature_type} features from {len(audio_files)} {language} tracks...")
        for fpath in tqdm(audio_files, desc=language):
            try:
                feat = extractor(str(fpath))
                all_features.append(feat)
                metadata_rows.append({
                    "song_id": song_id,
                    "filename": fpath.name,
                    "language": language,
                    "path": str(fpath),
                })
                song_id += 1
            except Exception as e:
                print(f"  Error processing {fpath.name}: {e}")

    if not all_features:
        raise ValueError("No audio files found! Check your data directories.")

    features = np.array(all_features, dtype=np.float32)
    metadata = pd.DataFrame(metadata_rows)
    return features, metadata


# ============================================================
# Synthetic Dataset Generator (for testing without real audio)
# ============================================================

def generate_synthetic_dataset(n_samples: int = 500,
                                n_features: int = 40,
                                n_clusters: int = 5,
                                n_languages: int = 2,
                                random_state: int = RANDOM_STATE) -> tuple:
    """
    Generate a synthetic dataset that mimics MFCC-like features
    from a hybrid language music collection.

    Creates clusters with different characteristics to simulate
    different genres/languages.

    Args:
        n_samples: total number of samples
        n_features: feature dimensionality (e.g., 2*N_MFCC = 40)
        n_clusters: number of underlying clusters
        n_languages: number of languages (2 = English + Bangla)
        random_state: random seed

    Returns:
        features: np.ndarray (n_samples, n_features)
        metadata: pd.DataFrame with columns: song_id, language, genre, cluster_label
    """
    rng = np.random.RandomState(random_state)

    languages = ["english", "bangla"]
    genres = ["pop", "rock", "classical", "folk", "electronic"]

    samples_per_cluster = n_samples // n_clusters
    all_features = []
    metadata_rows = []

    for cluster_id in range(n_clusters):
        # Each cluster has a different center and spread
        center = rng.randn(n_features) * 3 + cluster_id * 1.5
        spread = rng.uniform(0.5, 2.0)
        cluster_features = rng.randn(samples_per_cluster, n_features) * spread + center

        # Assign language (roughly split across clusters with some mixing)
        lang_idx = cluster_id % n_languages
        for i in range(samples_per_cluster):
            # 80% dominant language, 20% other
            if rng.rand() < 0.8:
                lang = languages[lang_idx]
            else:
                lang = languages[(lang_idx + 1) % n_languages]

            genre = genres[cluster_id % len(genres)]
            song_id = len(all_features)
            all_features.append(cluster_features[i])
            metadata_rows.append({
                "song_id": song_id,
                "language": lang,
                "genre": genre,
                "cluster_label": cluster_id,
            })

    features = np.array(all_features, dtype=np.float32)
    metadata = pd.DataFrame(metadata_rows)

    # Shuffle
    perm = rng.permutation(len(features))
    features = features[perm]
    metadata = metadata.iloc[perm].reset_index(drop=True)
    metadata["song_id"] = range(len(metadata))

    return features, metadata


# ============================================================
# PyTorch Dataset
# ============================================================

class MusicFeatureDataset(Dataset):
    """PyTorch Dataset for music feature vectors."""

    def __init__(self, features: np.ndarray, metadata: pd.DataFrame = None):
        """
        Args:
            features: np.ndarray of shape (n_samples, feature_dim)
            metadata: optional DataFrame with metadata
        """
        self.features = torch.FloatTensor(features)
        self.metadata = metadata
        self.n_samples = features.shape[0]
        self.feature_dim = features.shape[1]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.features[idx]

    def get_dataloader(self, batch_size: int = BATCH_SIZE,
                       shuffle: bool = True) -> DataLoader:
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle,
                          drop_last=False)


# ============================================================
# Feature Normalization
# ============================================================

def normalize_features(features: np.ndarray,
                       method: str = "standard") -> tuple:
    """
    Normalize features.

    Args:
        features: (n_samples, feature_dim)
        method: "standard" (zero mean, unit var) or "minmax" (0-1 range)

    Returns:
        normalized_features, scaler
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    normalized = scaler.fit_transform(features)
    return normalized.astype(np.float32), scaler


# ============================================================
# Save / Load helpers
# ============================================================

def save_features(features: np.ndarray, metadata: pd.DataFrame,
                  feature_name: str = "mfcc"):
    """Save extracted features and metadata to disk."""
    feat_path = FEATURES_DIR / f"{feature_name}_features.npy"
    meta_path = FEATURES_DIR / f"{feature_name}_metadata.csv"
    np.save(feat_path, features)
    metadata.to_csv(meta_path, index=False)
    print(f"Saved features to {feat_path}")
    print(f"Saved metadata to {meta_path}")


def load_features(feature_name: str = "mfcc") -> tuple:
    """Load previously saved features and metadata."""
    feat_path = FEATURES_DIR / f"{feature_name}_features.npy"
    meta_path = FEATURES_DIR / f"{feature_name}_metadata.csv"
    features = np.load(feat_path)
    metadata = pd.read_csv(meta_path)
    print(f"Loaded features: {features.shape} from {feat_path}")
    return features, metadata


# ============================================================
# Multi-Modal Dataset (Medium + Hard tasks)
# ============================================================

class MultiModalMusicDataset(Dataset):
    """
    PyTorch Dataset combining audio features and lyrics embeddings.

    Args:
        audio_features: np.ndarray of shape (n_samples, audio_dim)
        lyrics_embeddings: np.ndarray of shape (n_samples, lyrics_dim)
        metadata: optional DataFrame
        fusion: "concat" — returns a single concatenated tensor (early fusion)
                "separate" — returns dict {"audio": tensor, "lyrics": tensor}
    """

    def __init__(self, audio_features: np.ndarray,
                 lyrics_embeddings: np.ndarray,
                 metadata: pd.DataFrame = None,
                 fusion: str = "concat"):
        assert len(audio_features) == len(lyrics_embeddings), (
            f"audio_features ({len(audio_features)}) and lyrics_embeddings "
            f"({len(lyrics_embeddings)}) must have the same length."
        )
        assert fusion in ("concat", "separate"), \
            "fusion must be 'concat' or 'separate'"

        self.audio = torch.FloatTensor(audio_features)
        self.lyrics = torch.FloatTensor(lyrics_embeddings)
        self.metadata = metadata
        self.fusion = fusion
        self.n_samples = len(audio_features)
        self.audio_dim = audio_features.shape[1]
        self.lyrics_dim = lyrics_embeddings.shape[1]
        self.feature_dim = (
            self.audio_dim + self.lyrics_dim if fusion == "concat"
            else self.audio_dim
        )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self.fusion == "concat":
            return torch.cat([self.audio[idx], self.lyrics[idx]], dim=-1)
        return {"audio": self.audio[idx], "lyrics": self.lyrics[idx]}

    def get_dataloader(self, batch_size: int = BATCH_SIZE,
                       shuffle: bool = True) -> DataLoader:
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle,
                          drop_last=False)

    def get_numpy(self) -> tuple:
        """Return audio and lyrics as numpy arrays (for clustering)."""
        return self.audio.numpy(), self.lyrics.numpy()


def load_lyrics_embeddings(embeddings_name: str = "lyrics_embeddings") -> np.ndarray:
    """Load previously saved lyrics embeddings from disk."""
    emb_path = LYRICS_EMB_DIR / f"{embeddings_name}.npy"
    if not emb_path.exists():
        raise FileNotFoundError(
            f"Lyrics embeddings not found at {emb_path}. "
            "Run src.lyrics.extract_and_save_lyrics_embeddings() first."
        )
    embeddings = np.load(emb_path)
    print(f"Loaded lyrics embeddings: {embeddings.shape} from {emb_path}")
    return embeddings.astype(np.float32)


