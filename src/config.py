"""
Configuration file for the VAE Music Clustering project.
Contains hyperparameters, paths, and settings.
"""

import os
from pathlib import Path

# ============================================================
# Project Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
AUDIO_DIR = DATA_DIR / "audio"
AUDIO_ENGLISH_DIR = AUDIO_DIR / "english"
AUDIO_BANGLA_DIR = AUDIO_DIR / "bangla"
LYRICS_DIR = DATA_DIR / "lyrics"
FEATURES_DIR = DATA_DIR / "features"
MFCC_DIR = FEATURES_DIR / "mfcc"
MEL_SPEC_DIR = FEATURES_DIR / "mel_spectrograms"
LYRICS_EMB_DIR = FEATURES_DIR / "lyrics_embeddings"
METADATA_PATH = DATA_DIR / "metadata.csv"

RESULTS_DIR = PROJECT_ROOT / "results"
LATENT_VIS_DIR = RESULTS_DIR / "latent_visualization"
CLUSTER_PLOTS_DIR = RESULTS_DIR / "cluster_plots"
RECONSTRUCTIONS_DIR = RESULTS_DIR / "reconstructions"
MODELS_DIR = RESULTS_DIR / "models"

# Create directories
for d in [
    AUDIO_ENGLISH_DIR, AUDIO_BANGLA_DIR, LYRICS_DIR,
    MFCC_DIR, MEL_SPEC_DIR, LYRICS_EMB_DIR,
    LATENT_VIS_DIR, CLUSTER_PLOTS_DIR, RECONSTRUCTIONS_DIR, MODELS_DIR,
    RESULTS_DIR / "clustering_metrics",
]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# Audio Processing
# ============================================================
SAMPLE_RATE = 22050          # Standard librosa sample rate
AUDIO_DURATION = 5           # Seconds per clip (trim/tile to this length for consistency)
N_MFCC = 20                 # Number of MFCC coefficients
N_MELS = 128                # Number of Mel bands
HOP_LENGTH = 512             # Hop length for STFT
N_FFT = 2048                 # FFT window size

# ============================================================
# VAE Hyperparameters
# ============================================================
LATENT_DIM = 32              # Latent space dimensionality
HIDDEN_DIMS = [512, 256]     # Hidden layer sizes for MLP-VAE
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
NUM_EPOCHS = 100
KL_WEIGHT = 1.0              # Weight for KL divergence (β for Beta-VAE)
EARLY_STOPPING_PATIENCE = 10

# ============================================================
# Clustering
# ============================================================
N_CLUSTERS = 5               # Default number of clusters (tune with elbow method)
RANDOM_STATE = 42            # For reproducibility

# ============================================================
# Visualization
# ============================================================
TSNE_PERPLEXITY = 30
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
FIG_SIZE = (10, 8)
DPI = 150

# ============================================================
# Device
# ============================================================
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
