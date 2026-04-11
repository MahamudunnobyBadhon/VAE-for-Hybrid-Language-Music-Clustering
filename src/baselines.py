"""
Baseline models for the Hard task comparison.

Contains:
  - Autoencoder: standard (non-variational) autoencoder baseline
  - ae_loss: MSE reconstruction loss (no KL term)
  - train_autoencoder: training loop for the Autoencoder
  - spectral_clustering_baseline: SpectralClustering on raw features
  - direct_feature_kmeans: K-Means on raw features (no dimensionality reduction)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List
from pathlib import Path

from src.config import (
    DEVICE, LEARNING_RATE, NUM_EPOCHS, EARLY_STOPPING_PATIENCE,
    MODELS_DIR, N_CLUSTERS, RANDOM_STATE,
)


# ============================================================
# Autoencoder (non-variational)
# ============================================================

class Autoencoder(nn.Module):
    """
    Standard (non-variational) Autoencoder — baseline for Hard task comparison.

    Architecture is identical to BasicVAE encoder/decoder but:
      - No mu / logvar split: encodes directly to a deterministic latent vector
      - No KL divergence term in loss (pure reconstruction)
      - get_latent() returns z directly (no sampling)

    Args:
        input_dim: input feature dimension
        latent_dim: bottleneck dimension
        hidden_dims: hidden layer sizes (reversed for decoder)
    """

    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: List[int] = None):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        if hidden_dims is None:
            hidden_dims = [512, 256]
        self.hidden_dims = hidden_dims

        # ----- Encoder -----
        enc_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            enc_layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            ]
            in_dim = h_dim
        enc_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # ----- Decoder -----
        dec_layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            dec_layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            ]
            in_dim = h_dim
        dec_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent vector (deterministic)."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Returns dict with:
            reconstruction: reconstructed input
            z: latent vector
        """
        z = self.encode(x)
        reconstruction = self.decode(z)
        return {"reconstruction": reconstruction, "z": z}

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation (no sampling)."""
        return self.encode(x)


def ae_loss(recon_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Autoencoder loss: pure MSE reconstruction (no KL term).

    Args:
        recon_x: reconstructed input
        x: original input

    Returns:
        scalar MSE loss
    """
    return torch.nn.functional.mse_loss(recon_x, x, reduction='mean')


def train_autoencoder(model: Autoencoder,
                      train_loader,
                      num_epochs: int = NUM_EPOCHS,
                      learning_rate: float = LEARNING_RATE,
                      device: torch.device = DEVICE,
                      patience: int = EARLY_STOPPING_PATIENCE,
                      model_name: str = "autoencoder") -> dict:
    """
    Train the Autoencoder baseline.

    Args:
        model: Autoencoder instance
        train_loader: DataLoader (same format as VAE — yields plain feature tensors)
        num_epochs: training epochs
        learning_rate: optimizer learning rate
        device: training device
        patience: early stopping patience
        model_name: checkpoint filename stem

    Returns:
        dict with model, history (recon losses), best_loss
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    history = {"recon": []}
    best_loss = float("inf")
    patience_counter = 0
    best_state = None

    print(f"\n{'='*60}")
    print(f" Training {model_name} (Autoencoder) on {device}")
    print(f" Epochs: {num_epochs}, LR: {learning_rate}")
    print(f"{'='*60}\n")

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            output = model(batch)
            loss = ae_loss(output["reconstruction"], batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        history["recon"].append(avg_loss)
        scheduler.step(avg_loss)

        if epoch % 10 == 0 or epoch == 1 or epoch == num_epochs:
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:3d}/{num_epochs} | "
                  f"Recon Loss: {avg_loss:.4f} | LR: {lr:.1e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    save_path = MODELS_DIR / f"{model_name}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": model.input_dim,
        "latent_dim": model.latent_dim,
        "hidden_dims": model.hidden_dims,
        "history": history,
    }, save_path)
    print(f"\n  Model saved to {save_path}")

    return {"model": model, "history": history, "best_loss": best_loss}


@torch.no_grad()
def extract_ae_latent(model: Autoencoder,
                      data_loader,
                      device: torch.device = DEVICE) -> np.ndarray:
    """Extract latent features from a trained Autoencoder."""
    model = model.to(device)
    model.eval()
    all_latent = []
    for batch in data_loader:
        batch = batch.to(device)
        z = model.get_latent(batch)
        all_latent.append(z.cpu().numpy())
    return np.concatenate(all_latent, axis=0)


# ============================================================
# Spectral Clustering baseline
# ============================================================

def spectral_clustering_baseline(features: np.ndarray,
                                  n_clusters: int = N_CLUSTERS,
                                  n_neighbors: int = 10,
                                  max_samples: int = 500,
                                  random_state: int = RANDOM_STATE) -> np.ndarray:
    """
    Spectral Clustering on raw features — "direct spectral feature clustering" baseline.

    Uses nearest-neighbour affinity graph construction. Note: expensive on
    large datasets (O(n²) memory for the affinity matrix). Randomly subsamples
    to max_samples if n_samples > max_samples to avoid memory issues.

    Returns labels for the FULL dataset: subsampled points are clustered with
    SpectralClustering; remaining points are assigned to the nearest subsampled
    centroid using K-Means on the same reduced features.

    Args:
        features: (n_samples, n_features)
        n_clusters: number of clusters
        n_neighbors: number of neighbours for affinity graph
        max_samples: maximum samples to use for spectral clustering (default 500)
        random_state: random seed

    Returns:
        cluster_labels: (n_samples,) labels for all input samples
    """
    from sklearn.cluster import SpectralClustering, KMeans

    n_samples = len(features)
    rng = np.random.RandomState(random_state)

    if n_samples > max_samples:
        print(f"  Note: Spectral clustering subsampling {n_samples} → {max_samples} "
              f"samples (O(n²) memory constraint).")
        idx = rng.choice(n_samples, size=max_samples, replace=False)
        sub_features = features[idx]
    else:
        idx = np.arange(n_samples)
        sub_features = features

    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity='nearest_neighbors',
        n_neighbors=n_neighbors,
        random_state=random_state,
        n_jobs=-1,
    )
    sub_labels = sc.fit_predict(sub_features)
    print(f"  Spectral clustering: {len(set(sub_labels))} clusters on {len(sub_features)} samples")

    if n_samples <= max_samples:
        return sub_labels

    # Assign remaining points by nearest centroid (KMeans on full data, initialised
    # from spectral cluster centres)
    centroids = np.array([
        sub_features[sub_labels == k].mean(axis=0)
        for k in range(n_clusters)
        if (sub_labels == k).any()
    ])
    km = KMeans(n_clusters=len(centroids), init=centroids, n_init=1,
                max_iter=300, random_state=random_state)
    all_labels = km.fit_predict(features)
    print(f"  Full-dataset assignment via nearest centroid: shape={all_labels.shape}")
    return all_labels


# ============================================================
# Direct feature K-Means baseline
# ============================================================

def direct_feature_kmeans(features: np.ndarray,
                           n_clusters: int = N_CLUSTERS,
                           random_state: int = RANDOM_STATE) -> np.ndarray:
    """
    K-Means directly on raw/normalized features without any VAE or AE.

    Serves as the "direct spectral/combined feature clustering" baseline
    for the Hard task comparison table.

    Args:
        features: (n_samples, n_features) — typically combined audio features
        n_clusters: number of clusters
        random_state: random seed

    Returns:
        cluster_labels: (n_samples,)
    """
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_clusters, random_state=random_state,
                n_init=10, max_iter=300)
    labels = km.fit_predict(features)
    print(f"  Direct K-Means: inertia={km.inertia_:.1f}")
    return labels
