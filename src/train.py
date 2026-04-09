"""
Training module for VAE models.

Handles:
  - Training loop with KL annealing
  - Validation
  - Early stopping
  - Latent feature extraction
  - Model saving/loading
"""

import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from pathlib import Path

from src.vae import BasicVAE, vae_loss, CVAE, MultiModalVAE
from src.config import (
    DEVICE, LEARNING_RATE, NUM_EPOCHS, KL_WEIGHT,
    EARLY_STOPPING_PATIENCE, MODELS_DIR,
)


def train_vae(model: BasicVAE,
              train_loader,
              num_epochs: int = NUM_EPOCHS,
              learning_rate: float = LEARNING_RATE,
              kl_weight: float = KL_WEIGHT,
              kl_annealing: bool = True,
              device: torch.device = DEVICE,
              patience: int = EARLY_STOPPING_PATIENCE,
              model_name: str = "basic_vae") -> dict:
    """
    Train a VAE model.

    Args:
        model: VAE model
        train_loader: DataLoader
        num_epochs: number of training epochs
        learning_rate: optimizer learning rate
        kl_weight: final KL weight (beta)
        kl_annealing: whether to gradually increase KL weight
        device: training device
        patience: early stopping patience
        model_name: name for saving

    Returns:
        dict with training history and trained model
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    history = {
        "total": [],
        "recon": [],
        "kl": [],
    }

    best_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    print(f"\n{'='*60}")
    print(f" Training {model_name} on {device}")
    print(f" Epochs: {num_epochs}, LR: {learning_rate}, KL Weight: {kl_weight}")
    print(f" KL Annealing: {kl_annealing}")
    print(f"{'='*60}\n")

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_total = 0
        epoch_recon = 0
        epoch_kl = 0
        n_batches = 0

        # KL annealing: linearly increase from 0 to kl_weight over first 30% of epochs
        if kl_annealing:
            anneal_epochs = max(1, int(0.3 * num_epochs))
            current_kl_weight = min(kl_weight, kl_weight * epoch / anneal_epochs)
        else:
            current_kl_weight = kl_weight

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            output = model(batch)
            losses = vae_loss(
                output["reconstruction"], batch,
                output["mu"], output["logvar"],
                kl_weight=current_kl_weight
            )

            losses["total_loss"].backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_total += losses["total_loss"].item()
            epoch_recon += losses["recon_loss"].item()
            epoch_kl += losses["kl_loss"].item()
            n_batches += 1

        # Average losses
        avg_total = epoch_total / n_batches
        avg_recon = epoch_recon / n_batches
        avg_kl = epoch_kl / n_batches

        history["total"].append(avg_total)
        history["recon"].append(avg_recon)
        history["kl"].append(avg_kl)

        scheduler.step(avg_total)

        # Print every 10 epochs or last epoch
        if epoch % 10 == 0 or epoch == 1 or epoch == num_epochs:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:3d}/{num_epochs} | "
                  f"Total: {avg_total:.4f} | "
                  f"Recon: {avg_recon:.4f} | "
                  f"KL: {avg_kl:.4f} | "
                  f"beta: {current_kl_weight:.3f} | "
                  f"LR: {current_lr:.1e}")

        # Early stopping
        if avg_total < best_loss:
            best_loss = avg_total
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save model
    save_path = MODELS_DIR / f"{model_name}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": model.input_dim,
        "latent_dim": model.latent_dim,
        "hidden_dims": getattr(model, "hidden_dims", None),
        "history": history,
    }, save_path)
    print(f"\n  Model saved to {save_path}")

    return {
        "model": model,
        "history": history,
        "best_loss": best_loss,
    }


@torch.no_grad()
def extract_latent_features(model: BasicVAE,
                            data_loader,
                            device: torch.device = DEVICE) -> np.ndarray:
    """
    Extract latent features (mu) from a trained VAE.

    Args:
        model: trained VAE
        data_loader: DataLoader
        device: device

    Returns:
        latent_features: np.ndarray (n_samples, latent_dim)
    """
    model = model.to(device)
    model.eval()

    all_latent = []
    for batch in data_loader:
        batch = batch.to(device)
        mu = model.get_latent(batch)
        all_latent.append(mu.cpu().numpy())

    return np.concatenate(all_latent, axis=0)



# ============================================================
# CVAE Training - Hard Task
# ============================================================

def train_cvae(model: CVAE,
               train_loader,
               num_epochs: int = NUM_EPOCHS,
               learning_rate: float = LEARNING_RATE,
               kl_weight: float = KL_WEIGHT,
               kl_annealing: bool = True,
               device: torch.device = DEVICE,
               patience: int = EARLY_STOPPING_PATIENCE,
               model_name: str = "cvae") -> dict:
    """
    Train a Conditional VAE.

    The DataLoader must yield tuples (x, c) where:
      x: feature tensor (B, input_dim)
      c: condition one-hot tensor (B, condition_dim)

    Args:
        model: CVAE instance
        train_loader: DataLoader yielding (x, c) tuples
        num_epochs: number of training epochs
        learning_rate: optimizer learning rate
        kl_weight: final KL weight (beta)
        kl_annealing: if True, linearly anneal KL weight
        device: training device
        patience: early stopping patience
        model_name: checkpoint filename stem

    Returns:
        dict with model, history, best_loss
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    history = {"total": [], "recon": [], "kl": []}
    best_loss = float("inf")
    patience_counter = 0
    best_state = None

    print(f"\n{'='*60}")
    print(f" Training {model_name} (CVAE) on {device}")
    print(f" Epochs: {num_epochs}, LR: {learning_rate}, KL Weight: {kl_weight}")
    print(f"{'='*60}\n")

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_total = epoch_recon = epoch_kl = 0.0
        n_batches = 0

        if kl_annealing:
            anneal_epochs = max(1, int(0.3 * num_epochs))
            current_kl = min(kl_weight, kl_weight * epoch / anneal_epochs)
        else:
            current_kl = kl_weight

        for x, c in train_loader:
            x, c = x.to(device), c.to(device)
            optimizer.zero_grad()

            output = model(x, c)
            losses = vae_loss(output["reconstruction"], x,
                              output["mu"], output["logvar"],
                              kl_weight=current_kl)
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_total += losses["total_loss"].item()
            epoch_recon += losses["recon_loss"].item()
            epoch_kl += losses["kl_loss"].item()
            n_batches += 1

        avg_total = epoch_total / n_batches
        avg_recon = epoch_recon / n_batches
        avg_kl = epoch_kl / n_batches

        history["total"].append(avg_total)
        history["recon"].append(avg_recon)
        history["kl"].append(avg_kl)
        scheduler.step(avg_total)

        if epoch % 10 == 0 or epoch == 1 or epoch == num_epochs:
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:3d}/{num_epochs} | "
                  f"Total: {avg_total:.4f} | Recon: {avg_recon:.4f} | "
                  f"KL: {avg_kl:.4f} | beta: {current_kl:.3f} | LR: {lr:.1e}")

        if avg_total < best_loss:
            best_loss = avg_total
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
        "condition_dim": model.condition_dim,
        "hidden_dims": model.hidden_dims,
        "history": history,
    }, save_path)
    print(f"\n  Model saved to {save_path}")

    return {"model": model, "history": history, "best_loss": best_loss}


@torch.no_grad()
def extract_latent_cvae(model: CVAE,
                        data_loader,
                        device: torch.device = DEVICE) -> np.ndarray:
    """
    Extract latent features (mu) from a trained CVAE.

    The DataLoader must yield (x, c) tuples.
    """
    model = model.to(device)
    model.eval()
    all_latent = []
    for x, c in data_loader:
        x, c = x.to(device), c.to(device)
        mu = model.get_latent(x, c)
        all_latent.append(mu.cpu().numpy())
    return np.concatenate(all_latent, axis=0)


# ============================================================
# MultiModalVAE Training - Hard Task
# ============================================================

def train_multimodal_vae(model: MultiModalVAE,
                          train_loader,
                          num_epochs: int = NUM_EPOCHS,
                          learning_rate: float = LEARNING_RATE,
                          kl_weight: float = KL_WEIGHT,
                          kl_annealing: bool = True,
                          device: torch.device = DEVICE,
                          patience: int = EARLY_STOPPING_PATIENCE,
                          model_name: str = "multimodal_vae") -> dict:
    """
    Train a MultiModalVAE.

    The DataLoader must yield dicts with keys "audio" and "lyrics".

    Args:
        model: MultiModalVAE instance
        train_loader: DataLoader yielding {"audio": tensor, "lyrics": tensor}
        num_epochs: training epochs
        learning_rate: optimizer learning rate
        kl_weight: KL divergence weight
        kl_annealing: if True, linearly anneal KL weight
        device: training device
        patience: early stopping patience
        model_name: checkpoint filename stem

    Returns:
        dict with model, history, best_loss
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    history = {"total": [], "recon": [], "kl": []}
    best_loss = float("inf")
    patience_counter = 0
    best_state = None

    print(f"\n{'='*60}")
    print(f" Training {model_name} (MultiModalVAE) on {device}")
    print(f" Epochs: {num_epochs}, LR: {learning_rate}, KL Weight: {kl_weight}")
    print(f"{'='*60}\n")

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_total = epoch_recon = epoch_kl = 0.0
        n_batches = 0

        if kl_annealing:
            anneal_epochs = max(1, int(0.3 * num_epochs))
            current_kl = min(kl_weight, kl_weight * epoch / anneal_epochs)
        else:
            current_kl = kl_weight

        for batch in train_loader:
            audio = batch["audio"].to(device)
            lyrics = batch["lyrics"].to(device)
            optimizer.zero_grad()

            output = model(audio, lyrics)
            losses = vae_loss(output["reconstruction"], audio,
                              output["mu"], output["logvar"],
                              kl_weight=current_kl)
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_total += losses["total_loss"].item()
            epoch_recon += losses["recon_loss"].item()
            epoch_kl += losses["kl_loss"].item()
            n_batches += 1

        avg_total = epoch_total / n_batches
        avg_recon = epoch_recon / n_batches
        avg_kl = epoch_kl / n_batches

        history["total"].append(avg_total)
        history["recon"].append(avg_recon)
        history["kl"].append(avg_kl)
        scheduler.step(avg_total)

        if epoch % 10 == 0 or epoch == 1 or epoch == num_epochs:
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:3d}/{num_epochs} | "
                  f"Total: {avg_total:.4f} | Recon: {avg_recon:.4f} | "
                  f"KL: {avg_kl:.4f} | beta: {current_kl:.3f} | LR: {lr:.1e}")

        if avg_total < best_loss:
            best_loss = avg_total
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
        "audio_dim": model.audio_dim,
        "lyrics_dim": model.lyrics_dim,
        "latent_dim": model.latent_dim,
        "hidden_dims": model.hidden_dims,
        "history": history,
    }, save_path)
    print(f"\n  Model saved to {save_path}")

    return {"model": model, "history": history, "best_loss": best_loss}


@torch.no_grad()
def extract_latent_multimodal(model: MultiModalVAE,
                               data_loader,
                               device: torch.device = DEVICE) -> np.ndarray:
    """
    Extract latent features (mu) from a trained MultiModalVAE.

    The DataLoader must yield dicts with "audio" and "lyrics" keys.
    """
    model = model.to(device)
    model.eval()
    all_latent = []
    for batch in data_loader:
        audio = batch["audio"].to(device)
        lyrics = batch["lyrics"].to(device)
        mu = model.get_latent(audio, lyrics)
        all_latent.append(mu.cpu().numpy())
    return np.concatenate(all_latent, axis=0)
