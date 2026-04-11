"""
Visualization module for VAE Music Clustering project.

Contains:
  - t-SNE visualization
  - UMAP visualization
  - Latent space plots
  - Cluster distribution plots
  - Elbow method plot
  - Training loss curves
  - Latent traversal (Beta-VAE disentanglement)
  - Reconstruction examples
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (works without display)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from pathlib import Path
from typing import Optional

from src.config import (
    TSNE_PERPLEXITY, UMAP_N_NEIGHBORS, UMAP_MIN_DIST,
    FIG_SIZE, DPI, LATENT_VIS_DIR, CLUSTER_PLOTS_DIR, RANDOM_STATE,
)


def plot_tsne(features: np.ndarray,
              labels: np.ndarray,
              title: str = "t-SNE Visualization",
              label_names: dict = None,
              save_path: str = None,
              perplexity: int = TSNE_PERPLEXITY,
              color_field: str = "Cluster"):
    """
    Create t-SNE visualization of features colored by labels.

    Args:
        features: (n_samples, n_features)
        labels: (n_samples,) cluster or class labels
        title: plot title
        label_names: optional dict mapping label -> display name
        save_path: path to save figure (if None, saves to default location)
        perplexity: t-SNE perplexity parameter
        color_field: legend title
    """
    print(f"Computing t-SNE (perplexity={perplexity})...")
    tsne = TSNE(n_components=2, perplexity=perplexity,
                random_state=RANDOM_STATE, max_iter=1000)
    embedding = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    unique_labels = sorted(np.unique(labels))

    cmap = matplotlib.colormaps["tab10"].resampled(len(unique_labels))
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        name = label_names[label] if label_names else str(label)
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=[cmap(idx)], label=name, alpha=0.7, s=30, edgecolors='white',
                   linewidth=0.3)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
    ax.legend(title=color_field, fontsize=9, title_fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is None:
        save_path = LATENT_VIS_DIR / f"tsne_{title.replace(' ', '_').lower()}.png"
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved t-SNE plot to {save_path}")
    return embedding


def plot_umap(features: np.ndarray,
              labels: np.ndarray,
              title: str = "UMAP Visualization",
              label_names: dict = None,
              save_path: str = None,
              color_field: str = "Cluster"):
    """
    Create UMAP visualization of features colored by labels.
    """
    import umap

    print(f"Computing UMAP (n_neighbors={UMAP_N_NEIGHBORS})...")
    reducer = umap.UMAP(n_neighbors=UMAP_N_NEIGHBORS,
                        min_dist=UMAP_MIN_DIST,
                        random_state=RANDOM_STATE)
    embedding = reducer.fit_transform(features)

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    unique_labels = sorted(np.unique(labels))
    cmap = matplotlib.colormaps["tab10"].resampled(len(unique_labels))

    for idx, label in enumerate(unique_labels):
        mask = labels == label
        name = label_names[label] if label_names else str(label)
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=[cmap(idx)], label=name, alpha=0.7, s=30, edgecolors='white',
                   linewidth=0.3)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("UMAP Dimension 1", fontsize=11)
    ax.set_ylabel("UMAP Dimension 2", fontsize=11)
    ax.legend(title=color_field, fontsize=9, title_fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is None:
        save_path = LATENT_VIS_DIR / f"umap_{title.replace(' ', '_').lower()}.png"
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved UMAP plot to {save_path}")
    return embedding


def plot_latent_space_by_language(features_2d: np.ndarray,
                                  languages: np.ndarray,
                                  title: str = "Latent Space by Language",
                                  save_path: str = None):
    """Plot 2D features colored by language."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    for lang in sorted(np.unique(languages)):
        mask = languages == lang
        ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                   label=lang.capitalize(), alpha=0.7, s=30,
                   edgecolors='white', linewidth=0.3)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Dimension 1", fontsize=11)
    ax.set_ylabel("Dimension 2", fontsize=11)
    ax.legend(title="Language", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is None:
        save_path = LATENT_VIS_DIR / f"{title.replace(' ', '_').lower()}.png"
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved language plot to {save_path}")


def plot_cluster_distribution(labels_pred: np.ndarray,
                               languages: np.ndarray = None,
                               title: str = "Cluster Distribution",
                               save_path: str = None):
    """
    Plot cluster size distribution, optionally broken down by language.
    """
    import pandas as pd

    fig, axes = plt.subplots(1, 2 if languages is not None else 1,
                             figsize=(14 if languages is not None else 8, 6))

    if languages is None:
        ax = axes
        unique, counts = np.unique(labels_pred, return_counts=True)
        bars = ax.bar(unique, counts, color=sns.color_palette("tab10", len(unique)),
                      edgecolor='black', linewidth=0.5)
        ax.set_xlabel("Cluster", fontsize=11)
        ax.set_ylabel("Number of Samples", fontsize=11)
        ax.set_title(title, fontsize=14, fontweight='bold')
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    str(count), ha='center', fontsize=9)
    else:
        # Left: overall distribution
        ax1 = axes[0]
        unique, counts = np.unique(labels_pred, return_counts=True)
        ax1.bar(unique, counts, color=sns.color_palette("tab10", len(unique)),
                edgecolor='black', linewidth=0.5)
        ax1.set_xlabel("Cluster", fontsize=11)
        ax1.set_ylabel("Number of Samples", fontsize=11)
        ax1.set_title("Cluster Sizes", fontsize=13, fontweight='bold')

        # Right: language breakdown per cluster
        ax2 = axes[1]
        df = pd.DataFrame({"cluster": labels_pred, "language": languages})
        ct = pd.crosstab(df["cluster"], df["language"])
        ct.plot(kind="bar", stacked=True, ax=ax2, colormap="Set2",
                edgecolor='black', linewidth=0.5)
        ax2.set_xlabel("Cluster", fontsize=11)
        ax2.set_ylabel("Count", fontsize=11)
        ax2.set_title("Language Distribution per Cluster", fontsize=13, fontweight='bold')
        ax2.legend(title="Language", fontsize=9)
        ax2.tick_params(axis='x', rotation=0)

    plt.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path is None:
        save_path = CLUSTER_PLOTS_DIR / f"{title.replace(' ', '_').lower()}.png"
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved cluster distribution to {save_path}")


def plot_elbow(k_range: list, inertias: list, silhouettes: list,
               save_path: str = None):
    """Plot elbow method: inertia and silhouette score vs K."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel("Number of Clusters (K)", fontsize=11)
    ax1.set_ylabel("Inertia", fontsize=11)
    ax1.set_title("Elbow Method", fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    ax2.plot(k_range, silhouettes, 'rs-', linewidth=2, markersize=8)
    ax2.set_xlabel("Number of Clusters (K)", fontsize=11)
    ax2.set_ylabel("Silhouette Score", fontsize=11)
    ax2.set_title("Silhouette Score vs K", fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    best_k = k_range[np.argmax(silhouettes)]
    ax2.axvline(x=best_k, color='green', linestyle='--', label=f'Best K={best_k}')
    ax2.legend(fontsize=10)

    plt.tight_layout()
    if save_path is None:
        save_path = CLUSTER_PLOTS_DIR / "elbow_method.png"
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved elbow plot to {save_path}")


def plot_training_curves(train_losses: dict, save_path: str = None):
    """
    Plot training loss curves.

    Args:
        train_losses: dict with keys 'total', 'recon', 'kl'
                      each mapping to a list of per-epoch losses
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (name, losses) in zip(axes, train_losses.items()):
        ax.plot(range(1, len(losses) + 1), losses, linewidth=2)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(f"{name.upper()} Loss", fontsize=11)
        ax.set_title(f"{name.capitalize()} Loss", fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is None:
        save_path = LATENT_VIS_DIR / "training_curves.png"
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved training curves to {save_path}")


def plot_comparison_table(df, save_path: str = None):
    """
    Create a visual comparison table as a figure.

    Args:
        df: pd.DataFrame from compare_methods()
    """
    fig, ax = plt.subplots(figsize=(12, 3 + 0.5 * len(df)))
    ax.axis('off')

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        rowLabels=df.index,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Style header
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', fontweight='bold')
        elif c == -1:
            cell.set_facecolor('#D9E2F3')
            cell.set_text_props(fontweight='bold')

    plt.title("Clustering Method Comparison", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()

    if save_path is None:
        save_path = CLUSTER_PLOTS_DIR / "comparison_table.png"
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved comparison table to {save_path}")


# ============================================================
# Hard Task: Latent Traversal + Reconstruction Examples
# ============================================================

def plot_latent_traversal(model,
                           base_latent: np.ndarray,
                           dim: int,
                           n_steps: int = 10,
                           value_range: tuple = (-3.0, 3.0),
                           save_path: str = None,
                           device=None):
    """
    Visualize Beta-VAE disentanglement by traversing a single latent dimension.

    Interpolates latent dimension `dim` from value_range[0] to value_range[1]
    while keeping all other dimensions fixed at base_latent. Decodes each
    interpolated vector and shows how the decoded feature vector changes.

    The resulting plot is a heatmap where:
      - rows = interpolation steps (bottom = min, top = max value)
      - columns = feature dimensions (first 64 shown for readability)

    Args:
        model: trained BetaVAE (or any VAE with a decode() method)
        base_latent: (latent_dim,) base latent vector (e.g. dataset mean)
        dim: latent dimension index to traverse
        n_steps: number of interpolation steps
        value_range: (min_value, max_value) for the traversed dimension
        save_path: path to save figure; defaults to LATENT_VIS_DIR
        device: torch device; defaults to CPU if not specified
    """
    import torch
    from src.config import RECONSTRUCTIONS_DIR

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    model = model.to(device)

    values = np.linspace(value_range[0], value_range[1], n_steps)
    decoded_vectors = []

    with torch.no_grad():
        for val in values:
            z = torch.FloatTensor(base_latent.copy()).unsqueeze(0).to(device)
            z[0, dim] = val
            recon = model.decode(z)
            decoded_vectors.append(recon.cpu().numpy().squeeze())

    decoded_matrix = np.stack(decoded_vectors, axis=0)  # (n_steps, feature_dim)

    # Show first 64 feature dims for readability
    n_show = min(64, decoded_matrix.shape[1])
    decoded_matrix = decoded_matrix[:, :n_show]

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(decoded_matrix, aspect='auto', cmap='RdBu_r', origin='lower')
    plt.colorbar(im, ax=ax, label='Decoded Feature Value')

    step_labels = [f"{v:.1f}" for v in values]
    ax.set_yticks(range(n_steps))
    ax.set_yticklabels(step_labels, fontsize=8)
    ax.set_ylabel(f"Latent dim {dim} value", fontsize=11)
    ax.set_xlabel(f"Feature dimension (first {n_show})", fontsize=11)
    ax.set_title(f"Latent Traversal — Dimension {dim}", fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path is None:
        save_path = LATENT_VIS_DIR / f"latent_traversal_dim_{dim}.png"
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved latent traversal (dim {dim}) to {save_path}")


def plot_reconstruction_examples(model,
                                  features: np.ndarray,
                                  n_examples: int = 6,
                                  n_dims_shown: int = 20,
                                  save_path: str = None,
                                  device=None):
    """
    Plot original vs. reconstructed feature vectors for n_examples samples.

    Each subplot shows a bar chart with original (blue) and reconstructed
    (orange) values for the first n_dims_shown feature dimensions.

    Args:
        model: trained VAE (BasicVAE, BetaVAE, or ConvVAE)
        features: (n_samples, feature_dim) numpy array
        n_examples: number of example samples to show
        n_dims_shown: number of feature dimensions to plot per example
        save_path: path to save figure; defaults to RECONSTRUCTIONS_DIR
        device: torch device; defaults to model's device
    """
    import torch
    from src.config import RECONSTRUCTIONS_DIR

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    model = model.to(device)

    indices = np.random.choice(len(features), size=n_examples, replace=False)
    n_cols = min(3, n_examples)
    n_rows = int(np.ceil(n_examples / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 4 * n_rows))
    if n_examples == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    x_pos = np.arange(n_dims_shown)
    width = 0.35

    with torch.no_grad():
        for plot_idx, sample_idx in enumerate(indices):
            row, col = divmod(plot_idx, n_cols)
            ax = axes[row, col]

            x = torch.FloatTensor(features[sample_idx]).unsqueeze(0).to(device)
            output = model(x)
            original = features[sample_idx, :n_dims_shown]
            reconstructed = output["reconstruction"].cpu().numpy().squeeze()[:n_dims_shown]

            ax.bar(x_pos - width / 2, original, width,
                   label='Original', color='steelblue', alpha=0.8)
            ax.bar(x_pos + width / 2, reconstructed, width,
                   label='Reconstructed', color='coral', alpha=0.8)

            ax.set_title(f"Sample {sample_idx}", fontsize=11)
            ax.set_xlabel(f"Feature dim (first {n_dims_shown})", fontsize=9)
            ax.set_ylabel("Value", fontsize=9)
            if plot_idx == 0:
                ax.legend(fontsize=9)

    # Hide empty axes
    for plot_idx in range(n_examples, n_rows * n_cols):
        row, col = divmod(plot_idx, n_cols)
        axes[row, col].set_visible(False)

    plt.suptitle("VAE Reconstruction Examples", fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path is None:
        save_path = RECONSTRUCTIONS_DIR / "reconstruction_examples.png"
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved reconstruction examples to {save_path}")
