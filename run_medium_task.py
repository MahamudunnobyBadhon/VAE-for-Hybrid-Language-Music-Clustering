"""
==========================================================
 MEDIUM TASK: ConvVAE + Hybrid Features + Multi-Clustering
==========================================================

This script runs the complete Medium task pipeline:
  1. Load mel features + lyrics embeddings + metadata
  2. Parse genre labels from filenames (for ARI/NMI ground truth)
  3. Create hybrid features (audio + lyrics)
  4. Train ConvVAE on mel-spectrogram features (pure audio)
  5. Train BasicVAE on hybrid features (audio + lyrics)
  6. Extract latent features from both models
  7. Run K-Means, Agglomerative Clustering, DBSCAN
  8. Evaluate all methods (Silhouette, DB, ARI, NMI, Purity)
  9. Comparison table across all methods
 10. t-SNE / UMAP visualizations -> save to results/medium/

Usage:
    python run_medium_task.py                          # synthetic fallback
    python run_medium_task.py --use-real-audio         # real GTZAN + Bangla audio
    python run_medium_task.py --n-clusters 10          # tune number of clusters
    python run_medium_task.py --skip-umap --epochs 50  # faster run

Author: CSE715 Neural Networks Project
"""

import sys
import os
import re
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    LATENT_DIM, HIDDEN_DIMS, BATCH_SIZE, NUM_EPOCHS, KL_WEIGHT,
    LEARNING_RATE, N_CLUSTERS, RANDOM_STATE, DEVICE,
    RESULTS_DIR, AUDIO_ENGLISH_DIR, AUDIO_BANGLA_DIR,
    N_MFCC, N_MELS, FEATURES_DIR,
)
from src.dataset import (
    generate_synthetic_dataset, normalize_features,
    MusicFeatureDataset, MultiModalMusicDataset,
    extract_features_from_directory, save_features, load_features,
    load_lyrics_embeddings, create_hybrid_features,
)
from src.vae import BasicVAE, ConvVAE
from src.train import train_vae, extract_latent_features
from src.clustering import (
    kmeans_clustering, agglomerative_clustering,
    dbscan_clustering, tune_dbscan,
    pca_kmeans_baseline, find_optimal_k,
)
from src.evaluation import evaluate_clustering, compare_methods
from src.visualization import (
    plot_tsne, plot_umap, plot_cluster_distribution,
    plot_training_curves, plot_comparison_table,
    plot_latent_space_by_language,
)
from src.lyrics import extract_and_save_lyrics_embeddings


# ============================================================
# Helpers
# ============================================================

def set_seed(seed: int = RANDOM_STATE):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_genre_from_filename(filename: str) -> str:
    """
    Extract genre label from audio filename.
    Supported:
      bangla_classical_002.wav        -> "classical"
      blues.00000.wav                 -> "blues"
      english_blues_blues.00000.wav   -> "blues"
      english_magna_00001.mp3         -> "untagged"
    """
    stem = Path(filename).stem.lower()
    # Bangla synthetic: bangla_<genre>_<num>
    m = re.match(r"^bangla_(\w+)_\d+$", stem)
    if m:
        return m.group(1)
    # GTZAN doubled: english_<genre>_<genre>.<number>
    m = re.match(r"^english_([a-z]+)_\1\.\d+", stem)
    if m:
        return m.group(1)
    # GTZAN plain: <genre>.<num>
    m = re.match(r"^([a-z]+)\.\d+$", stem)
    if m:
        return m.group(1)
    # MagnaTagATune: english_magna_<number>
    if "magna" in stem:
        return "untagged"
    # Fallback
    return "unknown"


def add_genre_labels(metadata: pd.DataFrame) -> pd.DataFrame:
    """Parse genre from filename column and add as 'genre' column."""
    metadata = metadata.copy()
    if "genre" not in metadata.columns:
        if "filename" in metadata.columns:
            metadata["genre"] = metadata["filename"].apply(parse_genre_from_filename)
        else:
            metadata["genre"] = "unknown"
    return metadata


def encode_labels(series: pd.Series) -> np.ndarray:
    """Encode string labels to integer indices."""
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    return le.fit_transform(series.fillna("unknown").astype(str))


def safe_evaluate_dbscan(features, labels, labels_true, method_name):
    """Evaluate DBSCAN results, filtering noise points (label == -1)."""
    mask = labels != -1
    n_noise = np.sum(~mask)
    n_valid = np.sum(mask)
    if n_valid < 10 or len(set(labels[mask])) < 2:
        print(f"  Warning: DBSCAN produced {n_noise} noise points "
              f"and {len(set(labels[mask]))} valid clusters - skipping metrics.")
        return {
            "method": method_name,
            "silhouette_score": float("nan"),
            "calinski_harabasz_index": float("nan"),
            "davies_bouldin_index": float("nan"),
            "adjusted_rand_index": float("nan"),
            "normalized_mutual_info": float("nan"),
            "cluster_purity": float("nan"),
        }
    true_filtered = labels_true[mask] if labels_true is not None else None
    return evaluate_clustering(features[mask], labels[mask],
                               labels_true=true_filtered,
                               method_name=method_name)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Medium Task: ConvVAE + Hybrid Clustering")
    parser.add_argument("--use-real-audio", action="store_true",
                        help="Extract features from real audio files")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Synthetic sample count (default: 500)")
    parser.add_argument("--n-clusters", type=int, default=N_CLUSTERS,
                        help=f"Number of clusters (default: {N_CLUSTERS})")
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM,
                        help=f"Latent dimension (default: {LATENT_DIM})")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help=f"Training epochs per model (default: {NUM_EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help=f"Learning rate (default: {LEARNING_RATE})")
    parser.add_argument("--skip-umap", action="store_true",
                        help="Skip UMAP visualizations (faster)")
    parser.add_argument("--genius-csv", type=str, default=None,
                        help="Path to Genius song lyrics CSV (optional)")
    args = parser.parse_args()

    set_seed()

    # Output directory for medium task
    medium_dir = RESULTS_DIR / "medium"
    medium_dir.mkdir(parents=True, exist_ok=True)

    n_clusters = args.n_clusters
    latent_dim = args.latent_dim

    print("\n" + "=" * 65)
    print(" CSE715 - Medium Task: ConvVAE + Hybrid Features + Multi-Clustering")
    print("=" * 65)
    print(f" Device: {DEVICE}  |  Latent dim: {latent_dim}  |  K: {n_clusters}")
    print(f" Epochs: {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")
    print("=" * 65)

    # ===========================================================
    # STEP 1: Load Mel Features + Metadata
    # ===========================================================
    print("\n[STEP 1] Loading mel-spectrogram features + metadata...")

    if args.use_real_audio:
        audio_dirs = {"english": AUDIO_ENGLISH_DIR, "bangla": AUDIO_BANGLA_DIR}
        mel_features, metadata = extract_features_from_directory(
            audio_dirs, feature_type="mel")
        save_features(mel_features, metadata, "mel")
    else:
        try:
            mel_features, metadata = load_features("mel")
        except FileNotFoundError:
            print("  Mel features not found - using synthetic fallback.")
            mel_features, metadata = generate_synthetic_dataset(
                n_samples=args.n_samples,
                n_features=2 * N_MELS,
                n_clusters=n_clusters,
            )

    metadata = add_genre_labels(metadata)
    print(f"  Dataset: {mel_features.shape[0]} samples, {mel_features.shape[1]} mel features")
    print(f"  Languages: {metadata['language'].value_counts().to_dict()}")
    print(f"  Genres: {metadata['genre'].value_counts().to_dict()}")

    # Ground truth: genre labels encoded as integers
    ground_truth = encode_labels(metadata["genre"])
    languages = metadata["language"].values

    # ===========================================================
    # STEP 2: Generate / Load Lyrics Embeddings
    # ===========================================================
    print("\n[STEP 2] Loading lyrics embeddings...")
    lyrics_emb_path = FEATURES_DIR / "lyrics_embeddings" / "lyrics_embeddings.npy"

    if lyrics_emb_path.exists():
        lyrics_embeddings = load_lyrics_embeddings()
    else:
        print("  No embeddings found - generating proxy lyrics via LaBSE...")
        lyrics_embeddings = extract_and_save_lyrics_embeddings(
            metadata,
            genius_csv_path=args.genius_csv,
            model_name="paraphrase-multilingual-MiniLM-L12-v2",
        )

    # Align lengths (lyrics may have been generated from a different metadata)
    if len(lyrics_embeddings) != len(mel_features):
        print(f"  Warning: embedding length mismatch "
              f"({len(lyrics_embeddings)} vs {len(mel_features)}). "
              "Re-generating embeddings for current metadata.")
        lyrics_embeddings = extract_and_save_lyrics_embeddings(
            metadata,
            genius_csv_path=args.genius_csv,
            model_name="paraphrase-multilingual-MiniLM-L12-v2",
        )

    print(f"  Lyrics embeddings shape: {lyrics_embeddings.shape}")

    # ===========================================================
    # STEP 3: Normalize Mel Features + Create Hybrid Features
    # ===========================================================
    print("\n[STEP 3] Normalizing features + creating hybrid representation...")
    mel_norm, _ = normalize_features(mel_features, method="standard")
    hybrid_features = create_hybrid_features(mel_features, lyrics_embeddings,
                                             audio_weight=0.7, lyrics_weight=0.3)
    print(f"  Mel normalized: {mel_norm.shape}")
    print(f"  Hybrid (audio + lyrics): {hybrid_features.shape}")

    # ===========================================================
    # STEP 4: Train ConvVAE on Mel Features
    # ===========================================================
    print("\n[STEP 4] Training ConvVAE on mel-spectrogram features...")
    mel_dataset = MusicFeatureDataset(mel_norm, metadata)
    mel_loader = mel_dataset.get_dataloader(batch_size=args.batch_size, shuffle=True)
    mel_loader_eval = mel_dataset.get_dataloader(batch_size=args.batch_size, shuffle=False)

    conv_vae = ConvVAE(
        input_dim=mel_norm.shape[1],
        latent_dim=latent_dim,
        channels=[32, 64, 128],
    )
    print(f"  ConvVAE parameters: {sum(p.numel() for p in conv_vae.parameters()):,}")

    conv_result = train_vae(
        model=conv_vae,
        train_loader=mel_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        kl_weight=KL_WEIGHT,
        kl_annealing=True,
        model_name="conv_vae_medium",
    )
    plot_training_curves(conv_result["history"],
                         save_path=str(medium_dir / "conv_vae_training_curves.png"))

    # ===========================================================
    # STEP 5: Train BasicVAE on Hybrid Features
    # ===========================================================
    print("\n[STEP 5] Training BasicVAE on hybrid (audio + lyrics) features...")
    hybrid_dataset = MusicFeatureDataset(hybrid_features, metadata)
    hybrid_loader = hybrid_dataset.get_dataloader(batch_size=args.batch_size, shuffle=True)
    hybrid_loader_eval = hybrid_dataset.get_dataloader(batch_size=args.batch_size, shuffle=False)

    hybrid_vae = BasicVAE(
        input_dim=hybrid_features.shape[1],
        latent_dim=latent_dim,
        hidden_dims=HIDDEN_DIMS,
    )
    print(f"  HybridVAE parameters: {sum(p.numel() for p in hybrid_vae.parameters()):,}")

    hybrid_result = train_vae(
        model=hybrid_vae,
        train_loader=hybrid_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        kl_weight=KL_WEIGHT,
        kl_annealing=True,
        model_name="hybrid_vae_medium",
    )
    plot_training_curves(hybrid_result["history"],
                         save_path=str(medium_dir / "hybrid_vae_training_curves.png"))

    # ===========================================================
    # STEP 6: Extract Latent Features
    # ===========================================================
    print("\n[STEP 6] Extracting latent features...")
    conv_latent = extract_latent_features(conv_result["model"], mel_loader_eval)
    hybrid_latent = extract_latent_features(hybrid_result["model"], hybrid_loader_eval)
    print(f"  ConvVAE latent: {conv_latent.shape}")
    print(f"  HybridVAE latent: {hybrid_latent.shape}")

    # PCA baseline (for comparison)
    pca_labels, pca_features = pca_kmeans_baseline(
        mel_norm, n_components=latent_dim, n_clusters=n_clusters)

    # ===========================================================
    # STEP 7: Clustering - K-Means, Agglomerative, DBSCAN
    # ===========================================================
    print("\n[STEP 7] Clustering (K-Means + Agglomerative + DBSCAN)...")

    # K-Means
    conv_km_labels = kmeans_clustering(conv_latent, n_clusters=n_clusters)
    hybrid_km_labels = kmeans_clustering(hybrid_latent, n_clusters=n_clusters)

    # Agglomerative
    print("  Running Agglomerative Clustering...")
    conv_agg_labels = agglomerative_clustering(conv_latent, n_clusters=n_clusters)
    hybrid_agg_labels = agglomerative_clustering(hybrid_latent, n_clusters=n_clusters)

    # DBSCAN (auto-tune on ConvVAE latent space)
    print("  Tuning DBSCAN on ConvVAE latent space...")
    dbscan_result = tune_dbscan(conv_latent)
    best_eps = dbscan_result["best_eps"]
    best_min_s = dbscan_result["best_min_samples"]
    conv_db_labels = dbscan_result["best_labels"]
    hybrid_db_labels = dbscan_clustering(hybrid_latent, eps=best_eps, min_samples=best_min_s)

    # ===========================================================
    # STEP 8: Evaluation
    # ===========================================================
    print("\n[STEP 8] Evaluating all methods...")

    all_results = []

    # ConvVAE methods
    all_results.append(evaluate_clustering(
        conv_latent, conv_km_labels, labels_true=ground_truth,
        method_name="ConvVAE + K-Means"))

    all_results.append(evaluate_clustering(
        conv_latent, conv_agg_labels, labels_true=ground_truth,
        method_name="ConvVAE + Agglomerative"))

    all_results.append(safe_evaluate_dbscan(
        conv_latent, conv_db_labels, ground_truth,
        method_name="ConvVAE + DBSCAN"))

    # HybridVAE methods
    all_results.append(evaluate_clustering(
        hybrid_latent, hybrid_km_labels, labels_true=ground_truth,
        method_name="HybridVAE (audio+lyrics) + K-Means"))

    all_results.append(evaluate_clustering(
        hybrid_latent, hybrid_agg_labels, labels_true=ground_truth,
        method_name="HybridVAE (audio+lyrics) + Agglomerative"))

    all_results.append(safe_evaluate_dbscan(
        hybrid_latent, hybrid_db_labels, ground_truth,
        method_name="HybridVAE + DBSCAN"))

    # PCA baseline
    all_results.append(evaluate_clustering(
        pca_features, pca_labels, labels_true=ground_truth,
        method_name="PCA + K-Means (baseline)"))

    # Comparison table
    comparison_df = compare_methods(all_results)
    metrics_path = medium_dir / "clustering_metrics.csv"
    comparison_df.to_csv(metrics_path)
    print(f"  Saved metrics to {metrics_path}")

    # ===========================================================
    # STEP 9: Visualizations
    # ===========================================================
    print("\n[STEP 9] Creating visualizations...")

    # ConvVAE latent space - clusters (also returns 2D embedding for language plot)
    conv_tsne_emb = plot_tsne(conv_latent, conv_km_labels,
                              title="ConvVAE Latent Space - K-Means Clusters",
                              color_field="Cluster",
                              save_path=str(medium_dir / "conv_vae_tsne_kmeans.png"))

    # ConvVAE latent space - by language (reuse t-SNE embedding)
    plot_latent_space_by_language(
        conv_tsne_emb, languages,
        title="ConvVAE Latent Space by Language",
        save_path=str(medium_dir / "conv_vae_tsne_language.png"))

    # HybridVAE latent space - clusters
    hybrid_tsne_emb = plot_tsne(hybrid_latent, hybrid_km_labels,
                                title="HybridVAE (audio+lyrics) Latent Space - K-Means",
                                color_field="Cluster",
                                save_path=str(medium_dir / "hybrid_vae_tsne_kmeans.png"))

    # HybridVAE latent space - by language
    plot_latent_space_by_language(
        hybrid_tsne_emb, languages,
        title="HybridVAE Latent Space by Language",
        save_path=str(medium_dir / "hybrid_vae_tsne_language.png"))

    # Cluster distributions
    plot_cluster_distribution(conv_km_labels, languages,
                              title="ConvVAE K-Means Cluster Distribution",
                              save_path=str(medium_dir / "conv_vae_cluster_dist.png"))

    plot_cluster_distribution(hybrid_km_labels, languages,
                              title="HybridVAE Cluster Distribution",
                              save_path=str(medium_dir / "hybrid_vae_cluster_dist.png"))

    # Comparison table figure
    plot_comparison_table(comparison_df,
                          save_path=str(medium_dir / "comparison_table.png"))

    # UMAP (optional)
    if not args.skip_umap:
        try:
            plot_umap(conv_latent, conv_km_labels,
                      title="ConvVAE Latent Space - UMAP",
                      save_path=str(medium_dir / "conv_vae_umap.png"))
            plot_umap(hybrid_latent, hybrid_km_labels,
                      title="HybridVAE Latent Space - UMAP",
                      save_path=str(medium_dir / "hybrid_vae_umap.png"))
        except Exception as e:
            print(f"  UMAP skipped: {e}")

    # ===========================================================
    # SUMMARY
    # ===========================================================
    print("\n" + "=" * 65)
    print(" MEDIUM TASK COMPLETE!")
    print("=" * 65)
    print(f"\n Results saved to: {medium_dir}")
    print("\n Key Results (Silhouette | Davies-Bouldin | ARI):")
    for r in all_results:
        sil = r.get("silhouette_score", float("nan"))
        db = r.get("davies_bouldin_index", float("nan"))
        ari = r.get("adjusted_rand_index")
        ari_str = f"{ari:.4f}" if ari is not None else "N/A"
        print(f"   {r['method']:45s} | {sil:.4f} | {db:.4f} | {ari_str}")

    print("\n Done! Check the results/medium/ folder for all outputs.\n")


if __name__ == "__main__":
    main()
