"""
==========================================================
 EASY TASK: Basic VAE + K-Means Music Clustering
==========================================================

This script runs the complete Easy task pipeline:
  1. Generate/load dataset (synthetic for testing OR real audio)
  2. Normalize features
  3. Train basic MLP-VAE
  4. Extract latent representations
  5. K-Means clustering on VAE latent space
  6. Baseline: PCA + K-Means on raw features
  7. Evaluate both with Silhouette Score & Calinski-Harabasz Index
  8. Visualize with t-SNE
  9. Save all results

Usage:
    python run_easy_task.py                    # Use synthetic data (testing)
    python run_easy_task.py --use-real-audio    # Use real audio files

Author: CSE715 Neural Networks Project
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    LATENT_DIM, HIDDEN_DIMS, BATCH_SIZE, NUM_EPOCHS, KL_WEIGHT,
    LEARNING_RATE, N_CLUSTERS, RANDOM_STATE, DEVICE,
    RESULTS_DIR, CLUSTER_PLOTS_DIR, LATENT_VIS_DIR,
    AUDIO_ENGLISH_DIR, AUDIO_BANGLA_DIR, N_MFCC,
)
from src.dataset import (
    generate_synthetic_dataset, normalize_features,
    MusicFeatureDataset, extract_features_from_directory,
    save_features, load_features,
)
from src.vae import BasicVAE
from src.train import train_vae, extract_latent_features
from src.clustering import kmeans_clustering, pca_kmeans_baseline, find_optimal_k
from src.evaluation import evaluate_clustering, compare_methods
from src.visualization import (
    plot_tsne, plot_cluster_distribution, plot_elbow,
    plot_training_curves, plot_comparison_table,
    plot_latent_space_by_language,
)


def set_seed(seed=RANDOM_STATE):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Easy Task: Basic VAE + K-Means")
    parser.add_argument("--use-real-audio", action="store_true",
                        help="Use real audio files instead of synthetic data")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Number of synthetic samples (default: 500)")
    parser.add_argument("--n-clusters", type=int, default=N_CLUSTERS,
                        help=f"Number of clusters (default: {N_CLUSTERS})")
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM,
                        help=f"Latent dimension (default: {LATENT_DIM})")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help=f"Training epochs (default: {NUM_EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help=f"Learning rate (default: {LEARNING_RATE})")
    parser.add_argument("--skip-umap", action="store_true",
                        help="Skip UMAP visualization (faster)")
    args = parser.parse_args()

    set_seed()
    n_clusters = args.n_clusters
    latent_dim = args.latent_dim

    print("\n" + "=" * 60)
    print(" CSE715 - Easy Task: Basic VAE + K-Means Clustering")
    print("=" * 60)
    print(f" Device: {DEVICE}")
    print(f" Latent dim: {latent_dim}, Clusters: {n_clusters}")
    print(f" Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print("=" * 60)

    # ===========================================================
    # STEP 1: Load or Generate Dataset
    # ===========================================================
    print("\n[STEP 1] Preparing dataset...")

    if args.use_real_audio:
        print("  Using real audio files...")
        audio_dirs = {
            "english": AUDIO_ENGLISH_DIR,
            "bangla": AUDIO_BANGLA_DIR,
        }
        features, metadata = extract_features_from_directory(
            audio_dirs, feature_type="mfcc"
        )
        save_features(features, metadata, "mfcc")
        ground_truth = None  # No ground truth for unsupervised
        languages = metadata["language"].values
    else:
        print("  Generating synthetic dataset for testing...")
        features, metadata = generate_synthetic_dataset(
            n_samples=args.n_samples,
            n_features=2 * N_MFCC,  # Simulates MFCC mean+std
            n_clusters=n_clusters,
        )
        ground_truth = metadata["cluster_label"].values
        languages = metadata["language"].values
        print(f"  Generated {features.shape[0]} samples, {features.shape[1]} features")
        print(f"  Languages: {metadata['language'].value_counts().to_dict()}")
        print(f"  Genres: {metadata['genre'].value_counts().to_dict()}")

    raw_features = features.copy()

    # ===========================================================
    # STEP 2: Normalize Features
    # ===========================================================
    print("\n[STEP 2] Normalizing features...")
    features_norm, scaler = normalize_features(features, method="standard")
    print(f"  Normalized shape: {features_norm.shape}")
    print(f"  Mean ~= {features_norm.mean():.4f}, Std ~= {features_norm.std():.4f}")

    # ===========================================================
    # STEP 3: Find Optimal K (Elbow Method)
    # ===========================================================
    print("\n[STEP 3] Finding optimal number of clusters (elbow method)...")
    k_results = find_optimal_k(features_norm, k_range=range(2, 11))
    plot_elbow(k_results["k_range"], k_results["inertias"],
               k_results["silhouette_scores"])
    best_k = k_results["best_k"]
    print(f"  Using K = {n_clusters} (specified) | Best by silhouette: {best_k}")

    # ===========================================================
    # STEP 4: Train Basic VAE
    # ===========================================================
    print("\n[STEP 4] Training Basic MLP-VAE...")
    input_dim = features_norm.shape[1]

    dataset = MusicFeatureDataset(features_norm, metadata)
    train_loader = dataset.get_dataloader(batch_size=args.batch_size, shuffle=True)

    model = BasicVAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=HIDDEN_DIMS,
    )
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    train_result = train_vae(
        model=model,
        train_loader=train_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        kl_weight=KL_WEIGHT,
        kl_annealing=True,
        model_name="basic_vae_easy",
    )

    # Plot training curves
    plot_training_curves(train_result["history"])

    # ===========================================================
    # STEP 5: Extract Latent Features
    # ===========================================================
    print("\n[STEP 5] Extracting latent features...")
    # Use non-shuffled loader for consistent ordering
    eval_loader = dataset.get_dataloader(batch_size=args.batch_size, shuffle=False)
    latent_features = extract_latent_features(train_result["model"], eval_loader)
    print(f"  Latent features shape: {latent_features.shape}")

    # ===========================================================
    # STEP 6: Clustering
    # ===========================================================
    print("\n[STEP 6] Clustering...")

    # 6a: VAE + K-Means
    print("  6a: VAE Latent + K-Means")
    vae_kmeans_labels = kmeans_clustering(latent_features, n_clusters=n_clusters)

    # 6b: PCA + K-Means Baseline
    print("  6b: PCA + K-Means Baseline")
    pca_labels, pca_features = pca_kmeans_baseline(
        features_norm, n_components=latent_dim, n_clusters=n_clusters
    )

    # 6c: Raw Features + K-Means Baseline
    print("  6c: Raw Features + K-Means Baseline")
    raw_kmeans_labels = kmeans_clustering(features_norm, n_clusters=n_clusters)

    # ===========================================================
    # STEP 7: Evaluation
    # ===========================================================
    print("\n[STEP 7] Evaluating clustering quality...")

    results = []

    # VAE + K-Means
    r1 = evaluate_clustering(
        latent_features, vae_kmeans_labels,
        labels_true=ground_truth,
        method_name="VAE + K-Means"
    )
    results.append(r1)

    # PCA + K-Means
    r2 = evaluate_clustering(
        pca_features, pca_labels,
        labels_true=ground_truth,
        method_name="PCA + K-Means"
    )
    results.append(r2)

    # Raw + K-Means
    r3 = evaluate_clustering(
        features_norm, raw_kmeans_labels,
        labels_true=ground_truth,
        method_name="Raw Features + K-Means"
    )
    results.append(r3)

    # Comparison table
    comparison_df = compare_methods(results)
    comparison_df.to_csv(RESULTS_DIR / "clustering_metrics.csv")
    print(f"  Saved metrics to {RESULTS_DIR / 'clustering_metrics.csv'}")

    # ===========================================================
    # STEP 8: Visualization
    # ===========================================================
    print("\n[STEP 8] Creating visualizations...")

    # t-SNE of VAE latent space colored by cluster
    tsne_emb = plot_tsne(
        latent_features, vae_kmeans_labels,
        title="VAE Latent Space - K-Means Clusters",
        color_field="Cluster"
    )

    # t-SNE of VAE latent space colored by language
    plot_latent_space_by_language(
        tsne_emb, languages,
        title="VAE Latent Space by Language (t-SNE)"
    )

    # t-SNE of PCA baseline colored by cluster
    plot_tsne(
        pca_features, pca_labels,
        title="PCA Baseline - K-Means Clusters",
        color_field="Cluster"
    )

    # Cluster distributions
    plot_cluster_distribution(
        vae_kmeans_labels, languages,
        title="VAE K-Means Cluster Distribution"
    )

    # Comparison table as figure
    plot_comparison_table(comparison_df)

    # UMAP (optional - can be slow)
    if not args.skip_umap:
        try:
            from src.visualization import plot_umap
            plot_umap(
                latent_features, vae_kmeans_labels,
                title="VAE Latent Space - UMAP",
                color_field="Cluster"
            )
        except Exception as e:
            print(f"  UMAP skipped due to error: {e}")

    # ===========================================================
    # SUMMARY
    # ===========================================================
    print("\n" + "=" * 60)
    print(" EASY TASK COMPLETE!")
    print("=" * 60)
    print(f"\n Results saved to: {RESULTS_DIR}")
    print(f" Visualizations saved to:")
    print(f"   - {LATENT_VIS_DIR}")
    print(f"   - {CLUSTER_PLOTS_DIR}")
    print(f"\n Key Results:")
    for r in results:
        sil = r['silhouette_score']
        ch = r['calinski_harabasz_index']
        m = r['method']
        print(f"   {m:30s} | Silhouette: {sil:.4f} | CH Index: {ch:.1f}")
        if r.get('adjusted_rand_index') is not None:
            print(f"   {'':30s} | ARI: {r['adjusted_rand_index']:.4f} | NMI: {r['normalized_mutual_info']:.4f}")

    print("\n Done! Check the results/ folder for all outputs.\n")


if __name__ == "__main__":
    main()
