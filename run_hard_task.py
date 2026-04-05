"""
==========================================================
 HARD TASK: Beta-VAE + CVAE + MultiModal + Full Comparison
==========================================================

This script runs the complete Hard task pipeline:
  1. Load combined audio features + lyrics embeddings + metadata
  2. Build condition vectors (language + genre one-hot) for CVAE
  3. Train BetaVAE (beta=4) on audio features
  4. Train CVAE conditioned on language + genre
  5. Train MultiModalVAE on audio + lyrics
  6. Train Autoencoder baseline
  7. K-Means clustering on all latent spaces
  8. Direct spectral feature K-Means baseline
  9. Evaluate ALL methods: Silhouette, NMI, ARI, Purity, DB, CH
 10. Latent traversal visualizations (BetaVAE, dims 0–5)
 11. Reconstruction examples for each model type
 12. Full comparison table -> results/hard/

Usage:
    python run_hard_task.py                        # synthetic fallback
    python run_hard_task.py --use-real-audio       # real audio + lyrics
    python run_hard_task.py --beta 4.0 --n-clusters 10 --epochs 60
    python run_hard_task.py --skip-multimodal      # skip MultiModalVAE (faster)

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
    LATENT_DIM, HIDDEN_DIMS, BATCH_SIZE, NUM_EPOCHS,
    LEARNING_RATE, N_CLUSTERS, RANDOM_STATE, DEVICE,
    RESULTS_DIR, AUDIO_ENGLISH_DIR, AUDIO_BANGLA_DIR,
    N_MELS, FEATURES_DIR,
)
from src.dataset import (
    generate_synthetic_dataset, normalize_features,
    MusicFeatureDataset, MultiModalMusicDataset,
    extract_features_from_directory, save_features, load_features,
    load_lyrics_embeddings, create_hybrid_features,
)
from src.vae import BasicVAE, BetaVAE, CVAE, MultiModalVAE
from src.train import (
    train_vae, extract_latent_features,
    train_cvae, extract_latent_cvae,
    train_multimodal_vae, extract_latent_multimodal,
)
from src.clustering import kmeans_clustering, gmm_clustering, pca_kmeans_baseline, find_optimal_k
from src.baselines import (
    Autoencoder, train_autoencoder, extract_ae_latent,
    spectral_clustering_baseline, direct_feature_kmeans,
)
from src.evaluation import evaluate_clustering, compare_methods
from src.visualization import (
    plot_tsne, plot_umap, plot_cluster_distribution,
    plot_training_curves, plot_comparison_table,
    plot_latent_space_by_language,
    plot_latent_traversal, plot_reconstruction_examples,
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
    stem = Path(filename).stem.lower()
    # Bangla: bangla_<genre>_<number>
    m = re.match(r"^bangla_(\w+)_\d+$", stem)
    if m:
        return m.group(1)
    # GTZAN doubled: english_<genre>_<genre>.<number>
    m = re.match(r"^english_([a-z]+)_\1\.\d+", stem)
    if m:
        return m.group(1)
    # GTZAN plain: <genre>.<number>
    m = re.match(r"^([a-z]+)\.\d+$", stem)
    if m:
        return m.group(1)
    # MagnaTagATune: english_magna_<number>
    if "magna" in stem:
        return "untagged"
    # Fallback
    return "unknown"


def add_genre_labels(metadata: pd.DataFrame) -> pd.DataFrame:
    metadata = metadata.copy()
    if "genre" not in metadata.columns:
        if "filename" in metadata.columns:
            metadata["genre"] = metadata["filename"].apply(parse_genre_from_filename)
        else:
            metadata["genre"] = "unknown"
    return metadata


def encode_labels(series: pd.Series) -> np.ndarray:
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    return le.fit_transform(series.fillna("unknown").astype(str))


def build_condition_vectors(metadata: pd.DataFrame,
                             condition_type: str = "language") -> tuple:
    """
    Build one-hot condition vectors for CVAE.

    Args:
        metadata: DataFrame with 'language' and optionally 'genre'
        condition_type: "language" (2-class) or "language_genre" (combined)

    Returns:
        conditions: np.ndarray (n_samples, condition_dim) float32
        condition_dim: int
        label_to_idx: dict
    """
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder

    if condition_type == "language":
        labels = metadata["language"].fillna("unknown").astype(str)
    else:  # language_genre
        lang = metadata["language"].fillna("unknown").astype(str)
        genre = metadata.get("genre", pd.Series(["misc"] * len(metadata))).fillna("misc").astype(str)
        labels = lang + "_" + genre

    le = LabelEncoder()
    label_ints = le.fit_transform(labels)
    ohe = OneHotEncoder(sparse_output=False, dtype=np.float32)
    conditions = ohe.fit_transform(label_ints.reshape(-1, 1))
    label_to_idx = dict(zip(le.classes_, range(len(le.classes_))))
    return conditions, conditions.shape[1], label_to_idx


class ConditionedDataset(torch.utils.data.Dataset):
    """Dataset that returns (features, condition) tuples for CVAE training."""

    def __init__(self, features: np.ndarray, conditions: np.ndarray):
        assert len(features) == len(conditions)
        self.features = torch.FloatTensor(features)
        self.conditions = torch.FloatTensor(conditions)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.conditions[idx]

    def get_dataloader(self, batch_size: int = BATCH_SIZE, shuffle: bool = True):
        return torch.utils.data.DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def safe_evaluate(features, labels, ground_truth, method_name):
    """Evaluate clustering, filtering DBSCAN noise if present."""
    mask = labels != -1
    if not np.all(mask):
        n_noise = np.sum(~mask)
        if np.sum(mask) < 10 or len(set(labels[mask])) < 2:
            print(f"  Warning: {method_name} - too many noise points ({n_noise})")
            return {
                "method": method_name,
                **{k: float("nan") for k in [
                    "silhouette_score", "calinski_harabasz_index",
                    "davies_bouldin_index", "adjusted_rand_index",
                    "normalized_mutual_info", "cluster_purity"
                ]},
            }
        features = features[mask]
        ground_truth = ground_truth[mask] if ground_truth is not None else None
        labels = labels[mask]
    return evaluate_clustering(features, labels, labels_true=ground_truth,
                               method_name=method_name)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Hard Task: Beta-VAE + CVAE + MultiModal")
    parser.add_argument("--use-real-audio", action="store_true")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--n-clusters", type=int, default=N_CLUSTERS)
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--beta", type=float, default=4.0,
                        help="Beta value for BetaVAE (default: 4.0)")
    parser.add_argument("--skip-umap", action="store_true")
    parser.add_argument("--skip-multimodal", action="store_true",
                        help="Skip MultiModalVAE (faster run)")
    parser.add_argument("--genius-csv", type=str, default=None)
    args = parser.parse_args()

    set_seed()

    hard_dir = RESULTS_DIR / "hard"
    hard_dir.mkdir(parents=True, exist_ok=True)

    n_clusters = args.n_clusters
    latent_dim = args.latent_dim

    print("\n" + "=" * 65)
    print(" CSE715 - Hard Task: Beta-VAE + CVAE + MultiModal + Full Comparison")
    print("=" * 65)
    print(f" Device: {DEVICE}  |  Latent dim: {latent_dim}  |  K: {n_clusters}")
    print(f" Epochs: {args.epochs}  |  Beta: {args.beta}  |  LR: {args.lr}")
    print("=" * 65)

    # ===========================================================
    # STEP 1: Load Combined Audio Features + Metadata
    # ===========================================================
    print("\n[STEP 1] Loading combined audio features + metadata...")

    if args.use_real_audio:
        audio_dirs = {"english": AUDIO_ENGLISH_DIR, "bangla": AUDIO_BANGLA_DIR}
        features, metadata = extract_features_from_directory(
            audio_dirs, feature_type="combined")
        save_features(features, metadata, "combined")
    else:
        try:
            features, metadata = load_features("combined")
        except FileNotFoundError:
            print("  Combined features not found - using mel features or synthetic.")
            try:
                features, metadata = load_features("mel")
            except FileNotFoundError:
                print("  Falling back to synthetic dataset.")
                features, metadata = generate_synthetic_dataset(
                    n_samples=args.n_samples,
                    n_features=2 * N_MELS,
                    n_clusters=n_clusters,
                )

    metadata = add_genre_labels(metadata)
    print(f"  Dataset: {features.shape[0]} samples, {features.shape[1]} features")
    print(f"  Languages: {metadata['language'].value_counts().to_dict()}")
    print(f"  Genres: {metadata['genre'].value_counts().to_dict()}")

    ground_truth = encode_labels(metadata["genre"])
    languages = metadata["language"].values
    features_norm, _ = normalize_features(features, method="standard")

    # ===========================================================
    # STEP 2: Load / Generate Lyrics Embeddings
    # ===========================================================
    print("\n[STEP 2] Loading lyrics embeddings...")
    lyrics_emb_path = FEATURES_DIR / "lyrics_embeddings" / "lyrics_embeddings.npy"

    if lyrics_emb_path.exists():
        lyrics_embeddings = load_lyrics_embeddings()
    else:
        print("  Generating proxy lyrics embeddings...")
        lyrics_embeddings = extract_and_save_lyrics_embeddings(
            metadata,
            genius_csv_path=args.genius_csv,
            model_name="paraphrase-multilingual-MiniLM-L12-v2",
        )

    if len(lyrics_embeddings) != len(features):
        lyrics_embeddings = extract_and_save_lyrics_embeddings(
            metadata,
            genius_csv_path=args.genius_csv,
            model_name="paraphrase-multilingual-MiniLM-L12-v2",
        )
    print(f"  Lyrics embeddings: {lyrics_embeddings.shape}")

    # ===========================================================
    # STEP 3: Build Condition Vectors for CVAE
    # ===========================================================
    print("\n[STEP 3] Building CVAE condition vectors (language only)...")
    conditions, condition_dim, label_to_idx = build_condition_vectors(
        metadata, condition_type="language")
    print(f"  Condition dim: {condition_dim} | Classes: {label_to_idx}")

    # Datasets
    audio_dataset = MusicFeatureDataset(features_norm, metadata)
    audio_loader = audio_dataset.get_dataloader(batch_size=args.batch_size, shuffle=True)
    audio_loader_eval = audio_dataset.get_dataloader(batch_size=args.batch_size, shuffle=False)

    cond_dataset = ConditionedDataset(features_norm, conditions)
    cond_loader = cond_dataset.get_dataloader(batch_size=args.batch_size, shuffle=True)
    cond_loader_eval = cond_dataset.get_dataloader(batch_size=args.batch_size, shuffle=False)

    hybrid_features = create_hybrid_features(features, lyrics_embeddings)
    mm_dataset = MultiModalMusicDataset(features_norm, lyrics_embeddings,
                                        metadata, fusion="separate")
    mm_loader = mm_dataset.get_dataloader(batch_size=args.batch_size, shuffle=True)
    mm_loader_eval = mm_dataset.get_dataloader(batch_size=args.batch_size, shuffle=False)

    # ===========================================================
    # STEP 4: Train BetaVAE
    # ===========================================================
    print(f"\n[STEP 4] Training BetaVAE (beta={args.beta})...")
    beta_vae = BetaVAE(
        input_dim=features_norm.shape[1],
        latent_dim=latent_dim,
        hidden_dims=HIDDEN_DIMS,
        beta=args.beta,
    )
    print(f"  BetaVAE parameters: {sum(p.numel() for p in beta_vae.parameters()):,}")

    beta_result = train_vae(
        model=beta_vae,
        train_loader=audio_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        kl_weight=args.beta,
        kl_annealing=True,
        model_name="beta_vae_hard",
    )
    plot_training_curves(beta_result["history"],
                         save_path=str(hard_dir / "beta_vae_training.png"))

    # ===========================================================
    # STEP 5: Train CVAE
    # ===========================================================
    print(f"\n[STEP 5] Training CVAE (condition_dim={condition_dim})...")
    cvae_model = CVAE(
        input_dim=features_norm.shape[1],
        latent_dim=latent_dim,
        condition_dim=condition_dim,
        hidden_dims=HIDDEN_DIMS,
    )
    print(f"  CVAE parameters: {sum(p.numel() for p in cvae_model.parameters()):,}")

    cvae_result = train_cvae(
        model=cvae_model,
        train_loader=cond_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        kl_weight=1.0,
        kl_annealing=True,
        model_name="cvae_hard",
    )
    plot_training_curves(cvae_result["history"],
                         save_path=str(hard_dir / "cvae_training.png"))

    # ===========================================================
    # STEP 6: Train MultiModalVAE (optional)
    # ===========================================================
    mm_latent = None
    if not args.skip_multimodal:
        print(f"\n[STEP 6] Training MultiModalVAE (audio + lyrics)...")
        mm_vae = MultiModalVAE(
            audio_dim=features_norm.shape[1],
            lyrics_dim=lyrics_embeddings.shape[1],
            latent_dim=latent_dim,
            hidden_dims=HIDDEN_DIMS,
        )
        print(f"  MultiModalVAE parameters: {sum(p.numel() for p in mm_vae.parameters()):,}")

        mm_result = train_multimodal_vae(
            model=mm_vae,
            train_loader=mm_loader,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            kl_weight=1.0,
            kl_annealing=True,
            model_name="multimodal_vae_hard",
        )
        plot_training_curves(mm_result["history"],
                             save_path=str(hard_dir / "multimodal_vae_training.png"))
        mm_latent = extract_latent_multimodal(mm_result["model"], mm_loader_eval)
        print(f"  MultiModalVAE latent: {mm_latent.shape}")
    else:
        print("\n[STEP 6] Skipping MultiModalVAE (--skip-multimodal)")

    # ===========================================================
    # STEP 7: Train Autoencoder Baseline
    # ===========================================================
    print("\n[STEP 7] Training Autoencoder baseline...")
    ae_model = Autoencoder(
        input_dim=features_norm.shape[1],
        latent_dim=latent_dim,
        hidden_dims=HIDDEN_DIMS,
    )
    ae_result = train_autoencoder(
        model=ae_model,
        train_loader=audio_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        model_name="autoencoder_hard",
    )

    # ===========================================================
    # STEP 8: Extract All Latent Features
    # ===========================================================
    print("\n[STEP 8] Extracting latent features...")
    beta_latent = extract_latent_features(beta_result["model"], audio_loader_eval)
    cvae_latent = extract_latent_cvae(cvae_result["model"], cond_loader_eval)
    ae_latent = extract_ae_latent(ae_result["model"], audio_loader_eval)

    print(f"  BetaVAE latent:      {beta_latent.shape}")
    print(f"  CVAE latent:         {cvae_latent.shape}")
    if mm_latent is not None:
        print(f"  MultiModalVAE latent: {mm_latent.shape}")
    print(f"  Autoencoder latent:  {ae_latent.shape}")

    # ===========================================================
    # STEP 9: Clustering
    # ===========================================================
    print("\n[STEP 9] Clustering all latent spaces...")

    beta_labels = kmeans_clustering(beta_latent, n_clusters=n_clusters)
    cvae_labels = kmeans_clustering(cvae_latent, n_clusters=n_clusters)
    ae_labels = kmeans_clustering(ae_latent, n_clusters=n_clusters)
    pca_labels, pca_features = pca_kmeans_baseline(
        features_norm, n_components=latent_dim, n_clusters=n_clusters)
    direct_labels = direct_feature_kmeans(features_norm, n_clusters=n_clusters)

    mm_labels = None
    if mm_latent is not None:
        mm_labels = kmeans_clustering(mm_latent, n_clusters=n_clusters)

    # Optimal K analysis on best latent space
    print("\n[STEP 9b] Optimal K analysis...")
    best_latent_for_k = mm_latent if mm_latent is not None else cvae_latent
    optimal_k_result = find_optimal_k(best_latent_for_k, k_range=range(2, 25))
    print(f"  Optimal K by silhouette: {optimal_k_result['best_k']}")

    # ===========================================================
    # STEP 10: Evaluate All Methods
    # ===========================================================
    print("\n[STEP 10] Evaluating all methods (full metric suite)...")

    all_results = []

    all_results.append(evaluate_clustering(
        beta_latent, beta_labels, labels_true=ground_truth,
        method_name=f"BetaVAE (beta={args.beta}) + K-Means"))

    all_results.append(evaluate_clustering(
        cvae_latent, cvae_labels, labels_true=ground_truth,
        method_name="CVAE + K-Means"))

    if mm_latent is not None and mm_labels is not None:
        all_results.append(evaluate_clustering(
            mm_latent, mm_labels, labels_true=ground_truth,
            method_name="MultiModalVAE (audio+lyrics) + K-Means"))

    all_results.append(evaluate_clustering(
        ae_latent, ae_labels, labels_true=ground_truth,
        method_name="Autoencoder + K-Means (baseline)"))

    all_results.append(evaluate_clustering(
        pca_features, pca_labels, labels_true=ground_truth,
        method_name="PCA + K-Means (baseline)"))

    all_results.append(evaluate_clustering(
        features_norm, direct_labels, labels_true=ground_truth,
        method_name="Raw Features + K-Means (baseline)"))

    # GMM clustering on VAE latent spaces
    for name, lat in [
        (f"BetaVAE (beta={args.beta}) + GMM", beta_latent),
        ("CVAE + GMM", cvae_latent),
    ]:
        gmm_labels = gmm_clustering(lat, n_clusters=n_clusters)
        all_results.append(evaluate_clustering(
            lat, gmm_labels, labels_true=ground_truth, method_name=name))

    if mm_latent is not None:
        gmm_mm = gmm_clustering(mm_latent, n_clusters=n_clusters)
        all_results.append(evaluate_clustering(
            mm_latent, gmm_mm, labels_true=ground_truth,
            method_name="MultiModalVAE (audio+lyrics) + GMM"))

    # Comparison table
    comparison_df = compare_methods(all_results)
    metrics_path = hard_dir / "all_methods_comparison.csv"
    comparison_df.to_csv(metrics_path)
    print(f"  Saved full comparison to {metrics_path}")

    # ------------------------------------------------------------------
    # STEP 10b: Filtered evaluation (exclude untagged MagnaTagATune tracks)
    # ------------------------------------------------------------------
    labeled_mask = metadata["genre"] != "untagged"
    n_labeled = labeled_mask.sum()
    print(f"\n[STEP 10b] Filtered evaluation on {n_labeled} labeled tracks "
          f"(excluding {(~labeled_mask).sum()} untagged MagnaTagATune)...")

    filtered_gt = encode_labels(metadata.loc[labeled_mask, "genre"])
    filtered_results = []

    latent_map = {
        f"BetaVAE (beta={args.beta}) + K-Means": (beta_latent, beta_labels),
        "CVAE + K-Means": (cvae_latent, cvae_labels),
        "Autoencoder + K-Means (baseline)": (ae_latent, ae_labels),
        "PCA + K-Means (baseline)": (pca_features, pca_labels),
        "Raw Features + K-Means (baseline)": (features_norm, direct_labels),
    }
    if mm_latent is not None and mm_labels is not None:
        latent_map["MultiModalVAE (audio+lyrics) + K-Means"] = (mm_latent, mm_labels)

    for name, (lat, lab) in latent_map.items():
        filtered_results.append(evaluate_clustering(
            lat[labeled_mask], lab[labeled_mask], labels_true=filtered_gt,
            method_name=name + " [labeled only]"))

    filtered_df = compare_methods(filtered_results)
    filtered_path = hard_dir / "filtered_methods_comparison.csv"
    filtered_df.to_csv(filtered_path)
    print(f"  Saved filtered comparison to {filtered_path}")

    # ------------------------------------------------------------------
    # STEP 10c: Language-level clustering (K=2 sanity check)
    # ------------------------------------------------------------------
    print("\n[STEP 10c] Language clustering sanity check (K=2)...")
    lang_gt = encode_labels(metadata["language"])
    lang_results = []
    for name, (lat, _) in latent_map.items():
        lang_labels_k2 = kmeans_clustering(lat, n_clusters=2)
        lang_results.append(evaluate_clustering(
            lat, lang_labels_k2, labels_true=lang_gt,
            method_name=name + " [K=2 language]"))

    lang_df = compare_methods(lang_results)
    lang_path = hard_dir / "language_clustering_comparison.csv"
    lang_df.to_csv(lang_path)
    print(f"  Saved language clustering results to {lang_path}")

    # ===========================================================
    # STEP 11: Latent Traversal Visualizations (BetaVAE)
    # ===========================================================
    print("\n[STEP 11] Latent traversal visualizations (BetaVAE)...")
    base_latent = np.mean(beta_latent, axis=0)  # dataset mean as base vector
    for dim in range(min(6, latent_dim)):
        plot_latent_traversal(
            model=beta_result["model"],
            base_latent=base_latent,
            dim=dim,
            n_steps=10,
            value_range=(-3.0, 3.0),
            save_path=str(hard_dir / f"latent_traversal_dim_{dim}.png"),
        )

    # ===========================================================
    # STEP 12: Reconstruction Examples
    # ===========================================================
    print("\n[STEP 12] Reconstruction examples...")

    # BetaVAE reconstructions
    plot_reconstruction_examples(
        model=beta_result["model"],
        features=features_norm,
        n_examples=6,
        save_path=str(hard_dir / "beta_vae_reconstructions.png"),
    )

    # Autoencoder reconstructions
    plot_reconstruction_examples(
        model=ae_result["model"],
        features=features_norm,
        n_examples=6,
        save_path=str(hard_dir / "autoencoder_reconstructions.png"),
    )

    # CVAE reconstructions - use a mean condition vector (mean over dataset)
    mean_condition = torch.FloatTensor(conditions.mean(axis=0)).unsqueeze(0)

    class CVAEWrapper(torch.nn.Module):
        """Wraps CVAE with a fixed mean condition for reconstruction plotting."""
        def __init__(self, cvae, cond):
            super().__init__()
            self.cvae = cvae
            self.cond = cond

        def forward(self, x):
            c = self.cond.to(x.device).expand(x.size(0), -1)
            return self.cvae(x, c)

        def decode(self, z):
            c = self.cond.to(z.device).expand(z.size(0), -1)
            return self.cvae.decode(z, c)

    cvae_wrapper = CVAEWrapper(cvae_result["model"], mean_condition)
    plot_reconstruction_examples(
        model=cvae_wrapper,
        features=features_norm,
        n_examples=6,
        save_path=str(hard_dir / "cvae_reconstructions.png"),
    )

    # ===========================================================
    # STEP 13: t-SNE Visualizations
    # ===========================================================
    print("\n[STEP 13] t-SNE visualizations...")

    beta_tsne = plot_tsne(beta_latent, beta_labels,
                          title=f"BetaVAE (beta={args.beta}) Latent Space - K-Means",
                          color_field="Cluster",
                          save_path=str(hard_dir / "beta_vae_tsne_clusters.png"))
    plot_latent_space_by_language(beta_tsne, languages,
                                  title="BetaVAE Latent Space by Language",
                                  save_path=str(hard_dir / "beta_vae_tsne_language.png"))

    cvae_tsne = plot_tsne(cvae_latent, cvae_labels,
                          title="CVAE Latent Space - K-Means",
                          color_field="Cluster",
                          save_path=str(hard_dir / "cvae_tsne_clusters.png"))
    plot_latent_space_by_language(cvae_tsne, languages,
                                  title="CVAE Latent Space by Language",
                                  save_path=str(hard_dir / "cvae_tsne_language.png"))

    if not args.skip_umap:
        try:
            plot_umap(beta_latent, beta_labels,
                      title=f"BetaVAE (beta={args.beta}) UMAP",
                      save_path=str(hard_dir / "beta_vae_umap.png"))
        except Exception as e:
            print(f"  UMAP skipped: {e}")

    # Comparison table figure
    plot_comparison_table(comparison_df,
                          save_path=str(hard_dir / "hard_comparison_table.png"))

    # ===========================================================
    # SUMMARY
    # ===========================================================
    print("\n" + "=" * 65)
    print(" HARD TASK COMPLETE!")
    print("=" * 65)
    print(f"\n Results saved to: {hard_dir}")
    print("\n Full Method Comparison (Silhouette | ARI | NMI | Purity):")
    for r in all_results:
        sil = r.get("silhouette_score", float("nan"))
        ari = r.get("adjusted_rand_index") or float("nan")
        nmi = r.get("normalized_mutual_info") or float("nan")
        pur = r.get("cluster_purity") or float("nan")
        print(f"   {r['method']:50s} | {sil:.4f} | {ari:.4f} | {nmi:.4f} | {pur:.4f}")

    print("\n Done! Check the results/hard/ folder for all outputs.\n")


if __name__ == "__main__":
    main()
