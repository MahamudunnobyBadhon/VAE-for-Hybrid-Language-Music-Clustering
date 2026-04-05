"""
Fine-Tuning Script: Systematic hyperparameter search for VAE Music Clustering.

Runs a grid over key hyperparameters (beta, latent_dim, epochs, n_clusters, lr)
for all VAE models, evaluates with corrected genre labels, and logs results to CSV.

Usage:
    python run_finetune.py --use-real-audio
    python run_finetune.py --use-real-audio --quick   # reduced grid for faster runs
"""

import argparse
import csv
import re
import time
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import (
    DEVICE, FEATURES_DIR, RESULTS_DIR, MODELS_DIR,
    BATCH_SIZE, N_MELS, HIDDEN_DIMS,
)
from src.vae import BasicVAE, BetaVAE, CVAE, MultiModalVAE, ConvVAE
from src.train import (
    train_vae, extract_latent_features,
    train_cvae, extract_latent_cvae,
    train_multimodal_vae, extract_latent_multimodal,
)
from src.clustering import (
    kmeans_clustering, gmm_clustering, agglomerative_clustering,
    find_optimal_k,
)
from src.evaluation import evaluate_clustering, compare_methods
from src.dataset import (
    MusicFeatureDataset, MultiModalMusicDataset,
    normalize_features, load_features, generate_synthetic_dataset,
)


# ============================================================
# Genre parsing (fixed version)
# ============================================================

def parse_genre_from_filename(filename: str) -> str:
    stem = Path(filename).stem.lower()
    m = re.match(r"^bangla_(\w+)_\d+$", stem)
    if m:
        return m.group(1)
    m = re.match(r"^english_([a-z]+)_\1\.\d+", stem)
    if m:
        return m.group(1)
    m = re.match(r"^([a-z]+)\.\d+$", stem)
    if m:
        return m.group(1)
    if "magna" in stem:
        return "untagged"
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


def build_condition_vectors(metadata, condition_type="language"):
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    if condition_type == "language":
        labels = metadata["language"].fillna("unknown").astype(str)
    else:
        lang = metadata["language"].fillna("unknown").astype(str)
        genre = metadata.get("genre", pd.Series(["misc"] * len(metadata))).fillna("misc").astype(str)
        labels = lang + "_" + genre
    le = LabelEncoder()
    label_ints = le.fit_transform(labels)
    ohe = OneHotEncoder(sparse_output=False, dtype=np.float32)
    conditions = ohe.fit_transform(label_ints.reshape(-1, 1))
    return conditions, conditions.shape[1], dict(zip(le.classes_, range(len(le.classes_))))


class ConditionedDataset(torch.utils.data.Dataset):
    def __init__(self, features, conditions):
        self.features = torch.FloatTensor(features)
        self.conditions = torch.FloatTensor(conditions)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.conditions[idx]

    def get_dataloader(self, batch_size=BATCH_SIZE, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, drop_last=False)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Fine-Tuning: Systematic VAE Hyperparameter Search")
    parser.add_argument("--use-real-audio", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Reduced grid for faster runs")
    parser.add_argument("--skip-multimodal", action="store_true", help="Skip MultiModalVAE")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    finetune_dir = RESULTS_DIR / "finetune"
    finetune_dir.mkdir(parents=True, exist_ok=True)
    results_csv = finetune_dir / "finetune_results.csv"

    # ----------------------------------------------------------
    # Define hyperparameter grid
    # ----------------------------------------------------------
    if args.quick:
        betas = [1.0, 2.0]
        latent_dims = [16, 32]
        epochs_list = [60]
        k_values = [2, 10, 18]
        lrs = [1e-3]
    else:
        betas = [0.5, 1.0, 2.0, 4.0]
        latent_dims = [16, 32, 64]
        epochs_list = [60, 100]
        k_values = [2, 10, 15, 18]
        lrs = [1e-3, 5e-4]

    # ----------------------------------------------------------
    # Load data
    # ----------------------------------------------------------
    print("\n" + "=" * 65)
    print(" FINE-TUNING: Loading data...")
    print("=" * 65)

    if args.use_real_audio:
        try:
            features_mel, metadata = load_features("mel")
        except FileNotFoundError:
            print("Mel features not found. Run build_dataset.py first.")
            return
    else:
        features_mel, metadata = generate_synthetic_dataset(
            n_samples=2000, n_features=2 * N_MELS, n_clusters=10,
        )

    metadata = add_genre_labels(metadata)
    genre_dist = metadata["genre"].value_counts().to_dict()
    print(f"  Dataset: {features_mel.shape[0]} samples, {features_mel.shape[1]} features")
    print(f"  Genres: {genre_dist}")

    # Ground truth (all tracks)
    ground_truth_all = encode_labels(metadata["genre"])
    lang_gt = encode_labels(metadata["language"])

    # Ground truth (labeled only — excluding MagnaTagATune)
    labeled_mask = metadata["genre"] != "untagged"
    n_labeled = labeled_mask.sum()
    n_untagged = (~labeled_mask).sum()
    print(f"  Labeled tracks: {n_labeled}, Untagged (MagnaTagATune): {n_untagged}")
    ground_truth_labeled = encode_labels(metadata.loc[labeled_mask, "genre"])

    features_norm, scaler = normalize_features(features_mel, method="standard")

    # Lyrics embeddings for MultiModalVAE
    lyrics_emb = None
    if not args.skip_multimodal:
        lyrics_path = FEATURES_DIR / "lyrics_embeddings" / "lyrics_embeddings.npy"
        if lyrics_path.exists():
            lyrics_emb = np.load(lyrics_path)
            if len(lyrics_emb) != len(features_mel):
                print(f"  Warning: lyrics shape mismatch ({len(lyrics_emb)} vs {len(features_mel)}), skipping multimodal")
                lyrics_emb = None
        else:
            print("  Lyrics embeddings not found, skipping MultiModalVAE.")

    # ----------------------------------------------------------
    # CSV header
    # ----------------------------------------------------------
    fieldnames = [
        "model", "feature_type", "latent_dim", "beta", "lr", "epochs",
        "n_clusters", "clustering", "eval_set",
        "silhouette_score", "calinski_harabasz_index", "davies_bouldin_index",
        "adjusted_rand_index", "normalized_mutual_info", "cluster_purity",
    ]
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    total_experiments = 0

    def log_result(row):
        nonlocal total_experiments
        total_experiments += 1
        with open(results_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

    def evaluate_and_log(latent, model_name, feature_type, latent_dim, beta, lr,
                         epochs, clustering_methods=None):
        """Evaluate a latent space with multiple K values and clustering methods."""
        if clustering_methods is None:
            clustering_methods = {"KMeans": kmeans_clustering, "GMM": gmm_clustering}

        for k in k_values:
            for clust_name, clust_fn in clustering_methods.items():
                labels = clust_fn(latent, n_clusters=k)

                # Evaluate on ALL tracks
                metrics_all = evaluate_clustering(
                    latent, labels, labels_true=ground_truth_all,
                    method_name=f"{model_name}+{clust_name}")
                log_result({
                    "model": model_name, "feature_type": feature_type,
                    "latent_dim": latent_dim, "beta": beta, "lr": lr,
                    "epochs": epochs, "n_clusters": k, "clustering": clust_name,
                    "eval_set": "all",
                    **{f: metrics_all.get(f) for f in fieldnames[9:]},
                })

                # Evaluate on LABELED tracks only
                metrics_filt = evaluate_clustering(
                    latent[labeled_mask], labels[labeled_mask],
                    labels_true=ground_truth_labeled,
                    method_name=f"{model_name}+{clust_name} [labeled]")
                log_result({
                    "model": model_name, "feature_type": feature_type,
                    "latent_dim": latent_dim, "beta": beta, "lr": lr,
                    "epochs": epochs, "n_clusters": k, "clustering": clust_name,
                    "eval_set": "labeled_only",
                    **{f: metrics_filt.get(f) for f in fieldnames[9:]},
                })

                # K=2 language evaluation (only once per latent space)
                if k == k_values[0]:
                    lang_labels = kmeans_clustering(latent, n_clusters=2)
                    metrics_lang = evaluate_clustering(
                        latent, lang_labels, labels_true=lang_gt,
                        method_name=f"{model_name} [K=2 language]")
                    log_result({
                        "model": model_name, "feature_type": feature_type,
                        "latent_dim": latent_dim, "beta": beta, "lr": lr,
                        "epochs": epochs, "n_clusters": 2, "clustering": "KMeans",
                        "eval_set": "language",
                        **{f: metrics_lang.get(f) for f in fieldnames[9:]},
                    })

    # ----------------------------------------------------------
    # Run experiments
    # ----------------------------------------------------------
    input_dim = features_norm.shape[1]
    batch_size = args.batch_size

    print(f"\n{'='*65}")
    print(f" Starting hyperparameter grid search")
    print(f" Grid: betas={betas}, latent_dims={latent_dims}, "
          f"epochs={epochs_list}, K={k_values}, lr={lrs}")
    print(f"{'='*65}\n")

    # ---- 1. BasicVAE / BetaVAE ----
    for latent_dim, lr, epochs in product(latent_dims, lrs, epochs_list):
        for beta in betas:
            model_name = f"BetaVAE_b{beta}" if beta != 1.0 else "BasicVAE"
            print(f"\n>>> {model_name} | latent={latent_dim}, lr={lr}, "
                  f"epochs={epochs}, beta={beta}")

            dataset = MusicFeatureDataset(features_norm)
            loader = dataset.get_dataloader(batch_size=batch_size)

            model = BetaVAE(input_dim=input_dim, latent_dim=latent_dim,
                            hidden_dims=HIDDEN_DIMS, beta=beta).to(DEVICE)

            t0 = time.time()
            result = train_vae(model, loader, num_epochs=epochs,
                               learning_rate=lr, kl_weight=beta,
                               model_name=f"finetune_{model_name}_ld{latent_dim}")
            elapsed = time.time() - t0
            print(f"    Trained in {elapsed:.0f}s, best loss: {result['best_loss']:.4f}")

            latent = extract_latent_features(result["model"], loader)
            evaluate_and_log(latent, model_name, "mel", latent_dim, beta, lr, epochs)

    # ---- 2. ConvVAE ----
    for latent_dim, lr, epochs in product(latent_dims, lrs, epochs_list):
        model_name = "ConvVAE"
        print(f"\n>>> {model_name} | latent={latent_dim}, lr={lr}, epochs={epochs}")

        dataset = MusicFeatureDataset(features_norm)
        loader = dataset.get_dataloader(batch_size=batch_size)

        model = ConvVAE(input_dim=input_dim, latent_dim=latent_dim).to(DEVICE)

        t0 = time.time()
        result = train_vae(model, loader, num_epochs=epochs,
                           learning_rate=lr, kl_weight=1.0,
                           model_name=f"finetune_convvae_ld{latent_dim}")
        elapsed = time.time() - t0
        print(f"    Trained in {elapsed:.0f}s, best loss: {result['best_loss']:.4f}")

        latent = extract_latent_features(result["model"], loader)
        evaluate_and_log(latent, model_name, "mel", latent_dim, 1.0, lr, epochs)

    # ---- 3. CVAE (conditioned on language) ----
    for latent_dim, lr, epochs in product(latent_dims, lrs, epochs_list):
        model_name = "CVAE"
        print(f"\n>>> {model_name} | latent={latent_dim}, lr={lr}, epochs={epochs}")

        conditions, cond_dim, _ = build_condition_vectors(metadata, "language")
        cond_dataset = ConditionedDataset(features_norm, conditions)
        cond_loader = cond_dataset.get_dataloader(batch_size=batch_size)

        model = CVAE(input_dim=input_dim, latent_dim=latent_dim,
                     condition_dim=cond_dim, hidden_dims=HIDDEN_DIMS).to(DEVICE)

        t0 = time.time()
        result = train_cvae(model, cond_loader, num_epochs=epochs,
                            learning_rate=lr, kl_weight=1.0,
                            model_name=f"finetune_cvae_ld{latent_dim}")
        elapsed = time.time() - t0
        print(f"    Trained in {elapsed:.0f}s, best loss: {result['best_loss']:.4f}")

        latent = extract_latent_cvae(result["model"], cond_loader)
        evaluate_and_log(latent, model_name, "mel", latent_dim, 1.0, lr, epochs)

    # ---- 4. MultiModalVAE ----
    if lyrics_emb is not None and not args.skip_multimodal:
        lyrics_norm = (lyrics_emb - lyrics_emb.mean(0)) / (lyrics_emb.std(0) + 1e-8)

        for latent_dim, lr, epochs in product(latent_dims, lrs, epochs_list):
            model_name = "MultiModalVAE"
            print(f"\n>>> {model_name} | latent={latent_dim}, lr={lr}, epochs={epochs}")

            mm_dataset = MultiModalMusicDataset(features_norm, lyrics_norm)
            mm_loader = DataLoader(mm_dataset, batch_size=batch_size,
                                   shuffle=True, drop_last=False)

            model = MultiModalVAE(
                audio_dim=input_dim, lyrics_dim=lyrics_norm.shape[1],
                latent_dim=latent_dim, hidden_dims=HIDDEN_DIMS,
            ).to(DEVICE)

            t0 = time.time()
            result = train_multimodal_vae(
                model, mm_loader, num_epochs=epochs,
                learning_rate=lr, kl_weight=1.0,
                model_name=f"finetune_mmvae_ld{latent_dim}")
            elapsed = time.time() - t0
            print(f"    Trained in {elapsed:.0f}s, best loss: {result['best_loss']:.4f}")

            latent = extract_latent_multimodal(result["model"], mm_loader)
            evaluate_and_log(latent, model_name, "mel+lyrics", latent_dim, 1.0, lr, epochs)

    # ----------------------------------------------------------
    # Baselines (PCA + KMeans, Raw + KMeans)
    # ----------------------------------------------------------
    print("\n>>> Baselines...")
    from sklearn.decomposition import PCA

    for k in k_values:
        # PCA baseline
        for n_comp in latent_dims:
            pca = PCA(n_components=n_comp, random_state=args.seed)
            pca_feat = pca.fit_transform(features_norm)

            for clust_name, clust_fn in [("KMeans", kmeans_clustering), ("GMM", gmm_clustering)]:
                labels = clust_fn(pca_feat, n_clusters=k)

                metrics = evaluate_clustering(
                    pca_feat, labels, labels_true=ground_truth_labeled,
                    method_name=f"PCA({n_comp})+{clust_name}")
                log_result({
                    "model": f"PCA({n_comp})", "feature_type": "mel",
                    "latent_dim": n_comp, "beta": "N/A", "lr": "N/A",
                    "epochs": "N/A", "n_clusters": k, "clustering": clust_name,
                    "eval_set": "labeled_only",
                    **{f: metrics.get(f) for f in fieldnames[9:]},
                })

        # Raw features baseline
        labels = kmeans_clustering(features_norm, n_clusters=k)
        metrics = evaluate_clustering(
            features_norm, labels, labels_true=ground_truth_labeled,
            method_name="Raw+KMeans")
        log_result({
            "model": "Raw", "feature_type": "mel",
            "latent_dim": input_dim, "beta": "N/A", "lr": "N/A",
            "epochs": "N/A", "n_clusters": k, "clustering": "KMeans",
            "eval_set": "labeled_only",
            **{f: metrics.get(f) for f in fieldnames[9:]},
        })

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print(f"\n{'='*65}")
    print(f" FINE-TUNING COMPLETE!")
    print(f" Total experiments logged: {total_experiments}")
    print(f" Results saved to: {results_csv}")
    print(f"{'='*65}")

    # Print top results
    df = pd.read_csv(results_csv)

    print("\n--- Top 10 by Silhouette (labeled_only) ---")
    top_sil = df[df["eval_set"] == "labeled_only"].nlargest(10, "silhouette_score")
    print(top_sil[["model", "latent_dim", "beta", "n_clusters", "clustering",
                    "silhouette_score", "adjusted_rand_index"]].to_string(index=False))

    print("\n--- Top 10 by ARI (labeled_only) ---")
    top_ari = df[df["eval_set"] == "labeled_only"].nlargest(10, "adjusted_rand_index")
    print(top_ari[["model", "latent_dim", "beta", "n_clusters", "clustering",
                    "silhouette_score", "adjusted_rand_index"]].to_string(index=False))

    print("\n--- Language Separation (K=2) ---")
    lang = df[df["eval_set"] == "language"]
    if not lang.empty:
        print(lang[["model", "latent_dim", "beta",
                     "silhouette_score", "adjusted_rand_index"]].to_string(index=False))

    # Save best configs summary
    summary_path = finetune_dir / "best_configs_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=== Top 5 by Silhouette (labeled_only) ===\n")
        f.write(top_sil.head().to_string(index=False))
        f.write("\n\n=== Top 5 by ARI (labeled_only) ===\n")
        f.write(top_ari.head().to_string(index=False))
        if not lang.empty:
            f.write("\n\n=== Language Separation (K=2) ===\n")
            f.write(lang.to_string(index=False))
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
