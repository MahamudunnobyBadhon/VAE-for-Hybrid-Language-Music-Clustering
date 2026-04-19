"""
HDBSCAN evaluation on pre-trained Hard task latents.

Loads saved model checkpoints (BetaVAE, CVAE, MultiModalVAE), extracts
latent representations, then runs HDBSCAN with hyperparameter search.
Saves results to results/hard/hdbscan_results.csv.

Usage:
    python run_hdbscan.py
"""

import sys
import os
import re
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    LATENT_DIM, HIDDEN_DIMS, BATCH_SIZE, RANDOM_STATE, DEVICE,
    RESULTS_DIR, FEATURES_DIR,
)
from src.dataset import (
    normalize_features, MusicFeatureDataset, MultiModalMusicDataset,
    load_features, load_lyrics_embeddings,
)
from src.vae import BetaVAE, CVAE, MultiModalVAE
from src.train import (
    extract_latent_features, extract_latent_cvae, extract_latent_multimodal,
)
from src.clustering import tune_hdbscan
from src.evaluation import evaluate_clustering


# ============================================================
# Helpers (same as run_hard_task.py)
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


def encode_labels(series: pd.Series) -> np.ndarray:
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    return le.fit_transform(series.fillna("unknown").astype(str))


def build_condition_vectors(metadata: pd.DataFrame) -> tuple:
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    lang = metadata["language"].fillna("unknown").astype(str)
    genre = metadata.get("genre", pd.Series(["misc"] * len(metadata))).fillna("misc").astype(str)
    labels = lang + "_" + genre
    le = LabelEncoder()
    label_ints = le.fit_transform(labels)
    ohe = OneHotEncoder(sparse_output=False, dtype=np.float32)
    conditions = ohe.fit_transform(label_ints.reshape(-1, 1))
    return conditions, conditions.shape[1]


class ConditionedDataset(torch.utils.data.Dataset):
    def __init__(self, features: np.ndarray, conditions: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.conditions = torch.FloatTensor(conditions)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.conditions[idx]

    def get_dataloader(self, batch_size=BATCH_SIZE, shuffle=False):
        return torch.utils.data.DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def load_model_from_checkpoint(ckpt_path: Path, model_cls, **model_kwargs):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = model_cls(**model_kwargs)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ============================================================
# Main
# ============================================================

def main():
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    hard_dir = RESULTS_DIR / "hard"
    hard_dir.mkdir(parents=True, exist_ok=True)
    models_dir = RESULTS_DIR / "models"

    print("\n" + "=" * 65)
    print(" HDBSCAN on Pre-trained Hard Task Latents")
    print("=" * 65)
    print(f" Device: {DEVICE}")

    # ----------------------------------------------------------
    # Step 1: Load features + metadata
    # ----------------------------------------------------------
    print("\n[1] Loading combined audio features + metadata...")
    features, metadata = load_features("combined")
    if "genre" not in metadata.columns:
        metadata["genre"] = metadata["filename"].apply(parse_genre_from_filename)
    print(f"    {features.shape[0]} samples, {features.shape[1]} features")

    ground_truth = encode_labels(metadata["genre"])
    languages = metadata["language"].values
    features_norm, _ = normalize_features(features, method="standard")

    # ----------------------------------------------------------
    # Step 2: Load lyrics embeddings
    # ----------------------------------------------------------
    print("\n[2] Loading lyrics embeddings...")
    lyrics_embeddings = load_lyrics_embeddings()
    if len(lyrics_embeddings) != len(features):
        raise RuntimeError(
            f"Lyrics length mismatch: {len(lyrics_embeddings)} vs {len(features)}"
        )
    print(f"    {lyrics_embeddings.shape}")

    # ----------------------------------------------------------
    # Step 3: Build condition vectors for CVAE
    # ----------------------------------------------------------
    print("\n[3] Building CVAE condition vectors...")
    conditions, condition_dim = build_condition_vectors(metadata)
    print(f"    condition_dim={condition_dim}")

    # ----------------------------------------------------------
    # Step 4: Build data loaders (eval mode, no shuffle)
    # ----------------------------------------------------------
    audio_loader = MusicFeatureDataset(features_norm, metadata).get_dataloader(
        batch_size=BATCH_SIZE, shuffle=False)
    cond_loader = ConditionedDataset(features_norm, conditions).get_dataloader(
        batch_size=BATCH_SIZE, shuffle=False)
    mm_loader = MultiModalMusicDataset(
        features_norm, lyrics_embeddings, metadata, fusion="separate"
    ).get_dataloader(batch_size=BATCH_SIZE, shuffle=False)

    # ----------------------------------------------------------
    # Step 5: Load model checkpoints + extract latents
    # ----------------------------------------------------------
    print("\n[4] Loading model checkpoints...")

    # BetaVAE
    beta_ckpt = models_dir / "beta_vae_hard.pt"
    beta_model = load_model_from_checkpoint(
        beta_ckpt, BetaVAE,
        input_dim=features_norm.shape[1],
        latent_dim=LATENT_DIM,
        hidden_dims=HIDDEN_DIMS,
    )
    print(f"    Loaded BetaVAE from {beta_ckpt.name}")

    # CVAE
    cvae_ckpt = models_dir / "cvae_hard.pt"
    cvae_ckpt_data = torch.load(cvae_ckpt, map_location="cpu")
    saved_condition_dim = cvae_ckpt_data.get("condition_dim", condition_dim)
    cvae_model = CVAE(
        input_dim=features_norm.shape[1],
        latent_dim=LATENT_DIM,
        condition_dim=saved_condition_dim,
        hidden_dims=HIDDEN_DIMS,
    )
    cvae_model.load_state_dict(cvae_ckpt_data["model_state_dict"])
    cvae_model.eval()
    print(f"    Loaded CVAE from {cvae_ckpt.name} (condition_dim={saved_condition_dim})")

    # Rebuild conditions to match saved condition_dim if different
    if saved_condition_dim != condition_dim:
        print(f"    Warning: condition_dim mismatch ({condition_dim} vs saved {saved_condition_dim}). "
              f"Using language-only conditions.")
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        le = LabelEncoder()
        lang_ints = le.fit_transform(metadata["language"].fillna("unknown"))
        ohe = OneHotEncoder(sparse_output=False, dtype=np.float32)
        conditions = ohe.fit_transform(lang_ints.reshape(-1, 1))
        cond_loader = ConditionedDataset(features_norm, conditions).get_dataloader(
            batch_size=BATCH_SIZE, shuffle=False)

    # MultiModalVAE
    mm_ckpt = models_dir / "multimodal_vae_hard.pt"
    mm_model = load_model_from_checkpoint(
        mm_ckpt, MultiModalVAE,
        audio_dim=features_norm.shape[1],
        lyrics_dim=lyrics_embeddings.shape[1],
        latent_dim=LATENT_DIM,
        hidden_dims=HIDDEN_DIMS,
    )
    print(f"    Loaded MultiModalVAE from {mm_ckpt.name}")

    print("\n[5] Extracting latent representations...")
    beta_latent = extract_latent_features(beta_model, audio_loader)
    print(f"    BetaVAE latent: {beta_latent.shape}")
    cvae_latent = extract_latent_cvae(cvae_model, cond_loader)
    print(f"    CVAE latent:    {cvae_latent.shape}")
    mm_latent = extract_latent_multimodal(mm_model, mm_loader)
    print(f"    MMVAE latent:   {mm_latent.shape}")

    # ----------------------------------------------------------
    # Step 6: HDBSCAN tuning on each latent space
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    # UMAP projection: 32d -> 10d before HDBSCAN
    # HDBSCAN suffers from curse of dimensionality in 32d;
    # UMAP preserves local density structure that HDBSCAN needs.
    # ----------------------------------------------------------
    print("\n[6] UMAP projection (32d -> 10d) for HDBSCAN...")
    import umap
    reducer = umap.UMAP(n_components=10, n_neighbors=30, min_dist=0.0,
                        random_state=RANDOM_STATE, n_jobs=1)

    latent_spaces_raw = {
        "BetaVAE": (beta_latent, ground_truth, languages),
        "CVAE":    (cvae_latent, ground_truth, languages),
        "MMVAE":   (mm_latent,   ground_truth, languages),
    }

    latent_spaces = {}
    for model_name, (latent, gt, langs) in latent_spaces_raw.items():
        print(f"    Projecting {model_name} latent with UMAP...")
        projected = reducer.fit_transform(latent)
        latent_spaces[f"{model_name}+UMAP+HDBSCAN"] = (projected, latent, gt, langs)

    all_results = []

    for name, (projected, latent_orig, gt, langs) in latent_spaces.items():
        print(f"\n{'='*50}")
        print(f" HDBSCAN on {name}")
        print(f"{'='*50}")

        tune_result = tune_hdbscan(
            projected,
            min_cluster_sizes=[50, 100, 200, 300, 500],
            min_samples_list=[None, 5, 10],
        )

        best_labels = tune_result["best_labels"]
        n_found = tune_result["n_clusters_found"]
        noise_count = int(np.sum(best_labels == -1))
        noise_pct = noise_count / len(best_labels) * 100

        print(f"\n  Best config: mcs={tune_result['best_min_cluster_size']}, "
              f"ms={tune_result['best_min_samples']}")
        print(f"  Clusters found: {n_found}  |  Noise points: {noise_count} ({noise_pct:.1f}%)")

        # Evaluate on non-noise points using original latent space
        mask = best_labels != -1
        n_valid = int(np.sum(mask))
        print(f"  Evaluating on {n_valid} non-noise samples (original latent space)...")

        metrics = evaluate_clustering(
            features=latent_orig[mask],
            labels_pred=best_labels[mask],
            labels_true=gt[mask],
        )

        # Language purity
        lang_labels = best_labels[mask]
        lang_vals = langs[mask]
        lang_purity_scores = []
        for c in set(lang_labels):
            cluster_langs = lang_vals[lang_labels == c]
            lang_ints = np.array([{"bangla": 0, "english": 1}.get(l, 2)
                                   for l in cluster_langs])
            counts = np.bincount(lang_ints, minlength=3)
            lang_purity_scores.append(counts.max() / len(cluster_langs))
        lang_purity = float(np.mean(lang_purity_scores))

        row = {
            "model": name,
            "n_clusters": n_found,
            "noise_count": noise_count,
            "noise_pct": round(noise_pct, 1),
            "min_cluster_size": tune_result["best_min_cluster_size"],
            "min_samples": str(tune_result["best_min_samples"]),
            "silhouette": round(metrics.get("silhouette_score", 0) or 0, 4),
            "calinski_harabasz": round(metrics.get("calinski_harabasz_index", 0) or 0, 1),
            "davies_bouldin": round(metrics.get("davies_bouldin_index", 0) or 0, 4),
            "ari": round(metrics.get("adjusted_rand_index", 0) or 0, 4),
            "nmi": round(metrics.get("normalized_mutual_info", 0) or 0, 4),
            "purity": round(metrics.get("cluster_purity", 0) or 0, 4),
            "lang_purity": round(lang_purity, 4),
        }
        all_results.append(row)

        print(f"\n  Results for {name}:")
        for k, v in row.items():
            if k != "model":
                print(f"    {k:25s}: {v}")

    # ----------------------------------------------------------
    # Step 7: Save results
    # ----------------------------------------------------------
    results_df = pd.DataFrame(all_results)
    out_path = hard_dir / "hdbscan_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n{'='*65}")
    print(f" Results saved to {out_path}")
    print(f"{'='*65}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
