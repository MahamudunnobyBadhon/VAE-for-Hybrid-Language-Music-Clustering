"""
Report-v2 evaluation script.

Purpose:
- Re-evaluate all saved main-task models with stronger clustering search.
- Produce reproducible CSVs for a new report version without touching old outputs.

Search space:
- Methods: KMeans, GMM, Agglomerative (ward/average/complete)
- Fixed K per target evaluation:
  - language: K=2
  - genre_all: K=10
  - genre_labeled: K=18 (exclude untagged)

Outputs:
- results/report_v2/all_candidates.csv
- results/report_v2/best_by_model_eval.csv
- results/report_v2/leaderboard.csv
"""

import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from src.vae import BasicVAE, BetaVAE, CVAE, ConvVAE, MultiModalVAE

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
FEATURES_DIR = ROOT / "data" / "features"
MODELS_DIR = ROOT / "results" / "models"
OUT_DIR = ROOT / "results" / "report_v2"


def cluster_purity(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    from collections import Counter

    contingency = {}
    for t, p in zip(labels_true, labels_pred):
        if p not in contingency:
            contingency[p] = Counter()
        contingency[p][t] += 1
    total_correct = sum(c.most_common(1)[0][1] for c in contingency.values())
    return total_correct / len(labels_true)


def compute_metrics(features: np.ndarray, labels_true: np.ndarray, labels_pred: np.ndarray) -> Dict[str, float]:
    return {
        "silhouette_score": float(silhouette_score(features, labels_pred)),
        "calinski_harabasz_index": float(calinski_harabasz_score(features, labels_pred)),
        "davies_bouldin_index": float(davies_bouldin_score(features, labels_pred)),
        "adjusted_rand_index": float(adjusted_rand_score(labels_true, labels_pred)),
        "normalized_mutual_info": float(normalized_mutual_info_score(labels_true, labels_pred)),
        "cluster_purity": float(cluster_purity(labels_true, labels_pred)),
    }


def parse_genre(filename: str) -> str:
    import re

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


def cluster_with_method(features: np.ndarray, method: str, k: int, seed: int = 42) -> np.ndarray:
    if method.startswith("Agglomerative") and len(features) > 8000:
        raise ValueError("Agglomerative skipped for n>8000 to avoid O(n^2) runtime/memory")

    if method == "KMeans":
        return KMeans(n_clusters=k, random_state=seed, n_init=30, max_iter=500).fit_predict(features)
    if method == "GMM":
        return GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=seed,
            n_init=10,
            max_iter=500,
        ).fit_predict(features)
    if method == "AgglomerativeWard":
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(features)
    if method == "AgglomerativeAverage":
        return AgglomerativeClustering(n_clusters=k, linkage="average").fit_predict(features)
    if method == "AgglomerativeComplete":
        return AgglomerativeClustering(n_clusters=k, linkage="complete").fit_predict(features)
    raise ValueError(f"Unsupported method: {method}")


def extract_latent_basic(model: BasicVAE, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
    model.eval()
    out = []
    xt = torch.FloatTensor(x)
    with torch.no_grad():
        for i in range(0, len(xt), batch_size):
            mu, _ = model.encode(xt[i : i + batch_size])
            out.append(mu.cpu().numpy())
    return np.concatenate(out, axis=0)


def extract_latent_conv(model: ConvVAE, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
    model.eval()
    out = []
    xt = torch.FloatTensor(x)
    with torch.no_grad():
        for i in range(0, len(xt), batch_size):
            mu, _ = model.encode(xt[i : i + batch_size])
            out.append(mu.cpu().numpy())
    return np.concatenate(out, axis=0)


def extract_latent_cvae(model: CVAE, x: np.ndarray, c: np.ndarray, batch_size: int = 512) -> np.ndarray:
    model.eval()
    out = []
    xt = torch.FloatTensor(x)
    ct = torch.FloatTensor(c)
    with torch.no_grad():
        for i in range(0, len(xt), batch_size):
            mu, _ = model.encode(xt[i : i + batch_size], ct[i : i + batch_size])
            out.append(mu.cpu().numpy())
    return np.concatenate(out, axis=0)


def extract_latent_mm(model: MultiModalVAE, audio: np.ndarray, lyrics: np.ndarray, batch_size: int = 512) -> np.ndarray:
    model.eval()
    out = []
    at = torch.FloatTensor(audio)
    lt = torch.FloatTensor(lyrics)
    with torch.no_grad():
        for i in range(0, len(at), batch_size):
            batch = model(at[i : i + batch_size], lt[i : i + batch_size])
            mu = batch["mu"] if isinstance(batch, dict) else batch[1]
            out.append(mu.cpu().numpy())
    return np.concatenate(out, axis=0)


def choose_best(df: pd.DataFrame) -> pd.Series:
    # Prioritize ARI for label alignment; use silhouette as tie-breaker.
    idx = df.sort_values(["adjusted_rand_index", "silhouette_score"], ascending=False).index[0]
    return df.loc[idx]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    mfcc = np.load(FEATURES_DIR / "mfcc_features.npy")
    mel = np.load(FEATURES_DIR / "mel_features.npy")
    combined = np.load(FEATURES_DIR / "combined_features.npy")
    lyrics = np.load(FEATURES_DIR / "lyrics_embeddings" / "lyrics_embeddings.npy")
    meta = pd.read_csv(FEATURES_DIR / "combined_metadata.csv")

    n = min(len(mfcc), len(mel), len(combined), len(lyrics), len(meta))
    mfcc, mel, combined, lyrics, meta = (
        mfcc[:n],
        mel[:n],
        combined[:n],
        lyrics[:n],
        meta.iloc[:n].reset_index(drop=True),
    )

    genre = meta["filename"].astype(str).apply(parse_genre).values
    lang = meta["language"].fillna("unknown").astype(str).values

    le_lang = LabelEncoder()
    lang_y = le_lang.fit_transform(lang)

    le_genre = LabelEncoder()
    genre_y = le_genre.fit_transform(genre)

    labeled_mask = genre != "untagged"

    # Normalization
    mfcc_n = StandardScaler().fit_transform(mfcc)
    mel_n = StandardScaler().fit_transform(mel)
    combined_n = StandardScaler().fit_transform(combined)
    lyrics_n = StandardScaler().fit_transform(lyrics)
    hybrid_n = np.concatenate([mel_n, lyrics_n], axis=1)

    # CVAE conditions: language+genre gives richer conditioning.
    lang_genre = (pd.Series(lang) + "_" + pd.Series(genre)).values
    lg_le = LabelEncoder()
    lg_int = lg_le.fit_transform(lang_genre)
    lg_ohe = OneHotEncoder(sparse_output=False, dtype=np.float32)
    conds_lg = lg_ohe.fit_transform(lg_int.reshape(-1, 1))

    # ---- Load trained checkpoints ----
    ck_easy = torch.load(MODELS_DIR / "basic_vae_easy.pt", map_location="cpu", weights_only=False)
    m_easy = BasicVAE(ck_easy["input_dim"], ck_easy["latent_dim"], ck_easy["hidden_dims"])
    m_easy.load_state_dict(ck_easy["model_state_dict"])

    ck_conv = torch.load(MODELS_DIR / "conv_vae_medium.pt", map_location="cpu", weights_only=False)
    m_conv = ConvVAE(ck_conv["input_dim"], ck_conv["latent_dim"])
    m_conv.load_state_dict(ck_conv["model_state_dict"])

    ck_hybrid = torch.load(MODELS_DIR / "hybrid_vae_medium.pt", map_location="cpu", weights_only=False)
    m_hybrid = BasicVAE(ck_hybrid["input_dim"], ck_hybrid["latent_dim"], ck_hybrid["hidden_dims"])
    m_hybrid.load_state_dict(ck_hybrid["model_state_dict"])

    ck_beta = torch.load(MODELS_DIR / "beta_vae_hard.pt", map_location="cpu", weights_only=False)
    m_beta = BetaVAE(ck_beta["input_dim"], ck_beta["latent_dim"], ck_beta["hidden_dims"])
    m_beta.load_state_dict(ck_beta["model_state_dict"])

    ck_cvae = torch.load(MODELS_DIR / "cvae_hard.pt", map_location="cpu", weights_only=False)
    m_cvae = CVAE(ck_cvae["input_dim"], ck_cvae["latent_dim"], ck_cvae["condition_dim"], ck_cvae["hidden_dims"])
    m_cvae.load_state_dict(ck_cvae["model_state_dict"])

    ck_mm = torch.load(MODELS_DIR / "multimodal_vae_hard.pt", map_location="cpu", weights_only=False)
    m_mm = MultiModalVAE(ck_mm["audio_dim"], ck_mm["lyrics_dim"], ck_mm["latent_dim"], ck_mm["hidden_dims"])
    m_mm.load_state_dict(ck_mm["model_state_dict"])

    # ---- Extract latent features ----
    latents: Dict[str, np.ndarray] = {
        "Easy-BasicVAE": extract_latent_basic(m_easy, mfcc_n),
        "Medium-ConvVAE": extract_latent_conv(m_conv, mel_n),
        "Medium-HybridVAE": extract_latent_basic(m_hybrid, hybrid_n),
        "Hard-BetaVAE": extract_latent_basic(m_beta, combined_n),
        "Hard-CVAE": extract_latent_cvae(m_cvae, combined_n, conds_lg),
        "Hard-MultiModalVAE": extract_latent_mm(m_mm, combined_n, lyrics_n),
        "Baseline-PCA32-mel": PCA(n_components=32, random_state=42).fit_transform(mel_n),
        "Baseline-PCA32-combined": PCA(n_components=32, random_state=42).fit_transform(combined_n),
    }

    eval_specs: List[Tuple[str, int, np.ndarray, np.ndarray]] = [
        ("language", 2, lang_y, np.ones(n, dtype=bool)),
        ("genre_all", 10, genre_y, np.ones(n, dtype=bool)),
        ("genre_labeled", 18, genre_y[labeled_mask], labeled_mask),
    ]

    methods = ["KMeans", "GMM", "AgglomerativeWard", "AgglomerativeComplete"]

    rows = []

    for model_name, z in latents.items():
        print(f"Evaluating: {model_name}")
        for eval_name, k, y_true, mask in eval_specs:
            z_eval = z[mask]
            for method in methods:
                try:
                    labels = cluster_with_method(z_eval, method, k)
                    metrics = compute_metrics(z_eval, y_true, labels)
                    rows.append(
                        {
                            "model": model_name,
                            "eval_type": eval_name,
                            "k": k,
                            "method": method,
                            **metrics,
                        }
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "model": model_name,
                            "eval_type": eval_name,
                            "k": k,
                            "method": method,
                            "method_error": str(exc),
                        }
                    )

    df = pd.DataFrame(rows)
    all_path = OUT_DIR / "all_candidates.csv"
    df.to_csv(all_path, index=False)

    valid = df.dropna(subset=["silhouette_score", "adjusted_rand_index", "normalized_mutual_info", "cluster_purity"])
    best_rows = []
    for (model_name, eval_name), grp in valid.groupby(["model", "eval_type"], sort=False):
        best_rows.append(choose_best(grp))

    best_df = pd.DataFrame(best_rows).reset_index(drop=True)
    best_path = OUT_DIR / "best_by_model_eval.csv"
    best_df.to_csv(best_path, index=False)

    leaderboard = (
        best_df[
            [
                "model",
                "eval_type",
                "k",
                "method",
                "silhouette_score",
                "calinski_harabasz_index",
                "davies_bouldin_index",
                "adjusted_rand_index",
                "normalized_mutual_info",
                "cluster_purity",
            ]
        ]
        .sort_values(["eval_type", "adjusted_rand_index", "silhouette_score"], ascending=[True, False, False])
        .reset_index(drop=True)
    )
    leaderboard_path = OUT_DIR / "leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)

    print("\nSaved:")
    print(f"- {all_path}")
    print(f"- {best_path}")
    print(f"- {leaderboard_path}")


if __name__ == "__main__":
    main()
