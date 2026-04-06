"""
Multi-K evaluation of all saved main-task VAE models.
Evaluates each model at K=2, K=10, K=18 with:
  - K=2  -> language ground truth (English vs Bangla)
  - K=10 -> genre ground truth (all tracks)
  - K=18 -> genre ground truth (labeled tracks only, excludes MagnaTagATune)
"""

import sys
import warnings
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score
)

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).parent
DATA_DIR  = ROOT / "data" / "features"
MODEL_DIR = ROOT / "results" / "models"
LYRICS_DIR = DATA_DIR / "lyrics_embeddings"
OUT_CSV   = ROOT / "results" / "multi_k_eval.csv"

sys.path.insert(0, str(ROOT))
from src.vae import BasicVAE, ConvVAE, BetaVAE, CVAE, MultiModalVAE


def cluster_purity(true, pred):
    from collections import Counter
    return sum(Counter(true[pred == c]).most_common(1)[0][1]
               for c in np.unique(pred)) / len(true)


def run_kmeans(Z, k, gt):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Z)
    p  = km.labels_
    ss  = silhouette_score(Z, p, sample_size=min(5000, len(Z)))
    chi = calinski_harabasz_score(Z, p)
    dbi = davies_bouldin_score(Z, p)
    ari = adjusted_rand_score(gt, p)
    nmi = normalized_mutual_info_score(gt, p)
    pur = cluster_purity(gt, p)
    return dict(SS=ss, CHI=chi, DBI=dbi, ARI=ari, NMI=nmi, Purity=pur)


def extract(model, feats, batch=512):
    model.eval()
    zs = []
    ft = torch.FloatTensor(feats)
    with torch.no_grad():
        for i in range(0, len(ft), batch):
            mu, _ = model.encode(ft[i:i+batch])
            zs.append(mu.cpu().numpy())
    return np.concatenate(zs)


def extract_cvae(model, feats, conds, batch=512):
    model.eval()
    zs = []
    ft, cd = torch.FloatTensor(feats), torch.FloatTensor(conds)
    with torch.no_grad():
        for i in range(0, len(ft), batch):
            mu, _ = model.encode(ft[i:i+batch], cd[i:i+batch])
            zs.append(mu.cpu().numpy())
    return np.concatenate(zs)


def extract_mm(model, audio, lyrics, batch=512):
    model.eval()
    zs = []
    a, l = torch.FloatTensor(audio), torch.FloatTensor(lyrics)
    with torch.no_grad():
        for i in range(0, len(a), batch):
            out = model(a[i:i+batch], l[i:i+batch])
            mu  = out["mu"] if isinstance(out, dict) else out[1]
            zs.append(mu.cpu().numpy())
    return np.concatenate(zs)


def main():
    print("Loading features and metadata ...")
    mfcc   = np.load(DATA_DIR / "mfcc_features.npy")
    mel    = np.load(DATA_DIR / "mel_features.npy")
    comb   = np.load(DATA_DIR / "combined_features.npy")
    lyrics = np.load(LYRICS_DIR / "lyrics_embeddings.npy")
    meta   = pd.read_csv(DATA_DIR / "combined_metadata.csv")

    N = min(len(mfcc), len(mel), len(comb), len(lyrics), len(meta))
    mfcc, mel, comb, lyrics, meta = (
        mfcc[:N], mel[:N], comb[:N], lyrics[:N], meta.iloc[:N].reset_index(drop=True))

    # --- ground truth labels ---
    lang_gt  = meta["language"].values                       # 'english' / 'bangla'
    genre_gt = np.array([fn.split("_")[1] if len(fn.split("_")) >= 2 else "unknown"
                         for fn in meta["filename"]])
    genre_gt[genre_gt == "magna"] = "untagged"

    le_lang  = LabelEncoder();  lang_enc  = le_lang.fit_transform(lang_gt)
    le_genre = LabelEncoder();  genre_enc = le_genre.fit_transform(genre_gt)
    labeled  = genre_gt != "untagged"                        # ~11K labeled tracks

    mfcc_n  = StandardScaler().fit_transform(mfcc)
    mel_n   = StandardScaler().fit_transform(mel)
    comb_n  = StandardScaler().fit_transform(comb)
    lyrics_n = StandardScaler().fit_transform(lyrics)
    hybrid_n = np.concatenate([mel_n, lyrics_n], axis=1)    # for HybridVAE

    ohe   = OneHotEncoder(sparse_output=False)
    conds = ohe.fit_transform(lang_enc.reshape(-1, 1)).astype(np.float32)

    results = []

    def record(task, model_name, k, eval_type, m):
        results.append({"Task": task, "Model": model_name, "K": k,
                         "EvalType": eval_type, **m})
        print(f"  {task:<32}K={k:2d} [{eval_type:<14}]  "
              f"SS={m['SS']:.3f}  ARI={m['ARI']:.3f}  NMI={m['NMI']:.3f}  Purity={m['Purity']:.3f}")

    def eval_ks(task, name, Z):
        record(task, name, 2,  "language",      run_kmeans(Z,           2,  lang_enc))
        record(task, name, 10, "genre_all",     run_kmeans(Z,           10, genre_enc))
        record(task, name, 18, "genre_labeled", run_kmeans(Z[labeled],  18, genre_enc[labeled]))

    # ── 1. EASY: BasicVAE (MFCC 40-dim) ─────────────────────────────────────
    print("\n[EASY] BasicVAE on MFCC")
    ck = torch.load(MODEL_DIR / "basic_vae_easy.pt",
                    map_location="cpu", weights_only=False)
    m = BasicVAE(ck["input_dim"], ck["latent_dim"], ck["hidden_dims"])
    m.load_state_dict(ck["model_state_dict"])
    eval_ks("Easy – BasicVAE", "BasicVAE (MFCC, d=40)", extract(m, mfcc_n))

    pca = PCA(n_components=32).fit_transform(mfcc_n)
    eval_ks("Easy – PCA baseline", "PCA (MFCC)+KMeans", pca)

    # ── 2. MEDIUM: ConvVAE (Mel 256-dim) ─────────────────────────────────────
    print("\n[MEDIUM] ConvVAE on Mel")
    ck = torch.load(MODEL_DIR / "conv_vae_medium.pt",
                    map_location="cpu", weights_only=False)
    m = ConvVAE(ck["input_dim"], ck["latent_dim"])
    m.load_state_dict(ck["model_state_dict"])
    eval_ks("Medium – ConvVAE", "ConvVAE (Mel, d=256)", extract(m, mel_n))

    # ── 3. MEDIUM: HybridVAE (Mel+Lyrics 640-dim) ────────────────────────────
    print("\n[MEDIUM] HybridVAE on Mel+Lyrics")
    ck = torch.load(MODEL_DIR / "hybrid_vae_medium.pt",
                    map_location="cpu", weights_only=False)
    m = BasicVAE(ck["input_dim"], ck["latent_dim"], ck["hidden_dims"])
    m.load_state_dict(ck["model_state_dict"])
    eval_ks("Medium – HybridVAE", "HybridVAE (Mel+Lyrics, d=640)", extract(m, hybrid_n))

    # ── 4. HARD: BetaVAE (Combined 90-dim) ───────────────────────────────────
    print("\n[HARD] BetaVAE on Combined")
    ck = torch.load(MODEL_DIR / "beta_vae_hard.pt",
                    map_location="cpu", weights_only=False)
    m = BetaVAE(ck["input_dim"], ck["latent_dim"], ck["hidden_dims"])
    m.load_state_dict(ck["model_state_dict"])
    eval_ks("Hard – BetaVAE (b=4)", "BetaVAE (Combined, d=90)", extract(m, comb_n))

    # ── 5. HARD: CVAE (Combined 90-dim + lang condition) ─────────────────────
    print("\n[HARD] CVAE on Combined + language condition")
    ck = torch.load(MODEL_DIR / "cvae_hard.pt",
                    map_location="cpu", weights_only=False)
    m = CVAE(ck["input_dim"], ck["latent_dim"], ck["condition_dim"], ck["hidden_dims"])
    m.load_state_dict(ck["model_state_dict"])
    eval_ks("Hard – CVAE", "CVAE (Combined+Lang, d=90)", extract_cvae(m, comb_n, conds))

    # ── 6. HARD: MultiModalVAE (Combined 90-dim + Lyrics 384-dim) ────────────
    print("\n[HARD] MultiModalVAE on Combined+Lyrics")
    ck = torch.load(MODEL_DIR / "multimodal_vae_hard.pt",
                    map_location="cpu", weights_only=False)
    m = MultiModalVAE(ck["audio_dim"], ck["lyrics_dim"], ck["latent_dim"], ck["hidden_dims"])
    m.load_state_dict(ck["model_state_dict"])
    eval_ks("Hard – MultiModalVAE", "MultiModalVAE (Comb+Lyrics, d=474)", extract_mm(m, comb_n, lyrics_n))

    # ── Save ──────────────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False, float_format="%.4f")
    print(f"\nSaved -> {OUT_CSV}")

    print("\n" + "="*100)
    print(f"{'Task':<34} {'K':>3}  {'EvalType':<16}  {'SS':>6}  {'CHI':>9}  "
          f"{'ARI':>6}  {'NMI':>6}  {'Purity':>6}")
    print("="*100)
    for _, r in df.iterrows():
        print(f"{r['Task']:<34} {r['K']:>3}  {r['EvalType']:<16}  "
              f"{r['SS']:>6.3f}  {r['CHI']:>9.0f}  "
              f"{r['ARI']:>6.3f}  {r['NMI']:>6.3f}  {r['Purity']:>6.3f}")


if __name__ == "__main__":
    main()
