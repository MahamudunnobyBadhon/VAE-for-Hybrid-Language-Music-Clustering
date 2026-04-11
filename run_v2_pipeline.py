"""
v2 Full Pipeline: Train → Exhaustive Clustering → Finetune

Fixes applied vs v1:
  1. Features from 3s center window (no clip-length bias)
  2. Language-neutral proxy lyrics (no label leakage)

Steps:
  1. Load v2 features from data/features/v2/
  2. Train all 6 models (Easy BasicVAE, Medium ConvVAE, Medium HybridVAE,
     Hard BetaVAE, Hard CVAE, Hard MultiModalVAE)
  3. Extract latent vectors from each trained checkpoint
  4. Exhaustive clustering search:
       KMeans (n_init=30, max_iter=500)
       GMM full covariance (n_init=10, max_iter=500)
       Agglomerative Ward
       Agglomerative Complete
     at K ∈ {2, 10, 18}
  5. Select best method per model/K by Silhouette Score
  6. Finetune: beta ∈ {1.0, 2.0} × latent_dim ∈ {16, 32} grid search
  7. Save full leaderboard to results/v2/leaderboard_v2.csv

Usage:
    python run_v2_pipeline.py [--skip-train] [--skip-finetune]
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import re
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score
)

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).parent
V2_DIR    = ROOT / "data" / "features" / "v2"
MODEL_DIR = ROOT / "results" / "models" / "v2"
OUT_DIR   = ROOT / "results" / "v2"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from src.vae import BasicVAE, ConvVAE, BetaVAE, CVAE, MultiModalVAE
from src.train import train_vae, train_cvae, train_multimodal_vae
from src.config import DEVICE, LEARNING_RATE, NUM_EPOCHS, KL_WEIGHT, EARLY_STOPPING_PATIENCE

# ── hypers ────────────────────────────────────────────────────────────────────
LATENT_DIM    = 32
BETA          = 1.0
HIDDEN_DIMS   = [512, 256]
EPOCHS_EASY   = 100
EPOCHS_MEDIUM = 80
EPOCHS_HARD   = 60

# ── helpers ───────────────────────────────────────────────────────────────────
def cluster_purity(true, pred):
    from collections import Counter
    return sum(Counter(true[pred == c]).most_common(1)[0][1]
               for c in np.unique(pred)) / len(true)


def evaluate(Z, labels_true, k, method_name):
    if len(np.unique(labels_true)) < 2:
        return None
    ss  = silhouette_score(Z, labels_true, sample_size=min(5000, len(Z)))
    # We only need ss for selection; full metrics computed below
    return ss


def full_metrics(Z, pred, true):
    ss  = silhouette_score(Z, pred, sample_size=min(5000, len(Z)))
    chi = calinski_harabasz_score(Z, pred)
    dbi = davies_bouldin_score(Z, pred)
    ari = adjusted_rand_score(true, pred)
    nmi = normalized_mutual_info_score(true, pred)
    pur = cluster_purity(np.array(true), pred)
    return dict(SS=ss, CHI=chi, DBI=dbi, ARI=ari, NMI=nmi, Purity=pur)


def exhaustive_cluster(Z, k, gt):
    """Try KMeans, GMM(full), Agglom-Ward, Agglom-Complete. Return best by SS."""
    candidates = {}

    # KMeans
    km = KMeans(n_clusters=k, n_init=30, max_iter=500, random_state=42).fit(Z)
    candidates["KMeans"] = km.labels_

    # GMM full covariance
    try:
        gm = GaussianMixture(n_components=k, covariance_type="full",
                              n_init=10, max_iter=500, random_state=42).fit(Z)
        candidates["GMM"] = gm.predict(Z)
    except Exception:
        pass

    # Agglomerative Ward (needs at least k+1 samples)
    if len(Z) > k:
        try:
            ag = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(Z)
            candidates["Agglom-Ward"] = ag.labels_
        except Exception:
            pass
        try:
            ag2 = AgglomerativeClustering(n_clusters=k, linkage="complete").fit(Z)
            candidates["Agglom-Complete"] = ag2.labels_
        except Exception:
            pass

    # Pick best by Silhouette
    best_name, best_pred, best_ss = None, None, -2.0
    for name, pred in candidates.items():
        if len(np.unique(pred)) < 2:
            continue
        ss = silhouette_score(Z, pred, sample_size=min(5000, len(Z)))
        if ss > best_ss:
            best_ss, best_name, best_pred = ss, name, pred

    if best_pred is None:
        best_name, best_pred = "KMeans", candidates["KMeans"]

    m = full_metrics(Z, best_pred, gt)
    m["method"] = best_name
    return m


def parse_genre(filename: str) -> str:
    stem = Path(filename).stem.lower()
    m = re.match(r"^bangla_(\w+)_\d+$", stem)
    if m: return m.group(1)
    m = re.match(r"^english_([a-z]+)_\1\.\d+", stem)
    if m: return m.group(1)
    m = re.match(r"^([a-z]+)\.\d+$", stem)
    if m: return m.group(1)
    if "magna" in stem: return "untagged"
    return "unknown"


# ── data loading ──────────────────────────────────────────────────────────────
def load_v2_data():
    print("\n[LOAD] Loading v2 features...")
    mfcc    = np.load(V2_DIR / "mfcc_features_v2.npy")
    mel     = np.load(V2_DIR / "mel_features_v2.npy")
    comb    = np.load(V2_DIR / "combined_features_v2.npy")
    lyrics  = np.load(V2_DIR / "lyrics_embeddings_v2.npy")
    meta    = pd.read_csv(V2_DIR / "metadata_v2.csv")

    N = min(len(mfcc), len(mel), len(comb), len(lyrics), len(meta))
    mfcc, mel, comb, lyrics, meta = (
        mfcc[:N], mel[:N], comb[:N], lyrics[:N], meta.iloc[:N].reset_index(drop=True))

    # Genre labels
    meta["genre"] = meta["filename"].apply(parse_genre)

    # Ground truth encoders
    le_lang  = LabelEncoder(); lang_enc  = le_lang.fit_transform(meta["language"])
    le_genre = LabelEncoder(); genre_enc = le_genre.fit_transform(meta["genre"])
    labeled  = meta["genre"] != "untagged"

    # Normalize
    mfcc_n   = StandardScaler().fit_transform(mfcc)
    mel_n    = StandardScaler().fit_transform(mel)
    comb_n   = StandardScaler().fit_transform(comb)
    lyrics_n = StandardScaler().fit_transform(lyrics)
    hybrid_n = np.concatenate([mel_n, lyrics_n], axis=1)  # 640-dim

    # CVAE conditions: language+genre one-hot (same as run_hard_task.py v2)
    lang_col  = meta["language"].fillna("unknown").astype(str)
    genre_col = meta["genre"].astype(str)
    lg_labels = lang_col + "_" + genre_col
    le_lg = LabelEncoder(); lg_enc = le_lg.fit_transform(lg_labels)
    ohe_lg = OneHotEncoder(sparse_output=False)
    conds_lg = ohe_lg.fit_transform(lg_enc.reshape(-1, 1)).astype(np.float32)
    condition_dim = conds_lg.shape[1]

    print(f"  N={N}  mfcc:{mfcc_n.shape}  mel:{mel_n.shape}  "
          f"comb:{comb_n.shape}  lyrics:{lyrics_n.shape}")
    print(f"  CVAE condition_dim={condition_dim}  labeled tracks={labeled.sum()}")

    return dict(
        mfcc_n=mfcc_n, mel_n=mel_n, comb_n=comb_n,
        lyrics_n=lyrics_n, hybrid_n=hybrid_n,
        conds_lg=conds_lg, condition_dim=condition_dim,
        lang_enc=lang_enc, genre_enc=genre_enc, labeled=labeled,
        meta=meta, N=N,
    )


# ── DataLoader helpers ────────────────────────────────────────────────────────
from torch.utils.data import DataLoader

class _SimpleDS(torch.utils.data.Dataset):
    def __init__(self, arr): self.t = torch.FloatTensor(arr)
    def __len__(self): return len(self.t)
    def __getitem__(self, i): return self.t[i]

def make_loader(arr, batch=64):
    return DataLoader(_SimpleDS(arr), batch_size=batch, shuffle=True, drop_last=False)

class _XYDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]

def make_loader_xy(x, y, batch=64):
    return DataLoader(_XYDataset(x, y), batch_size=batch, shuffle=True, drop_last=False)

def make_loader_dict(audio, lyrics, batch=64):
    ta, tl = torch.FloatTensor(audio), torch.FloatTensor(lyrics)
    class DictDS(torch.utils.data.Dataset):
        def __len__(self): return len(ta)
        def __getitem__(self, i): return {"audio": ta[i], "lyrics": tl[i]}
    return DataLoader(DictDS(), batch_size=batch, shuffle=True)


# ── latent extraction ─────────────────────────────────────────────────────────
@torch.no_grad()
def extract_latent(model, feats, batch=512):
    model.eval()
    zs = []
    ft = torch.FloatTensor(feats)
    for i in range(0, len(ft), batch):
        mu, _ = model.encode(ft[i:i+batch])
        zs.append(mu.cpu().numpy())
    return np.concatenate(zs)

@torch.no_grad()
def extract_latent_cvae(model, feats, conds, batch=512):
    model.eval()
    zs = []
    ft, cd = torch.FloatTensor(feats), torch.FloatTensor(conds)
    for i in range(0, len(ft), batch):
        mu, _ = model.encode(ft[i:i+batch], cd[i:i+batch])
        zs.append(mu.cpu().numpy())
    return np.concatenate(zs)

@torch.no_grad()
def extract_latent_mm(model, audio, lyrics, batch=512):
    model.eval()
    zs = []
    a, l = torch.FloatTensor(audio), torch.FloatTensor(lyrics)
    for i in range(0, len(a), batch):
        out = model(a[i:i+batch], l[i:i+batch])
        zs.append(out["mu"].cpu().numpy())
    return np.concatenate(zs)


# ── eval at all K ─────────────────────────────────────────────────────────────
def eval_all_k(Z, task, model_name, lang_enc, genre_enc, labeled, results):
    for k, eval_type, gt in [
        (2,  "language",      lang_enc),
        (10, "genre_all",     genre_enc),
        (18, "genre_labeled", genre_enc[labeled]),
    ]:
        Zk = Z if eval_type != "genre_labeled" else Z[labeled]
        m = exhaustive_cluster(Zk, k, gt)
        m.update(task=task, model=model_name, k=k, eval_type=eval_type)
        results.append(m)
        print(f"  {task:<28} K={k:2d} [{eval_type:<14}]  "
              f"SS={m['SS']:.3f}  ARI={m['ARI']:.3f}  "
              f"NMI={m['NMI']:.3f}  method={m['method']}")


# ── TRAINING ──────────────────────────────────────────────────────────────────
def train_all(data, skip=False):
    results = []

    # ── EASY: BasicVAE on MFCC ────────────────────────────────────────────────
    ckpt_path = MODEL_DIR / "basic_vae_easy_v2.pt"
    print(f"\n{'='*60}\n[EASY] BasicVAE on MFCC (40-dim)\n{'='*60}")
    if skip and ckpt_path.exists():
        print("  Skipping training (checkpoint exists)")
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        m  = BasicVAE(ck["input_dim"], ck["latent_dim"], ck["hidden_dims"])
        m.load_state_dict(ck["model_state_dict"])
    else:
        m = BasicVAE(40, LATENT_DIM, HIDDEN_DIMS)
        loader = make_loader(data["mfcc_n"])
        res = train_vae(m, loader, num_epochs=EPOCHS_EASY, kl_weight=BETA,
                        model_name="basic_vae_easy_v2")
        m = res["model"]
        torch.save({"model_state_dict": m.state_dict(), "input_dim": 40,
                    "latent_dim": LATENT_DIM, "hidden_dims": HIDDEN_DIMS},
                   ckpt_path)

    Z = extract_latent(m, data["mfcc_n"])
    eval_all_k(Z, "Easy-BasicVAE", "BasicVAE(MFCC,v2)",
               data["lang_enc"], data["genre_enc"], data["labeled"], results)

    # ── MEDIUM: ConvVAE on Mel ────────────────────────────────────────────────
    ckpt_path = MODEL_DIR / "conv_vae_medium_v2.pt"
    print(f"\n{'='*60}\n[MEDIUM] ConvVAE on Mel (256-dim)\n{'='*60}")
    if skip and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        m  = ConvVAE(ck["input_dim"], ck["latent_dim"])
        m.load_state_dict(ck["model_state_dict"])
    else:
        m = ConvVAE(256, LATENT_DIM)
        loader = make_loader(data["mel_n"])
        res = train_vae(m, loader, num_epochs=EPOCHS_MEDIUM, kl_weight=BETA,
                        model_name="conv_vae_medium_v2")
        m = res["model"]
        torch.save({"model_state_dict": m.state_dict(), "input_dim": 256,
                    "latent_dim": LATENT_DIM, "hidden_dims": HIDDEN_DIMS},
                   ckpt_path)

    Z = extract_latent(m, data["mel_n"])
    eval_all_k(Z, "Medium-ConvVAE", "ConvVAE(Mel,v2)",
               data["lang_enc"], data["genre_enc"], data["labeled"], results)

    # ── MEDIUM: HybridVAE on Mel+Lyrics ───────────────────────────────────────
    ckpt_path = MODEL_DIR / "hybrid_vae_medium_v2.pt"
    print(f"\n{'='*60}\n[MEDIUM] HybridVAE on Mel+Lyrics (640-dim)\n{'='*60}")
    if skip and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        m  = BasicVAE(ck["input_dim"], ck["latent_dim"], ck["hidden_dims"])
        m.load_state_dict(ck["model_state_dict"])
    else:
        m = BasicVAE(640, LATENT_DIM, HIDDEN_DIMS)
        loader = make_loader(data["hybrid_n"])
        res = train_vae(m, loader, num_epochs=EPOCHS_MEDIUM, kl_weight=BETA,
                        model_name="hybrid_vae_medium_v2")
        m = res["model"]
        torch.save({"model_state_dict": m.state_dict(), "input_dim": 640,
                    "latent_dim": LATENT_DIM, "hidden_dims": HIDDEN_DIMS},
                   ckpt_path)

    Z = extract_latent(m, data["hybrid_n"])
    eval_all_k(Z, "Medium-HybridVAE", "HybridVAE(Mel+Lyrics,v2)",
               data["lang_enc"], data["genre_enc"], data["labeled"], results)

    # ── HARD: BetaVAE on Combined ─────────────────────────────────────────────
    ckpt_path = MODEL_DIR / "beta_vae_hard_v2.pt"
    print(f"\n{'='*60}\n[HARD] BetaVAE (b=1) on Combined (90-dim)\n{'='*60}")
    if skip and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        m  = BetaVAE(ck["input_dim"], ck["latent_dim"], ck["hidden_dims"])
        m.load_state_dict(ck["model_state_dict"])
    else:
        m = BetaVAE(data["comb_n"].shape[1], LATENT_DIM, HIDDEN_DIMS, beta=BETA)
        loader = make_loader(data["comb_n"])
        res = train_vae(m, loader, num_epochs=EPOCHS_HARD, kl_weight=BETA,
                        model_name="beta_vae_hard_v2")
        m = res["model"]
        torch.save({"model_state_dict": m.state_dict(),
                    "input_dim": data["comb_n"].shape[1],
                    "latent_dim": LATENT_DIM, "hidden_dims": HIDDEN_DIMS},
                   ckpt_path)

    Z = extract_latent(m, data["comb_n"])
    eval_all_k(Z, "Hard-BetaVAE", "BetaVAE(Combined,v2)",
               data["lang_enc"], data["genre_enc"], data["labeled"], results)

    # ── HARD: CVAE on Combined + lang+genre ───────────────────────────────────
    ckpt_path = MODEL_DIR / "cvae_hard_v2.pt"
    cond_dim  = data["condition_dim"]
    print(f"\n{'='*60}\n[HARD] CVAE on Combined+LangGenre cond (90+{cond_dim})\n{'='*60}")
    if skip and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        m  = CVAE(ck["input_dim"], ck["latent_dim"], ck["condition_dim"], ck["hidden_dims"])
        m.load_state_dict(ck["model_state_dict"])
    else:
        m = CVAE(data["comb_n"].shape[1], LATENT_DIM, cond_dim, HIDDEN_DIMS)
        loader = make_loader_xy(data["comb_n"], data["conds_lg"])
        res = train_cvae(m, loader, num_epochs=EPOCHS_HARD, kl_weight=BETA,
                         model_name="cvae_hard_v2")
        m = res["model"]
        torch.save({"model_state_dict": m.state_dict(),
                    "input_dim": data["comb_n"].shape[1],
                    "latent_dim": LATENT_DIM, "condition_dim": cond_dim,
                    "hidden_dims": HIDDEN_DIMS}, ckpt_path)

    Z = extract_latent_cvae(m, data["comb_n"], data["conds_lg"])
    eval_all_k(Z, "Hard-CVAE", "CVAE(Combined+LangGenre,v2)",
               data["lang_enc"], data["genre_enc"], data["labeled"], results)

    # ── HARD: MultiModalVAE on Combined+Lyrics ────────────────────────────────
    ckpt_path = MODEL_DIR / "multimodal_vae_hard_v2.pt"
    print(f"\n{'='*60}\n[HARD] MultiModalVAE on Combined+Lyrics (90+384)\n{'='*60}")
    if skip and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        m  = MultiModalVAE(ck["audio_dim"], ck["lyrics_dim"],
                           ck["latent_dim"], ck["hidden_dims"])
        m.load_state_dict(ck["model_state_dict"])
    else:
        m = MultiModalVAE(data["comb_n"].shape[1], 384, LATENT_DIM, HIDDEN_DIMS)
        loader = make_loader_dict(data["comb_n"], data["lyrics_n"])
        res = train_multimodal_vae(m, loader, num_epochs=EPOCHS_HARD, kl_weight=BETA,
                                   model_name="multimodal_vae_hard_v2")
        m = res["model"]
        torch.save({"model_state_dict": m.state_dict(),
                    "audio_dim": data["comb_n"].shape[1],
                    "lyrics_dim": 384, "latent_dim": LATENT_DIM,
                    "hidden_dims": HIDDEN_DIMS}, ckpt_path)

    Z = extract_latent_mm(m, data["comb_n"], data["lyrics_n"])
    eval_all_k(Z, "Hard-MultiModalVAE", "MultiModalVAE(Comb+Lyrics,v2)",
               data["lang_enc"], data["genre_enc"], data["labeled"], results)

    # ── PCA baselines ─────────────────────────────────────────────────────────
    from sklearn.decomposition import PCA
    print(f"\n{'='*60}\n[BASELINE] PCA\n{'='*60}")
    for feat_name, feat in [("MFCC", data["mfcc_n"]), ("Combined", data["comb_n"])]:
        Z = PCA(n_components=32).fit_transform(feat)
        eval_all_k(Z, f"Baseline-PCA32-{feat_name.lower()}",
                   f"PCA32({feat_name})",
                   data["lang_enc"], data["genre_enc"], data["labeled"], results)

    return results


# ── FINETUNE ──────────────────────────────────────────────────────────────────
def finetune_v2(data):
    """Grid search: beta × latent_dim × clustering, on v2 features."""
    from src.vae import vae_loss
    print(f"\n{'='*60}\n[FINETUNE] Hyperparameter grid search\n{'='*60}")

    grid_results = []
    betas      = [1.0, 2.0]
    latent_dims = [16, 32]

    configs = [
        ("BasicVAE",      "mfcc",   40,  None, None),
        ("ConvVAE",       "mel",    256, None, None),
        ("HybridVAE",     "hybrid", 640, None, None),
        ("BetaVAE",       "comb",   data["comb_n"].shape[1],  None, None),
        ("CVAE",          "comb",   data["comb_n"].shape[1],  data["conds_lg"], data["condition_dim"]),
        ("MultiModalVAE", "comb",   data["comb_n"].shape[1],  data["lyrics_n"], None),
    ]
    feat_map = {
        "mfcc":   data["mfcc_n"],
        "mel":    data["mel_n"],
        "hybrid": data["hybrid_n"],
        "comb":   data["comb_n"],
    }

    for beta in betas:
        for ld in latent_dims:
            for (mname, feat_key, idim, aux, aux_dim) in configs:
                feat = feat_map[feat_key]
                ckpt_path = MODEL_DIR / f"finetune_{mname}_b{beta}_ld{ld}_v2.pt"

                # Build and train model
                try:
                    if mname == "BasicVAE" or mname == "HybridVAE":
                        m = BasicVAE(idim, ld, HIDDEN_DIMS)
                        loader = make_loader(feat)
                        res = train_vae(m, loader, num_epochs=60, kl_weight=beta,
                                        model_name=f"ft_{mname}_b{beta}_ld{ld}_v2",
                                        kl_annealing=True)
                    elif mname == "ConvVAE":
                        m = ConvVAE(idim, ld)
                        loader = make_loader(feat)
                        res = train_vae(m, loader, num_epochs=60, kl_weight=beta,
                                        model_name=f"ft_{mname}_b{beta}_ld{ld}_v2")
                    elif mname == "BetaVAE":
                        m = BetaVAE(idim, ld, HIDDEN_DIMS, beta=beta)
                        loader = make_loader(feat)
                        res = train_vae(m, loader, num_epochs=60, kl_weight=beta,
                                        model_name=f"ft_{mname}_b{beta}_ld{ld}_v2")
                    elif mname == "CVAE":
                        m = CVAE(idim, ld, aux_dim, HIDDEN_DIMS)
                        loader = make_loader_xy(feat, aux)
                        res = train_cvae(m, loader, num_epochs=60, kl_weight=beta,
                                         model_name=f"ft_{mname}_b{beta}_ld{ld}_v2")
                    elif mname == "MultiModalVAE":
                        m = MultiModalVAE(idim, 384, ld, HIDDEN_DIMS)
                        loader = make_loader_dict(feat, data["lyrics_n"])
                        res = train_multimodal_vae(m, loader, num_epochs=60, kl_weight=beta,
                                                    model_name=f"ft_{mname}_b{beta}_ld{ld}_v2")
                    m = res["model"]
                except Exception as e:
                    print(f"  SKIP {mname} b={beta} ld={ld}: {e}")
                    continue

                # Extract latents and evaluate
                if mname == "CVAE":
                    Z = extract_latent_cvae(m, feat, aux)
                elif mname == "MultiModalVAE":
                    Z = extract_latent_mm(m, feat, data["lyrics_n"])
                else:
                    Z = extract_latent(m, feat)

                for k, eval_type, gt in [
                    (2,  "language",      data["lang_enc"]),
                    (10, "genre_all",     data["genre_enc"]),
                    (18, "genre_labeled", data["genre_enc"][data["labeled"]]),
                ]:
                    Zk = Z if eval_type != "genre_labeled" else Z[data["labeled"]]
                    try:
                        em = exhaustive_cluster(Zk, k, gt)
                    except Exception:
                        continue
                    em.update(model=mname, beta=beta, latent_dim=ld,
                              k=k, eval_type=eval_type)
                    grid_results.append(em)
                    print(f"  ft {mname:15s} b={beta} ld={ld} K={k:2d} "
                          f"[{eval_type:<14}]  SS={em['SS']:.3f}  "
                          f"ARI={em['ARI']:.3f}  method={em['method']}")

    return grid_results


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train",    action="store_true",
                        help="Skip training; load existing v2 checkpoints")
    parser.add_argument("--skip-finetune", action="store_true",
                        help="Skip finetune grid search")
    args = parser.parse_args()

    data = load_v2_data()

    # ── Training + exhaustive clustering ──────────────────────────────────────
    results = train_all(data, skip=args.skip_train)
    df_main = pd.DataFrame(results)
    df_main.to_csv(OUT_DIR / "leaderboard_v2.csv", index=False, float_format="%.4f")
    print(f"\nMain leaderboard saved -> {OUT_DIR / 'leaderboard_v2.csv'}")

    # ── Finetune ──────────────────────────────────────────────────────────────
    if not args.skip_finetune:
        ft_results = finetune_v2(data)
        df_ft = pd.DataFrame(ft_results)
        df_ft.to_csv(OUT_DIR / "finetune_v2.csv", index=False, float_format="%.4f")
        print(f"Finetune results saved -> {OUT_DIR / 'finetune_v2.csv'}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*110)
    print(f"{'Task':<30} {'K':>3}  {'EvalType':<16}  {'SS':>6}  {'CHI':>9}  "
          f"{'ARI':>6}  {'NMI':>6}  {'Method'}")
    print("="*110)
    for _, r in df_main.sort_values(["task","k"]).iterrows():
        print(f"{r['task']:<30} {r['k']:>3}  {r['eval_type']:<16}  "
              f"{r['SS']:>6.3f}  {r['CHI']:>9.0f}  "
              f"{r['ARI']:>6.3f}  {r['NMI']:>6.3f}  {r['method']}")


if __name__ == "__main__":
    main()
