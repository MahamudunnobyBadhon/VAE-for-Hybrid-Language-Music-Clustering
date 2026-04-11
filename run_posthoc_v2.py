"""
run_posthoc_v2.py  — Parallel post-hoc clustering + fast finetune

Runs exhaustive clustering + limited finetune on whatever v2 checkpoints
exist right now.  Safe to run in parallel with run_v2_pipeline.py.

Usage:
    # Process all available checkpoints
    python run_posthoc_v2.py

    # Process only specific models
    python run_posthoc_v2.py --models BasicVAE ConvVAE HybridVAE

    # Skip finetune (clustering only, fastest)
    python run_posthoc_v2.py --no-finetune

Finetune is limited to beta={1.0,2.0} x latent_dim=32, 30 epochs to stay fast.
Results are merged into results/v2/leaderboard_v2.csv.
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import argparse, warnings, re
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score,
)
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).parent
V2_DIR    = ROOT / "data" / "features" / "v2"
MODEL_DIR = ROOT / "results" / "models" / "v2"
OUT_DIR   = ROOT / "results" / "v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from src.vae import BasicVAE, ConvVAE, BetaVAE, CVAE, MultiModalVAE
from src.train import train_vae, train_cvae, train_multimodal_vae

LATENT_DIM  = 32
HIDDEN_DIMS = [512, 256]

# ── model registry: filename → (type, feat_key) ──────────────────────────────
CHECKPOINT_REGISTRY = {
    "basic_vae_easy_v2.pt":       ("BasicVAE",      "mfcc"),
    "conv_vae_medium_v2.pt":      ("ConvVAE",        "mel"),
    "hybrid_vae_medium_v2.pt":    ("HybridVAE",      "hybrid"),
    "beta_vae_hard_v2.pt":        ("BetaVAE",        "comb"),
    "cvae_hard_v2.pt":            ("CVAE",           "comb"),
    "multimodal_vae_hard_v2.pt":  ("MultiModalVAE",  "comb"),
}

TASK_MAP = {
    "BasicVAE":     "Easy-BasicVAE",
    "ConvVAE":      "Medium-ConvVAE",
    "HybridVAE":    "Medium-HybridVAE",
    "BetaVAE":      "Hard-BetaVAE",
    "CVAE":         "Hard-CVAE",
    "MultiModalVAE":"Hard-MultiModalVAE",
}

# ── helpers ───────────────────────────────────────────────────────────────────
def cluster_purity(true, pred):
    from collections import Counter
    return sum(Counter(true[pred == c]).most_common(1)[0][1]
               for c in np.unique(pred)) / len(true)

def full_metrics(Z, pred, true):
    ss  = silhouette_score(Z, pred, sample_size=min(5000, len(Z)))
    chi = calinski_harabasz_score(Z, pred)
    dbi = davies_bouldin_score(Z, pred)
    ari = adjusted_rand_score(true, pred)
    nmi = normalized_mutual_info_score(true, pred)
    pur = cluster_purity(np.array(true), pred)
    return dict(SS=ss, CHI=chi, DBI=dbi, ARI=ari, NMI=nmi, Purity=pur)

def exhaustive_cluster(Z, k, gt):
    candidates = {}
    km = KMeans(n_clusters=k, n_init=30, max_iter=500, random_state=42).fit(Z)
    candidates["KMeans"] = km.labels_
    try:
        gm = GaussianMixture(n_components=k, covariance_type="full",
                              n_init=10, max_iter=500, random_state=42).fit(Z)
        candidates["GMM"] = gm.predict(Z)
    except Exception:
        pass
    if len(Z) > k:
        for link in ("ward", "complete"):
            try:
                ag = AgglomerativeClustering(n_clusters=k, linkage=link).fit(Z)
                candidates[f"Agglom-{link.title()}"] = ag.labels_
            except Exception:
                pass
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

def parse_genre(filename):
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
def load_data():
    print("[LOAD] Loading v2 features...")
    mfcc   = np.load(V2_DIR / "mfcc_features_v2.npy")
    mel    = np.load(V2_DIR / "mel_features_v2.npy")
    comb   = np.load(V2_DIR / "combined_features_v2.npy")
    lyrics = np.load(V2_DIR / "lyrics_embeddings_v2.npy")
    meta   = pd.read_csv(V2_DIR / "metadata_v2.csv")
    N = min(len(mfcc), len(mel), len(comb), len(lyrics), len(meta))
    mfcc, mel, comb, lyrics, meta = (
        mfcc[:N], mel[:N], comb[:N], lyrics[:N], meta.iloc[:N].reset_index(drop=True))
    meta["genre"] = meta["filename"].apply(parse_genre)
    le_lang  = LabelEncoder(); lang_enc  = le_lang.fit_transform(meta["language"])
    le_genre = LabelEncoder(); genre_enc = le_genre.fit_transform(meta["genre"])
    labeled  = meta["genre"] != "untagged"
    mfcc_n   = StandardScaler().fit_transform(mfcc)
    mel_n    = StandardScaler().fit_transform(mel)
    comb_n   = StandardScaler().fit_transform(comb)
    lyrics_n = StandardScaler().fit_transform(lyrics)
    hybrid_n = np.concatenate([mel_n, lyrics_n], axis=1)
    lang_col  = meta["language"].fillna("unknown").astype(str)
    genre_col = meta["genre"].astype(str)
    lg_labels = lang_col + "_" + genre_col
    le_lg = LabelEncoder(); lg_enc = le_lg.fit_transform(lg_labels)
    ohe_lg = OneHotEncoder(sparse_output=False)
    conds_lg = ohe_lg.fit_transform(lg_enc.reshape(-1, 1)).astype(np.float32)
    condition_dim = conds_lg.shape[1]
    print(f"  N={N}  mfcc:{mfcc_n.shape}  mel:{mel_n.shape}  "
          f"comb:{comb_n.shape}  lyrics:{lyrics_n.shape}  cond_dim={condition_dim}")
    return dict(
        mfcc_n=mfcc_n, mel_n=mel_n, comb_n=comb_n, lyrics_n=lyrics_n,
        hybrid_n=hybrid_n, conds_lg=conds_lg, condition_dim=condition_dim,
        lang_enc=lang_enc, genre_enc=genre_enc, labeled=labeled, meta=meta, N=N,
    )

feat_map_key = None  # set after data loaded

def get_feat(data, key):
    return {"mfcc": data["mfcc_n"], "mel": data["mel_n"],
            "comb": data["comb_n"], "hybrid": data["hybrid_n"]}[key]

# ── DataLoader helpers ────────────────────────────────────────────────────────
class _SimpleDS(torch.utils.data.Dataset):
    def __init__(self, arr): self.t = torch.FloatTensor(arr)
    def __len__(self): return len(self.t)
    def __getitem__(self, i): return self.t[i]

class _XYDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = torch.FloatTensor(x); self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]

class _DictDS(torch.utils.data.Dataset):
    def __init__(self, a, l):
        self.a = torch.FloatTensor(a); self.l = torch.FloatTensor(l)
    def __len__(self): return len(self.a)
    def __getitem__(self, i): return {"audio": self.a[i], "lyrics": self.l[i]}

def make_loader(arr, bs=64):
    return DataLoader(_SimpleDS(arr), batch_size=bs, shuffle=True, drop_last=False)
def make_loader_xy(x, y, bs=64):
    return DataLoader(_XYDataset(x, y), batch_size=bs, shuffle=True, drop_last=False)
def make_loader_dict(a, l, bs=64):
    return DataLoader(_DictDS(a, l), batch_size=bs, shuffle=True)

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

# ── load checkpoint + extract Z ───────────────────────────────────────────────
def load_and_extract(ckpt_file, mtype, data):
    ck = torch.load(MODEL_DIR / ckpt_file, map_location="cpu", weights_only=False)
    if mtype == "BasicVAE":
        m = BasicVAE(ck["input_dim"], ck["latent_dim"], ck["hidden_dims"])
        m.load_state_dict(ck["model_state_dict"])
        return extract_latent(m, data["mfcc_n"])
    elif mtype == "ConvVAE":
        m = ConvVAE(ck["input_dim"], ck["latent_dim"])
        m.load_state_dict(ck["model_state_dict"])
        return extract_latent(m, data["mel_n"])
    elif mtype == "HybridVAE":
        m = BasicVAE(ck["input_dim"], ck["latent_dim"], ck["hidden_dims"])
        m.load_state_dict(ck["model_state_dict"])
        return extract_latent(m, data["hybrid_n"])
    elif mtype == "BetaVAE":
        m = BetaVAE(ck["input_dim"], ck["latent_dim"], ck["hidden_dims"],
                    beta=ck.get("beta", 1.0))
        m.load_state_dict(ck["model_state_dict"])
        return extract_latent(m, data["comb_n"])
    elif mtype == "CVAE":
        m = CVAE(ck["input_dim"], ck["latent_dim"],
                 ck["condition_dim"], ck["hidden_dims"])
        m.load_state_dict(ck["model_state_dict"])
        return extract_latent_cvae(m, data["comb_n"], data["conds_lg"])
    elif mtype == "MultiModalVAE":
        m = MultiModalVAE(ck["audio_dim"], ck["lyrics_dim"],
                          ck["latent_dim"], ck["hidden_dims"])
        m.load_state_dict(ck["model_state_dict"])
        return extract_latent_mm(m, data["comb_n"], data["lyrics_n"])
    else:
        raise ValueError(f"Unknown model type: {mtype}")

# ── eval at all K ─────────────────────────────────────────────────────────────
def eval_all_k(Z, task, model_name, data):
    rows = []
    for k, eval_type, gt in [
        (2,  "language",      data["lang_enc"]),
        (10, "genre_all",     data["genre_enc"]),
        (18, "genre_labeled", data["genre_enc"][data["labeled"]]),
    ]:
        Zk = Z if eval_type != "genre_labeled" else Z[data["labeled"]]
        m = exhaustive_cluster(Zk, k, gt)
        m.update(task=task, model=model_name, k=k, eval_type=eval_type)
        rows.append(m)
        print(f"  {task:<28} K={k:2d} [{eval_type:<14}]  "
              f"SS={m['SS']:.3f}  ARI={m['ARI']:.3f}  "
              f"NMI={m['NMI']:.3f}  method={m['method']}")
    return rows

# ── quick finetune ────────────────────────────────────────────────────────────
def quick_finetune(mtype, data, betas=(1.0, 2.0), latent_dim=32, epochs=30):
    """Fast finetune: 2 beta values, fixed latent_dim, 30 epochs."""
    rows = []
    feat_key = {"BasicVAE": "mfcc", "ConvVAE": "mel", "HybridVAE": "hybrid",
                "BetaVAE": "comb", "CVAE": "comb", "MultiModalVAE": "comb"}[mtype]
    feat = get_feat(data, feat_key)
    idim = feat.shape[1]
    for beta in betas:
        try:
            if mtype in ("BasicVAE", "HybridVAE"):
                m = BasicVAE(idim, latent_dim, HIDDEN_DIMS)
                res = train_vae(m, make_loader(feat), num_epochs=epochs,
                                kl_weight=beta, kl_annealing=True,
                                model_name=f"qft_{mtype}_b{beta}_v2")
                Z = extract_latent(res["model"], feat)
            elif mtype == "ConvVAE":
                m = ConvVAE(idim, latent_dim)
                res = train_vae(m, make_loader(feat), num_epochs=epochs,
                                kl_weight=beta,
                                model_name=f"qft_{mtype}_b{beta}_v2")
                Z = extract_latent(res["model"], feat)
            elif mtype == "BetaVAE":
                m = BetaVAE(idim, latent_dim, HIDDEN_DIMS, beta=beta)
                res = train_vae(m, make_loader(feat), num_epochs=epochs,
                                kl_weight=beta,
                                model_name=f"qft_{mtype}_b{beta}_v2")
                Z = extract_latent(res["model"], feat)
            elif mtype == "CVAE":
                m = CVAE(idim, latent_dim, data["condition_dim"], HIDDEN_DIMS)
                res = train_cvae(m, make_loader_xy(feat, data["conds_lg"]),
                                 num_epochs=epochs, kl_weight=beta,
                                 model_name=f"qft_{mtype}_b{beta}_v2")
                Z = extract_latent_cvae(res["model"], feat, data["conds_lg"])
            elif mtype == "MultiModalVAE":
                m = MultiModalVAE(idim, 384, latent_dim, HIDDEN_DIMS)
                res = train_multimodal_vae(
                    m, make_loader_dict(feat, data["lyrics_n"]),
                    num_epochs=epochs, kl_weight=beta,
                    model_name=f"qft_{mtype}_b{beta}_v2")
                Z = extract_latent_mm(res["model"], feat, data["lyrics_n"])
        except Exception as e:
            print(f"  SKIP finetune {mtype} b={beta}: {e}")
            continue

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
            em.update(model=mtype, beta=beta, latent_dim=latent_dim,
                      k=k, eval_type=eval_type)
            rows.append(em)
            print(f"  qft {mtype:15s} b={beta} ld={latent_dim} K={k:2d} "
                  f"[{eval_type:<14}]  SS={em['SS']:.3f}  ARI={em['ARI']:.3f}")
    return rows

# ── merge into leaderboard ────────────────────────────────────────────────────
def merge_csv(path, new_rows, key_cols):
    new_df = pd.DataFrame(new_rows)
    if path.exists():
        old_df = pd.read_csv(path)
        # Drop old rows for same models (update them)
        if key_cols:
            mask = old_df[key_cols[0]].isin(new_df[key_cols[0]].unique())
            for col in key_cols[1:]:
                mask &= old_df[col].isin(new_df[col].unique())
            old_df = old_df[~mask]
        df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        df = new_df
    df.to_csv(path, index=False, float_format="%.4f")
    return df

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*",
                        help="Model types to process (default: all available)")
    parser.add_argument("--no-finetune", action="store_true",
                        help="Skip quick finetune, do clustering only")
    args = parser.parse_args()

    # Find available checkpoints
    available = {}
    for fname, (mtype, fkey) in CHECKPOINT_REGISTRY.items():
        if (MODEL_DIR / fname).exists():
            available[mtype] = fname

    if args.models:
        requested = set(args.models)
        available = {k: v for k, v in available.items() if k in requested}

    if not available:
        print("No checkpoints found. Run run_v2_pipeline.py first.")
        return

    print(f"\nFound {len(available)} checkpoint(s): {list(available.keys())}")
    data = load_data()

    # ── Exhaustive clustering ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("[CLUSTERING] Exhaustive clustering on available models")
    print("="*60)
    cluster_rows = []
    for mtype, fname in available.items():
        print(f"\n  Loading {fname} ...")
        try:
            Z = load_and_extract(fname, mtype, data)
            task = TASK_MAP.get(mtype, mtype)
            rows = eval_all_k(Z, task, mtype, data)
            cluster_rows.extend(rows)
        except Exception as e:
            print(f"  ERROR {mtype}: {e}")

    if cluster_rows:
        lb_path = OUT_DIR / "leaderboard_v2.csv"
        merge_csv(lb_path, cluster_rows, ["model", "k", "eval_type"])
        print(f"\nLeaderboard updated -> {lb_path}  ({len(cluster_rows)} new rows)")

    # ── Quick finetune ────────────────────────────────────────────────────────
    if not args.no_finetune:
        print("\n" + "="*60)
        print("[FINETUNE] Quick grid: beta={1.0,2.0} x latent_dim=32 x 30 epochs")
        print("="*60)
        ft_rows = []
        for mtype in available:
            print(f"\n  Finetuning {mtype} ...")
            rows = quick_finetune(mtype, data)
            ft_rows.extend(rows)

        if ft_rows:
            ft_path = OUT_DIR / "finetune_quick_v2.csv"
            merge_csv(ft_path, ft_rows, ["model", "beta", "k", "eval_type"])
            print(f"\nFinetune results saved -> {ft_path}  ({len(ft_rows)} new rows)")

    print("\nDONE.")

if __name__ == "__main__":
    main()
