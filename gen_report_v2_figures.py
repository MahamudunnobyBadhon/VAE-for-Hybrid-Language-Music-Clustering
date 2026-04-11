"""
Generate figures for report-v2.
Saves all figures to report/figures/ (creates if absent).

Figures produced:
  1. fig_v2_language_bar.pdf  -- ARI/NMI/Purity bar chart (language K=2) all models
  2. fig_v2_genre_all_bar.pdf -- ARI/NMI/Purity bar chart (genre_all K=10) all models
  3. fig_v2_genre_lab_bar.pdf -- ARI/NMI/Purity bar chart (genre_labeled K=18) all models
  4. fig_v2_v1v2_delta.pdf   -- grouped bar chart showing v1→v2 improvement
  5. fig_v2_tsne_hybrid.pdf  -- t-SNE of HybridVAE latent space (language + genre coloured)
  6. fig_v2_method_heatmap.pdf -- heatmap of ARI by (model x eval_type) with method annotation
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

FIGURES_DIR = os.path.join("report", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

LEADERBOARD = "results/report_v2/leaderboard.csv"

# ─── load results ────────────────────────────────────────────────────────────
df = pd.read_csv(LEADERBOARD)
# shorter display names
NAME_MAP = {
    "Medium-HybridVAE":         "HybridVAE",
    "Medium-ConvVAE":           "ConvVAE",
    "Easy-BasicVAE":            "BasicVAE",
    "Hard-BetaVAE":             "BetaVAE",
    "Hard-CVAE":                "CVAE",
    "Hard-MultiModalVAE":       "MMoVAE",
    "Baseline-PCA32-mel":       "PCA-mel",
    "Baseline-PCA32-combined":  "PCA-comb",
}
df["short_name"] = df["model"].map(NAME_MAP)

# colour palette by model type
COLOURS = {
    "HybridVAE":  "#2196F3",
    "ConvVAE":    "#03A9F4",
    "BasicVAE":   "#B3E5FC",
    "BetaVAE":    "#FF9800",
    "CVAE":       "#F57C00",
    "MMoVAE":     "#FFF176",
    "PCA-mel":    "#A5D6A7",
    "PCA-comb":   "#66BB6A",
}

MODEL_ORDER = ["BasicVAE", "ConvVAE", "HybridVAE", "BetaVAE", "CVAE", "MMoVAE", "PCA-mel", "PCA-comb"]

# ─── helper: grouped metric bar chart ────────────────────────────────────────
def metric_bar_chart(subset, title, filename,
                     metrics=("adjusted_rand_index","normalized_mutual_info","cluster_purity"),
                     metric_labels=("ARI", "NMI", "Purity")):
    subset = subset.set_index("short_name").reindex(MODEL_ORDER).dropna(how="all")
    models = list(subset.index)
    n = len(models)
    m = len(metrics)
    x = np.arange(n)
    width = 0.22
    offsets = np.linspace(-(m-1)/2*width, (m-1)/2*width, m)
    colours_bars = ["#1565C0", "#E65100", "#2E7D32"]

    fig, ax = plt.subplots(figsize=(max(9, n*1.1), 4.5))
    for i, (metric, label, c) in enumerate(zip(metrics, metric_labels, colours_bars)):
        vals = [subset.loc[m, metric] if m in subset.index else 0 for m in models]
        bars = ax.bar(x + offsets[i], vals, width, label=label, color=c, alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            if v > 0.02:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    # colour x-tick labels by model type
    for tick, model in zip(ax.get_xticklabels(), models):
        tick.set_color(COLOURS.get(model, "black"))
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    # also save PNG for PDF embedding
    fig.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# Figure 1: language K=2
lang_df = df[df["eval_type"] == "language"].copy()
metric_bar_chart(lang_df, "Language Separation (K=2) — All Models", "fig_v2_language_bar.pdf",
                 metrics=("adjusted_rand_index","normalized_mutual_info","cluster_purity"),
                 metric_labels=("ARI", "NMI", "Purity"))

# Figure 2: genre_all K=10
ga_df = df[df["eval_type"] == "genre_all"].copy()
metric_bar_chart(ga_df, "Genre Clustering — All Tracks (K=10)", "fig_v2_genre_all_bar.pdf")

# Figure 3: genre_labeled K=18
gl_df = df[df["eval_type"] == "genre_labeled"].copy()
metric_bar_chart(gl_df, "Genre Clustering — Labeled Tracks Only (K=18)", "fig_v2_genre_lab_bar.pdf")


# ─── Figure 4: v1 vs v2 delta bar chart ──────────────────────────────────────
v1_data = {
    # (model_short, eval_type): {metric: v1_value}
    ("HybridVAE", "language"):     {"ARI": 0.991, "NMI": 0.978, "Purity": 0.998},
    ("HybridVAE", "genre_all"):    {"ARI": 0.209, "NMI": 0.384, "Purity": 0.637},
    ("HybridVAE", "genre_labeled"):{"ARI": None,  "NMI": None,  "Purity": None},
    ("CVAE",      "genre_all"):    {"ARI": 0.071, "NMI": 0.225, "Purity": 0.468},
}
v2_data = {
    ("HybridVAE", "language"):      {"ARI": 1.000, "NMI": 1.000, "Purity": 1.000},
    ("HybridVAE", "genre_all"):     {"ARI": 0.332, "NMI": 0.499, "Purity": 0.660},
    ("HybridVAE", "genre_labeled"): {"ARI": 0.269, "NMI": 0.473, "Purity": 0.487},
    ("CVAE",      "genre_all"):     {"ARI": 0.053, "NMI": 0.245, "Purity": 0.543},
}

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
metrics_delta = ["ARI", "NMI", "Purity"]
colors_v = {"v1": "#90CAF9", "v2": "#1565C0"}

cases = [
    ("HybridVAE\nlanguage K=2",   ("HybridVAE","language")),
    ("HybridVAE\ngenre-all K=10", ("HybridVAE","genre_all")),
    ("HybridVAE\ngenre-lab K=18", ("HybridVAE","genre_labeled")),
]

for ax, (case_label, key) in zip(axes, cases):
    v1 = v1_data.get(key, {})
    v2 = v2_data.get(key, {})
    xs = np.arange(len(metrics_delta))
    w = 0.32
    v1_vals = [v1.get(m) or 0.0 for m in metrics_delta]
    v2_vals = [v2.get(m) or 0.0 for m in metrics_delta]
    b1 = ax.bar(xs - w/2, v1_vals, w, label="v1", color=colors_v["v1"], edgecolor="white")
    b2 = ax.bar(xs + w/2, v2_vals, w, label="v2", color=colors_v["v2"], edgecolor="white")
    for bar, val in zip(b2, v2_vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, color="#0D47A1")
    ax.set_xticks(xs)
    ax.set_xticklabels(metrics_delta, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_title(case_label, fontsize=10, fontweight="bold")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

fig.suptitle("v1 → v2 Improvement (Exhaustive Clustering Search, No Retraining)", fontsize=11, fontweight="bold")
fig.tight_layout()
path = os.path.join(FIGURES_DIR, "fig_v2_v1v2_delta.pdf")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")


# ─── Figure 5: ARI heatmap  ──────────────────────────────────────────────────
pivot = df.pivot_table(index="short_name", columns="eval_type",
                        values="adjusted_rand_index", aggfunc="max")
pivot = pivot.reindex(MODEL_ORDER).dropna(how="all")
pivot = pivot.reindex(columns=["language","genre_all","genre_labeled"])
pivot.columns = ["Language\n(K=2)", "Genre-All\n(K=10)", "Genre-Labeled\n(K=18)"]

method_pivot = df.pivot_table(index="short_name", columns="eval_type",
                               values="method", aggfunc="first")
method_pivot = method_pivot.reindex(MODEL_ORDER).dropna(how="all")
method_pivot = method_pivot.reindex(columns=["language","genre_all","genre_labeled"])

fig, ax = plt.subplots(figsize=(7, 5))
data = pivot.values.astype(float)
im = ax.imshow(data, cmap="Blues", vmin=0, vmax=1.0, aspect="auto")
plt.colorbar(im, ax=ax, label="ARI", fraction=0.03, pad=0.04)
ax.set_xticks(range(pivot.shape[1]))
ax.set_xticklabels(pivot.columns, fontsize=9)
ax.set_yticks(range(pivot.shape[0]))
ax.set_yticklabels(pivot.index, fontsize=9)
ax.set_title("ARI Heatmap by Model and Evaluation Protocol\n(Best clustering method per cell)", fontsize=10, fontweight="bold")

for i, model in enumerate(pivot.index):
    for j, col in enumerate(["language","genre_all","genre_labeled"]):
        val = data[i, j]
        meth = method_pivot.loc[model, col] if model in method_pivot.index else ""
        if np.isnan(val):
            continue
        cell_text = f"{val:.3f}\n{meth}" if meth else f"{val:.3f}"
        text_color = "white" if val > 0.5 else "black"
        ax.text(j, i, cell_text, ha="center", va="center",
                fontsize=7.5, color=text_color, fontweight="bold" if val >= 0.9 else "normal")

fig.tight_layout()
path = os.path.join(FIGURES_DIR, "fig_v2_ari_heatmap.pdf")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")


# ─── Figure 6: t-SNE of HybridVAE (if checkpoint exists) ────────────────────
HYBRID_CKPT = "results/models/hybrid_vae_medium.pt"
MEL_FEATURES = "data/features/mel_features.npy"
MEL_META     = "data/features/mel_metadata.csv"
LYRICS_EMB   = "data/features/lyrics_embeddings/lyrics_embeddings.npy"

tsne_possible = all(os.path.exists(p) for p in [HYBRID_CKPT, MEL_FEATURES, MEL_META, LYRICS_EMB])

if tsne_possible:
    try:
        import torch
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
        sys.path.insert(0, "src")
        from vae import BasicVAE  # HybridVAE is BasicVAE trained on mel+lyrics

        device = torch.device("cpu")
        mel_features = np.load(MEL_FEATURES).astype(np.float32)
        lyrics_emb   = np.load(LYRICS_EMB).astype(np.float32)
        meta         = pd.read_csv(MEL_META)

        # Align lengths
        n = min(len(mel_features), len(lyrics_emb), len(meta))
        mel_features = mel_features[:n]
        lyrics_emb   = lyrics_emb[:n]
        meta         = meta.iloc[:n].reset_index(drop=True)

        # Concatenate
        combined = np.concatenate([mel_features, lyrics_emb], axis=1)
        scaler = StandardScaler()
        combined_s = scaler.fit_transform(combined)

        # Load model — HybridVAE checkpoint is a BasicVAE on the concatenated input
        ckpt      = torch.load(HYBRID_CKPT, map_location=device)
        input_dim  = ckpt.get("input_dim", combined.shape[1])
        latent_dim = ckpt.get("latent_dim", 32)
        hidden_dims = ckpt.get("hidden_dims", [512, 256])
        model = BasicVAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Encode in batches
        BS = 512
        mu_list = []
        with torch.no_grad():
            for i in range(0, len(combined_s), BS):
                batch = torch.FloatTensor(combined_s[i:i+BS])
                mu, _ = model.encode(batch)
                mu_list.append(mu.numpy())
        latents = np.vstack(mu_list)

        # Subsample for speed
        rng = np.random.RandomState(42)
        idx = rng.choice(len(latents), size=min(5000, len(latents)), replace=False)
        z   = latents[idx]
        m   = meta.iloc[idx]

        # t-SNE
        tsne = TSNE(n_components=2, perplexity=40, max_iter=1000, random_state=42)
        z2d = tsne.fit_transform(z)

        # --- subplot 1: coloured by language ---
        lang_labels = m["language"].values if "language" in m.columns else np.zeros(len(m))
        languages   = sorted(set(lang_labels))
        lang_cmap   = {"english": "#1565C0", "bangla": "#E65100"}

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        for lang in languages:
            mask = lang_labels == lang
            axes[0].scatter(z2d[mask, 0], z2d[mask, 1],
                            c=lang_cmap.get(lang, "gray"), s=5, alpha=0.55,
                            label=lang.capitalize(), rasterized=True)
        axes[0].set_title("HybridVAE Latent Space\n(coloured by language)", fontsize=10, fontweight="bold")
        axes[0].legend(markerscale=3, fontsize=9, loc="best")
        axes[0].set_xticks([]); axes[0].set_yticks([])
        axes[0].text(0.02, 0.02, "ARI=1.000 (GMM, K=2)", transform=axes[0].transAxes,
                     fontsize=8, color="#1B5E20",
                     bbox=dict(facecolor="white", edgecolor="#1B5E20", boxstyle="round,pad=0.3"))

        # --- subplot 2: coloured by genre ---
        if "genre" in m.columns:
            genre_labels = m["genre"].fillna("unknown").values
            genres_unique = [g for g in sorted(set(genre_labels)) if g != "untagged"][:18]
            genre_palette = plt.cm.tab20(np.linspace(0, 1, len(genres_unique)))
            genre_colour_map = {g: genre_palette[i] for i, g in enumerate(genres_unique)}
            genre_colour_map["untagged"] = (0.7, 0.7, 0.7, 0.3)

            for g in genres_unique:
                mask = genre_labels == g
                if mask.sum() == 0: continue
                axes[1].scatter(z2d[mask, 0], z2d[mask, 1],
                                color=genre_colour_map[g], s=5, alpha=0.55,
                                label=g, rasterized=True)
            # untagged
            mask = genre_labels == "untagged"
            if mask.sum() > 0:
                axes[1].scatter(z2d[mask, 0], z2d[mask, 1],
                                color=(0.7, 0.7, 0.7), s=3, alpha=0.2, label="untagged", rasterized=True)
            axes[1].set_title("HybridVAE Latent Space\n(coloured by genre)", fontsize=10, fontweight="bold")
            axes[1].legend(markerscale=3, fontsize=6.5, loc="best", ncol=2)
        else:
            axes[1].set_visible(False)

        axes[1].set_xticks([]); axes[1].set_yticks([])
        fig.suptitle("t-SNE of HybridVAE Latent Space (5,000 samples)", fontsize=11, fontweight="bold")
        fig.tight_layout()
        path = os.path.join(FIGURES_DIR, "fig_v2_tsne_hybrid.pdf")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")
    except Exception as e:
        warnings.warn(f"t-SNE figure skipped: {e}")
else:
    print("t-SNE skipped: required files not found")
    print(f"  Needed: {HYBRID_CKPT}, {MEL_FEATURES}, {MEL_META}, {LYRICS_EMB}")

print("\nAll report-v2 figures generated.")
