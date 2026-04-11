"""
Generate all new figures for report_v3.tex.
Reads from results/v2/leaderboard_v2.csv (corrected numbers).
Saves to report/figures/ as fig_v3_*.pdf + .png

Figures produced:
  fig_v3_journey.pdf        -- 3-stage narrative: v1 claim → diagnosis → v2 corrected
  fig_v3_why_k.pdf          -- rationale for K=2/10/18 with ground-truth label distribution
  fig_v3_corrected_lang.pdf -- language bar chart (v2 corrected numbers)
  fig_v3_corrected_genre.pdf -- genre bar chart (v2 corrected numbers, K=10 and K=18)
  fig_v3_finetune.pdf       -- v1 full grid vs v2 quick-finetune comparison
  fig_v3_tsne_v2.pdf        -- t-SNE of v2 HybridVAE checkpoint (v2 features)
  fig_v3_progression.pdf    -- step-by-step ARI progression across all 3 stages
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

def savefig(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

# ─── Load corrected v2 leaderboard ───────────────────────────────────────────
df = pd.read_csv("results/v2/leaderboard_v2.csv")
# normalise model short names
MODEL_SHORT = {
    "BasicVAE(MFCC,v2)":              "BasicVAE",
    "ConvVAE(Mel,v2)":                "ConvVAE",
    "HybridVAE(Mel+Lyrics,v2)":       "HybridVAE",
    "BetaVAE(Combined,v2)":           "BetaVAE",
    "CVAE(Combined+LangGenre,v2)":    "CVAE",
    "MultiModalVAE(Comb+Lyrics,v2)":  "MMoVAE",
    "PCA32(MFCC)":                    "PCA-MFCC",
    "PCA32(Combined)":                "PCA-Comb",
}
df["short"] = df["model"].map(MODEL_SHORT).fillna(df["model"])

VAE_ORDER   = ["BasicVAE","ConvVAE","HybridVAE","BetaVAE","CVAE","MMoVAE"]
FULL_ORDER  = VAE_ORDER + ["PCA-MFCC","PCA-Comb"]
MODEL_COLORS = {
    "BasicVAE":  "#90CAF9",
    "ConvVAE":   "#42A5F5",
    "HybridVAE": "#1565C0",
    "BetaVAE":   "#FFCC80",
    "CVAE":      "#EF6C00",
    "MMoVAE":    "#FFD54F",
    "PCA-MFCC":  "#A5D6A7",
    "PCA-Comb":  "#388E3C",
}

def get(subset, short, metric):
    rows = subset[subset["short"] == short]
    return float(rows[metric].iloc[0]) if len(rows) > 0 else 0.0

# ─── Figure 1: Journey narrative ─────────────────────────────────────────────
fig = plt.figure(figsize=(15, 5.5))
gs  = GridSpec(1, 3, figure=fig, wspace=0.06)

stage_data = [
    {
        "title":    "Stage 1 — v1 Original\n(K-Means, full-clip features,\nlanguage-tagged lyrics)",
        "color":    "#E3F2FD",
        "border":   "#1565C0",
        "headline": "HybridVAE K=2\nARI = 0.991",
        "hc":       "#1B5E20",
        "bullets": [
            "MLP-VAE → ConvVAE → HybridVAE",
            "→ CVAE / BetaVAE / MultiModal",
            "K-Means only, K=10 only",
            "Proxy lyrics: 'bangla folk music'",
            "Full clip: 30s English / 3s Bangla",
            "Result: near-perfect separation",
        ],
        "icon": "✓",
    },
    {
        "title":    "Stage 2 — Artefact Diagnosis\n(Two bugs identified\nand quantified)",
        "color":    "#FFF3E0",
        "border":   "#E65100",
        "headline": "Two causes found\nARI = 1.000 artefactual",
        "hc":       "#BF360C",
        "bullets": [
            "Bug 1 — Label leakage:",
            "  'bangla' word in MiniLM input",
            "  384-dim embedding pre-separates",
            "Bug 2 — Clip-length bias:",
            "  σ_English≈51 vs σ_Bangla≈27",
            "  Length-correlated MFCC variance",
        ],
        "icon": "!",
    },
    {
        "title":    "Stage 3 — v2 Corrected\n(Exhaustive clustering,\nboth bugs fixed, retrained)",
        "color":    "#E8F5E9",
        "border":   "#2E7D32",
        "headline": "HybridVAE K=2\nARI = 0.484  (genuine)",
        "hc":       "#1B5E20",
        "bullets": [
            "Fix 1: language-neutral lyrics",
            "Fix 2: center 3s window extraction",
            "Retrained all 6 VAE models",
            "K-Means + GMM + Aggl.-Ward/Comp.",
            "Evaluated at K=2 / 10 / 18",
            "Genre ARI: +54% at K=10",
        ],
        "icon": "✓✓",
    },
]

for i, sd in enumerate(stage_data):
    ax = fig.add_subplot(gs[i])
    ax.set_facecolor(sd["color"])
    for spine in ax.spines.values():
        spine.set_edgecolor(sd["border"])
        spine.set_linewidth(2.5)
    ax.set_xticks([]); ax.set_yticks([])

    ax.text(0.5, 0.97, sd["title"], transform=ax.transAxes,
            ha="center", va="top", fontsize=9.5, fontweight="bold",
            color=sd["border"])
    ax.axhline(y=0.78, xmin=0.05, xmax=0.95, color=sd["border"], linewidth=0.8, alpha=0.5)
    ax.text(0.5, 0.73, sd["headline"], transform=ax.transAxes,
            ha="center", va="top", fontsize=11, fontweight="bold",
            color=sd["hc"],
            bbox=dict(facecolor="white", edgecolor=sd["border"],
                      boxstyle="round,pad=0.3", alpha=0.9))
    for j, b in enumerate(sd["bullets"]):
        ax.text(0.07, 0.55 - j*0.085, b, transform=ax.transAxes,
                ha="left", va="top", fontsize=8.5, color="#333333")

    if i < 2:
        ax.annotate("", xy=(1.0, 0.5), xytext=(0.93, 0.5),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color=sd["border"],
                                    lw=2, connectionstyle="arc3,rad=0"))

fig.suptitle("Research Journey: v1 Baseline → Artefact Discovery → v2 Corrected",
             fontsize=12, fontweight="bold", y=1.02)
savefig(fig, "fig_v3_journey.pdf")


# ─── Figure 2: Why K=2 / K=10 / K=18 ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Panel 1: K=2 — language distribution
langs = ["English\n(GTZAN+MagnaTag)", "Bangla\n(BanglaBeats)"]
counts_lang = [10000, 10020]
colors_lang = ["#1565C0", "#E65100"]
axes[0].bar(langs, counts_lang, color=colors_lang, edgecolor="white", width=0.5)
axes[0].set_title("K = 2  →  Language Separation\n(binary English / Bangla)",
                  fontsize=10, fontweight="bold")
axes[0].set_ylabel("Track count", fontsize=9)
axes[0].set_ylim(0, 14000)
for i, v in enumerate(counts_lang):
    axes[0].text(i, v + 200, f"{v:,}", ha="center", fontsize=9, fontweight="bold")
axes[0].text(0.5, 0.15, "Near-equal split\n→ K=2 natural choice", transform=axes[0].transAxes,
             ha="center", fontsize=8.5, color="#333",
             bbox=dict(facecolor="#E3F2FD", edgecolor="#1565C0", boxstyle="round,pad=0.3"))
axes[0].grid(axis="y", alpha=0.3)

# Panel 2: K=10 — genre distribution over all tracks
genres_all = ["Blues", "Classical", "Country", "Disco", "Hip-Hop",
              "Jazz", "Metal", "Pop", "Reggae", "Rock", "+Bangla\n(8 genres)",
              "MagnaTag\n(untagged)"]
# approximate: GTZAN 100/genre=10 genres, BanglaBeats ~1252/genre, MagnaTag untagged
counts_genre = [100]*10 + [10020, 9000]
colors_genre = ["#42A5F5"]*10 + ["#EF6C00", "#A5D6A7"]
axes[1].bar(range(len(genres_all)), counts_genre, color=colors_genre, edgecolor="white")
axes[1].set_xticks(range(len(genres_all)))
axes[1].set_xticklabels(genres_all, rotation=45, ha="right", fontsize=7.5)
axes[1].set_title("K = 10  →  Genre Clustering (All Tracks)\n(10 GTZAN genres; Bangla+MagnaTag mapped to 10)",
                  fontsize=10, fontweight="bold")
axes[1].set_ylabel("Track count (approx.)", fontsize=9)
axes[1].text(0.5, 0.8, "All 20,020 tracks\nMagnaTag → 'untagged'", transform=axes[1].transAxes,
             ha="center", fontsize=8.5, color="#333",
             bbox=dict(facecolor="#FFF8E1", edgecolor="#F57C00", boxstyle="round,pad=0.3"))
axes[1].grid(axis="y", alpha=0.3)

# Panel 3: K=18 — labeled genres only
genres_lab_en = ["Blues","Classical","Country","Disco","HipHop",
                 "Jazz","Metal","Pop","Reggae","Rock"]
genres_lab_bn = ["Adhunik","Folk","HipHop","Indie","Metal","Patriotic","Rock","Spiritual"]
counts_en = [100]*10
counts_bn = [int(10020/8)]*8
axes[2].bar(range(10), counts_en, color="#42A5F5", edgecolor="white", label="GTZAN (English)")
axes[2].bar(range(10, 18), counts_bn, color="#EF6C00", edgecolor="white", label="BanglaBeats (Bangla)")
axes[2].set_xticks(range(18))
axes[2].set_xticklabels(genres_lab_en + genres_lab_bn, rotation=45, ha="right", fontsize=7)
axes[2].set_title("K = 18  →  Genre Clustering (Labeled Only)\n(11,020 tracks; MagnaTagATune excluded — untagged)",
                  fontsize=10, fontweight="bold")
axes[2].set_ylabel("Track count (approx.)", fontsize=9)
axes[2].legend(fontsize=8)
axes[2].text(0.5, 0.8, "Excludes 9,000 MagnaTag\nuntagged tracks", transform=axes[2].transAxes,
             ha="center", fontsize=8.5, color="#333",
             bbox=dict(facecolor="#E8F5E9", edgecolor="#2E7D32", boxstyle="round,pad=0.3"))
axes[2].grid(axis="y", alpha=0.3)

fig.suptitle("Rationale for Three Evaluation Granularities: K = 2, 10, 18",
             fontsize=12, fontweight="bold")
fig.tight_layout()
savefig(fig, "fig_v3_why_k.pdf")


# ─── Figure 3: Corrected v2 language bar ─────────────────────────────────────
lang_df = df[df["eval_type"] == "language"].copy()
models_plot = [m for m in FULL_ORDER if m in lang_df["short"].values]

fig, ax = plt.subplots(figsize=(11, 4.5))
metrics   = ["ARI",  "NMI",  "Purity"]
mc        = ["#1565C0", "#E65100", "#2E7D32"]
x         = np.arange(len(models_plot))
w         = 0.25
offsets   = [-w, 0, w]

for mi, (met, col) in enumerate(zip(metrics, mc)):
    vals = [get(lang_df, m, met) for m in models_plot]
    bars = ax.bar(x + offsets[mi], vals, w, label=met, color=col, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, vals):
        if v > 0.03:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)

ax.set_xticks(x)
ax.set_xticklabels(models_plot, fontsize=9)
for tick, m in zip(ax.get_xticklabels(), models_plot):
    tick.set_color(MODEL_COLORS.get(m, "black"))
ax.set_ylabel("Score", fontsize=10)
ax.set_ylim(0, 1.12)
ax.set_title("Language Separation (K=2) — v2 Corrected Results\n"
             "HybridVAE: genuine ARI=0.484 after removing label leakage & clip-length bias",
             fontsize=10, fontweight="bold")
ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
ax.legend(fontsize=9, loc="upper right")
ax.grid(axis="y", alpha=0.3)
# Annotate HybridVAE
hybrid_x = models_plot.index("HybridVAE")
ax.annotate("Genuine ARI=0.484\n(was 0.991 in v1)", xy=(hybrid_x - w, 0.484+0.01),
            xytext=(hybrid_x - w - 0.8, 0.75),
            arrowprops=dict(arrowstyle="->", color="#1565C0"),
            fontsize=8, color="#1565C0", fontweight="bold")
fig.tight_layout()
savefig(fig, "fig_v3_corrected_lang.pdf")


# ─── Figure 4: Genre bars (K=10 and K=18 side-by-side) ───────────────────────
ga_df = df[df["eval_type"] == "genre_all"].copy()
gl_df = df[df["eval_type"] == "genre_labeled"].copy()
models_plot = [m for m in FULL_ORDER if m in ga_df["short"].values]

fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
for ax, sub_df, title_suffix in zip(axes, [ga_df, gl_df],
                                    ["Genre-All (K=10, N=20,020)", "Genre-Labeled (K=18, N=11,020)"]):
    x = np.arange(len(models_plot))
    w = 0.25
    for mi, (met, col) in enumerate(zip(["ARI","NMI","Purity"], mc)):
        vals = [get(sub_df, m, met) for m in models_plot]
        bars = ax.bar(x + [-w, 0, w][mi], vals, w, label=met, color=col, alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            if v > 0.03:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(models_plot, fontsize=8.5, rotation=15, ha="right")
    for tick, m in zip(ax.get_xticklabels(), models_plot):
        tick.set_color(MODEL_COLORS.get(m, "black"))
    ax.set_ylabel("Score", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"v2 Corrected — {title_suffix}", fontsize=10, fontweight="bold")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
savefig(fig, "fig_v3_corrected_genre.pdf")


# ─── Figure 5: Finetune comparison — v1 full grid vs v2 quick grid ───────────
# v1 full grid (156 configs): from report.tex table
# v2 quick grid (12-36 rows): from results/v2/finetune_quick_v2.csv
ft_v2 = pd.read_csv("results/v2/finetune_quick_v2.csv")

# best per (model, k) across betas
def best_ft(ft_df, model, k, metric="ARI"):
    sub = ft_df[(ft_df["model"] == model) & (ft_df["k"] == k)]
    if len(sub) == 0:
        return 0.0
    return float(sub[metric].max())

# v1 known best SS at K=10 labeled from the hyperparameter table in v1 report
v1_best_ss = {
    "BasicVAE":   (32, 1.0, 0.507),  # (dz, beta, SS)
    "MMoVAE":     (32, 1.0, 0.502),
    "CVAE":       (16, 1.0, 0.493),
    "BetaVAE":    (32, 2.0, 0.382),
}

# v2 quick grid best SS at K=10
def best_v2_ss(model_name, k=10):
    sub = ft_v2[(ft_v2["model"] == model_name) & (ft_v2["k"] == k)]
    if len(sub) == 0:
        return 0.0
    return float(sub["SS"].max())

ft_models_v2 = ["BasicVAE", "ConvVAE", "HybridVAE", "BetaVAE", "CVAE", "MultiModalVAE"]
ft_models_v1 = ["BasicVAE", None,        None,        "BetaVAE", "CVAE", "MMoVAE"]  # v1 names

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Panel 1: Beta comparison on ARI at K=10
models_show = ["BasicVAE", "BetaVAE", "CVAE", "HybridVAE", "MultiModalVAE"]
beta1_ari = [best_ft(ft_v2, m, 10, "ARI") for m in models_show]
beta2_rows = ft_v2[(ft_v2["beta"] == 2.0) & (ft_v2["k"] == 10)]
beta2_ari  = [get(beta2_rows.rename(columns={"model": "short"}), m, "ARI") for m in models_show]
x = np.arange(len(models_show))
axes[0].bar(x - 0.2, beta1_ari, 0.38, label="β=1.0", color="#1565C0", alpha=0.85)
axes[0].bar(x + 0.2, beta2_ari, 0.38, label="β=2.0", color="#E65100", alpha=0.85)
for i, (b1, b2) in enumerate(zip(beta1_ari, beta2_ari)):
    axes[0].text(i - 0.2, b1 + 0.005, f"{b1:.3f}", ha="center", va="bottom", fontsize=7.5)
    axes[0].text(i + 0.2, b2 + 0.005, f"{b2:.3f}", ha="center", va="bottom", fontsize=7.5)
axes[0].set_xticks(x)
axes[0].set_xticklabels(models_show, rotation=20, ha="right", fontsize=8.5)
axes[0].set_ylabel("ARI (K=10, genre-all)", fontsize=9)
axes[0].set_title("v2 Quick Grid: β=1 vs β=2\n(dz=32, 30 epochs)", fontsize=10, fontweight="bold")
axes[0].legend(fontsize=9)
axes[0].grid(axis="y", alpha=0.3)
axes[0].set_ylim(0, max(max(beta1_ari), max(beta2_ari)) * 1.35 + 0.05)

# Panel 2: K sensitivity — HybridVAE best ARI at K=2/10/18 for β=1 and β=2
ks = [2, 10, 18]
hyb_b1 = [best_ft(ft_v2[ft_v2["beta"] == 1.0], "HybridVAE", k) for k in ks]
hyb_b2 = [best_ft(ft_v2[ft_v2["beta"] == 2.0], "HybridVAE", k) for k in ks]
x = np.arange(3)
axes[1].bar(x - 0.2, hyb_b1, 0.38, label="β=1.0", color="#1565C0", alpha=0.85)
axes[1].bar(x + 0.2, hyb_b2, 0.38, label="β=2.0", color="#E65100", alpha=0.85)
for i, (b1, b2) in enumerate(zip(hyb_b1, hyb_b2)):
    axes[1].text(i - 0.2, b1 + 0.005, f"{b1:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    axes[1].text(i + 0.2, b2 + 0.005, f"{b2:.3f}", ha="center", va="bottom", fontsize=8.5)
axes[1].set_xticks(x)
axes[1].set_xticklabels(["K=2\n(language)", "K=10\n(genre-all)", "K=18\n(genre-lab.)"], fontsize=9)
axes[1].set_ylabel("Best ARI", fontsize=9)
axes[1].set_title("HybridVAE: ARI at K=2/10/18\nv2 Quick Finetune Grid", fontsize=10, fontweight="bold")
axes[1].legend(fontsize=9)
axes[1].grid(axis="y", alpha=0.3)
axes[1].set_ylim(0, 1.1)

# Panel 3: v1 full grid vs v2 quick grid — SS comparison at K=10
ft_models_cmp = ["BasicVAE", "BetaVAE", "CVAE", "MMoVAE/MultiModal"]
ft_map_v2 = {"BasicVAE": "BasicVAE", "BetaVAE": "BetaVAE",
             "CVAE": "CVAE", "MMoVAE/MultiModal": "MultiModalVAE"}
v1_ss_vals = [0.507, 0.382, 0.493, 0.502]
v2_ss_vals = [best_v2_ss(ft_map_v2[m]) for m in ft_models_cmp]

x = np.arange(len(ft_models_cmp))
axes[2].bar(x - 0.2, v1_ss_vals, 0.38, label="v1 full grid\n(156 configs, 60 ep)", color="#4CAF50", alpha=0.85)
axes[2].bar(x + 0.2, v2_ss_vals, 0.38, label="v2 quick grid\n(β∈{1,2}, 30 ep)", color="#FF9800", alpha=0.85)
for i, (v1s, v2s) in enumerate(zip(v1_ss_vals, v2_ss_vals)):
    axes[2].text(i - 0.2, v1s + 0.005, f"{v1s:.3f}", ha="center", va="bottom", fontsize=8)
    axes[2].text(i + 0.2, v2s + 0.005, f"{v2s:.3f}", ha="center", va="bottom", fontsize=8)
axes[2].set_xticks(x)
axes[2].set_xticklabels(ft_models_cmp, rotation=15, ha="right", fontsize=8.5)
axes[2].set_ylabel("Silhouette Score (K=10)", fontsize=9)
axes[2].set_title("v1 Full Grid vs v2 Quick Grid\nSS Comparison (β=1.0 optimal in both)",
                  fontsize=10, fontweight="bold")
axes[2].legend(fontsize=8.5)
axes[2].grid(axis="y", alpha=0.3)
axes[2].set_ylim(0, 0.75)

fig.tight_layout()
savefig(fig, "fig_v3_finetune.pdf")


# ─── Figure 6: Step-by-step ARI progression across all 3 stages ──────────────
# v1 K=2 lang, v1 K=10 genre, v2-corrected K=2 lang, v2-corrected K=10 genre

stages    = ["v1\n(K-Means only)", "v2-corrected\n(best algorithm)"]
stage_idx = [0, 1]

v1_k2_ari = {
    "BasicVAE":  0.001, "ConvVAE":  0.055, "HybridVAE": 0.991,
    "BetaVAE":   0.009, "CVAE":     0.004, "MMoVAE":    0.005,
}
v1_k10_ari = {
    "BasicVAE":  0.022, "ConvVAE":  0.019, "HybridVAE": 0.269,
    "BetaVAE":   0.105, "CVAE":     0.052, "MMoVAE":    0.035,
}

lang_df2   = df[df["eval_type"] == "language"]
genre_df2  = df[df["eval_type"] == "genre_all"]

v2_k2_ari = {m: get(lang_df2,  m, "ARI") for m in VAE_ORDER}
v2_k10_ari= {m: get(genre_df2, m, "ARI") for m in VAE_ORDER}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(len(VAE_ORDER))
w = 0.38

for ax, (v1d, v2d, title) in zip(axes, [
    (v1_k2_ari,  v2_k2_ari,  "Language Separation (K=2)\nARI: v1 (K-Means) → v2 (best algorithm)"),
    (v1_k10_ari, v2_k10_ari, "Genre Clustering (K=10)\nARI: v1 (K-Means) → v2 (best algorithm)"),
]):
    v1_vals = [v1d.get(m, 0) for m in VAE_ORDER]
    v2_vals = [v2d.get(m, 0) for m in VAE_ORDER]
    b1 = ax.bar(x - w/2, v1_vals, w, label="v1 (K-Means only)", color="#90CAF9", edgecolor="white")
    b2 = ax.bar(x + w/2, v2_vals, w, label="v2 corrected (best algorithm)", color="#1565C0", edgecolor="white", alpha=0.9)
    for bar, v in zip(b1, v1_vals):
        if v > 0.01:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.008,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, color="#546E7A")
    for bar, v in zip(b2, v2_vals):
        if v > 0.01:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.008,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#0D47A1")
    ax.set_xticks(x)
    ax.set_xticklabels(VAE_ORDER, fontsize=9)
    ax.set_ylabel("ARI", fontsize=10)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.2 if "K=2" in title else 0.65)
    if "K=2" in title:
        ax.axhline(0.991, color="#E57373", linestyle=":", linewidth=1.2, alpha=0.8)
        ax.text(5.5, 0.995, "v1 ARI=0.991\n(artefactual)", ha="right", fontsize=7.5, color="#C62828")
        ax.annotate("", xy=(2 + w/2, 0.484), xytext=(2 + w/2, 0.991 - 0.01),
                    arrowprops=dict(arrowstyle="-[", color="#E65100", lw=1.5))
        ax.text(2 + w/2 + 0.08, 0.74, "artefact\nremoved", fontsize=7.5, color="#E65100")

fig.suptitle("Step-by-Step ARI Improvement: v1 Baseline → v2 Corrected\n"
             "(v2 gains from: bug fixes + model retraining + exhaustive clustering)",
             fontsize=11, fontweight="bold")
fig.tight_layout()
savefig(fig, "fig_v3_progression.pdf")


# ─── Figure 7: t-SNE of v2 HybridVAE checkpoint ─────────────────────────────
HYBRID_V2_CKPT = "results/models/v2/hybrid_vae_medium_v2.pt"
MEL_FEATURES   = "data/features/mel_features.npy"
MEL_META       = "data/features/mel_metadata.csv"
LYRICS_EMB     = "data/features/lyrics_embeddings/lyrics_embeddings.npy"

required = [HYBRID_V2_CKPT, MEL_FEATURES, MEL_META, LYRICS_EMB]
if all(os.path.exists(p) for p in required):
    try:
        import torch
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
        sys.path.insert(0, "src")
        from vae import BasicVAE

        device = torch.device("cpu")
        mel  = np.load(MEL_FEATURES).astype(np.float32)
        lyr  = np.load(LYRICS_EMB).astype(np.float32)
        meta = pd.read_csv(MEL_META)

        n = min(len(mel), len(lyr), len(meta))
        combined = np.concatenate([mel[:n], lyr[:n]], axis=1)
        scaler = StandardScaler()
        combined_s = scaler.fit_transform(combined)

        ckpt = torch.load(HYBRID_V2_CKPT, map_location=device)
        input_dim  = combined.shape[1]
        latent_dim = ckpt.get("latent_dim", 32)
        hidden_dims = ckpt.get("hidden_dims", [512, 256])
        model = BasicVAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        mu_list = []
        with torch.no_grad():
            for i in range(0, len(combined_s), 512):
                batch = torch.FloatTensor(combined_s[i:i+512])
                mu, _ = model.encode(batch)
                mu_list.append(mu.numpy())
        latents = np.vstack(mu_list)

        rng = np.random.RandomState(42)
        idx = rng.choice(len(latents), size=min(5000, len(latents)), replace=False)
        z   = latents[idx]
        m   = meta.iloc[:n].iloc[idx]

        tsne = TSNE(n_components=2, perplexity=40, max_iter=1000, random_state=42)
        z2d  = tsne.fit_transform(z)

        lang_col = "language" if "language" in m.columns else None
        genre_col= "genre"    if "genre"    in m.columns else None

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

        if lang_col:
            lang_labels = m[lang_col].values
            for lang, c in [("english","#1565C0"),("bangla","#E65100")]:
                mask = lang_labels == lang
                axes[0].scatter(z2d[mask,0], z2d[mask,1], c=c, s=5, alpha=0.55,
                                label=lang.capitalize(), rasterized=True)
            axes[0].set_title("v2 HybridVAE Latent Space\n(coloured by language — ARI=0.484, K=2)",
                              fontsize=10, fontweight="bold")
            axes[0].legend(markerscale=3, fontsize=9)
            axes[0].text(0.02, 0.03, "Genuine ARI=0.484\n(language-neutral lyrics, 3s window)",
                         transform=axes[0].transAxes, fontsize=8, color="#1B5E20",
                         bbox=dict(facecolor="white", edgecolor="#1B5E20", boxstyle="round,pad=0.3"))

        if genre_col:
            genre_labels = m[genre_col].fillna("unknown").values
            genres_show  = [g for g in sorted(set(genre_labels)) if g != "untagged"][:18]
            cmap = plt.cm.tab20(np.linspace(0, 1, len(genres_show)))
            gcmap = {g: cmap[i] for i, g in enumerate(genres_show)}
            gcmap["untagged"] = (0.7, 0.7, 0.7, 0.3)
            for g in genres_show:
                mask = genre_labels == g
                axes[1].scatter(z2d[mask,0], z2d[mask,1], color=gcmap[g], s=5, alpha=0.6,
                                label=g, rasterized=True)
            mask = genre_labels == "untagged"
            if mask.sum() > 0:
                axes[1].scatter(z2d[mask,0], z2d[mask,1], color=(0.7,0.7,0.7), s=3,
                                alpha=0.2, label="untagged", rasterized=True)
            axes[1].set_title("v2 HybridVAE Latent Space\n(coloured by genre — ARI=0.436, K=18)",
                              fontsize=10, fontweight="bold")
            axes[1].legend(markerscale=2, fontsize=6.5, ncol=2, loc="lower right")

        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])

        fig.suptitle("t-SNE of v2 HybridVAE Latent Space\n(5,000 samples, v2 checkpoint + language-neutral lyrics)",
                     fontsize=11, fontweight="bold")
        fig.tight_layout()
        savefig(fig, "fig_v3_tsne_v2.pdf")

    except Exception as e:
        print(f"t-SNE skipped: {e}")
        import traceback; traceback.print_exc()
else:
    missing = [p for p in required if not os.path.exists(p)]
    print(f"t-SNE skipped — missing files: {missing}")

print("\nAll fig_v3_* figures generated.")
