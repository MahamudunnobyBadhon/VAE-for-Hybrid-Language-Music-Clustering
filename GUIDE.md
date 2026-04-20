# CSE715 Project — Complete Setup & Run Guide
## VAE for Hybrid Language Music Clustering (v1 → v2 → v3)

---

## Table of Contents
1. [Prerequisites](#1-prerequisites)
2. [Environment Setup](#2-environment-setup)
3. [Dataset Acquisition](#3-dataset-acquisition)
4. [Feature Extraction](#4-feature-extraction)
5. [Stage 1 — v1 Baseline Pipeline](#5-stage-1--v1-baseline-pipeline)
6. [Stage 2 — Artefact Fixes (v2 Corrections)](#6-stage-2--artefact-fixes-v2-corrections)
7. [Stage 3 — v2 Corrected Pipeline](#7-stage-3--v2-corrected-pipeline)
8. [HDBSCAN Analysis](#8-hdbscan-analysis)
9. [Generating Report v3 PDF](#9-generating-report-v3-pdf)
10. [File Reference](#10-file-reference)
11. [Source Module Reference](#11-source-module-reference)
12. [Results Directory Layout](#12-results-directory-layout)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10 – 3.13 | 3.11 recommended |
| pip | latest | `python -m pip install --upgrade pip` |
| Git | any | for cloning |
| LaTeX (optional) | MiKTeX / TeX Live | only needed to compile the PDF report |
| FFmpeg | any | bundled as `ffmpeg.exe` in project root |
| ~8 GB disk | — | audio files + features |
| ~4 GB RAM | — | minimum for 20K-sample training |
| GPU (optional) | CUDA 11+ | CPU works; GPU is ~5–10× faster |

---

## 2. Environment Setup

```bash
# Clone the repository
git clone https://github.com/MahamudunnobyBadhon/VAE-for-Hybrid-Language-Music-Clustering.git
cd VAE-for-Hybrid-Language-Music-Clustering

# Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

**requirements.txt installs:**
```
torch>=2.0.0          # Deep learning
torchaudio>=2.0.0     # Audio I/O
librosa>=0.10.0       # Audio feature extraction
scikit-learn>=1.3.0   # Clustering, metrics, PCA
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
umap-learn>=0.5.0     # UMAP dimensionality reduction
tqdm>=4.65.0          # Progress bars
soundfile>=0.12.0     # Audio file I/O
sentence-transformers>=2.2.0  # LaBSE / MiniLM lyrics embeddings
hdbscan>=0.8.0        # HDBSCAN clustering
```

> **GPU note:** If you have a CUDA GPU, install the matching PyTorch CUDA build from
> https://pytorch.org/get-started/locally/ instead of the default CPU version.

---

## 3. Dataset Acquisition

The project uses three datasets totalling **20,020 tracks**:

| Dataset | Language | Tracks | Duration | Genres |
|---------|----------|--------|----------|--------|
| GTZAN | English | 1,000 | 30 s | 10 |
| MagnaTagATune | English | 9,000 | 29 s | multi-tag |
| BanglaBeats | Bangla | 10,020 | 3 s | 8 |

### Option A — Automated download (recommended)

```bash
# Download both datasets and auto-extract features
python build_dataset.py --extract
```

This downloads audio to `data/audio/english/` and `data/audio/bangla/`,
then runs feature extraction automatically.

### Option B — Manual placement

If you already have the audio files:

```
data/
  audio/
    english/   ← WAV files from GTZAN + MagnaTagATune
    bangla/    ← WAV files from BanglaBeats
```

File naming conventions expected:
- GTZAN: `blues.00000.wav`, `rock.00001.wav`, … (genre.number.wav)
- BanglaBeats: `bangla_folk_001.wav`, `bangla_classical_002.wav`, … (bangla_genre_number.wav)
- MagnaTagATune: `english_magna_0001.wav`, … (english_magna_number.wav)

### Option C — Download Bangla only

```bash
# Search and download Bangla tracks from YouTube (uses yt-dlp)
python download_bangla.py --n 200

# Dry run to preview what would be downloaded
python download_bangla.py --n 200 --dry-run
```

---

## 4. Feature Extraction

Feature extraction produces NumPy arrays saved to `data/features/`.

### v1 Feature Extraction (original)

Feature extraction runs automatically inside each task script when `--use-real-audio`
is passed. To run standalone:

```bash
# The run_easy_task.py script handles MFCC extraction
python run_easy_task.py --use-real-audio

# For Mel + combined features (used by medium and hard tasks)
python run_medium_task.py --use-real-audio
```

Extracted files:
```
data/features/
  mfcc_features.npy       # (20020, 40)  — 20 MFCC coefficients × mean+std
  mfcc_metadata.csv       # track filenames, language, genre labels
  mel_features.npy        # (20019, 256) — 128 Mel bands × mean+std
  mel_metadata.csv
  combined_features.npy   # (20019, 90)  — MFCC + Mel + Chroma + Spectral
  combined_metadata.csv
```

### v2 Feature Extraction (3 s center-window fix)

The v1 extraction used different durations for English (30 s) and Bangla (3 s), which
introduced a clip-length variance bias. v2 fixes this by extracting a 3 s center window
for all tracks:

```bash
python reextract_features_3s.py
```

Output: `data/features/v2/` — same structure as above but with uniform 3 s window.

### Lyrics Embeddings

Lyrics embeddings are generated from proxy text (genre + language descriptions).
They are created automatically on first run. To regenerate v2 embeddings manually:

```bash
python reembed_lyrics_v2.py
```

Output: `data/features/v2/lyrics_embeddings_v2.npy` — (20019, 384)

> **What are proxy lyrics?** Since we do not have actual lyrics for all 20K tracks,
> `src/lyrics.py` generates short genre descriptions (e.g., "rock music", "folk music")
> and embeds them with a multilingual sentence transformer. The v2 fix uses
> language-neutral descriptions only, to prevent label leakage.

---

## 5. Stage 1 — v1 Baseline Pipeline

Run the three task scripts in order. Each trains a VAE, extracts latents, clusters,
evaluates, and saves plots + CSVs.

### Easy Task — BasicVAE + K-Means on MFCC

```bash
python run_easy_task.py --use-real-audio --n-clusters 10 --epochs 100
```

**What it does:**
1. Loads MFCC features (20,020 × 40)
2. Trains `BasicVAE` (MLP: 40 → 512 → 256 → 32 → 256 → 512 → 40)
3. Extracts 32-dim latent vectors
4. Runs K-Means and PCA+K-Means baseline
5. Evaluates: Silhouette, CHI, DBI, ARI, NMI, Purity
6. Saves results to `results/easy/`

**Key outputs:**
```
results/easy/
  easy_results.csv          # metrics summary
  easy_tsne_clusters.png    # t-SNE visualization
  easy_training.png         # loss curves
  easy_comparison_table.png # PCA vs VAE table
```

**Expected results (K=10):** SS ≈ 0.43, CHI ≈ 69,332, ARI ≈ low

---

### Medium Task — ConvVAE + HybridVAE on Mel Spectrograms

```bash
python run_medium_task.py --use-real-audio --n-clusters 10 --epochs 80
```

**What it does:**
1. Loads Mel + lyrics features
2. Trains `ConvVAE` on audio only
3. Trains `HybridVAE` on audio + lyrics fused via concatenation
4. Runs K-Means, Agglomerative (Ward), DBSCAN
5. Evaluates all models and clustering methods
6. Saves results to `results/medium/`

**Key outputs:**
```
results/medium/
  medium_results.csv
  conv_vae_tsne.png
  hybrid_vae_tsne.png
  medium_comparison.png
```

**Expected results:** ConvVAE SS ≈ 0.43; HybridVAE ARI ≈ 0.21 (language signal from lyrics)

---

### Hard Task — BetaVAE + CVAE + MultiModalVAE

```bash
python run_hard_task.py --use-real-audio --n-clusters 10 --epochs 60 --beta 1.0
```

**What it does:**
1. Loads combined audio + lyrics features
2. Trains `BetaVAE` (β=1 for standard VAE regularisation)
3. Trains `CVAE` conditioned on language+genre one-hot vectors (22-dim condition)
4. Trains `MultiModalVAE` with separate audio and lyrics encoders fused at latent level
5. Trains `Autoencoder` baseline (no KL term)
6. Runs K-Means + Spectral Clustering on all latent spaces
7. Full comparison table: all 5 models × all 6 metrics
8. Latent traversal visualizations (BetaVAE dims 0–5)
9. Saves results to `results/hard/`

```bash
# Optional: skip MultiModalVAE for faster testing
python run_hard_task.py --use-real-audio --skip-multimodal
```

**Key outputs:**
```
results/hard/
  all_methods_comparison.csv      # full comparison table
  beta_vae_tsne_clusters.png
  cvae_tsne_language.png
  latent_traversal_dim_0.png      # dims 0–5
  hard_comparison_table.png
```

**Expected results:** CVAE SS ≈ 0.46, MultiModalVAE CHI ≈ 81,020

---

### Multi-K Evaluation (K = 2, 10, 18)

After the three tasks, evaluate all saved models across all cluster counts:

```bash
python run_multi_k_eval.py
```

This loads every checkpoint in `results/models/` and evaluates at K=2 (language binary),
K=10 (all-genre), K=18 (labeled-genre only, excluding MagnaTagATune).

Output: `results/multi_k_eval.csv`

---

### Hyperparameter Grid Search (v1)

```bash
# Full 156-configuration grid (takes several hours on CPU)
python run_finetune.py --use-real-audio

# Quick reduced grid for testing
python run_finetune.py --use-real-audio --quick
```

Grid covers: β ∈ {1.0, 2.0}, d_z ∈ {16, 32}, K ∈ {2, 10, 18}, clustering ∈ {K-Means, GMM}

Output: `results/finetune/finetune_results.csv`, `best_configs_summary.txt`

---

## 6. Stage 2 — Artefact Fixes (v2 Corrections)

Two methodological issues were discovered in the v1 pipeline:

### Issue 1 — Label Leakage in Proxy Lyrics

**Problem:** v1 proxy lyrics included language tags (e.g., "bangla folk music"),
causing the MiniLM embedding to trivially separate English from Bangla *before* the
VAE processed any audio. This inflated HybridVAE ARI from genuine ≈0.48 to artefactual
≈0.99.

**Fix:**
```bash
python reembed_lyrics_v2.py
```

This regenerates lyrics embeddings using language-neutral genre descriptions only
(e.g., "folk music" instead of "bangla folk music").

Output: `data/features/v2/lyrics_embeddings_v2.npy`

---

### Issue 2 — Clip-Length Bias

**Problem:** English clips are 30 s, Bangla clips are 3 s. Mean-pooled spectral features
have lower temporal variance for shorter clips (English MFCC std ≈ 51, Bangla ≈ 27).
This makes language separation appear easier than it truly is.

**Fix:**
```bash
python reextract_features_3s.py
```

This re-extracts a 3 s center window from all tracks, normalizing clip length.

Output: `data/features/v2/` — corrected features for all models

---

## 7. Stage 3 — v2 Corrected Pipeline

After running the artefact fixes above, retrain all models on corrected features:

```bash
# Full v2 pipeline: train → cluster → finetune in sequence
python run_v2_pipeline.py

# If models are already trained, skip training phase
python run_v2_pipeline.py --skip-train

# To skip the quick finetune grid at the end
python run_v2_pipeline.py --skip-finetune
```

**What it does (in sequence):**
1. Loads `data/features/v2/` corrected features
2. Retrains all 6 models (BasicVAE, ConvVAE, HybridVAE, BetaVAE, CVAE, MultiModalVAE)
3. Runs exhaustive clustering: K-Means, GMM, Agglomerative-Ward, Agglomerative-Complete at K ∈ {2, 10, 18}
4. Runs quick finetune grid: β ∈ {1.0, 2.0} × each model
5. Saves comprehensive leaderboard

**Key outputs:**
```
results/v2/
  leaderboard_v2.csv         # all model × method × K combinations ranked
  finetune_quick_v2.csv      # beta sensitivity results
results/models/
  *_v2.pt                    # v2 model checkpoints
```

### Post-Hoc Evaluation (report v2)

To run the full evaluation suite used in the report (can run alongside v2 pipeline):

```bash
python run_report_v2_eval.py
```

Output: `results/report_v2/leaderboard.csv`, `best_by_model_eval.csv`

### Post-Hoc Clustering on v2 Checkpoints

```bash
# Re-cluster all v2 checkpoints with exhaustive algorithm search
python run_posthoc_v2.py

# Only specific models
python run_posthoc_v2.py --models HybridVAE MultiModalVAE

# Clustering only (skip finetune step)
python run_posthoc_v2.py --no-finetune
```

### Generate Report v3 Figures

After all results are in, generate all figures for the v3 report:

```bash
python gen_report_v3_figures.py
```

Figures are saved to `report/figures/` and referenced directly by `report_v3.tex`.

---

## 8. HDBSCAN Analysis

HDBSCAN uses pre-trained Hard task checkpoints — **no retraining needed**.

```bash
python run_hdbscan.py
```

**What it does:**
1. Loads pre-trained `beta_vae_hard.pt`, `cvae_hard.pt`, `multimodal_vae_hard.pt`
2. Reconstructs the same data loaders used during training
3. Extracts 32-dim latent vectors from each model
4. Projects latents to 10 dimensions with UMAP (n_neighbors=30, min_dist=0)
5. Tunes HDBSCAN hyperparameters: min_cluster_size ∈ {50,100,200,300,500}, min_samples ∈ {5,10,None}
6. Evaluates on non-noise points using original 32-dim latent space
7. Saves results to `results/hard/hdbscan_results.csv`

**Key outputs:**
```
results/hard/
  hdbscan_results.csv    # SS, CHI, ARI, NMI, Purity, noise% per model
```

**Expected results:**

| Model | K | SS | ARI | NMI | Noise% |
|-------|---|----|-----|-----|--------|
| BetaVAE+UMAP+HDBSCAN | 4 | 0.339 | **0.255** | 0.349 | 35.3% |
| CVAE+UMAP+HDBSCAN | 207 | 0.093 | 0.027 | 0.250 | 2.1% |
| MMVAE+UMAP+HDBSCAN | 17 | 0.155 | 0.213 | 0.344 | 30.2% |

> BetaVAE+UMAP+HDBSCAN achieves the **highest ARI of any single model** (0.255), finding
> 4 natural density clusters in the latent space without specifying K in advance.

---

## 9. Generating Report v3 PDF

The report is at `report/report_v3.tex`. A `tectonic.exe` LaTeX compiler is bundled
in the project root.

```bash
cd report

# Using bundled tectonic (Windows)
..\tectonic.exe report_v3.tex

# Using system pdflatex (if installed)
pdflatex report_v3.tex
pdflatex report_v3.tex   # run twice to resolve cross-references
```

Output: `report/report_v3.pdf`

> Run pdflatex **twice** — the first pass writes `.aux` files for cross-references;
> the second pass reads them to resolve `\ref{}` and `\cite{}` correctly.

---

## 10. File Reference

### Root-Level Scripts

| Script | Purpose | Key CLI flags |
|--------|---------|---------------|
| `build_dataset.py` | Download GTZAN + BanglaBeats audio | `--extract`, `--gtzan-only`, `--bangla-only` |
| `download_bangla.py` | Download Bangla music from YouTube | `--n N`, `--dry-run` |
| `reextract_features_3s.py` | Re-extract features with 3 s window (v2 fix) | none |
| `reembed_lyrics_v2.py` | Re-embed lyrics without language tags (v2 fix) | none |
| `run_easy_task.py` | Easy: BasicVAE + MFCC + K-Means | `--use-real-audio`, `--n-clusters`, `--epochs`, `--latent-dim`, `--skip-umap` |
| `run_medium_task.py` | Medium: ConvVAE + HybridVAE + Agglomerative/DBSCAN | same as easy + `--genius-csv` |
| `run_hard_task.py` | Hard: BetaVAE + CVAE + MultiModalVAE | same as easy + `--beta`, `--skip-multimodal` |
| `run_finetune.py` | Hyperparameter grid search (v1, 156 configs) | `--use-real-audio`, `--quick` |
| `run_multi_k_eval.py` | Evaluate all checkpoints at K=2/10/18 | none |
| `run_v2_pipeline.py` | Full v2 corrected pipeline | `--skip-train`, `--skip-finetune` |
| `run_posthoc_v2.py` | Post-hoc clustering on v2 checkpoints | `--models`, `--no-finetune` |
| `run_report_v2_eval.py` | Evaluation suite for report v2 | none |
| `run_hdbscan.py` | HDBSCAN on pre-trained Hard task latents | none |
| `gen_report_figures.py` | Generate figures for report v1 | none |
| `gen_report_v2_figures.py` | Generate figures for report v2 | none |
| `gen_report_v3_figures.py` | Generate figures for report v3 | none |

### Report Files

| File | Description |
|------|-------------|
| `report/report_v3.tex` | **Current report** (NeurIPS format, complete) |
| `report/report_v3.pdf` | Compiled PDF |
| `report/report_v2.tex` | Previous version (v2 corrected, no HDBSCAN) |
| `report/report.tex` | Original v1 report |
| `report/neurips_2024.sty` | NeurIPS LaTeX style file |
| `report/figures/` | All figures (PNG/PDF) used in the report |

### Other Key Files

| File | Description |
|------|-------------|
| `requirements.txt` | Python package dependencies |
| `ffmpeg.exe` | FFmpeg binary for audio processing (Windows) |
| `tectonic.exe` | LaTeX compiler (Windows) |
| `src/config.py` | All paths and hyperparameter constants |
| `data/features/combined_metadata.csv` | Ground truth labels for all 20,019 tracks |
| `results/v2/leaderboard_v2.csv` | Best v2 results across all models and K values |
| `results/hard/hdbscan_results.csv` | HDBSCAN evaluation results |

---

## 11. Source Module Reference

### `src/config.py` — Global Configuration

All paths and hyperparameters. Import constants from here:

```python
from src.config import DEVICE, LATENT_DIM, N_CLUSTERS, FEATURES_DIR, MODELS_DIR
```

Key constants:

| Constant | Value | Description |
|----------|-------|-------------|
| `LATENT_DIM` | 32 | VAE latent space dimension |
| `HIDDEN_DIMS` | [512, 256] | Encoder/decoder hidden layer sizes |
| `BATCH_SIZE` | 64 | Training batch size |
| `NUM_EPOCHS` | 100 | Default training epochs |
| `LEARNING_RATE` | 1e-3 | Adam optimizer LR |
| `KL_WEIGHT` | 1.0 | Beta value (β=1 = standard VAE) |
| `N_CLUSTERS` | 5 | Default K for clustering |
| `RANDOM_STATE` | 42 | Reproducibility seed |
| `DEVICE` | auto | `cuda` if available, else `cpu` |

---

### `src/vae.py` — Model Architectures

| Class | Task | Architecture | Input |
|-------|------|-------------|-------|
| `BasicVAE` | Easy | MLP: 40→512→256→32→256→512→40 | MFCC (40-dim) |
| `ConvVAE` | Medium | Conv1d + MLP: 256→32→256 | Mel (256-dim) |
| `BetaVAE` | Hard | Same as BasicVAE + β parameter | Combined (90-dim) |
| `CVAE` | Hard | concat(x, c) encoder: (90+22)→32 | Combined + one-hot |
| `MultiModalVAE` | Hard | Dual encoder: audio(90)+lyrics(384)→32 | Combined + Lyrics |

All models expose `get_latent(x)` → `mu` (deterministic latent vector for clustering).

---

### `src/clustering.py` — Clustering Algorithms

| Function | Algorithm | Key parameter |
|----------|-----------|---------------|
| `kmeans_clustering(features, n_clusters)` | K-Means | `n_clusters` |
| `gmm_clustering(features, n_clusters)` | Gaussian Mixture | `n_clusters` |
| `agglomerative_clustering(features, n_clusters, linkage)` | Hierarchical | `linkage` ∈ {ward, complete, average} |
| `dbscan_clustering(features, eps, min_samples)` | DBSCAN | `eps`, `min_samples` |
| `tune_dbscan(features)` | DBSCAN grid search | auto-tunes eps, min_samples |
| `hdbscan_clustering(features, min_cluster_size)` | HDBSCAN | `min_cluster_size` |
| `tune_hdbscan(features)` | HDBSCAN grid search | auto-tunes min_cluster_size, min_samples |
| `pca_kmeans_baseline(features, n_components, n_clusters)` | PCA + K-Means | `n_components` |
| `find_optimal_k(features, k_range)` | Elbow + Silhouette | `k_range` |

> DBSCAN and HDBSCAN return label `-1` for noise points. Always filter with
> `mask = labels != -1` before computing Silhouette Score.

---

### `src/evaluation.py` — Metrics

```python
from src.evaluation import evaluate_clustering

metrics = evaluate_clustering(
    features=latent_vectors,   # (n_samples, latent_dim)
    labels_pred=cluster_labels,
    labels_true=genre_labels,  # optional
)
# Returns dict with keys:
# silhouette_score, calinski_harabasz_index, davies_bouldin_index,
# adjusted_rand_index, normalized_mutual_info, cluster_purity
```

| Metric | Abbreviation | Range | Better |
|--------|-------------|-------|--------|
| Silhouette Score | SS | −1 to 1 | Higher |
| Calinski-Harabasz Index | CHI | 0 to ∞ | Higher |
| Davies-Bouldin Index | DBI | 0 to ∞ | Lower |
| Adjusted Rand Index | ARI | −1 to 1 | Higher |
| Normalized Mutual Info | NMI | 0 to 1 | Higher |
| Cluster Purity | — | 0 to 1 | Higher |

---

### `src/train.py` — Training Functions

```python
from src.train import train_vae, extract_latent_features

# Train
result = train_vae(model, train_loader, num_epochs=60, model_name="my_model")
# Saves checkpoint to results/models/my_model.pt

# Extract latents
latent = extract_latent_features(result["model"], eval_loader)
# Returns np.ndarray (n_samples, latent_dim)
```

All training functions implement:
- Adam optimizer with ReduceLROnPlateau scheduler
- KL annealing (gradually increases KL weight from 0 to β over first 50% of epochs)
- Early stopping (patience=10 epochs)
- Auto-save best checkpoint to `results/models/`

---

### `src/dataset.py` — Data Loading

```python
from src.dataset import load_features, normalize_features, MusicFeatureDataset

# Load pre-extracted features
features, metadata = load_features("combined")  # or "mfcc", "mel"

# Normalize
features_norm, scaler = normalize_features(features, method="standard", return_scaler=True)

# PyTorch Dataset
dataset = MusicFeatureDataset(features_norm, metadata)
loader = dataset.get_dataloader(batch_size=64, shuffle=True)
```

---

## 12. Results Directory Layout

```
results/
├── models/
│   ├── basic_vae_easy.pt           # Easy task checkpoint
│   ├── conv_vae_medium.pt          # Medium: ConvVAE
│   ├── hybrid_vae_medium.pt        # Medium: HybridVAE
│   ├── beta_vae_hard.pt            # Hard: BetaVAE (β=1)
│   ├── cvae_hard.pt                # Hard: CVAE
│   ├── multimodal_vae_hard.pt      # Hard: MultiModalVAE
│   ├── autoencoder_hard.pt         # Hard: Autoencoder baseline
│   ├── *_v2.pt                     # v2 corrected checkpoints
│   ├── finetune_*.pt               # v1 hyperparameter grid models
│   └── qft_*.pt                    # v2 quick finetune models
├── easy/                           # Easy task plots + CSV
├── medium/                         # Medium task plots + CSV
├── hard/
│   ├── all_methods_comparison.csv  # Full comparison table
│   ├── hdbscan_results.csv         # HDBSCAN evaluation
│   └── *.png                       # Plots
├── v2/
│   ├── leaderboard_v2.csv          # Best v2 results
│   └── finetune_quick_v2.csv       # Beta sensitivity
├── report_v2/
│   ├── leaderboard.csv
│   └── best_by_model_eval.csv
├── finetune/
│   ├── finetune_results.csv        # 156-config grid results
│   └── best_configs_summary.txt
└── multi_k_eval.csv                # K=2/10/18 comparison
```

---

## 13. Troubleshooting

### "ModuleNotFoundError: No module named 'src'"
Run all scripts from the **project root directory**, not from inside `src/`:
```bash
cd VAE-for-Hybrid-Language-Music-Clustering
python run_easy_task.py ...
```

### "FileNotFoundError: combined_features.npy not found"
Features haven't been extracted yet. Run:
```bash
python run_hard_task.py --use-real-audio   # extracts combined features
```
Or place audio in `data/audio/` then run `build_dataset.py --extract`.

### "CUDA out of memory"
Reduce batch size in `src/config.py`:
```python
BATCH_SIZE = 32   # default is 64
```

### HDBSCAN produces all noise (>80% noise points)
This happens when running HDBSCAN on raw high-dimensional features. The `run_hdbscan.py`
script correctly applies UMAP projection first. If you call `hdbscan_clustering()` directly,
always project to ≤10 dimensions first:
```python
import umap
reducer = umap.UMAP(n_components=10, min_dist=0.0, random_state=42)
projected = reducer.fit_transform(latent_32d)
labels = hdbscan_clustering(projected, min_cluster_size=300)
```

### pdflatex not found
Use the bundled `tectonic.exe`:
```bash
cd report
..\tectonic.exe report_v3.tex
```
Or install MiKTeX from https://miktex.org/download.

### Low ARI with high Silhouette Score
This is expected and not a bug. Silhouette measures internal compactness; ARI measures
alignment with genre ground truth. With K=10 clusters and 18 true genre classes, perfect
ARI is geometrically impossible. See Section 7 of the report for full discussion.

---

## Complete Run Order (Fresh Start → Report v3)

```bash
# 1. Setup
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# 2. Get data
python build_dataset.py --extract

# 3. Stage 1 — v1 baseline
python run_easy_task.py   --use-real-audio --n-clusters 10 --epochs 100
python run_medium_task.py --use-real-audio --n-clusters 10 --epochs 80
python run_hard_task.py   --use-real-audio --n-clusters 10 --epochs 60 --beta 1.0
python run_multi_k_eval.py
python run_finetune.py    --use-real-audio --quick

# 4. Stage 2 — fix artefacts
python reextract_features_3s.py
python reembed_lyrics_v2.py

# 5. Stage 3 — v2 corrected
python run_v2_pipeline.py
python run_report_v2_eval.py

# 6. HDBSCAN (uses v1 hard checkpoints, no retraining)
python run_hdbscan.py

# 7. Generate figures + compile report
python gen_report_v3_figures.py
cd report && ..\tectonic.exe report_v3.tex
```

Total runtime on CPU: approximately **6–10 hours** (dominated by v2 pipeline training).
With a GPU: approximately **1–2 hours**.
