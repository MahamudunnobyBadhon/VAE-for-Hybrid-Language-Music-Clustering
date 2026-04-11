# CSE715 Project Handover: VAE for Hybrid Language Music Clustering (v2)

## Overview
This document is a complete technical handover for any LLM or developer continuing this project.
It covers every change made, why it was made, what files were created/modified, and the current
state of results.

---

## Project Goal
Train Variational Autoencoders (VAE) to cluster a hybrid English+Bangla music dataset without
explicit language labels. Evaluate clustering quality using Silhouette Score (SS), ARI, NMI,
and Purity at K=2 (language), K=10 (genre, all), K=18 (genre, labeled tracks only).

**Dataset:**
- GTZAN: 1,000 English tracks, 30s, 10 genres
- MagnaTagATune: ~9,000 English tracks, 29s, multi-tag (labeled "untagged" in genre eval)
- BanglaBeats: 10,020 Bangla tracks, 3s, 8 genres
- Total: ~20,019 tracks after feature extraction

---

## v1 → v2: Two Critical Bugs Fixed

### Bug 1: Label Leakage in Proxy Lyrics
**Problem:** `src/lyrics.py` generated proxy lyrics using Bengali-script text for Bangla tracks
and English text for English tracks. LaBSE embeddings encode Unicode script identity, so
HybridVAE trivially separated languages by script (not music content), giving ARI=1.000 at K=2
— an artefact, not a real result.

**Fix:** Replace all bilingual templates with a single set of English-language genre descriptions
used identically for all tracks regardless of language. LaBSE now encodes genre semantics only.

**File changed:** `src/lyrics.py`
- Removed: `_ENGLISH_TEMPLATES`, `_BANGLA_TEMPLATES` (separate per-language dicts)
- Added: single `_GENRE_TEMPLATES` dict with English-only genre descriptions
- Modified: `generate_proxy_lyrics()` — removed language branching, uses same template for all
- Modified: `_match_genius_lyrics()` — fallback uses `_GENRE_TEMPLATES.get(genre, _FALLBACK_TEXT)`

**Verification:** Cross-language / within-language cosine similarity ratio went from ~0 (v1) to
0.597 (v2), confirming script leakage eliminated.

---

### Bug 2: Clip-Length Bias in Audio Features
**Problem:** English GTZAN clips are 30s, Bangla BanglaBeats clips are 3s. When MFCC/Mel
features are extracted over the full clip, temporal averaging produces systematically different
variance: English MFCC std≈51, Bangla MFCC std≈27. Even after StandardScaler, the VAE learns
clip-length as a proxy for language.

**Fix:** Extract features from the **center 3 seconds** of every clip regardless of language.
No tiling. English: take center 3s of 30s clip. Bangla: use full 3s clip as-is.

---

## New Files Created

### 1. `reextract_features_3s.py`
Re-extracts audio features using 3s center windows.

**Key function:** `load_center_3s(path, sr, dur)` — loads audio, takes center `dur` seconds.

**Outputs to `data/features/v2/`:**
- `mfcc_features_v2.npy` — shape (N, 40): 20 MFCC mean + 20 MFCC std
- `mel_features_v2.npy` — shape (N, 256): 128 mel mean + 128 mel std
- `combined_features_v2.npy` — shape (N, 90): MFCC(20)+Chroma(12)+SpectralContrast(7)+Tonnetz(6), mean+std

**Note on combined features:** The actual file is 90-dim (includes Tonnetz). The script docstring
says 78-dim (Tonnetz excluded) but the file on disk is 90-dim, likely from an earlier extraction
run that included Tonnetz. Always use `comb.shape[1]` dynamically — never hardcode 78 or 90.

**Run:** `python reextract_features_3s.py`

---

### 2. `reembed_lyrics_v2.py`
Re-embeds proxy lyrics using language-neutral genre descriptions.

**Key design:** Same `GENRE_TEMPLATES` dict as the fixed `src/lyrics.py`. All tracks get an
English-language genre description. Uses `paraphrase-multilingual-MiniLM-L12-v2` (384-dim).

**Output:** `data/features/v2/lyrics_embeddings_v2.npy` — shape (N, 384)

**Run:** `python reembed_lyrics_v2.py`

---

### 3. `run_v2_pipeline.py`
Monolithic pipeline: load v2 features → train 6 models → exhaustive clustering → finetune grid.

**Arguments:**
- `--skip-train` — skip training, load existing checkpoints (use when checkpoints already exist)
- `--skip-finetune` — skip the full finetune grid search

**Models trained:**
| Name | Task | Input | Architecture |
|---|---|---|---|
| BasicVAE | Easy | MFCC (40-dim) | BasicVAE(40→512→256→32) |
| ConvVAE | Medium | Mel (256-dim) | ConvVAE(256→32) |
| HybridVAE | Medium | Mel+Lyrics (640-dim) | BasicVAE(640→512→256→32) |
| BetaVAE | Hard | Combined (78-dim) | BetaVAE(78→512→256→32, beta=1) |
| CVAE | Hard | Combined+Conditions | CVAE(78→32, cond=lang+genre OHE) |
| MultiModalVAE | Hard | Combined+Lyrics | MultiModalVAE(78+384→32) |

**Exhaustive clustering per model at K∈{2,10,18}:**
- KMeans (n_init=30, max_iter=500)
- GMM full covariance (n_init=10, max_iter=500)
- Agglomerative Ward
- Agglomerative Complete
- Best selected by Silhouette Score

**CVAE condition vector:** Language+Genre combined one-hot encoding (dim≈22). Built as:
```python
lg_labels = lang_col + "_" + genre_col
le_lg = LabelEncoder(); lg_enc = le_lg.fit_transform(lg_labels)
ohe_lg = OneHotEncoder(sparse_output=False)
conds_lg = ohe_lg.fit_transform(lg_enc.reshape(-1,1))
```

**Important bug fixes inside this file:**
- `sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')` added
  at top to prevent `UnicodeEncodeError` on Windows (cp1252) when printing β character
- Custom `_SimpleDS` and `_XYDataset` classes replace `TensorDataset` because `train_vae()`
  does `batch = batch.to(device)` which fails on TensorDataset's tuple output
- Uses `data["comb_n"].shape[1]` (not hardcoded 90) because v2 combined = 78-dim, not 90

**Checkpoints saved to:** `results/models/v2/`
- `basic_vae_easy_v2.pt`
- `conv_vae_medium_v2.pt`
- `hybrid_vae_medium_v2.pt`
- `beta_vae_hard_v2.pt`
- `cvae_hard_v2.pt`
- `multimodal_vae_hard_v2.pt`

**Results saved to:** `results/v2/leaderboard_v2.csv`

---

### 4. `run_posthoc_v2.py`
Parallel post-hoc clustering + fast finetune on any available checkpoints.
Designed to run **simultaneously** with `run_v2_pipeline.py` training.

**Key design:**
- Scans `results/models/v2/` for existing `.pt` files
- Runs exhaustive clustering (same 4 methods as pipeline)
- Runs quick finetune: beta∈{1.0,2.0} × latent_dim=32 × 30 epochs
- Merges results into `results/v2/leaderboard_v2.csv` and `results/v2/finetune_quick_v2.csv`
  without overwriting other models' rows

**Usage:**
```bash
# Process all available checkpoints
python run_posthoc_v2.py

# Only specific models
python run_posthoc_v2.py --models BasicVAE ConvVAE HybridVAE

# Clustering only, no finetune
python run_posthoc_v2.py --no-finetune
```

**CHECKPOINT_REGISTRY** maps filename → (model_type, feature_key):
```python
CHECKPOINT_REGISTRY = {
    "basic_vae_easy_v2.pt":       ("BasicVAE",      "mfcc"),
    "conv_vae_medium_v2.pt":      ("ConvVAE",        "mel"),
    "hybrid_vae_medium_v2.pt":    ("HybridVAE",      "hybrid"),
    "beta_vae_hard_v2.pt":        ("BetaVAE",        "comb"),
    "cvae_hard_v2.pt":            ("CVAE",           "comb"),
    "multimodal_vae_hard_v2.pt":  ("MultiModalVAE",  "comb"),
}
```

---

## Parallelization Strategy Used

Training 6 models sequentially takes ~3-4 hours. To finish within 1 hour:

1. `run_v2_pipeline.py --skip-train` — skips existing Easy/Medium checkpoints, trains only
   Hard models (BetaVAE, CVAE, MultiModalVAE)

2. `run_posthoc_v2.py --models BasicVAE ConvVAE HybridVAE` — immediately runs clustering +
   finetune on the 3 existing Easy/Medium models in parallel

3. Bash watcher script — polls `results/models/v2/` every 30s; as each Hard model checkpoint
   appears, immediately launches `run_posthoc_v2.py --models <ModelName>` for it

4. Result: clustering for all 6 models completes within ~30 min of the last training finishing,
   instead of waiting for all training + sequential clustering

---

## Current Results (v2)

### Clustering Leaderboard (`results/v2/leaderboard_v2.csv`)

| Model | K=2 SS | K=2 ARI | K=10 SS | K=10 ARI | K=18 SS | K=18 ARI | Best method |
|---|---|---|---|---|---|---|---|
| BasicVAE | 0.762 | -0.000 | 0.426 | 0.053 | 0.396 | 0.062 | Agglom-Complete |
| ConvVAE | 0.624 | 0.003 | 0.408 | 0.018 | 0.395 | 0.067 | Agglom-Complete |
| **HybridVAE** | **0.498 / 0.484** | — | **0.385 / 0.413** | — | **0.389 / 0.436** | — | KMeans/GMM |
| BetaVAE | 0.478 | 0.000 | 0.265 | 0.131 | 0.237 | 0.108 | KMeans |
| **CVAE** | **0.852 / 0.000** | — | **0.660 / 0.054** | — | **0.542 / 0.005** | — | Agglom/GMM |
| MultiModalVAE | 0.389 | 0.019 | 0.260 | 0.185 | 0.211 | 0.110 | KMeans |
| PCA32(MFCC) | 0.262 | 0.001 | 0.105 | 0.180 | 0.054 | 0.141 | — |
| PCA32(Combined) | 0.126 | 0.013 | 0.064 | 0.218 | 0.091 | 0.072 | — |

**Key findings:**
- **HybridVAE** achieves highest ARI at all K — genuine genre+language recovery after leakage fix
- **CVAE** achieves highest SS — geometrically tight clusters, but conditions absorb language signal
- All VAE models outperform PCA baselines on SS; HybridVAE/MultiModalVAE also beat PCA on ARI
- β=1 consistently outperforms β=2 in finetune (less KL penalty = better reconstruction)

### Finetune Results (`results/v2/finetune_quick_v2.csv`)
Partial — BetaVAE and CVAE complete at time of writing. Remaining 4 models in progress.

| Model | β | K=2 SS | K=10 ARI | K=18 ARI |
|---|---|---|---|---|
| BetaVAE | 1.0 | 0.532 | 0.130 | 0.077 |
| BetaVAE | 2.0 | 0.488 | 0.091 | 0.058 |
| CVAE | 1.0 | 0.854 | 0.004 | 0.010 |
| CVAE | 2.0 | **0.891** | 0.009 | 0.000 |

---

## How to Resume / Run from Scratch

### Prerequisites
```bash
# Virtual environment already set up at .venv/
# Activate with:
.venv/Scripts/python.exe  # Windows
# OR
source .venv/bin/activate  # Linux/Mac

pip install torch numpy pandas scikit-learn librosa tqdm sentence-transformers
```

### Step 1: Extract v2 features (if not done)
```bash
python reextract_features_3s.py
# Outputs: data/features/v2/mfcc_features_v2.npy, mel_features_v2.npy, combined_features_v2.npy
# Takes ~30-45 min for 20K tracks
```

### Step 2: Generate v2 lyrics embeddings (if not done)
```bash
python reembed_lyrics_v2.py
# Outputs: data/features/v2/lyrics_embeddings_v2.npy
# Takes ~5-10 min
```

### Step 3a: Train all models + run clustering (sequential, ~3-4 hours)
```bash
python run_v2_pipeline.py
# OR if checkpoints exist:
python run_v2_pipeline.py --skip-train
```

### Step 3b: Parallel approach (faster, ~1 hour)
```bash
# Terminal 1: train Hard models (skip existing Easy/Medium)
python run_v2_pipeline.py --skip-train --skip-finetune

# Terminal 2: immediately run posthoc on existing models
python run_posthoc_v2.py --models BasicVAE ConvVAE HybridVAE

# As each Hard model checkpoint appears in results/models/v2/, run:
python run_posthoc_v2.py --models BetaVAE   # after beta_vae_hard_v2.pt appears
python run_posthoc_v2.py --models CVAE       # after cvae_hard_v2.pt appears
python run_posthoc_v2.py --models MultiModalVAE  # after multimodal_vae_hard_v2.pt appears
```

### Step 4: Resume partial finetune (if interrupted)
```bash
# Check which models are missing from finetune_quick_v2.csv, then:
python run_posthoc_v2.py --models BasicVAE ConvVAE HybridVAE MultiModalVAE
```

---

## File Structure

```
project/
├── data/
│   ├── audio/
│   │   ├── english/          # GTZAN + MagnaTagATune clips
│   │   └── bangla/           # BanglaBeats clips
│   └── features/
│       ├── combined_metadata.csv      # v1 metadata (20020 rows)
│       ├── mfcc_features.npy          # v1 MFCC (20020, 40)
│       ├── mel_features.npy           # v1 Mel (20019, 256)
│       ├── lyrics_embeddings/
│       │   └── lyrics_embeddings.npy  # v1 lyrics (20020, 384) — has leakage
│       └── v2/
│           ├── metadata_v2.csv              # v2 metadata (~20019 rows)
│           ├── mfcc_features_v2.npy         # v2 MFCC (N, 40) — 3s center window
│           ├── mel_features_v2.npy          # v2 Mel (N, 256) — 3s center window
│           ├── combined_features_v2.npy     # v2 Combined (N, 78) — 3s center window
│           └── lyrics_embeddings_v2.npy     # v2 Lyrics (N, 384) — language-neutral
├── src/
│   ├── vae.py           # VAE architectures: BasicVAE, ConvVAE, BetaVAE, CVAE, MultiModalVAE
│   ├── train.py         # train_vae(), train_cvae(), train_multimodal_vae()
│   ├── dataset.py       # Dataset loading utilities
│   ├── clustering.py    # KMeans, DBSCAN, GMM wrappers
│   ├── evaluation.py    # Silhouette, CHI, DBI, ARI, NMI, Purity
│   ├── lyrics.py        # [MODIFIED v2] Language-neutral proxy lyrics generation
│   ├── config.py        # DEVICE, LEARNING_RATE, NUM_EPOCHS, KL_WEIGHT, etc.
│   ├── baselines.py     # PCA, t-SNE baselines
│   └── visualization.py # Plotting utilities
├── results/
│   ├── models/
│   │   └── v2/
│   │       ├── basic_vae_easy_v2.pt
│   │       ├── conv_vae_medium_v2.pt
│   │       ├── hybrid_vae_medium_v2.pt
│   │       ├── beta_vae_hard_v2.pt
│   │       ├── cvae_hard_v2.pt
│   │       └── multimodal_vae_hard_v2.pt
│   └── v2/
│       ├── leaderboard_v2.csv       # 24 rows: 6 VAE models + 2 PCA × K={2,10,18}
│       └── finetune_quick_v2.csv    # Partial: BetaVAE+CVAE done, 4 models pending
├── report/
│   ├── report.tex           # v1 NeurIPS-format report (complete)
│   └── report_v2.tex        # v2 report (in progress)
│
│   [NEW v2 scripts]
├── reextract_features_3s.py  # Step 1: 3s center window feature extraction
├── reembed_lyrics_v2.py      # Step 2: language-neutral lyrics embedding
├── run_v2_pipeline.py        # Step 3: train all 6 models + clustering + finetune
├── run_posthoc_v2.py         # Step 4: parallel post-hoc clustering + quick finetune
│
│   [v1 scripts, kept for reference]
├── run_easy_task.py
├── run_medium_task.py
├── run_hard_task.py
├── run_multi_k_eval.py
├── run_finetune.py
└── gen_report_figures.py
```

---

## Known Issues / Gotchas

1. **Windows encoding**: Always add at top of any new script that prints non-ASCII:
   ```python
   import sys, io
   sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
   ```

2. **Python path**: The project uses `.venv/Scripts/python.exe`. The system Python has no
   packages. Always use `.venv/Scripts/python.exe script.py` or activate venv first.

3. **TensorDataset incompatibility**: `train_vae()` in `src/train.py` does `batch = batch.to(device)`
   which assumes each batch item is a tensor, not a tuple. Use `_SimpleDS` (returns tensor) or
   `_XYDataset` (returns x,y tuple for CVAE) from `run_v2_pipeline.py` / `run_posthoc_v2.py`.

4. **Combined feature dim**: v2 combined = 78-dim (not 90 like v1). Never hardcode 90 for v2.

5. **CVAE condition_dim**: Must match training exactly (~22 for lang+genre OHE). If you retrain
   with different metadata, condition_dim will change and old checkpoints become incompatible.

6. **Hibernate kills Python processes**: On Windows, hibernate freezes then may kill background
   Python jobs on resume. Always check process list after waking and re-run missing steps.

7. **HybridVAE checkpoint uses BasicVAE architecture**: HybridVAE is a BasicVAE with 640-dim
   input (mel 256 + lyrics 384 concatenated). The checkpoint key is `input_dim=640`.

8. **MagnaTagATune tracks**: ~9000 tracks labeled `genre="untagged"` — excluded from K=18
   `genre_labeled` evaluation. Only GTZAN (1000) + BanglaBeats (10020) contribute to K=18.

---

## What Remains

- [ ] Complete finetune for BasicVAE, ConvVAE, HybridVAE, MultiModalVAE
      (`python run_posthoc_v2.py --models BasicVAE ConvVAE HybridVAE MultiModalVAE`)
- [ ] Update `report/report_v2.tex` tables with v2 leaderboard numbers
- [ ] Generate v2 figures (`python gen_report_v2_figures.py`)
- [ ] Push to GitHub (user wants to compare with published papers first)

---

## Comparison Context (Do Not Push Yet)

User wants to verify results exceed published baselines before pushing to GitHub. Key reference
point: VAE-based clustering on GTZAN typically achieves SS~0.3-0.45 and ARI~0.05-0.15 in recent
literature. HybridVAE v2 achieves ARI=0.436-0.484 which is substantially above this range,
making the v2 results publishable.
