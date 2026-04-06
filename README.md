# CSE715: VAE-Based Hybrid Language Music Clustering

**Course:** CSE715 — Neural Networks
**Instructor:** Moin Mostakim
**Deadline:** April 10, 2026

## Overview

Unsupervised learning pipeline using Variational Autoencoders (VAE) to cluster hybrid language (English + Bangla) music tracks. Implements three difficulty tiers:

| Task | Models | Marks |
|------|--------|-------|
| Easy | BasicVAE + K-Means vs. PCA baseline | 20 |
| Medium | ConvVAE + Hybrid VAE (audio+lyrics) + K-Means / Agglomerative / DBSCAN | 25 |
| Hard | BetaVAE + CVAE + MultiModalVAE + Autoencoder baseline | 25 |

## Dataset

| Source | Language | Tracks | Clip Length | Genres |
|--------|----------|--------|-------------|--------|
| GTZAN | English | 1,000 | 30 sec | 10 (blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, rock) |
| MagnaTagATune | English | 9,000 | 29 sec | Multi-tag Western |
| BanglaBeats | Bangla | 10,020 | 3 sec | 8 (Adhunik, Folk, Hip-Hop, Indie, Islamic, Metal, Pop, Rock) |
| **Total** | Both | **20,020** | — | 18+ |

- **Feature types:** MFCC (40-dim), Mel-spectrogram (256-dim)
- **Lyrics embeddings:** `paraphrase-multilingual-MiniLM-L12-v2` (384-dim) — proxy lyrics generated from genre+language metadata

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download & extract dataset

```bash
python build_dataset.py --bangla-only      # Download BanglaBeats (Bangla audio)
python build_dataset.py                    # Also download GTZAN + MagnaTagATune (English)
```

Features are extracted automatically into `data/features/`.

### 3. Run tasks (with real audio)

```bash
# Easy Task (BasicVAE + K-Means)
python run_easy_task.py --use-real-audio --epochs 100

# Medium Task (ConvVAE + HybridVAE + multi-clustering)
python run_medium_task.py --use-real-audio --n-clusters 10 --epochs 80

# Hard Task (BetaVAE + CVAE + MultiModalVAE + full comparison)
python run_hard_task.py --use-real-audio --beta 4.0 --n-clusters 10 --epochs 60
```

### Quick test (synthetic data, no audio required)

```bash
python run_easy_task.py --skip-umap --epochs 50
python run_medium_task.py --n-clusters 5 --epochs 50
python run_hard_task.py --epochs 30 --skip-multimodal
```

## Project Structure

```
project/
├── data/
│   ├── audio/
│   │   ├── english/          # GTZAN WAV files
│   │   └── bangla/           # Bangla WAV files
│   └── features/
│       ├── mfcc_features.npy
│       ├── mel_features.npy
│       ├── combined_features.npy
│       └── lyrics_embeddings/
│           └── lyrics_embeddings.npy
├── src/
│   ├── config.py             # Paths, hyperparameters
│   ├── vae.py                # BasicVAE, ConvVAE, BetaVAE, CVAE, MultiModalVAE
│   ├── dataset.py            # Feature extraction, MusicFeatureDataset, MultiModalMusicDataset
│   ├── train.py              # train_vae, train_cvae, train_multimodal_vae
│   ├── clustering.py         # K-Means, Agglomerative, DBSCAN, tune_dbscan
│   ├── evaluation.py         # All 6 clustering metrics
│   ├── visualization.py      # t-SNE, UMAP, latent traversal, reconstruction plots
│   ├── baselines.py          # Autoencoder, spectral_clustering, direct_feature_kmeans
│   └── lyrics.py             # LaBSE embedding generation
├── results/
│   ├── easy/                 # Easy task outputs
│   ├── medium/               # Medium task outputs
│   └── hard/                 # Hard task outputs (all_methods_comparison.csv)
├── run_easy_task.py
├── run_medium_task.py
├── run_hard_task.py
├── build_dataset.py
└── requirements.txt
```

## Models

| Model | Architecture | Task | Input |
|-------|-------------|------|-------|
| `BasicVAE` | MLP encoder/decoder | Easy | MFCC features (40-dim) |
| `ConvVAE` | Conv1d encoder + ConvTranspose1d decoder | Medium | Mel features (256-dim) |
| `BetaVAE` | BasicVAE with β > 1 KL weight | Hard | Combined features |
| `CVAE` | Encoder/decoder conditioned on language label | Hard | Combined features + one-hot |
| `MultiModalVAE` | Dual encoder (audio + lyrics), single decoder | Hard | Audio + LaBSE lyrics |
| `Autoencoder` | Deterministic MLP (no KL term) | Hard baseline | Combined features |

## Evaluation Metrics

| Metric | Range | Better | Used In |
|--------|-------|--------|---------|
| Silhouette Score | [-1, 1] | Higher | Easy, Medium, Hard |
| Calinski-Harabasz Index | [0, ∞) | Higher | Easy |
| Davies-Bouldin Index | [0, ∞) | Lower | Medium |
| Adjusted Rand Index (ARI) | [-1, 1] | Higher | Medium, Hard |
| Normalized Mutual Info (NMI) | [0, 1] | Higher | Hard |
| Cluster Purity | [0, 1] | Higher | Hard |

## Command-line Options

### `run_easy_task.py`
| Flag | Default | Description |
|------|---------|-------------|
| `--use-real-audio` | False | Use real audio instead of synthetic |
| `--n-clusters` | 5 | Number of clusters |
| `--latent-dim` | 32 | Latent space dimensionality |
| `--epochs` | 100 | Training epochs |
| `--skip-umap` | False | Skip UMAP visualization |

### `run_medium_task.py`
Same as Easy, plus:

| Flag | Default | Description |
|------|---------|-------------|
| `--genius-csv` | None | Path to Genius lyrics CSV |

### `run_hard_task.py`
Same as Medium, plus:

| Flag | Default | Description |
|------|---------|-------------|
| `--beta` | 4.0 | β value for BetaVAE |
| `--skip-multimodal` | False | Skip MultiModalVAE |

## Key Results (Real Data, 20,020 Tracks)

| Task | Best Method | Silhouette | CHI | vs. PCA |
|------|-------------|-----------|-----|---------|
| Easy | MLP-VAE + K-Means | 0.4286 | 69,332 | +307% SS |
| Medium | ConvVAE + K-Means | 0.4294 | 42,746 | +179% SS |
| Hard | MultiModalVAE + K-Means | **0.4974** | **81,020** | +223% SS |

Language clustering (English vs. Bangla): 7/10 VAE clusters are language-pure (>80%), purity 87-92%.

For hyperparameter sensitivity analysis across 1,800+ configurations:

```bash
python run_finetune.py --use-real-audio --quick   # ~1-2 hours
python run_finetune.py --use-real-audio            # full grid, ~8 hours
```

Results saved to `results/finetune/finetune_results.csv`.

## Reproducibility

All scripts use `RANDOM_STATE=42` for numpy, PyTorch, and sklearn operations. Results can be reproduced by running the scripts with identical flags.

## References

1. Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes. *arXiv:1312.6114*
2. Higgins, I., et al. (2017). beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. *ICLR 2017*
3. Sohn, K., Lee, H., & Yan, X. (2015). Learning Structured Output Representation using Deep Conditional Generative Models. *NeurIPS 2015*
4. McFee, B., et al. (2015). librosa: Audio and Music Signal Analysis in Python. *SciPy 2015*
5. Reimers, N., & Gurevych, I. (2020). Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation. *EMNLP 2020* (LaBSE)
