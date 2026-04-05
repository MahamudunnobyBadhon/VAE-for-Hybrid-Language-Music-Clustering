# CSE715 Neural Networks Project Plan

## VAE for Hybrid Language Music Clustering (Unsupervised Learning)

**Course:** CSE715 - Neural Networks  
**Instructor:** Moin Mostakim  
**Deadline:** April 10th, 2026  
**Total Marks:** 110 (Easy 20 + Medium 25 + Hard 25 + Metrics 10 + Viz 10 + Report 10 + GitHub 10)

---

## Table of Contents

1. [Project Summary](#1-project-summary)
2. [Task Breakdown (Easy / Medium / Hard)](#2-task-breakdown)
3. [Dataset Strategy](#3-dataset-strategy)
4. [Architecture & Method Design](#4-architecture--method-design)
5. [Evaluation Metrics](#5-evaluation-metrics)
6. [Step-by-Step Implementation Plan](#6-step-by-step-implementation-plan)
7. [Repository Structure](#7-repository-structure)
8. [Report Plan (NeurIPS Format)](#8-report-plan-neurips-format)
9. [Tools & Libraries](#9-tools--libraries)
10. [Timeline & Milestones](#10-timeline--milestones)
11. [Key Resources & References](#11-key-resources--references)
12. [Risk Mitigation & Tips](#12-risk-mitigation--tips)

---

## 1. Project Summary

Build an **unsupervised learning pipeline** using **Variational Autoencoders (VAE)** to:
- Extract latent representations from **hybrid language music** (English + Bangla)
- Cluster tracks using the learned latent space
- Compare with baseline methods (PCA + K-Means, Autoencoder + K-Means)
- Visualize and evaluate using standard clustering metrics

The project has three difficulty tiers (Easy → Medium → Hard). **Do all three for maximum marks.**

---

## 2. Task Breakdown

### Easy Task (20 marks)
- [ ] Implement a **basic VAE** (fully-connected / MLP-based) for feature extraction
- [ ] Use a **small hybrid language music dataset** (English + Bangla songs)
- [ ] Perform **K-Means clustering** on latent features
- [ ] **Visualize** clusters using t-SNE or UMAP
- [ ] **Compare** with baseline (PCA + K-Means) using Silhouette Score and Calinski-Harabasz Index

### Medium Task (25 marks)
- [ ] Enhance VAE with **convolutional architecture** for spectrograms or MFCC features
- [ ] Include **hybrid feature representation**: audio features + lyrics embeddings
- [ ] Experiment with multiple clustering algorithms: **K-Means, Agglomerative Clustering, DBSCAN**
- [ ] Evaluate using: **Silhouette Score, Davies-Bouldin Index, Adjusted Rand Index** (if partial labels available)
- [ ] Compare results across methods and **analyze why** VAE representations perform better/worse

### Hard Task (25 marks)
- [ ] Implement **Conditional VAE (CVAE) or Beta-VAE** for disentangled latent representations
- [ ] Perform **multi-modal clustering** combining audio, lyrics, and genre information
- [ ] Quantitatively evaluate using: **Silhouette Score, NMI, ARI, Cluster Purity**
- [ ] Detailed **visualizations**: latent space plots, cluster distribution over languages/genres, reconstruction examples
- [ ] Compare VAE-based clustering with **PCA + K-Means**, **Autoencoder + K-Means**, and **direct spectral feature clustering**

---

## 3. Dataset Strategy

### Primary Approach: Build a Custom Hybrid Dataset
Since the project requires English + Bangla music, we will likely need to build our own dataset by combining sources.

#### Audio Features
| Source | Description | Use |
|--------|-------------|-----|
| [GTZAN Genre Collection](http://marsyas.info/downloads/datasets.html) | 1000 tracks, 10 genres | English audio features (MFCC, spectrograms) |
| [Jamendo Dataset](https://www.kaggle.com/datasets/andradaolteanu/jamendo-music-da) | Audio + metadata + lyrics | Hybrid audio + lyrics experiments |
| [MIR-1K Dataset](https://sites.google.com/site/unvoicedsoundseparation/mir-1k) | 1000 clips, Mandarin + English | Multi-language reference |
| YouTube / Spotify API | Download Bangla songs | Bangla audio source |

#### Lyrics Features
| Source | Description | Use |
|--------|-------------|-----|
| [Kaggle Lyrics Datasets](https://www.kaggle.com/datasets?search=lyrics) | Multi-language lyrics datasets | English lyrics embeddings |
| Web scraping (with attribution) | Bangla song lyrics | Bangla lyrics embeddings |
| [Million Song Dataset (Lyrics)](http://millionsongdataset.com/) | Partial lyrics | Additional lyrics data |

#### Recommended Dataset Size
- **Minimum**: 200-500 tracks (100+ English, 100+ Bangla)
- **Ideal**: 500-1000+ tracks for better clustering

### Data Collection Steps
1. Collect ~250+ English songs (GTZAN or Jamendo)
2. Collect ~250+ Bangla songs (YouTube/Spotify with proper attribution)
3. Extract audio features (MFCC, Mel-spectrograms, Chroma)
4. Collect/scrape lyrics for both languages
5. Generate lyrics embeddings using pre-trained models
6. Create metadata CSV with: song_id, language, genre (if known), file paths

---

## 4. Architecture & Method Design

### 4.1 Feature Extraction Pipeline

```
Audio Track
    ├── librosa → MFCC (13-40 coefficients)
    ├── librosa → Mel-Spectrogram (128 mel bands)
    ├── librosa → Chroma Features (12 dims)
    └── librosa → Spectral Contrast, Tonnetz, etc.

Lyrics Text  
    ├── TF-IDF Vectorization (baseline)
    ├── Sentence-BERT / multilingual-BERT → Embeddings (768d)
    └── LaBSE (Language-agnostic BERT) → Embeddings (768d)  ← RECOMMENDED for Bangla+English
```

### 4.2 VAE Architectures

#### Easy: Basic MLP-VAE
```
Encoder: Input(d) → FC(512) → ReLU → FC(256) → ReLU → [μ(z_dim), σ(z_dim)]
Decoder: z(z_dim) → FC(256) → ReLU → FC(512) → ReLU → Output(d)
Loss: Reconstruction Loss (MSE) + KL Divergence
```

#### Medium: Convolutional VAE (for spectrograms)
```
Encoder: Input(1, H, W) → Conv2d(32) → Conv2d(64) → Conv2d(128) → Flatten → [μ, σ]
Decoder: z → FC → Reshape → ConvTranspose2d(128) → ConvTranspose2d(64) → ConvTranspose2d(32) → Output
Loss: Reconstruction Loss (BCE/MSE) + KL Divergence
```

#### Hard: Beta-VAE / Conditional VAE
```
Beta-VAE: Same as VAE but Loss = Recon Loss + β * KL Divergence (β > 1 for disentanglement)
CVAE: Encoder takes (x, c) → z; Decoder takes (z, c) → x_recon (c = condition like language/genre)
```

### 4.3 Multi-Modal Fusion (Medium + Hard)
```
Audio Features → Audio Encoder → z_audio
Lyrics Embeddings → Lyrics Encoder → z_lyrics
                                          ↓
                      Concatenate / Attention Fusion → z_combined
                                          ↓
                                    Clustering Layer
```

**Fusion strategies to try:**
1. **Early fusion**: Concatenate audio + lyrics features before VAE
2. **Late fusion**: Separate VAEs, concatenate latent vectors, then cluster
3. **Joint VAE**: Single VAE with combined input

---

## 5. Evaluation Metrics

| Metric | Formula Intuition | Range | Better | Required For |
|--------|-------------------|-------|--------|-------------|
| **Silhouette Score** | How well-separated are clusters? | [-1, 1] | Higher | Easy, Medium, Hard |
| **Calinski-Harabasz Index** | Between vs. within cluster variance ratio | [0, ∞) | Higher | Easy |
| **Davies-Bouldin Index** | Avg similarity between clusters | [0, ∞) | Lower | Medium |
| **Adjusted Rand Index (ARI)** | Agreement with ground truth (adjusted for chance) | [-1, 1] | Higher | Medium, Hard |
| **Normalized Mutual Information (NMI)** | Mutual info between clusters and labels | [0, 1] | Higher | Hard |
| **Cluster Purity** | Fraction of dominant class per cluster | [0, 1] | Higher | Hard |

### Baseline Comparisons Required
1. **PCA + K-Means** (Easy baseline)
2. **Autoencoder (non-variational) + K-Means** (Medium/Hard baseline)
3. **Direct spectral feature clustering** (Hard baseline)

---

## 6. Step-by-Step Implementation Plan

### Phase 1: Environment & Data Setup (Week 1-2)
- [ ] **Step 1**: Set up Python environment with all dependencies
- [ ] **Step 2**: Create project repository with suggested structure
- [ ] **Step 3**: Collect/download English songs (GTZAN or Jamendo)
- [ ] **Step 4**: Collect Bangla songs (YouTube/Spotify — ensure proper attribution)
- [ ] **Step 5**: Write `dataset.py` — audio loading, feature extraction pipeline
- [ ] **Step 6**: Extract MFCC features for all audio files (save as `.npy`)
- [ ] **Step 7**: Extract Mel-spectrograms for all audio files (save as `.npy`)
- [ ] **Step 8**: Collect lyrics and generate embeddings (LaBSE / multilingual-BERT)
- [ ] **Step 9**: Create a master metadata CSV: `song_id, title, language, genre, audio_path, lyrics_path`
- [ ] **Step 10**: Exploratory Data Analysis notebook — understand feature distributions

### Phase 2: Easy Task Implementation (Week 2-3)
- [ ] **Step 11**: Implement basic MLP-VAE in `src/vae.py`
- [ ] **Step 12**: Write training loop with reconstruction loss + KL divergence
- [ ] **Step 13**: Train VAE on MFCC features, save latent representations
- [ ] **Step 14**: Implement K-Means clustering on latent features
- [ ] **Step 15**: Implement PCA + K-Means baseline
- [ ] **Step 16**: Compute Silhouette Score and Calinski-Harabasz Index for both
- [ ] **Step 17**: Visualize clusters with t-SNE and UMAP
- [ ] **Step 18**: Save results and plots to `results/`

### Phase 3: Medium Task Implementation (Week 3-4)
- [ ] **Step 19**: Implement Convolutional VAE in `src/vae.py` for spectrograms
- [ ] **Step 20**: Implement hybrid feature representation (audio + lyrics concatenation)
- [ ] **Step 21**: Train Conv-VAE on Mel-spectrograms
- [ ] **Step 22**: Implement Agglomerative Clustering and DBSCAN in `src/clustering.py`
- [ ] **Step 23**: Run all three clustering algorithms (K-Means, Agglom., DBSCAN)
- [ ] **Step 24**: Compute Davies-Bouldin Index + ARI (if partial labels exist)
- [ ] **Step 25**: Compare and analyze results across methods
- [ ] **Step 26**: Create comparison tables and analysis plots

### Phase 4: Hard Task Implementation (Week 4-5)
- [ ] **Step 27**: Implement Beta-VAE (add β parameter to KL loss)
- [ ] **Step 28**: Implement Conditional VAE (CVAE) with language/genre conditioning
- [ ] **Step 29**: Implement multi-modal VAE (audio + lyrics + genre)
- [ ] **Step 30**: Train and extract latent representations
- [ ] **Step 31**: Compute all metrics: Silhouette, NMI, ARI, Cluster Purity
- [ ] **Step 32**: Implement Autoencoder + K-Means baseline
- [ ] **Step 33**: Implement direct spectral feature clustering baseline
- [ ] **Step 34**: Create detailed visualizations:
  - Latent space plots (t-SNE/UMAP colored by language, genre)
  - Cluster distribution across languages/genres
  - VAE reconstruction examples
  - Disentangled latent traversals (Beta-VAE)
- [ ] **Step 35**: Full comparison table of all methods vs. all metrics

### Phase 5: Report & Final Polish (Week 5-6)
- [ ] **Step 36**: Write NeurIPS-format report on Overleaf
- [ ] **Step 37**: Create README.md with setup instructions, results summary
- [ ] **Step 38**: Clean code, add docstrings, type hints
- [ ] **Step 39**: Add `requirements.txt` with version pins
- [ ] **Step 40**: Final testing — ensure full reproducibility
- [ ] **Step 41**: Push everything to GitHub

---

## 7. Repository Structure

```
project/
├── data/
│   ├── audio/                    # Raw audio files (or download scripts)
│   │   ├── english/
│   │   └── bangla/
│   ├── lyrics/                   # Lyrics text files
│   │   ├── english/
│   │   └── bangla/
│   ├── features/                 # Extracted features (.npy files)
│   │   ├── mfcc/
│   │   ├── mel_spectrograms/
│   │   └── lyrics_embeddings/
│   └── metadata.csv              # Master metadata file
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_easy_task_basic_vae.ipynb
│   ├── 03_medium_task_conv_vae.ipynb
│   └── 04_hard_task_beta_cvae.ipynb
├── src/
│   ├── __init__.py
│   ├── vae.py                    # VAE, Conv-VAE, Beta-VAE, CVAE models
│   ├── dataset.py                # Data loading, feature extraction, PyTorch Dataset
│   ├── clustering.py             # K-Means, Agglomerative, DBSCAN wrappers
│   ├── evaluation.py             # All clustering metrics
│   ├── visualization.py          # t-SNE, UMAP, latent space plots
│   ├── baselines.py              # PCA+KMeans, AE+KMeans, spectral clustering
│   ├── train.py                  # Training loop for VAE models
│   └── config.py                 # Hyperparameters and paths
├── results/
│   ├── latent_visualization/     # t-SNE/UMAP plots
│   ├── cluster_plots/            # Cluster distribution plots
│   ├── reconstructions/          # VAE reconstruction examples
│   ├── clustering_metrics.csv    # All metrics in tabular form
│   └── comparison_table.csv      # Methods vs. metrics comparison
├── report/
│   └── figures/                  # Figures for the NeurIPS paper
├── README.md
├── requirements.txt
├── setup.py                      # Optional: make project installable
└── .gitignore
```

---

## 8. Report Plan (NeurIPS Format)

**Template:** [NeurIPS 2024 Overleaf Template](https://www.overleaf.com/latex/templates/neurips-2024/tpsbbrdqcmsh)

### Report Sections

| # | Section | Content | Approx. Length |
|---|---------|---------|----------------|
| 1 | **Abstract** | Goal, method, key findings | ~150 words |
| 2 | **Introduction** | Motivation: why hybrid language music clustering? Why VAE? | 0.5-1 page |
| 3 | **Related Work** | VAE literature, music representation learning, clustering methods | 0.5-1 page |
| 4 | **Method** | VAE architecture details, feature extraction pipeline, multi-modal fusion, clustering | 1-2 pages |
| 5 | **Experiments** | Dataset description, preprocessing, hyperparameters, training details | 1 page |
| 6 | **Results** | Metrics tables, latent space visualizations, reconstruction examples, comparisons | 1-2 pages |
| 7 | **Discussion** | Interpretation of clusters, why certain methods work better, limitations | 0.5-1 page |
| 8 | **Conclusion** | Summary and future work | 0.25 page |
| 9 | **References** | 15-25 references | 0.5-1 page |

### Key Figures for Report
1. **Architecture diagram** of VAE / CVAE / Beta-VAE
2. **Feature extraction pipeline** diagram
3. **t-SNE / UMAP latent space** colored by language and genre
4. **Cluster distribution** bar charts
5. **Comparison table** of all methods vs. metrics
6. **Reconstruction examples** from VAE decoder
7. **Latent traversal** visualizations (Beta-VAE disentanglement)
8. **Training curves** (loss vs. epoch)

---

## 9. Tools & Libraries

### Core
| Library | Purpose | Install |
|---------|---------|---------|
| **PyTorch** | VAE implementation, training | `pip install torch torchvision torchaudio` |
| **librosa** | Audio feature extraction (MFCC, Mel-spectrograms) | `pip install librosa` |
| **scikit-learn** | Clustering (K-Means, DBSCAN, Agglomerative), metrics, PCA | `pip install scikit-learn` |
| **numpy** | Numerical operations | `pip install numpy` |
| **pandas** | Data management | `pip install pandas` |

### Visualization
| Library | Purpose | Install |
|---------|---------|---------|
| **matplotlib** | Plotting | `pip install matplotlib` |
| **seaborn** | Statistical visualization | `pip install seaborn` |
| **umap-learn** | UMAP dimensionality reduction | `pip install umap-learn` |

### NLP / Lyrics Embeddings
| Library | Purpose | Install |
|---------|---------|---------|
| **sentence-transformers** | LaBSE / multilingual-BERT embeddings | `pip install sentence-transformers` |
| **transformers** | Hugging Face models | `pip install transformers` |

### Audio Collection
| Library | Purpose | Install |
|---------|---------|---------|
| **yt-dlp** | Download audio from YouTube | `pip install yt-dlp` |
| **spotipy** | Spotify API (metadata, audio features) | `pip install spotipy` |
| **pydub** | Audio format conversion | `pip install pydub` |

### Others
| Library | Purpose | Install |
|---------|---------|---------|
| **tensorboard** | Training monitoring | `pip install tensorboard` |
| **tqdm** | Progress bars | `pip install tqdm` |
| **soundfile** | Audio I/O | `pip install soundfile` |

### Full `requirements.txt`
```
torch>=2.0.0
torchaudio>=2.0.0
librosa>=0.10.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
umap-learn>=0.5.0
sentence-transformers>=2.2.0
transformers>=4.30.0
yt-dlp>=2023.0.0
pydub>=0.25.0
soundfile>=0.12.0
tensorboard>=2.13.0
tqdm>=4.65.0
```

---

## 10. Timeline & Milestones

| Week | Dates (Approx.) | Tasks | Deliverable |
|------|-----------------|-------|-------------|
| **Week 1** | Mar 1-7 | Environment setup, data collection starts | Working env, initial dataset |
| **Week 2** | Mar 8-14 | Feature extraction, EDA, basic VAE | Feature files, EDA notebook |
| **Week 3** | Mar 15-21 | Easy task complete, start Medium task | Easy task results, Conv-VAE draft |
| **Week 4** | Mar 22-28 | Medium task complete, start Hard task | Medium task results, Beta-VAE/CVAE draft |
| **Week 5** | Mar 29-Apr 4 | Hard task complete, all comparisons | Full results, comparison tables |
| **Week 6** | Apr 5-10 | Report writing, code cleanup, final push | **SUBMIT by Apr 10** |

---

## 11. Key Resources & References

### Papers to Read
1. **Kingma & Welling (2014)** — "Auto-Encoding Variational Bayes" (original VAE paper)
   - https://arxiv.org/abs/1312.6114
2. **Higgins et al. (2017)** — "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"
   - https://openreview.net/forum?id=Sy2fzU9gl
3. **Sohn et al. (2015)** — "Learning Structured Output Representation using Deep Conditional Generative Models" (CVAE)
   - https://papers.nips.cc/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html
4. **van den Oord et al. (2017)** — "Neural Discrete Representation Learning" (VQ-VAE, good context)
   - https://arxiv.org/abs/1711.00937
5. **McFee et al. (2015)** — "librosa: Audio and Music Signal Analysis in Python"
   - https://conference.scipy.org/proceedings/scipy2015/brian_mcfee.html

### Tutorials & Guides
- **PyTorch VAE Tutorial**: https://pytorch.org/tutorials/beginner/variational_autoencoder.html
- **VAE from Scratch (Toward Data Science)**: Search "VAE tutorial PyTorch" on Medium/TDS
- **librosa Quickstart**: https://librosa.org/doc/latest/tutorial.html
- **UMAP Documentation**: https://umap-learn.readthedocs.io/
- **Sentence-BERT / LaBSE**: https://www.sbert.net/docs/pretrained_models.html
- **NeurIPS 2024 LaTeX Template**: https://www.overleaf.com/latex/templates/neurips-2024/tpsbbrdqcmsh

### Dataset Links
- Million Song Dataset: http://millionsongdataset.com/
- GTZAN Genre Collection: http://marsyas.info/downloads/datasets.html
- Jamendo Dataset: https://www.kaggle.com/datasets/andradaolteanu/jamendo-music-da
- MIR-1K Dataset: https://sites.google.com/site/unvoicedsoundseparation/mir-1k
- Lakh MIDI Dataset: https://colinraffel.com/projects/lmd/
- Kaggle Lyrics Datasets: https://www.kaggle.com/datasets?search=lyrics

---

## 12. Risk Mitigation & Tips

### Common Pitfalls
| Risk | Mitigation |
|------|-----------|
| **KL collapse** (VAE ignores latent space) | Use KL annealing (gradually increase KL weight during training), use cyclical annealing |
| **Poor clustering on raw features** | Normalize features, try different latent dimensions (8, 16, 32, 64) |
| **Bangla songs hard to find** | Use YouTube with yt-dlp, collect from Bangla music streaming sites |
| **Lyrics not available for all songs** | Use audio-only VAE as fallback; lyrics embeddings are optional for Easy task |
| **Spectrograms different sizes** | Pad/trim all audio to fixed duration (e.g., 30s) before feature extraction |
| **Overfitting VAE** | Use dropout, early stopping, monitor reconstruction on validation set |
| **DBSCAN sensitive to parameters** | Use HDBSCAN or grid search over eps and min_samples |
| **Report too short/vague** | Follow NeurIPS template strictly, include enough figures and tables |

### Hyperparameters to Tune
- **Latent dimension**: 8, 16, 32, 64 (try multiple)
- **Learning rate**: 1e-3, 5e-4, 1e-4
- **β (Beta-VAE)**: 1, 2, 4, 8, 10
- **Number of clusters (K)**: Use elbow method + silhouette analysis
- **Audio segment length**: 10s, 20s, 30s
- **MFCC coefficients**: 13, 20, 40
- **Batch size**: 32, 64, 128

### Pro Tips
1. **Start with the Easy task first** — get a working pipeline end-to-end, then iterate
2. **Version your experiments** — use TensorBoard or Weights & Biases to track runs
3. **Save intermediate results** — don't re-extract features every time
4. **Use GPU** — PyTorch with CUDA for faster training (Google Colab if no local GPU)
5. **Test with small data first** — validate pipeline on 50 songs before scaling up
6. **Write the report incrementally** — don't leave it all for the last day
7. **Make your GitHub repo clean** — README with clear setup instructions, example outputs
8. **Use fixed random seeds** — for reproducibility (`torch.manual_seed(42)`, `np.random.seed(42)`)

---

## Quick Start Checklist

```
[ ] 1. Clone/create repo with folder structure above
[ ] 2. pip install -r requirements.txt
[ ] 3. Collect 200+ English songs (GTZAN)
[ ] 4. Collect 200+ Bangla songs (YouTube/yt-dlp)
[ ] 5. Run feature extraction (MFCC + Mel-spectrograms)
[ ] 6. Train basic VAE on MFCC features
[ ] 7. Run K-Means on latent space → visualize with t-SNE
[ ] 8. Compute Silhouette Score → compare with PCA+KMeans baseline
[ ] 9. ✅ Easy task done! Proceed to Medium...
```

---

*Last updated: February 28, 2026*
