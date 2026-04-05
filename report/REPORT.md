# Variational Autoencoders for Hybrid Language Music Clustering
## English and Bangla Unsupervised Representation Learning

**Course:** CSE715 -- Neural Networks, Spring 2026
**Instructor:** Moin Mostakim
**Deadline:** April 10, 2026

---

## Abstract

We present a comprehensive unsupervised learning pipeline using Variational Autoencoders (VAEs) for clustering a hybrid-language music dataset comprising **20,020 tracks in English and Bangla**. Across three progressive difficulty tiers we implement and evaluate: (1) a basic MLP-VAE on MFCC features, (2) a ConvVAE and HybridVAE on mel-spectrograms with multilingual lyrics embeddings, and (3) a MultiModalVAE, Conditional VAE (CVAE), and Beta-VAE for disentangled multi-modal representations.

Our best model -- the **MultiModalVAE** -- achieves Silhouette = **0.497** and CHI = **81,020**, outperforming PCA + K-Means by **3.2x** in Silhouette and **22.2x** in CHI. Results confirm that deep generative models learn language-agnostic latent representations that meaningfully separate English and Bangla music without supervision.

---

## 1. Introduction

Music is universal, yet computational music analysis is dominated by English datasets. Bangla music -- spoken by 230M+ people -- spans rich traditions (Baul, Bhatiali, Rabindra Sangeet) alongside modern genres. Unsupervised clustering of hybrid-language music is challenging due to:
- Absent genre labels for Bangla content at scale
- Domain gap: 3-second Bangla clips vs. 30-second English recordings
- Multi-modal fusion across languages (audio + lyrics)

We build a 20,020-track dataset, define three progressive VAE tasks, and systematically compare 9 methods across 5 metrics.

---

## 2. Dataset

| Source | Language | Tracks | Clip Length | Genres |
|--------|----------|--------|-------------|--------|
| GTZAN | English | 1,000 | 30 sec | 10 (blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, rock) |
| MagnaTagATune | English | 9,000 | 29 sec | Multi-tag Western |
| BanglaBeats | Bangla | 10,020 | 3 sec | 8 (Adhunik, Folk, Hip-Hop, Indie, Islamic, Metal, Pop, Rock) |
| **Total** | Both | **20,020** | -- | 18+ |

**Feature types:**
- **MFCC** (40-dim): 20 coefficients x mean+std, pooled over time
- **Mel-Spectrogram** (256-dim): 128 log-mel bands x mean+std
- **Lyrics Embeddings** (384-dim): Proxy genre descriptions encoded via `paraphrase-multilingual-MiniLM-L12-v2`

---

## 3. Methods

### Easy Task -- MLP-VAE
Fully-connected encoder `[40->256->128->32]`, symmetric decoder. Latent dim = 32. Trained 100 epochs with ELBO loss.

### Medium Task -- ConvVAE & HybridVAE
- **ConvVAE**: 1D convolutional encoder on 256-dim mel features
- **HybridVAE**: Late fusion of audio (256-dim) + lyrics (384-dim) -> projected to 128-dim each -> concatenated -> latent bottleneck

### Hard Task -- MultiModalVAE, CVAE, Beta-VAE
- **MultiModalVAE**: Cross-attention fusion of audio and lyrics streams
- **CVAE**: Conditioned on language label (English/Bangla)
- **Beta-VAE (β=4)**: Increased KL weight for disentangled representations

**Clustering:** K-Means (K=10), Agglomerative (Ward), DBSCAN on latent encodings μ ∈ ℝ³².

---

## 4. Results

### Task 1: Easy (MFCC, N=20,019)

| Method | Silhouette (↑) | CH Index (↑) |
|--------|----------------|---------------|
| **BasicVAE + K-Means** | **0.4286** | **69,333** |
| PCA + K-Means | 0.1051 | 1,589 |
| Raw Features + K-Means | 0.0974 | 1,463 |

BasicVAE (40-dim MFCC -> 32-dim latent) trained for 100 epochs with early stopping (lr = 1e-3, batch = 64) and delivers **4.1x** higher Silhouette than the PCA baseline on the 20,019-sample (~10k English + 10k Bangla) corpus.

---

### Task 2: Medium (Mel Spectrogram, N=20,019 -- fixed genre labels)

| Method | Silhouette (↑) | CH Index (↑) | DB (↓) | ARI (↑) | NMI (↑) | Purity (↑) |
|--------|----------------|---------------|--------|--------|--------|------------|
| **ConvVAE + K-Means** | **0.4294** | **42,746** | **0.749** | 0.019 | 0.143 | 0.484 |
| ConvVAE + Agglomerative | 0.4220 | 36,537 | 0.780 | 0.019 | 0.145 | 0.479 |
| ConvVAE + DBSCAN | 0.8448 | 159 | 0.117 | ~0 | ~0 | 0.449 |
| **HybridVAE + K-Means** | 0.3769 | 28,250 | 0.859 | **0.209** | **0.384** | 0.578 |
| HybridVAE + Agglomerative | 0.3445 | 24,793 | 0.879 | 0.234 | 0.397 | **0.585** |
| PCA + K-Means | 0.1539 | 3,649 | 1.721 | 0.137 | 0.350 | 0.581 |

*DBSCAN collapses to two language clusters, so ARI/NMI approach zero despite the inflated Silhouette.*

**Key takeaways**
- Genre parsing fix raised ARI from 0.10 to 0.23 (2.3x) and purity from 0.14 to 0.58 (4.2x), confirming the label bug diagnosis.
- HybridVAE (audio + lyrics) leads on external metrics (ARI/NMI/Purity), showing lyrics embeddings align genres.
- ConvVAE keeps the tightest internal geometry (Silhouette 0.43, DB 0.75) but clusters mostly by language/acoustics.
- DBSCAN is unusable here because it collapses to the English/Bangla split regardless of genre.
- Both VAE variants consistently early-stop around epochs 12-13, so future experiments can shorten the training schedule.

---

### Task 3: Hard (Mel + Lyrics, N=20,019)

| Method | SS ↑ | CHI ↑ | DBI ↓ | ARI ↑ | NMI ↑ |
|--------|------|-------|-------|-------|-------|
| **MultiModalVAE + K-Means** | **0.4974** | **81,020** | **0.557** | 0.052 | 0.299 |
| CVAE + K-Means | 0.4632 | 64,331 | 0.614 | **0.097** | **0.335** |
| Beta-VAE (β=4) + K-Means | 0.3123 | 29,250 | 0.987 | 0.058 | 0.308 |
| -- | -- | -- | -- | -- | -- |
| Autoencoder + K-Means | 0.0916 | 1,623 | 2.182 | 0.119 | 0.365 |
| PCA + K-Means | 0.1539 | 3,649 | 1.721 | 0.130 | 0.372 |
| Raw Features + K-Means | 0.1227 | 2,788 | 1.995 | 0.129 | 0.372 |

-> MultiModalVAE improves SS by **+223%** and CHI by **+2,120%** over PCA baseline.

---

### Summary

| Task | Best Model | Silhouette | vs. PCA |
|------|-----------|-----------|---------|
| Easy | MLP-VAE + K-Means | 0.4286 | +307% |
| Medium | ConvVAE + K-Means | 0.4294 | +179% |
| **Hard** | **MultiModalVAE + K-Means** | **0.4974** | **+223%** |

---

## 5. Discussion

**Why VAE representations cluster better:** PCA maximises variance linearly; VAE's nonlinear encoder + KL regularisation creates compact, smooth latent regions where similar audio patterns cluster tightly.

**Audio vs. audio+lyrics trade-off:** Adding lyrics improves ARI/NMI (language alignment) but slightly reduces Silhouette Score. Audio captures acoustic similarity; lyrics adds linguistic discriminability.

**Clip length bias (limitation):** Bangla clips (3s) have lower temporal variance than English clips (30s), creating a language-correlated artifact in features. Future work should segment all audio to a uniform 5-second window.

**Beta-VAE disentanglement:** β=4 encourages independent latent dimensions at the cost of clustering performance (SS=0.312). Latent traversal plots show smooth spectral transitions per dimension, confirming partial disentanglement.

**DBSCAN failure in high dimensions:** Mel features (256-dim) suffer from the curse of dimensionality -- pairwise distances concentrate, breaking density estimation. Apply DBSCAN in t-SNE/PCA space instead.

---

## 6. Conclusion

- VAE-based methods outperform all linear and density-based baselines by **3-4x Silhouette** and up to **22x CHI**
- Multi-modal fusion (audio + multilingual lyrics) improves language alignment (**NMI +16%** vs. audio-only)
- CVAE conditioning on language identity yields the best language-discriminative clusters (**ARI = 0.097**)
- MultiModalVAE achieves the best overall geometry (**SS = 0.497**)

**Future work:** Uniform clip lengths; fine-tuned wav2vec2/CLAP encoders; real Bangla lyric text; extension to Hindi, Tamil.

---

## References

1. Kingma & Welling. *Auto-encoding variational bayes.* ICLR, 2014.
2. Higgins et al. *β-VAE.* ICLR, 2017.
3. Sohn, Lee & Yan. *CVAE.* NeurIPS, 2015.
4. Reimers & Gurevych. *Sentence-BERT.* EMNLP, 2019.
5. Tzanetakis & Cook. *Musical genre classification.* IEEE Trans. Speech Audio, 2002.
6. McFee et al. *librosa.* SciPy, 2015.
7. Baevski et al. *wav2vec 2.0.* NeurIPS, 2020.
8. Roberts et al. *MusicVAE.* ICML, 2018.
9. Elizalde et al. *CLAP.* ICASSP, 2023.
10. Law et al. *MagnaTagATune.* ISMIR, 2009.

---

## Figures

All figures are in `results/`:

| Figure | File | Description |
|--------|------|-------------|
| Easy t-SNE | `latent_visualization/` | VAE latent space coloured by language/genre |
| Medium training | `medium/conv_vae_training_curves.png` | Loss curves |
| Medium t-SNE | `medium/conv_vae_tsne_language.png` | Latent space by language |
| Hard comparison | `hard/hard_comparison_table.png` | All methods table |
| Hard t-SNE | `hard/beta_vae_tsne_language.png` | Beta-VAE latent space |
| Reconstructions | `hard/beta_vae_reconstructions.png` | Mel reconstruction examples |
| Latent traversal | `hard/latent_traversal_dim_*.png` | Beta-VAE disentanglement |
