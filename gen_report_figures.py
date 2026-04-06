"""Generate comparison figures for the report."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

FIG = "report/figures"
os.makedirs(FIG, exist_ok=True)

# ── 1. Multi-K comparison ────────────────────────────────────────────────
models = ['BasicVAE\n(Easy)', 'ConvVAE\n(Medium)', 'HybridVAE\n(Medium)',
          'BetaVAE\n(Hard)', 'CVAE\n(Hard)', 'MultiModal\n(Hard)', 'PCA\nBaseline']

ari_k2  = [0.001, 0.055, 0.991, 0.000, 0.179, 0.006, 0.002]
ari_k10 = [0.022, 0.019, 0.269, 0.013, 0.073, 0.122, 0.155]
ari_k18 = [0.053, 0.046, 0.215, 0.002, 0.112, 0.109, 0.144]
ss_k2   = [0.605, 0.660, 0.789, 0.943, 0.514, 0.390, 0.162]
ss_k10  = [0.430, 0.428, 0.409, 0.927, 0.271, 0.258, 0.102]
ss_k18  = [0.429, 0.323, 0.341, 0.901, 0.208, 0.208, 0.053]

x = np.arange(len(models))
w = 0.25
cols = ['#2196F3', '#4CAF50', '#FF9800']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Multi-K Evaluation: K=2 (Language), K=10 (All Genres), K=18 (Labeled Genres)',
             fontsize=12, fontweight='bold')

ax = axes[0]
ax.bar(x - w, ari_k2,  w, label='K=2 (language)',        color=cols[0], alpha=0.85)
ax.bar(x,     ari_k10, w, label='K=10 (genre, all)',      color=cols[1], alpha=0.85)
ax.bar(x + w, ari_k18, w, label='K=18 (genre, labeled)',  color=cols[2], alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9)
ax.set_ylabel('Adjusted Rand Index (ARI)'); ax.set_title('ARI by Model and K')
ax.legend(fontsize=9); ax.set_ylim(0, 1.1)
ax.annotate('ARI=0.991\n(near-perfect\nlanguage sep.)',
            xy=(x[2]-w, 0.991), xytext=(x[2]-w+0.9, 0.78),
            arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5),
            fontsize=8, color='darkred', ha='center')

ax = axes[1]
ax.bar(x - w, ss_k2,  w, label='K=2 (language)',        color=cols[0], alpha=0.85)
ax.bar(x,     ss_k10, w, label='K=10 (genre, all)',      color=cols[1], alpha=0.85)
ax.bar(x + w, ss_k18, w, label='K=18 (genre, labeled)',  color=cols[2], alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9)
ax.set_ylabel('Silhouette Score'); ax.set_title('Silhouette Score by Model and K')
ax.legend(fontsize=9); ax.set_ylim(0, 1.1)
ax.annotate('SS=0.943\n(over-reg., NOT\nmeaningful)',
            xy=(x[3]-w, 0.943), xytext=(x[3]-w+1.0, 0.75),
            arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5),
            fontsize=8, color='darkred', ha='center')

plt.tight_layout()
plt.savefig(f'{FIG}/multi_k_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved multi_k_comparison.png")

# ── 2. Before vs After Finetuning ────────────────────────────────────────
model_names = ['BasicVAE', 'BetaVAE\n(b=2)', 'ConvVAE', 'CVAE', 'MultiModal']
before_ss  = [0.430, 0.927, 0.429, 0.271, 0.258]   # main task K=10
after_ss   = [0.507, 0.382, 0.448, 0.493, 0.502]   # finetune best K=10
before_ari = [0.022, 0.013, 0.019, 0.073, 0.122]   # main task K=10 ARI
after_ari  = [0.053, 0.002, 0.046, 0.112, 0.109]   # main task K=18 ARI (best achievable)

x = np.arange(len(model_names))
w = 0.35

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Before vs. After Hyperparameter Finetuning\n'
             '(grid search over beta, latent_dim; left = main task, right = best finetune config)',
             fontsize=11, fontweight='bold')

ax = axes[0]
ax.bar(x - w/2, before_ss, w, label='Main task (K=10)', color='#5C9BD6', alpha=0.9)
ax.bar(x + w/2, after_ss,  w, label='Finetune best (K=10)', color='#E8823A', alpha=0.9)
ax.set_xticks(x); ax.set_xticklabels(model_names, fontsize=10)
ax.set_ylabel('Silhouette Score'); ax.set_title('Silhouette Score')
ax.legend(fontsize=9); ax.set_ylim(0, 1.05)
for i, (b, a) in enumerate(zip(before_ss, after_ss)):
    pct = (a - b) / b * 100
    ax.text(i + w/2, a + 0.01, f'{pct:+.0f}%', ha='center', fontsize=8,
            color='green' if pct > 0 else 'red')

ax = axes[1]
ax.bar(x - w/2, before_ari, w, label='K=10 (main task)', color='#5C9BD6', alpha=0.9)
ax.bar(x + w/2, after_ari,  w, label='K=18 (best after)', color='#E8823A', alpha=0.9)
ax.set_xticks(x); ax.set_xticklabels(model_names, fontsize=10)
ax.set_ylabel('Adjusted Rand Index (ARI)'); ax.set_title('ARI: K=10 vs K=18')
ax.legend(fontsize=9); ax.set_ylim(0, 0.2)

improvements = [(a-b)/b*100 if b > 0 else 0 for a,b in zip(after_ss, before_ss)]
ax = axes[2]
bar_colors = ['#4CAF50' if v > 0 else '#F44336' for v in improvements]
bars = ax.bar(x, improvements, color=bar_colors, alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(model_names, fontsize=10)
ax.set_ylabel('SS % Change'); ax.set_title('SS Change After Finetuning (%)')
ax.axhline(0, color='black', lw=0.8, linestyle='--')
for bar, val in zip(bars, improvements):
    ypos = bar.get_height() + 0.5 if val >= 0 else bar.get_height() - 2.0
    ax.text(bar.get_x() + bar.get_width()/2, ypos, f'{val:+.1f}%',
            ha='center', fontsize=10, fontweight='bold')
ax.set_ylim(-80, 25)

plt.tight_layout()
plt.savefig(f'{FIG}/finetune_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved finetune_comparison.png")

# ── 3. Dataset Pipeline ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 3.5))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

steps = [
    ("GTZAN\n1,000 tracks\n30-sec WAV\n10 genres (EN)", '#1565C0'),
    ("MagnaTagATune\n9,000 tracks\n29-sec MP3\nmulti-tag (EN)", '#1976D2'),
    ("BanglaBeats\n10,020 tracks\n3-sec WAV\n8 genres (BN)", '#E65100'),
    ("Feature\nExtraction\nMFCC 40d\nMel 256d\nCombined 90d", '#6A1B9A'),
    ("Lyrics\nEmbeddings\nProxy text\n+ LaBSE\n384d", '#880E4F'),
    ("VAE Training\nEasy: MLP-VAE\nMedium: Conv/Hybrid\nHard: MM/CVAE/Beta", '#1B5E20'),
    ("Evaluation\nK=2,10,18\nARI/SS/NMI\nPurity\n156 configs", '#B71C1C'),
]

n = len(steps)
box_w = 0.12
spacing = (1.0 - n * box_w) / (n + 1)

for i, (label, color) in enumerate(steps):
    x0 = spacing + i * (box_w + spacing)
    y0 = 0.15
    rect = mpatches.FancyBboxPatch((x0, y0), box_w, 0.65,
                                    boxstyle="round,pad=0.02",
                                    facecolor=color, edgecolor='white',
                                    linewidth=2, alpha=0.9)
    ax.add_patch(rect)
    ax.text(x0 + box_w/2, y0 + 0.325, label, ha='center', va='center',
            fontsize=7.5, color='white', fontweight='bold',
            family='monospace')
    if i < n - 1:
        x_arrow_start = x0 + box_w + 0.005
        x_arrow_end   = x0 + box_w + spacing - 0.005
        ax.annotate('', xy=(x_arrow_end, 0.48), xytext=(x_arrow_start, 0.48),
                    arrowprops=dict(arrowstyle='->', lw=2, color='#444'))

ax.text(0.5, 0.93, 'Dataset Construction and Full Pipeline Overview',
        ha='center', va='center', fontsize=12, fontweight='bold',
        transform=ax.transAxes)
ax.text(0.5, 0.06, 'Total: 20,020 tracks  |  English: 10,000 (GTZAN + MagnaTagATune)  |  Bangla: 10,020 (BanglaBeats)',
        ha='center', va='center', fontsize=9, color='#333',
        transform=ax.transAxes)

plt.tight_layout()
plt.savefig(f'{FIG}/dataset_pipeline.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved dataset_pipeline.png")
print(f"Total figures: {len([f for f in os.listdir(FIG) if f.endswith('.png')])}")
