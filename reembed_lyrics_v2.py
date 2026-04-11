"""
v2 Lyrics Re-embedding: Language-neutral genre descriptions.

Fixes v1 label leakage:
  v1: Bangla tracks → Bengali script text → LaBSE trivially separates languages
      by text script, not music content. ARI=1.0 at K=2 is artefactual.
  v2: ALL tracks → English-language genre descriptions only (same script).
      LaBSE must distinguish tracks by genre semantics, not script identity.

Outputs saved to data/features/v2/:
  lyrics_embeddings_v2.npy   shape (N, 384)

Requires: pip install sentence-transformers
Uses model: paraphrase-multilingual-MiniLM-L12-v2 (420MB, 384-dim)
  - Lighter than LaBSE but still multilingual-capable
  - With genre-only English text, model choice doesn't matter much
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR   = Path("data")
FEAT_DIR   = DATA_DIR / "features"
OUT_DIR    = FEAT_DIR / "v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

META_CSV   = OUT_DIR / "metadata_v2.csv"        # use v2 metadata (same tracks)
FALLBACK   = FEAT_DIR / "combined_metadata.csv"  # fallback if v2 not yet created
OUT_EMB    = OUT_DIR / "lyrics_embeddings_v2.npy"

# Language-neutral genre templates (same English text for both languages)
GENRE_TEMPLATES = {
    "blues":      "soulful string melodies with slow rhythm and expressive emotional phrasing",
    "classical":  "orchestral composition with rich harmonic texture and instrumental arrangement",
    "country":    "acoustic string instruments with narrative vocal delivery and rural themes",
    "disco":      "dance music with prominent bass line and regular upbeat rhythmic pattern",
    "electronic": "synthesized tones with programmed beats and layered digital sound design",
    "folk":       "acoustic instruments with traditional melodic storytelling and simple structure",
    "hiphop":     "rhythmic spoken vocal delivery over percussive beat with urban production",
    "jazz":       "improvised melodic phrases over complex chord progressions with swing rhythm",
    "metal":      "distorted string instruments with high tempo percussion and powerful dynamics",
    "pop":        "catchy vocal melody with polished studio production and modern arrangement",
    "reggae":     "offbeat rhythmic pattern with bass-heavy groove and syncopated percussion",
    "rock":       "electric string instruments with energetic vocal and driving rhythmic section",
    "devotional": "meditative spiritual vocal performance with reverent tone and gentle instrumentation",
    "modern":     "contemporary melodic composition with polished production and clear vocal",
    "adhunik":    "contemporary melodic composition with polished production and clear vocal",
    "islamic":    "devotional vocal recitation with spiritual tone and restrained instrumentation",
    "untagged":   "instrumental composition with varied melodic and rhythmic elements",
    "misc":       "musical composition with diverse instrumentation and vocal performance",
    "unknown":    "musical composition with instrumentation and structured melodic content",
}
FALLBACK_TEXT = "musical composition with instrumentation and structured melodic content"


import re

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


def build_texts(meta: pd.DataFrame) -> list:
    texts = []
    for _, row in meta.iterrows():
        genre = row.get("genre", None)
        if not genre or str(genre) in ("nan", "None", ""):
            fname = str(row.get("filename", ""))
            genre = parse_genre(fname) if fname else "unknown"
        genre = str(genre).lower().strip()
        texts.append(GENRE_TEMPLATES.get(genre, FALLBACK_TEXT))
    return texts


def main():
    print("=" * 60)
    print(" v2 Lyrics Re-embedding: Language-neutral genre descriptions")
    print("=" * 60)

    meta_path = META_CSV if META_CSV.exists() else FALLBACK
    meta = pd.read_csv(meta_path)
    print(f"Loaded metadata: {len(meta)} tracks from {meta_path}")

    texts = build_texts(meta)

    # Sanity check: confirm no language word in any text
    leaky = [t for t in texts if "english" in t.lower() or "bangla" in t.lower()
             or "বাংলা" in t]
    print(f"Language-leaky texts: {len(leaky)} (should be 0)")

    # Genre distribution
    genres = {}
    for t in texts:
        genres[t[:40]] = genres.get(t[:40], 0) + 1
    print(f"Unique template types: {len(genres)}")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("Run: pip install sentence-transformers")

    print("\nLoading sentence transformer (paraphrase-multilingual-MiniLM-L12-v2)...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    print(f"Encoding {len(texts)} texts...")
    embeddings = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    print(f"Embeddings shape: {embeddings.shape}")
    np.save(OUT_EMB, embeddings)
    print(f"Saved -> {OUT_EMB}")

    # Bias check: cosine similarity between random English and Bangla embeddings
    if "language" in meta.columns:
        eng_idx = meta[meta["language"] == "english"].index[:100]
        ban_idx = meta[meta["language"] == "bangla"].index[:100]
        if len(eng_idx) > 0 and len(ban_idx) > 0:
            eng_emb = embeddings[eng_idx]
            ban_emb = embeddings[ban_idx]
            cross_sim = (eng_emb @ ban_emb.T).mean()
            within_eng = (eng_emb @ eng_emb.T).mean()
            print(f"\nLeakage check (cosine similarity):")
            print(f"  Within-English: {within_eng:.4f}")
            print(f"  Cross-language: {cross_sim:.4f}")
            print(f"  Ratio cross/within: {cross_sim/within_eng:.4f}")
            print("  (v2 fix: ratio should be close to 1.0, v1 was near 0)")


if __name__ == "__main__":
    main()
