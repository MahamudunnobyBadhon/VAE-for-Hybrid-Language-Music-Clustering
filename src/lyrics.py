"""
Lyrics embedding module for the VAE Music Clustering project.

Handles:
  - Generating proxy lyrics text from song metadata (genre + language)
  - Loading and filtering the Genius song lyrics dataset (if available)
  - Embedding lyrics using LaBSE (Language-agnostic BERT Sentence Embeddings)
  - Saving/loading lyrics embeddings

Usage (no Genius dataset):
    from src.lyrics import extract_and_save_lyrics_embeddings
    embeddings = extract_and_save_lyrics_embeddings(metadata)

Usage (with Genius dataset CSV):
    embeddings = extract_and_save_lyrics_embeddings(
        metadata, genius_csv_path="path/to/song_lyrics_with_language.csv"
    )
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import LYRICS_EMB_DIR, RANDOM_STATE


# ============================================================
# Genre-descriptive proxy text templates
# ============================================================

# English genre descriptions
_ENGLISH_TEMPLATES = {
    "blues":      "blues music with soulful guitar melodies and emotional lyrics about hardship",
    "classical":  "classical orchestral music with rich harmonies and instrumental arrangements",
    "country":    "country music with acoustic guitar storytelling and rural themes",
    "disco":      "disco dance music with funky bass lines and upbeat rhythms",
    "electronic": "electronic music with synthesizers beats and digital soundscapes",
    "folk":       "folk music with acoustic instruments and traditional storytelling lyrics",
    "hiphop":     "hip hop music with rhythmic rap verses and urban beats",
    "jazz":       "jazz music with improvised melodies and complex chord progressions",
    "metal":      "heavy metal music with distorted guitars and powerful percussion",
    "pop":        "pop music with catchy melodies and modern production",
    "reggae":     "reggae music with offbeat rhythms and social commentary lyrics",
    "rock":       "rock music with electric guitars and energetic performances",
    "misc":       "contemporary music with diverse instrumentation and vocal performances",
}

# Bangla genre descriptions
_BANGLA_TEMPLATES = {
    "classical":  "বাংলা শাস্ত্রীয় সংগীত ঐতিহ্যবাহী রাগ এবং তাল সহ",
    "devotional": "ভক্তিমূলক বাংলা গান রবীন্দ্রনাথ নজরুলের ধর্মীয় ভাব সহ",
    "folk":       "বাংলা লোকসংগীত বাউল ভাটিয়ালি মুর্শিদী সুর সহ",
    "modern":     "আধুনিক বাংলা গান সমসাময়িক সুর এবং কথা সহ",
    "rock":       "বাংলা রক সংগীত বৈদ্যুতিক গিটার এবং শক্তিশালী ছন্দ সহ",
    "pop":        "বাংলা পপ সংগীত সুরেলা সুর এবং আধুনিক প্রযোজনা সহ",
    "misc":       "বাংলা সংগীত বিভিন্ন বাদ্যযন্ত্র এবং কণ্ঠ পরিবেশনা সহ",
}

# Fallbacks when genre is unknown
_FALLBACK = {
    "english": "english song with contemporary music production and vocal performance",
    "bangla":  "বাংলা গান আধুনিক সুর এবং কণ্ঠ পরিবেশনা সহ",
    "bn":      "বাংলা গান আধুনিক সুর এবং কণ্ঠ পরিবেশনা সহ",
    "en":      "english song with contemporary music production and vocal performance",
}


def _parse_genre_from_filename(filename: str, language: str) -> str:
    """
    Extract genre label from audio filename conventions used in this project.

    Supported conventions:
      - Bangla synthetic: bangla_classical_002.wav  → "classical"
      - GTZAN:            blues.00000.wav           → "blues"
      - General fallback: any word before first dot or underscore
    """
    fname = Path(filename).stem.lower()

    # Bangla synthetic: bangla_<genre>_<number>
    m = re.match(r"^bangla_(\w+)_\d+$", fname)
    if m:
        return m.group(1)

    # GTZAN doubled: english_<genre>_<genre>.<number>
    m = re.match(r"^english_([a-z]+)_\1\.\d+", fname)
    if m:
        return m.group(1)

    # GTZAN convention: <genre>.<number>
    m = re.match(r"^([a-z]+)\.\d+$", fname)
    if m:
        return m.group(1)

    # MagnaTagATune: english_magna_<number>
    if "magna" in fname:
        return "untagged"

    # Fallback: first token before underscore or dot
    return re.split(r"[._]", fname)[0]


def generate_proxy_lyrics(metadata: pd.DataFrame) -> list:
    """
    Generate genre-descriptive proxy text for each song in metadata.

    Uses the song's language and genre (parsed from filename if 'genre'
    column is absent) to select a template description. This produces
    semantically meaningful embeddings even when no real lyrics are
    available.

    Args:
        metadata: DataFrame with columns 'language' and optionally
                  'genre' and 'filename'.

    Returns:
        List of strings, one per row in metadata.
    """
    texts = []
    for _, row in metadata.iterrows():
        lang = str(row.get("language", "english")).lower().strip()

        # Resolve genre
        if "genre" in row and pd.notna(row["genre"]):
            genre = str(row["genre"]).lower().strip()
        elif "filename" in row and pd.notna(row["filename"]):
            genre = _parse_genre_from_filename(str(row["filename"]), lang)
        else:
            genre = "misc"

        # Select template
        if lang in ("bangla", "bn"):
            text = _BANGLA_TEMPLATES.get(genre, _FALLBACK.get("bangla"))
        else:
            text = _ENGLISH_TEMPLATES.get(genre, _FALLBACK.get("english"))

        texts.append(text)

    return texts


# ============================================================
# Genius Dataset Loader (optional)
# ============================================================

def load_genius_lyrics(genius_csv_path: str,
                       languages: list = None,
                       max_per_tag: int = 50,
                       random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Load and filter the Genius Song Lyrics dataset from Kaggle.

    Expected CSV columns: title, tag, artist, year, views, features,
                          lyrics, id, language_cld3, language_ft, language

    Args:
        genius_csv_path: path to the Genius CSV file
        languages: list of language codes to keep, e.g. ["en", "bn"].
                   Defaults to ["en", "bn"].
        max_per_tag: maximum rows to sample per (language, tag) group
        random_state: random seed for sampling

    Returns:
        DataFrame with columns: language, tag, lyrics (cleaned)
    """
    if languages is None:
        languages = ["en", "bn"]

    print(f"Loading Genius dataset from {genius_csv_path}...")
    df = pd.read_csv(genius_csv_path, low_memory=False,
                     usecols=["tag", "lyrics", "language"])

    # Keep only rows with known, agreed-upon language
    df = df[df["language"].isin(languages)].copy()
    print(f"  Rows after language filter ({languages}): {len(df)}")

    if df.empty:
        raise ValueError(
            f"No rows found for languages {languages}. "
            "Check that the 'language' column exists and contains these codes."
        )

    # Clean lyrics: strip section headers like [Verse 1], [Chorus]
    df["lyrics"] = df["lyrics"].fillna("").apply(_clean_genius_lyrics)

    # Drop empty lyrics
    df = df[df["lyrics"].str.len() > 20]

    # Sample max_per_tag per (language, tag) group for balance
    rng = np.random.RandomState(random_state)
    groups = []
    for (lang, tag), grp in df.groupby(["language", "tag"]):
        if len(grp) > max_per_tag:
            grp = grp.sample(n=max_per_tag, random_state=rng.randint(0, 9999))
        groups.append(grp)

    result = pd.concat(groups, ignore_index=True)
    print(f"  Final Genius sample: {len(result)} rows "
          f"({result['language'].value_counts().to_dict()})")
    return result[["language", "tag", "lyrics"]]


def _clean_genius_lyrics(text: str) -> str:
    """Remove Genius metadata annotations and normalize whitespace."""
    # Remove [section headers] like [Verse 1], [Chorus], [Bridge], etc.
    text = re.sub(r"\[.*?\]", " ", text)
    # Collapse multiple newlines/spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# LaBSE Embedding
# ============================================================

def embed_lyrics_labse(texts: list,
                        model_name: str = "LaBSE",
                        batch_size: int = 64,
                        normalize: bool = True) -> np.ndarray:
    """
    Embed a list of text strings using a multilingual sentence transformer.

    Args:
        texts: list of strings to embed
        model_name: sentence-transformers model name.
                    "LaBSE" (1.9 GB, 768-dim) is recommended for Bangla+English.
                    "paraphrase-multilingual-MiniLM-L12-v2" (420 MB, 384-dim)
                    is a lighter alternative.
        batch_size: number of texts per encoding batch
        normalize: L2-normalize embeddings (recommended for cosine similarity)

    Returns:
        np.ndarray of shape (n_texts, embedding_dim)
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is not installed. "
            "Run: pip install sentence-transformers"
        )

    print(f"Loading sentence transformer model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Encoding {len(texts)} texts (batch_size={batch_size})...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings.astype(np.float32)


# ============================================================
# Orchestration: Extract and Save
# ============================================================

def extract_and_save_lyrics_embeddings(
        metadata: pd.DataFrame,
        genius_csv_path: str = None,
        embeddings_name: str = "lyrics_embeddings",
        model_name: str = "LaBSE",
        batch_size: int = 64,
) -> np.ndarray:
    """
    Generate lyrics embeddings for all songs in metadata and save to disk.

    Strategy:
      1. If genius_csv_path is provided, attempt to match songs by genre/language
         to real Genius lyrics. Falls back to proxy text for unmatched songs.
      2. Without Genius CSV, generates proxy text from genre + language labels.

    Args:
        metadata: DataFrame with 'language' column and optionally 'genre',
                  'filename' columns.
        genius_csv_path: optional path to the Genius CSV file.
        embeddings_name: filename stem for saved .npy file.
        model_name: sentence-transformers model to use.
        batch_size: encoding batch size.

    Returns:
        np.ndarray of shape (len(metadata), embedding_dim)
    """
    output_path = LYRICS_EMB_DIR / f"{embeddings_name}.npy"

    # Build text list for each song
    if genius_csv_path and Path(genius_csv_path).exists():
        print("Genius dataset found — using real lyrics where available.")
        texts = _match_genius_lyrics(metadata, genius_csv_path)
    else:
        if genius_csv_path:
            print(f"Warning: Genius CSV not found at {genius_csv_path}. "
                  "Using proxy lyrics instead.")
        print("Generating proxy lyrics from genre + language metadata...")
        texts = generate_proxy_lyrics(metadata)

    # Embed
    embeddings = embed_lyrics_labse(texts, model_name=model_name,
                                     batch_size=batch_size)

    # Save
    np.save(output_path, embeddings)
    print(f"Saved lyrics embeddings to {output_path}  shape={embeddings.shape}")
    return embeddings


def _match_genius_lyrics(metadata: pd.DataFrame,
                          genius_csv_path: str) -> list:
    """
    For each song in metadata, find a representative lyric from the Genius
    dataset with the same language and (approximate) genre, then fall back
    to proxy text if no match is found.

    Returns a list of strings, one per metadata row.
    """
    rng = np.random.RandomState(RANDOM_STATE)

    # Load Genius grouped by (language, tag)
    genius = load_genius_lyrics(genius_csv_path)
    # Build lookup: (lang_code, tag) -> list of lyrics strings
    lookup = {}
    lang_map = {"english": "en", "bangla": "bn", "en": "en", "bn": "bn"}
    for _, row in genius.iterrows():
        key = (row["language"], row["tag"].lower())
        lookup.setdefault(key, []).append(row["lyrics"])

    texts = []
    for _, row in metadata.iterrows():
        lang_raw = str(row.get("language", "english")).lower().strip()
        lang_code = lang_map.get(lang_raw, "en")

        if "genre" in row and pd.notna(row["genre"]):
            genre = str(row["genre"]).lower().strip()
        elif "filename" in row and pd.notna(row["filename"]):
            genre = _parse_genre_from_filename(str(row["filename"]), lang_raw)
        else:
            genre = "misc"

        key = (lang_code, genre)
        if key in lookup and lookup[key]:
            # Pick a random lyric from the pool for this (language, genre)
            texts.append(rng.choice(lookup[key]))
        else:
            # Fall back to proxy text
            dummy_row = pd.Series({"language": lang_raw, "genre": genre})
            texts.append(generate_proxy_lyrics(pd.DataFrame([dummy_row]))[0])

    return texts


# ============================================================
# Load helper (used by run scripts)
# ============================================================

def load_lyrics_embeddings_from_file(
        embeddings_name: str = "lyrics_embeddings") -> np.ndarray:
    """Load previously saved lyrics embeddings."""
    emb_path = LYRICS_EMB_DIR / f"{embeddings_name}.npy"
    if not emb_path.exists():
        raise FileNotFoundError(
            f"Lyrics embeddings not found at {emb_path}. "
            "Run extract_and_save_lyrics_embeddings() first."
        )
    embeddings = np.load(emb_path)
    print(f"Loaded lyrics embeddings: {embeddings.shape} from {emb_path}")
    return embeddings.astype(np.float32)
