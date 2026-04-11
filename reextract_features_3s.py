"""
v2 Feature Re-extraction: Fixed 3-second center window for ALL clips.

Fixes two v1 limitations:
  1. Clip-length bias: English (30s) vs Bangla (3s) → different feature variance.
     Fix: extract center 3s from every clip regardless of language.
  2. Tiling artifact: v1 tiled Bangla 3s→5s, creating duplicate spectral patterns.
     Fix: no tiling needed since 3s is the target.

Outputs saved to data/features/v2/:
  mfcc_features_v2.npy   shape (N, 40)   -- 20 MFCC mean + 20 MFCC std
  mel_features_v2.npy    shape (N, 256)  -- 128 mel mean + 128 mel std
  metadata_v2.csv        -- same row order as feature arrays

v1 files (data/features/mfcc_features.npy etc.) are NOT overwritten.
"""

import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm

# ── config (must match src/config.py) ──────────────────────────────────────
SR          = 22050
TARGET_DUR  = 3.0          # seconds — Bangla natural clip length
N_MFCC      = 20
N_MELS      = 128
HOP_LENGTH  = 512
N_FFT       = 2048

DATA_DIR    = Path("data")
AUDIO_DIR   = DATA_DIR / "audio"
FEAT_DIR    = DATA_DIR / "features"
META_CSV    = FEAT_DIR / "combined_metadata.csv"

OUT_DIR     = FEAT_DIR / "v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_MFCC    = OUT_DIR / "mfcc_features_v2.npy"
OUT_MEL     = OUT_DIR / "mel_features_v2.npy"
OUT_COMB    = OUT_DIR / "combined_features_v2.npy"
OUT_META    = OUT_DIR / "metadata_v2.csv"


def load_center_3s(path: Path, sr: int = SR, dur: float = TARGET_DUR) -> np.ndarray:
    """Load audio and return the center `dur` seconds. No tiling."""
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    target_samples = int(dur * sr)
    if len(y) <= target_samples:
        # Already at or below target length — use as-is
        return y
    # Take center window
    center = len(y) // 2
    half   = target_samples // 2
    return y[center - half : center - half + target_samples]


def extract_mfcc(y: np.ndarray, sr: int = SR) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                  hop_length=HOP_LENGTH, n_fft=N_FFT)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])  # (40,)


def extract_mel(y: np.ndarray, sr: int = SR) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS,
                                          hop_length=HOP_LENGTH, n_fft=N_FFT)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return np.concatenate([mel_db.mean(axis=1), mel_db.std(axis=1)])  # (256,)


def extract_combined(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """MFCC(20) + Chroma(12) + SpectralContrast(7) → mean+std = 78-dim.
    Tonnetz excluded: it requires harmonic separation (~10x slower, ~1.8s/file).
    78-dim is sufficient for the combined feature VAE."""
    mfcc     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                     hop_length=HOP_LENGTH, n_fft=N_FFT)
    chroma   = librosa.feature.chroma_stft(y=y, sr=sr,
                                            hop_length=HOP_LENGTH, n_fft=N_FFT)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr,
                                                  hop_length=HOP_LENGTH, n_fft=N_FFT)
    parts = []
    for feat in [mfcc, chroma, contrast]:
        parts.extend([feat.mean(axis=1), feat.std(axis=1)])
    return np.concatenate(parts)  # (78,)


def main():
    print("=" * 60)
    print(" v2 Feature Extraction: 3s center window (no tiling)")
    print("=" * 60)

    meta = pd.read_csv(META_CSV)
    print(f"Loaded metadata: {len(meta)} tracks")

    mfcc_list, mel_list, comb_list, valid_rows = [], [], [], []
    skipped = 0

    # Skip re-extraction if v2 files already exist (only extract combined if missing)
    mfcc_exists = OUT_MFCC.exists()
    mel_exists  = OUT_MEL.exists()
    comb_exists = OUT_COMB.exists()

    if mfcc_exists and mel_exists and comb_exists:
        print("All v2 feature files already exist — skipping extraction.")
        return

    for idx, row in tqdm(meta.iterrows(), total=len(meta), desc="Extracting"):
        audio_path = Path(str(row.get("path", "")).strip())
        if not audio_path.exists():
            lang  = str(row.get("language", "english")).lower()
            fname = str(row.get("filename", ""))
            audio_path = AUDIO_DIR / lang / fname
        if not audio_path.exists():
            skipped += 1
            continue

        try:
            y = load_center_3s(audio_path)
            mfcc_list.append(extract_mfcc(y))
            mel_list.append(extract_mel(y))
            comb_list.append(extract_combined(y))
            valid_rows.append(row)
        except Exception as e:
            skipped += 1

    print(f"\nExtracted: {len(mfcc_list)}  Skipped: {skipped}")

    mfcc_arr = np.array(mfcc_list, dtype=np.float32)
    mel_arr  = np.array(mel_list,  dtype=np.float32)
    comb_arr = np.array(comb_list, dtype=np.float32)
    meta_out = pd.DataFrame(valid_rows).reset_index(drop=True)

    np.save(OUT_MFCC, mfcc_arr)
    np.save(OUT_MEL,  mel_arr)
    np.save(OUT_COMB, comb_arr)
    meta_out.to_csv(OUT_META, index=False)

    print(f"\nSaved:")
    print(f"  MFCC     : {mfcc_arr.shape}  -> {OUT_MFCC}")
    print(f"  Mel      : {mel_arr.shape}   -> {OUT_MEL}")
    print(f"  Combined : {comb_arr.shape}  -> {OUT_COMB}")
    print(f"  Meta     : {len(meta_out)} rows -> {OUT_META}")

    # Bias check: per-language MFCC std (should be similar after fix)
    print("\nBias check — MFCC overall std per language:")
    print("  (v1 showed English ~51, Bangla ~27 — v2 should be similar)")
    for lang in ["english", "bangla"]:
        mask = meta_out["language"] == lang
        if mask.sum() > 0:
            std = mfcc_arr[mask].std()
            print(f"  {lang:10s}: {mask.sum():6d} tracks,  std = {std:.4f}")


if __name__ == "__main__":
    main()
