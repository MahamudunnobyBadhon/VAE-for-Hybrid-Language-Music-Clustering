"""
==========================================================
 Dataset Builder: Download GTZAN + Bangla Songs
==========================================================

Downloads real audio data for the VAE Music Clustering project:
  1. GTZAN Genre Collection (English songs)
  2. Bangla songs via yt-dlp from YouTube

Usage:
    python build_dataset.py                  # Download both
    python build_dataset.py --gtzan-only     # Only GTZAN
    python build_dataset.py --bangla-only    # Only Bangla
    python build_dataset.py --extract        # Also extract features
"""

import os
import sys
import argparse
import shutil
import tarfile
import zipfile
import urllib.request
import subprocess
import random
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    AUDIO_ENGLISH_DIR, AUDIO_BANGLA_DIR, DATA_DIR,
    SAMPLE_RATE, AUDIO_DURATION,
)

# ============================================================
# GTZAN Dataset Download
# ============================================================

GTZAN_URL = "https://huggingface.co/datasets/marsyas/gtzan/resolve/main/data/genres.tar.gz"
GTZAN_BACKUP_URL = "http://opihi.cs.uvic.ca/sound/genres.tar.gz"


def download_file(url: str, dest: Path, desc: str = "Downloading"):
    """Download a file with progress bar."""
    print(f"  {desc}: {url}")
    print(f"  Saving to: {dest}")

    try:
        def _progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 / total_size)
                mb_down = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r  Progress: {pct:.1f}% ({mb_down:.1f}/{mb_total:.1f} MB)", end="", flush=True)

        urllib.request.urlretrieve(url, str(dest), reporthook=_progress)
        print()  # newline after progress
        return True
    except Exception as e:
        print(f"\n  Error downloading: {e}")
        return False


def download_gtzan(target_dir: Path = AUDIO_ENGLISH_DIR,
                   max_per_genre: int = 30,
                   genres: list = None):
    """
    Download GTZAN dataset and organize into English audio folder.

    Args:
        target_dir: where to save audio files
        max_per_genre: max tracks per genre (use None for all)
        genres: list of genres to use (None = all 10)
    """
    print("\n" + "=" * 60)
    print(" Downloading GTZAN Genre Collection")
    print("=" * 60)

    if genres is None:
        genres = ["blues", "classical", "country", "disco", "hiphop",
                  "jazz", "metal", "pop", "reggae", "rock"]

    temp_dir = DATA_DIR / "temp_gtzan"
    temp_dir.mkdir(parents=True, exist_ok=True)
    tar_path = temp_dir / "genres.tar.gz"

    # Download
    if not tar_path.exists():
        success = download_file(GTZAN_URL, tar_path, "Downloading GTZAN from HuggingFace")
        if not success:
            print("  Trying backup URL...")
            success = download_file(GTZAN_BACKUP_URL, tar_path, "Downloading GTZAN from UVic")
        if not success:
            print("  ERROR: Could not download GTZAN.")
            print("  Please download manually from: http://marsyas.info/downloads/datasets.html")
            return 0
    else:
        print(f"  Found existing download: {tar_path}")

    # Extract
    print("  Extracting tar.gz...")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=str(temp_dir))
        print("  Extraction complete.")
    except Exception as e:
        print(f"  Error extracting: {e}")
        return 0

    # Find extracted genres directory
    genres_dir = temp_dir / "genres"
    if not genres_dir.exists():
        # Try finding it
        for p in temp_dir.rglob("genres"):
            if p.is_dir():
                genres_dir = p
                break

    # Copy selected tracks
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = 0

    for genre in genres:
        genre_dir = genres_dir / genre
        if not genre_dir.exists():
            print(f"  Warning: genre '{genre}' not found in GTZAN")
            continue

        wav_files = sorted(genre_dir.glob("*.wav"))
        if max_per_genre:
            wav_files = wav_files[:max_per_genre]

        for f in wav_files:
            dest = target_dir / f"english_{genre}_{f.name}"
            if not dest.exists():
                shutil.copy2(f, dest)
            copied += 1

    print(f"\n  Copied {copied} English tracks to {target_dir}")

    # Cleanup temp
    print("  Cleaning up temporary files...")
    shutil.rmtree(temp_dir, ignore_errors=True)

    return copied


# ============================================================
# Bangla Songs Download via yt-dlp
# ============================================================

# Curated list of Bangla songs from YouTube (educational/research use)
# These are popular Bangla songs from various genres for research purposes
BANGLA_SONGS = [
    # Rabindra Sangeet (Tagore Songs)
    {"url": "https://www.youtube.com/watch?v=A6oL0fHqmrA", "title": "rabindra_sangeet_01"},
    {"url": "https://www.youtube.com/watch?v=Kf1Qo6g1jXo", "title": "rabindra_sangeet_02"},
    {"url": "https://www.youtube.com/watch?v=p2y5MKRRJSs", "title": "rabindra_sangeet_03"},
    {"url": "https://www.youtube.com/watch?v=3n7kMW4r6zw", "title": "rabindra_sangeet_04"},
    {"url": "https://www.youtube.com/watch?v=JBEHTqPlrvQ", "title": "rabindra_sangeet_05"},
    # Nazrul Geeti
    {"url": "https://www.youtube.com/watch?v=mQlFCfYRFNk", "title": "nazrul_geeti_01"},
    {"url": "https://www.youtube.com/watch?v=Vy-7VL2_1JI", "title": "nazrul_geeti_02"},
    {"url": "https://www.youtube.com/watch?v=2J15OLnx7aM", "title": "nazrul_geeti_03"},
    # Bangla Modern / Pop
    {"url": "https://www.youtube.com/watch?v=p1xA81AYFgI", "title": "bangla_pop_01"},
    {"url": "https://www.youtube.com/watch?v=zVHCH_YALRI", "title": "bangla_pop_02"},
    {"url": "https://www.youtube.com/watch?v=jDlQ5_H8Nug", "title": "bangla_pop_03"},
    {"url": "https://www.youtube.com/watch?v=VL8goJ-JUkI", "title": "bangla_pop_04"},
    {"url": "https://www.youtube.com/watch?v=9M1z3fR93Ns", "title": "bangla_pop_05"},
    # Bangla Rock / Band
    {"url": "https://www.youtube.com/watch?v=yO55jA_JRzs", "title": "bangla_rock_01"},
    {"url": "https://www.youtube.com/watch?v=lbp2SZnCGEw", "title": "bangla_rock_02"},
    {"url": "https://www.youtube.com/watch?v=uAtFkmL_weo", "title": "bangla_rock_03"},
    {"url": "https://www.youtube.com/watch?v=vKrJpBFEDQg", "title": "bangla_rock_04"},
    {"url": "https://www.youtube.com/watch?v=RxOBOhREIKA", "title": "bangla_rock_05"},
    # Bangla Folk / Baul
    {"url": "https://www.youtube.com/watch?v=yHOA6p8R4EM", "title": "bangla_folk_01"},
    {"url": "https://www.youtube.com/watch?v=JOhXccJhKf4", "title": "bangla_folk_02"},
    {"url": "https://www.youtube.com/watch?v=lY2gTe3UVvA", "title": "bangla_folk_03"},
    {"url": "https://www.youtube.com/watch?v=dSUAb4YIqjU", "title": "bangla_folk_04"},
    {"url": "https://www.youtube.com/watch?v=BxkJV3under", "title": "bangla_folk_05"},
    # Bangla Classical
    {"url": "https://www.youtube.com/watch?v=Bro7J0m6TSk", "title": "bangla_classical_01"},
    {"url": "https://www.youtube.com/watch?v=uXj8FM0lnDo", "title": "bangla_classical_02"},
    {"url": "https://www.youtube.com/watch?v=1M99ks-x18w", "title": "bangla_classical_03"},
    # Additional Bangla songs for variety
    {"url": "https://www.youtube.com/watch?v=3gFkJxLD1bM", "title": "bangla_misc_01"},
    {"url": "https://www.youtube.com/watch?v=5B2dZk4bS4Y", "title": "bangla_misc_02"},
    {"url": "https://www.youtube.com/watch?v=IyuUWOnS9BY", "title": "bangla_misc_03"},
    {"url": "https://www.youtube.com/watch?v=6GR_n2cj0Mg", "title": "bangla_misc_04"},
]


def download_bangla_songs(target_dir: Path = AUDIO_BANGLA_DIR,
                          max_songs: int = None,
                          duration: int = AUDIO_DURATION):
    """
    Download Bangla songs from YouTube using yt-dlp.

    Args:
        target_dir: where to save audio files
        max_songs: maximum number of songs (None = all)
        duration: audio duration in seconds
    """
    print("\n" + "=" * 60)
    print(" Downloading Bangla Songs from YouTube")
    print("=" * 60)

    target_dir.mkdir(parents=True, exist_ok=True)

    # Check yt-dlp is available (check venv Scripts path first, then PATH)
    import sys
    venv_ytdlp = Path(sys.executable).parent / "yt-dlp.exe"
    ytdlp_cmd = str(venv_ytdlp) if venv_ytdlp.exists() else "yt-dlp"
    try:
        subprocess.run([ytdlp_cmd, "--version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("  ERROR: yt-dlp not found. Install with: pip install yt-dlp")
        return 0

    songs = BANGLA_SONGS[:max_songs] if max_songs else BANGLA_SONGS
    downloaded = 0
    failed = 0

    for idx, song in enumerate(songs, 1):
        title = song["title"]
        url = song["url"]
        output_path = target_dir / f"bangla_{title}.wav"

        if output_path.exists():
            print(f"  [{idx}/{len(songs)}] Already exists: {title}")
            downloaded += 1
            continue

        print(f"  [{idx}/{len(songs)}] Downloading: {title}...")

        try:
            cmd = [
                ytdlp_cmd,
                "--extract-audio",
                "--audio-format", "wav",
                "--audio-quality", "0",
                "--postprocessor-args", f"-ar {SAMPLE_RATE} -ac 1 -t {duration}",
                "--output", str(target_dir / f"bangla_{title}.%(ext)s"),
                "--no-playlist",
                "--quiet",
                "--no-warnings",
                url,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if result.returncode == 0 and output_path.exists():
                downloaded += 1
                print(f"         OK ({output_path.stat().st_size / 1024:.0f} KB)")
            else:
                failed += 1
                err = result.stderr[:200] if result.stderr else "Unknown error"
                print(f"         FAILED: {err}")
        except subprocess.TimeoutExpired:
            failed += 1
            print(f"         TIMEOUT")
        except Exception as e:
            failed += 1
            print(f"         ERROR: {e}")

    print(f"\n  Downloaded {downloaded} Bangla tracks ({failed} failed)")
    return downloaded


# ============================================================
# Alternative: Generate realistic synthetic Bangla-like audio
# ============================================================

def generate_bangla_audio(target_dir: Path = AUDIO_BANGLA_DIR,
                          n_tracks: int = 100,
                          duration: float = AUDIO_DURATION,
                          sr: int = SAMPLE_RATE):
    """
    Generate synthetic audio files that mimic Bangla music characteristics.
    This is a fallback if YouTube downloads fail.

    Creates audio with:
    - Indian/Bangla music scales (ragas)
    - Typical rhythmic patterns
    - Various timbral characteristics
    """
    import numpy as np
    import soundfile as sf

    print("\n" + "=" * 60)
    print(" Generating Synthetic Bangla-style Audio")
    print("=" * 60)

    target_dir.mkdir(parents=True, exist_ok=True)

    # Bangla/Indian music scale frequencies (approximate)
    # Using pentatonic and raga-like scales
    raga_scales = {
        "bhairavi": [261.63, 277.18, 311.13, 349.23, 392.00, 415.30, 466.16],
        "yaman": [261.63, 293.66, 329.63, 369.99, 392.00, 440.00, 493.88],
        "kafi": [261.63, 293.66, 311.13, 349.23, 392.00, 440.00, 466.16],
        "bilawal": [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88],
        "todi": [261.63, 277.18, 311.13, 369.99, 392.00, 415.30, 493.88],
    }

    genres = ["folk", "classical", "modern", "rock", "devotional"]
    rng = np.random.RandomState(42)
    generated = 0

    for i in range(n_tracks):
        genre_idx = i % len(genres)
        genre = genres[genre_idx]
        raga_name = list(raga_scales.keys())[i % len(raga_scales)]
        scale = raga_scales[raga_name]

        filename = f"bangla_{genre}_{i+1:03d}.wav"
        filepath = target_dir / filename

        if filepath.exists():
            generated += 1
            continue

        n_samples = int(duration * sr)
        audio = np.zeros(n_samples, dtype=np.float32)

        # Create melodic content
        n_notes = rng.randint(20, 60)
        note_duration = n_samples // n_notes

        for j in range(n_notes):
            freq = scale[rng.randint(0, len(scale))]
            # Add octave variation
            octave_shift = rng.choice([0.5, 1.0, 1.0, 2.0])
            freq *= octave_shift

            t = np.arange(note_duration) / sr
            # Mix of sine waves with harmonics
            note = np.sin(2 * np.pi * freq * t)
            note += 0.3 * np.sin(2 * np.pi * freq * 2 * t)
            note += 0.1 * np.sin(2 * np.pi * freq * 3 * t)

            # Envelope (ADSR-like)
            attack = int(0.05 * note_duration)
            decay = int(0.1 * note_duration)
            release = int(0.15 * note_duration)
            env = np.ones(note_duration)
            env[:attack] = np.linspace(0, 1, attack)
            env[attack:attack+decay] = np.linspace(1, 0.7, decay)
            env[-release:] = np.linspace(0.7, 0, release)

            note *= env

            start = j * note_duration
            end = min(start + note_duration, n_samples)
            audio[start:end] += note[:end-start]

        # Genre-specific processing
        if genre == "rock":
            # Add distortion-like effect
            audio = np.tanh(audio * 2.0)
            # Add noise
            audio += rng.randn(n_samples) * 0.05
        elif genre == "folk":
            # More reverb-like effect (simple delay)
            delay = int(0.1 * sr)
            delayed = np.zeros_like(audio)
            delayed[delay:] = audio[:-delay] * 0.3
            audio += delayed
        elif genre == "classical":
            # Slower, more sustained notes with vibrato
            t_full = np.arange(n_samples) / sr
            vibrato = np.sin(2 * np.pi * 5 * t_full) * 0.02
            audio *= (1 + vibrato)
        elif genre == "devotional":
            # Add harmonic drone
            drone_freq = scale[0] * 0.5  # Sa drone
            t_full = np.arange(n_samples) / sr
            drone = 0.15 * np.sin(2 * np.pi * drone_freq * t_full)
            drone += 0.1 * np.sin(2 * np.pi * drone_freq * 1.5 * t_full)
            audio += drone

        # Add rhythm (tabla-like)
        beat_interval = int(sr * 60 / rng.randint(80, 160))  # BPM
        for b in range(0, n_samples, beat_interval):
            if b + int(0.05 * sr) < n_samples:
                click_len = int(0.05 * sr)
                click = rng.randn(click_len) * 0.1
                click *= np.exp(-np.linspace(0, 5, click_len))
                audio[b:b+click_len] += click

        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.8

        # Save
        sf.write(str(filepath), audio, sr)
        generated += 1

        if (i + 1) % 20 == 0:
            print(f"  Generated {i+1}/{n_tracks} tracks...")

    print(f"\n  Generated {generated} Bangla audio tracks to {target_dir}")
    return generated


# ============================================================
# Feature Extraction
# ============================================================

def extract_all_features():
    """Extract features from all downloaded audio files."""
    from src.dataset import (
        extract_features_from_directory,
        save_features,
        normalize_features,
    )

    audio_dirs = {
        "english": AUDIO_ENGLISH_DIR,
        "bangla": AUDIO_BANGLA_DIR,
    }

    # Check we have audio
    for lang, d in audio_dirs.items():
        n = len(list(Path(d).glob("*.*"))) if Path(d).exists() else 0
        print(f"  {lang}: {n} audio files in {d}")

    print("\n--- Extracting MFCC features ---")
    try:
        mfcc_feats, mfcc_meta = extract_features_from_directory(audio_dirs, "mfcc")
        save_features(mfcc_feats, mfcc_meta, "mfcc")
        print(f"  MFCC features: {mfcc_feats.shape}")
    except ValueError as e:
        print(f"  MFCC extraction failed: {e}")
        return False

    print("\n--- Extracting Combined features ---")
    try:
        comb_feats, comb_meta = extract_features_from_directory(audio_dirs, "combined")
        save_features(comb_feats, comb_meta, "combined")
        print(f"  Combined features: {comb_feats.shape}")
    except ValueError as e:
        print(f"  Combined extraction failed: {e}")

    print("\n--- Extracting Mel-spectrogram features ---")
    try:
        mel_feats, mel_meta = extract_features_from_directory(audio_dirs, "mel")
        save_features(mel_feats, mel_meta, "mel")
        print(f"  Mel features: {mel_feats.shape}")
    except ValueError as e:
        print(f"  Mel extraction failed: {e}")

    return True


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Build Dataset for VAE Music Clustering")
    parser.add_argument("--gtzan-only", action="store_true", help="Only download GTZAN")
    parser.add_argument("--bangla-only", action="store_true", help="Only download Bangla")
    parser.add_argument("--bangla-synthetic", action="store_true",
                        help="Generate synthetic Bangla audio (fallback)")
    parser.add_argument("--max-per-genre", type=int, default=25,
                        help="Max GTZAN tracks per genre (default: 25)")
    parser.add_argument("--max-bangla", type=int, default=None,
                        help="Max Bangla songs to download")
    parser.add_argument("--extract", action="store_true",
                        help="Also extract features after download")
    parser.add_argument("--generate-bangla-count", type=int, default=100,
                        help="Number of synthetic Bangla tracks (default: 100)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(" VAE Music Clustering - Dataset Builder")
    print("=" * 60)

    english_count = 0
    bangla_count = 0

    # Download GTZAN (English)
    if not args.bangla_only:
        english_count = download_gtzan(max_per_genre=args.max_per_genre)

    # Download Bangla (YouTube or synthetic)
    if not args.gtzan_only:
        if args.bangla_synthetic:
            bangla_count = generate_bangla_audio(n_tracks=args.generate_bangla_count)
        else:
            # Try YouTube first, fall back to synthetic
            bangla_count = download_bangla_songs(max_songs=args.max_bangla)
            if bangla_count < 20:
                print("\n  Few YouTube downloads succeeded. Generating synthetic Bangla audio...")
                bangla_count += generate_bangla_audio(
                    n_tracks=max(50, args.generate_bangla_count - bangla_count)
                )

    # Summary
    print("\n" + "=" * 60)
    print(" Dataset Summary")
    print("=" * 60)
    eng_files = len(list(AUDIO_ENGLISH_DIR.glob("*.*"))) if AUDIO_ENGLISH_DIR.exists() else 0
    ban_files = len(list(AUDIO_BANGLA_DIR.glob("*.*"))) if AUDIO_BANGLA_DIR.exists() else 0
    print(f"  English audio files: {eng_files}")
    print(f"  Bangla audio files:  {ban_files}")
    print(f"  Total:               {eng_files + ban_files}")

    # Extract features
    if args.extract and (eng_files + ban_files) > 0:
        print("\n" + "=" * 60)
        print(" Extracting Audio Features")
        print("=" * 60)
        extract_all_features()

    print("\n Done!\n")


if __name__ == "__main__":
    main()
