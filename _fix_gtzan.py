"""
Fix GTZAN extraction: download, extract properly, filter out macOS resource forks.
"""
import tarfile
import shutil
import urllib.request
import os
from pathlib import Path

DATA_DIR = Path("data")
AUDIO_ENGLISH_DIR = DATA_DIR / "audio" / "english"
temp_dir = DATA_DIR / "temp_gtzan"
temp_dir.mkdir(parents=True, exist_ok=True)
tar_path = temp_dir / "genres.tar.gz"

GTZAN_URL = "https://huggingface.co/datasets/marsyas/gtzan/resolve/main/data/genres.tar.gz"

# Step 1: Download
if not tar_path.exists():
    print("Downloading GTZAN (~1.2 GB)...")
    def progress(bn, bs, ts):
        d = bn * bs
        if ts > 0:
            pct = min(100, d * 100 / ts)
            print(f"\r  {pct:.1f}% ({d/(1024*1024):.0f} MB)", end="", flush=True)
    urllib.request.urlretrieve(GTZAN_URL, str(tar_path), reporthook=progress)
    print(f"\n  Downloaded: {tar_path.stat().st_size / (1024*1024):.0f} MB")
else:
    print(f"  Using existing: {tar_path} ({tar_path.stat().st_size / (1024*1024):.0f} MB)")

# Step 2: Extract
print("Extracting tar.gz...")
with tarfile.open(tar_path, "r:gz") as tar:
    tar.extractall(path=str(temp_dir))
print("  Extraction complete")

# Step 3: Find all audio files (excluding macOS resource forks)
genres = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]

# Look for the genres directory
genres_dir = None
for candidate in [temp_dir / "genres", temp_dir / "genres" / "genres"]:
    if candidate.exists() and any(candidate.iterdir()):
        genres_dir = candidate
        break

if genres_dir is None:
    # Search recursively
    for p in temp_dir.rglob("*"):
        if p.is_dir() and p.name in genres:
            genres_dir = p.parent
            break

print(f"  Found genres dir: {genres_dir}")

# List one genre to see structure
if genres_dir:
    test_genre = genres_dir / genres[0]
    if test_genre.exists():
        files = sorted(test_genre.iterdir())
        print(f"  Sample files in {test_genre.name}/:")
        for f in files[:10]:
            print(f"    {f.name} ({f.stat().st_size} bytes)")

# Step 4: Copy valid audio files
AUDIO_ENGLISH_DIR.mkdir(parents=True, exist_ok=True)
# Clean old corrupt files
for f in AUDIO_ENGLISH_DIR.glob("english_*"):
    f.unlink()

copied = 0
max_per_genre = 25
for genre in genres:
    genre_dir = genres_dir / genre
    if not genre_dir.exists():
        print(f"  WARNING: {genre} not found")
        continue

    # Get all audio files, exclude macOS resource forks (._prefix)
    audio_files = sorted([
        f for f in genre_dir.iterdir()
        if f.is_file()
        and not f.name.startswith("._")
        and f.suffix.lower() in {'.wav', '.au', '.mp3', '.flac', '.ogg'}
        and f.stat().st_size > 1000  # Must be > 1KB
    ])

    selected = audio_files[:max_per_genre]
    for f in selected:
        dest = AUDIO_ENGLISH_DIR / f"english_{genre}_{f.stem}.wav"
        if f.suffix.lower() == '.wav':
            shutil.copy2(f, dest)
        else:
            # Convert .au to .wav using ffmpeg
            os.system(f'ffmpeg -y -i "{f}" "{dest}" -loglevel quiet')
        copied += 1

    print(f"  {genre}: copied {len(selected)} tracks")

print(f"\nTotal: {copied} English tracks copied to {AUDIO_ENGLISH_DIR}")

# Check file sizes
sizes = [f.stat().st_size for f in AUDIO_ENGLISH_DIR.glob("english_*")]
if sizes:
    print(f"  Average size: {sum(sizes)/len(sizes)/1024:.0f} KB")
    print(f"  Total size: {sum(sizes)/(1024*1024):.0f} MB")

# Cleanup
print("Cleaning up temp files...")
shutil.rmtree(temp_dir, ignore_errors=True)
print("Done!")
