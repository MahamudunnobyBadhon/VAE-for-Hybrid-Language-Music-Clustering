"""
Download real Bangla music using yt-dlp YouTube search.
No hardcoded URLs - searches by genre keywords and downloads top results.

Usage:
    python download_bangla.py              # Download ~50 songs (default)
    python download_bangla.py --n 100      # Download ~100 songs
    python download_bangla.py --dry-run    # List what would be downloaded
"""

import subprocess
import sys
import argparse
from pathlib import Path

BANGLA_DIR = Path("data/audio/bangla")
VENV_YTDLP = Path(sys.executable).parent / "yt-dlp.exe"
YTDLP = str(VENV_YTDLP) if VENV_YTDLP.exists() else "yt-dlp"
FFMPEG_DIR = str(Path(__file__).parent)  # project root has ffmpeg.exe

# Search queries: (query, genre_label, n_songs)
SEARCH_QUERIES = [
    ("Rabindra Sangeet Bengali classical vocal",         "classical",   8),
    ("Nazrul Geeti Bengali song",                        "classical",   5),
    ("Baul song Bangladesh folk music",                  "folk",        8),
    ("Bhatiali boat song Bangladesh",                    "folk",        5),
    ("Bengali modern song Bangla gaan",                  "modern",      8),
    ("Bangla band song rock music Bangladesh",           "rock",        8),
    ("Bengali devotional song kirtan bhajan",            "devotional",  8),
    ("Lalon Fakir folk song Bangladesh",                 "folk",        5),
    ("Bangla pop song new",                              "modern",      5),
]


def run_ytdlp(args, timeout=120):
    env = {"PATH": FFMPEG_DIR + ";" + __import__("os").environ.get("PATH", "")}
    return subprocess.run(
        [YTDLP] + args,
        capture_output=True, text=True,
        timeout=timeout, env={**__import__("os").environ, **env}
    )


def download_genre(query, genre, n, dry_run=False):
    search = f"ytsearch{n}:{query}"
    out_tmpl = str(BANGLA_DIR / f"bangla_{genre}_%(autonumber)03d.%(ext)s")

    if dry_run:
        print(f"  [{genre}] Would search: {query!r} ({n} songs)")
        return 0

    print(f"\n  Searching: {query!r} ({n} songs)...")
    args = [
        search,
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "5",          # medium quality, faster
        "--ffmpeg-location", FFMPEG_DIR,
        "--output", out_tmpl,
        "--no-playlist",
        "--match-filter", "duration < 600",  # skip >10min videos
        "--quiet",
        "--progress",
        "--no-warnings",
        "--ignore-errors",
    ]

    result = run_ytdlp(args, timeout=300)
    downloaded = len(list(BANGLA_DIR.glob(f"bangla_{genre}_*.wav")))
    if result.returncode != 0 and result.stderr:
        print(f"    Warning: {result.stderr[:200]}")
    print(f"    Done. {genre} files now: {downloaded}")
    return downloaded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None,
                        help="Total target songs (scales queries proportionally)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    BANGLA_DIR.mkdir(parents=True, exist_ok=True)

    # Check yt-dlp
    r = run_ytdlp(["--version"])
    if r.returncode != 0:
        print("ERROR: yt-dlp not working. Run: .venv/Scripts/pip install yt-dlp")
        sys.exit(1)
    print(f"yt-dlp version: {r.stdout.strip()}")
    print(f"ffmpeg location: {FFMPEG_DIR}")
    print(f"Output dir: {BANGLA_DIR.absolute()}\n")

    # Scale query counts if --n specified
    queries = list(SEARCH_QUERIES)
    if args.n:
        total_default = sum(n for _, _, n in queries)
        scale = args.n / total_default
        queries = [(q, g, max(1, int(n * scale))) for q, g, n in queries]

    total = 0
    for query, genre, n in queries:
        total += download_genre(query, genre, n, args.dry_run)

    if not args.dry_run:
        all_bangla = list(BANGLA_DIR.glob("bangla_*.wav"))
        print(f"\nTotal Bangla WAV files: {len(all_bangla)}")
        by_genre = {}
        for f in all_bangla:
            parts = f.stem.split("_")
            g = parts[1] if len(parts) > 1 else "unknown"
            by_genre[g] = by_genre.get(g, 0) + 1
        for g, c in sorted(by_genre.items()):
            print(f"  {g}: {c}")


if __name__ == "__main__":
    main()
