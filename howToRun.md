# With synthetic data (for testing):
python run_easy_task.py --skip-umap --epochs 50

# With real audio files (place .wav/.mp3 in data/audio/english/ and data/audio/bangla/):
python run_easy_task.py --use-real-audio --epochs 100