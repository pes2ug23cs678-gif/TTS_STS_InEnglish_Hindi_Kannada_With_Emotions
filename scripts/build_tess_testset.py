#!/usr/bin/env python3
"""Build testset.csv from TESS file names (SPEAKER_word_emotion.wav).

Usage: python3 build_tess_testset.py <tess_folder> testset.csv
"""
import csv
import sys
from collections import Counter
from pathlib import Path

# TESS emotion suffix -> label set used by ab_emotion.py.
# Check these against the emotion names in your copy of the dataset.
EMOTION_TO_GOLD = {
    "angry": "anger", "disgust": "disgust", "fear": "fear", "happy": "joy",
    "neutral": "neutral", "sad": "sadness", "ps": "surprise",
}

if len(sys.argv) != 3:
    sys.exit("usage: python3 build_tess_testset.py YAF_disgust entered_texts.csv")
root, out = Path(sys.argv[1]), Path(sys.argv[2])
rows, unmapped, seen = [], Counter(), Counter()
for i, wav in enumerate(sorted(root.rglob("*.wav"))):
    parts = wav.stem.split("_")
    if len(parts) != 3:
        continue
    _speaker, word, emo = parts
    seen[emo.lower()] += 1
    gold = EMOTION_TO_GOLD.get(emo.lower())
    if gold is None:
        unmapped[emo.lower()] += 1
        gold = ""
    rows.append({"id": f"{i:05d}", "text": f"Say the word {word}",
                 "audio_path": str(wav), "gold": gold})

with open(out, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["id", "text", "audio_path", "gold"])
    w.writeheader()
    w.writerows(rows)

print("emotion suffixes found:", dict(seen))
print("unmapped (left unlabelled):", dict(unmapped))
print(f"wrote {len(rows)} rows to {out}")
