#!/usr/bin/env python3
"""Compare emotion classifiers on a hand-labelled test set, fairly.

Fairness rules built into this script:
  - You label the sentences BEFORE looking at any model output.
    'make-template' writes the sentences with an empty label column.
  - Every model gets the same sentences and the same scoring code.
  - The 28-label GoEmotions models are mapped to the same 7 labels using GO_TO_BASIC.
    Freeze that table before running. Do not change it after seeing results.

Usage:
  python emotion_compare.py make-template     # step 1: writes emotion_test_set.csv
  (fill in the 'label' column by hand with one of: anger, disgust, fear, joy, neutral, sadness, surprise)
  python emotion_compare.py evaluate          # step 2: runs every model in MODELS

Install once:  pip install transformers torch
The first run downloads each model, and the RoBERTa-large models are large.
"""
import argparse
import csv
import gc
import pathlib
import statistics
import sys
import time

BASE = pathlib.Path(__file__).resolve().parent
TEST_SET = BASE / "emotion_test_set.csv"
RESULTS = BASE / "emotion_compare_results.csv"
SENTENCE_SOURCES = [BASE / "translation_cases.csv", BASE / "demo_sentences.txt"]

LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

MODELS = {
    "incumbent: distilroberta-base (7 labels)": "j-hartmann/emotion-english-distilroberta-base",
    "roberta-large (7 labels)": "j-hartmann/emotion-english-roberta-large",
    "roberta-large GoEmotions (28 labels)": "tasinhoque/roberta-large-go-emotions",
    "roberta-base GoEmotions (28 labels)": "SamLowe/roberta-base-go_emotions",
}

# GoEmotions label -> one of the 7 basic labels. Frozen before testing.
GO_TO_BASIC = {
    "anger": "anger", "annoyance": "anger",
    "disgust": "disgust", "disapproval": "disgust",
    "fear": "fear", "nervousness": "fear",
    "joy": "joy", "amusement": "joy", "excitement": "joy", "gratitude": "joy",
    "love": "joy", "optimism": "joy", "pride": "joy", "relief": "joy",
    "admiration": "joy", "approval": "joy", "caring": "joy", "desire": "joy",
    "sadness": "sadness", "disappointment": "sadness", "embarrassment": "sadness",
    "grief": "sadness", "remorse": "sadness",
    "surprise": "surprise", "confusion": "surprise", "curiosity": "surprise",
    "realization": "surprise",
    "neutral": "neutral",
}


def make_template():
    if TEST_SET.exists():
        sys.exit(f"{TEST_SET.name} already exists. Delete it first if you want a new template.")
    sentences = []
    csv_source = SENTENCE_SOURCES[0]
    with open(csv_source, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            text = (row.get("english") or "").strip()
            if text:
                sentences.append(text)
    txt_source = SENTENCE_SOURCES[1]
    for raw in txt_source.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            sentences.append(line)
    with open(TEST_SET, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text", "label"])
        for i, text in enumerate(sentences, start=1):
            writer.writerow([i, text, ""])
    print(f"Wrote {len(sentences)} sentences to {TEST_SET.name}.")
    print("Fill in the label column by hand, then run: python emotion_compare.py evaluate")


def load_labelled():
    if not TEST_SET.exists():
        sys.exit("Run 'make-template' first.")
    rows = []
    with open(TEST_SET, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            label = (row.get("label") or "").strip().lower()
            if not label:
                continue
            if label not in LABELS:
                sys.exit(f"Row {row['id']}: label '{label}' is not one of {LABELS}")
            rows.append((row["text"], label))
    if not rows:
        sys.exit("No labelled rows yet. Fill in the label column first.")
    return rows


def to_basic_scores(all_scores):
    """Adds up the model's probabilities into the 7 basic labels."""
    totals = {label: 0.0 for label in LABELS}
    unmapped = set()
    for item in all_scores:
        raw = item["label"].lower()
        if raw in LABELS:
            basic = raw
        else:
            basic = GO_TO_BASIC.get(raw)
        if basic is None:
            unmapped.add(raw)
            continue
        totals[basic] += float(item["score"])
    return totals, unmapped


def macro_f1(results):
    scores = []
    for label in sorted({g for _, g, _, _ in results}):
        tp = sum(1 for _, gold, pred, _ in results if gold == label and pred == label)
        fp = sum(1 for _, gold, pred, _ in results if gold != label and pred == label)
        fn = sum(1 for _, gold, pred, _ in results if gold == label and pred != label)
        if tp + fp + fn == 0:
            continue  # label never appears in gold or predictions
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else 0.0

QUANTISE = True  # int8 dynamic quantisation for every model, so the speed comparison is fair

def load_pipeline(model_id):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).eval()
    if QUANTISE:
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    return pipeline("text-classification", model=model, tokenizer=tok, top_k=None, device=-1)

def evaluate():
    from transformers import pipeline

    rows = load_labelled()
    gold_counts = {label: 0 for label in LABELS}
    for _, gold in rows:
        gold_counts[gold] += 1
    print(f"Labelled sentences: {len(rows)}")
    print("Gold label counts:", gold_counts)

    all_rows = []
    summary = []
    for name, model_id in MODELS.items():
        print(f"\n>>> {name}  ({model_id})", flush=True)
        clf = load_pipeline(model_id)
        clf(rows[0][0])  # warm-up call, not timed

        results, times, unmapped_all = [], [], set()
        for text, gold in rows:
            t0 = time.perf_counter()
            out = clf(text)
            times.append(time.perf_counter() - t0)
            if out and isinstance(out[0], list):
                out = out[0]
            totals, unmapped = to_basic_scores(out)
            unmapped_all |= unmapped
            pred = max(LABELS, key=lambda label: totals[label])
            results.append((text, gold, pred, round(times[-1], 3)))

        accuracy = sum(1 for _, gold, pred, _ in results if gold == pred) / len(results)
        f1 = macro_f1(results)
        median_s = statistics.median(times)
        summary.append((name, accuracy, f1, median_s))
        for text, gold, pred, seconds in results:
            all_rows.append([name, text, gold, pred, seconds])
        if unmapped_all:
            print(f"  Warning: labels not in the mapping table (counted as no vote): {sorted(unmapped_all)}")

        del clf
        gc.collect()

    with open(RESULTS, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "text", "gold", "predicted", "seconds"])
        writer.writerows(all_rows)

    print("\n=== Summary (same sentences, same scoring) ===")
    print(f"{'model':<44}{'accuracy':>10}{'macro-F1':>10}{'median s':>10}")
    for name, accuracy, f1, median_s in summary:
        print(f"{name:<44}{accuracy:>10.3f}{f1:>10.3f}{median_s:>10.2f}")
    print(f"\nPer-sentence results written to {RESULTS.name}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare emotion classifiers on a labelled test set.")
    parser.add_argument("command", choices=["make-template", "evaluate"])
    args = parser.parse_args()
    if args.command == "make-template":
        make_template()
    else:
        evaluate()
