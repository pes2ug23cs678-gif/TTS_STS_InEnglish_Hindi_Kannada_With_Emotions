#!/usr/bin/env python3
"""Type sentences, see what each emotion model predicts, and score them against your expected labels.
Needs ab_emotions.py in the same folder.
Optional environment variables:
  XLMR_MODEL, XLMR_LABEL_MAP (JSON), EMOTION2VEC_MODEL
Every entry is appended to entered_texts.csv.
"""
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from ab_emotions import LABELS, Emotion2Vec, TextClassifier
LOG = Path("entered_texts.csv")

def load_models():
    models = [TextClassifier("distilroberta", "j-hartmann/emotion-english-distilroberta-base")]
    if os.getenv("XLMR_MODEL"):
        models.append(TextClassifier("xlmr", os.environ["XLMR_MODEL"],
                                     json.loads(os.getenv("XLMR_LABEL_MAP", "{}"))))
    if os.getenv("EMOTION2VEC_MODEL"):
        models.append(Emotion2Vec(os.environ["EMOTION2VEC_MODEL"]))
    return models

def ask(prompt):
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt) as exc:
        raise SystemExit from exc

def ask_gold():
    while True:
        g = ask("  expected emotion " + str(LABELS) + " (Enter to skip): ").lower()
        if g == "" or g in LABELS:
            return g
        print("  not a valid label")

def precision_report(entries, models):
    labeled = [(gold, preds) for gold, preds in entries if gold]
    if not labeled:
        print("\nNo expected labels entered, so nothing was scored.")
        return
    print(f"\n=== {len(labeled)} labelled entries ===")
    for m in models:
        pairs = [(gold, preds[m.name]) for gold, preds in labeled if preds.get(m.name)]
        if not pairs:
            print(f"\n{m.name}: no scorable entries")
            continue
        acc = sum(g == p for g, p in pairs) / len(pairs)
        print(f"\n{m.name}: accuracy {acc:.2f} on {len(pairs)} entries")
        print(f"  {'label':<10}{'precision':>10}{'recall':>9}{'predicted':>11}{'expected':>10}")
        for lab in LABELS:
            n_pred = sum(p == lab for _, p in pairs)
            n_gold = sum(g == lab for g, _ in pairs)
            if n_pred == 0 and n_gold == 0:
                continue
            tp = sum(g == lab and p == lab for g, p in pairs)
            prec = f"{tp / n_pred:.2f}" if n_pred else "-"
            rec = f"{tp / n_gold:.2f}" if n_gold else "-"
            print(f"  {lab:<10}{prec:>10}{rec:>9}{n_pred:>11}{n_gold:>10}")

def main():
    models = load_models()
    has_e2v = any(isinstance(m, Emotion2Vec) for m in models)
    fields = ["timestamp", "text", "gold", "audio_path"]
    for m in models:
        fields += [f"{m.name}_pred", f"{m.name}_top_prob"]

    new_file = not LOG.exists()
    entries = []
    with open(LOG, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            writer.writeheader()
        print("Type a sentence. Empty line to finish.")
        while True:
            text = ask("\ntext> ")
            if not text:
                break
            gold = ask_gold()
            audio = ask("  audio file path (Enter if none): ") if has_e2v else ""
            row = {"timestamp": datetime.now().isoformat(timespec="seconds"),
                   "text": text, "gold": gold, "audio_path": audio}
            preds = {}
            for m in models:
                probs = m.predict({"text": text, "audio_path": audio})
                if probs is None:
                    preds[m.name] = None
                    row[f"{m.name}_pred"] = ""
                    row[f"{m.name}_top_prob"] = ""
                    print(f"  {m.name:<13} n/a")
                    continue
                top = max(probs, key=probs.get)
                preds[m.name] = top
                row[f"{m.name}_pred"] = top
                row[f"{m.name}_top_prob"] = round(probs[top], 3)
                ranked = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
                detail = ", ".join(f"{k} {v:.2f}" for k, v in ranked)
                print(f"  {m.name:<13} {top:<9} {probs[top]:.2f}   ({detail})")
            writer.writerow(row)
            f.flush()
            entries.append((gold, preds))

    precision_report(entries, models)
    print(f"\nAll entries are in {LOG.resolve()}")

if __name__ == "__main__":
    main()
