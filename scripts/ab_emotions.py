#!/usr/bin/env python3
"""A/B benchmark for emotion classifiers on the same labelled rows.

Backends:
  - distilroberta : text, same model the app uses today
  - xlmr          : text, any XLM-R emotion checkpoint you pass with --xlmr
  - emotion2vec   : audio, read from audio_path (funasr, iic/emotion2vec_plus_large)

CSV columns: id, text, audio_path, gold
Rows without a valid gold label are skipped. Metrics use only rows every
selected backend answered, so all models are scored on the same set.
"""
import argparse
import csv
import json
import random
import statistics
import sys
import time
from pathlib import Path

LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

# emotion2vec_plus_large outputs: angry, happy, neutral, sad, unknown.
# It cannot produce disgust, fear or surprise. "unknown" is dropped.
EMOTION2VEC_MAP = {
    "angry": "anger", "happy": "joy", "neutral": "neutral", "sad": "sadness",
}


def normalize(probs):
    """Restrict to LABELS and renormalise so every backend yields a distribution over the same labels."""
    total = sum(probs.get(l, 0.0) for l in LABELS)
    return {l: (probs.get(l, 0.0) / total if total else 0.0) for l in LABELS}


class TextClassifier:
    """DistilRoBERTa or XLM-R. One code path for both, so only the checkpoint differs."""

    def __init__(self, name, model_id, label_map=None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.name = name
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.mod = AutoModelForSequenceClassification.from_pretrained(model_id).eval()
        label_map = {k.lower(): v.lower() for k, v in (label_map or {}).items()}
        self.index_to_label = {}
        for i, raw in self.mod.config.id2label.items():
            raw = str(raw).lower()
            mapped = label_map.get(raw, raw)
            if mapped not in LABELS:
                raise ValueError(
                    f"{name}: checkpoint label '{raw}' is not one of {LABELS}; "
                    f"map it with --xlmr-label-map")
            self.index_to_label[int(i)] = mapped

    def predict(self, row):
        enc = self.tok(row["text"], return_tensors="pt", truncation=True)
        with self.torch.no_grad():
            logits = self.mod(**enc).logits[0]
        probs = self.torch.softmax(logits, dim=-1).tolist()
        out = {}
        for i, p in enumerate(probs):
            lab = self.index_to_label[i]
            out[lab] = out.get(lab, 0.0) + p
        return normalize(out)


class Emotion2Vec:
    """Audio emotion backend wrapping a funasr emotion2vec model."""
    name = "emotion2vec"

    def __init__(self, model_id):
        from funasr import AutoModel
        self.model = AutoModel(model=model_id)

    def predict(self, row):
        if not row.get("audio_path"):
            return None
        res = self.model.generate(row["audio_path"], granularity="utterance", extract_embedding=False)
        item = res[0]
        out = {}
        for raw, score in zip(item["labels"], item["scores"]):
            key = str(raw).rsplit("/", maxsplit=1)[-1].strip().lower()   # handles "生气/angry" as well as "angry"
            lab = EMOTION2VEC_MAP.get(key)
            if lab:
                out[lab] = out.get(lab, 0.0) + float(score)
        if not out:
            return None
        return normalize(out)


def score(gold, pred):
    """Accuracy and macro-F1 without extra dependencies."""
    acc = sum(g == p for g, p in zip(gold, pred)) / len(gold)
    f1s = []
    for lab in sorted(set(gold) | set(pred)):
        tp = sum(g == lab and p == lab for g, p in zip(gold, pred))
        fp = sum(g != lab and p == lab for g, p in zip(gold, pred))
        fn = sum(g == lab and p != lab for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return acc, sum(f1s) / len(f1s)


def p90(values):
    s = sorted(values)
    return s[int(0.9 * (len(s) - 1))]


def main():
    # Single CLI entry point: argument parsing plus the full benchmark loop.
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--testset", required=True)
    ap.add_argument("--out", default="ab_emotion_out")
    ap.add_argument("--distil", default="j-hartmann/emotion-english-distilroberta-base")
    ap.add_argument("--xlmr", help="XLM-R emotion checkpoint ID (choose and verify one)")
    ap.add_argument("--xlmr-label-map", default="{}", help='JSON, e.g. \'{"happiness": "joy"}\'')
    ap.add_argument("--emotion2vec", help="funasr model ID, e.g. iic/emotion2vec_plus_large")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--no-text", action="store_true", help="skip DistilRoBERTa and XLM-R (audio-only run)")
    ap.add_argument("--all-labels", action="store_true",
                    help="keep rows whose label emotion2vec cannot output (default: drop them when emotion2vec runs)")
    ap.add_argument("--per-label", type=int, default=0, help="random sample of N rows per gold label (0 = all)")
    ap.add_argument("--seed", type=int, default=0, help="seed for --per-label sampling")
    args = ap.parse_args()

    with open(args.testset, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for i, r in enumerate(rows):
        r["id"] = (r.get("id") or str(i)).strip()
        r["gold"] = (r.get("gold") or "").strip().lower()
    valid = [r for r in rows if r["gold"] in LABELS]
    if len(valid) < len(rows):
        print(f"skipped {len(rows) - len(valid)} rows without a valid gold label", file=sys.stderr)
    if not valid:
        sys.exit("no labelled rows: fill the gold column first")

    if args.emotion2vec and not args.all_labels:
        allowed = set(EMOTION2VEC_MAP.values())
        dropped = [r for r in valid if r["gold"] not in allowed]
        valid = [r for r in valid if r["gold"] in allowed]
        print(f"emotion2vec cannot output these labels, so {len(dropped)} rows were dropped "
              f"(use --all-labels to keep them)", file=sys.stderr)
        if not valid:
            sys.exit("no rows left after restricting to emotion2vec labels")

    if args.per_label > 0:
        rng = random.Random(args.seed)
        sampled = []
        for lab in LABELS:
            group = [r for r in valid if r["gold"] == lab]
            sampled += rng.sample(group, min(args.per_label, len(group)))
        valid = sampled

    backends = []
    if not args.no_text:
        backends.append(TextClassifier("distilroberta", args.distil))
        if args.xlmr:
            backends.append(TextClassifier("xlmr", args.xlmr, json.loads(args.xlmr_label_map)))
    if args.emotion2vec:
        backends.append(Emotion2Vec(args.emotion2vec))
    if not backends:
        sys.exit("no backend selected")

    for b in backends:                       # warm up so model loading is not timed
        for r in valid[:args.warmup]:
            b.predict(r)

    results = {b.name: {} for b in backends}  # name -> id -> (label, probs, seconds)
    for r in valid:
        for b in backends:
            t0 = time.perf_counter()
            probs = b.predict(r)
            dt = time.perf_counter() - t0
            if probs is not None:
                results[b.name][r["id"]] = (max(probs, key=probs.get), probs, dt)

    common = [r for r in valid if all(r["id"] in results[b.name] for b in backends)]
    if not common:
        sys.exit("no row was answered by every backend")

    gold = [r["gold"] for r in common]
    summary = {
        "n_rows_total": len(rows),
        "n_rows_scored": len(common),
        "latency_note": "text and audio backends do different work; latency is not like-for-like",
        "models": {},
        "pairwise_agreement": {},
    }
    for b in backends:
        preds = [results[b.name][r["id"]][0] for r in common]
        lats = [results[b.name][r["id"]][2] for r in common]
        acc, mf1 = score(gold, preds)
        summary["models"][b.name] = {
            "accuracy": round(acc, 3),
            "macro_f1": round(mf1, 3),
            "latency_mean_s": round(statistics.mean(lats), 4),
            "latency_p90_s": round(p90(lats), 4),
        }
    names = [b.name for b in backends]
    for i, a in enumerate(names):
        for c in names[i + 1:]:
            same = sum(results[a][r["id"]][0] == results[c][r["id"]][0] for r in common)
            summary["pairwise_agreement"][f"{a}~{c}"] = round(same / len(common), 3)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "per_row.csv", "w", encoding="utf-8", newline="") as f:
        cols = ["id", "gold"] + [f"{n}_{k}" for n in names for k in ("pred", "probs_json", "sec")]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in common:
            row = {"id": r["id"], "gold": r["gold"]}
            for n in names:
                lab, probs, dt = results[n][r["id"]]
                row[f"{n}_pred"] = lab
                row[f"{n}_probs_json"] = json.dumps({k: round(v, 4) for k, v in probs.items()})
                row[f"{n}_sec"] = round(dt, 4)
            w.writerow(row)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
