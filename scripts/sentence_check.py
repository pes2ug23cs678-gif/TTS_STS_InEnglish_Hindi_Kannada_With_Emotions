#!/usr/bin/env python3
"""Times translation alone on the approved test sentences, with no pipeline and no emotion model.

Run from the project folder (where models.py and rules.json are):
  python sentence_check.py              # 3 runs per sentence and language
  python sentence_check.py --runs 5

Compare the output with the NMT + emotion column in pipeline_report.csv:
  - If the times match, the sentences are the cause.
  - If these times are much lower, the pipeline is adding the cost.
"""
import argparse
import csv
import pathlib
import statistics
import sys
import time

BASE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

LANG_MAP = {"hi": "hin_Deva", "kn": "kan_Knda"}


def load_sentences(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [row["english"] for row in csv.DictReader(f)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default=str(BASE / "translation_cases.csv"))
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    from models import load_neural_pipeline, translate_text
    import torch

    _engines = load_neural_pipeline()
    trans_tok, trans_mod = _engines["en_indic"]
    sentences = load_sentences(args.cases)

    def translate(text, lang):
        return translate_text(text, "eng_Latn", LANG_MAP[lang], trans_tok, trans_mod, [], {})

    translate(sentences[0], "hi")  # warm-up, not timed

    print(f"torch threads: {torch.get_num_threads()}  |  sentences: {len(sentences)}  |  runs: {args.runs}\n")
    print(f"{'lang':<6}{'chars':>7}{'median (s)':>12}{'min (s)':>10}{'max (s)':>10}  sentence")
    for lang in LANG_MAP:
        for text in sentences:
            times = []
            for _ in range(args.runs):
                t0 = time.perf_counter()
                translate(text, lang)
                times.append(time.perf_counter() - t0)
            print(f"{lang:<6}{len(text):>7}{statistics.median(times):>12.2f}"
                  f"{min(times):>10.2f}{max(times):>10.2f}  {text[:50]}")


if __name__ == "__main__":
    sys.exit(main())
