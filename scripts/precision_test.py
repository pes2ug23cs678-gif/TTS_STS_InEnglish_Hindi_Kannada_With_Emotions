#!/usr/bin/env python3
"""Precision test: scores four translation settings against the approved references.

Settings: 1B and 200M models, each at beam 1 and beam 4. The models are loaded and run with
the same functions as ab_translate_cpu.py, so the four settings are compared on equal terms.

Run from the project folder, with ab_translate_cpu.py and translation_cases.csv alongside:
  pip install sacrebleu
  python precision_test.py                 # all sentences, Hindi and Kannada
  python precision_test.py --limit 10      # quicker check

Outputs:
  precision_report.csv    one row per sentence, language and setting, with chrF and exact match
  precision_summary.txt   mean and median per setting, and wrong-script counts
  blind_rating_sheet.csv  outputs shuffled and unlabelled, for native speakers to rate 1-5
  blind_key.csv           the key that maps the sheet's options back to settings (keep private)

Caveat: the approved references were produced by the 1B model at beam 4, so that setting has
an advantage on chrF and exact match. The blind rating sheet is the fairer comparison.
"""
import argparse
import csv
import pathlib
import random
import re
import statistics
import sys

BASE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

SETTINGS = [
    ("1B-beam1", "ai4bharat/indictrans2-en-indic-1B", 1),
    ("1B-beam4", "ai4bharat/indictrans2-en-indic-1B", 4),
    ("200M-beam1", "ai4bharat/indictrans2-en-indic-dist-200M", 1),
    ("200M-beam4", "ai4bharat/indictrans2-en-indic-dist-200M", 4),
]
LANGS = {"hi": ("hin_Deva", "hi_expected", r"[ऀ-ॿ]"),
         "kn": ("kan_Knda", "kn_expected", r"[ಀ-೿]")}
FIELDS = ["english", "lang", "setting", "output", "expected", "chrf", "exact", "wrong_script"]


def norm(text):
    return re.sub(r"\s+", " ", text or "").strip()


def canon(text):
    """Comparison only: treats Hindi chandrabindu and anusvara as the same."""
    return norm(text).replace("ँ", "ं")


def script_share(text, pattern):
    letters = re.findall(r"\w", text)
    if not letters:
        return 0.0
    return len(re.findall(pattern, text)) / len(letters)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default=str(BASE / "translation_cases.csv"))
    ap.add_argument("--limit", type=int, default=0, help="only the first N sentences (0 = all)")
    args = ap.parse_args()

    try:
        from sacrebleu.metrics import CHRF
    except ImportError:
        print("sacrebleu is not installed. Run: pip install sacrebleu")
        return 1
    chrf = CHRF()

    from ab_translate import load_model, translate
    from IndicTransToolkit.processor import IndicProcessor

    proc = IndicProcessor(inference=True)  # required input format; the harness uses it too

    with open(args.cases, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]

    # Load each model once, then run both beam sizes on it.
    outputs = {}  # (english, lang, setting) -> output
    for model_name in ["1B", "200M"]:
        model_id = next(mid for name, mid, _ in SETTINGS if name.startswith(model_name))
        tok, mod = load_model(model_id, "cpu", True)
        for name, mid, beams in SETTINGS:
            if not name.startswith(model_name):
                continue
            for row in rows:
                for lang, (code, _, _) in LANGS.items():
                    out = translate(tok, mod, proc, row["english"], code, beams, "cpu")
                    outputs[(row["english"], lang, name)] = norm(out)
            print(f">>> {name} done", flush=True)

    report = []
    for row in rows:
        for lang, (_, col, pattern) in LANGS.items():
            expected = norm(row.get(col, ""))
            if not expected:
                continue
            for name, _, _ in SETTINGS:
                out = outputs[(row["english"], lang, name)]
                report.append({
                    "english": row["english"], "lang": lang, "setting": name,
                    "output": out, "expected": expected,
                    "chrf": round(chrf.sentence_score(out, [expected]).score, 2),
                    "exact": canon(out) == canon(expected),
                    "wrong_script": script_share(out, pattern) < 0.5,
                })

    with open(BASE / "precision_report.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(report)

    lines = []
    for name, _, _ in SETTINGS:
        items = [r for r in report if r["setting"] == name]
        scores = [r["chrf"] for r in items]
        exact = sum(r["exact"] for r in items)
        wrong = sum(r["wrong_script"] for r in items)
        lines.append(f"{name:<12} chrF mean {statistics.mean(scores):6.2f}  "
                     f"median {statistics.median(scores):6.2f}  "
                     f"exact {exact}/{len(items)}  wrong script {wrong}")
    summary = "\n".join(lines)
    (BASE / "precision_summary.txt").write_text(summary + "\n", encoding="utf-8")
    print("\n" + summary)

    # Blind rating sheet: options shuffled per item, names kept only in the key file.
    rng = random.Random(7)
    sheet, key = [], []
    item_id = 0
    for row in rows:
        for lang in LANGS:
            item_id += 1
            names = [name for name, _, _ in SETTINGS]
            rng.shuffle(names)
            entry = {"item": item_id, "english": row["english"], "lang": lang}
            for label, name in zip("ABCD", names):
                entry[f"option_{label}"] = outputs[(row["english"], lang, name)]
                entry[f"rating_{label}"] = ""
                key.append({"item": item_id, "option": label, "setting": name})
            sheet.append(entry)

    sheet_fields = ["item", "english", "lang"] + [f"{p}_{l}" for l in "ABCD" for p in ("option", "rating")]
    with open(BASE / "blind_rating_sheet.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sheet_fields)
        writer.writeheader()
        writer.writerows(sheet)
    with open(BASE / "blind_key.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item", "option", "setting"])
        writer.writeheader()
        writer.writerows(key)

    print(f"\nWritten: precision_report.csv, precision_summary.txt, blind_rating_sheet.csv, blind_key.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
