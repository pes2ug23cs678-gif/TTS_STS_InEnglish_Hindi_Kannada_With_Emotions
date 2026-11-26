#!/usr/bin/env python3
"""Beam check: compares translation models and beam sizes on the same sentences.
Reports the median time per sentence and how many outputs exactly match the approved wording.
Caveat: the approved wording was produced by the 1B model at beam 4, so that setting has an
advantage on exact match. Exact match is not a quality measure on its own; read the outputs too.
Run from the project folder, with ab_translate_cpu.py and translation_cases.csv alongside:
  python beam_check.py                                  # 1B model, beams 3 and 4 (default)
  python beam_check.py --model 200m --beams 4,6         # 200M model, beams 4 and 6
  python beam_check.py --limit 10                       # quicker check
Writes beam_check_<model>_report.csv and beam_check_<model>_summary.txt.
"""
import argparse
import csv
import pathlib
import re
import statistics
import sys
import time

BASE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

MODELS = {
    "1b": "ai4bharat/indictrans2-en-indic-1B",
    "200m": "ai4bharat/indictrans2-en-indic-dist-200M",
}
LANGS = {"hi": ("hin_Deva", "hi_expected"), "kn": ("kan_Knda", "kn_expected")}

def norm(text):
    return re.sub(r"\s+", " ", text or "").strip()

def canon(text):
    return norm(text).replace("ँ", "ं")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODELS), default="1b")
    ap.add_argument("--beams", default="3,4", help="comma list of beam sizes, e.g. 4,6")
    ap.add_argument("--cases", default=str(BASE / "translation_cases.csv"))
    ap.add_argument("--limit", type=int, default=0, help="only the first N sentences (0 = all)")
    args = ap.parse_args()
    beams_list = [int(b) for b in args.beams.split(",")]
    from ab_translate import load_model, translate
    from IndicTransToolkit.processor import IndicProcessor

    proc = IndicProcessor(inference=True)
    with open(args.cases, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]

    tok, mod = load_model(MODELS[args.model], "cpu", True)
    translate(tok, mod, proc, "Warm up.", "hin_Deva", beams_list[0], "cpu")  # first call is slower; not timed

    report = []
    for row in rows:
        for lang, (code, col) in LANGS.items():
            expected = norm(row.get(col, ""))
            if not expected:
                continue
            for beams in beams_list:
                t0 = time.perf_counter()
                out = norm(translate(tok, mod, proc, row["english"], code, beams, "cpu"))
                secs = time.perf_counter() - t0
                report.append({
                    "english": row["english"], "lang": lang, "beams": beams,
                    "seconds": round(secs, 2), "exact": canon(out) == canon(expected),
                    "output": out, "expected": expected,
                })

    report_path = BASE / f"beam_check_{args.model}_report.csv"
    with open(report_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        writer.writeheader()
        writer.writerows(report)

    lines = [f"model {args.model} ({MODELS[args.model]})"]
    for beams in beams_list:
        items = [r for r in report if r["beams"] == beams]
        times = [r["seconds"] for r in items]
        exact = sum(r["exact"] for r in items)
        lines.append(f"beam {beams}: median {statistics.median(times):.2f}s per sentence, "
                     f"mean {statistics.mean(times):.2f}s, exact match {exact}/{len(items)}")
    summary = "\n".join(lines)
    (BASE / f"beam_check_{args.model}_summary.txt").write_text(summary + "\n", encoding="utf-8")
    print("\n" + summary)
    print(f"\nWritten: {report_path.name}, beam_check_{args.model}_summary.txt")
    return 0

if __name__ == "__main__":
    sys.exit(main())
