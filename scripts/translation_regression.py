#!/usr/bin/env python3
"""Automated translation regression test for the STS app.

Each English case goes through the same steps as the app: name masking, IndicTrans2,
name restoration, and the surface rules. The result is compared with the approved
Hindi and Kannada text in translation_cases.csv.

Usage (run from the project folder, where models.py and rules.json are):
  python translation_regression.py               # run every case; exit 1 if any FAIL
  python translation_regression.py --langs hi    # Hindi only
  python translation_regression.py --update      # write current outputs into the cases file

Statuses:
  PASS   output matches the approved text
  FAIL   output differs from the approved text (no note on the case)
  KNOWN  output differs, but the case has a note (a known issue kept on purpose)
  NEW    no approved text yet (filled in by --update)

Files:
  translation_cases.csv  english, hi_expected, kn_expected, note
  translation_report.csv one row per sentence and language, with timing
"""
import argparse
import csv
import pathlib
import re
import statistics
import sys
import time

BASE = pathlib.Path(__file__).resolve().parent
LANG_MAP = {"en": "eng_Latn", "kn": "kan_Knda", "hi": "hin_Deva"}
FIELDS = ["english", "hi_expected", "kn_expected", "note"]


def norm(text):
    return re.sub(r"\s+", " ", text or "").strip()


def canon(text):
    """For comparison only: treats Hindi ँ (chandrabindu) and ं (anusvara) as the same."""
    return norm(text).replace("ँ", "ं")


def build_runner():
    """Loads the models once and returns run(english, lang) -> final text, as the app does."""
    from models import load_neural_pipeline, translate_text
    from entity_masker import mask_named_entities, unmask_and_restore_entities
    from rules_engine import RulesEngine

    _engines = load_neural_pipeline()
    trans_tok, trans_mod = _engines["en_indic"]
    engine = RulesEngine(str(BASE / "rules.json"))

    def run(english, lang):
        masked, ledger = mask_named_entities(english)
        raw = translate_text(masked, LANG_MAP["en"], LANG_MAP[lang],
                             trans_tok, trans_mod, [], {})
        restored = unmask_and_restore_entities(raw, ledger, lang)
        return engine.apply_conversational_surface_rules(restored, lang)

    return run


def load_cases(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save_cases(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDS})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default=str(BASE / "translation_cases.csv"))
    ap.add_argument("--report", default=str(BASE / "translation_report.csv"))
    ap.add_argument("--langs", default="hi,kn", help="comma list, e.g. hi,kn")
    ap.add_argument("--update", action="store_true",
                    help="write current outputs into the cases file (review the report first)")
    args = ap.parse_args()

    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    rows = load_cases(args.cases)
    run = build_runner()

    report, times = [], []
    counts = {"PASS": 0, "FAIL": 0, "KNOWN": 0, "NEW": 0}
    for row in rows:
        english = row["english"]
        for lang in langs:
            col = f"{lang}_expected"
            t0 = time.perf_counter()
            actual = norm(run(english, lang))
            seconds = time.perf_counter() - t0
            times.append(seconds)

            expected = norm(row.get(col, ""))
            if not expected:
                status = "NEW"
            elif canon(actual) == canon(expected):
                status = "PASS"
            elif (row.get("note") or "").strip():
                status = "KNOWN"
            else:
                status = "FAIL"
            counts[status] += 1

            report.append({"english": english, "lang": lang, "status": status,
                           "expected": expected, "actual": actual,
                           "seconds": round(seconds, 2), "note": row.get("note", "")})

            if args.update and status in ("FAIL", "NEW"):
                row[col] = actual

    with open(args.report, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(report[0].keys()) if report else ["english"])
        writer.writeheader()
        writer.writerows(report)

    if args.update:
        save_cases(args.cases, rows)

    print(f"\nPASS {counts['PASS']}  FAIL {counts['FAIL']}  KNOWN {counts['KNOWN']}  NEW {counts['NEW']}")
    if times:
        p90 = statistics.quantiles(times, n=10)[-1] if len(times) >= 10 else max(times)
        print(f"Time per sentence: median {statistics.median(times):.2f}s, p90 {p90:.2f}s, max {max(times):.2f}s")
    for item in report:
        if item["status"] == "FAIL":
            print(f"\nFAIL [{item['lang']}] {item['english']}\n  expected: {item['expected']}\n  actual:   {item['actual']}")
    print(f"\nReport written to {args.report}")

    if args.update:
        print(f"Cases file updated: {args.cases}")
        return 0
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
