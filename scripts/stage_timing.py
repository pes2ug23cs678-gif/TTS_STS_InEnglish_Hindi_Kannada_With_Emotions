#!/usr/bin/env python3
"""Times the translation and emotion stages separately, then together, to find the slowdown.

Run from the project folder (where models.py, rules.json, and config.py are):
  python stage_timing.py              # 10 runs per case
  python stage_timing.py --runs 5     # fewer runs for a quick check

It runs three ways on the same sentences:
  A. translation alone
  B. emotion alone
  C. translation and emotion run together, as the pipeline does (asyncio.gather)
  D. translation, then emotion, one after the other

If C is much slower than A + B, the two models are competing for the CPU.
"""
import argparse
import asyncio
import pathlib
import statistics
import sys
import time

BASE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

LANG_MAP = {"en": "eng_Latn", "hi": "hin_Deva"}
SENTENCES = [
    "Good morning. My name is Likith, and I am your tour guide",
    "The river flows through the forest and joins the sea.",
    "Mythology describes nymphs as heavenly dancers in the court of the gods.",
    "Please give me a cup of coffee along with biscuits.",
]

def med(values):
    return statistics.median(values) if values else float("nan")

async def timed(fn, *args):
    t0 = time.perf_counter()
    result = await asyncio.to_thread(fn, *args)
    return result, time.perf_counter() - t0

async def main_async(runs):
    import torch
    from models import load_neural_pipeline, translate_text, detect_text_emotion

    _engines = load_neural_pipeline()
    trans_tok, trans_mod = _engines["en_indic"]
    emo_tok, emo_mod = _engines["emotion"]

    def translate(text):
        return translate_text(text, LANG_MAP["en"], LANG_MAP["hi"], trans_tok, trans_mod, [], {})

    def emotion(text):
        return detect_text_emotion(text, emo_tok, emo_mod)

    # Warm up both models so the first run does not skew the numbers
    translate(SENTENCES[0])
    emotion(SENTENCES[0])

    print(f"torch threads: {torch.get_num_threads()}  |  sentences: {len(SENTENCES)}  |  runs each: {runs}\n")

    results = {"A translation alone": [], "B emotion alone": [],
               "C together (pipeline style)": [], "D one after the other": []}

    for _ in range(runs):
        for text in SENTENCES:
            _, a = await timed(translate, text)
            results["A translation alone"].append(a)

            _, b = await timed(emotion, text)
            results["B emotion alone"].append(b)

            t0 = time.perf_counter()
            await asyncio.gather(asyncio.to_thread(translate, text), asyncio.to_thread(emotion, text))
            results["C together (pipeline style)"].append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            translate(text)
            emotion(text)
            results["D one after the other"].append(time.perf_counter() - t0)

    print(f"{'case':<30}{'median (s)':>12}{'p90 (s)':>12}{'max (s)':>12}")
    for name, values in results.items():
        p90 = statistics.quantiles(values, n=10)[-1] if len(values) >= 10 else max(values)
        print(f"{name:<30}{med(values):>12.2f}{p90:>12.2f}{max(values):>12.2f}")

    a, b = med(results["A translation alone"]), med(results["B emotion alone"])
    c, d = med(results["C together (pipeline style)"]), med(results["D one after the other"])
    print(f"\nA + B (sum of alone runs): {a + b:.2f}s")
    print(f"C (together):              {c:.2f}s")
    print(f"D (one after the other):   {d:.2f}s")
    if c > d * 1.3:
        print("\nTogether is much slower than one after the other: the two models are competing for the CPU.")
    else:
        print("\nTogether is not much slower than one after the other: contention is not the main cause.")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=10)
    args = ap.parse_args()
    asyncio.run(main_async(args.runs))
    return 0

if __name__ == "__main__":
    sys.exit(main())
