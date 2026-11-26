#!/usr/bin/env python3
"""Checks whether running translation in a worker thread is slower than running it on the main thread.

Run from the project folder (where models.py and rules.json are):
  python thread_check.py              # 10 runs
  python thread_check.py --runs 5     # quicker check

Each run covers the same sentences three ways, rotating the order each run so that
warm-up or heat does not favour one method:
  MAIN         translation called directly on the main thread
  THREAD       translation called through asyncio.to_thread (as the pipeline does)
  THREAD_FIX   same as THREAD, but torch.set_num_threads is set again inside the worker first

If THREAD is much slower than MAIN, the thread hop is the cause.
If THREAD_FIX is close to MAIN, setting the thread count inside the worker fixes it.
"""
import argparse
import asyncio
import pathlib
import statistics
import sys
import time

BASE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

SENTENCES = [
    "Good morning. My name is Likith, and I am your tour guide",
    "The river flows through the forest and joins the sea.",
    "Mythology describes nymphs as heavenly dancers in the court of the gods.",
    "Please give me a cup of coffee along with biscuits.",
]
METHODS = ["MAIN", "THREAD", "THREAD_FIX"]


async def main_async(runs):
    import torch
    from models import load_neural_pipeline, translate_text

    _engines = load_neural_pipeline()
    trans_tok, trans_mod = _engines["en_indic"]
    thread_count = torch.get_num_threads()

    def translate(text):
        return translate_text(text, "eng_Latn", "hin_Deva", trans_tok, trans_mod, [], {})

    def translate_fixed(text):
        torch.set_num_threads(thread_count)
        return translate(text)

    translate(SENTENCES[0])  # warm-up, not timed

    print(f"torch threads: {thread_count}  |  sentences: {len(SENTENCES)}  |  runs: {runs}\n")

    times = {m: [] for m in METHODS}
    for run in range(runs):
        shift = run % len(METHODS)
        order = METHODS[shift:] + METHODS[:shift]
        for text in SENTENCES:
            for method in order:
                t0 = time.perf_counter()
                if method == "MAIN":
                    translate(text)
                elif method == "THREAD":
                    await asyncio.to_thread(translate, text)
                else:
                    await asyncio.to_thread(translate_fixed, text)
                times[method].append(time.perf_counter() - t0)

    print(f"{'method':<12}{'median (s)':>12}{'p90 (s)':>12}{'max (s)':>12}")
    for name in METHODS:
        values = times[name]
        p90 = statistics.quantiles(values, n=10)[-1]
        print(f"{name:<12}{statistics.median(values):>12.2f}{p90:>12.2f}{max(values):>12.2f}")

    m = statistics.median(times["MAIN"])
    t = statistics.median(times["THREAD"])
    f = statistics.median(times["THREAD_FIX"])
    print(f"\nTHREAD / MAIN median ratio:      {t / m:.2f}")
    print(f"THREAD_FIX / MAIN median ratio:  {f / m:.2f}")
    if t > m * 1.3:
        print("THREAD is much slower: the thread hop is part of the slowdown.")
    else:
        print("THREAD is not much slower: the thread hop is not the cause.")
    if f <= m * 1.1:
        print("THREAD_FIX is close to MAIN: setting the thread count inside the worker fixes it.")
    else:
        print("THREAD_FIX is still slower than MAIN: the thread count is not the fix.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=10)
    args = ap.parse_args()
    asyncio.run(main_async(args.runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
