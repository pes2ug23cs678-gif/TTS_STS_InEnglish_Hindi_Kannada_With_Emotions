#!/usr/bin/env python3
"""
Compare TTS engines on the same sentences: Edge-TTS (incumbent baseline) and
Indic-Parler-TTS (local model, slow on CPU).

Run from combined_app with the project virtualenv active:
    python tts_compare.py                                # edge (with emotion prosody) vs edge_plain (no prosody)
    python tts_compare.py --engines edge                 # Edge only
    python tts_compare.py --runs 3                       # more timing runs

Outputs (tts_compare_out/):
    results.csv              latency, audio length and real-time factor per engine, row and run
    audio/<engine>/          every generated clip (for checking pronunciation)
    blind/                   clips for listening, loudness matched, names hidden (A, B, C...)
    private/blind_key.csv    which letter is which engine. Do NOT open until ratings are done.
"""
import argparse
import asyncio
import csv
import io
import random
import time
from pathlib import Path

from pydub import AudioSegment

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "tts_compare_out"

try:
    from tts.loudness import normalize_loudness
except ImportError:  # loudness.py not copied into this project yet
    normalize_loudness = None


def build_engine(name):
    """Returns (engine, audio format) for a name. Imports happen here so unused engines are skipped."""
    if name in ("edge", "edge_plain"):
        from tts.edge_engine import EdgeTTSEngine
        return EdgeTTSEngine(), "mp3"
    if name == "parler":
        from tts.parler_engine import ParlerTTSEngine
        return ParlerTTSEngine(), "wav"
    if name in ("mms", "indicf5", "xtts"):
        from tts import alt_engines
        cls = {"mms": alt_engines.MMSTTSEngine,
               "indicf5": alt_engines.IndicF5Engine,
               "xtts": alt_engines.XTTSEngine}[name]
        return cls(), "wav"
    raise SystemExit(f"Unknown engine '{name}'. Use edge, edge_plain, parler, mms, indicf5 or xtts.")


def emo_meta(row):
    return {
        "detected_emo": row["emotion"],
        "label": row["emotion"],
        "avd": {
            "arousal": float(row["arousal"]),
            "valence": float(row["valence"]),
            "dominance": 0.0,
        },
    }


def audio_seconds(audio_bytes, fmt):
    return len(AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)) / 1000.0


def to_mp3_bytes(audio_bytes, fmt):
    seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
    buf = io.BytesIO()
    seg.export(buf, format="mp3", bitrate="128k")
    return buf.getvalue()


async def time_engine(name, engine, fmt, rows, runs, audio_dir):
    results, clips = [], {}
    for i, row in enumerate(rows, 1):
        for r in range(1, runs + 1):
            t0 = time.perf_counter()
            try:
                # edge_plain is the ablation baseline: same voice, no emotion prosody
                meta = None if name == "edge_plain" else emo_meta(row)
                audio = await engine.synthesize(
                    row["text"], row["lang"], emotion_meta=meta)
                latency = time.perf_counter() - t0
                secs = audio_seconds(audio, fmt)
                status = "ok"
                clips[(i, r)] = (audio, fmt)
                (audio_dir / name).mkdir(parents=True, exist_ok=True)
                (audio_dir / name / f"row{i:02d}_run{r}.{fmt}").write_bytes(audio)
            except Exception as exc:  # record failures instead of stopping the run
                latency = time.perf_counter() - t0
                secs = 0.0
                status = f"error: {type(exc).__name__}: {exc}"[:200]
            rtf = (latency / secs) if secs else float("nan")
            results.append({
                "engine": name, "row": i, "lang": row["lang"], "emotion": row["emotion"],
                "run": r, "latency_s": round(latency, 2), "audio_s": round(secs, 2),
                "realtime_factor": round(rtf, 2) if secs else "", "status": status,
            })
            print(f"[{name}] row {i:02d} run {r}: {latency:6.2f}s  audio {secs:5.2f}s  {status}", flush=True)
    return results, clips


def write_blind(rows, clips_by_engine, blind_dir, key_path):
    """One blind set per sentence, using only the engines that produced audio for that sentence.
    An engine that cannot speak a language (for example XTTS and Kannada) is left out of that sentence
    rather than removing the whole sentence."""
    blind_dir.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    engines = list(clips_by_engine)
    rng = random.Random(2026)
    key_rows, max_letters = [], 0
    for i, row in enumerate(rows, 1):
        available = [e for e in engines if (i, 1) in clips_by_engine[e]]
        if len(available) < 2:
            continue  # nothing to compare for this sentence
        rng.shuffle(available)
        letters = [chr(ord("A") + k) for k in range(len(available))]
        max_letters = max(max_letters, len(letters))
        entry = {"row": i, "text": row["text"]}
        for letter, engine in zip(letters, available):
            audio, fmt = clips_by_engine[engine][(i, 1)]
            data = to_mp3_bytes(audio, fmt)
            if normalize_loudness is not None:
                data = normalize_loudness(data, target_lufs=-12.0)
            (blind_dir / f"row{i:02d}_{letter}.mp3").write_bytes(data)
            entry[letter] = engine
        key_rows.append(entry)
    letters_all = [chr(ord("A") + k) for k in range(max_letters)]
    fieldnames = ["row"] + letters_all + ["text"]
    with key_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(key_rows)


def write_results(results):
    OUT.mkdir(exist_ok=True)
    with (OUT / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def summarise(results):
    print("\n=== Summary ===")
    for name in sorted({r["engine"] for r in results}):
        ok = [r for r in results if r["engine"] == name and r["status"] == "ok"]
        failed = len([r for r in results if r["engine"] == name and r["status"] != "ok"])
        if not ok:
            print(f"{name}: no successful clips ({failed} failed)")
            continue
        lat = sorted(r["latency_s"] for r in ok)
        rtf = sorted(float(r["realtime_factor"]) for r in ok)
        median = lambda xs: xs[len(xs) // 2]
        print(f"{name}: {len(ok)} ok, {failed} failed | latency median {median(lat):.2f}s, "
              f"max {lat[-1]:.2f}s | real-time factor median {median(rtf):.2f}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default=str(ROOT / "tts_compare_inputs.csv"))
    # Every engine that does not need a paid API. Sarvam, ElevenLabs and Bhashini are excluded.
    ap.add_argument("--engines", default="edge,edge_plain,parler,mms,indicf5,xtts")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--skip-blind", action="store_true")
    args = ap.parse_args()

    names = [n.strip() for n in args.engines.split(",") if n.strip()]
    with open(args.inputs, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("text", "").strip()]
    print(f"{len(rows)} sentences, {args.runs} run(s) each, engines: {', '.join(names)}")

    all_results, clips_by_engine = [], {}
    for name in names:
        try:
            engine, fmt = build_engine(name)
            if name == "parler":
                t0 = time.perf_counter()
                engine._lazy_load()  # load once, timed separately from synthesis
                print(f"[parler] model load: {time.perf_counter() - t0:.1f}s (not included in clip latency)")
            res, clips = await time_engine(name, engine, fmt, rows, args.runs, OUT / "audio")
        except Exception as exc:  # one broken engine must not discard the others' results
            msg = f"error: {type(exc).__name__}: {exc}"[:200]
            print(f"[{name}] could not run: {msg}", flush=True)
            res = [{"engine": name, "row": i, "lang": r["lang"], "emotion": r["emotion"],
                    "run": 1, "latency_s": 0.0, "audio_s": 0.0, "realtime_factor": "",
                    "status": msg} for i, r in enumerate(rows, 1)]
            clips = {}
        all_results += res
        clips_by_engine[name] = clips
        write_results(all_results)  # saved after every engine, so a later crash keeps earlier results

    summarise(all_results)
    if not args.skip_blind and len(names) >= 2:
        write_blind(rows, clips_by_engine, OUT / "blind", OUT / "private" / "blind_key.csv")
        print(f"\nBlind clips written to {OUT / 'blind'} (key in private/blind_key.csv, do not open until rated).")


if __name__ == "__main__":
    asyncio.run(main())
