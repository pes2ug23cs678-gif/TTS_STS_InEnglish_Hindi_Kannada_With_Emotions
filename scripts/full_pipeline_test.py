#!/usr/bin/env python3
"""Full pipeline test: runs each case through STSPipeline.process, the same entry point the app uses.

Each run covers name masking, translation, emotion detection, the sensitivity guard, the
surface rules, speech synthesis, and timing. The extra checks cover the Edge fallback and the
guard, using one sentence each.

Run from the project folder (where pipeline.py, models.py, rules.json, and config.py are):
  python full_pipeline_test.py                       # Edge TTS for every case (free)
  python full_pipeline_test.py --engine elevenlabs   # uses ElevenLabs credits
  python full_pipeline_test.py --cases 5            # quick check on the first 5 sentences
  python full_pipeline_test.py --save-audio audio/  # keep the MP3 files to listen to

Checks per sentence and language:
  TEXT     output matches the approved text in translation_cases.csv (ElevenLabs trims to 300 characters)
  AUDIO    speech bytes were returned
  GUARD    sensitivity guard applied exactly when the emotion label is in NO_TAG_LABELS
  BUDGET   end-to-end time is within the 5.2-second target (reported as SLOW, not a failure)

Fallback and guard checks (one sentence each):
  FALLBACK the primary engine raises, and the pipeline retries and still returns audio
  GUARD    a 'disgust' emotion sets suppress_tags in the telemetry

Exit code is 1 if any TEXT, AUDIO, GUARD, or FALLBACK check fails.
"""
import argparse
import asyncio
import csv
import pathlib
import re
import statistics
import sys
import time

BASE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

LANG_MAP = {"en": "eng_Latn", "kn": "kan_Knda", "hi": "hin_Deva"}
BUDGET_SEC = 5.2
ELEVENLABS_TRIM = 300
FIELDS = ["english", "lang", "status", "expected", "actual", "engine", "audio_bytes",
          "e2e_sec", "masking_sec", "nmt_emotion_sec", "rules_sec", "tts_sec", "guard", "note"]


def norm(text):
    return re.sub(r"\s+", " ", text or "").strip()


def canon(text):
    """Comparison only: treats Hindi chandrabindu and anusvara as the same."""
    return norm(text).replace("ँ", "ं")


def load_cases(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def build_pipeline(engine, skip_emotion=False):
    """Loads the models and builds STSPipeline the same way app.get_pipeline() does."""
    from models import load_neural_pipeline, translate_text, detect_text_emotion
    from pipeline import STSPipeline

    import torch

    _engines = load_neural_pipeline()
    trans_tok, trans_mod = _engines["en_indic"]
    emo_tok, emo_mod = _engines["emotion"]
    print(f"[thread check] torch threads at build: {torch.get_num_threads()}")

    def translator_fn(text, src_lang, tgt_lang):
        torch.set_num_threads(4)  # test only: match the threads set at build time
        print(f"[thread check] torch threads inside translator: {torch.get_num_threads()}")
        return translate_text(text, LANG_MAP[src_lang], LANG_MAP[tgt_lang],
                              trans_tok, trans_mod, [], {})

    def emotion_fn(text):
        if skip_emotion:  # test only: neutral label, no emotion model run
            return {"label": "neutral", "detected_emo": "neutral", "conf": 1.0, "prob_dist": {"neutral": 1.0}}
        label, conf, prob_dist = detect_text_emotion(text, emo_tok, emo_mod)
        return {"label": label, "detected_emo": label, "conf": conf, "prob_dist": prob_dist}

    pipe = STSPipeline(translator_fn=translator_fn, emotion_fn=emotion_fn,
                       rules_path=str(BASE / "rules.json"), default_tts_engine=engine)
    return pipe, emotion_fn


def guard_expected(emotion_meta):
    from config import NO_TAG_LABELS
    label = str(emotion_meta.get("detected_emo", emotion_meta.get("label", ""))).lower()
    return label in NO_TAG_LABELS


async def run_cases(pipe, rows, langs, engine, save_dir, gender):
    report = []
    for row in rows:
        for lang in langs:
            expected = norm(row.get(f"{lang}_expected", ""))
            item = {"english": row["english"], "lang": lang, "expected": expected,
                    "note": row.get("note", "")}
            try:
                res = await pipe.process(row["english"], gender, "en", lang,
                                         preferred_tts_engine=engine)
            except Exception as exc:  # noqa: BLE001 - report any pipeline error, keep going
                item.update(status="FAIL", actual="", engine="", audio_bytes=0, e2e_sec="",
                            masking_sec="", nmt_emotion_sec="", rules_sec="", tts_sec="",
                            guard="", note=f"{item['note']} | error: {type(exc).__name__}: {exc}".strip(" |"))
                report.append(item)
                continue

            tel = res.telemetry
            tts = tel.get("tts", {})
            actual = norm(res.translated_text)
            problems = []

            if not expected:
                status = "NEW"
            else:
                want = expected[:ELEVENLABS_TRIM] if engine == "elevenlabs" else expected
                if canon(actual) != canon(want):
                    problems.append("text differs")
                status = "FAIL" if problems else "PASS"

            if len(res.audio_bytes or b"") < 1000:
                problems.append("no audio")
                status = "FAIL"

            guard_ok = bool(tel.get("sensitivity_guard_applied")) == guard_expected(res.emotion_meta)
            if not guard_ok:
                problems.append("guard mismatch")
                status = "FAIL"

            if tel.get("total_e2e_latency_sec", 0) > BUDGET_SEC and status == "PASS":
                status = "SLOW"

            if save_dir and res.audio_bytes:
                ext = ".mp3" if tts.get("audio_format") == "audio/mpeg" else ".wav"
                safe = re.sub(r"[^A-Za-z0-9]+", "_", row["english"])[:40]
                (save_dir / f"{safe}_{lang}{ext}").write_bytes(res.audio_bytes)

            item.update(status=status, actual=actual, engine=tts.get("engine_used", ""),
                        audio_bytes=len(res.audio_bytes or b""),
                        e2e_sec=tel.get("total_e2e_latency_sec", ""),
                        masking_sec=tel.get("entity_masking_sec", ""),
                        nmt_emotion_sec=tel.get("nmt_and_emotion_sec", ""),
                        rules_sec=tel.get("rules_post_processing_sec", ""),
                        tts_sec=tts.get("latency_sec", ""),
                        guard=tel.get("sensitivity_guard_applied", False),
                        note=" | ".join(filter(None, [item["note"], "; ".join(problems)])))
            report.append(item)
    return report


async def run_fallback_check(pipe, gender, engine):
    """Makes the first TTS call raise, so the pipeline must retry through Edge."""
    dispatcher = pipe.tts_dispatcher
    original = dispatcher.synthesize
    calls = {"n": 0}

    async def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated primary failure")
        return await original(*args, **kwargs)

    dispatcher.synthesize = flaky
    try:
        res = await pipe.process("Good morning. My name is Likith.", gender, "en", "hi",
                                 preferred_tts_engine=engine)
    finally:
        dispatcher.synthesize = original
    ok = calls["n"] == 2 and len(res.audio_bytes or b"") >= 1000
    return ok, f"TTS calls: {calls['n']}, audio bytes: {len(res.audio_bytes or b'')}"


async def run_guard_check(pipe_factory, gender, engine):
    """Forces a 'disgust' emotion and checks that the guard suppresses tags."""
    pipe = pipe_factory(lambda text: {"label": "disgust", "detected_emo": "disgust",
                                      "conf": 0.9, "prob_dist": {"disgust": 0.9}})
    res = await pipe.process("It is disgusting how the street has been left unclean.",
                             gender, "en", "hi", preferred_tts_engine=engine)
    ok = res.telemetry.get("sensitivity_guard_applied") is True and \
        res.emotion_meta.get("suppress_tags") is True
    return ok, f"guard applied: {res.telemetry.get('sensitivity_guard_applied')}"


def summarize(report):
    counts = {}
    for item in report:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    print("\nStatus counts: " + "  ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    e2e = [float(i["e2e_sec"]) for i in report if i["e2e_sec"] not in ("", None)]
    if e2e:
        p90 = statistics.quantiles(e2e, n=10)[-1] if len(e2e) >= 10 else max(e2e)
        print(f"End-to-end per sentence: median {statistics.median(e2e):.2f}s, "
              f"p90 {p90:.2f}s, max {max(e2e):.2f}s (target {BUDGET_SEC}s)")
        over = sum(1 for x in e2e if x > BUDGET_SEC)
        print(f"Over budget: {over} of {len(e2e)}")
    for stage, key in [("masking", "masking_sec"), ("NMT + emotion", "nmt_emotion_sec"),
                       ("rules", "rules_sec"), ("speech", "tts_sec")]:
        vals = [float(i[key]) for i in report if i[key] not in ("", None)]
        if vals:
            print(f"  {stage:<14} median {statistics.median(vals):.3f}s")
    for item in report:
        if item["status"] in ("FAIL", "SLOW"):
            print(f"\n{item['status']} [{item['lang']}] {item['english']}\n"
                  f"  note: {item['note']}\n  expected: {item['expected']}\n  actual:   {item['actual']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default=str(BASE / "translation_cases.csv"))
    ap.add_argument("--report", default=str(BASE / "pipeline_report.csv"))
    ap.add_argument("--langs", default="hi,kn")
    ap.add_argument("--engine", default="edge", choices=["edge", "elevenlabs"])
    ap.add_argument("--limit", type=int, default=0, help="only the first N sentences (0 = all)")
    ap.add_argument("--save-audio", default="", help="folder to save the audio files")
    ap.add_argument("--gender", default="female", choices=["female", "male"])
    ap.add_argument("--skip-emotion", action="store_true", help="test only: skip the emotion model")
    args = ap.parse_args()

    rows = load_cases(args.cases)
    if args.limit:
        rows = rows[: args.limit]
    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    save_dir = None
    if args.save_audio:
        save_dir = pathlib.Path(args.save_audio)
        save_dir.mkdir(parents=True, exist_ok=True)

    pipe, emotion_fn = build_pipeline(args.engine, args.skip_emotion)

    async def main_async():
        report = await run_cases(pipe, rows, langs, args.engine, save_dir, args.gender)
        fb_ok, fb_note = await run_fallback_check(pipe, args.gender, args.engine)
        guard_ok, guard_note = await run_guard_check(
            lambda fn: _pipeline_with_emotion(pipe, fn, args.engine), args.gender, args.engine)
        return report, (fb_ok, fb_note), (guard_ok, guard_note)

    report, (fb_ok, fb_note), (guard_ok, guard_note) = asyncio.run(main_async())

    with open(args.report, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report)

    summarize(report)
    print(f"\nFALLBACK check: {'PASS' if fb_ok else 'FAIL'} ({fb_note})")
    print(f"GUARD check:    {'PASS' if guard_ok else 'FAIL'} ({guard_note})")
    print(f"\nReport written to {args.report}")

    failed = any(i["status"] == "FAIL" for i in report) or not fb_ok or not guard_ok
    return 1 if failed else 0


def _pipeline_with_emotion(base_pipe, emotion_fn, engine):
    """Same pipeline, with the emotion function replaced. Shares the loaded models."""
    from pipeline import STSPipeline
    pipe = STSPipeline(translator_fn=base_pipe.translator_fn, emotion_fn=emotion_fn,
                       rules_path=str(BASE / "rules.json"), default_tts_engine=engine)
    return pipe


if __name__ == "__main__":
    sys.exit(main())
