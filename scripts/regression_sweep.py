#!/usr/bin/env python
"""
Regression sweep across all six translation directions.

Runs a curated battery of inputs through the real TEXT path
    mask_named_entities -> translate_route -> unmask_and_restore_entities -> surface rules
(the same chain the app uses to produce `translated_text`, minus TTS) and AUTO-FLAGS
suspicious output so you don't have to read every line.

Run from the project folder (NewEnv - Copy - Copy):
    python regression_sweep.py
or, to be explicit about the interpreter:
    C:\\Users\\liki5\\Python312_Project\\clean_env\\Scripts\\python.exe regression_sweep.py

Loading all six directions' models is memory-heavy on an 8 GB machine. If it struggles,
comment out directions in DIRECTIONS below and run them in smaller groups.
"""
import re
import sys

try:
    from models import load_neural_pipeline, translate_route
    from entity_masker import mask_named_entities, unmask_and_restore_entities
    from rules_engine import apply_conversational_surface_rules
except Exception as exc:  # noqa: BLE001
    print("IMPORT FAILED:", type(exc).__name__, exc)
    print("Run this from inside the project folder with the venv active.")
    sys.exit(1)


def translate_text_path(text, src, tgt, engines):
    """Mirror the app's text pipeline (no TTS)."""
    masked, ledger = mask_named_entities(text)
    raw = translate_route(masked, src, tgt, engines)
    if isinstance(raw, str) and raw.startswith("[TRANSLATION_ERROR"):
        return raw
    unmasked = unmask_and_restore_entities(raw, ledger, tgt)
    try:
        return apply_conversational_surface_rules(unmasked, tgt)
    except Exception:  # noqa: BLE001
        return unmasked


# --- Red-flag detectors (run on the OUTPUT) ---------------------------------
RE_CONTRACTION_MANGLE = re.compile(r"\b[A-Za-z]+,\s+(?:m|s|re|ve|ll|d|t)\b")   # "I, m"
RE_ABBR_DANDA = re.compile(r"(?:डॉ|श्री|प्रो|कु|सुश्री|[A-Z])।")               # "डॉ।"
RE_WORD_LOOP = re.compile(r"(?<!\S)(\S+)(?:\s+\1(?!\S)){3,}")                   # 4+ repeats left


def flags(src, out, tgt_lang):
    f = []
    if not out or not out.strip():
        f.append("EMPTY_OUTPUT")
        return f
    if out.startswith("[TRANSLATION_ERROR"):
        f.append("TRANSLATION_ERROR")
        return f
    if RE_CONTRACTION_MANGLE.search(out):
        f.append("CONTRACTION_MANGLED(comma)")
    if tgt_lang == "hi" and RE_ABBR_DANDA.search(out):
        f.append("ABBR_DANDA")
    if RE_WORD_LOOP.search(out):
        f.append("WORD_LOOP_4plus")
    if tgt_lang in ("hi", "kn"):
        latin = len(re.findall(r"[A-Za-z]", out))
        if latin > 8:                         # names can be Latin; only flag heavy leakage
            f.append("LATIN_IN_%s(%d)" % (tgt_lang.upper(), latin))
    if len(out.split()) < max(2, len(src.split()) // 3):
        f.append("SUSPICIOUSLY_SHORT")
    return f


# --- Input battery (choose inputs IN the source language of each direction) --
BATTERY = {
    "en": [
        "I'm very happy today!",
        "Dr. Smith arrived at 4:30 a.m. and left by noon.",
        "The package, which I ordered last week, finally showed up.",
        "No, no, no! This won't work and it isn't fair.",
        "Michael Jackson sang at the concert in Bangalore.",
        "It costs 25 rupees and takes 3.5 hours.",
        "Wait... what?! I can't believe she didn't tell us.",
    ],
    "hi": [
        "मैं बहुत खुश हूँ।",
        "मुझे नहीं लगता कि यह काम करेगा।",
        "बहुत बहुत धन्यवाद, आपने बहुत मदद की।",
        "डॉ. शर्मा जल्दी जल्दी अस्पताल पहुँचे।",
        "कल मेरी परीक्षा है, इसलिए मैं रात भर पढ़ाई कर रहा हूँ।",
        "क्या?! यह कैसे हुआ... मुझे विश्वास नहीं हो रहा।",
        "यह 25 रुपये का है और 3 घंटे लगेंगे।",
    ],
    "kn": [
        "ನಾನು ತುಂಬಾ ಸಂತೋಷವಾಗಿದ್ದೇನೆ.",
        "ಇದು ಕೆಲಸ ಮಾಡುತ್ತದೆ ಎಂದು ನನಗೆ ಅನಿಸುವುದಿಲ್ಲ.",
        "ತುಂಬಾ ತುಂಬಾ ಧನ್ಯವಾದಗಳು, ನೀವು ತುಂಬಾ ಸಹಾಯ ಮಾಡಿದಿರಿ.",
        "ಡಾ. ಶರ್ಮಾ ಬೇಗ ಬೇಗ ಆಸ್ಪತ್ರೆಗೆ ಬಂದರು.",
        "ಇದು 25 ರೂಪಾಯಿ ಆಗುತ್ತದೆ ಮತ್ತು 3 ಗಂಟೆ ತೆಗೆದುಕೊಳ್ಳುತ್ತದೆ.",
        "ಏನು?! ಇದು ಹೇಗೆ ಆಯಿತು... ನನಗೆ ನಂಬಲಾಗುತ್ತಿಲ್ಲ.",
    ],
}

DIRECTIONS = [
    ("en", "hi"), ("en", "kn"),
    ("hi", "en"), ("hi", "kn"),
    ("kn", "en"), ("kn", "hi"),
]


def main():
    print("Loading translation models (this can take a minute, and a lot of RAM)...")
    engines = load_neural_pipeline()
    total_flags = 0
    flagged_lines = []
    for src_lang, tgt_lang in DIRECTIONS:
        print("\n" + "=" * 72)
        print("DIRECTION  %s -> %s" % (src_lang, tgt_lang))
        print("=" * 72)
        for src in BATTERY[src_lang]:
            try:
                out = translate_text_path(src, src_lang, tgt_lang, engines)
            except Exception as exc:  # noqa: BLE001
                out = "[CRASH: %s: %s]" % (type(exc).__name__, exc)
            fl = flags(src, out, tgt_lang)
            total_flags += len(fl)
            tag = ("   FLAG -> " + ", ".join(fl)) if fl else "   ok"
            print("IN : " + src)
            print("OUT: " + out + tag)
            print("-" * 64)
            if fl:
                flagged_lines.append("[%s->%s] %s  ==>  %s  (%s)"
                                     % (src_lang, tgt_lang, src, out, ", ".join(fl)))
    print("\n" + "#" * 72)
    print("SWEEP COMPLETE.  Total flags: %d" % total_flags)
    if flagged_lines:
        print("\nFlagged items to review:")
        for line in flagged_lines:
            print("  - " + line)
    else:
        print("No automatic red flags. Still eyeball the outputs for meaning/fluency.")


if __name__ == "__main__":
    main()
