"""Startup preflight checks for the Trilingual Emotion STS engine.

Catches the two failures that otherwise only surface as a wall of traceback at
warm-up time, and turns them into a short, actionable report:

  1. The gated IndicTrans2 / Indic-Parler-TTS models need a HuggingFace token.
     Without one the first download dies with "401 Unauthorized / gated repo".
  2. IndicTrans2 needs IndicTransToolkit's IndicProcessor; without it the model
     only transliterates instead of translating.

Nothing here downloads a model or loads weights -- it is cheap and safe to run
on every start. Call `report()` for a printed summary, or `summary_lines()` to
embed the status in a UI.
"""
from __future__ import annotations

import os
import shutil
from typing import List, Tuple

# config.py loads .env and normalizes the token into the environment on import.
try:
    from config import HF_TOKEN
except Exception:  # noqa: BLE001 - config should import, but never let preflight crash
    HF_TOKEN = os.getenv("HF_TOKEN", "")

OK, WARN, ERR = "ok", "warn", "error"
_ICON = {OK: "✓", WARN: "!", ERR: "✗"}

TRANSLATION_REPOS = (
    "ai4bharat/indictrans2-en-indic-1B",
    "ai4bharat/indictrans2-indic-en-1B",
)


def _hf_cache_dir() -> str:
    return os.getenv("HF_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface")


def _model_maybe_cached(repo_id: str) -> bool:
    """True if a repo looks already downloaded in the HF cache (so no token needed)."""
    base = os.path.join(_hf_cache_dir(), "hub")
    folder = "models--" + repo_id.replace("/", "--")
    return os.path.isdir(os.path.join(base, folder))


def _check_indic_processor() -> Tuple[str, str]:
    try:
        from models import INDIC_PROCESSOR, INDIC_PROCESSOR_ERROR
    except Exception as err:  # noqa: BLE001 - importing models pulls torch; degrade gracefully
        return WARN, f"Could not import the translation layer to verify IndicProcessor ({err})."
    if INDIC_PROCESSOR is not None:
        return OK, "IndicTransToolkit IndicProcessor loaded (translations will be correct)."
    return ERR, ("IndicTransToolkit IndicProcessor NOT available "
                 f"({INDIC_PROCESSOR_ERROR}). IndicTrans2 would only transliterate. "
                 "Fix: pip install IndicTransToolkit")


def _check_token() -> Tuple[str, str]:
    cached = all(_model_maybe_cached(r) for r in TRANSLATION_REPOS)
    if HF_TOKEN:
        return OK, "HuggingFace token found (gated IndicTrans2 / Parler models can download)."
    if cached:
        return OK, "No HF token, but the gated translation models are already in the cache."
    return ERR, ("No HuggingFace token set and the gated IndicTrans2 models are not cached. "
                 "Downloads will fail with HTTP 401. Fix: put HF_TOKEN=... in .env "
                 "(see .env.example) after accepting the model licenses on huggingface.co.")


def _check_device() -> Tuple[str, str]:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return OK, f"CUDA GPU detected ({name}); expressive Indic-Parler-TTS will be used."
        return WARN, ("No CUDA GPU; running on CPU. Voice falls back to Edge-TTS "
                      "(needs internet). Set STS_FORCE_TTS_ENGINE to override.")
    except Exception as err:  # noqa: BLE001
        return WARN, f"Could not query torch/CUDA ({err})."


def _check_optional() -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    # Local expressive voice
    try:
        import parler_tts  # noqa: F401
        out.append((OK, "parler_tts installed (local expressive voice available)."))
    except Exception:  # noqa: BLE001
        out.append((WARN, "parler_tts not installed; Parler voice disabled. "
                           "Enable it with: pip install -r requirements_parler.txt"))
    # Vocal-tone emotion fusion
    try:
        import funasr  # noqa: F401
        out.append((OK, "funasr installed (vocal-tone emotion2vec fusion available)."))
    except Exception:  # noqa: BLE001
        out.append((WARN, "funasr not installed; speech-to-speech uses text emotion only."))
    # Internet-dependent cloud voice fallback
    if not shutil.which("ffmpeg"):
        out.append((WARN, "ffmpeg not found (optional; only needed for loudness normalization)."))
    return out


def summary_lines() -> List[Tuple[str, str]]:
    """Return a list of (level, message) tuples describing readiness."""
    lines = [
        _check_token(),
        _check_indic_processor(),
        _check_device(),
    ]
    lines.extend(_check_optional())
    return lines


def report(prefix: str = "[preflight]") -> bool:
    """Print the readiness report. Returns False if any hard error was found."""
    ok = True
    for level, msg in summary_lines():
        if level == ERR:
            ok = False
        print(f"{prefix} {_ICON[level]} {msg}", flush=True)
    if not ok:
        print(f"{prefix} --> The app will still start, but translation will fail until the "
              f"errors above are fixed. See .env.example and README.md.", flush=True)
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if report() else 1)
