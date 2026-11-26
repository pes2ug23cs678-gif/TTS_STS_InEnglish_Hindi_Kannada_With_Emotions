"""Script-level input-language validation, shared by app.py and the REST backend.

Mirrors the SCRIPT-level guard in app.py: catches text whose script does not match the selected
input language -- an unsupported script (Malayalam/Tamil/Gujarati/... -> 'other') or a different
one of the three. It does NOT do same-script disambiguation (Marathi-under-Hindi); that needs the
fastText/IndicLID models and stays in app.py. Pure and dependency-free.
"""
import re as _re

_SCRIPT_RE = {
    "hi": _re.compile(r"[\u0900-\u097F]"),   # Devanagari
    "kn": _re.compile(r"[\u0C80-\u0CFF]"),   # Kannada
    "en": _re.compile(r"[A-Za-z]"),          # Latin
}
_ANY_LETTER_RE = _re.compile(r"[^\W\d_]", _re.UNICODE)
_LANG_NAMES = {"en": "English", "hi": "Hindi", "kn": "Kannada",
               "other": "an unsupported script"}


def detect_text_script_lang(text):
    text = text or ""
    counts = {code: len(rx.findall(text)) for code, rx in _SCRIPT_RE.items()}
    known = sum(counts.values())
    total_letters = len(_ANY_LETTER_RE.findall(text))
    if total_letters == 0:
        return None
    if (total_letters - known) > known:
        return "other"
    return max(counts, key=counts.get)


def validate_input_script(text, src_code):
    text = (text or "").strip()
    if not text:
        return True, ""
    detected = detect_text_script_lang(text)
    if detected is None:
        return True, ""
    if detected != src_code:
        name = _LANG_NAMES.get(src_code, src_code)
        if detected == "other":
            return False, (
                f"Input Language is set to {name}, but the text is in a script this system "
                f"does not support - only English, Hindi and Kannada are available. "
                f"Please enter {name} text."
            )
        return False, (
            f"Input Language is set to {name}, but the text looks like "
            f"{_LANG_NAMES.get(detected, detected)}. Please type {name} text, or change the "
            f"Input Language to {_LANG_NAMES.get(detected, detected)}."
        )
    return True, ""
