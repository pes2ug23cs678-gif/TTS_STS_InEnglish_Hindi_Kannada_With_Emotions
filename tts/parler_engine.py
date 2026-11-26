"""Local AI4Bharat Indic-Parler-TTS engine with emotion-conditioned descriptions."""
import io
import os
import time
import asyncio
import logging
import re
from typing import Optional, Dict, Any
import numpy as np
import torch
import soundfile as sf
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
from config import AI4BHARAT_SPEAKERS, PARLER_PARAMS, HF_TOKEN
from .base import BaseTTSEngine
logger = logging.getLogger(__name__)

# Peak-amplitude floor (0..1) below which a clip is treated as silent/degenerate. Real TTS
# speech peaks well above this; the all-silence waveform Parler occasionally emits for short
# prompts sits far below it (observed ~0.008 on a blank "I am Likith" -> Hindi render).
_SILENCE_PEAK = 0.02
# Parler can draw a silent waveform under sampling; re-sample a few times before giving up.
_MAX_GEN_ATTEMPTS = 3
# Fastest plausible speaking rate (characters of prompt per second of audio); a clip implying
# a FASTER rate than this has almost certainly dropped part of its text (Parler does this as a
# whole-clip early cut-off or a partial tail drop). This rate is SCRIPT-DEPENDENT and must be,
# or the guard misfires:
#   * Devanagari / Kannada: each char packs more sound, so real speech is only ~9-13 cps. 15 is
#     a good ceiling -- it still flags the ~18-cps tail-drop case.
#   * Latin (English output): characters are "thinner", so ordinary English speech runs 15-20
#     cps. Judging it with the Indic ceiling of 15 flags EVERY normal English render as
#     "truncated" (measured: 33 chars in 2.19s = 15.07 cps, 57 chars in 3.09s = 18.4 cps -- both
#     perfectly complete audio), wasting 3 slow re-samples per chunk. Latin needs ~24.
# A false flag only costs an extra re-sample, never a wrong output. Tune with STS_MAX_CPS
# (Indic) and STS_MAX_CPS_LATIN (Latin).
try:
    # Default raised 15 -> 18: Kannada aksharas span several Unicode code points (vowel signs,
    # anusvara, conjuncts), so len() over-counts characters vs syllables and the real
    # code-points/sec rate (~16-17) sat just above the old 15 ceiling -- falsely flagging
    # COMPLETE Kannada renders as truncated and forcing them onto the Edge fallback.
    _MAX_CHARS_PER_SEC = max(10.0, float(os.getenv("STS_MAX_CPS", "18")))
except (TypeError, ValueError):
    _MAX_CHARS_PER_SEC = 18.0
try:
    _MAX_CPS_LATIN = max(10.0, float(os.getenv("STS_MAX_CPS_LATIN", "24")))
except (TypeError, ValueError):
    _MAX_CPS_LATIN = 24.0


def _cps_ceiling(chunk: str) -> float:
    """Pick the truncation ceiling by the chunk's dominant script: Latin text speaks faster per
    character than Devanagari/Kannada, so it gets the higher ceiling."""
    latin = sum(1 for c in chunk if c.isascii() and c.isalpha())
    indic = sum(1 for c in chunk if "ऀ" <= c <= "ൿ")
    return _MAX_CPS_LATIN if latin > indic else _MAX_CHARS_PER_SEC


# Minimum plausible speaking rate (chars/sec): a clip running FAR SLOWER than this for its text
# is a stretched / stuck render (Parler sometimes stumbles on a long, rare proper noun and draws
# a slow, slurred take). Script-dependent like the ceiling; tune via STS_MIN_CPS / STS_MIN_CPS_LATIN.
try:
    _MIN_CHARS_PER_SEC = max(2.0, float(os.getenv("STS_MIN_CPS", "6")))
except (TypeError, ValueError):
    _MIN_CHARS_PER_SEC = 6.0
try:
    _MIN_CPS_LATIN = max(2.0, float(os.getenv("STS_MIN_CPS_LATIN", "9")))
except (TypeError, ValueError):
    _MIN_CPS_LATIN = 9.0


# A stretched render is FAR longer than its text warrants. Short chunks have a fixed onset/
# offset overhead that makes their chars/sec look low even when perfect, so we (a) exempt very
# short chunks and (b) add a fixed pad to the allowed duration. Tune via STS_STRETCH_* env vars.
try:
    _STRETCH_PAD_S = max(0.0, float(os.getenv("STS_STRETCH_PAD_S", "1.0")))
except (TypeError, ValueError):
    _STRETCH_PAD_S = 1.0
try:
    _STRETCH_MIN_CHARS = max(0, int(os.getenv("STS_STRETCH_MIN_CHARS", "18")))
except (TypeError, ValueError):
    _STRETCH_MIN_CHARS = 18


def _cps_floor(chunk: str) -> float:
    "Minimum plausible chars/sec by dominant script (Latin speaks faster -> a higher floor)."
    latin = sum(1 for c in chunk if c.isascii() and c.isalpha())
    indic = sum(1 for c in chunk if "ऀ" <= c <= "ൿ")
    return _MIN_CPS_LATIN if latin > indic else _MIN_CHARS_PER_SEC

# A render cut off mid-word ends LOUD; a complete, vowel-final render decays into Parler trailing
# silence. Duration alone cannot separate a fast-but-complete clip from a clipped one, so we also
# test the end energy (on the edge-trimmed clip, where only the final sound remains).
_ABRUPT_TAIL_MS = 150
try:
    _ABRUPT_END_RATIO = max(0.15, float(os.getenv("STS_ABRUPT_END_RATIO", "0.35")))
except (TypeError, ValueError):
    _ABRUPT_END_RATIO = 0.35
# English/Latin words frequently END ON CONSONANTS (higher end-energy even when COMPLETE), so the
# Indic-tuned 0.35 ceiling over-flags complete English chunks and wastes retries. Latin gets a more
# lenient ceiling; vowel-final Devanagari/Kannada keep the strict one.
try:
    _ABRUPT_END_RATIO_LATIN = max(0.15, float(os.getenv("STS_ABRUPT_END_RATIO_LATIN", "0.55")))
except (TypeError, ValueError):
    _ABRUPT_END_RATIO_LATIN = 0.55


def _abrupt_threshold_for(chunk):
    """Abrupt-end ceiling by dominant script: Latin (consonant-final) lenient, Indic (vowel-final) strict."""
    latin = sum(1 for c in chunk if c.isascii() and c.isalpha())
    indic = sum(1 for c in chunk if "ऀ" <= c <= "ൿ")
    return _ABRUPT_END_RATIO_LATIN if latin > indic else _ABRUPT_END_RATIO


def _ends_abruptly(arr, sr):
    """(is_abrupt, ratio): ratio = final ~150 ms peak / whole-clip peak. Above _ABRUPT_END_RATIO
    the clip ends mid-sound (cut off); a complete, vowel-final render decays soft and reads low."""
    if arr is None or arr.size == 0:
        return False, 0.0
    peak = float(np.max(np.abs(arr)))
    if peak <= 1e-6:
        return False, 0.0
    tail_n = max(1, int(sr * _ABRUPT_TAIL_MS / 1000.0))
    ratio = float(np.max(np.abs(arr[-tail_n:]))) / peak
    return ratio > _ABRUPT_END_RATIO, ratio

# --- Post-processing of the generated waveform -----------------------------------------
_TARGET_PEAK = 0.7      # ~ -3 dBFS; a uniform output level so quiet and loud renders match
_TARGET_RMS  = 0.10     # -20 dBFS; RMS loudness target for perceived-loudness normalisation
_MAX_GAIN    = 15.0     # hard cap: prevent catastrophic gain on near-silent renders
_TRIM_THRESH = 0.02     # amplitude floor: below this at the clip edges is treated as dead air
_TRIM_PAD_MS = 40       # keep a little air around speech so it doesn't clip in/out abruptly
_CHUNK_GAP_MS = 70      # short, natural pause inserted between stitched chunks (one per clause)

# Max characters per synthesis chunk. Hindi/English render cleanly at 70, but Parler's KANNADA
# voice -- the female one especially -- DEGENERATES on long chunks (measured: a 35-word sentence
# came out quiet, 45% silence, dragging to 18.8s, with only the tail intelligible). Feeding it
# shorter chunks keeps each render inside the length it handles reliably. Tunable per machine:
# lower STS_CHUNK_LIMIT_KN further (e.g. 40) if the Kannada female still degrades.
try:
    _CHUNK_LIMIT = max(30, int(os.getenv("STS_CHUNK_LIMIT", "70")))
except (TypeError, ValueError):
    _CHUNK_LIMIT = 70
try:
    _CHUNK_LIMIT_KN = max(24, int(os.getenv("STS_CHUNK_LIMIT_KN", "48")))
except (TypeError, ValueError):
    _CHUNK_LIMIT_KN = 48


def _chunk_limit_for(text: str) -> int:
    """Kannada text gets the tighter chunk ceiling; Devanagari/Latin keep the default."""
    kn = sum(1 for c in text if "ಀ" <= c <= "೿")
    other = sum(1 for c in text
                if ("ऀ" <= c <= "ॿ") or (c.isascii() and c.isalpha()))
    return _CHUNK_LIMIT_KN if kn > other else _CHUNK_LIMIT


def _trim_silence(arr: np.ndarray, sr: int, thresh: float = _TRIM_THRESH,
                  pad_ms: int = _TRIM_PAD_MS) -> np.ndarray:
    """Strip the leading/trailing near-silence Parler pads onto clips (observed ~1.7 s of
    dead air before speech, ~2 s after), keeping a short pad so it eases in and out."""
    if arr is None or not arr.size:
        return arr
    idx = np.nonzero(np.abs(arr) > thresh)[0]
    if idx.size == 0:
        return arr
    pad = int(sr * pad_ms / 1000)
    start = max(0, int(idx[0]) - pad)
    end = min(arr.size, int(idx[-1]) + 1 + pad)
    return arr[start:end]


def _normalize_peak(arr: np.ndarray, target: float = _TARGET_PEAK) -> np.ndarray:
    """Scale to a consistent peak so a quiet render (the female clip peaked at ~0.18) and a
    loud one (the male at ~0.83) come out at the same, clearly audible level. target < 1.0
    so there is no clipping."""
    if arr is None or not arr.size:
        return arr
    peak = float(np.max(np.abs(arr)))
    if peak < 1e-5:
        return arr
    return (arr * (target / peak)).astype(np.float32)



def _normalize_loudness(arr: np.ndarray, target_rms: float = _TARGET_RMS,
                        max_gain: float = _MAX_GAIN) -> np.ndarray:
    """RMS-normalize to consistent perceived loudness; peak-clamp to prevent distortion."""
    if arr is None or not arr.size:
        return arr
    rms = float(np.sqrt(np.mean(arr ** 2)))
    if rms < 1e-6:
        return arr
    gain = min(target_rms / rms, max_gain)
    return np.clip(arr * gain, -0.99, 0.99).astype(np.float32)

# Dashes used as asides read as glued hyphens in Parler (no pause); commas are what the
# model turns into pauses. Convert em/en/figure dashes, spaced hyphens, and a hyphen glued
# between two Devanagari letters (how an em-dash survives translation, e.g. "पैकेज-जो") to
# a comma pause. A hyphen between digits (ranges like "9-5") is left alone.
_DASH_CHARS_RE = re.compile(r"\s*[‒–—―]\s*")
_SPACED_HYPHEN_RE = re.compile(r"\s+-\s+")
_DEVA_HYPHEN_RE = re.compile(r"(?<=[ऀ-ॿ])-(?=[ऀ-ॿ])")


def _dashes_to_pauses(text: str) -> str:
    "Replace dash-style punctuation with comma pauses so asides are spoken with a beat."
    text = _DASH_CHARS_RE.sub(", ", text)
    text = _SPACED_HYPHEN_RE.sub(", ", text)
    text = _DEVA_HYPHEN_RE.sub(", ", text)
    return text


# Digits read wrong in Parler Hindi (a bare "25" is not spoken as "पच्चीस"): the model has
# no Hindi number normalization, so we spell integers out in Hindi words for the TTS prompt
# only (the on-screen translation keeps the digits). Hindi 0-99 are irregular, so a full
# table is required; hundreds/thousands/lakh/crore are composed from it.
_HI_0_99 = [
    "शून्य", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ",
    "दस", "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह", "अठारह", "उन्नीस",
    "बीस", "इक्कीस", "बाईस", "तेईस", "चौबीस", "पच्चीस", "छब्बीस", "सत्ताईस", "अट्ठाईस", "उनतीस",
    "तीस", "इकतीस", "बत्तीस", "तैंतीस", "चौंतीस", "पैंतीस", "छत्तीस", "सैंतीस", "अड़तीस", "उनतालीस",
    "चालीस", "इकतालीस", "बयालीस", "तैंतालीस", "चवालीस", "पैंतालीस", "छियालीस", "सैंतालीस", "अड़तालीस", "उनचास",
    "पचास", "इक्यावन", "बावन", "तिरपन", "चौवन", "पचपन", "छप्पन", "सत्तावन", "अट्ठावन", "उनसठ",
    "साठ", "इकसठ", "बासठ", "तिरसठ", "चौंसठ", "पैंसठ", "छियासठ", "सड़सठ", "अड़सठ", "उनहत्तर",
    "सत्तर", "इकहत्तर", "बहत्तर", "तिहत्तर", "चौहत्तर", "पचहत्तर", "छिहत्तर", "सतहत्तर", "अठहत्तर", "उन्यासी",
    "अस्सी", "इक्यासी", "बयासी", "तिरासी", "चौरासी", "पचासी", "छियासी", "सत्तासी", "अट्ठासी", "नवासी",
    "नब्बे", "इक्यानवे", "बानवे", "तिरानवे", "चौरानवे", "पचानवे", "छियानवे", "सत्तानवे", "अट्ठानवे", "निन्यानवे",
]
_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
# An integer not part of a decimal (so "4.30" is left untouched -- time/decimal handling is
# a separate concern and reading its parts as cardinals would be wrong).
_INT_TOKEN_RE = re.compile(r"(?<![\d.])\d+(?![\d.])")

# An IDENTIFIER number (reference / account / OTP / PIN / phone) is read DIGIT-BY-DIGIT, not as a
# cardinal quantity: "reference number 849201" -> "eight four nine two zero one", not "eight lakh".
# Detected by an ID cue word just before the number, with only separators (spaces/commas/hyphens)
# in between -- so an intervening ordinary word (a real quantity) is NOT affected.
_HI_ID_CUES = ("नंबर", "संख्या", "रेफरेंस", "रजिस्टर्ड", "ओटीपी", "पिन", "मोबाइल", "खाता",
               "number", "OTP", "PIN", "otp", "pin")
_KN_ID_CUES = ("ಸಂಖ್ಯೆ", "ನಂಬರ್", "ಉಲ್ಲೇಖ", "ನೋಂದಾಯಿತ", "ಒಟಿಪಿ", "ಪಿನ್", "ಮೊಬೈಲ್", "ಖಾತೆ",
               "number", "OTP", "PIN", "otp", "pin")
_ID_GAP_RE = re.compile(r"[\s\d,\.\-:/]*")   # only separators/digits may sit between cue and number


def _is_id_number_context(s, pos, cues):
    """True if the number at pos follows an ID cue word with only separators between -- an
    identifier (reference/phone/OTP) read digit-by-digit, not a cardinal quantity."""
    pre = s[:pos]
    for cue in cues:
        idx = pre.rfind(cue)
        if idx != -1 and _ID_GAP_RE.fullmatch(pre[idx + len(cue):]):
            return True
    return False


def _hi_int_to_words(n: int) -> str:
    "Hindi words for a non-negative integer, Indian place values (हज़ार/लाख/करोड़)."
    if n < 100:
        return _HI_0_99[n]
    parts = []
    crore, n = divmod(n, 10000000)
    lakh, n = divmod(n, 100000)
    thousand, n = divmod(n, 1000)
    hundred, rest = divmod(n, 100)
    if crore:
        parts.append(_hi_int_to_words(crore) + " करोड़")
    if lakh:
        parts.append(_hi_int_to_words(lakh) + " लाख")
    if thousand:
        parts.append(_hi_int_to_words(thousand) + " हज़ार")
    if hundred:
        parts.append(_HI_0_99[hundred] + " सौ")
    if rest:
        parts.append(_HI_0_99[rest])
    return " ".join(parts)



# --- TTS-only pronunciation fixes -----------------------------------------------------
# Words Parler mispronounces. Keys match whole words, case-insensitively, and ONLY the
# text sent to the model is changed -- the on-screen / translated text is untouched.
# Tune by ear: if a respelling still sounds off, try another form (e.g. "saiklone",
# "sigh clone"). Add more entries as you find them.
_TTS_PRON_FIX = {
    "cyclone": "sigh-clone",
    "lyari": "lee-yaari",
}
_TTS_PRON_RE = (re.compile(r"\b(" + "|".join(re.escape(k) for k in _TTS_PRON_FIX) + r")\b",
                           re.IGNORECASE) if _TTS_PRON_FIX else None)


def _fix_pronunciations(text: str) -> str:
    "Respell tricky words for the TTS prompt only (display text is unaffected)."
    if not _TTS_PRON_RE:
        return text
    return _TTS_PRON_RE.sub(lambda m: _TTS_PRON_FIX[m.group(0).lower()], text)


# --- English number spelling for the Parler prompt (English output now uses Parler) -----
# English integers/currency are spelled out so the model says them, not misreads the digits
# or the rupee sign. Prompt-only: the on-screen translation keeps the digits (e.g. "Rs.1500").
_EN_0_19 = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen"]
_EN_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_EN_ID_CUES = ("number", "reference", "registered", "account", "mobile", "phone",
    "OTP", "PIN", "otp", "pin", "ID", "id")


def _en_int_to_words(n: int) -> str:
    # Non-negative integer -> English words, Indian place values (thousand / lakh / crore).
    if n < 20:
        return _EN_0_19[n]
    if n < 100:
        t, r = divmod(n, 10)
        return _EN_TENS[t] + ("-" + _EN_0_19[r] if r else "")
    if n < 1000:
        h, r = divmod(n, 100)
        return _EN_0_19[h] + " hundred" + (" " + _en_int_to_words(r) if r else "")
    parts = []
    crore, n = divmod(n, 10000000)
    lakh, n = divmod(n, 100000)
    thousand, rest = divmod(n, 1000)
    if crore:
        parts.append(_en_int_to_words(crore) + " crore")
    if lakh:
        parts.append(_en_int_to_words(lakh) + " lakh")
    if thousand:
        parts.append(_en_int_to_words(thousand) + " thousand")
    if rest:
        parts.append(_en_int_to_words(rest))
    return " ".join(parts)


def _numbers_to_english_words(text: str) -> str:
    # Spell integers and rupee amounts out in English for the TTS prompt only.
    text = re.sub(r"(?<=\d),(?=\d)", "", text)   # 1,500 -> 1500

    def _cur(m):
        try:
            return _en_int_to_words(int(m.group(1))) + " rupees"
        except Exception:
            return m.group(0)
    text = re.sub(r"(?:₹|Rs\.?|INR)\s?(\d+)", _cur, text, flags=re.IGNORECASE)

    def _repl(m):
        tok = m.group(0)
        try:
            if (len(tok) >= 10 or (len(tok) > 1 and tok[0] == "0")
                    or _is_id_number_context(m.string, m.start(), _EN_ID_CUES)):
                return " ".join(_EN_0_19[int(d)] for d in tok)
            return _en_int_to_words(int(tok))
        except Exception:
            return tok
    return _INT_TOKEN_RE.sub(_repl, text)


def _numbers_to_hindi_words(text: str) -> str:
    """Spell integers out in Hindi for the TTS prompt. Long digit strings (>=10 digits, e.g.
    phone numbers) and zero-padded ones are read digit-by-digit, which is how they are said;
    other integers become a single cardinal (25 -> पच्चीस, 2025 -> दो हज़ार पच्चीस)."""
    text = text.translate(_DEVA_DIGITS)
    text = re.sub(r"(?<=\d),(?=\d)", "", text)   # merge digit-grouping commas: 2,12,300 -> 212300

    def repl(m: "re.Match") -> str:
        tok = m.group(0)
        try:
            if (len(tok) >= 10 or (len(tok) > 1 and tok[0] == "0")
                    or _is_id_number_context(m.string, m.start(), _HI_ID_CUES)):
                return " ".join(_HI_0_99[int(d)] for d in tok)
            return _hi_int_to_words(int(tok))
        except Exception:  # noqa: BLE001 - never let number handling break synthesis
            return tok

    return _INT_TOKEN_RE.sub(repl, text)


# Kannada number words -- Parler has no Kannada number normalization either, so a bare "25" is
# misread. Kannada 0-99 are SANDHI-regular: tens-stem + unit, where a vowel-initial unit's
# leading vowel becomes a vowel sign on the stem's final consonant (25 = ಇಪ್ಪತ್ತು + ಐದು ->
# ಇಪ್ಪತ್ತೈದು), so they are GENERATED rather than tabled; hundreds/thousands/lakh/crore compose
# from them with the ನೂರಾ / ಸಾವಿರದ / ಲಕ್ಷದ combining forms before a remainder.
_KN_ONES = ["", "ಒಂದು", "ಎರಡು", "ಮೂರು", "ನಾಲ್ಕು", "ಐದು", "ಆರು", "ಏಳು", "ಎಂಟು", "ಒಂಬತ್ತು"]
_KN_TEENS = ["ಹತ್ತು", "ಹನ್ನೊಂದು", "ಹನ್ನೆರಡು", "ಹದಿಮೂರು", "ಹದಿನಾಲ್ಕು",
             "ಹದಿನೈದು", "ಹದಿನಾರು", "ಹದಿನೇಳು", "ಹದಿನೆಂಟು", "ಹತ್ತೊಂಬತ್ತು"]
_KN_TENS = ["", "", "ಇಪ್ಪತ್ತು", "ಮೂವತ್ತು", "ನಲವತ್ತು", "ಐವತ್ತು",
            "ಅರವತ್ತು", "ಎಪ್ಪತ್ತು", "ಎಂಬತ್ತು", "ತೊಂಬತ್ತು"]
# Native hundreds (confirmed by a Kannada speaker): 2-5 and 9 contract (ಇನ್ನೂರು, ಮುನ್ನೂರು,
# ನಾನೂರು, ಐನೂರು, ಒಂಬೈನೂರು); 6-8 stay as two words (ಆರು ನೂರು, ...). Before a REMAINDER each drops
# its final vowel: ಇನ್ನೂರು -> ಇನ್ನೂರ ಐವತ್ತು (250), ಆರು ನೂರು -> ಆರು ನೂರ ಐವತ್ತು (650).
_KN_HUNDREDS = ["", "ನೂರು", "ಇನ್ನೂರು", "ಮುನ್ನೂರು", "ನಾನೂರು", "ಐನೂರು",
                "ಆರು ನೂರು", "ಏಳು ನೂರು", "ಎಂಟು ನೂರು", "ಒಂಬೈನೂರು"]
_KN_VSIGN = {"ಅ": "", "ಆ": "ಾ", "ಇ": "ಿ", "ಈ": "ೀ", "ಉ": "ು", "ಊ": "ೂ",
             "ಎ": "ೆ", "ಏ": "ೇ", "ಐ": "ೈ", "ಒ": "ೊ", "ಓ": "ೋ", "ಔ": "ೌ"}
_KN_DIGITS = str.maketrans("೦೧೨೩೪೫೬೭೮೯", "0123456789")


def _kn_0_99(n: int) -> str:
    "Kannada words for 0-99, applying the tens+unit sandhi."
    if n == 0:
        return "ಸೊನ್ನೆ"
    if n < 10:
        return _KN_ONES[n]
    if n < 20:
        return _KN_TEENS[n - 10]
    t, u = divmod(n, 10)
    if u == 0:
        return _KN_TENS[t]
    stem = _KN_TENS[t][:-1]
    uw = _KN_ONES[u]
    if uw[0] in _KN_VSIGN:
        return stem + _KN_VSIGN[uw[0]] + uw[1:]
    return stem + uw


def _kn_int_to_words(n: int) -> str:
    "Kannada words for a non-negative integer, Indian place values (ಸಾವಿರ/ಲಕ್ಷ/ಕೋಟಿ)."
    if n < 100:
        return _kn_0_99(n)
    parts = []
    crore, n = divmod(n, 10000000)
    lakh, n = divmod(n, 100000)
    thousand, n = divmod(n, 1000)
    hundred, rest = divmod(n, 100)
    if crore:
        parts.append(_kn_int_to_words(crore) + " ಕೋಟಿ")
    if lakh:
        parts.append(_kn_0_99(lakh) + (" ಲಕ್ಷದ" if (thousand or hundred or rest) else " ಲಕ್ಷ"))
    if thousand:
        parts.append(_kn_0_99(thousand) + (" ಸಾವಿರದ" if (hundred or rest) else " ಸಾವಿರ"))
    if hundred:
        hw = _KN_HUNDREDS[hundred]
        parts.append(hw[:-1] if rest else hw)   # drop the final vowel before a remainder (ಇನ್ನೂರ ಐವತ್ತು)
    if rest:
        parts.append(_kn_0_99(rest))
    return " ".join(parts)


def _numbers_to_kannada_words(text: str) -> str:
    """Spell integers out in Kannada for the TTS prompt (25 -> ಇಪ್ಪತ್ತೈದು). Long digit strings
    (>=10 digits) and zero-padded ones are read digit-by-digit, as in the Hindi path."""
    text = text.translate(_KN_DIGITS)
    text = re.sub(r"(?<=\d),(?=\d)", "", text)   # merge digit-grouping commas: 2,12,300 -> 212300

    def repl(m: "re.Match") -> str:
        tok = m.group(0)
        try:
            if (len(tok) >= 10 or (len(tok) > 1 and tok[0] == "0")
                    or _is_id_number_context(m.string, m.start(), _KN_ID_CUES)):
                return " ".join(_kn_0_99(int(d)) for d in tok)
            return _kn_int_to_words(int(tok))
        except Exception:  # noqa: BLE001 - never let number handling break synthesis
            return tok

    return _INT_TOKEN_RE.sub(repl, text)
EMOTION_STYLE = {
    "joy": "joyful, bright and warmly cheerful",
    "happy": "joyful, bright and warmly cheerful",
    "excited": "excited, energetic and enthusiastic",
    "anger": "loud, sharp and highly agitated, with emphatic stress and clear frustration",
    "angry": "loud, sharp and highly agitated, with emphatic stress and clear frustration",
    "sadness": "sorrowful, heartfelt and deeply subdued",
    "sad": "sorrowful, heartfelt and deeply subdued",
    "fear": "tense, haunted and hushed with unease",
    "disgust": "disgusted and deeply uncomfortable",
    "surprise": "astonished and wide-eyed, with sudden shifts in energy",
    "neutral": "conversational, clear and naturally engaging",
}
# Per-emotion pace phrases: richer than pure arousal buckets, used as the first
# choice in _describe() before falling back to arousal-magnitude bucketing.
_EMOTION_PACE = {
    "anger":   "a rapid, forceful pace with sharp, raised pitch",
    "angry":   "a rapid, forceful pace with sharp, raised pitch",
    "joy":     "a lively, upbeat pace with bright, cheerful pitch",
    "happy":   "a lively, upbeat pace with bright, cheerful pitch",
    "excited": "an energetic, bounding pace with high, vibrant pitch",
    "sadness": "a slow, heavy pace with low, subdued pitch",
    "sad":     "a slow, heavy pace with low, subdued pitch",
    "fear":    "a trembling, hushed pace with unsteady pitch",
    "disgust": "a measured, contemptuous pace with flat, cold pitch",
    "surprise":"a sudden, wide-awake pace with sharply rising pitch",
}
# Intensity adverb from arousal magnitude -- pushes Parler toward a more emotive read
# for strongly-felt lines (what otherwise sounds flat).
def _intensity_word(arousal: float) -> str:
    a = abs(float(arousal or 0.0))
    if a >= 0.45:
        return "very "
    if a >= 0.22:
        return "quite "
    return ""


def _expressivity_phrase(label: str, arousal: float) -> str:
    """Wording on Indic-Parler's documented *expressivity* dimension (expressive <-> monotone).

    This is the main lever for Hindi: the model's explicit emotion *keywords* are only
    tested for 10 Indic languages (Hindi is NOT among them), so for Hindi the emotion
    descriptor words carry little, while the expressivity / pitch / rate dimensions still
    work across every language. We therefore assert an expressive (non-flat) delivery for
    ANY felt emotion -- including low-arousal ones like sadness, which are expressive
    without being energetic -- instead of only for high-arousal lines. Returns "" for
    neutral/suppressed, where a plain read is wanted.

    Only positive wording is used; we never say "not monotone", because the description
    encoder keys on the token "monotone" itself and the negation is not understood -- so
    mentioning it at all drags the delivery toward flat, the very thing we are fixing.
    """
    if label in ("neutral", ""):
        # Even neutral gets a minimum so Parler never goes fully flat.
        return "naturally expressive"
    strong = abs(float(arousal or 0.0)) >= 0.38
    # "animated" only when the emotion carries energy (arousal >= 0); low-energy emotions
    # (sad, subdued, fearful-withdrawn) stay expressive but not animated.
    if float(arousal or 0.0) >= 0.0:
        return "very expressive and animated" if strong else "expressive and animated"
    return "very expressive" if strong else "expressive"


def _hf_auth_kwargs() -> dict:
    return {"token": HF_TOKEN} if HF_TOKEN else {}
class ParlerTTSEngine(BaseTTSEngine):
    """Local Indic-Parler-TTS engine with emotion-conditioned descriptions."""
    def __init__(self, model_id: str = "ai4bharat/indic-parler-tts", device: Optional[str] = None):
        "Initialization"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, os.getenv("STS_PARLER_DTYPE", "").strip().lower(), torch.bfloat16 if (self.device.startswith("cuda") and torch.cuda.is_bf16_supported()) else torch.float32)
        self.model_id = model_id
        self._model = self._tok = self._desc_tok = None
    def _lazy_load(self):
        "Loading only when and where required for that matter!!!"
        if self._model is not None:
            return
        auth = _hf_auth_kwargs()
        self._model = ParlerTTSForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=self.dtype, **auth).to(self.device).eval()
        self._tok = AutoTokenizer.from_pretrained(self.model_id, **auth)
        if getattr(self._model.config.text_encoder, "_name_or_path", None):
            enc_id = getattr(self._model.config.text_encoder, "_name_or_path", None)
        else:
            enc_id = "google/flan-t5-large"
        self._desc_tok = AutoTokenizer.from_pretrained(enc_id, **auth)
    def warmup(self):
        """Call once at startup so the first user request isn't slow."""
        self._lazy_load()
    def is_language_supported(self, lang_code: str) -> bool:
        "Sees if the given language is supported or not"
        return str(lang_code).lower() in {"hi", "kn", "bn", "mr", "ta", "te", "en"}
    def _describe(self, lang_code: str, gender: str, emotion_meta: Optional[Dict[str, Any]]) -> str:
        "Build the natural-language style description that conditions Parler's delivery."
        gkey = "male" if str(gender).lower() == "male" else "female"
        # Use a named speaker when one is configured for this language; otherwise fall back to
        # a generic description (naming a Hindi speaker for English speech hurts the accent).
        speaker = AI4BHARAT_SPEAKERS.get(lang_code, {}).get(gkey)
        voice_intro = f"{speaker}'s voice" if speaker else f"A {gkey} speaker's voice"
        emo = (emotion_meta or {})
        # Honor the sensitivity guard: no emotional styling when tags are suppressed.
        suppress = bool(emo.get("suppress_tags"))
        label = str(emo.get("detected_emo", emo.get("label", "neutral"))).lower()
        avd = emo.get("avd") if isinstance(emo.get("avd"), dict) else {}
        arousal = float(avd.get("arousal", 0.0) or 0.0)
        # Per-emotion arousal floor: emotion2vec blended 50/50 with neutral text
        # yields low arousal (e.g. 0.46 for anger). These emotions intrinsically
        # have high arousal, so we enforce a minimum so pace/pitch stay energetic.
        _AROUSAL_FLOORS = {"anger": 0.72, "angry": 0.72, "excited": 0.78,
                           "fear": 0.65, "surprise": 0.65}
        if not suppress and label in _AROUSAL_FLOORS:
            arousal = max(arousal, _AROUSAL_FLOORS[label])
        if suppress:
            expressivity = ""
            pace = "a moderate speed and pitch"
            style = "expressive and natural"
        else:
            style = EMOTION_STYLE.get(label, "expressive and natural")
            # Scale the emotion with how strongly it was felt (arousal), so expressive
            # lines are read expressively instead of flat. Neutral stays plain.
            if label not in ("neutral",):
                style = f"{_intensity_word(arousal)}{style}"
            # The expressivity-dimension cue -- the lever that actually lifts Hindi off a
            # flat read (see _expressivity_phrase). Always present for a felt emotion.
            expressivity = _expressivity_phrase(label, arousal)
            # Continuous prosody transfer: describe the source speaker's measured delivery
            # (pace + pitch dynamics) when available, else fall back to the arousal buckets.
            pace = None
            prosody = emo.get("prosody")
            if prosody:
                try:
                    from prosody import to_parler_phrase
                    pace = to_parler_phrase(prosody)
                except Exception:  # noqa: BLE001
                    pace = None
            if not pace:
                # Per-emotion pace takes priority over generic arousal buckets.
                pace = _EMOTION_PACE.get(label)
            if not pace:
                if arousal > 0.65:
                    pace = "a rapid, urgent pace with elevated pitch"
                elif arousal > 0.38:
                    pace = "a fast pace and high pitch"
                elif arousal < -0.25:
                    pace = "a slow pace and low pitch"
                else:
                    pace = "a moderate speed and pitch"
        # Lead with the expressivity cue, folded INTO the voice clause (where Parler
        # conditions on it), rather than a trailing sentence. Positive wording only: we
        # never say "flat" or "monotone", even negated -- the description encoder keys on
        # those tokens and would pull the delivery toward the flatness we are removing.
        lead = f"{expressivity}, {style}" if expressivity else style
        return (f"{voice_intro} is {lead}, delivered with {pace}. "
                f"This is very clear audio of high recording quality, with the voice close up.")
    @staticmethod
    def _chunk(text: str, limit: int = 70, min_len: int = 25) -> list:
        """Split text into MEDIUM pieces Parler renders reliably -- not too long, not too short.

        Parler-TTS fails at both extremes: long inputs garble/stutter ("...हर रह से रह रह...
        silshim"), while very short prompts (a 6-char "हर डर।") come out SILENT. And grouping
        several sentences with their dandas (।) made it drop a clause. So we break the text
        into units (sentences -> commas -> words), then regroup them into chunks of roughly
        min_len..limit characters, JOINING units within a chunk with commas (a flowing phrase)
        instead of dandas. Staccato fragments ("हर रहस्य। हर डर।") thus ride inside a normal-
        length, comma-flowed chunk -- neither tiny-and-silent nor dropped. Short inputs that
        already fit are returned unchanged (punctuation intact).
        """
        if len(text) <= limit:
            return [text]
        # 1. Break into units no longer than the limit. Strip each unit's terminal sentence
        #    punctuation (we rejoin units with commas, so trailing dandas must go).
        units = []
        for sent in re.split(r'(?<=[.!।?])\s+', text):
            sent = sent.strip().rstrip(".!।?,،").strip()
            if not sent:
                continue
            if len(sent) <= limit:
                units.append(sent)
                continue
            for clause in re.split(r'(?<=[,،])\s+', sent):
                clause = clause.strip().rstrip(",،").strip()
                while len(clause) > limit:
                    cut = clause.rfind(" ", 0, limit)
                    cut = cut if cut > 0 else limit
                    units.append(clause[:cut].strip())
                    clause = clause[cut:].strip()
                if clause:
                    units.append(clause)
        # 2. Regroup units into comma-joined chunks of up to `limit` chars, with one rule:
        #    a SHORT fragment must never be a chunk's TAIL. Parler reliably renders the start
        #    of a chunk but drops or repeats trailing fragments -- a staccato "...हर रहस्य,
        #    हर डर" at the end of a chunk comes out as "हर रहस्य, हर रहस्य" or just vanishes.
        #    So when the running chunk is already substantial and the next unit is short, we
        #    start a NEW chunk with that fragment, letting it LEAD and flow into the clause
        #    after it (where Parler commits to generating it).
        out, cur = [], ""
        for u in units:
            if not cur:
                cur = u
            elif len(u) < min_len <= len(cur):
                out.append(cur)       # don't let a short fragment trail -> it leads next chunk
                cur = u
            elif len(cur) + len(u) + 2 <= limit:
                cur = f"{cur}, {u}"
            else:
                out.append(cur)
                cur = u
        if cur:
            # A short final remainder has nowhere to lead, so fold it into the previous chunk.
            if out and len(cur) < min_len:
                out[-1] = f"{out[-1]}, {cur}"
            else:
                out.append(cur)
        return out or [text]
    @staticmethod
    def _peak(arr: Optional[np.ndarray]) -> float:
        "Peak absolute amplitude of a waveform (0.0 for empty), used for silence detection."
        return float(np.max(np.abs(arr))) if arr is not None and arr.size else 0.0

    def _generate_chunk(self, prompt, desc, temperature: float) -> np.ndarray:
        "Run one Parler generation and coerce the output to a flat float32 mono array."
        _dev = self.device.split(":")[0]
        with torch.inference_mode():
            with torch.autocast(device_type=_dev, dtype=self.dtype, enabled=self.device.startswith("cuda")):
                out = self._model.generate(
                    input_ids=desc.input_ids, attention_mask=desc.attention_mask,
                    prompt_input_ids=prompt.input_ids, prompt_attention_mask=prompt.attention_mask,
                    do_sample=PARLER_PARAMS.get("do_sample", True),
                    temperature=temperature)
        # Parler returns an audio waveform tensor; coerce to a flat float32 mono array.
        arr = np.asarray(out.float().cpu().numpy().squeeze(), dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        return arr

    def _generate(self, text: str, description: str, lang_code: str = "", gender: str = "female") -> np.ndarray:
        "Output form: numpy array"
        self._lazy_load()
        desc = self._desc_tok(description, return_tensors="pt").to(self.device)
        base_temp = float(PARLER_PARAMS.get("temperature", 0.7))
        sr = int(getattr(self._model.config, "sampling_rate", 44100))
        parts = []
        # Hindi gender agreement: normalize 1st-person verb endings to match speaker.
        # IndicTrans2 sometimes drifts between masc/fem forms mid-translation.
        if str(lang_code).lower() == "hi":
            import re as _rg
            if str(gender).lower() == "male":
                text = _rg.sub(r"ूंगी", "ूंगा", text)   # करूंगी→करूंगा
                text = _rg.sub(r"ूँगी", "ूँगा", text)   # alt chandrabindu
                text = _rg.sub(r"ऊंगी", "ऊंगा", text)
                text = _rg.sub(r"ऊँगी", "ऊँगा", text)
                text = _rg.sub(r"ती हूं", "ता हूं", text)  # ती हूं→ता हूं
                text = _rg.sub(r"ती हूँ", "ता हूँ", text)
                text = _rg.sub(r"वाली\s+हूँ", "वाला हूँ", text)
                text = _rg.sub(r"वाली\s+हूं", "वाला हूं", text)
            else:
                text = _rg.sub(r"ूंगा", "ूंगी", text)
                text = _rg.sub(r"ूँगा", "ूँगी", text)
                text = _rg.sub(r"ऊंगा", "ऊंगी", text)
                text = _rg.sub(r"ऊँगा", "ऊँगी", text)
                text = _rg.sub(r"ता हूं", "ती हूं", text)
                text = _rg.sub(r"ता हूँ", "ती हूँ", text)
                text = _rg.sub(r"वाला\s+हूँ", "वाली हूँ", text)
                text = _rg.sub(r"वाला\s+हूं", "वाली हूं", text)
                text = _rg.sub(r"रहा हूं", "रही हूं", text)
                text = _rg.sub(r"रहा हूँ", "रही हूँ", text)
                text = _rg.sub(r"रहा था", "रही थी", text)
        # Merge subordinate-clause splits before chunking:
        # MT sometimes produces "X. To do Y" as two sentences; rejoin with comma.
        import re as _re
        text = _re.sub(r"(?<=[a-zA-Z])\.\s+([Tt]o\s+[a-z])",
                       lambda m: ", " + m.group(1).lower(), text)
        for chunk in self._chunk(text, limit=_chunk_limit_for(text)):
            prompt = self._tok(chunk, return_tensors="pt").to(self.device)
            # Shortest plausible duration for this chunk's text; a render shorter than this
            # was cut off mid-sentence (see _MAX_CHARS_PER_SEC).
            n_chars = len(chunk.strip())
            min_sec = n_chars / _cps_ceiling(chunk)
            max_sec = n_chars / _cps_floor(chunk) + _STRETCH_PAD_S
            arr, reason = None, "no attempt"
            best_arr, best_ratio = None, float("inf")
            # Parler intermittently (a) emits an all-silent waveform or (b) stops generating
            # mid-sentence (a truncated, abruptly-cut clip). Both are sampling artifacts, so
            # each attempt reseeds and nudges temperature to draw a different -- hopefully
            # clean and complete -- waveform instead of the same bad one.
            for attempt in range(_MAX_GEN_ATTEMPTS):
                try:
                    torch.manual_seed((time.time_ns() + attempt) & 0x7FFFFFFF)
                except Exception:  # noqa: BLE001 - seeding is best-effort
                    pass
                arr = self._generate_chunk(prompt, desc, base_temp + 0.15 * attempt)
                peak = self._peak(arr)
                dur = arr.size / sr if arr is not None else 0.0
                if peak < _SILENCE_PEAK:
                    reason = f"near-silent (peak={peak:.4f})"
                elif dur + 1e-3 < min_sec:
                    reason = f"truncated ({dur:.2f}s of audio for {n_chars} chars, " \
                             f"expected >= {min_sec:.2f}s)"
                else:
                    arr = _trim_silence(arr, sr)
                    if n_chars >= _STRETCH_MIN_CHARS and (arr.size / sr) > max_sec:
                        # Stretched / stuck render: FAR longer than the text warrants (Parler
                        # occasionally stumbles on a long/rare proper noun). Trimmed duration,
                        # so Parler's trailing-silence padding can't false-flag a normal clip.
                        reason = (f"stretched ({arr.size / sr:.2f}s of audio for {n_chars} "
                                  f"chars, expected <= {max_sec:.2f}s)")
                    else:
                        _abr_thr = _abrupt_threshold_for(chunk)
                        _, ratio = _ends_abruptly(arr, sr)
                        if ratio > _abr_thr:
                            reason = f"cut off mid-word (end-loudness {ratio:.2f} > {_abr_thr:.2f})"
                            if ratio < best_ratio:
                                best_ratio, best_arr = ratio, arr
                        else:
                            reason = ""
                            break
                logger.warning("Parler render rejected: %s (attempt %d/%d); retrying.",
                               reason, attempt + 1, _MAX_GEN_ATTEMPTS)
            if reason:
                # Still unusable after retries: raise so the dispatcher falls back to Edge,
                # which reliably produces a complete, audible utterance, instead of returning
                # a blank or cut-off clip.
                if best_arr is not None and reason.startswith("cut off"):
                    logger.warning("Parler still ends abruptly after %d attempts; using "
                                   "least-abrupt take (end-loudness %.2f).", _MAX_GEN_ATTEMPTS, best_ratio)
                    arr = best_arr
                else:
                    raise RuntimeError(
                        f"Parler-TTS output unusable after {_MAX_GEN_ATTEMPTS} attempts: {reason}.")
            # Trim each chunk's edge-silence BEFORE stitching, so the joins between clauses
            # don't gape (Parler pads every chunk with up to ~1 s of silence; untrimmed, that
            # stacks at each join into the long pauses heard between clauses).
            parts.append(_trim_silence(arr, sr))
        if not parts:
            raise RuntimeError("Parler-TTS produced no audio.")
        if len(parts) == 1:
            audio = parts[0]
        else:
            # Stitch chunks with a short, natural pause instead of their padded silence.
            gap = np.zeros(int(sr * _CHUNK_GAP_MS / 1000.0), dtype=np.float32)
            stitched = []
            for i, part in enumerate(parts):
                if i:
                    stitched.append(gap)
                stitched.append(part)
            audio = np.concatenate(stitched)
        # Final cleanup: trim the overall edges, then set a uniform, audible level.
        audio = _trim_silence(audio, sr)
        audio = _normalize_loudness(audio)
        return audio
    async def synthesize(self, text: str, lang_code: str, gender: str = "female",
                         emotion_meta: Optional[Dict[str, Any]] = None,
                         voice: Optional[str] = None) -> bytes:
        "Output form: bytes"
        clean = _dashes_to_pauses(text.strip())
        # Spell numbers out in Hindi so "25" is spoken "पच्चीस", not misread. Hindi output
        # only (lang_code "hi"/"hin_*"); other languages need their own number words.
        if str(lang_code).lower().startswith("hi"):
            clean = _numbers_to_hindi_words(clean)
        elif str(lang_code).lower().startswith("kn"):
            clean = _numbers_to_kannada_words(clean)
        elif str(lang_code).lower().startswith("en"):
            clean = _numbers_to_english_words(clean)
        clean = _fix_pronunciations(clean)
        if not clean:
            raise ValueError("Empty text for Parler synthesis.")
        description = voice or self._describe(str(lang_code).lower(), gender, emotion_meta)
        audio = await asyncio.to_thread(self._generate, clean, description, lang_code, gender)
        buf = io.BytesIO()
        sf.write(buf, audio, self._model.config.sampling_rate, format="WAV")
        return buf.getvalue()