"""Deterministic post-translation text sanitizer and audio-payload verifier."""
import io
import re
import soundfile as sf
# Pre-compile patterns once at module load
# Collapse ONLY runs of 4+ identical consecutive words (true NMT degeneration loops, e.g.
# "the the the the"). We deliberately do NOT touch 2-3 repeats: Hindi reduplication is
# grammatical ("जल्दी जल्दी" = hurriedly, "धीरे धीरे" = slowly), "बहुत बहुत धन्यवाद" means "thank
# you very much", and emphatic repetition ("नहीं नहीं नहीं", "very very") is exactly the emotional
# signal this engine exists to preserve. (ASR transcripts get their own stutter cleaner upstream.)
RE_REPEATED_WORDS = re.compile(r'(?<!\S)(\S+)(?:\s+\1(?!\S)){3,}', flags=re.IGNORECASE)
RE_SPACED_PUNCT = re.compile(r'\s+([,.?!।])')
# Collapse only runs of the SAME terminal mark (!!!->!, ???->?, ।।->।). Periods are excluded so
# an ellipsis "..." survives (hesitation cue + TTS pause), and because the marks must match,
# mixed emphasis like "?!" is left intact.
RE_DUPLICATE_PUNCT = re.compile(r'([?!।])\1+')
# IndicTrans2's English decode renders a contraction apostrophe as a COMMA token, so "I'm"
# comes out of the model as "I , m" and MosesDetokenizer faithfully yields "I, m". Repair the
# standard enclitics that follow such a mangled comma ("I, m"->"I'm", "it, s"->"it's",
# "we, re"->"we're", "they, ve"->"they've", "you, ll"->"you'll", "I, d"->"I'd", "n, t"->"n't").
# The enclitic must be a whole lowercase token (\b both sides) so ordinary commas ("Yes, sir",
# "No, really", "Plan, B") are never touched.
RE_EN_CONTRACTION = re.compile(r"\b([A-Za-z]+),[ \t]+(m|s|re|ve|ll|d|t)\b")
# Rejoin negatives Moses leaves spaced after the repair above ("do n't"->"don't",
# "ca n't"->"can't", "wo n't"->"won't", "is n't"->"isn't").
RE_EN_NT = re.compile(r"\b([A-Za-z]+) (n't)\b")
# Language-specific known SentencePiece/BPE truncation patches
# Replaced (?<=^|\s) with (^|\s) capturing groups to guarantee fixed width
KN_PATCHES = [
    (re.compile(r'^\s*ವವನ್ನು\b'), 'ಶವವನ್ನು'),
    (re.compile(r'\bಫೋಟೋ ತೆಗೆಯಲು ಪೋಸ್\b'), 'ಫೋಟೋಗೆ ಪೋಸ್'),
    (re.compile(r'\b೮ನೇ ವೇದಿಕೆ\b|\b8ನೇ ವೇದಿಕೆ\b'), '8ನೇ ಪ್ಲಾಟ್‌ಫಾರ್ಮ್'),
    (re.compile(r'\bತಾಯಿಗಿಂತ ದೊಡ್ಡವನು\b'), 'ತಾಯಿಗಿಂತ ದೊಡ್ಡವರು'),
    (re.compile(r'\bಉಸಿರಾಟವನ್ನು ದೂರ ತೆಗೆದುಕೊಳ್ಳಿ\b'), 'ಉಸಿರನ್ನೇ ಕಸಿದುಕೋ'),
    (re.compile(r'\bದೂರ ಸ್ಲೈಡ್\b'), 'ಮರೆಯಾಗಿ ಹೋದೆ'),
    (re.compile(r'(^|\s)ರಿಯತಮೆ(?=\s|[?.!,]|$)'), r'\1ಪ್ರಿಯತಮೆ'),
    (re.compile(r'(^|\s)ನಿರ್ಮಿಸಿದೆವು(?=\s|[?.!,]|$)'), r'\1ಕಟ್ಟಿದೆವು'),
    (re.compile(r'ಅದು ಕಾಣಿಸಿಕೊಳ್ಳುವವರೆಗೆ ಅದನ್ನು'), 'ಅದು ಕಾಣಿಸುವವರೆಗೆ'),
    # 'सुबह होना' = dawn breaking; MT maps to 'ಬೆಳಿಗ್ಗೆ ಏಳಲು' (waking up) — wrong semantic.
    (re.compile(r'ಬೆಳಿಗ್ಗೆ\s+ಏಳಲು\s+ಎಷ್ಟು\s+ಸಮಯ\s+ತೆಗೆದುಕೊಳ್ಳುತ್ತದೆ'),
     'ಬೆಳಗಾಗಲು ಎಷ್ಟು ಹೊತ್ತು ಬೇಕಾಗುತ್ತದೆ'),
]

EN_PATCHES = [
    # 'सुबह होना' = dawn breaking; MT renders as 'to get up in the morning' (waking up) — wrong.
    (re.compile(r'how long it takes to get up in the morning', re.IGNORECASE),
     'how long it takes for morning to come'),
]

HI_PATCHES = [
    (re.compile(r'\bउल्टी करने की इच्छा\b'), 'उल्टी आ रही'),
    (re.compile(r'\bआंखों से संपर्क\b'), 'नज़रें'),
    (re.compile(r'\bएक माँ से बड़ी\b'), 'माँ से बड़ी'),
    (re.compile(r'\bमेरी सांस दूर ले लो\b'), 'मेरी सांसें थाम लो'),
    (re.compile(r'\bआप बारी और कहते हैं\b'), 'तुम मुड़कर कहते हो'),
    (re.compile(r'\bचोट मत करो\b'), 'चोट मत पहुँचाइए'),
    (re.compile(r'प्रकट\s+न\s+हो\s+जाए'), 'साफ़ दिखने न लगे'),
    (re.compile(r'(^|\s)वादा\s+करें(?=\s|[?.!,।]|$)'), r'\1वादा करो'),
    (re.compile(r'(^|\s)(तस्वीर|फोटो)\s+लें(?=\s|[?.!,।]|$)'), r'\1\2 लो'),
]
# --- Acronym spell-out repair -------------------------------------------------
# IndicTrans2 renders English acronyms letter-by-letter and terminates each letter
# with a Devanagari danda ('।') or an ASCII dot, e.g. "BES" -> "बी। ई। एस।".
# Each danda is a sentence terminator, so TTS reads long pauses between the letters
# and the speech sounds broken. We detect runs of 2+ danda/dot-separated *letter
# names* and rejoin them as spaced letters ("बी ई एस"), which TTS reads smoothly as
# an acronym. Only known letter names are collapsed, so ordinary short words that
# happen to end a sentence (e.g. "हाँ। ठीक है।") are never touched.
HI_LETTER_NAMES = ["डब्ल्यू", "क्यू", "एक्स", "ज़ेड", "जेड", "वाई", "एच", "एफ",
                   "एल", "एम", "एन", "एस", "आई", "आर", "बी", "सी", "डी", "जी",
                   "के", "पी", "टी", "यू", "वी", "जे", "ए", "ई", "ओ"]
KN_LETTER_NAMES = ["ಡಬ್ಲ್ಯೂ", "ಕ್ಯೂ", "ಎಕ್ಸ್", "ಜೆಡ್", "ಹೆಚ್", "ಎಫ್", "ಎಲ್", "ಎಂ",
                   "ಎನ್", "ಎಸ್", "ಆರ್", "ವೈ", "ಬಿ", "ಸಿ", "ಡಿ", "ಜಿ", "ಕೆ", "ಪಿ",
                   "ಟಿ", "ಯು", "ವಿ", "ಜೆ", "ಐ", "ಎ", "ಇ", "ಓ"]


def _build_acronym_regex(names):
    alt = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    run = re.compile(rf"(?:(?:{alt})\s*[।.]+\s*){{2,}}")
    letters = re.compile(alt)
    return run, letters


RE_HI_ACRONYM, RE_HI_LETTER = _build_acronym_regex(HI_LETTER_NAMES)
RE_KN_ACRONYM, RE_KN_LETTER = _build_acronym_regex(KN_LETTER_NAMES)


def _despell_acronyms(text: str, run_re, letter_re) -> str:
    def repl(m):
        return " ".join(letter_re.findall(m.group(0))) + " "
    return run_re.sub(repl, text)


def verify_and_clean_output(text: str, lang: str) -> str:
    """
    Deterministic sub-millisecond lexical sanitizer and orthographic verifier.
    Catches BPE clipping, stutter loops, and punctuation artifacts.
    """
    if not text:
        return ""
    # Accept either ISO codes ('en'/'hi'/'kn') OR IndicTrans2 tags
    # ('eng_Latn'/'hin_Deva'/'kan_Knda'), so the language branches below always fire
    # no matter which form the caller passes.
    lang = {"eng_Latn": "en", "hin_Deva": "hi", "kan_Knda": "kn",
            "eng": "en", "hin": "hi", "kan": "kn"}.get(lang, lang)
    out = text.strip()
    # 1. Repetition suppression (stutters / autoregressive loops)
    out = RE_REPEATED_WORDS.sub(r'\1', out)
    # 2. Targeted subword & morphological repair
    if lang == "kn":
        for pat, repl in KN_PATCHES:
            out = pat.sub(repl, out)
        # Rejoin acronyms spelled out with dandas/dots (e.g. "ಬಿ. ಇ. ಎಸ್." -> "ಬಿ ಇ ಎಸ್")
        out = _despell_acronyms(out, RE_KN_ACRONYM, RE_KN_LETTER)
    elif lang == "hi":
        for pat, repl in HI_PATCHES:
            out = pat.sub(repl, out)
        # Enforce uniform Purna Viram over trailing ASCII dots, but NOT on abbreviation/acronym
        # periods: a period after a single Latin letter ("U.S.A.") or after a standard Hindi
        # title abbreviation (डॉ./श्री./प्रो./कु./सुश्री) is not a sentence end, and converting it
        # to a danda makes the TTS pause mid-name (the old "डॉ।" fragmentation bug).
        out = re.sub(r'(?<![A-Za-z])(?<!डॉ)(?<!श्री)(?<!प्रो)(?<!कु)(?<!सुश्री)\.(?=\s|$)', '।', out)
        # Indic->Indic models sometimes emit the danda already attached to a title abbreviation
        # ("डॉ।"), which no period-based rule can catch and which makes TTS pause mid-name.
        # Normalize a known abbreviation + danda back to a period so the name stays together.
        out = re.sub(r'(डॉ|श्री|प्रो|कु|सुश्री)।', r'\1.', out)
        # Rejoin acronyms spelled out with dandas (e.g. "बी। ई। एस।" -> "बी ई एस"),
        # so the TTS engine does not read a full-stop pause after every letter.
        out = _despell_acronyms(out, RE_HI_ACRONYM, RE_HI_LETTER)
    elif lang == "en":
        for pat, repl in EN_PATCHES:
            out = pat.sub(repl, out)
        # Repair contraction apostrophes the model emits as commas (see RE_EN_CONTRACTION).
        out = RE_EN_CONTRACTION.sub(r"\1'\2", out)
        out = RE_EN_NT.sub(r"\1\2", out)
        # IndicTrans2 often prepends a spurious discourse connective ("And"/"But") to a
        # standalone translation. In single-sentence input there is no prior clause for it to
        # join, so it is almost always an artifact. Strip it only when it begins the text and is
        # a WHOLE word followed by a space (so "Android"/"Anderson"/"Butter" are never touched),
        # then re-capitalise the new first letter.
        _stripped = re.sub(r"^(?:And|But)\s+", "", out)
        if _stripped != out and _stripped:
            out = _stripped[0].upper() + _stripped[1:]
    # 3. Punctuation coherence
    out = RE_SPACED_PUNCT.sub(r'\1', out)
    out = RE_DUPLICATE_PUNCT.sub(r'\1', out)
    return out
# ==================== Audio Verification & SLA Assertions ====================
def verify_audio_payload(audio_bytes: bytes, expected_engine: str = None) -> dict:
    """
    Validates audio byte integrity, minimum payload size, format decodability,
    and valid sampling rate (24kHz Edge-TTS / 44.1kHz AI4Bharat).
    """
    if not audio_bytes or len(audio_bytes) < 1024:
        if audio_bytes:
            raise ValueError(f"Audio payload too small or empty: {len(audio_bytes)} bytes")
        raise ValueError("Audio payload too small or empty: 0 bytes")
    # Read binary header via soundfile buffer
    buf = io.BytesIO(audio_bytes)
    try:
        with sf.SoundFile(buf) as sound_file:
            sr = sound_file.samplerate
            channels = sound_file.channels
            frames = len(sound_file)
            duration = frames / float(sr)
            # Assert valid frequency profiles
            if expected_engine == "AI4Bharat" and sr != 44100:
                raise AssertionError(f"Expected 44100 Hz for AI4Bharat, got {sr} Hz")
            if expected_engine == "Edge-TTS" and sr not in [24000, 48000]:
                raise AssertionError(f"Expected 24000/48000 Hz for Edge-TTS, got {sr} Hz")
            return {
                "valid": True,
                "sampling_rate": sr,
                "channels": channels,
                "duration_seconds": round(duration, 3),
                "payload_bytes": len(audio_bytes)
            }
    except Exception as e:
        # Fallback verification for raw MP3 byte streams (Edge-TTS default stream)
        if audio_bytes[:3] == b'ID3' or audio_bytes[:2] in [b'\xff\xfb', b'\xff\xf3']:
            return {
                "valid": True,
                "format": "mp3",
                "payload_bytes": len(audio_bytes)
            }
        raise AssertionError(f"Corrupted or invalid audio stream: {str(e)}") from e

def assert_latency_sla(total_elapsed: float, budget_seconds: float = 1.2) -> bool:
    """
    Asserts end-to-end processing meets target response budget.
    """
    if total_elapsed > budget_seconds:
        print(f"Latency SLA breached: {total_elapsed:.3f}s > {budget_seconds:.3f}s")
        return False
    return True

