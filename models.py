"""Neural engine loaders and translation/emotion inference for the STS pipeline."""
import os
import logging
import threading
import re
import streamlit as st
import torch

logger = logging.getLogger(__name__)
import torch.nn.functional as F
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification
)
from asr.whisper_engine import load_whisper_engine
from config import (
    TRANSLATION_MODEL_EN_INDIC,
    TRANSLATION_MODEL_INDIC_EN,
    TRANSLATION_MODEL_INDIC_INDIC,
    EMOTION_MODEL_ID,
    LANG_MAP,
    DISPLAY_TO_CODE,
    ROBERTA_TO_PROSODY_MAP,
    DEVICE,
    TORCH_DTYPE,
    TRANSLATION_DEVICE,
    TRANSLATION_DTYPE,
    ASR_DEVICE,
    HF_TOKEN,
)
from rules_engine import apply_conversational_surface_rules
from verifier import verify_and_clean_output
# Attempt to load indic-transliteration for script alignment
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
except ImportError:
    sanscript = None
    transliterate = None
# Load the IndicTransToolkit pre/post-processor. IndicTrans2 is TRAINED to read a
# specific normalized+tokenized input and to emit output in an intermediate form that
# MUST be post-processed back into native script. The IndicProcessor does both halves.
# Without it, the model's raw decode looks like romanized/transliterated text rather
# than a real translation -- which is exactly the "it only transliterates" bug.
# So we try every known import path and, if none works, we record why and make the
# translation path fail LOUDLY (see _require_indic_processor) instead of silently
# emitting transliteration.
INDIC_PROCESSOR = None
INDIC_PROCESSOR_ERROR = None

def _init_indic_processor():
    """Instantiate IndicProcessor from whichever module path this version exposes."""
    last_err = None
    for _import in (
        lambda: __import__("IndicTransToolkit.processor",
                           fromlist=["IndicProcessor"]).IndicProcessor,
        lambda: __import__("IndicTransToolkit", fromlist=["IndicProcessor"]).IndicProcessor,
    ):
        try:
            cls = _import()
            return cls(inference=True), None
        except Exception as err:  # noqa: BLE001 - try the next import path
            last_err = err
    return None, last_err

INDIC_PROCESSOR, INDIC_PROCESSOR_ERROR = _init_indic_processor()
if INDIC_PROCESSOR is None:
    logger.warning(
        "IndicTransToolkit's IndicProcessor could not be loaded (%s). IndicTrans2 "
        "CANNOT produce correct translations without it and would only transliterate, "
        "so translation will raise until it is installed: pip install IndicTransToolkit",
        INDIC_PROCESSOR_ERROR,
    )

def _require_indic_processor():
    """Raise a clear, actionable error rather than let the model transliterate."""
    if INDIC_PROCESSOR is None:
        raise RuntimeError(
            "IndicTrans2 requires IndicTransToolkit's IndicProcessor for correct "
            "translation (without it the model only transliterates). It failed to load: "
            f"{type(INDIC_PROCESSOR_ERROR).__name__}: {INDIC_PROCESSOR_ERROR}. "
            "Fix: pip install IndicTransToolkit  (or: pip install -r requirements.txt)."
        )
# Fixed regex: fixed-width patterns only
RE_CLAUSE_SPLIT = re.compile(r'[\r\n]+|(?<=[.!?])\s+')
RE_TRAILING_PUNCT = re.compile(r'[\.।\?!]+$')

# --- Sentence splitting that respects abbreviations ------------------------------------
# The naive "split on [.!?] + whitespace" treated every abbreviation period ("Dr.", "St.",
# "N.", "a.m.") as a sentence boundary, so ONE sentence was shattered into fragments, each
# translated out of context and rejoined with Hindi dandas ("।"). That is exactly why
# "Dr. Smith ... 123 N. Maple St." came out as "डॉ। ... 123 एन ...। Maple St।" and the
# speech was read as four choppy sentences. Shield those periods before splitting so only
# real sentence breaks split.
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')
_PROTECT_DOT = ""  # private-use sentinel standing in for a non-breaking period
_ABBR_PROTECT_PATTERNS = (
    # Titles / common abbreviations (case-insensitive), their trailing period included.
    re.compile(r'\b(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr|St|Mt|Ave|Blvd|Rd|No|vs|etc|Inc|Ltd|Co|'
               r'Corp|Dept|Univ|Gen|Col|Capt|Lt|Sgt|Gov|Sen|Rep|Pres|approx|Fig)\.',
               re.IGNORECASE),
    re.compile(r'\b[A-Z]\.'),                       # single uppercase initials: "N." "S." "E."
    re.compile(r'\b[ap]\.\s?m\.', re.IGNORECASE),   # "a.m." / "p.m." (also "a. m.")
    re.compile(r'\b(?:e\.g|i\.e|u\.s|u\.k)\.', re.IGNORECASE),
    re.compile(r'\d\.\d'),                          # decimals: "4.30"
)

def _split_sentences(line: str) -> list:
    """Split a line into sentences without breaking on abbreviation/initial/decimal dots."""
    protected = line
    for pat in _ABBR_PROTECT_PATTERNS:
        protected = pat.sub(lambda m: m.group(0).replace(".", _PROTECT_DOT), protected)
    return [c.replace(_PROTECT_DOT, ".").strip()
            for c in _SENT_SPLIT_RE.split(protected) if c.strip()]


# --- Kannada unpunctuated multi-clause splitter ------------------------------------------
# IndicTrans2 is sentence-level: fed a long unpunctuated Kannada run-on (three clauses
# with no danda/period) it treats the whole thing as one sentence and emits EOS early,
# silently dropping the tail clause. This splitter cuts at:
#   (a) coordinator words  -- ಮತ್ತು / ಅಥವಾ / ಆದರೆ
#   (b) clause-final verb endings (ಿತು, ಿಲ್ಲ, ಿದೆ, ೇನೆ, ...) when followed by another
#       Kannada word (not at end-of-string, so we don't split the final predicate off).
# Activated only for Kannada source text; disable with STS_KN_CLAUSE_SPLIT=0.
_KN_COORDINATOR_RE = re.compile(r'(?<=[^\s])\s+(ಮತ್ತು|ಅಥವಾ|ಆದರೆ)\s+')
_KN_VERB_FINAL_RE = re.compile(
    r'(ಿತು|ಿತ್ತು|ಿದೆ|ಿಲ್ಲ|ಿದ್ದಾರೆ|ಿದ್ದಾನೆ|ಿದ್ದಾಳೆ|ೇನೆ|ಿರುವೆ|ಿದ್ದೆ)'
    r'(\s+)(?=[ಀ-೿])'   # only split when ANOTHER Kannada word follows
)
_KN_MIN_CLAUSE_CHARS = 8   # don't split into fragments shorter than this


def _split_kn_clauses(text: str) -> list:
    """Split unpunctuated Kannada text at coordinator words and clause-final verb endings."""
    # Step 1: split at coordinator words; drop the coordinator token itself
    # (re.split with one capture group yields [before, coord1, after1, coord2, after2, ...])
    raw = _KN_COORDINATOR_RE.split(text)
    segments = [raw[i].strip() for i in range(0, len(raw), 2) if raw[i].strip()]
    if not segments:
        segments = [text.strip()]

    # Step 2: for any still-long punctuation-free segment, split after the FIRST
    # clause-final verb ending that is followed by another Kannada word
    result = []
    for seg in segments:
        if len(seg) < 40 or re.search(r'[.!?।]', seg):
            result.append(seg)
            continue
        m = _KN_VERB_FINAL_RE.search(seg)
        if m:
            left = seg[:m.end(1)].strip()
            right = seg[m.end():].strip()
            if len(left) >= _KN_MIN_CLAUSE_CHARS and len(right) >= _KN_MIN_CLAUSE_CHARS:
                result.append(left)
                result.append(right)
                continue
        result.append(seg)

    return [s for s in result if s.strip()]
def _hf_auth_kwargs() -> dict:
    """Pass the HF token to from_pretrained for the gated IndicTrans2 repos."""
    return {"token": HF_TOKEN} if HF_TOKEN else {}

def _load_translation_model(model_id: str):
    """Loads one IndicTrans2 seq2seq model + tokenizer, quantizing on CPU, and warms it."""
    auth = _hf_auth_kwargs()
    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, **auth)
        mod = AutoModelForSeq2SeqLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=TRANSLATION_DTYPE,
            low_cpu_mem_usage=True, **auth).to(TRANSLATION_DEVICE)
    except Exception as err:  # noqa: BLE001 - turn the raw 401 into an actionable message
        msg = str(err)
        if "401" in msg or "gated" in msg.lower() or "restricted" in msg.lower():
            raise RuntimeError(
                f"Access to the gated model '{model_id}' was refused (HTTP 401). "
                "Set a HuggingFace token with access to it: accept the license at "
                f"https://huggingface.co/{model_id} , create a READ token at "
                "https://huggingface.co/settings/tokens , and put it in .env as "
                "HF_TOKEN=... (see .env.example)."
            ) from err
        raise
    if TRANSLATION_DEVICE == "cpu":
        mod = torch.ao.quantization.quantize_dynamic(mod, {torch.nn.Linear}, dtype=torch.qint8)
    mod.generation_config.max_length = None
    mod.eval()
    try:
        with torch.inference_mode():
            warm = tok(["warmup"], return_tensors="pt").to(TRANSLATION_DEVICE)
            _ = mod.generate(**warm, max_new_tokens=2, max_length=None)
    except Exception:
        pass
    return tok, mod

def _load_emotion_model():
    """Loads + quantizes (on CPU) + warms the English RoBERTa emotion classifier."""
    auth = _hf_auth_kwargs()
    emo_tok = AutoTokenizer.from_pretrained(EMOTION_MODEL_ID, **auth)
    emo_mod = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL_ID,
        torch_dtype=TORCH_DTYPE, low_cpu_mem_usage=True, **auth).to(DEVICE)
    if DEVICE == "cpu":
        emo_mod = torch.ao.quantization.quantize_dynamic(emo_mod,
            {torch.nn.Linear}, dtype=torch.qint8)
    emo_mod.eval()
    try:
        with torch.inference_mode():
            _ = emo_mod(**emo_tok(["warmup"], return_tensors="pt").to(DEVICE))
    except Exception:
        pass
    return emo_tok, emo_mod

class NeuralEngines:
    """Lazy engine registry: each model is loaded on first use, not at startup.

    This avoids loading models a session never needs — e.g. the Indic->English 1B model
    during an English-only run (saves a whole 1B model + its RAM), or the Whisper ASR model
    when the user only types text. Access with engines["en_indic"] / .en_indic etc.
    """
    _LOADERS = {
        "en_indic": lambda: _load_translation_model(TRANSLATION_MODEL_EN_INDIC),
        "indic_en": lambda: _load_translation_model(TRANSLATION_MODEL_INDIC_EN),
        "indic_indic": lambda: _load_translation_model(TRANSLATION_MODEL_INDIC_INDIC),
        "emotion": _load_emotion_model,
        "asr": lambda: load_whisper_engine(device=ASR_DEVICE),
    }

    def __init__(self):
        self._cache = {}
        self._lock = threading.Lock()

    def __getitem__(self, key):
        if key not in self._cache:
            with self._lock:
                if key not in self._cache:
                    logger.info("Lazy-loading neural engine: %s", key)
                    self._cache[key] = NeuralEngines._LOADERS[key]()
        return self._cache[key]

    en_indic = property(lambda self: self["en_indic"])
    indic_en = property(lambda self: self["indic_en"])
    emotion = property(lambda self: self["emotion"])
    asr = property(lambda self: self["asr"])

@st.cache_resource(show_spinner="Preparing neural engines (models load on first use)...")
def load_neural_pipeline():
    """Return a lazy engine registry. Model weights load on first use, keeping startup fast
    and skipping models this session never touches."""
    return NeuralEngines()

def _iso2(code: str) -> str:
    """Normalize any language spelling ('English', 'hin_Deva', 'hi', ...) to 'en'/'hi'/'kn'."""
    key = str(code).strip().lower()
    return DISPLAY_TO_CODE.get(key, key[:2])

def translate_route(text: str, src_lang: str, tgt_lang: str, engines: dict,
                    pre_map=None, surface_rules=None) -> str:
    """Translate between any pair of {en, hi, kn}, choosing the right model.

    - en -> hi/kn : en_indic model
    - hi/kn -> en : indic_en model
    - hi <-> kn   : pivot through English (src -> en -> tgt)
    """
    src = _iso2(src_lang)
    tgt = _iso2(tgt_lang)
    clean = str(text).strip()
    if not clean or src == tgt:
        return clean
    pre_map = pre_map or []
    surface_rules = surface_rules or {}

    if src == "en":
        tok, mod = engines["en_indic"]
        return translate_text(clean, "en", tgt, tok, mod, pre_map, surface_rules)
    if tgt == "en":
        tok, mod = engines["indic_en"]
        return translate_text(clean, src, "en", tok, mod, pre_map, surface_rules)
    # indic -> indic (e.g. kn -> hi): prefer the DIRECT indic-indic model -- one hop, no
    # English pivot, so translation error does not compound. Fall back to the English pivot
    # if the direct model is disabled (TRANSLATION_MODEL_INDIC_INDIC="") or fails to load.
    if TRANSLATION_MODEL_INDIC_INDIC:
        try:
            tok_ii, mod_ii = engines["indic_indic"]
            direct = translate_text(clean, src, tgt, tok_ii, mod_ii, pre_map, surface_rules)
            if not direct.startswith("[TRANSLATION_ERROR"):
                return direct
            logger.warning("Direct indic-indic translation errored (%s); pivoting through "
                           "English.", direct)
        except Exception as err:  # noqa: BLE001 - any load/inference failure -> pivot route
            logger.warning("Direct indic-indic model unavailable (%s: %s); pivoting through "
                           "English.", type(err).__name__, err)
    # Fallback: pivot through English (src -> en -> tgt).
    tok_ie, mod_ie = engines["indic_en"]
    english = translate_text(clean, src, "en", tok_ie, mod_ie, pre_map, surface_rules)
    if english.startswith("[TRANSLATION_ERROR"):
        return english
    tok_ei, mod_ei = engines["en_indic"]
    return translate_text(english, "en", tgt, tok_ei, mod_ei, pre_map, surface_rules)
@torch.inference_mode()
def detect_text_emotion(text: str, emo_tokenizer, emo_model) -> tuple[str, float, dict]:
    """Returns the dominant emotion, its confidence, and the full probability spread."""
    inputs = emo_tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
    logits = emo_model(**inputs).logits
    probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
    id2label = emo_model.config.id2label
    scores = {id2label[i].lower(): float(probs[i]) for i in range(len(probs))}
    best_raw_label = max(scores, key=scores.get)
    canonical_emotion = ROBERTA_TO_PROSODY_MAP.get(best_raw_label, "neutral")
    confidence = scores[best_raw_label]
    prob_dist = {ROBERTA_TO_PROSODY_MAP.get(k, k): round(v, 3) for k, v in scores.items()}
    return canonical_emotion, confidence, prob_dist
def _generate_translation(trans_mod, inputs, dynamic_max):
    """Runs IndicTrans2 generation, retrying without the KV cache if the model's
    custom decoder hits the cache-format incompatibility that surfaces in some
    transformers builds as "'NoneType' object has no attribute 'shape'".
    """
    gen_kwargs = dict(
        max_new_tokens=dynamic_max,
        min_new_tokens=1,
        num_beams=4,
        repetition_penalty=1.15,
        do_sample=False,
    )
    try:
        return trans_mod.generate(**inputs, use_cache=True, **gen_kwargs)
    except AttributeError as err:
        # IndicTrans2's remote decoder expects the legacy (tuple) KV-cache layout.
        # Newer transformers hand it a Cache object, so past_key_values comes back
        # as None and the attention code raises AttributeError on `.shape`.
        # Retrying with the cache disabled avoids that path (slower, but correct).
        logger.warning(
            "generate() failed with cache enabled (%s); retrying with use_cache=False.",
            err,
        )
        return trans_mod.generate(**inputs, use_cache=False, **gen_kwargs)

@torch.inference_mode()
def _translate_single_clause(clause: str, src_lang: str,
tgt_lang: str, trans_tok, trans_mod,
pre_map, surface_rules) -> str:
    if not clause.strip():
        return ""
    normalized = clause.strip()
    if pre_map:
        for pat, rep in pre_map:
            normalized = pat.sub(rep, normalized)
    # Pad punctuation to prevent BPE token fusion on short imperatives
    normalized = re.sub(r'([!?,।])', r' \1 ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    src_code = LANG_MAP.get(src_lang, src_lang)
    tgt_code = LANG_MAP.get(tgt_lang, tgt_lang)
    # Pre-process via IndicProcessor. It is REQUIRED: feeding IndicTrans2 raw tag-prefixed
    # text and decoding without post-processing produces romanized/transliterated output,
    # not a translation. Fail loudly instead of silently transliterating.
    _require_indic_processor()
    batch = INDIC_PROCESSOR.preprocess_batch([normalized], src_lang=src_code, tgt_lang=tgt_code)
    inputs = trans_tok(batch, truncation=True, padding="longest",
    return_tensors="pt", return_attention_mask=True).to(TRANSLATION_DEVICE)
    input_len = inputs["input_ids"].shape[1]
    dynamic_max = min(int(input_len * 2.0) + 24, 512)
    generated = _generate_translation(trans_mod, inputs, dynamic_max)
    decoded = trans_tok.batch_decode(generated, skip_special_tokens=True)
    # Post-process back into native script (required; see _require_indic_processor).
    try:
        decoded_text = INDIC_PROCESSOR.postprocess_batch(decoded, lang=tgt_code)[0].strip()
    except Exception as err:  # noqa: BLE001
        logger.warning("IndicProcessor.postprocess_batch failed (%s); using raw decode "
                       "for this clause, which may read as transliteration.", err)
        decoded_text = decoded[0].strip()
    # Pass ISO code (e.g. 'kn', 'hi') to downstream rules and verifier
    ISO_MAP = {"kan_Knda": "kn", "hin_Deva": "hi", "eng_Latn": "en", "kn": "kn", "hi": "hi"}
    iso_tgt = ISO_MAP.get(tgt_code, ISO_MAP.get(tgt_lang, tgt_lang[:2]))
    # Convert Devanagari script representation to native Kannada script if needed
    if iso_tgt == "kn" and re.search(r'[\u0900-\u097F]', decoded_text):
        if transliterate and sanscript:
            decoded_text = transliterate(decoded_text, sanscript.DEVANAGARI, sanscript.KANNADA)
    # Detach virama-fused Devanagari imperative verbs (e.g., 'शब्देखें' -> 'शब्द देखें')
    if iso_tgt == "hi":
        decoded_text = re.sub(r'शब्(देख(?:ें|िए)|सुन(?:ें|िए)|पढ़(?:ें|िए))',
        r'शब्द \1', decoded_text)
        decoded_text = re.sub(r'(शब्द)(देख(?:ें|िए)|सुन(?:ें|िए)|पढ़(?:ें|िए))', r'\1 \2', decoded_text)
    # Restore time colons the model emits as visarga (e.g. '3ः30' -> '3:30')
    decoded_text = re.sub(r'(?<=\d)[ःಃ](?=\d)', ':', decoded_text)
    # Execute surface rules whether passed as file path or config dict
    try:
        if surface_rules:
            surface_fixed = apply_conversational_surface_rules(decoded_text, iso_tgt, surface_rules)
        else:
            surface_fixed = decoded_text
    except Exception:
        surface_fixed = decoded_text
    return verify_and_clean_output(surface_fixed, iso_tgt)
def _translate_keeping_colons(clause: str, src_lang: str,
tgt_lang: str, trans_tok, trans_mod,
pre_map, surface_rules) -> str:
    """Translate a clause so the NMT model never sees a mid-sentence colon.

    The colon is split out, each side is translated on its own, and the parts are
    rejoined with a colon. Colons between two digits (e.g. 10:30) are left alone.
    """
    if not re.search(r'(?<!\d):(?!\d)', clause):
        return _translate_single_clause(clause, src_lang,
        tgt_lang, trans_tok, trans_mod,
        pre_map, surface_rules)
    pieces = [p.strip() for p in re.split(r'(?<!\d):(?!\d)', clause) if p.strip()]
    translated = [_translate_single_clause(p, src_lang,
    tgt_lang, trans_tok, trans_mod,
    pre_map, surface_rules).rstrip(".!।?") for p in pieces]
    return ": ".join(t for t in translated if t)

def translate_text(text: str, src_lang: str, tgt_lang: str,
trans_tok, trans_mod, pre_map, surface_rules) -> str:
    """Translates text line-by-line and clause-by-clause into the target language."""
    clean = str(text).strip()
    if not clean or src_lang == tgt_lang:
        return clean
    try:
        # 1. Preserve explicit lines/bullets first
        raw_lines = [line.strip() for line in clean.splitlines() if line.strip()]
        translated_lines = []
        for line in raw_lines:
            # If line has multiple full sentences, split by punctuation -- but not on
            # abbreviation/initial/decimal periods (see _split_sentences), which otherwise
            # fragment one sentence into mis-translated, danda-joined pieces.
            clauses = _split_sentences(line)
            # Kannada unpunctuated run-ons: IndicTrans2 exits early on the first
            # natural stopping point, dropping tail clauses. Re-split at coordinators
            # and clause-final verb endings so it sees one clause at a time.
            if os.getenv("STS_KN_CLAUSE_SPLIT", "1") == "1" and \
                    src_lang in ("kn", "kan_Knda"):
                _exp = []
                for _c in clauses:
                    if len(_c) >= 40 and not re.search(r'[.!?\u0964]', _c):
                        _exp.extend(_split_kn_clauses(_c))
                    else:
                        _exp.append(_c)
                clauses = _exp
            line_parts = []
            for clause in clauses:
                part = _translate_keeping_colons(clause, src_lang,
                tgt_lang, trans_tok, trans_mod,
                pre_map, surface_rules)
                if part:
                    line_parts.append(part.strip())
            if line_parts:
                # Join sentences within the same line
                punct = "। " if (tgt_lang == "hi" or "hin" in str(tgt_lang)) else ". "
                assembled_line = punct.join(p.rstrip(".!।?") for p in line_parts)
                # Respect original terminal punctuation be it !, ?, or :
                if line.endswith(":"):
                    assembled_line = assembled_line.rstrip(".!।?") + ":"
                elif line.endswith("!"):
                    assembled_line = assembled_line.rstrip(".!।?") + "!"
                elif line.endswith("?"):
                    assembled_line = assembled_line.rstrip(".!।?") + "?"
                else:
                    assembled_line = assembled_line.rstrip(".!।?") + punct
                translated_lines.append(assembled_line)
        # Join list items / paragraphs cleanly with newlines
        return "\n".join(translated_lines)
    except Exception as e:
        # Log the full traceback so the real failure point is visible, instead of
        # collapsing every error into an opaque "[TRANSLATION_ERROR: ...]" string.
        logger.exception("Translation failed for text=%r (src=%s, tgt=%s)",
                         text, src_lang, tgt_lang)
        return f"[TRANSLATION_ERROR: {type(e).__name__}: {e}]"
