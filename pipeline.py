"pipeline.py"
import asyncio
import os
import re
import time
import logging
import socket
from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch
from entity_masker import mask_named_entities, unmask_and_restore_entities
from rules_engine import RulesEngine
from tts import TTSDispatcher
from affect import fuse_text_and_audio
from config import (ROUTING_CONFIG, NO_TAG_LABELS, TTS_HARD_TIMEOUT_SEC,
                    DISPLAY_TO_CODE, SUPPORTED_LANGS, AUDIO_EMOTION_WEIGHT)
logger = logging.getLogger(__name__)

# Default ON: an unpunctuated Kannada run-on (the normal output of speech ASR, which emits no
# full stops) is translated by IndicTrans2 as ONE sentence, so clauses bleed together and numbers
# mis-parse (e.g. "ಒಂದು ಸಾವಿರ ಪೇಮೆಂಟ್ ... ತಲುಪಿದೆ" came out as "one thousand payments" instead of
# "a payment of one thousand reached me"). Splitting at finite-verb boundaries first gives the NMT
# clean single sentences. Set STS_KN_CLAUSE_SPLIT=0 to disable.
_STS_KN_CLAUSE_SPLIT = os.getenv("STS_KN_CLAUSE_SPLIT", "1") != "0"

# Matches a Kannada word ending with a finite-verb suffix that marks a clause boundary. Used to
# insert ". " so IndicTrans2 sees separate sentences instead of one run-on. Only genuine
# SENTENCE-FINAL conjugations are listed (1st/2nd/3rd-person present/future/past + reported
# speech + modal), never the non-finite/subordinate forms (ಇರೋದು, ಇರೋದ್ರಿಂದ, ಮಾತಾಡ್ತಾ ...) that
# continue a clause. Longer suffixes precede their shorter sub-forms so the longest one matches.
_KN_VERB_END_RE = re.compile(
    r'(\S*(?:ವಾಗಿದೆ|ವಾಗಿಲ್ಲ|ಆಗಿದೆ|ಆಗಿಲ್ಲ|'
    r'ುತ್ತೇನೆ|ತ್ತೇನೆ|ುತ್ತೀನಿ|ತ್ತೀನಿ|ುತ್ತೇವೆ|ತ್ತೇವೆ|ುತ್ತೀರಿ|ತ್ತೀರಿ|'
    r'ುತ್ತಾನೆ|ತ್ತಾನೆ|ುತ್ತಾಳೆ|ತ್ತಾಳೆ|ುತ್ತಾರೆ|ತ್ತಾರೆ|ುತ್ತದೆ|ತ್ತದೆ|'
    r'ಅಂತಾರೆ|ಎಂದರು|ಎಂದನು|ಎಂದನಳು|ಬೇಕು|ಬೇಡ|ಲ್ಲವೇ|ಲ್ಲವ|'
    r'ಆಯಿತು|ಆಯ್ತು|ಯಿತು|ಕೊಟ್ಟರು|ಮಾಡಿದರು|ಹೋದರು|ಬಂದರು))\s+'
)


def _kn_insert_boundaries(text: str) -> str:
    """Insert sentence boundaries after Kannada finite-verb endings in unpunctuated run-ons."""
    if re.search(r'[.!?।]', text):
        return text  # already punctuated — leave as-is
    return _KN_VERB_END_RE.sub(r'\1. ', text)


# Non-standard Kannada source forms -> the standard WRITTEN form IndicTrans2 expects. Two kinds:
#   (a) colloquial/spoken contractions Whisper transcribes faithfully ("ಹುಡ್ಕಿ", "ಮಾಡ್ತಾ",
#       "ಹೋಗ್ತೀನಿ") — IndicTrans2, trained on formal written Kannada, does not recognise them and
#       TRANSLITERATES them ("hudki hudki ... took two hours") instead of translating;
#   (b) ASR spelling glitches — a stray conjunct the model mis-transcribes ("ಕದಲಲ್ಲಿಲ್ಲ" for the
#       correct "ಕದಲಲಿಲ್ಲ"), which can break the NMT's parse of the whole word (wrong meaning AND
#       wrong tense: "no one stirred" -> "no one is bothered").
# Each entry is a FAITHFUL spelling normalisation (same word, canonical form), never a meaning
# change, so it is safe. STARTER set — extend it as you meet more forms (you know the dialect and
# the common ASR slips better than the model does).
_KN_COLLOQUIAL = {
    "ಹುಡ್ಕಿ": "ಹುಡುಕಿ",         # searched
    "ಮಾಡ್ತಾ": "ಮಾಡುತ್ತಾ",       # doing
    "ಮಾಡ್ತೀನಿ": "ಮಾಡುತ್ತೇನೆ",   # I will do
    "ಮಾಡ್ತೇವೆ": "ಮಾಡುತ್ತೇವೆ",   # we will do
    "ಹೋಗ್ತಾ": "ಹೋಗುತ್ತಾ",       # going
    "ಹೋಗ್ತೀನಿ": "ಹೋಗುತ್ತೇನೆ",   # I will go
    "ಬರ್ತಾ": "ಬರುತ್ತಾ",         # coming
    "ಬರ್ತೀನಿ": "ಬರುತ್ತೇನೆ",     # I will come
    "ಮಾತಾಡ್ತಾ": "ಮಾತನಾಡುತ್ತಾ",  # speaking
    "ಆಡ್ತಾ": "ಆಡುತ್ತಾ",         # playing
    "ಕೊಡ್ತೀನಿ": "ಕೊಡುತ್ತೇನೆ",   # I will give
    "ತಗೊಂಡು": "ತೆಗೆದುಕೊಂಡು",   # having taken
    "ಆಯ್ತು": "ಆಯಿತು",           # happened / it's done
    "ಇರೋದು": "ಇರುವುದು",         # being / the fact of being
    "ಇರ್ತೀನಿ": "ಇರುತ್ತೇನೆ",     # I will be
    # ASR spelling glitches (stray conjunct) -> correct spelling:
    "ಕದಲಲ್ಲಿಲ್ಲ": "ಕದಲಲಿಲ್ಲ",   # did not stir / no one moved
}
# Whole-Kannada-word match only: a key must not sit inside a longer Kannada word (so "ಹುಡ್ಕಿಯನ್ನು"
# is left alone), guarded by Kannada-letter look-around on both sides. Longest keys first.
_KN_COLLOQUIAL_RE = re.compile(
    r'(?<![ಀ-೿])(' +
    '|'.join(sorted(map(re.escape, _KN_COLLOQUIAL), key=len, reverse=True)) +
    r')(?![ಀ-೿])'
)


def _kn_normalize_colloquial(text: str) -> str:
    """Rewrite common spoken-Kannada contractions to their standard spelling so IndicTrans2
    translates them instead of transliterating them."""
    if not text:
        return text
    return _KN_COLLOQUIAL_RE.sub(lambda m: _KN_COLLOQUIAL[m.group(1)], text)


# Negative-concord repair. Kannada forms a universal negative ("no one", "nothing") by adding the
# emphatic clitic -ೂ to a question word that is paired with a negative: ಯಾರೂ … ಇಲ್ಲ = "no one",
# ಏನೂ … ಇಲ್ಲ = "nothing". ASR routinely drops that long vowel (ಯಾರೂ -> ಯಾರು, ಏನೂ -> ಏನು), which
# flips the meaning from "no one" to the question "who", so IndicTrans2 renders "who didn't move"
# instead of "no one moved". We restore the clitic ONLY when a negative follows within the clause,
# so a genuine "ಯಾರು ಬಂದರು?" ("who came?") is never touched. The negative token is a negative-past
# verb (…ಲಿಲ್ಲ), standalone ಇಲ್ಲ, or ಅಲ್ಲ ("is not").
_KN_QWORD_EMPHATIC = {"ಯಾರು": "ಯಾರೂ", "ಏನು": "ಏನೂ"}
_KN_NEG_CONCORD_RE = re.compile(
    r'(?<![ಀ-೿])(ಯಾರು|ಏನು)(?![ಀ-೿])'
    r'((?:\s+[^\s.।!?]+){0,3}?\s+\S*?(?:ಲಿಲ್ಲ|ಇಲ್ಲ|ಅಲ್ಲ)(?![ಀ-೿]))'
)


def _kn_fix_negative_concord(text: str) -> str:
    """Lengthen a question word to its emphatic 'no one'/'nothing' form when a negative follows
    in the same clause (ASR drops the long vowel). Genuine questions are left untouched."""
    if not text:
        return text
    return _KN_NEG_CONCORD_RE.sub(
        lambda m: _KN_QWORD_EMPHATIC[m.group(1)] + m.group(2), text)


def _iso(code: str) -> str:
    """Normalize any language spelling ('English', 'hin_Deva', 'hi', ...) to 'en'/'hi'/'kn'."""
    key = str(code).strip().lower()
    return DISPLAY_TO_CODE.get(key, key[:2])
# How long a request may wait for its turn on the translator before it is refused
NMT_QUEUE_WAIT_SEC = 20.0
# A TTS retry needs at least this much of the overall time budget left, or it is skipped
TTS_RETRY_MIN_BUDGET_SEC = 2.0


class PipelineBusy(RuntimeError):
    "Raised when a request waits too long for the translator."
def _is_edge_reachable(host: str = "speech.platform.bing.com",
port: int = 443,
timeout: float = 1.0) -> bool:
    """Non-blocking pre-flight socket probe ensuring port 443 is open and responsive."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, OSError):
        return False
def _apply_sensitivity_guard(emotion_meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a copy of emotion_meta with suppress_tags set when the label is in NO_TAG_LABELS."""
    emo = dict(emotion_meta or {})
    label = str(emo.get("detected_emo", emo.get("label", ""))).lower()
    if label in NO_TAG_LABELS:
        emo["suppress_tags"] = True
        logger.info(f"Sensitivity guard: suppressing emotion tags for label '{label}'.")
    return emo
@dataclass
class PipelineResult:
    "This is the main class we are talking about over here!!"
    source_text: str
    translated_text: str
    audio_bytes: bytes
    target_language: str
    emotion_meta: Dict[str, Any]
    telemetry: Dict[str, Any]
    @property
    def ui_meta(self) -> Dict[str, Any]:
        """Normalized presentation contract decoupling internal telemetry from the UI layer."""
        tts = self.telemetry.get("tts", {})
        emo = self.emotion_meta or {}
        detected_emo = emo.get("detected_emo", emo.get("label", "neutral"))
        conf = emo.get("conf", emo.get("confidence", 0.95))
        # Report the prosody Edge was actually asked for; ElevenLabs uses no prosody here.
        if tts.get("engine_used") == "edge":
            from tts.edge_engine import EdgeTTSEngine
            rate, pitch, _ = EdgeTTSEngine()._derive_prosody(emo)
        else:
            rate, pitch = "+0%", "+0Hz"
        return {
            "status": "SUCCESS" if self.audio_bytes else "FAILED",
            "engine": tts.get("engine_used", "edge"),
            "requested_engine": tts.get("requested_engine", tts.get("engine_used", "edge")),
            "fallback": bool(tts.get("fallback_triggered")),
            "engine_error": tts.get("error"),
            "audio_format": tts.get("audio_format",
            "audio/mpeg" if tts.get("engine_used") == "edge" else "audio/wav"),
            "duration": tts.get("latency_sec", 0.0),
            "total_e2e_time": self.telemetry.get("total_e2e_latency_sec", 0.0),
            "pitch": pitch, "rate": rate, "detected_emo": detected_emo,
            "emotion_source": emo.get("emotion_source", "text"),
            "text_label": emo.get("text_label"), "audio_label": emo.get("audio_label"),
            "confidence": conf, "prob_dist": emo.get("prob_dist", {detected_emo: conf}),
            "avd": emo.get("avd", {"arousal": emo.get("arousal", 0.0),
            "valence": emo.get("valence", 0.0),
            "dominance": emo.get("dominance", 0.0)})
        }
async def _timed_step(name, fn, *args):
    "Runs fn in a worker thread"
    return await asyncio.to_thread(fn, *args)


class STSPipeline:
    "STSPipeline class"
    def __init__(self, translator_fn, emotion_fn,
    rules_path: str = "rules.json",
    default_tts_engine: str = "elevenlabs"):
        "Initialization part over here!!"
        self.translator_fn = translator_fn
        self.emotion_fn = emotion_fn
        self.rules_engine = RulesEngine(rules_path)
        self.tts_dispatcher = TTSDispatcher(primary_engine=default_tts_engine)
        self._nmt_lock = asyncio.Lock()

    async def _translate_exclusive(self, masked_text: str, src_lang: str, tgt_lang: str):
        "Runs one translation at a time. The lock is released only when the worker thread really ends."
        try:
            await asyncio.wait_for(self._nmt_lock.acquire(), timeout=NMT_QUEUE_WAIT_SEC)
        except asyncio.TimeoutError as err:
            raise PipelineBusy("The server is busy with another translation; please try again shortly.") from err
        fut = asyncio.ensure_future(
            asyncio.to_thread(self.translator_fn, masked_text, src_lang, tgt_lang))
        # Released when the thread finishes, even if the request that started it was dropped
        fut.add_done_callback(lambda _f: self._nmt_lock.release())
        result = await asyncio.shield(fut)
        return result
    async def _translate(self, masked_text: str, src_lang: str, tgt_lang: str) -> str:
        "Translate one (masked) text between two languages, honoring the NMT lock."
        if _iso(src_lang) == _iso(tgt_lang):
            return masked_text
        return await self._translate_exclusive(masked_text, src_lang, tgt_lang)

    async def _english_pivot(self, masked_text: str, src_lang: str) -> str:
        "Return the English (masked) form of the source, translating only if needed."
        if _iso(src_lang) == "en":
            return masked_text
        return await self._translate(masked_text, src_lang, "en")

    async def _synthesize(self, refined_text: str, tgt_lang: str,
                          emotion_meta: Dict[str, Any], gender: str,
                          preferred_tts_engine: Optional[str]):
        "Run TTS with Edge socket pre-flight, an overall time budget, and one fallback retry."
        selected_engine = preferred_tts_engine or self.tts_dispatcher.primary_engine
        requested_engine = selected_engine
        # English output -> Edge-TTS, which has native Indian-English voices (en-IN-Neerja /
        # en-IN-Prabhat). Indic-Parler's English drifts to a neutral/Western accent, so for a
        # reliably Indian-accented English we route it to Edge; Hindi/Kannada stay on the
        # expressive local Parler engine. Opt out with STS_ENGLISH_EDGE=0. Falls back to Parler
        # below if Edge is unreachable (offline).
        _is_en = str(tgt_lang).lower().startswith("en") or "eng" in str(tgt_lang).lower()
        if (_is_en and os.getenv("STS_ENGLISH_EDGE", "1") != "0"
                and "edge" in self.tts_dispatcher.engines):
            selected_engine = "edge"
            requested_engine = "edge"
        if selected_engine == "elevenlabs" and len(refined_text) > 300:
            logger.warning(
                f"[FREE TIER] Trimming refined_text from {len(refined_text)} to 300 chars.")
            refined_text = refined_text[:300]
        # If Edge-TTS is selected but the socket is unreachable, route immediately to local tier
        if selected_engine == "edge":
            edge_ok = await asyncio.to_thread(_is_edge_reachable)
            if not edge_ok:
                if os.getenv("ELEVENLABS_API_KEY"):
                    logger.warning("Edge-TTS socket unreachable. Rerouting to ElevenLabs.")
                    selected_engine = "elevenlabs"
                elif "parler" in self.tts_dispatcher.engines:
                    # Offline: fall back to the local expressive engine so English still gets
                    # audio (with the Indian-accent description cue) instead of failing.
                    logger.warning("Edge-TTS socket unreachable; falling back to local Parler "
                                   "for %s.", tgt_lang)
                    selected_engine = "parler"
                else:
                    logger.warning("Edge-TTS socket unreachable and no ElevenLabs key set; "
                                   "no offline engine is enabled, so synthesis will fail.")
        # Local engines (Parler) are slow and may DOWNLOAD the model on first use, so the
        # cloud-tier 25 s budget would always time them out and bounce to Edge. Give the local
        # engine a much larger budget (override with PARLER_TIMEOUT_SEC).
        if selected_engine == "parler":
            hard_timeout = float(os.getenv("PARLER_TIMEOUT_SEC", "600"))
        else:
            hard_timeout = float(TTS_HARD_TIMEOUT_SEC)
        try:
            audio_bytes, tts_telemetry = await asyncio.wait_for(
                self.tts_dispatcher.synthesize(text=refined_text, lang_code=tgt_lang,
                emotion_meta=emotion_meta, preferred_engine=selected_engine, gender=gender),
                timeout=hard_timeout)
            # Record what was asked for, so the UI can report when the dispatcher's own chain
            # already fell back (e.g. Parler not registered -> Edge).
            tts_telemetry.setdefault("requested_engine", requested_engine)
            return audio_bytes, tts_telemetry
        except Exception as err:
            reason = f"{type(err).__name__}: {err or 'timed out'}"
            logger.warning(f"Primary TTS '{requested_engine}' failed ({reason}). "
                           "Retrying via Edge so the user still gets audio.")
            # Always give Edge a fresh standard budget so a slow local failure still yields audio.
            audio_bytes, tts_telemetry = await asyncio.wait_for(
                self.tts_dispatcher.synthesize(text=refined_text, lang_code=tgt_lang,
                emotion_meta=emotion_meta, preferred_engine="edge", gender=gender),
                timeout=float(TTS_HARD_TIMEOUT_SEC))
            tts_telemetry["requested_engine"] = requested_engine
            tts_telemetry["fallback_triggered"] = True
            tts_telemetry["error"] = reason
            return audio_bytes, tts_telemetry

    async def _prepare(self, source_text: str, src_lang: str, audio_emotion=None, prosody=None):
        """Mask entities, compute the English form, and detect emotion.

        Text emotion is detected on the English form (the text model is English-only).
        When `audio_emotion` (from emotion2vec on the spoken input) is supplied, it is fused
        with the text emotion so the speaker's TONE is captured, not just their words.
        When `prosody` (measured source delivery) is supplied, it rides along in emotion_meta
        so the TTS engines can mirror the speaker's pitch/pace/loudness continuously.
        Returns (masked_text, ledger, eng_masked, emotion_meta).
        """
        # Kannada source preprocessing (speech is casual and unpunctuated):
        #   1) normalise spoken contractions to standard spelling so IndicTrans2 translates
        #      them instead of transliterating ("ಹುಡ್ಕಿ" -> "ಹುಡುಕಿ", not "hudki");
        #   2) clause-split the unpunctuated run-on so the NMT gets separate sentences rather
        #      than one huge clause it misreads. Normalise first, so any verb endings it fixes
        #      (…ತೀನಿ -> …ತ್ತೇನೆ) also become clean boundaries for the splitter.
        if _iso(src_lang) == "kn":
            source_text = _kn_normalize_colloquial(source_text)
            source_text = _kn_fix_negative_concord(source_text)
            if _STS_KN_CLAUSE_SPLIT:
                source_text = _kn_insert_boundaries(source_text)
        # Indian/Western digit-grouping commas (2,12,300) make IndicTrans2 drop a group when it
        # copies the number; strip them so it sees one clean token (212300). The spoken number
        # words are produced later in the TTS stage from this clean form.
        source_text = re.sub(r"(?<=\d),(?=\d)", "", source_text)
        masked_text, ledger = mask_named_entities(source_text)
        eng_masked = await self._english_pivot(masked_text, src_lang)
        eng_text = unmask_and_restore_entities(eng_masked, ledger, "en")
        emotion_meta = await asyncio.to_thread(self.emotion_fn, eng_text)
        if audio_emotion:
            emotion_meta = fuse_text_and_audio(emotion_meta, audio_emotion, AUDIO_EMOTION_WEIGHT)
        else:
            emotion_meta.setdefault("emotion_source", "text")
        emotion_meta = _apply_sensitivity_guard(emotion_meta)
        if prosody:
            emotion_meta["prosody"] = prosody  # continuous prosody transfer (speech input)
        return masked_text, ledger, eng_masked, emotion_meta

    async def _build_target(self, source_text, masked_text, eng_masked, ledger,
                            src_lang, tgt_lang, emotion_meta, gender,
                            preferred_tts_engine, start_time) -> PipelineResult:
        "Translate to one target language, clean it, synthesize audio, and package the result."
        src, tgt = _iso(src_lang), _iso(tgt_lang)
        if tgt == src:
            raw = masked_text
        elif tgt == "en":
            raw = eng_masked                               # already have the English form
        elif src == "en":
            raw = await self._translate(masked_text, "en", tgt)
        else:
            raw = await self._translate(eng_masked, "en", tgt)   # indic->indic via English
        unmasked = unmask_and_restore_entities(raw, ledger, tgt)
        refined = self.rules_engine.apply_conversational_surface_rules(unmasked, tgt)
        audio_bytes, tts_telemetry = await self._synthesize(
            refined, tgt, emotion_meta, gender, preferred_tts_engine)
        telemetry = {"tts": tts_telemetry,
                     "total_e2e_latency_sec": round(time.perf_counter() - start_time, 3)}
        return PipelineResult(source_text=source_text, translated_text=refined,
                              audio_bytes=audio_bytes, target_language=tgt,
                              emotion_meta=emotion_meta, telemetry=telemetry)

    async def process(self, source_text: str, gender: str,
    src_lang: str = "en", tgt_lang: str = "kn",
    preferred_tts_engine: Optional[str] = None,
    audio_emotion: Optional[Dict[str, Any]] = None,
    prosody: Optional[Dict[str, Any]] = None) -> PipelineResult:
        "Translate the source into a single target language and synthesize speech."
        start_time = time.perf_counter()
        try:
            masked_text, ledger, eng_masked, emotion_meta = await self._prepare(
                source_text, src_lang, audio_emotion, prosody)
            return await self._build_target(source_text, masked_text, eng_masked, ledger,
                src_lang, tgt_lang, emotion_meta, gender, preferred_tts_engine, start_time)
        finally:
            if torch.cuda.is_available():
                torch.cuda.ipc_collect()

    async def process_all_targets(self, source_text: str, gender: str,
    src_lang: str = "en",
    preferred_tts_engine: Optional[str] = None,
    audio_emotion: Optional[Dict[str, Any]] = None,
    prosody: Optional[Dict[str, Any]] = None) -> Dict[str, PipelineResult]:
        """Translate the source into the OTHER two languages and synthesize speech for each.

        If the input is English, outputs are Hindi and Kannada; if Hindi, outputs are
        English and Kannada; if Kannada, outputs are English and Hindi. Returns a dict
        keyed by the 2-letter target code ('en'/'hi'/'kn').
        """
        start_time = time.perf_counter()
        src = _iso(src_lang)
        targets = [lang for lang in SUPPORTED_LANGS if lang != src]
        try:
            masked_text, ledger, eng_masked, emotion_meta = await self._prepare(
                source_text, src_lang, audio_emotion, prosody)
            results: Dict[str, PipelineResult] = {}
            for tgt in targets:
                results[tgt] = await self._build_target(
                    source_text, masked_text, eng_masked, ledger, src, tgt,
                    emotion_meta, gender, preferred_tts_engine, start_time)
            return results
        finally:
            if torch.cuda.is_available():
                torch.cuda.ipc_collect()

