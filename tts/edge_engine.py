"edge_engine.py"
import asyncio
import os
import re
import logging
from typing import Optional, Dict, Any
from .base import BaseTTSEngine
from .loudness import normalize_loudness
logger = logging.getLogger(__name__)

# edge-tts is imported defensively (like the ElevenLabs engine's optional import): a machine
# that hasn't installed it must not crash the whole app at startup. It is only needed when
# EdgeTTSEngine actually synthesizes; until then the class imports fine. Install it with:
#   pip install edge-tts   (it is pinned in requirements.txt as edge-tts==7.3.1)
try:
    import edge_tts
    HAS_EDGE_TTS = True
except ImportError:
    edge_tts = None
    HAS_EDGE_TTS = False
    logger.warning("The 'edge-tts' package is not installed; EdgeTTSEngine is unavailable "
                   "and the dispatcher will fall back to another engine. "
                   "Install it with: pip install edge-tts")

# Global expressiveness dial. 1.0 = the widened defaults below; raise toward ~1.4 for
# more dramatic delivery, lower toward 0.6 for calmer. Env override: STS_EXPRESSIVENESS.
try:
    EXPRESSIVENESS = max(0.0, min(2.0, float(os.getenv("STS_EXPRESSIVENESS", "1.0"))))
except (TypeError, ValueError):
    EXPRESSIVENESS = 1.0
class EdgeTTSEngine(BaseTTSEngine):
    "THis is the main class we would be working with over here"
    DEFAULT_VOICE_MAP = {
        "kn": "kn-IN-SapnaNeural",
        "kn_male": "kn-IN-GaganNeural",
        "hi": "hi-IN-SwaraNeural",
        "hi_male": "hi-IN-MadhurNeural",
        # Indian-English neural voices (was en-US-* -> that read as a Western accent).
        "en": "en-IN-NeerjaNeural",
        "en_male": "en-IN-PrabhatNeural",
    }
    def _normalize_lang(self, lang_code: str) -> str:
        "this is the _normalize_lang code"
        code = str(lang_code).lower().strip()
        if "kan" in code or code == "kn":
            return "kn"
        if "hin" in code or code == "hi":
            return "hi"
        if "eng" in code or code == "en":
            return "en"
        return code
    def is_language_supported(self, lang_code: str) -> bool:
        "let us see if the given language is supported over here or not"
        normalized = self._normalize_lang(lang_code)
        return normalized in {"kn", "hi", "en", "te", "ta", "ml", "mr", "bn", "gu"}
    def _derive_prosody(self, emotion_meta: Optional[Dict[str, Any]]) -> tuple[str, str, str]:
        if not emotion_meta or emotion_meta.get("suppress_tags"):
            return "+0%", "+0Hz", "+0%"
        # Continuous prosody transfer: if the source speaker's measured prosody is present
        # (speech-to-speech), mirror their actual delivery instead of the discrete emotion map.
        prosody = emotion_meta.get("prosody")
        if prosody:
            try:
                from prosody import to_edge_prosody
                mapped = to_edge_prosody(prosody)
                if mapped:
                    return mapped
            except Exception:  # noqa: BLE001 - fall back to emotion-derived prosody
                pass
        avd = emotion_meta.get("avd") if isinstance(emotion_meta.get("avd"), dict) else {}
        try:
            arousal = float(avd.get("arousal", emotion_meta.get("arousal", 0.0)) or 0.0)
            valence = float(avd.get("valence", 0.0) or 0.0)
        except (TypeError, ValueError):
            return "+0%", "+0Hz", "+0%"
        arousal = max(-1.0, min(1.0, arousal))
        valence = max(-1.0, min(1.0, valence))
        k = EXPRESSIVENESS
        # Carry audible affect instead of a flat monotone. The old caps (rate ±12%,
        # pitch ±40Hz) were so tight the voice sounded emotionless; these widened ranges
        # let anger/excitement and sadness actually read while staying intelligible.
        # Pitch and volume carry most of the emotion (a human cue), rate a little less so
        # the words stay followable.
        rate = int(round(arousal * 42 * k))
        rate = max(-18, min(24, rate))
        pitch = int(round((arousal * 36 + valence * 30) * k))
        pitch = max(-60, min(60, pitch))
        volume = int(round((arousal * 30 + max(0.0, valence) * 14) * k))
        volume = max(-25, min(42, volume))
        # Tiny dead-zone on rate only, so genuinely neutral lines are not sped up/slowed;
        # pitch and volume stay responsive so mild emotion still colours the delivery.
        if abs(rate) < 2:
            rate = 0
        return f"{rate:+d}%", f"{pitch:+d}Hz", f"{volume:+d}%"
    # A single utterance that stalls at the WebSocket stage is cut off after this many seconds
    CLAUSE_TIMEOUT_SEC = 15.0
    # Text at or below this length is spoken in one continuous pass (more natural intonation);
    # longer text is split into clauses for reliability.
    SINGLE_PASS_MAX_CHARS = 320

    async def _stream_clause(self, text: str, voice: str, rate: str, pitch: str, volume: str) -> list:
        "Streams one sentence from Edge, giving up after CLAUSE_TIMEOUT_SEC so a stall cannot hang the request"
        async def _collect() -> list:
            if not HAS_EDGE_TTS:
                raise RuntimeError("edge-tts is not installed (pip install edge-tts).")
            communicator = edge_tts.Communicate(text=text, voice=voice,
                rate=rate, pitch=pitch, volume=volume)
            parts = []
            async for chunk in communicator.stream():
                if chunk["type"] == "audio":
                    parts.append(chunk["data"])
            return parts
        return await asyncio.wait_for(_collect(), timeout=self.CLAUSE_TIMEOUT_SEC)

    async def synthesize(
        self,
        text: str,
        lang_code: str,
        emotion_meta: Optional[Dict[str, Any]] = None,
        voice: Optional[str] = None,
        gender: str = "female"
    ) -> bytes:
        "Returns in terms of bytes"
        clean_text = text.strip()
        if not clean_text or not re.search(r'\w|[\u0900-\u0D7F]', clean_text):
            raise ValueError("Input text contains no pronounceable linguistic content.")
        normalized_lang = self._normalize_lang(lang_code)
        if not self.is_language_supported(normalized_lang):
            raise ValueError(f"Language '{lang_code}' unsupported by EdgeTTSEngine.")
        # Resolve voice using language and gender keys
        voice_key = f"{normalized_lang}_male" if str(gender).lower() == "male" else normalized_lang
        default_voice = self.DEFAULT_VOICE_MAP.get(voice_key,
        self.DEFAULT_VOICE_MAP.get(normalized_lang,
        "kn-IN-SapnaNeural"))
        selected_voice = voice or default_voice
        rate_str, pitch_str, volume_str = self._derive_prosody(emotion_meta)

        async def _render(segments: list) -> list:
            """Synthesize each segment, retrying once with neutral prosody on failure."""
            parts = []
            for seg in segments:
                try:
                    chunk_bytes = await self._stream_clause(
                        seg, selected_voice, rate_str, pitch_str, volume_str)
                    if not chunk_bytes:
                        raise RuntimeError("Empty stream from dynamic prosody")
                except Exception as e:
                    logger.warning(f"Prosody failed for '{seg[:30]}' "
                                   f"({type(e).__name__}: {e}). Retrying with neutral prosody..")
                    chunk_bytes = await self._stream_clause(
                        seg, selected_voice, "+0%", "+0Hz", "+0%")
                if chunk_bytes:
                    parts.append(b"".join(chunk_bytes))
            return parts

        # Prefer ONE continuous pass: the voice keeps a natural, connected intonation across
        # the whole utterance instead of resetting (and sounding choppy/robotic) every clause.
        # Only long text is split into clauses, which also guards against WebSocket payload drops.
        accumulated_audio = []
        if len(clean_text) <= self.SINGLE_PASS_MAX_CHARS:
            try:
                accumulated_audio = await _render([clean_text])
            except Exception as e:
                logger.warning(f"Single-pass synthesis failed ({type(e).__name__}: {e}); "
                               "falling back to clause-by-clause.")
                accumulated_audio = []
        if not accumulated_audio:
            clauses = [c.strip() for c in re.split(r'(?<=[.!।?])\s+', clean_text) if c.strip()]
            if not clauses:
                clauses = [clean_text]
            accumulated_audio = await _render(clauses)

        if not accumulated_audio:
            raise RuntimeError(f"Edge-TTS returned zero audio frames for voice {selected_voice}.")
        # Raise the level to a consistent, louder target (falls back to the original on error)
        return normalize_loudness(b"".join(accumulated_audio))