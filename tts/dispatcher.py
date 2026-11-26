"dispatcher.py"
import os
import time
import logging
from typing import Optional, Dict, Any, List
from .base import BaseTTSEngine
from .edge_engine import EdgeTTSEngine
from .elevenlabs_engine import ElevenLabsTTSEngine

logger = logging.getLogger(__name__)

# Indic-Parler-TTS is a local, expressive engine (English + Indic, 44.1 kHz WAV output).
# It is enabled by default; set ENABLE_PARLER=0 to skip it (e.g. on machines where the
# `parler_tts` package is not installed, or that are too small to load it).
ENABLE_PARLER = os.getenv("ENABLE_PARLER", "1") != "0"

# Which engines to try, in order, after the chosen one fails.
FALLBACK_ORDER = {"elevenlabs": ["edge", "parler"], "edge": ["parler"],
                  "parler": ["edge", "elevenlabs"]}
MPEG_ENGINES = {"edge", "elevenlabs"}  # both return MP3 bytes; parler returns WAV


class TTSDispatcher:
    """Chooses a TTS engine, applies an ordered fallback chain, and records telemetry."""

    def __init__(self, primary_engine: str = "edge", sla_budget_sec: float = 0.75):
        self.sla_budget_sec = sla_budget_sec
        self.engines: Dict[str, BaseTTSEngine] = {
            "edge": EdgeTTSEngine(),
            "elevenlabs": ElevenLabsTTSEngine(),
        }
        if ENABLE_PARLER:
            # Loaded defensively: a missing `parler_tts` package (or any import/init error)
            # must not break the whole dispatcher — it just means Parler is unavailable and
            # the engine falls back to Edge/ElevenLabs.
            try:
                from .parler_engine import ParlerTTSEngine  # imported only when enabled
                self.engines["parler"] = ParlerTTSEngine()
                logger.info("Indic-Parler-TTS engine registered.")
            except Exception as err:  # noqa: BLE001
                logger.warning("Parler-TTS unavailable (%s: %s); continuing without it. "
                               "Install it with requirements_parler.txt to enable.",
                               type(err).__name__, err)
        self.primary_engine = primary_engine

    def resolve_engine(self, preferred: Optional[str], emotion_meta: Optional[Dict[str, Any]] = None) -> str:
        """Maps a requested name ('auto', 'fallback', None, unknown) to an engine that exists."""
        key = preferred or self.primary_engine
        if key == "auto":
            avd = (emotion_meta or {}).get("avd") or {}
            intensity = max(abs(float(avd.get("arousal", 0.0) or 0.0)),
                            abs(float(avd.get("valence", 0.0) or 0.0)))
            el = self.engines.get("elevenlabs")
            key = "elevenlabs" if (intensity >= 0.4 and getattr(el, "client", None)) else "edge"
        return key if key in self.engines else "edge"

    def _chain(self, target: str) -> List[str]:
        later = [k for k in FALLBACK_ORDER.get(target, ["edge"]) if k in self.engines and k != target]
        return [target] + later

    async def synthesize(self, text: str, lang_code: str, gender: str,
                         emotion_meta: Optional[Dict[str, Any]] = None,
                         preferred_engine: Optional[str] = None, **kwargs) -> tuple[bytes, dict]:
        """Synthesizes audio, trying each engine in the chain until one succeeds."""
        target = self.resolve_engine(preferred_engine, emotion_meta)
        chain = self._chain(target)
        start = time.perf_counter()
        errors: List[str] = []

        for key in chain:
            try:
                audio = await self.engines[key].synthesize(
                    text=text, lang_code=lang_code, gender=gender, emotion_meta=emotion_meta, **kwargs)
            except Exception as e:  # noqa: BLE001 - any engine failure should move to the next engine
                logger.warning(f"TTS Engine '{key}' failed: {e}")
                errors.append(f"{key}: {e}")
                continue
            telemetry = {
                "engine_used": key,
                "latency_sec": round(time.perf_counter() - start, 3),
                "fallback_triggered": key != target,
                "audio_format": "audio/mpeg" if key in MPEG_ENGINES else "audio/wav",
            }
            if errors:
                telemetry["error"] = "; ".join(errors)
            return audio, telemetry

        raise RuntimeError("All TTS engines failed: " + " | ".join(errors))
