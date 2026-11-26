"elevenlabs_engine.py"
import os
import asyncio
import logging
from typing import Optional, Dict, Any
from .base import BaseTTSEngine
logger = logging.getLogger(__name__)
try:
    from elevenlabs.client import ElevenLabs
    HAS_ELEVENLABS = True
except ImportError:
    HAS_ELEVENLABS = False
try:
    from elevenlabs import VoiceSettings
except ImportError:
    VoiceSettings = None
# Emotion label -> v3 audio tag. Only applied on eleven_v3.
EMOTION_TAGS = {
    "joy": "[happy]", "happy": "[happy]", "excited": "[excited]",
    "anger": "[angry]", "angry": "[angry]",
    "sadness": "[sad]", "sad": "[sad]",
    "fear": "[nervous]", "surprise": "[surprised]",
}
class ElevenLabsTTSEngine(BaseTTSEngine):
    """ElevenLabs TTS with emotion-aware audio tags and stability."""
    def __init__(self):
        super().__init__()
        if not HAS_ELEVENLABS:
            logger.error("The 'elevenlabs' package is not installed.")
        api_key = os.getenv("ELEVENLABS_API_KEY")
        self.client = ElevenLabs(api_key=api_key) if (HAS_ELEVENLABS and api_key) else None
        self.default_voices = {
            "female": "WcoxGT0TxNJ1EzYYz2PU",
            "male": "CtCOpWj1ERLtLhf6BPTx",
        }
    def is_language_supported(self, lang_code: str) -> bool:
        "Returns true all the time"
        return True
    @staticmethod
    def _read_affect(emotion_meta: Optional[Dict[str, Any]]):
        "Way the reading is changed as per the emotions"
        emo = emotion_meta or {}
        label = str(emo.get("detected_emo", emo.get("label", "neutral"))).lower()
        avd = emo.get("avd") if isinstance(emo.get("avd"), dict) else {}
        arousal = float(avd.get("arousal", 0.0) or 0.0)
        valence = float(avd.get("valence", 0.0) or 0.0)
        return label, arousal, valence
    def _pick_tag(self, label: str, arousal: float, valence: float) -> str:
        "For the emotions"
        if label in ("joy", "happy") and arousal > 0.5:
            return "[excited]"
        tag = EMOTION_TAGS.get(label)
        if tag:
            return tag
        # Fallback from AVD when the label has no mapping
        if valence > 0.3:
            return "[happy]"
        if valence < -0.3 and arousal > 0.4:
            return "[angry]"
        if valence < -0.3:
            return "[sad]"
        return ""
    @staticmethod
    def _pick_stability(arousal: float, valence: float) -> float:
        "eleven_v3 stability: 0.0 Creative, 0.5 Natural, 1.0 Robust"
        return 0.0 if max(abs(arousal), abs(valence)) >= 0.55 else 0.5
    def _convert(self, text, voice_id, model_id, voice_settings) -> bytes:
        "this is the _convert part"
        kwargs = {"text":text,
        "voice_id":voice_id, "model_id":model_id,
        "output_format":"mp3_44100_128"}
        if voice_settings is not None:
            kwargs["voice_settings"] = voice_settings
        stream = self.client.text_to_speech.convert(**kwargs)
        audio = b"".join(chunk for chunk in stream if isinstance(chunk, bytes))
        if not audio:
            raise ValueError("ElevenLabs returned empty stream.")
        return audio
    async def synthesize(self, text: str,
    lang_code: str,
    gender: str = "female",
    emotion_meta: Optional[Dict[str, Any]] = None,
    **kwargs) -> bytes:
        "this is the synthesis into the elevenlabs api for that matter"
        if not self.client:
            raise RuntimeError("ElevenLabs client is not initialized or API key is missing.")
        voice_id = kwargs.get("voice_id",
        self.default_voices.get(gender.lower(),
        self.default_voices["female"]))
        model_id = kwargs.get("model_id", "eleven_v3")
        label, arousal, valence = self._read_affect(emotion_meta)
        voice_settings, spoken_text = None, text
        if model_id == "eleven_v3":
            suppress = bool((emotion_meta or {}).get("suppress_tags"))
            tag = "" if suppress else self._pick_tag(label, arousal, valence)
            if tag:
                spoken_text = f"{tag} {text}"
            if VoiceSettings is not None:
                voice_settings = VoiceSettings(stability=self._pick_stability(arousal, valence))
        elif VoiceSettings is not None:
            # Older models: no audio tags (they would be read aloud); use settings only
            intensity = min(max(abs(arousal), abs(valence)), 1.0)
            voice_settings = VoiceSettings(
                stability=round(0.55 - 0.30 * intensity, 2), similarity_boost=0.75,
                style=round(0.10 + 0.50 * intensity, 2), use_speaker_boost=True,
            )
        logger.info(f"ElevenLabs [{lang_code}] model={model_id} voice={voice_id} "
                    f"emotion={label} A={arousal:+.2f} V={valence:+.2f} "
                    f"tag={spoken_text[:12] if spoken_text != text else 'none'}")
        try:
            return await asyncio.to_thread(self._convert,
            spoken_text,
            voice_id,
            model_id, voice_settings)
        except Exception as e:
            logger.error(f"ElevenLabs synthesis error for lang '{lang_code}': {e}")
            raise
