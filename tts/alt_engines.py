"""Alternative offline TTS engines for the listening comparison in tts_compare.py.

Used only by tts_compare.py. None of these are wired into the live app.

  mms        Meta MMS-TTS (VITS). Kannada and Hindi. Verified loading code.
  indicf5    AI4Bharat IndicF5. UNVERIFIED call pattern: confirm on its model card on first run.
  xtts       Coqui XTTS-v2. Hindi only in this test. UNVERIFIED call pattern.

IndicF5 and XTTS need a reference recording of a speaker in each language:
    REF_AUDIO_DIR/ref_kn.wav + ref_kn.txt   (Kannada: a short clip and its exact transcript)
    REF_AUDIO_DIR/ref_hi.wav + ref_hi.txt   (Hindi, same format)
Set REF_AUDIO_DIR in the terminal, or it defaults to ./ref_audio.
"""
import asyncio
import io
import os
from pathlib import Path
from typing import Any, Dict, Optional

import soundfile as sf

from .base import BaseTTSEngine

REF_DIR = Path(os.environ.get("REF_AUDIO_DIR", "ref_audio"))


def _read_ref(lang: str):
    wav = REF_DIR / f"ref_{lang}.wav"
    txt = REF_DIR / f"ref_{lang}.txt"
    if not wav.exists() or not txt.exists():
        raise RuntimeError(f"Missing reference audio for '{lang}': expected {wav} and {txt}")
    return wav, txt.read_text(encoding="utf-8").strip()


def _to_wav_bytes(samples, rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples, rate, format="WAV")
    return buf.getvalue()


class MMSTTSEngine(BaseTTSEngine):
    MODELS = {"kn": "facebook/mms-tts-kan", "hi": "facebook/mms-tts-hin"}

    def __init__(self):
        self._models = {}

    def is_language_supported(self, lang_code: str) -> bool:
        return str(lang_code).lower() in self.MODELS

    def _load(self, lang: str):
        if lang not in self._models:
            import torch
            from transformers import AutoTokenizer, VitsModel
            model_id = self.MODELS[lang]
            model = VitsModel.from_pretrained(model_id).eval()
            tok = AutoTokenizer.from_pretrained(model_id)
            self._models[lang] = (model, tok, torch)
        return self._models[lang]

    def _generate(self, text: str, lang: str) -> bytes:
        model, tok, torch = self._load(lang)
        inputs = tok(text, return_tensors="pt")
        with torch.no_grad():
            wav = model(**inputs).waveform.squeeze().cpu().numpy()
        return _to_wav_bytes(wav, model.config.sampling_rate)

    async def synthesize(self, text: str, lang_code: str,
                         emotion_meta: Optional[Dict[str, Any]] = None,
                         voice: Optional[str] = None, gender: str = "female") -> bytes:
        lang = str(lang_code).lower()
        if not self.is_language_supported(lang):
            raise ValueError(f"MMS engine has no '{lang}' model in this test.")
        return await asyncio.to_thread(self._generate, text.strip(), lang)


class IndicF5Engine(BaseTTSEngine):
    """UNVERIFIED: follow the IndicF5 model card if the call below fails."""

    def __init__(self):
        self._model = None

    def is_language_supported(self, lang_code: str) -> bool:
        return str(lang_code).lower() in {"kn", "hi"}

    def _load(self):
        if self._model is None:
            from transformers import AutoModel
            self._model = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True)
        return self._model

    def _generate(self, text: str, lang: str) -> bytes:
        ref_wav, ref_text = _read_ref(lang)
        model = self._load()
        samples = model(text, ref_audio_path=str(ref_wav), ref_text=ref_text)
        return _to_wav_bytes(samples, 24000)

    async def synthesize(self, text: str, lang_code: str,
                         emotion_meta: Optional[Dict[str, Any]] = None,
                         voice: Optional[str] = None, gender: str = "female") -> bytes:
        lang = str(lang_code).lower()
        if not self.is_language_supported(lang):
            raise ValueError(f"IndicF5 engine does not cover '{lang}' in this test.")
        return await asyncio.to_thread(self._generate, text.strip(), lang)


class XTTSEngine(BaseTTSEngine):
    """UNVERIFIED call pattern. Hindi only in this test: XTTS-v2 has no Kannada."""

    def __init__(self):
        self._tts = None

    def is_language_supported(self, lang_code: str) -> bool:
        return str(lang_code).lower() == "hi"

    def _load(self):
        if self._tts is None:
            from TTS.api import TTS
            self._tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        return self._tts

    def _generate(self, text: str, lang: str) -> bytes:
        ref_wav, _ = _read_ref(lang)
        tts = self._load()
        samples = tts.tts(text=text, speaker_wav=str(ref_wav), language=lang)
        return _to_wav_bytes(samples, 24000)

    async def synthesize(self, text: str, lang_code: str,
                         emotion_meta: Optional[Dict[str, Any]] = None,
                         voice: Optional[str] = None, gender: str = "female") -> bytes:
        lang = str(lang_code).lower()
        if not self.is_language_supported(lang):
            raise ValueError("XTTS-v2 has no Kannada voice, so it is not tested for kn.")
        return await asyncio.to_thread(self._generate, text.strip(), lang)
