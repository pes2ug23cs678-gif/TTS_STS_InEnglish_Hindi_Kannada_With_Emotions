"""Optional speech-emotion recognition via FunASR emotion2vec_plus_large.

Reads emotion from the speaker's *tone* (acoustics/prosody), which a text classifier
cannot recover. Used only on the speech-to-speech path and fused with the text emotion
model. Degrades gracefully: if `funasr` or the model is unavailable, callers skip it and
fall back to text-only emotion.
"""
import io
import logging
import os
import tempfile
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

AUDIO_EMOTION_MODEL_ID = os.getenv("AUDIO_EMOTION_MODEL_ID", "iic/emotion2vec_plus_large")

# emotion2vec class names -> the canonical labels used by affect.py / the prosody map.
# "other"/"unknown" carry no affect and are dropped.
_E2V_TO_CANONICAL = {
    "angry": "anger", "happy": "joy", "sad": "sadness", "neutral": "neutral",
    "fearful": "fear", "fear": "fear", "disgusted": "disgust", "disgust": "disgust",
    "surprised": "surprise", "surprise": "surprise",
}


def _prepare_16k_mono(audio: Union[bytes, io.BytesIO, np.ndarray]) -> np.ndarray:
    """Return a 16 kHz mono float32 array from bytes / BytesIO / ndarray."""
    import soundfile as sf
    if isinstance(audio, np.ndarray):
        arr, sr = audio, 16000
    else:
        buf = io.BytesIO(audio) if isinstance(audio, (bytes, bytearray)) else audio
        arr, sr = sf.read(buf)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim > 1:                       # stereo -> mono
        arr = arr.mean(axis=1)
    if sr != 16000:
        import librosa
        arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
    return arr.astype(np.float32)


class AudioEmotionEngine:
    """Lazy wrapper around FunASR emotion2vec. Call `available()` before relying on it."""

    def __init__(self, model_id: str = AUDIO_EMOTION_MODEL_ID, device: Optional[str] = None):
        self.model_id = model_id
        self.device = device
        self._model = None

    @staticmethod
    def available() -> bool:
        """True if the FunASR runtime can be imported (does not download the model)."""
        try:
            import funasr  # noqa: F401
            return True
        except Exception as err:  # noqa: BLE001
            logger.info("Audio emotion unavailable (funasr import failed: %s).", err)
            return False

    def _lazy_load(self):
        if self._model is not None:
            return
        from funasr import AutoModel
        kwargs = {"model": self.model_id, "disable_update": True}
        if self.device:
            kwargs["device"] = self.device
        self._model = AutoModel(**kwargs)

    def warmup(self):
        """Load the model once (downloads on first call)."""
        self._lazy_load()

    def predict(self, audio: Union[bytes, io.BytesIO, np.ndarray]) -> dict:
        """Return {'label','conf','prob_dist'} from the speaker's tone, or raise on failure."""
        self._lazy_load()
        arr = _prepare_16k_mono(audio)
        # FunASR's emotion2vec reads most reliably from a file path, so stage a temp WAV.
        import soundfile as sf
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                sf.write(tmp_path, arr, 16000, format="WAV")
            res = self._model.generate(tmp_path, granularity="utterance",
                                       extract_embedding=False)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        if not res:
            raise RuntimeError("emotion2vec returned no result.")
        item = res[0]
        labels = item.get("labels", []) or []
        scores = item.get("scores", []) or []
        dist: dict = {}
        for lab, sc in zip(labels, scores):
            name = str(lab).split("/")[-1].strip().lower()   # handles "生气/angry" style labels
            canon = _E2V_TO_CANONICAL.get(name)
            if canon:
                dist[canon] = dist.get(canon, 0.0) + float(sc)
        total = sum(dist.values())
        if total <= 0:
            # Only "other"/"unknown" fired — treat as neutral rather than failing.
            return {"label": "neutral", "conf": 0.0, "prob_dist": {"neutral": 1.0}}
        dist = {k: v / total for k, v in dist.items()}
        label = max(dist, key=dist.get)
        return {"label": label, "conf": float(dist[label]), "prob_dist": dist}
