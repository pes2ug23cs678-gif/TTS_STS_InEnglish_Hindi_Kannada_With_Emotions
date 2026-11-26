import threading
from pathlib import Path

from config import configure_runtime_hardware
from models import (
    load_neural_pipeline,
    translate_route,
    detect_text_emotion,
)
from pipeline import STSPipeline
from affect import compute_avd
_runtime = None
_runtime_lock = threading.Lock()

_audio_emotion_engine = None
_audio_emotion_tried = False
_audio_emotion_lock = threading.Lock()


def get_audio_emotion_engine():
    """Lazy singleton speech-emotion engine (emotion2vec). Returns None if unavailable."""
    global _audio_emotion_engine, _audio_emotion_tried
    if _audio_emotion_tried:
        return _audio_emotion_engine
    with _audio_emotion_lock:
        if _audio_emotion_tried:
            return _audio_emotion_engine
        _audio_emotion_tried = True
        try:
            from config import ENABLE_AUDIO_EMOTION, AUDIO_EMOTION_MODEL_ID, DEVICE
            if not ENABLE_AUDIO_EMOTION:
                return None
            from emotion_audio import AudioEmotionEngine
            if not AudioEmotionEngine.available():
                return None
            _audio_emotion_engine = AudioEmotionEngine(
                model_id=AUDIO_EMOTION_MODEL_ID, device=DEVICE)
        except Exception:
            _audio_emotion_engine = None
    return _audio_emotion_engine


def audio_emotion_from_bytes(audio_bytes):
    """Best-effort vocal-tone emotion dict (with AVD) from raw audio, or None."""
    engine = get_audio_emotion_engine()
    if engine is None or not audio_bytes:
        return None
    try:
        raw = engine.predict(audio_bytes)
        a, v, d = (float(x) for x in compute_avd(raw["prob_dist"], raw["label"]))
        raw["avd"] = {"arousal": a, "valence": v, "dominance": d}
        return raw
    except Exception:
        return None


def get_ai_runtime():
    global _runtime

    if _runtime is not None:
        return _runtime

    with _runtime_lock:
        if _runtime is not None:
            return _runtime

        configure_runtime_hardware()

        engines = load_neural_pipeline()  # lazy: no weights load until first use

        def translator_adapter(
            text: str,
            src_lang: str,
            tgt_lang: str,
        ) -> str:
            # Routes across all 6 directions (en<->hi, en<->kn, hi<->kn via English pivot).
            result = translate_route(text, src_lang, tgt_lang, engines)

            if result.startswith("[TRANSLATION_ERROR:"):
                raise RuntimeError(result)

            return result

        def emotion_adapter(text: str) -> dict:
            emo_tok, emo_mod = engines["emotion"]  # loaded on first call, not at startup
            label, confidence, probabilities = detect_text_emotion(
                text,
                emo_tok,
                emo_mod,
            )
            a, v, d = (float(x) for x in compute_avd(probabilities, label))
            return {
                "label": label,
                "detected_emo": label,
                "conf": confidence,
                "prob_dist": probabilities,
                "avd": {"arousal": a, "valence": v, "dominance": d},
            }

        rules_path = Path(__file__).resolve().parent / "rules.json"

        pipeline_instance = STSPipeline(
            translator_fn=translator_adapter,
            emotion_fn=emotion_adapter,
            rules_path=str(rules_path),
            default_tts_engine="edge",
        )

        # Return the lazy engine registry; callers fetch engines["asr"] only when needed.
        _runtime = (pipeline_instance, engines)
        return _runtime
