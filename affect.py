# affect.py
"""Arousal-Valence-Dominance (AVD) mapping from emotion probabilities.
Vector order everywhere is (arousal, valence, dominance)."""
import numpy as np
EMOTION_AVD_BASIS = {
    "neutral":  np.array([ 0.00,  0.00,  0.00]),
    "happy":    np.array([ 0.65,  0.75,  0.55]),
    "joy":      np.array([ 0.70,  0.80,  0.60]),
    "excited":  np.array([ 0.85,  0.65,  0.70]),
    "sad":      np.array([-0.50, -0.70, -0.55]),
    "sadness":  np.array([-0.50, -0.70, -0.55]),
    "angry":    np.array([ 0.80, -0.65,  0.75]),
    "anger":    np.array([ 0.80, -0.65,  0.75]),
    "fear":     np.array([ 0.65, -0.60, -0.60]),
    "disgust":  np.array([ 0.30, -0.60,  0.20]),
}
def fuse_text_and_audio(text_meta: dict, audio_meta: dict, w_audio: float = 0.6) -> dict:
    """Fuse a text-emotion result (roberta-large) with an audio-emotion result (emotion2vec).

    - AVD is a weighted blend (audio carries tone/arousal better, so it is weighted higher).
    - The label stays the text model's fine-grained 7-class label, EXCEPT when the text says
      'neutral' but the voice is confidently emotional (the "said flatly but sounds angry"
      case) — then the audio label wins, which is the whole point of listening to the tone.
    """
    def _avd(meta):
        a = meta.get("avd") or {}
        return np.array([float(a.get("arousal", 0.0) or 0.0),
                         float(a.get("valence", 0.0) or 0.0),
                         float(a.get("dominance", 0.0) or 0.0)], dtype=np.float32)

    w_audio = max(0.0, min(1.0, float(w_audio)))
    fused = np.clip(w_audio * _avd(audio_meta) + (1.0 - w_audio) * _avd(text_meta), -1.0, 1.0)

    text_label = str(text_meta.get("label", text_meta.get("detected_emo", "neutral")))
    audio_label = str(audio_meta.get("label", "neutral"))
    audio_conf = float(audio_meta.get("conf", 0.0) or 0.0)
    label = text_label
    if text_label.lower() == "neutral" and audio_label.lower() != "neutral" and audio_conf >= 0.6:
        label = audio_label

    return {
        "label": label,
        "detected_emo": label,
        "conf": max(float(text_meta.get("conf", 0.0) or 0.0), audio_conf),
        "prob_dist": text_meta.get("prob_dist", {}),
        "avd": {"arousal": float(fused[0]), "valence": float(fused[1]),
                "dominance": float(fused[2])},
        "emotion_source": "fused(text+audio)",
        "text_label": text_label,
        "audio_label": audio_label,
    }


def compute_avd(prob_dist: dict, dominant_emo: str) -> np.ndarray:
    """Probability-weighted AVD vector in [-1, 1]^3."""
    zero = np.zeros(3, dtype=np.float32)
    if not prob_dist:
        return EMOTION_AVD_BASIS.get(dominant_emo, zero)
    vec = np.zeros(3, dtype=np.float32)
    total_weight = 0.0
    for emo, prob in prob_dist.items():
        key = emo.lower()
        if key in EMOTION_AVD_BASIS:
            vec += float(prob) * EMOTION_AVD_BASIS[key]
            total_weight += float(prob)
    if total_weight > 0:
        vec /= total_weight
    else:
        vec = EMOTION_AVD_BASIS.get(dominant_emo, zero)
    return np.clip(vec, -1.0, 1.0)
