"""Continuous prosody transfer.

Instead of mapping a discrete emotion label to a fixed prosody bucket, this measures how the
SOURCE speaker actually delivered the input (pitch height, pitch movement/expressiveness,
loudness and speaking rate) and maps those continuously onto the target TTS controls. The
synthesized speech then mirrors the real delivery, not just an emotion category.

Only meaningful on the speech-to-speech path (text input has no source audio).
"""
import logging
import math
import os

logger = logging.getLogger(__name__)

# Rough references for "unmarked" conversational speech, used to express the source
# relative to a neutral baseline.
_NEUTRAL_RATE_SYL = 4.2       # syllables per second
_NEUTRAL_ENERGY_DB = -26.0    # mean RMS level (dBFS)

# Shared expressiveness dial (same env var as the Edge engine). Mirroring the source
# speaker's delivery more strongly is what stops the output sounding flat on the
# speech-to-speech path.
try:
    _EXPRESSIVENESS = max(0.0, min(2.0, float(os.getenv("STS_EXPRESSIVENESS", "1.0"))))
except (TypeError, ValueError):
    _EXPRESSIVENESS = 1.0


def extract_prosody(audio):
    """Measure prosody features from audio (bytes / BytesIO / np.ndarray). Returns a dict or None.

    Keys: median_f0_hz, pitch_var_semitones, energy_db, rate_syl_per_sec, voiced_ratio, duration_sec.
    Degrades gracefully (returns None) if librosa is unavailable or the clip is too short.
    """
    try:
        import numpy as np
        import librosa
        from emotion_audio import _prepare_16k_mono

        sr = 16000
        y = _prepare_16k_mono(audio)
        y, _ = librosa.effects.trim(y, top_db=30)
        dur = float(len(y)) / sr
        if dur < 0.2:
            return None

        # Pitch (F0) via probabilistic YIN on voiced frames.
        f0, _voiced_flag, _voiced_prob = librosa.pyin(y, fmin=65, fmax=400, sr=sr)
        if f0 is not None:
            finite = np.isfinite(f0)
            f0v = f0[finite]
            voiced_ratio = float(np.mean(finite)) if f0.size else 0.0
        else:
            f0v, voiced_ratio = np.array([]), 0.0
        if f0v.size >= 5:
            median_f0 = float(np.median(f0v))
            semis = 12.0 * np.log2(np.clip(f0v, 1e-6, None) / max(median_f0, 1e-6))
            pitch_var = float(np.std(semis))               # semitone spread = expressiveness
        else:
            median_f0, pitch_var = 0.0, 0.0

        # Loudness (mean RMS in dBFS).
        rms = librosa.feature.rms(y=y)[0]
        energy_db = float(20.0 * np.log10(float(np.mean(rms)) + 1e-6))

        # Speaking rate proxy: onset-strength peaks per second (~syllable nuclei).
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            peaks = librosa.util.peak_pick(
                onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.15, wait=4)
            rate_syl = float(len(peaks)) / dur if dur > 0 else 0.0
        except Exception:
            rate_syl = 0.0

        return {
            "median_f0_hz": median_f0,
            "pitch_var_semitones": pitch_var,
            "energy_db": energy_db,
            "rate_syl_per_sec": rate_syl,
            "voiced_ratio": voiced_ratio,
            "duration_sec": dur,
        }
    except Exception as err:  # noqa: BLE001 - any failure just disables prosody transfer
        logger.info("Prosody extraction unavailable (%s: %s).", type(err).__name__, err)
        return None


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def to_edge_prosody(prosody):
    """Map measured source prosody -> Edge-TTS continuous deltas (rate%, pitchHz, volume%).

    Rate and loudness transfer directly (fast/loud source -> fast/loud output). Pitch combines
    how much the source's pitch moves (expressiveness) with its relative loudness, since a single
    Edge pitch delta cannot set a range. Returns (rate_str, pitch_str, volume_str) or None.
    """
    if not prosody:
        return None
    pitch_var = float(prosody.get("pitch_var_semitones", 0.0) or 0.0)
    rate_syl = float(prosody.get("rate_syl_per_sec", 0.0) or 0.0)
    energy_db = float(prosody.get("energy_db", _NEUTRAL_ENERGY_DB) or _NEUTRAL_ENERGY_DB)

    k = _EXPRESSIVENESS
    rate = _clamp(round((rate_syl / _NEUTRAL_RATE_SYL - 1.0) * 55 * k), -18, 24) if rate_syl > 0 else 0
    pitch = _clamp(round((pitch_var * 8.0 + (energy_db - _NEUTRAL_ENERGY_DB) * 1.0) * k), -60, 60)
    vol = _clamp(round((energy_db - _NEUTRAL_ENERGY_DB) * 3.0 * k), -25, 42)
    return f"{int(rate):+d}%", f"{int(pitch):+d}Hz", f"{int(vol):+d}%"


def to_parler_phrase(prosody):
    """Describe measured prosody in words for Parler's text-conditioned delivery, or None."""
    if not prosody:
        return None
    rate = float(prosody.get("rate_syl_per_sec", 0.0) or 0.0)
    var = float(prosody.get("pitch_var_semitones", 0.0) or 0.0)
    if rate <= 0:
        pace = "a moderate pace"
    elif rate > _NEUTRAL_RATE_SYL * 1.15:
        pace = "a fast pace"
    elif rate < _NEUTRAL_RATE_SYL * 0.85:
        pace = "a slow pace"
    else:
        pace = "a moderate pace"
    pitch = ("a highly expressive, varied pitch" if var > 2.5
             else "a fairly flat, steady pitch" if var < 1.0
             else "a natural pitch")
    return f"{pace} and {pitch}"


def summarize(prosody) -> str:
    """Short human-readable summary for the UI."""
    if not prosody:
        return ""
    return (f"pitch≈{prosody.get('median_f0_hz', 0):.0f}Hz, "
            f"variation≈{prosody.get('pitch_var_semitones', 0):.1f}st, "
            f"rate≈{prosody.get('rate_syl_per_sec', 0):.1f}/s, "
            f"loudness≈{prosody.get('energy_db', 0):.0f}dB")
