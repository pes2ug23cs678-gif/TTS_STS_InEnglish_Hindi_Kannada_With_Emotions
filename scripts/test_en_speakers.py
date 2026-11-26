"""
Compare Thoma vs Suresh for English male voice expressivity.
Run from ~/Python312_Project:
    python test_en_speakers.py
"""
import sys, os, numpy as np, soundfile as sf, torch
sys.path.insert(0, os.path.dirname(__file__))

from tts.parler_engine import ParlerTTSEngine

TEST_TEXT = "How many times have I told you to do this work properly! This is completely unacceptable!"
DESCRIPTION_TEMPLATE = (
    "{speaker}'s voice is very expressive and animated, loud, sharp and highly agitated, "
    "with emphatic stress and clear frustration, delivered with a rapid, urgent pace with elevated pitch."
)

SPEAKERS = ["Thoma", "Suresh"]
OUT_DIR = os.path.expanduser("~/speaker_test_en")
os.makedirs(OUT_DIR, exist_ok=True)

engine = ParlerTTSEngine()

results = []
for spk in SPEAKERS:
    desc = DESCRIPTION_TEMPLATE.format(speaker=spk)
    print(f"\n--- Generating: {spk} ---")
    print(f"Description: {desc}")
    audio = engine._generate(TEST_TEXT, desc, lang_code="en", gender="male")
    path = os.path.join(OUT_DIR, f"en_male_{spk.lower()}.wav")
    sf.write(path, audio, 16000)
    # F0 via numpy zero-crossing proxy (rough but fast)
    # Use scipy if available, else skip F0 and just show RMS
    rms = float(np.sqrt(np.mean(audio**2)))
    dur = len(audio) / 16000
    try:
        import parselmouth
        snd = parselmouth.Sound(path)
        pitch = snd.to_pitch()
        f0_vals = pitch.selected_array['frequency']
        f0_vals = f0_vals[f0_vals > 0]
        med_f0 = float(np.median(f0_vals)) if len(f0_vals) else 0
        std_f0 = float(np.std(f0_vals)) if len(f0_vals) else 0
    except ImportError:
        # fallback: librosa
        try:
            import librosa
            f0, voiced, _ = librosa.pyin(audio.astype(np.float32), fmin=50, fmax=500, sr=16000)
            f0_vals = f0[voiced.astype(bool)] if f0 is not None else np.array([])
            med_f0 = float(np.median(f0_vals)) if len(f0_vals) else 0
            std_f0 = float(np.std(f0_vals)) if len(f0_vals) else 0
        except Exception:
            med_f0, std_f0 = 0, 0
    results.append((spk, dur, med_f0, std_f0, rms, path))
    print(f"  dur={dur:.2f}s  F0={med_f0:.0f}Hz (std={std_f0:.0f})  rms={rms:.4f}  → {path}")
print("\n=== SUMMARY ===")
print(f"{'Speaker':<10} {'Dur':>7} {'Med F0':>8} {'F0 std':>8} {'RMS':>8}")
for spk, dur, f0, std, rms, _ in results:
    print(f"{spk:<10} {dur:>6.2f}s {f0:>7.0f}Hz {std:>7.0f}Hz {rms:>8.4f}")
print("\nAudio saved to:", OUT_DIR)
