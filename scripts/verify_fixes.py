"""End-to-end sanity check for the fixes, run inside your installed environment:

    source clean_env/bin/activate      # (Windows: clean_env\\Scripts\\activate)
    python verify_fixes.py

It prints the readiness report, then (if the models are available) translates a few
sentences so you can SEE that the output is a real translation and not a transliteration,
and shows the expressive prosody the voice will use for different emotions. Nothing is
saved; no microphone needed.
"""
import sys


def main():
    print("=" * 70)
    print("1) READINESS REPORT")
    print("=" * 70)
    try:
        from preflight import report
        report(prefix="   ")
    except Exception as exc:  # noqa: BLE001
        print("   preflight failed:", exc)

    print("\n" + "=" * 70)
    print("2) TRANSLATION (should be meaning, NOT transliteration)")
    print("=" * 70)
    # Kannada: "I am very happy today." / "Where is the station?"
    samples = [
        ("kn", "en", "ನಾನು ಇಂದು ತುಂಬಾ ಸಂತೋಷವಾಗಿದ್ದೇನೆ."),
        ("kn", "hi", "ನಿಲ್ದಾಣ ಎಲ್ಲಿದೆ?"),
        ("en", "hi", "Please close the door."),
        ("en", "kn", "The weather is beautiful today."),
    ]
    try:
        from models import load_neural_pipeline, translate_route
        engines = load_neural_pipeline()
        for src, tgt, text in samples:
            try:
                out = translate_route(text, src, tgt, engines)
            except Exception as exc:  # noqa: BLE001
                out = f"<error: {type(exc).__name__}: {exc}>"
            print(f"   [{src}->{tgt}] {text}")
            print(f"            => {out}\n")
        print("   Check: the right-hand side should be fluent target-language text,")
        print("   not the source words re-spelled in another script.")
    except Exception as exc:  # noqa: BLE001
        print("   Skipped (models unavailable):", exc)
        print("   Fix the readiness errors above (usually the HF token), then re-run.")

    print("=" * 70)
    print("3) EXPRESSIVE VOICE PROSODY (Edge-TTS deltas per emotion)")
    print("=" * 70)
    try:
        from tts.edge_engine import EdgeTTSEngine, EXPRESSIVENESS
        eng = EdgeTTSEngine()
        cases = {
            "neutral": {"arousal": 0.0, "valence": 0.0},
            "happy":   {"arousal": 0.7, "valence": 0.8},
            "angry":   {"arousal": 0.9, "valence": -0.7},
            "sad":     {"arousal": -0.6, "valence": -0.6},
            "excited": {"arousal": 0.95, "valence": 0.5},
        }
        print(f"   STS_EXPRESSIVENESS = {EXPRESSIVENESS}  (raise it for more drama)")
        for label, avd in cases.items():
            r, p, v = eng._derive_prosody({"detected_emo": label, "avd": avd})
            print(f"   {label:8s} -> rate {r:>5s}  pitch {p:>6s}  volume {v:>5s}")
        print("   Check: emotions are clearly differentiated (not all +0%).")
    except Exception as exc:  # noqa: BLE001
        print("   Skipped:", exc)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
