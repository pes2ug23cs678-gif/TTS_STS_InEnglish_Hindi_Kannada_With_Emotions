"""Diagnose why the app may still be using Edge instead of Parler.

Run from the project folder:   python check_parler.py
"""
import os

print("ENABLE_PARLER env :", os.getenv("ENABLE_PARLER", "(unset -> enabled by default)"))
print("PARLER_TIMEOUT_SEC:", os.getenv("PARLER_TIMEOUT_SEC", "(unset -> 600s default)"))

print("\n[1] Is the parler_tts runtime installed?")
try:
    import parler_tts  # noqa: F401
    print("    OK - parler_tts imports.")
except Exception as e:
    print(f"    FAILED -> {type(e).__name__}: {e}")
    print("    Fix: pip install git+https://github.com/huggingface/parler-tts.git")

print("\n[2] Does the dispatcher register the Parler engine?")
try:
    from tts.dispatcher import TTSDispatcher
    d = TTSDispatcher()
    print("    registered engines :", sorted(d.engines))
    print("    'parler' available :", "parler" in d.engines)
    print("    resolve('parler')  :", d.resolve_engine("parler"),
          "(if this is 'edge', Parler is NOT registered)")
except Exception as e:
    import traceback
    traceback.print_exc()

print("\n[3] If registered, this loads the model (downloads on first run; needs HF access).")
print("    The model ai4bharat/indic-parler-tts is GATED: request access + `huggingface-cli login`.")
try:
    from tts.dispatcher import TTSDispatcher
    d = TTSDispatcher()
    if "parler" in d.engines:
        d.engines["parler"].warmup()
        print("    OK - Parler model loaded; it should now be selectable and used.")
    else:
        print("    Skipped - Parler not registered (see [1]/[2]).")
except Exception as e:
    print(f"    Model load FAILED -> {type(e).__name__}: {e}")
