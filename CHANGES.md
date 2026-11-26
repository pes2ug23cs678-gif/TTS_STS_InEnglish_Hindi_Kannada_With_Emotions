# What changed and why

This revision fixes the two quality problems you reported (Kannada→Hindi / Kannada→English
coming out **transliterated** and the speech sounding **bland**), hardens the app against the
gated-model start-up crash, improves **heavy-accent** recognition, and adds a **one-click
installer** for Linux and Windows.

## 1. Translation: real translation, not transliteration
**Root cause.** IndicTrans2 is trained to read specially pre-processed input and to emit output
that *must* be post-processed back into native script by IndicTransToolkit's `IndicProcessor`.
The old code treated the processor as optional: if it wasn't active it silently fed the model
raw tag-prefixed text and returned the un-post-processed decode — which reads as
romanized/transliterated text, not a translation.

**Fix** (`models.py`):
- `IndicProcessor` is now loaded robustly (tries both module paths) and is **required**. If it
  can't be loaded, translation raises a clear, actionable error instead of silently
  transliterating (`_require_indic_processor`).
- Post-processing always runs; a failure is logged instead of silently returning raw decode.

## 2. Gated models no longer crash start-up (401)
**Root cause.** `ai4bharat/indictrans2-*-1B` (and `ai4bharat/indic-parler-tts`) are **gated** on
HuggingFace. Without a token the first download dies with `401 Unauthorized / gated repo`, which
killed the whole warm-up. The core pipeline never read any HF token.

**Fix**:
- `config.py` now loads `.env` and normalizes any HuggingFace token spelling
  (`HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, `HUGGINGFACEHUB_API_TOKEN`, …) into the environment
  **before** torch/transformers import, so every download authenticates.
- `models.py` / `tts/parler_engine.py` pass the token to `from_pretrained`, and a 401 is turned
  into a message that tells you exactly what to do.
- New `.env.example` documents the token and all optional settings.
- New `preflight.py` prints a readiness report (token / IndicProcessor / GPU / optional extras)
  at start-up for the Streamlit app, the desktop app, and the backend — so setup problems show
  up as one clear line, not a traceback.

## 3. Expressive, non-bland speech
- **Edge-TTS** (`tts/edge_engine.py`): the emotion→prosody ranges were widened a lot
  (rate ±12%→±24%, pitch ±40Hz→±60Hz, volume ±25%→±42%) with stronger mapping, so anger/joy/
  sadness actually read instead of sounding flat. Neutral/suppressed stays flat.
- **Prosody transfer** (`prosody.py`): the speech-to-speech path now mirrors the source
  speaker's pitch range / pace / loudness more strongly onto the output voice.
- **Parler-TTS** (`tts/parler_engine.py`): the style description now scales with how strongly the
  emotion was felt (e.g. *“very angry and forceful … expressive and emotive, not flat”*), and it
  authenticates for the gated model. Parler is the default voice on a GPU.
- New global dial **`STS_EXPRESSIVENESS`** (default `1.0`) scales delivery up/down across both
  Edge and prosody transfer.

## 4. Heavy-accent speech recognition
`asr/whisper_engine.py` decoding is tuned for accented/noisy speech: wider beam (`ASR_BEAM_SIZE`,
default 5), a temperature fallback ladder, `condition_on_previous_text=False` (stops accent-driven
repetition loops), confidence thresholds, and an optional per-language priming prompt
(`ASR_INITIAL_PROMPT_HI/KN/EN`). Still `large-v3` on GPU for the best accuracy.

## 5. Fully installable app (Linux + Windows)
- `install.sh` (Linux) and `install.ps1` + `install.bat` (Windows): build a self-contained
  `clean_env`, install GPU/CPU PyTorch (auto-detected), all requirements and the Parler voice,
  save your HF token to `.env`, run the readiness check, and create a double-click
  **“Trilingual Emotion STS”** launcher (desktop icon / Start-Menu shortcut).
- `run_desktop.bat` launches the native-window app on Windows.
- `INSTALL.md` is the step-by-step guide; `README.md` updated.

## How to verify
```bash
python preflight.py      # setup health
python verify_fixes.py   # shows real translations + per-emotion prosody
```
