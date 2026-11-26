Autonomous Trilingual Emotion STS Engine

An emotion-aware Speech-to-Speech / Text-to-Speech pipeline across English, Hindi and Kannada. It takes input in any one of the three languages (typed text or spoken audio) and produces expressive speech in a chosen output language, preserving the emotional tone of the original.
What it does

    Input: English, Hindi or Kannada — either typed text or a recorded/uploaded voice clip.
    Output: You choose the output language (one of the two that isn't your input), and only that language is produced — translated text plus natural, emotion-matched speech, along with the inferred affective state.
    All 6 directions: En↔Hi, En↔Kn, Hi↔Kn. Hindi↔Kannada pivots through English, so only two translation models are needed (plus an optional direct Indic↔Indic model).
    Emotion-aware: detected emotion is mapped to an Arousal–Valence–Dominance (AVD) vector that drives the voice's pitch, rate and volume.

Pipeline at a glance

speech ─► ASR (faster-whisper) ─┐
text   ─────────────────────────┤─► entity masking ─► IndicTrans2 NMT ─┐
                                 │   emotion (text + vocal tone)        ├─► unmask + transliterate
                                 │                                      ─► rules + verifier cleanup
                                 │                                      ─► expressive TTS
                                 └──────────────────────────────────────► output-language audio

Translation and emotion detection run together; the detected emotion conditions the TTS prosody so the output sounds like the input felt. On the voice path, the speaker's own tone (emotion2vec) and delivery (pitch/pace/loudness) are transferred onto the output.
Models
Stage	Model
Speech recognition	faster-whisper (CTranslate2) multilingual — base large-v3 / large-v3-turbo. Optional per-language fine-tunes via ASR_MODEL_KN / ASR_MODEL_HI (e.g. vasista22 Kannada / Hindi).
Translation (NMT)	AI4Bharat IndicTrans2 — en-indic-1B and indic-en-1B (Hi↔Kn pivots through English), plus optional direct indic-indic-dist-320M.
Emotion	Text: j-hartmann/emotion-english-roberta-large (on the English form). Voice tone: iic/emotion2vec_plus_large (FunASR), fused with the text label. Disable audio with ENABLE_AUDIO_EMOTION=0.
Speech synthesis	Indic-Parler-TTS (local, expressive; default on GPU) · Edge-TTS (fast) · ElevenLabs (cloud, if ELEVENLABS_API_KEY is set). Engine auto-selected with fallback.

    Resource note: two IndicTrans2 1B models, a Whisper model and the emotion model load together. First run downloads several GB and needs substantial RAM. On low-RAM machines, set the distilled models before launching:

    export TRANSLATION_MODEL_EN_INDIC=ai4bharat/indictrans2-en-indic-dist-200M
    export TRANSLATION_MODEL_INDIC_EN=ai4bharat/indictrans2-indic-en-dist-200M
    export ASR_MODEL_SIZE=small

Install

Requires Python 3.12 and (recommended) an NVIDIA GPU. A free HuggingFace token is needed for the gated models (see the note below).

python3.12 -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchaudio   # or .../cpu
pip install -r requirements.txt
pip install -r requirements_backend.txt
pip install git+https://github.com/huggingface/parler-tts.git   # local expressive GPU voice (optional)
cp .env.example .env     # then edit .env: set HF_TOKEN=... (and optional ASR_MODEL_* paths)
python preflight.py      # verify token / IndicProcessor / GPU

    A free HuggingFace token is required. IndicTrans2 and Indic-Parler-TTS are gated — without a token the app fails at start-up with 401 Unauthorized. Accept the licenses, create a READ token at https://huggingface.co/settings/tokens, and put it in .env as HF_TOKEN=... (see .env.example).

Run

bash start.sh

This starts the FastAPI backend (:8000) and the Streamlit UI, then prints the URL to open. Pick the input/output languages and speaker gender, then type text or record/upload speech. On the voice path, the speaker's vocal tone and delivery are transferred onto the output so it keeps the original feeling.

Runtime options (shell or .env): STS_EXPRESSIVENESS (default 1.0), AUDIO_EMOTION_WEIGHT, ASR_MODEL_KN / ASR_MODEL_HI (local fine-tuned CTranslate2 ASR dirs), ASR_CHUNK_SEC, STS_KN_CLAUSE_SPLIT. See start.sh for the tuned defaults.

    HTTPS (optional): if sts-cert.pem and sts-key.pem exist beside start.sh, the UI is served over HTTPS. Generate your own self-signed pair; never commit them (git-ignored):

    openssl req -x509 -newkey rsa:2048 -nodes -keyout sts-key.pem -out sts-cert.pem -days 365 -subj "/CN=localhost"

Project layout

app_backend.py      Streamlit UI (calls the FastAPI backend)
start.sh            Launch script — starts the backend + UI
backend_api/        FastAPI backend (auth, DB, /translate, /transcribe)
pipeline.py         STSPipeline: masking → NMT/emotion → cleanup → TTS
models.py           Neural engine loaders + translation/emotion inference
affect.py           Emotion → Arousal-Valence-Dominance (AVD) mapping
entity_masker.py    Protects proper nouns across translation
rules_engine.py     Surface-rule post-processing (rules.json)
verifier.py         Deterministic text cleanup + audio checks
config.py           Model IDs, language maps, prosody tables
asr/                faster-whisper speech recognition
tts/                Engine interface, dispatcher, and the TTS engines
tests/              Unit tests (auth, API, text-processing fixes)
scripts/            Dev & evaluation tooling (benchmarks, regression, timing)

Evaluation tooling

These dev/benchmark scripts live in scripts/. They import the core modules in the repo root, so run them from the root with both on the path:

PYTHONPATH="$PWD:$PWD/scripts" python scripts/translation_regression.py

    scripts/ab_translate.py — benchmarks IndicTrans2 model sizes / beam settings.
    scripts/translation_regression.py — marks each approved sentence PASS / FAIL / KNOWN / NEW.
    scripts/full_pipeline_test.py — runs rows through STSPipeline.process end to end.
    scripts/stage_timing.py — times translation, emotion, and both together.

Tests

pip install pytest && pytest tests/

License

MIT (code only) — see LICENSE. The machine-learning models have their own licenses and several are gated on HuggingFace; review and accept each before use.


# TTS_STS_InEnglish_Hindi_Kannada_With_Emotions
