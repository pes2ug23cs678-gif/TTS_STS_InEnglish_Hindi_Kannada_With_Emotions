#!/usr/bin/env bash
#
# start.sh — launch the Trilingual Emotion STS app:
#   FastAPI backend (uvicorn, :8000) + Streamlit UI (app_backend.py).
#
# Portable: no machine-specific paths. Put your own settings in a local .env
# (git-ignored) — config.py loads it automatically. See .env.example.
#
set -u
cd "$(dirname "$0")"

# --- Activate your Python environment BEFORE running this, or uncomment & edit: ---
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate sts

IP="$(hostname -I 2>/dev/null | awk '{print $1}')"; IP="${IP:-127.0.0.1}"
export STS_API_URL="${STS_API_URL:-http://127.0.0.1:8000}"

# --- Tuned runtime flags (override any of these via your shell or .env) ---
export STS_ENGLISH_EDGE="${STS_ENGLISH_EDGE:-0}"     # 0 = Parler for English (Indian accent)
export STS_KN_CHUNK="${STS_KN_CHUNK:-1}"             # chunk long Kannada audio
export STS_KN_CLAUSE_SPLIT="${STS_KN_CLAUSE_SPLIT:-1}"  # split unpunctuated KN run-ons before MT
export AUDIO_EMOTION_WEIGHT="${AUDIO_EMOTION_WEIGHT:-0.5}"   # 50% audio / 50% text emotion
export ASR_CHUNK_SEC="${ASR_CHUNK_SEC:-8}"           # long-form ASR window (seconds)
export ASR_NO_SPEECH_THRESHOLD="${ASR_NO_SPEECH_THRESHOLD:-0.8}"
export ASR_VAD_SILENCE_MS="${ASR_VAD_SILENCE_MS:-500}"
export ASR_INITIAL_PROMPT_KN="${ASR_INITIAL_PROMPT_KN:-}"

# --- Optional fine-tuned Indic ASR models ----------------------------------------
# Leave unset to use base multilingual Whisper. To use local CTranslate2 models,
# set these in your .env (NOT here — keep machine paths out of the repo), e.g.:
#   ASR_MODEL_KN=/abs/path/to/ct2_models/whisper-kannada-medium
#   ASR_MODEL_HI=/abs/path/to/ct2_models/whisper-hindi-large-v2
export ASR_MODEL_KN="${ASR_MODEL_KN:-}"
export ASR_MODEL_HI="${ASR_MODEL_HI:-}"

# --- Start the backend if it isn't already up ---
if ss -ltn 2>/dev/null | grep -q ":8000 "; then
  echo "[=] Backend already on :8000 — reusing it."; BACK_PID=""
else
  echo "[*] Starting backend (uvicorn) on :8000 ..."
  uvicorn backend_api.main:app --host 0.0.0.0 --port 8000 & BACK_PID=$!
fi

# --- Pick a free UI port ---
PORT="${STS_UI_PORT:-8502}"
while ss -ltn 2>/dev/null | grep -q ":$PORT "; do PORT=$((PORT+1)); done

cleanup(){ echo; echo "[*] Shutting down ..."; [ -n "${BACK_PID:-}" ] && kill "$BACK_PID" 2>/dev/null; exit 0; }
trap cleanup INT TERM

# --- Optional HTTPS: used only if BOTH cert and key exist beside this script. ---
# Generate your own (never commit them): they are git-ignored.
#   openssl req -x509 -newkey rsa:2048 -nodes -keyout sts-key.pem -out sts-cert.pem -days 365 -subj "/CN=localhost"
if [ -f sts-cert.pem ] && [ -f sts-key.pem ]; then SCHEME=https; else SCHEME=http; fi

echo; echo "=================================================="
echo "  Backend : http://$IP:8000"
echo "  APP URL : $SCHEME://$IP:$PORT"
echo "=================================================="; echo

if [ "$SCHEME" = "https" ]; then
  streamlit run app_backend.py \
    --server.address 0.0.0.0 --server.port "$PORT" \
    --server.sslCertFile sts-cert.pem --server.sslKeyFile sts-key.pem \
    --server.enableCORS false --server.enableXsrfProtection false
else
  streamlit run app_backend.py \
    --server.address 0.0.0.0 --server.port "$PORT" \
    --server.enableCORS false --server.enableXsrfProtection false
fi
cleanup

