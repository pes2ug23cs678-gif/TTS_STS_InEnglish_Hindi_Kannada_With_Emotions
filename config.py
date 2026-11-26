"""Central configuration: model IDs, language maps, prosody tables, and hardware setup."""
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load .env and the HuggingFace access token BEFORE torch/transformers import,
# so the gated IndicTrans2 and Indic-Parler-TTS models authenticate correctly.
# IndicTrans2 (en-indic-1B / indic-en-1B) and ai4bharat/indic-parler-tts are
# GATED on the Hub: without a token every download fails with HTTP 401, which
# is the crash that otherwise kills the whole pipeline at warm-up time.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent


def _load_env_files() -> None:
    """Load the project-local .env (and a .env found upward from CWD), if present."""
    try:
        from dotenv import load_dotenv
        load_dotenv(_PROJECT_ROOT / ".env")
        load_dotenv()  # default search from CWD upward; no-op if nothing found
    except Exception:  # noqa: BLE001 - dotenv is optional; fall back to a tiny parser
        env_path = _PROJECT_ROOT / ".env"
        try:
            if env_path.is_file():
                for raw in env_path.read_text(encoding="utf-8").splitlines():
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
        except Exception:  # noqa: BLE001
            pass


def _resolve_hf_token() -> str:
    """Return the HuggingFace token from any of the common env var spellings, and
    normalize it so transformers *and* huggingface_hub both see it."""
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN",
                "HUGGINGFACEHUB_API_TOKEN", "HF_API_TOKEN"):
        val = os.getenv(key)
        if val and val.strip():
            token = val.strip()
            # Publish under both canonical names so every library picks it up.
            os.environ["HF_TOKEN"] = token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = token
            return token
    return ""


_load_env_files()
HF_TOKEN = _resolve_hf_token()

import torch
from transformers.utils import logging as hf_logging

# Suppress sub-module overwrite warnings
hf_logging.set_verbosity_error()
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Automatic low-VRAM configuration.
# A small-VRAM GPU (e.g. a 4 GB laptop RTX 3050) cannot hold the two 1B IndicTrans2
# models + Parler-TTS + emotion2vec at once -- it OOM-crashes during translation. So we
# DETECT the GPU's memory and, when it is below a threshold, automatically switch to the
# lightweight configuration (distilled 200M translation models + no audio-emotion) that
# fits. No env vars needed; an explicit env var still overrides this choice either way.
# ---------------------------------------------------------------------------
def _detect_gpu_vram_gb() -> float:
    """Total VRAM of the first CUDA GPU in GB, or 0.0 on a CPU-only machine."""
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:  # noqa: BLE001 - any probe failure => treat as no GPU
        pass
    return 0.0


GPU_VRAM_GB = _detect_gpu_vram_gb()
# GPUs with less than this many GB use the lightweight models. 6 GB comfortably covers 4 GB
# cards; tune with STS_LOW_VRAM_GB. (A CPU-only box has no VRAM cap -- only speed -- so it
# keeps the full models.)
try:
    _LOW_VRAM_THRESHOLD_GB = float(os.getenv("STS_LOW_VRAM_GB", "6"))
except (TypeError, ValueError):
    _LOW_VRAM_THRESHOLD_GB = 6.0
LOW_VRAM_MODE = 0.0 < GPU_VRAM_GB < _LOW_VRAM_THRESHOLD_GB
# Hard override: STS_LOW_VRAM=1 forces lightweight mode on, =0 forces it off.
_low_vram_override = os.getenv("STS_LOW_VRAM")
if _low_vram_override is not None:
    LOW_VRAM_MODE = _low_vram_override.strip().lower() not in ("", "0", "false", "no")
def _auto_asr_model_size() -> str:
    """Pick the Whisper ASR size DYNAMICALLY by the host's RAM, leaving room for the rest of the
    resident pipeline (distilled translation + Parler + emotion ~5 GB). Whisper-small can't
    transcribe Kannada reliably, but a bigger ASR + the full pipeline OOMs an 8 GB box (observed:
    large-v3-turbo loaded, then translation+emotion on top killed the process). So we scale the
    ASR to the machine: 'small' where RAM is tight (keeps the pipeline alive), 'medium'/'large-v3'
    where there's genuine headroom. An explicit ASR_MODEL_SIZE always overrides this."""
    try:
        import psutil
        avail_gb = psutil.virtual_memory().available / (1024 ** 3)
        total_gb = psutil.virtual_memory().total / (1024 ** 3)
    except Exception:  # noqa: BLE001 - psutil missing -> safest choice
        return "small"
    # Cap by total-2 so a transient burst of free RAM can't pick a model the machine can't
    # actually sustain once every model is resident; then reserve the pipeline's footprint.
    budget = min(avail_gb, total_gb - 2.0) - 5.5
    if budget >= 3.5:
        return "large-v3"          # ~3 GB in RAM, best Indic accuracy (needs ~12 GB+ host)
    if budget >= 2.0:
        return "medium"            # ~1.5 GB, solid Indic accuracy (needs ~10 GB+ host)
    return "small"                 # tight RAM (e.g. 8 GB): only this fits beside the pipeline


if LOW_VRAM_MODE:
    _auto_asr = _auto_asr_model_size()
    logger.warning(
        "Low-VRAM GPU detected (%.1f GB < %.1f GB threshold): auto-selecting distilled "
        "200M translation models and disabling audio-emotion so the pipeline fits. "
        "Override with env vars (TRANSLATION_MODEL_*, ENABLE_AUDIO_EMOTION, STS_LOW_VRAM).",
        GPU_VRAM_GB, _LOW_VRAM_THRESHOLD_GB)
    # ASR size scales to available RAM (see _auto_asr_model_size): a bigger model transcribes
    # Kannada far better but, with the whole pipeline resident, only fits where there is RAM for
    # it. An explicit ASR_MODEL_SIZE still wins. whisper_engine also degrades on a load failure.
    logger.warning("Auto-selected ASR model size '%s' for the available system RAM "
                   "(override with ASR_MODEL_SIZE).", _auto_asr)
    os.environ.setdefault("ASR_MODEL_SIZE", _auto_asr)

EMOTION_MODEL_ID = "j-hartmann/emotion-english-roberta-large"

# ----------------- Speech (audio) emotion recognition -----------------
# emotion2vec reads emotion from the speaker's TONE on the speech-to-speech path, and is
# fused with the text model above. Enabled by default; set ENABLE_AUDIO_EMOTION=0 to skip.
# Default OFF on a low-VRAM GPU (emotion2vec is ~1.2 GB); an env var still overrides.
ENABLE_AUDIO_EMOTION = os.getenv("ENABLE_AUDIO_EMOTION",
                                 "0" if LOW_VRAM_MODE else "1") != "0"
AUDIO_EMOTION_MODEL_ID = os.getenv("AUDIO_EMOTION_MODEL_ID", "iic/emotion2vec_plus_large")
# Weight of the AUDIO (tone) AVD vector when fused with the text AVD (0..1). Audio carries
# arousal/tone better, so it is weighted a little higher by default.
AUDIO_EMOTION_WEIGHT = float(os.getenv("AUDIO_EMOTION_WEIGHT", "0.6"))
# ----------------------------------------------------------------------

# ----------------- AI4Bharat IndicTrans2 Configuration -----------------
# Two IndicTrans2 models give full trilingual coverage:
#   - EN_INDIC : English  -> Hindi / Kannada
#   - INDIC_EN : Hindi / Kannada -> English
# Hindi <-> Kannada is produced by pivoting through English (src -> en -> tgt),
# so no third (indic-indic) model is required.
# On a low-VRAM GPU default to the distilled 200M variants (~0.4 GB each vs ~2.2 GB),
# which is what lets translation fit alongside Parler. An env var still overrides.
_DEFAULT_EN_INDIC = ("ai4bharat/indictrans2-en-indic-dist-200M" if LOW_VRAM_MODE
                     else "ai4bharat/indictrans2-en-indic-1B")
_DEFAULT_INDIC_EN = ("ai4bharat/indictrans2-indic-en-dist-200M" if LOW_VRAM_MODE
                     else "ai4bharat/indictrans2-indic-en-1B")
TRANSLATION_MODEL_EN_INDIC = os.getenv("TRANSLATION_MODEL_EN_INDIC", _DEFAULT_EN_INDIC)
TRANSLATION_MODEL_INDIC_EN = os.getenv("TRANSLATION_MODEL_INDIC_EN", _DEFAULT_INDIC_EN)
# Direct Indic<->Indic model (e.g. Kannada <-> Hindi WITHOUT pivoting through English).
# The pivot route (kn->en->hi) runs translation twice, so its errors compound; this does it
# in ONE hop with a single ~0.6 GB model -- better quality and faster for the indic->indic
# path. Set TRANSLATION_MODEL_INDIC_INDIC="" to force the old English-pivot route.
TRANSLATION_MODEL_INDIC_INDIC = os.getenv(
    "TRANSLATION_MODEL_INDIC_INDIC", "ai4bharat/indictrans2-indic-indic-dist-320M").strip()
# Backward-compatible alias (en->indic was the historical single-model default)
TRANSLATION_MODEL_ID = TRANSLATION_MODEL_EN_INDIC

# IndicTrans2 ISO-639-3 Script/Lang Codes
LANG_MAP = {
    "en": "eng_Latn",
    "kn": "kan_Knda",
    "hi": "hin_Deva"
}
# The three languages the engine supports, and their display names for the UI.
SUPPORTED_LANGS = ("en", "hi", "kn")
LANG_DISPLAY = {"en": "English", "hi": "Hindi", "kn": "Kannada"}
# Map UI/display names (and a few aliases) back to internal 2-letter codes.
DISPLAY_TO_CODE = {
    "english": "en", "en": "en", "eng_latn": "en",
    "hindi": "hi", "hi": "hi", "hin_deva": "hi",
    "kannada": "kn", "kn": "kn", "kan_knda": "kn",
}
# -----------------------------------------------------------------------

# ----------------- AI4Bharat Indic-Parler-TTS Configuration -----------
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Run the (large) translation models on a different device than the rest. On a small-VRAM
# GPU, two IndicTrans2 1B models + Parler-TTS do not fit in 4 GB and translation OOM-crashes.
# Set STS_TRANSLATION_DEVICE=cpu to run translation on the CPU -- where it is int8-quantized
# to ~1 GB each (full 1B quality, in system RAM) -- freeing the GPU for Parler. Slightly
# slower per sentence, but no VRAM blow-up. Defaults to DEVICE (unchanged behaviour).
TRANSLATION_DEVICE = (os.getenv("STS_TRANSLATION_DEVICE", "").strip() or DEVICE)
TRANSLATION_DTYPE = torch.float32 if TRANSLATION_DEVICE == "cpu" else TORCH_DTYPE

# Whisper ASR loads large-v3 (~3 GB) on the GPU, and since it loads FIRST on the mic path
# (while VRAM is free), it then starves translation + Parler and the pipeline OOMs. On a
# low-VRAM GPU, run ASR on the CPU instead (it is fast enough there and frees ~3 GB of VRAM
# for the models that must be on the GPU). Override with STS_ASR_DEVICE.
ASR_DEVICE = (os.getenv("STS_ASR_DEVICE", "").strip()
              or ("cpu" if LOW_VRAM_MODE else DEVICE))

AI4BHARAT_SPEAKERS = {
    "kn": {"female": "Anu", "male": "Suresh"},
    "hi": {"female": "Anu", "male": "Suresh"},
    "en": {"female": "Mary", "male": "Thoma"},
}

PARLER_PARAMS = {
    "max_length": 600,
    "temperature": 0.7,
    "do_sample": True
}

# Dynamic Routing SLA (< 5.2s TTS Target)
ROUTING_CONFIG = {
    "expressive_threshold": 0.62,
    "max_words_local": 16,
    "sla_timeout_seconds": 5.2
}
# Hard cutoff for each TTS attempt. The 5.2 s SLA above is the latency target, not a cutoff;
# using it as a cutoff cancelled every synthesis longer than 5.2 s.
TTS_HARD_TIMEOUT_SEC = 25.0
# -----------------------------------------------------------------------

# Emotion labels whose delivery is suppressed (no emotion tags or prosody). Narrating violent or
# sexual content is often classified as disgust, and emotion tags then make it sound like a performance.
NO_TAG_LABELS = {"disgust"}

ROBERTA_TO_PROSODY_MAP = {
    "anger": "angry",
    "joy": "happy",
    "sadness": "sad",
    "fear": "fear",
    "disgust": "disgust",
    "surprise": "excited",
    "neutral": "neutral"
}

def configure_runtime_hardware():
    """Caps CPU threads and enables denormal flushing for stable low-latency inference."""
    try:
        available_cores = os.cpu_count() or 4
        # Default cap of 4 threads was measured fastest for this single-request workload
        # (more threads contend and slow it down). Override with STS_NUM_THREADS to tune.
        override = os.getenv("STS_NUM_THREADS")
        if override and override.isdigit() and int(override) > 0:
            allocated_threads = int(override)
        else:
            allocated_threads = max(1, min(available_cores, 4))
        torch.set_num_threads(allocated_threads)
        if hasattr(torch, "set_flush_denormal"):
            torch.set_flush_denormal(True)
    except Exception:
        pass
