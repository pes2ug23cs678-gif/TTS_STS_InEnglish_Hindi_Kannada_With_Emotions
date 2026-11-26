"""Low-latency multilingual speech-to-text using faster-whisper (CTranslate2).

Supports English, Hindi and Kannada. The model is multilingual (not the *.en
variant) so Hindi/Kannada speech can be transcribed as well as English.
"""
# asr/whisper_engine.py
import io
import os
import re
import logging
from collections import Counter
from typing import Union, Tuple, Optional, Any
import torch
import numpy as np
import soundfile as sf
import librosa
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# Languages this engine is used with, so auto-detect can be constrained if needed.
SUPPORTED_ASR_LANGS = {"en", "hi", "kn"}


def _int_env(name: str, default: int) -> int:
    try:
        v = int(os.getenv(name, ""))
        return v if v > 0 else default
    except (TypeError, ValueError):
        return default


# Accent-robust decoding defaults. Override the beam via ASR_BEAM_SIZE if you need
# to trade accuracy for speed (ASR_BEAM_SIZE=1 is fastest, 5 is the robust default).
_BEAM = _int_env("ASR_BEAM_SIZE", 5)
DECODE_PARAMS = {
    "beam_size": _BEAM,
    "best_of": _BEAM,
    # Temperature fallback ladder: start greedy, and only widen if the pass looks
    # unreliable (low avg logprob / high compression ratio). This is what rescues
    # heavily-accented words a single greedy pass would mangle.
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "patience": 1.0,
}

# Optional per-language priming text. Off by default — set via env var only
# (ASR_INITIAL_PROMPT_EN / _HI / _KN). A prompt that contains words common in
# everyday speech can bleed into the transcript and corrupt ASR output, so the
# default is None for all languages; the script-mismatch gate handles Devanagari
# contamination instead.
_INITIAL_PROMPTS = {
    "en": os.getenv("ASR_INITIAL_PROMPT_EN") or None,
    "hi": os.getenv("ASR_INITIAL_PROMPT_HI") or None,
    "kn": os.getenv("ASR_INITIAL_PROMPT_KN") or None,
}


def _initial_prompt(language: Optional[str]) -> Optional[str]:
    if not language:
        return None
    return _INITIAL_PROMPTS.get(str(language).lower()[:2])


# --- Input audio conditioning & quality gate (STEP 1 + 2) --------------------------------
# Speech recognition is only as good as the waveform it is fed. Marginal mic recordings
# (DC offset, rumble, too quiet, clipped) degrade Whisper badly. We (1) CONDITION the audio
# before ASR and (2) measure its quality and surface a WARNING so the user can re-record
# instead of trusting a garbage transcript. Disable conditioning with STS_ASR_PREPROCESS=0.
_ASR_PREPROCESS = os.getenv("STS_ASR_PREPROCESS", "1") != "0"
_CLIP_PEAK = 0.99        # at/above this a SAMPLE is at full-scale
_CLIP_FRACTION = 0.0005  # flag clipping only when this FRACTION of samples pin (flat-topping)
_QUIET_RMS = 0.012       # below this RMS the input is too quiet to recognise reliably
_MIN_DURATION_SEC = 1.2  # shorter than this gives Whisper too little to lock onto

# --- ASR hallucination guard (en / hi / kn) ----------------------------------------------
# Forced-language ASR on unclear / mismatched / non-speech audio makes Whisper HALLUCINATE:
# the transcript comes back script-contaminated -- mostly the forced script but laced with the
# Unicode replacement char, stray Cyrillic/CJK/other foreign letters, or (for Hindi/Kannada) the
# target script NEARLY ABSENT (the speech was really English, or pure gibberish). We key on that
# signature, not on word meaning, so it is language-agnostic. Letters are classified one-by-one
# with str.isalpha() (which skips digits, punctuation AND Indic combining matras, so the counts
# stay consistent); legitimate code-mixing (English nouns in a Hindi sentence) is allowed, while a
# foreign script, a garbled char, or an Indic transcript with almost no target script is flagged.
_HG_RANGES = {"en": None, "hi": (0x0900, 0x097F), "kn": (0x0C80, 0x0CFF)}
_HG_MIN_LETTERS = 8      # below this a transcript is too short to judge contamination reliably
_HG_MIN_TARGET_FRAC = 0.15  # forced-Indic with target script below this fraction = wrong language
_HG_PUNCT_RUN = re.compile(r"([^\w\s])(?:\s*\1){4,}")  # same punctuation char 5+ times (eg ", , ,")
_HG_REPEAT_TOK = 6       # a transcript this many tokens+ is checked for a dominant repeated token
_HG_REPEAT_FRAC = 0.6    # one token at/above this fraction of all tokens = a repetition loop


def _asr_hallucination_warning(transcript: Optional[str], language: Optional[str]) -> str:
    """Return a warning if a transcript looks like a Whisper hallucination, else ''. Catches a
    language-independent degenerate/repetition loop (a number then comma-spam, dot runs, one token
    repeated) first, then en/hi/kn script contamination; other/auto languages skip the script pass."""
    if not transcript:
        return ""
    # 1) Degenerate / repetition-loop output -- language-independent, and it has no letters for the
    # script checks to judge (e.g. the laptop small model emitting "2017 , , , , ,"). Catch it first.
    t = transcript.strip()
    if _HG_PUNCT_RUN.search(t):
        return ("Transcription looks unreliable (repetitive / looping output). "
                "Please re-record clearly, or use text input.")
    _toks = t.split()
    if len(_toks) >= _HG_REPEAT_TOK and Counter(_toks).most_common(1)[0][1] / len(_toks) >= _HG_REPEAT_FRAC:
        return ("Transcription looks unreliable (repetitive / looping output). "
                "Please re-record clearly, or use text input.")
    # 2) Script contamination -- needs a known/forced language and real letters to judge.
    if not language:
        return ""
    lang = str(language).lower()[:2]
    if lang not in _HG_RANGES:
        return ""
    if transcript.count("�") >= 2:   # 2+ replacement chars = a broken decode; a lone one is a
        # minor artifact (a single stray glyph in an otherwise-clean long transcript), not garble.
        return ("Transcription looks unreliable (garbled / hallucinated output). "
                "Please re-record clearly, or use text input.")
    rng = _HG_RANGES[lang]
    n_exp = n_lat = n_foreign = 0
    for ch in transcript:
        if not ch.isalpha():            # skip digits, punctuation, spaces and combining matras
            continue
        o = ord(ch)
        if 0x41 <= o <= 0x5A or 0x61 <= o <= 0x7A:
            n_lat += 1                   # Latin (expected for en; code-mixing for hi/kn)
        elif rng is not None and rng[0] <= o <= rng[1]:
            n_exp += 1                   # the forced Indic script
        else:
            n_foreign += 1              # Cyrillic / CJK / other script = contamination
    if lang == "en":
        total = n_lat + n_foreign
        if total < _HG_MIN_LETTERS:
            return ""
        if n_foreign >= 2:              # non-Latin letters in an English transcript
            return ("Transcription looks unreliable (mixed-script / hallucinated output). "
                    "Please re-record clearly, or use text input.")
        return ""
    total = n_exp + n_lat + n_foreign
    if total < _HG_MIN_LETTERS:
        return ""
    if n_foreign >= 2:                  # Cyrillic/CJK/... in a Hindi/Kannada transcript
        return ("Transcription looks unreliable (mixed-script / hallucinated output). "
                "Please re-record clearly, or use text input.")
    if n_exp < _HG_MIN_TARGET_FRAC * total and n_lat > n_exp:   # target script nearly absent
        return ("Transcription looks unreliable (the speech does not match the selected "
                "language). Please check the input language, or re-record / use text input.")
    return ""


def _assess_audio_quality(audio: np.ndarray, sr: int = 16000) -> dict:
    # Measure the input waveform and flag capture problems that hurt ASR. Returns peak/rms/
    # duration plus a human-readable warning (empty when the audio looks fine).
    if audio is None or not audio.size:
        return {"peak": 0.0, "rms": 0.0, "duration_sec": 0.0,
                "warning": "The audio is empty - nothing to transcribe."}
    absx = np.abs(audio)
    peak = float(np.max(absx))
    clipped_frac = float(np.mean(absx >= _CLIP_PEAK))  # real clipping = many pinned samples
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    dur = float(audio.size) / float(sr or 16000)
    problems = []
    if clipped_frac >= _CLIP_FRACTION:
        problems.append("clipped/distorted (recorded too loud or too close to the mic)")
    if rms < _QUIET_RMS:
        problems.append("very quiet (speak louder or closer to the mic)")
    if dur < _MIN_DURATION_SEC:
        problems.append("very short (%.1fs - give a fuller sentence)" % dur)
    warning = ""
    if problems:
        warning = ("Input audio quality is poor: " + "; ".join(problems) +
                   ". The transcript may be unreliable - consider re-recording.")
    return {"peak": round(peak, 4), "clipped_frac": round(clipped_frac, 5),
            "rms": round(rms, 4), "duration_sec": round(dur, 2), "warning": warning}


def _condition_audio(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    # Light, broadly-safe conditioning so Whisper sees a cleaner waveform: remove DC offset,
    # high-pass out rumble/handling noise, soft-limit clipped peaks, and normalise the level
    # so quiet recordings are boosted. Never changes WHAT was said - only how cleanly it reads.
    if audio is None or not audio.size:
        return audio
    x = np.asarray(audio, dtype=np.float32)
    x = x - float(np.mean(x))                      # 1) DC-offset removal
    try:                                           # 2) high-pass ~70 Hz (zero-phase)
        from scipy.signal import butter, filtfilt
        b, a = butter(2, 70.0 / (0.5 * sr), btype="high")
        if x.size > 12:
            x = filtfilt(b, a, x).astype(np.float32)
    except Exception:  # noqa: BLE001 - scipy optional / tiny clip => skip filtering
        pass
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak >= _CLIP_PEAK:                          # 3) soft de-clip distorted peaks
        x = np.tanh(x * 1.5).astype(np.float32)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1e-5:                                 # 4) peak-normalise (rescues quiet audio)
        x = (x * (0.95 / peak)).astype(np.float32)
    return x


# --- Long-form chunking for fine-tuned Indic ASR models ----------------------------------
# A fine-tuned Whisper (e.g. vasista22/whisper-kannada-medium, whisper-hindi-large-v2) is trained
# almost entirely on SHORT (<15 s) clips, so on a longer utterance it emits end-of-transcript early
# (the last clauses vanish — "...ಮನೆಯ" and nothing after) or drops a stretch in the MIDDLE (a
# whole sentence missing, leaving an orphan verb like "...करते हैं बच्चे मैदान में..."). The base
# multilingual model handles long-form fine; the fine-tuned one does not. The fix the pipeline used
# before the backend/UI refactor (the vestigial STS_KN_CHUNK flag): split long audio into short
# windows the fine-tuned model CAN transcribe in full, then join.
#
# CRITICAL: the split must be GAP-FREE. We cover the whole clip with CONTIGUOUS windows and cut at
# the MIDDLE of a silence (never mid-word, never dropping a sample). An earlier version cut at the
# EDGES of librosa's non-silent runs and dropped the silence BETWEEN windows — if a quietly-spoken
# passage was misread as silence there, it was lost. Covering every sample makes that impossible.
_ASR_CHUNK_TOP_DB = _int_env("ASR_CHUNK_TOP_DB", 30)   # quieter-than-peak by this many dB = silence


def _split_audio_on_silence(audio: np.ndarray, sr: int, max_window_sec: float) -> list:
    """Return CONTIGUOUS [(start, end), ...] sample ranges that cover the whole clip with no gaps
    and no overlap (so no audio is ever dropped). Each range is <= max_window_sec where a silence
    allows it; cuts fall at the midpoint of a silence so words are never split. A stretch of
    continuous speech longer than the window is hard-cut only as a last resort."""
    if audio is None or not audio.size:
        return [(0, 0)]
    n = int(audio.size)
    max_len = max(int(max_window_sec * sr), sr)       # never below 1 s
    if n <= max_len:
        return [(0, n)]
    try:
        intervals = librosa.effects.split(audio, top_db=_ASR_CHUNK_TOP_DB)
    except Exception:  # noqa: BLE001 - librosa hiccup => one window, nothing dropped
        intervals = None
    # Candidate cut points = the MIDPOINT of every silence gap between consecutive speech runs.
    gap_mids = []
    if intervals is not None and len(intervals) > 1:
        for (s0, e0), (s1, e1) in zip(intervals[:-1], intervals[1:]):
            if int(s1) > int(e0):                     # a real silence gap exists
                gap_mids.append((int(e0) + int(s1)) // 2)
    ranges = []
    start = 0
    while n - start > max_len:
        limit = start + max_len
        cut = None
        for m in gap_mids:                            # furthest silence midpoint that still fits
            if start < m <= limit:
                cut = m
            elif m > limit:
                break
        if cut is None:                               # no pause in this window -> hard cut
            cut = limit
        ranges.append((start, cut))
        start = cut
    ranges.append((start, n))
    return ranges


class FasterWhisperASR:
    """Low-latency multilingual speech recognition using faster-whisper (CTranslate2)."""
    def __init__(self, model_size: Optional[str] = None, device: Optional[str] = None,
                 compute_type: Optional[str] = None):
        requested = (device or ("cuda" if torch.cuda.is_available() else "cpu")).lower()
        use_cuda = requested.startswith("cuda") and torch.cuda.is_available()

        device_index = 0
        if use_cuda and ":" in requested:          # "cuda:0" -> device="cuda", index=0
            try:
                device_index = int(requested.split(":", 1)[1])
            except ValueError:
                device_index = 0

        # Multilingual models (no ".en" suffix). Override with ASR_MODEL_SIZE if needed.
        # Larger models transcribe Hindi/Kannada proper nouns far more reliably, which
        # directly improves translation quality (a mis-heard name can never be translated
        # back correctly). On GPU we can afford the full large-v3; on CPU, large-v3-turbo
        # gives most of that accuracy at a fraction of the cost.
        default_size = "large-v3" if use_cuda else "large-v3-turbo"
        size = model_size or os.getenv("ASR_MODEL_SIZE", default_size)

        # Degrade gracefully on two axes: (1) an older faster-whisper build may not know a
        # given model name, and (2) a small-VRAM laptop GPU (e.g. a 4 GB RTX 3050) already
        # holding the two IndicTrans2 1B models + Parler-TTS + emotion2vec has no room for
        # Whisper, so loading ANY size on CUDA fails with "out of memory". Previously every
        # fallback size still targeted CUDA, so they all OOM'd and the app crashed. Now, if
        # the GPU can't host it, we retry the whole ladder on CPU (int8), which is accurate
        # and fast enough for ASR and never crashes for lack of VRAM.
        cuda_ctype = compute_type or "float16"
        cpu_ctype = compute_type if (compute_type and not use_cuda) else "int8"
        plans = []
        if use_cuda:
            plans.append(("cuda", device_index, cuda_ctype,
                          [size, "large-v3", "medium", "small", "base"]))
            # CPU fallback: turbo keeps most of large-v3's accuracy at a fraction of the cost.
            plans.append(("cpu", 0, cpu_ctype,
                          [os.getenv("ASR_MODEL_SIZE", "large-v3-turbo"), "small", "base"]))
        else:
            plans.append(("cpu", device_index, cpu_ctype,
                          [size, "large-v3-turbo", "medium", "small", "base"]))

        last_err = None
        self.model = None
        self.model_size = None
        self.device = None
        for dev, didx, ctype, sizes in plans:
            candidates = []
            for s in sizes:
                if s and s not in candidates:
                    candidates.append(s)
            for s in candidates:
                try:
                    self.model = WhisperModel(
                        s, device=dev, device_index=didx, compute_type=ctype)
                    self.model_size = s
                    self.device = dev
                    if dev == "cpu" and use_cuda:
                        logger.warning(
                            "Whisper ASR could not load on the GPU (VRAM likely exhausted "
                            "by the translation/TTS models); running on CPU with '%s'.", s)
                    break
                except Exception as err:   # unknown name, OOM, download failure, etc.
                    last_err = err
                    continue
            if self.model is not None:
                break
        if self.model is None:
            raise RuntimeError(
                f"Could not load any faster-whisper model (tried GPU then CPU): {last_err}")

        # Optional per-language fine-tuned models (ASR_MODEL_KN / ASR_MODEL_HI env vars).
        # When set, Kannada/Hindi audio routes to the specialized model; English uses the main one.
        self._lang_models: dict = {}
        for _lang, _env in [("kn", "ASR_MODEL_KN"), ("hi", "ASR_MODEL_HI")]:
            _mid = os.getenv(_env, "").strip()
            if not _mid:
                continue
            try:
                _lm = WhisperModel(
                    _mid, device=self.device, device_index=device_index,
                    compute_type=cuda_ctype if self.device == "cuda" else cpu_ctype)
                self._lang_models[_lang] = _lm
                logger.info("Loaded %s-specific ASR model: %s", _lang, _mid)
            except Exception as _e:  # noqa: BLE001
                logger.warning("Could not load %s model '%s': %s — falling back to %s.",
                               _lang, _mid, _e, self.model_size)

    @staticmethod
    def _decode_with_ffmpeg(raw: bytes) -> Tuple[np.ndarray, int]:
        """Decode a container libsndfile cannot read (m4a/AAC, webm/opus, mp4) to 16 kHz mono
        float32 using ffmpeg. RECORDING gives WAV (soundfile reads it), but file-UPLOAD commonly
        provides m4a (iPhone voice memos) or webm (browser capture) — without this, those uploads
        fail at sf.read() and the whole pipeline errors with no output voice. ffmpeg reads the
        bytes on stdin and writes raw 32-bit float PCM on stdout, so no temp file is needed."""
        import shutil
        import subprocess
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "Could not decode the uploaded audio: libsndfile does not support this format "
                "(e.g. m4a/webm) and ffmpeg is not installed. Install ffmpeg, or upload a WAV/MP3.")
        proc = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", "pipe:0",
             "-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1"],
            input=raw, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0 or not proc.stdout:
            _err = (proc.stderr or b"").decode("utf-8", "ignore").strip()[:200]
            raise RuntimeError(f"ffmpeg could not decode the uploaded audio ({_err or 'no output'}).")
        return np.frombuffer(proc.stdout, dtype=np.float32).copy(), 16000

    def _prepare_audio(self, audio_data: Union[bytes, np.ndarray, io.BytesIO]) -> np.ndarray:
        """Converts raw audio bytes or arrays to a 16kHz mono float32 numpy array."""
        if isinstance(audio_data, (bytes, io.BytesIO)):
            raw = audio_data if isinstance(audio_data, bytes) else audio_data.getvalue()
            try:
                audio, sr = sf.read(io.BytesIO(raw))
            except Exception as _sf_err:  # noqa: BLE001 - unsupported container -> ffmpeg fallback
                logger.info("soundfile could not read the audio (%s); decoding via ffmpeg.",
                            _sf_err)
                audio, sr = self._decode_with_ffmpeg(raw)
        else:
            audio = audio_data
            sr = 16000
        # Convert stereo to mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        # Resample to 16kHz: faster-whisper assumes 16kHz when given a raw array
        if sr != 16000:
            audio = librosa.resample(np.asarray(audio, dtype=np.float32),
            orig_sr=sr, target_sr=16000)
        # Ensure float32 normalized in [-1.0, 1.0]
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        # STEP 2: assess capture quality BEFORE conditioning (so the warning reflects the real
        # recording), stash it for transcribe() to surface; STEP 1: condition the waveform.
        self._last_audio_quality = _assess_audio_quality(audio, 16000)
        if _ASR_PREPROCESS:
            audio = _condition_audio(audio, 16000)
        return audio
    def transcribe(self, audio_input: Union[bytes, np.ndarray, io.BytesIO],
                   language: Optional[str] = None) -> Tuple[str, dict]:
        """Transcribes speech audio into text with casing and punctuation.

        language: ISO 2-letter code ('en', 'hi', 'kn') to force that language,
        or None to let Whisper auto-detect the spoken language.

        Decoding is tuned for HEAVY ACCENTS and noisy phone audio: a wider beam,
        a temperature fallback ladder (so a low-confidence pass is retried with more
        exploration instead of emitting garbage), and condition_on_previous_text=False
        (accented speech otherwise triggers repetition/hallucination loops). These
        cost a little latency but markedly improve accuracy on non-native speakers.
        """
        audio_arr = self._prepare_audio(audio_input)
        _lang_model = (self._lang_models.get(language)
                       if (language and getattr(self, "_lang_models", None)) else None)
        _active_model = _lang_model or self.model
        _decode_kw = dict(
            task="transcribe",                       # never translate; always keep source language
            language=language,                       # None => auto-detect
            beam_size=DECODE_PARAMS["beam_size"],
            best_of=DECODE_PARAMS["best_of"],
            temperature=DECODE_PARAMS["temperature"],
            patience=DECODE_PARAMS["patience"],
            condition_on_previous_text=False,        # stops accent-driven loops
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.8,
            initial_prompt=_initial_prompt(language),
        )
        # VAD splits audio at every pause, then each resulting mini-segment is judged
        # independently for no_speech — a natural inter-clause breath followed by a
        # short trailing fragment routinely crosses the threshold and gets silently
        # dropped, cutting off the last few words. With VAD off, Whisper sees the full
        # audio and handles its own 30-second chunking internally (no silent dropping).
        # Enable VAD via ASR_VAD_FILTER=1 only if you need it (e.g., very long silences).
        _use_vad = os.getenv("ASR_VAD_FILTER", "0") != "0"

        # Long-form handling. A fine-tuned Indic model (vasista22/whisper-kannada-medium etc.,
        # loaded via ASR_MODEL_KN/_HI) terminates early on long audio and drops the tail, so
        # whenever a language-specific model is active we pre-split the clip at silences into
        # short windows it can transcribe in full. The base multilingual model only needs this
        # for genuinely long clips. ASR_CHUNK=1 forces it on, ASR_CHUNK=0 off; tune the window
        # with ASR_CHUNK_SEC (lang-specific) / ASR_CHUNK_SEC_BASE (base model).
        _dur_sec = float(len(audio_arr)) / 16000.0
        # ASR_CHUNK is the current flag; STS_KN_CHUNK is the historical name kept for
        # backward compatibility with existing start.sh files that still export it.
        _chunk_env = (os.getenv("ASR_CHUNK", "").strip()
                      or os.getenv("STS_KN_CHUNK", "").strip())
        if _chunk_env in ("0", "false", "no"):
            _do_chunk = False
        elif _chunk_env in ("1", "true", "yes"):
            _do_chunk = True
        elif _lang_model is not None:
            _win_specific = float(_int_env("ASR_CHUNK_SEC", 12))
            _do_chunk = _dur_sec > _win_specific
        else:
            _win_base = float(_int_env("ASR_CHUNK_SEC_BASE", 24))
            _do_chunk = _dur_sec > _win_base

        def _run(_arr, _vad):
            if _vad:
                _vad_silence_ms = _int_env("ASR_VAD_SILENCE_MS", 500)
                return _active_model.transcribe(
                    _arr, vad_filter=True,
                    vad_parameters={"min_silence_duration_ms": _vad_silence_ms}, **_decode_kw)
            return _active_model.transcribe(_arr, vad_filter=False, **_decode_kw)

        info = None
        if _do_chunk:
            _win = float(_int_env("ASR_CHUNK_SEC", 12)) if _lang_model is not None \
                else float(_int_env("ASR_CHUNK_SEC_BASE", 24))
            _ranges = _split_audio_on_silence(audio_arr, 16000, _win)
            _parts = []
            for _s, _e in _ranges:
                if _e <= _s:
                    continue
                _seg, _inf = _run(audio_arr[_s:_e], _use_vad)
                info = _inf  # keep the last window's info (language/duration telemetry)
                _parts.append(" ".join([seg.text.strip() for seg in _seg]).strip())
            transcript = " ".join([p for p in _parts if p]).strip()
            logger.info("ASR long-form chunking: %.1fs audio -> %d window(s), %d chars.",
                        _dur_sec, len(_ranges), len(transcript))
            if info is None:  # no usable windows -> fall through to a whole-clip pass
                _do_chunk = False
        if not _do_chunk:
            segments, info = _run(audio_arr, _use_vad)
            transcript = " ".join([seg.text.strip() for seg in segments]).strip()
        # VAD-fallback: only needed when VAD is on and strips the whole clip.
        _q0 = getattr(self, "_last_audio_quality", None) or {}
        _has_energy = float(_q0.get("rms", 0.0)) >= _QUIET_RMS
        if (_use_vad and not transcript and _has_energy
                and getattr(info, "duration_after_vad", info.duration) < 0.3):
            logger.warning("VAD removed all audio but the clip has energy; retrying ASR without VAD.")
            segments, info = _active_model.transcribe(audio_arr, vad_filter=False, **_decode_kw)
            transcript = " ".join([seg.text.strip() for seg in segments]).strip()
        telemetry = {
            "detected_language": info.language,
            "language_probability": round(info.language_probability, 3),
            # Report the TRUE full-clip duration (info.duration is only the last window when chunked).
            "audio_duration_sec": round(_dur_sec, 2)
        }
        # STEP 2 + v2 BACK-GATE: carry the pre-ASR quality assessment up, and if NOTHING was
        # transcribed, flag "no speech" (long silence / room noise / distortion the VAD rejected)
        # so the user is never left with a silent blank and no explanation.
        q = getattr(self, "_last_audio_quality", None)
        q = dict(q) if q else {}
        self._last_language_mismatch = False
        self._last_detected_language = info.language  # always expose for auto-suggest
        if not transcript:
            _ns = "No speech detected - please speak clearly into the mic."
            q["warning"] = (q.get("warning", "") + " " + _ns).strip() if q.get("warning") else _ns
            q["no_speech"] = True
        else:
            # Language mismatch: user forced a language but Whisper heard something different.
            _LANG_DISPLAY = {"en": "English", "hi": "Hindi", "kn": "Kannada"}
            if language and info.language and info.language != language:
                _chosen = _LANG_DISPLAY.get(language, language.upper())
                _detected = _LANG_DISPLAY.get(info.language, info.language.upper())
                _mismatch_msg = (
                    f"Please enter your speech in {_chosen} — {_detected} was detected. "
                    f"Re-record in {_chosen}, or switch Input Language to {_detected}."
                )
                q["warning"] = _mismatch_msg
                q["language_mismatch"] = True
                self._last_language_mismatch = True
            else:
                # --- Script-level mismatch gate (catches the common case where Whisper
                # reports info.language="kn" but the transcript is actually Devanagari,
                # or reports "hi" but outputs Kannada script). Language-ID alone misses this.
                _script_mismatch = False
                if language in ("kn", "hi") and transcript:
                    _alpha = sum(1 for c in transcript if c.isalpha())
                    if _alpha >= 5:
                        _deva = sum(1 for c in transcript if 0x0900 <= ord(c) <= 0x097F)
                        _knda = sum(1 for c in transcript if 0x0C80 <= ord(c) <= 0x0CFF)
                        if language == "kn" and _deva > _knda and _deva > _alpha * 0.25:
                            _script_mismatch = True
                            self._last_language_mismatch = True
                            self._last_detected_language = "hi"
                            q["warning"] = (
                                "Please enter your speech in Kannada — Hindi script was detected. "
                                "Re-record in Kannada, or switch Input Language to Hindi.")
                            q["language_mismatch"] = True
                        elif language == "hi" and _knda > _deva and _knda > _alpha * 0.25:
                            _script_mismatch = True
                            self._last_language_mismatch = True
                            self._last_detected_language = "kn"
                            q["warning"] = (
                                "Please enter your speech in Hindi — Kannada script was detected. "
                                "Re-record in Hindi, or switch Input Language to Kannada.")
                            q["language_mismatch"] = True
                if not _script_mismatch:
                    # Transcript present: guard against a script-contaminated hallucination on the
                    # forced language (falls back to the detected language when nothing was forced).
                    _hw = _asr_hallucination_warning(transcript, language or info.language)
                    if _hw:
                        q["warning"] = (q.get("warning", "") + " " + _hw).strip() if q.get("warning") else _hw
                        q["hallucination"] = True
        if q:
            telemetry["input_quality"] = q
            if q.get("warning"):
                logger.warning("ASR input quality: %s", q["warning"])
        # Expose the final warning and mismatch flag so the UI can surface / block on them.
        self._last_quality_warning = q.get("warning", "")
        return transcript, telemetry
def clean_asr_transcript(text: str) -> str:
    """Sanitizes raw Whisper transcripts to protect downstream NMT and Emotion engines."""
    if not text:
        return ""
    # 1. Deduplicate consecutive word repetitions (stutters like "I I I" -> "I", "and and" -> "and")
    cleaned = re.sub(r"\b(\w+)(?:\s+\1\b)+", r"\1", text, flags=re.IGNORECASE)
    # 2. Strip standard acoustic fillers that distort emotion/NMT syntax
    cleaned = re.sub(r"(?i)\b(um+|uh+|er+|ah+)\b[,.]?", "", cleaned)
    # 3. Collapse duplicate punctuation generated by Whisper decoders (e.g., "??", "!!", "....")
    cleaned = re.sub(r"([?.!,])\1+", r"\1", cleaned)
    # 3b. Drop stray Unicode replacement chars (minor decode artifacts) so they never reach NMT.
    cleaned = cleaned.replace("�", " ")
    # 4. Normalize multiple whitespaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned
# Lazy cache: {lang_code: hunspell.HunSpell instance | None}
# None means "tried to load but dictionary not found / library not installed".
_HUNSPELL_CACHE: dict = {}

_KN_DIC_SEARCH_DIRS = [
    # Project-local fallback: drop kn_IN.dic / kn_IN.aff here when the system
    # package (hunspell-kn) is unavailable — download from the LibreOffice
    # dictionaries repo (github.com/LibreOffice/dictionaries/tree/master/kn_IN).
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "dictionaries"),
    os.path.join(os.path.dirname(__file__), "dictionaries"),
    "/usr/share/hunspell",
    "/usr/share/myspell/dicts",
    "/usr/local/share/hunspell",
    "/opt/homebrew/share/hunspell",   # macOS Homebrew
]


def _get_hunspell(lang_code: str):
    """Return a cached HunSpell instance for lang_code, or None if unavailable."""
    if lang_code in _HUNSPELL_CACHE:
        return _HUNSPELL_CACHE[lang_code]
    _HUNSPELL_CACHE[lang_code] = None   # mark as attempted
    _dic_stems = {"kn": "kn_IN"}
    stem = _dic_stems.get(lang_code)
    if not stem:
        return None
    try:
        import hunspell as _hunspell_lib
        import os as _os
        for d in _KN_DIC_SEARCH_DIRS:
            dic = f"{d}/{stem}.dic"
            aff = f"{d}/{stem}.aff"
            if _os.path.exists(dic) and _os.path.exists(aff):
                try:
                    _HUNSPELL_CACHE[lang_code] = _hunspell_lib.HunSpell(dic, aff)
                    logger.info("Loaded Hunspell dictionary: %s", dic)
                except Exception as _e:  # noqa: BLE001
                    logger.warning("Failed to load Hunspell %s: %s", dic, _e)
                break
        else:
            logger.warning("Hunspell %s.dic not found; Kannada spell check disabled.", stem)
    except ImportError:
        logger.warning("hunspell Python package not installed; Kannada spell check disabled.")
    except Exception as _e:  # noqa: BLE001
        logger.warning("Hunspell init error: %s", _e)
    return _HUNSPELL_CACHE[lang_code]


def _polish_transcript(text: str, language: Optional[str]) -> str:
    """Post-process a clean transcript for better spelling and capitalisation.

    * Capitalises the first letter of every sentence (all languages).
    * Fixes lone 'i' → 'I' (English).
    * English: corrects obvious spelling mistakes via pyspellchecker — only purely
      lowercase tokens are checked; mixed-case / ALL-CAPS are left untouched
      (assumed proper nouns / acronyms).
    * Kannada: corrects misspelled words via Hunspell + kn_IN dictionary
      (requires: apt-get install hunspell-kn && pip install hunspell).
      Only replaces a token when Hunspell is certain (single top suggestion);
      unknown words / proper nouns with no suggestion are kept as-is.
    """
    if not text:
        return text
    # Capitalise first word of each sentence (Latin script only; Indic = no-op)
    text = re.sub(
        r'((?:^|(?<=[.!?])\s+))([a-z])',
        lambda m: m.group(1) + m.group(2).upper(),
        text,
    )
    if language == "en":
        text = re.sub(r'\bi\b', 'I', text)
        try:
            from spellchecker import SpellChecker  # pip install pyspellchecker
            _spell = SpellChecker()
            tokens = text.split()
            out = []
            for tok in tokens:
                core = re.sub(r'^[^\w]+|[^\w]+$', '', tok)
                if core and core.islower() and len(core) > 2 and core not in _spell:
                    best = _spell.correction(core)
                    if best:
                        tok = tok.replace(core, best, 1)
                out.append(tok)
            text = ' '.join(out)
        except ImportError:
            pass

    return text


def _normalize_script_for_language(text: str, language: Optional[str]) -> str:
    """Whisper's small multilingual model often transcribes KANNADA speech into Devanagari even
    when forced to 'kn' (e.g. "ನಾನು ಇಂದು..." comes back as "नानु इंदू..."). Downstream IndicTrans2
    is script-specific, so that Devanagari text breaks Kannada translation. When the spoken
    language is KNOWN (forced by the UI dropdown), rewrite the transcript into that language's own
    script. Devanagari<->Kannada is a faithful 1:1 Brahmic mapping, so phonetics are preserved
    (it fixes the SCRIPT, not any word Whisper genuinely misheard)."""
    if not text or not language:
        return text
    lang = str(language).lower()[:2]
    has_deva = bool(re.search(r"[ऀ-ॿ]", text))
    has_knda = bool(re.search(r"[ಀ-೿]", text))
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
    except Exception:  # noqa: BLE001 - library optional; leave transcript untouched
        return text
    try:
        if lang == "kn" and has_deva and not has_knda:
            return transliterate(text, sanscript.DEVANAGARI, sanscript.KANNADA)
        if lang == "hi" and has_knda and not has_deva:
            return transliterate(text, sanscript.KANNADA, sanscript.DEVANAGARI)
    except Exception:  # noqa: BLE001
        return text
    return text


def load_whisper_engine(device: Optional[str] = None) -> FasterWhisperASR:
    """Factory loader instantiated once within cache wrappers."""
    return FasterWhisperASR(device=device)
def transcribe_speech(asr_engine: Union[FasterWhisperASR, WhisperModel], audio_buffer: Any,
                      language: Optional[str] = None) -> str:
    """Transcribe speech to text. `language` forces a language, None auto-detects."""
    text, _ = transcribe_with_language(asr_engine, audio_buffer, language)
    return text


def transcribe_with_language(asr_engine: Union[FasterWhisperASR, WhisperModel],
                             audio_buffer: Any,
                             language: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """Transcribe speech and also report the (detected or forced) 2-letter language code.

    Returns (clean_transcript, lang_code). lang_code is 'en' / 'hi' / 'kn' when known.
    """
    if audio_buffer is None:
        return "", language
    if isinstance(asr_engine, FasterWhisperASR):
        raw_transcript, telemetry = asr_engine.transcribe(audio_buffer, language=language)
        detected = language or telemetry.get("detected_language")
        # If the script-level or language-ID mismatch gate fired inside transcribe(),
        # return empty string — the caller will show the warning instead of garbled text.
        if getattr(asr_engine, '_last_language_mismatch', False):
            return "", detected
        _normalized = _normalize_script_for_language(
            clean_asr_transcript(raw_transcript), language)
        try:
            clean = _polish_transcript(_normalized, language)
        except Exception as _pe:  # noqa: BLE001
            logger.warning("_polish_transcript failed (%s); using unpolished text.", _pe)
            clean = _normalized
        return clean, detected
    # Fallback if a raw WhisperModel instance is passed directly
    segments, info = asr_engine.transcribe(
        audio_buffer, task="transcribe", language=language,
        beam_size=DECODE_PARAMS["beam_size"], best_of=DECODE_PARAMS["best_of"],
        temperature=DECODE_PARAMS["temperature"], patience=DECODE_PARAMS["patience"],
        condition_on_previous_text=False, compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0, no_speech_threshold=0.8,
        initial_prompt=_initial_prompt(language),
        vad_filter=False)
    raw_transcript = " ".join([seg.text.strip() for seg in segments]).strip()
    detected = language or getattr(info, "language", None)
    clean = _normalize_script_for_language(clean_asr_transcript(raw_transcript), language)
    return clean, detected

