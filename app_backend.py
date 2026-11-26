"""Streamlit front end that mirrors app.py but sends the work to the FastAPI backend.

Same UX as app.py: pick the input language, pick ONE output language (of the other two),
type or speak, and get an expressive, emotion-matched translation + speech. The engine is
chosen on the server (no engine picker, like app.py). Voice input goes through /translate so
the speaker's vocal tone is captured server-side.

Start the backend first, then this front end:
    uvicorn backend_api.main:app --port 8000
    streamlit run app_backend.py

The backend URL defaults to http://127.0.0.1:8000 and can be changed with STS_API_URL.
"""
import base64
import json
import os
import time
from datetime import datetime

import httpx
import streamlit as st

from ui_theme import configure_page, render_hero, pill, inject_audio_mutex

API_URL = os.getenv("STS_API_URL", "http://127.0.0.1:8000")
TIMEOUT = httpx.Timeout(600.0, connect=10.0)  # local engines can be slow on first run
MAX_TEXT_CHARS = 500
MOS_LOG_PATH = "mos_evaluations.jsonl"

LANG_NAMES = ["English", "Hindi", "Kannada"]
NAME_TO_API = {"English": "english", "Hindi": "hindi", "Kannada": "kannada"}

configure_page("Trilingual Emotion STS Engine")
inject_audio_mutex()

# ------------------------------- UI helpers: emotion theming -------------------------------
_EMO_BADGE = {
    "happy":    ("#4caf50", "😊"), "joy":      ("#4caf50", "😊"),
    "excited":  ("#ff9800", "🤩"), "surprise": ("#00bcd4", "😲"),
    "sad":      ("#2196f3", "😢"), "sadness":  ("#2196f3", "😢"),
    "angry":    ("#f44336", "😠"), "anger":    ("#f44336", "😠"),
    "fear":     ("#9c27b0", "😨"), "disgust":  ("#795548", "🤢"),
    "neutral":  ("#9e9e9e", "😐"),
}
_EMO_BORDER = {
    "happy": "#2d6a2d", "joy": "#2d6a2d", "excited": "#7a4800",
    "sad": "#0d2a4a", "sadness": "#0d2a4a",
    "angry": "#5a1010", "anger": "#5a1010",
    "fear": "#3a0d5a", "disgust": "#3a2a0d",
    "surprise": "#0d3a3a", "neutral": "#2a2a2a",
}

def _emo_badge_html(label: str, conf: float) -> str:
    color, icon = _EMO_BADGE.get(label.lower(), ("#9e9e9e", "🎭"))
    return (f'<span style="background:{color};color:#fff;padding:3px 10px;'
            f'border-radius:12px;font-size:0.85em;font-weight:600">'
            f'{icon} {label.capitalize()} · {conf*100:.0f}%</span>')

def _avd_bars_html(avd: dict) -> str:
    def bar(label, val, color):
        pct = int((float(val) + 1) * 50)
        return (f'<div style="margin:3px 0;font-size:0.75em;color:#aaa">{label}'
                f'<div style="background:#333;border-radius:3px;height:5px;margin-top:2px">'
                f'<div style="background:{color};width:{pct}%;height:5px;border-radius:3px"></div>'
                f'</div></div>')
    return ('<div style="margin:6px 0 2px 0">'
            + bar("Arousal",   avd.get("arousal", 0),   "#e57373")
            + bar("Valence",   avd.get("valence", 0),   "#81c784")
            + bar("Dominance", avd.get("dominance", 0), "#64b5f6")
            + '</div>')

# ------------------------------- API helpers -------------------------------
def error_text(resp):
    try:
        return resp.json().get("detail", resp.text)
    except Exception:  # noqa: BLE001
        return resp.text


def mime_for(audio_bytes):
    return "audio/wav" if audio_bytes[:4] == b"RIFF" else "audio/mpeg"


def _base_url():
    """Current backend URL: the in-app override (sidebar field) if set, else the STS_API_URL
    default. Lets you switch between a LAN IP (same WiFi) and a Tailscale IP (any network) live."""
    try:
        return (st.session_state.get("api_url") or "").strip() or API_URL
    except Exception:  # noqa: BLE001 - session_state not ready yet
        return API_URL


def _client(token=None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return httpx.Client(base_url=_base_url(), timeout=TIMEOUT, headers=headers)


def backend_health():
    """Quick liveness probe so a backend that crashed on start-up (e.g. the gated
    model / missing HF token) is visible here, not just as a failed request later."""
    try:
        with httpx.Client(base_url=_base_url(), timeout=httpx.Timeout(4.0, connect=3.0)) as c:
            return c.get("/health").status_code == 200
    except Exception:  # noqa: BLE001
        return False


def login(email, password):
    with _client() as client:
        resp = client.post("/login", data={"username": email, "password": password})
    if resp.status_code != 200:
        raise RuntimeError(error_text(resp))
    return resp.json()["access_token"]


def register(name, email, password):
    with _client() as client:
        resp = client.post("/register", json={"name": name, "email": email, "password": password})
    if resp.status_code not in (200, 201):
        raise RuntimeError(error_text(resp))
    return resp.json()


def synthesize_text(token, text, input_lang, output_lang, gender):
    """Text -> translated text + speech (base64). Emotion is detected server-side (text model)."""
    with _client(token) as client:
        resp = client.post("/synthesize", data={
            "text": text, "input_lang": input_lang,
            "output_lang": output_lang, "gender": gender})
    if resp.status_code != 200:
        raise RuntimeError(error_text(resp))
    data = resp.json()
    data["audio"] = base64.b64decode(data["audio_base64"])
    return data


def translate_speech(token, audio_bytes, filename, input_lang, output_lang, gender):
    """Speech -> translated text + speech. Uses /translate so vocal TONE is fused server-side."""
    with _client(token) as client:
        resp = client.post("/translate",
            data={"input_lang": input_lang, "output_lang": output_lang, "gender": gender},
            files={"file": (filename, audio_bytes, "audio/wav")})
        if resp.status_code != 200:
            raise RuntimeError(error_text(resp))
        data = resp.json()
        audio_resp = client.get(data["output_audio_url"])
        if audio_resp.status_code != 200:
            raise RuntimeError(error_text(audio_resp))
        data["audio"] = audio_resp.content
    return data


def transcribe_audio(token, audio_bytes, filename, input_lang):
    """Speech -> recognised text only (ASR), so it can be shown/edited before translation."""
    with _client(token) as client:
        resp = client.post("/transcribe",
            data={"input_lang": input_lang},
            files={"file": (filename, audio_bytes, "audio/wav")})
    if resp.status_code != 200:
        raise RuntimeError(error_text(resp))
    return resp.json()   # {transcript, input_quality_warning}


def resynthesize_text(token, text, output_lang, gender, emotion):
    """Voice-only change: speak already-translated `text` without re-translating.
    `emotion` is the dict returned earlier by the backend (label + avd + suppress_tags)."""
    avd = (emotion or {}).get("avd", {}) or {}
    prosody = (emotion or {}).get("prosody")
    with _client(token) as client:
        resp = client.post("/resynthesize", data={
            "text": text, "output_lang": output_lang, "gender": gender,
            "emotion_label": (emotion or {}).get("label", "neutral"),
            "arousal": avd.get("arousal", 0.0), "valence": avd.get("valence", 0.0),
            "dominance": avd.get("dominance", 0.0),
            "suppress_tags": str(bool((emotion or {}).get("suppress_tags", False))).lower(),
            "prosody_json": json.dumps(prosody) if prosody else ""})
    if resp.status_code != 200:
        raise RuntimeError(error_text(resp))
    data = resp.json()
    data["audio"] = base64.b64decode(data["audio_base64"])
    return data


def log_mos(src_text, out_lang_name, emo, conf, scores, notes):
    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "input": {"source_text": src_text, "target_language": out_lang_name},
        "affective_state": {"dominant_emotion": emo, "confidence": round(float(conf), 4)},
        "mos_ratings": scores,
        "evaluator_notes": (notes or "").strip(),
    }
    with open(MOS_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ------------------------------- Header -------------------------------
render_hero("Trilingual Emotion STS Engine", center=True)

# ------------------------------- Auth gate (main body) -------------------------------
if "token" not in st.session_state:
    st.session_state["token"] = None

token = st.session_state["token"]

if not token:
    _lc, _mc, _rc = st.columns([1, 2, 1])
    with _mc:
        st.markdown("### Account")
        _mode = st.radio("", ["Log in", "Register"], horizontal=True,
                         label_visibility="collapsed")
        _email = st.text_input("Email")
        _password = st.text_input("Password", type="password")
        if _mode == "Register":
            _password2 = st.text_input("Confirm Password", type="password")
            _name = st.text_input("Your name")
        else:
            _password2 = ""
            _name = ""
        if st.button("Continue", use_container_width=True, type="primary"):
            if _mode == "Log in":
                try:
                    with st.status("Signing in …", expanded=True) as _s:
                        _tok = login(_email, _password)
                        _s.update(label="✓ Welcome back!", state="complete", expanded=False)
                    time.sleep(0.9)
                    st.session_state["token"] = _tok
                    st.session_state["email"] = _email
                    st.rerun()
                except Exception as _e:
                    st.error(f"Login failed: {_e}")
            else:
                if _password != _password2:
                    st.error("Passwords do not match — please re-enter.")
                elif not _password:
                    st.error("Password cannot be empty.")
                else:
                    try:
                        with st.status("Creating your account …", expanded=True) as _s:
                            register(_name, _email, _password)
                            _s.update(label="✓ Account created!", state="complete", expanded=False)
                        with st.status("Signing you in …", expanded=True) as _s:
                            _tok = login(_email, _password)
                            _s.update(label="✓ All set — loading your session!", state="complete", expanded=False)
                        time.sleep(1.1)
                        st.session_state["token"] = _tok
                        st.session_state["email"] = _email
                        st.rerun()
                    except Exception as _e:
                        st.error(f"Registration failed: {_e}")
    st.stop()

# Logged-in: show account strip in main body.
_al, _am, _ar = st.columns([1, 2, 1])
with _am:
    _ec1, _ec2 = st.columns([3, 1])
    _ec1.caption(f"Logged in as {st.session_state.get('email', '')}")
    if _ec2.button("Log out", key="logout_btn"):
        st.session_state["token"] = None
        st.session_state.pop("email", None)
        st.rerun()

col1, col2 = st.columns([1, 1], gap="large")

with col1.container(border=True):
    st.subheader("1. Input & Configuration")
    _API_TO_NAME = {v: k for k, v in NAME_TO_API.items()}  # reverse map for auto-suggest
    _suggested = st.session_state.get("_bk_suggested_lang", "")
    _lang_idx = LANG_NAMES.index(_suggested) if _suggested in LANG_NAMES else 0
    in_lang_name = st.selectbox("Input Language", LANG_NAMES, index=_lang_idx)
    src_api = NAME_TO_API[in_lang_name]
    # Clear stale recording/transcript/output when input language changes.
    if st.session_state.get("_bk_prev_in_lang") != in_lang_name:
        for _k in ["bk_audio_hash", "bk_asr_transcript", "bk_transcript_warning",
                   "bk_lang_mismatch", "bk_asr_error", "_bk_suggested_lang", "result", "trans_cache"]:
            st.session_state.pop(_k, None)
        st.session_state["_bk_reset_ctr"] = st.session_state.get("_bk_reset_ctr", 0) + 1
        st.session_state["_bk_prev_in_lang"] = in_lang_name
    _bk_audio_key = st.session_state.get("_bk_reset_ctr", 0)

    out_names = [n for n in LANG_NAMES if n != in_lang_name]
    out_lang_name = st.selectbox("Output Language", out_names, index=0,
        help="Choose which language to translate into (the two options are the languages "
             "other than your input). Only this language is produced.")
    out_api = NAME_TO_API[out_lang_name]

    st.markdown(pill(f"{in_lang_name} → {out_lang_name}", "good"), unsafe_allow_html=True)

    input_mode = st.radio("Input Modality",
        ["Text Input (TTS)", "Voice Input (Speech-to-Speech)"], horizontal=True)

    recorded = None
    voice_text = ""   # the (editable) recognised transcript for the voice path
    if input_mode.startswith("Text"):
        source_text = st.text_area(f"Input ({in_lang_name})", height=130,
            max_chars=MAX_TEXT_CHARS, placeholder=f"Type {in_lang_name} text here...")
    else:
        source = st.radio("Speech source", ["Record", "Upload file"], horizontal=True)
        if source == "Record" and hasattr(st, "audio_input"):
            recorded = st.audio_input("Record Speech Input", key=f"bk_rec_{_bk_audio_key}")
        else:
            recorded = st.file_uploader("Upload Speech Input (WAV/MP3/M4A)",
                type=["wav", "mp3", "m4a"], key=f"bk_upl_{_bk_audio_key}")
        if recorded is not None:
            _inp_bytes = recorded.getvalue()
            _inp_ext = "wav" if _inp_bytes[:4] == b"RIFF" else "mp3"
            st.audio(_inp_bytes, format=f"audio/{_inp_ext}")
            st.download_button("⬇️ Save input recording", data=_inp_bytes,
                file_name=f"input_{in_lang_name.lower()}.{_inp_ext}",
                mime=f"audio/{_inp_ext}", use_container_width=True)
            # Two-step voice flow (like app.py): transcribe FIRST so the recognised text can be
            # seen and edited BEFORE translation. Transcribe once per clip; a new clip re-runs ASR.
            import hashlib
            _ah = hashlib.md5(_inp_bytes).hexdigest() + ":" + src_api
            if st.session_state.get("bk_audio_hash") != _ah:
                try:
                    with st.spinner(f"Transcribing {in_lang_name} speech..."):
                        _td = transcribe_audio(token, recorded.getvalue(),
                                               getattr(recorded, "name", None) or "input.wav", src_api)
                    _CODE_TO_NAME = {"en": "English", "hi": "Hindi", "kn": "Kannada"}
                    if _td.get("language_mismatch"):
                        _det_code = _td.get("detected_language") or ""
                        st.session_state["bk_lang_mismatch"] = True
                        st.session_state["bk_asr_error"] = _td.get("input_quality_warning", "Language mismatch.")
                        st.session_state["_bk_suggested_lang"] = _CODE_TO_NAME.get(_det_code, "")
                        st.session_state["bk_asr_transcript"] = ""
                        st.session_state["bk_transcript_warning"] = ""
                    else:
                        st.session_state["bk_asr_transcript"] = _td.get("transcript", "")
                        st.session_state["bk_transcript_warning"] = _td.get("input_quality_warning", "")
                        st.session_state.pop("_bk_suggested_lang", None)
                    st.session_state["bk_audio_hash"] = _ah
                except Exception as _e:  # noqa: BLE001
                    st.session_state["bk_asr_transcript"] = ""
                    st.session_state["bk_transcript_warning"] = ""
                    st.session_state["bk_lang_mismatch"] = False
                    st.session_state["bk_asr_error"] = str(_e)
                    st.session_state["bk_audio_hash"] = _ah  # don't re-run on rerender
            _bk_err = st.session_state.get("bk_asr_error", "")
            if _bk_err:
                if st.session_state.get("bk_lang_mismatch"):
                    st.error(_bk_err)
                    # Force the audio widget to reset so the user must re-record in the correct language.
                    st.session_state["_bk_reset_ctr"] = st.session_state.get("_bk_reset_ctr", 0) + 1
                    for _k in ["bk_asr_error", "bk_lang_mismatch", "bk_audio_hash", "bk_asr_transcript"]:
                        st.session_state.pop(_k, None)
                else:
                    st.error(f"Transcription failed: {_bk_err}")
                st.stop()
            if st.session_state.get("bk_transcript_warning"):
                st.warning(st.session_state["bk_transcript_warning"])
            voice_text = st.text_area("Recognized Transcript (editable)",
                value=st.session_state.get("bk_asr_transcript", ""), height=110,
                help="Edit to fix any ASR error before translating. Unchanged -> the original "
                     "audio is translated (keeping vocal-tone emotion); edited -> the corrected "
                     "text is translated.")
        source_text = ""  # the voice path uses voice_text (the editable transcript) instead

    with st.expander("⚙️ Advanced Settings", expanded=False):
        gender = st.selectbox("Speaker Voice Gender", ["female", "male"], index=0)
        st.caption("Voice engine is selected automatically on the server.")
    run_btn = st.button("Run Pipeline", type="primary", use_container_width=True)

with col2.container(border=True):
    st.subheader(f"2. Output — {out_lang_name}")

    if "trans_cache" not in st.session_state:
        st.session_state["trans_cache"] = {}

    if run_btn:
        is_text = input_mode.startswith("Text")
        if is_text and not (source_text or "").strip():
            st.warning("Please enter some text first.")
        elif not is_text and recorded is None:
            st.warning("Please record or upload a speech clip first.")
        else:
            # Script validation for text input: reject if the wrong Indic script was typed
            if is_text:
                _txt = (source_text or "").strip()
                _has_deva = any(0x0900 <= ord(c) <= 0x097F for c in _txt)
                _has_knda = any(0x0C80 <= ord(c) <= 0x0CFF for c in _txt)
                _script_err = None
                if src_api == "kannada" and _has_deva:
                    _script_err = ("Input text appears to be in Hindi (Devanagari script). "
                                   "Please type in Kannada, or switch Input Language to Hindi.")
                elif src_api == "hindi" and _has_knda:
                    _script_err = ("Input text appears to be in Kannada script. "
                                   "Please type in Hindi, or switch Input Language to Kannada.")
                elif src_api == "english" and (_has_deva or _has_knda):
                    _script_err = ("Input text appears to contain Indic script. "
                                   "Please type in English, or switch Input Language.")
                if _script_err:
                    st.error(_script_err)
                    st.stop()

            # Cache key = what the TRANSLATION/EMOTION depend on (NOT gender). For voice it's
            # a hash of the clip; for text it's the text. A voice-only change hits the cache.
            if is_text:
                cache_key = ("text", src_api, out_api, source_text.strip())
            else:
                import hashlib
                cache_key = ("voice", src_api, out_api,
                             hashlib.md5(recorded.getvalue()).hexdigest(),
                             (voice_text or "").strip())
            cached = st.session_state["trans_cache"].get(cache_key)
            started = time.perf_counter()
            data = None
            # Fast path: voice-only change -> /resynthesize from cached translation/emotion.
            if cached:
                try:
                    with st.status(f"Voice change only — re-synthesizing in {gender} "
                                   f"(translation reused) ...", expanded=True) as status:
                        rdata = resynthesize_text(token, cached["output_text"], out_api,
                                                  gender, cached["emotion"])
                        if not rdata.get("audio"):
                            raise RuntimeError("re-synthesis returned no audio")
                        status.update(label=f"Voice updated to {gender}! (no re-translation)",
                                      state="complete", expanded=False)
                    data = {
                        "input_text": cached.get("input_text", ""),
                        "output_text": cached["output_text"],
                        "emotion_label": cached["emotion"].get("label", "neutral"),
                        "emotion_confidence": cached["emotion"].get("confidence", 0.0),
                        "audio": rdata["audio"],
                    }
                except Exception:  # noqa: BLE001 - self-heal silently via the full call
                    data = None
            try:
                if data is None:
                    with st.status(f"Translating {in_lang_name} → {out_lang_name} ...",
                                   expanded=True) as status:
                        if is_text:
                            st.write("Sending text to backend (translation, emotion, speech)...")
                            data = synthesize_text(token, source_text.strip(),
                                                   src_api, out_api, gender)
                        else:
                            fname = getattr(recorded, "name", None) or "input.wav"
                            _asr0 = (st.session_state.get("bk_asr_transcript", "") or "").strip()
                            _edited = (voice_text or "").strip()
                            if _edited and _edited != _asr0:
                                # Transcript was corrected -> translate the TEXT (text emotion).
                                st.write("Translating your edited transcript...")
                                data = synthesize_text(token, _edited, src_api, out_api, gender)
                            else:
                                # Unchanged -> full speech pipeline (keeps vocal-tone emotion).
                                st.write("Sending speech to backend (ASR, vocal-tone emotion, "
                                         "translation, speech)...")
                                data = translate_speech(token, recorded.getvalue(), fname,
                                                        src_api, out_api, gender)
                                if (data or {}).get("input_quality_warning"):
                                    st.warning(data["input_quality_warning"])
                        status.update(label=f"Complete! (Total: "
                                      f"{time.perf_counter() - started:.2f}s)",
                                      state="complete", expanded=False)
                    # Cache the translation + emotion so a later voice-only change is instant.
                    st.session_state["trans_cache"][cache_key] = {
                        "output_text": data.get("output_text", ""),
                        "input_text": data.get("input_text", ""),
                        "emotion": data.get("emotion") or {
                            "label": str(data.get("emotion_label", "neutral")),
                            "confidence": float(data.get("emotion_confidence") or 0.0),
                            "avd": {}, "suppress_tags": False},
                    }
            except Exception as err:  # noqa: BLE001
                st.error(f"The pipeline could not finish after "
                         f"{time.perf_counter() - started:.0f}s "
                         f"({type(err).__name__}): {err or 'no message'}")
                st.stop()
            st.session_state["result"] = {
                "out_lang_name": out_lang_name,
                "in_lang_name": in_lang_name,
                "input_text": data.get("input_text", ""),
                "input_warning": data.get("input_quality_warning", ""),
                "output_text": data.get("output_text", ""),
                "emotion_label": str(data.get("emotion_label", "neutral")),
                "emotion_conf": float(data.get("emotion_confidence") or 0.0),
                "audio": data.get("audio"),
                "voice": not is_text,
            }

    result = st.session_state.get("result")
    if result:
        emo_label = result["emotion_label"]
        conf = result["emotion_conf"]
        # Emotion badge + AVD bars
        st.markdown(_emo_badge_html(emo_label, conf), unsafe_allow_html=True)
        _avd = (st.session_state["trans_cache"].get(
            next(iter(st.session_state["trans_cache"]), None), {})
            .get("emotion", {}).get("avd")) or {}
        if _avd:
            st.markdown(_avd_bars_html(_avd), unsafe_allow_html=True)
        st.caption("🔊 Vocal tone + text emotion fused." if result.get("voice")
                   else "📝 Emotion from text. Use voice input to also capture vocal tone.")
        # Translation card with emotion-tinted left border
        _border_color = _EMO_BORDER.get(emo_label.lower(), "#2a2a2a")
        st.markdown(
            f'<div style="border-left:4px solid {_border_color};padding:10px 14px;'
            f'border-radius:4px;margin:8px 0;font-size:1.05em">'
            f'<strong>{result["out_lang_name"]}</strong><br>{result["output_text"]}</div>',
            unsafe_allow_html=True)
        if result.get("audio"):
            _ext = "wav" if result["audio"][:4] == b"RIFF" else "mp3"
            _acol, _dcol = st.columns([3, 1])
            _acol.audio(result["audio"], format=mime_for(result["audio"]), autoplay=False)
            _dcol.download_button("⬇️ Save", data=result["audio"],
                file_name=f"sts_{result.get('out_lang_name', 'out').lower()}.{_ext}",
                mime=mime_for(result["audio"]), use_container_width=True)


