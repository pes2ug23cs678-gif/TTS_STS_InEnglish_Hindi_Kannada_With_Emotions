import asyncio
import base64
import os
import threading
import uuid
import traceback
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from backend_api.auth import (
    create_token,
    get_current_user,
    hash_password,
    verify_password,
)
from backend_api.db import engine, get_db
from backend_api.models import Base, Translation, User
from backend_api.schemas import Token, UserCreate, UserOut
from pipeline import PipelineBusy

# Supported languages and their internal 2-letter codes.
LANG_NAME_TO_CODE = {"english": "en", "hindi": "hi", "kannada": "kn"}
LANG_CHOICES_MSG = "must be english, hindi or kannada"


def _auto_tts_engine(pipeline) -> str:
    """Select the voice engine automatically (no client choice), like the Streamlit app:
    GPU -> Parler-TTS; CPU -> Edge-TTS (Parler is far too slow on CPU). Set
    STS_FORCE_TTS_ENGINE to override. The dispatcher still falls back on failure."""
    import os
    import torch
    forced = os.getenv("STS_FORCE_TTS_ENGINE", "").strip().lower()
    if forced in ("parler", "edge", "elevenlabs"):
        return forced
    return "parler" if torch.cuda.is_available() else "edge"

app = FastAPI(title="Emotion-Aware Speech Translation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = PROJECT_ROOT / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

MAX_AUDIO_SIZE = 10 * 1024 * 1024
ALLOWED_AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".mp4"
}

def _preload_models():
    """Loads the speech, translation and emotion models in the background, so the first request is not slow."""
    try:
        from ai_runtime import get_ai_runtime

        get_ai_runtime()
        print("[WARMUP] models loaded; requests will now be served without the load delay", flush=True)
    except Exception:  # noqa: BLE001 - a failed warm-up is reported, not fatal
        traceback.print_exc()

WARMUP_SENTENCE = "Good morning guys."
WARMUP_LANGUAGES = ("kn", "hi")
_warmup_task = None

async def _warm_full_pipeline():
    """Runs one short sentence per language through the whole pipeline once, at start-up. Output is discarded."""
    try:
        from ai_runtime import get_ai_runtime

        pipeline, _ = await asyncio.to_thread(get_ai_runtime)
        for lang in WARMUP_LANGUAGES:
            await pipeline.process(WARMUP_SENTENCE, "female", src_lang="en",
                                   tgt_lang=lang, preferred_tts_engine="edge")
        print(f"[WARMUP] full pipeline warm-up done for {', '.join(WARMUP_LANGUAGES)}", flush=True)
    except Exception:  # noqa: BLE001 - a failed warm-up is reported, not fatal
        traceback.print_exc()

@app.on_event("startup")
async def create_tables():
    global _warmup_task
    Base.metadata.create_all(bind=engine)
    # Print the readiness report (token / IndicProcessor / GPU) before warm-up, so a
    # missing token or processor is reported clearly instead of as a warm-up traceback.
    try:
        from preflight import report
        report()
    except Exception:  # noqa: BLE001 - never let preflight stop the server
        pass
    # Set STS_PRELOAD_MODELS=0 to skip the warm-up (for example, when running tests).
    if os.getenv("STS_PRELOAD_MODELS", "1") == "1":
        threading.Thread(target=_preload_models, daemon=True).start()
        # Runs on the server's own event loop, so it shares the translation lock with real requests
        _warmup_task = asyncio.create_task(_warm_full_pipeline())

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/register", response_model=UserOut, status_code=201)
def register(payload: UserCreate, db: Session = Depends(get_db)):
    email = str(payload.email).strip().lower()

    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(status_code=409, detail="Email is already registered")

    user = User(
        name=payload.name.strip(),
        email=email,
        password_hash=hash_password(payload.password),
    )

    try:
        db.add(user)
        db.commit()
        db.refresh(user)
    except Exception:
        db.rollback()
        raise HTTPException(status_code=500, detail="Could not create account")

    return user

@app.post("/login", response_model=Token)
def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    email = form.username.strip().lower()
    user = db.query(User).filter(User.email == email).first()

    if not user or not verify_password(form.password, user.password_hash):
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return {"access_token": create_token(user.id), "token_type": "bearer"}

class AudioInput:
    def __init__(self, filename: str, data: bytes):
        self.name = filename
        self._data = data

    def getvalue(self):
        return self._data

def get_result_value(value, key, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _emotion_payload(emotion) -> dict:
    """Full emotion detail for responses, so a client can re-synthesize (voice-only change)
    WITHOUT re-running translation/emotion."""
    avd = get_result_value(emotion, "avd", {}) or {}
    label = str(get_result_value(emotion, "label",
                get_result_value(emotion, "detected_emo", "neutral")))
    conf = float(get_result_value(emotion, "conf",
                 get_result_value(emotion, "confidence", 0.0)) or 0.0)
    return {
        "label": label,
        "confidence": conf,
        "avd": {"arousal": float(avd.get("arousal", 0.0) or 0.0),
                "valence": float(avd.get("valence", 0.0) or 0.0),
                "dominance": float(avd.get("dominance", 0.0) or 0.0)},
        "suppress_tags": bool(get_result_value(emotion, "suppress_tags", False)),
        "source": str(get_result_value(emotion, "emotion_source", "text")),
        "prosody": get_result_value(emotion, "prosody", None),
    }

@app.post("/translate")
async def translate(
    file: UploadFile = File(...),
    output_lang: str = Form(...),
    input_lang: str = Form("english"),
    gender: str = Form("female"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    extension = Path(file.filename or "").suffix.lower()

    if extension not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported audio file format")

    if input_lang not in LANG_NAME_TO_CODE:
        raise HTTPException(status_code=400, detail=f"input_lang {LANG_CHOICES_MSG}")

    if output_lang not in LANG_NAME_TO_CODE:
        raise HTTPException(status_code=400, detail=f"output_lang {LANG_CHOICES_MSG}")

    if input_lang == output_lang:
        raise HTTPException(status_code=400, detail="output_lang must differ from input_lang")

    if gender not in {"female", "male"}:
        raise HTTPException(
            status_code=400,
            detail="gender must be female or male",
        )

    audio_data = await file.read()

    if not audio_data:
        raise HTTPException(status_code=400, detail="Uploaded audio is empty")

    if len(audio_data) > MAX_AUDIO_SIZE:
        raise HTTPException(status_code=413, detail="Audio file exceeds 10 MB")

    source_path = AUDIO_DIR / f"{uuid.uuid4().hex}{extension}"
    source_code = LANG_NAME_TO_CODE[input_lang]
    target_code = LANG_NAME_TO_CODE[output_lang]

    try:
        source_path.write_bytes(audio_data)

        from ai_runtime import get_ai_runtime
        from asr.whisper_engine import transcribe_speech

        pipeline, engines = await asyncio.to_thread(get_ai_runtime)

        transcript = await asyncio.to_thread(
            transcribe_speech,
            engines["asr"],
            audio_data,
            source_code,
        )

        if not transcript or not transcript.strip():
            raise HTTPException(
                status_code=422,
                detail="No speech could be recognized in the audio",
            )

        # Script-level guard (mirrors app.py): reject a transcript whose script does not match
        # the selected input language.
        from validation_lang import validate_input_script
        _ok, _msg = validate_input_script(transcript, source_code)
        if not _ok:
            raise HTTPException(status_code=422, detail=_msg)

        # Capture the speaker's vocal tone from the waveform (emotion2vec) and fuse it with
        # the text emotion inside the pipeline. Best-effort: None if the engine is unavailable.
        from ai_runtime import audio_emotion_from_bytes
        audio_emotion = await asyncio.to_thread(audio_emotion_from_bytes, audio_data)
        # Continuous prosody transfer: measure the speaker's delivery from the waveform.
        from prosody import extract_prosody
        prosody = await asyncio.to_thread(extract_prosody, audio_data)

        result = await pipeline.process(
            source_text=transcript.strip(),
            gender=gender,
            src_lang=source_code,
            tgt_lang=target_code,
            preferred_tts_engine=_auto_tts_engine(pipeline),
            audio_emotion=audio_emotion,
            prosody=prosody,
        )

        output_audio = result.audio_bytes
        if not output_audio:
            raise RuntimeError("The speech synthesis pipeline returned no audio")

        audio_format = "wav" if output_audio[:4] == b"RIFF" else "mp3"
        target_path = AUDIO_DIR / f"{uuid.uuid4().hex}.{audio_format}"
        target_path.write_bytes(output_audio)

        emotion = result.emotion_meta
        emotion_label = get_result_value(emotion, "label", "neutral")
        emotion_confidence = get_result_value(
            emotion, "conf", get_result_value(emotion, "confidence", 0.0)
        )

        record = Translation(
            user_id=current_user.id,
            input_lang=input_lang,
            output_lang=output_lang,
            input_text=transcript.strip(),
            input_audio_path=str(source_path.resolve()),
            output_text=result.translated_text,
            output_audio_path=str(target_path.resolve()),
            emotion_label=str(emotion_label),
            emotion_confidence=float(emotion_confidence or 0.0),
        )

        try:
            db.add(record)
            db.commit()
            db.refresh(record)
        except Exception:
            db.rollback()
            target_path.unlink(missing_ok=True)
            raise

        return {
            # Surface the ASR input-quality warning (clipping/quiet/short/no-speech) so API
            # clients get the same feedback the Streamlit UI shows. Empty string when fine.
            "input_quality_warning": getattr(engines["asr"], "_last_quality_warning", ""),
            "id": record.id,
            "input_lang": input_lang,
            "output_lang": output_lang,
            "input_text": record.input_text,
            "output_text": record.output_text,
            "emotion_label": record.emotion_label,
            "emotion_confidence": record.emotion_confidence,
            "emotion": _emotion_payload(result.emotion_meta),
            "created_at": record.created_at,
            "input_audio_url": f"/audio/{record.id}/input",
            "output_audio_url": f"/audio/{record.id}/output",
        }

    except HTTPException:
        raise
    except PipelineBusy as exc:
        source_path.unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "5"}) from exc
    except Exception as exc:
        traceback.print_exc()
        source_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail=f"Speech translation failed: {type(exc).__name__}",
        ) from exc
    finally:
        await file.close()

MAX_TEXT_CHARS = 500

@app.post("/synthesize")
async def synthesize(
    text: str = Form(...),
    output_lang: str = Form(...),
    input_lang: str = Form("english"),
    gender: str = Form("female"),
    current_user: User = Depends(get_current_user),
):
    """Text in (any of the three languages), translated and spoken in another language.

    Nothing is saved to the database; the audio is returned in the response (base64).
    """
    clean_text = text.strip()

    if not clean_text:
        raise HTTPException(status_code=400, detail="Text is empty")

    if len(clean_text) > MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"Text exceeds {MAX_TEXT_CHARS} characters",
        )

    if input_lang not in LANG_NAME_TO_CODE:
        raise HTTPException(status_code=400, detail=f"input_lang {LANG_CHOICES_MSG}")

    if output_lang not in LANG_NAME_TO_CODE:
        raise HTTPException(status_code=400, detail=f"output_lang {LANG_CHOICES_MSG}")

    if input_lang == output_lang:
        raise HTTPException(status_code=400, detail="output_lang must differ from input_lang")

    if gender not in {"female", "male"}:
        raise HTTPException(
            status_code=400,
            detail="gender must be female or male",
        )

    source_code = LANG_NAME_TO_CODE[input_lang]
    target_code = LANG_NAME_TO_CODE[output_lang]

    # Script-level input-language guard (mirrors app.py): reject text whose script does not
    # match the selected input language -- e.g. Malayalam/Tamil/Gujarati typed under Hindi.
    from validation_lang import validate_input_script
    _ok, _msg = validate_input_script(clean_text, source_code)
    if not _ok:
        raise HTTPException(status_code=422, detail=_msg)

    try:
        from ai_runtime import get_ai_runtime

        pipeline, _ = await asyncio.to_thread(get_ai_runtime)
        result = await pipeline.process(
            source_text=clean_text,
            gender=gender,
            src_lang=source_code,
            tgt_lang=target_code,
            preferred_tts_engine=_auto_tts_engine(pipeline),
        )
    except HTTPException:
        raise
    except PipelineBusy as exc:
        raise HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "5"}) from exc
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Speech synthesis failed: {type(exc).__name__}",
        ) from exc

    if not result.audio_bytes:
        raise HTTPException(status_code=500, detail="Speech synthesis returned no audio")

    emotion = result.emotion_meta
    return {
        "input_text": clean_text,
        "output_lang": output_lang,
        "output_text": result.translated_text,
        "emotion_label": str(get_result_value(emotion, "label", "neutral")),
        "emotion_confidence": float(get_result_value(
            emotion, "conf", get_result_value(emotion, "confidence", 0.0)) or 0.0),
        "emotion": _emotion_payload(emotion),
        "audio_format": "audio/wav" if result.audio_bytes[:4] == b"RIFF" else "audio/mpeg",
        "audio_base64": base64.b64encode(result.audio_bytes).decode("ascii"),
    }

@app.post("/resynthesize")
async def resynthesize(
    text: str = Form(...),
    output_lang: str = Form(...),
    gender: str = Form("female"),
    emotion_label: str = Form("neutral"),
    arousal: float = Form(0.0),
    valence: float = Form(0.0),
    dominance: float = Form(0.0),
    suppress_tags: bool = Form(False),
    prosody_json: str = Form(""),
    current_user: User = Depends(get_current_user),
):
    """Speak ALREADY-translated text without re-translating — for a voice-only change.

    The caller passes the previously returned target text + emotion (and optional measured
    prosody); only TTS runs, so switching the voice does not repeat the translation/emotion
    work, and the source speaker's prosody is preserved.
    """
    clean = text.strip()
    if not clean:
        raise HTTPException(status_code=400, detail="Text is empty")
    if output_lang not in LANG_NAME_TO_CODE:
        raise HTTPException(status_code=400, detail=f"output_lang {LANG_CHOICES_MSG}")
    if gender not in {"female", "male"}:
        raise HTTPException(status_code=400, detail="gender must be female or male")

    code = LANG_NAME_TO_CODE[output_lang]
    emotion_meta = {
        "label": emotion_label, "detected_emo": emotion_label,
        "avd": {"arousal": arousal, "valence": valence, "dominance": dominance},
        "suppress_tags": bool(suppress_tags),
    }
    if prosody_json:
        try:
            import json as _json
            prosody = _json.loads(prosody_json)
            if isinstance(prosody, dict) and prosody:
                emotion_meta["prosody"] = prosody
        except Exception:  # noqa: BLE001 - ignore malformed prosody, synthesize without it
            pass
    try:
        from ai_runtime import get_ai_runtime
        pipeline, _ = await asyncio.to_thread(get_ai_runtime)
        audio_bytes, _tel = await pipeline._synthesize(
            clean, code, emotion_meta, gender, _auto_tts_engine(pipeline))
    except PipelineBusy as exc:
        raise HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "5"}) from exc
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500,
                            detail=f"Speech synthesis failed: {type(exc).__name__}") from exc

    if not audio_bytes:
        raise HTTPException(status_code=500, detail="Speech synthesis returned no audio")

    return {
        "output_lang": output_lang,
        "output_text": clean,
        "audio_format": "audio/wav" if audio_bytes[:4] == b"RIFF" else "audio/mpeg",
        "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
    }

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    input_lang: str = Form("english"),
    current_user: User = Depends(get_current_user),
):
    """Speech in (English/Hindi/Kannada), text out in that language. Nothing is saved."""
    extension = Path(file.filename or "").suffix.lower()

    if extension not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported audio file format")

    if input_lang not in LANG_NAME_TO_CODE:
        raise HTTPException(status_code=400, detail=f"input_lang {LANG_CHOICES_MSG}")

    audio_data = await file.read()
    await file.close()

    if not audio_data:
        raise HTTPException(status_code=400, detail="Uploaded audio is empty")

    if len(audio_data) > MAX_AUDIO_SIZE:
        raise HTTPException(status_code=413, detail="Audio file exceeds 10 MB")

    source_code = LANG_NAME_TO_CODE[input_lang]

    try:
        from ai_runtime import get_ai_runtime
        from asr.whisper_engine import transcribe_speech

        _, engines = await asyncio.to_thread(get_ai_runtime)
        transcript = await asyncio.to_thread(transcribe_speech, engines["asr"], audio_data, source_code)
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Speech recognition failed: {type(exc).__name__}",
        ) from exc

    transcript = (transcript or "").strip()
    if getattr(engines["asr"], "_last_language_mismatch", False):
        raise HTTPException(
            status_code=422,
            detail=getattr(engines["asr"], "_last_quality_warning", "Language mismatch detected."),
        )
    if not transcript:
        raise HTTPException(
            status_code=422,
            detail="No speech could be recognized in the audio",
        )

    return {"transcript": transcript,
            "input_quality_warning": getattr(engines["asr"], "_last_quality_warning", "")}

@app.get("/history")
def history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    records = (
        db.query(Translation)
        .filter(Translation.user_id == current_user.id)
        .order_by(Translation.created_at.desc())
        .all()
    )

    return [
        {
            "id": record.id,
            "input_lang": record.input_lang,
            "output_lang": record.output_lang,
            "input_text": record.input_text,
            "output_text": record.output_text,
            "emotion_label": record.emotion_label,
            "emotion_confidence": record.emotion_confidence,
            "created_at": record.created_at,
            "input_audio_url": f"/audio/{record.id}/input",
            "output_audio_url": f"/audio/{record.id}/output",
        }
        for record in records
    ]

@app.get("/audio/{translation_id}/{kind}")
def get_audio(
    translation_id: int,
    kind: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if kind not in {"input", "output"}:
        raise HTTPException(status_code=400, detail="kind must be input or output")

    record = (
        db.query(Translation)
        .filter(
            Translation.id == translation_id,
            Translation.user_id == current_user.id,
        )
        .first()
    )

    if not record:
        raise HTTPException(status_code=404, detail="Translation not found")

    stored_path = (
        record.input_audio_path if kind == "input" else record.output_audio_path
    )
    path = Path(stored_path).resolve()

    if path.parent != AUDIO_DIR.resolve() or not path.is_file():
        raise HTTPException(status_code=404, detail="Audio file not found")

    media_type = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".webm": "audio/webm",
        ".mp4": "audio/mp4",
    }.get(path.suffix.lower(), "application/octet-stream")

    return FileResponse(path, media_type=media_type, filename=path.name)

