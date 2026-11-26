from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, field_validator

from backend_api.models import InputLang, OutputLang


class UserCreate(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(min_length=8, max_length=72)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 2:
            raise ValueError("Name must contain at least 2 non-space characters")
        return value

    @field_validator("password")
    @classmethod
    def validate_password(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Password cannot contain only whitespace")
        if len(value.encode("utf-8")) > 72:
            raise ValueError("Password must not exceed 72 bytes")
        return value


class UserOut(BaseModel):
    id: int
    name: str
    email: EmailStr

    model_config = {"from_attributes": True}


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TranslationOut(BaseModel):
    id: int
    input_lang: InputLang
    output_lang: OutputLang
    input_text: str
    output_text: str
    input_audio_path: str
    output_audio_path: str
    emotion_label: Optional[str] = None
    emotion_confidence: Optional[float] = None
    created_at: datetime

    model_config = {"from_attributes": True}
