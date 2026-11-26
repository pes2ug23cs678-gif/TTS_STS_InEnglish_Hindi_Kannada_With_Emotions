from datetime import datetime
import enum

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey,
    Enum,
    Text,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class InputLang(str, enum.Enum):
    hindi = "hindi"
    kannada = "kannada"
    english = "english"


class OutputLang(str, enum.Enum):
    hindi = "hindi"
    kannada = "kannada"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    translations = relationship(
        "Translation",
        back_populates="user",
        cascade="all, delete-orphan",
    )


class Translation(Base):
    __tablename__ = "translations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Stored as plain strings ("english"/"hindi"/"kannada") so any of the three
    # languages can be a source or a target without a Postgres ENUM migration.
    input_lang = Column(String, nullable=False)
    output_lang = Column(String, nullable=False)

    input_text = Column(Text, nullable=False)
    input_audio_path = Column(String, nullable=False)

    output_text = Column(Text, nullable=False)
    output_audio_path = Column(String, nullable=False)

    emotion_label = Column(String, nullable=True)
    emotion_confidence = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="translations")


class CorpusText(Base):
    __tablename__ = "corpus_texts"

    id = Column(Integer, primary_key=True, index=True)
    language = Column(String, nullable=False)
    split = Column(String, nullable=False)
    source_id = Column(String, nullable=True)
    text = Column(Text, nullable=False)


class AudioSample(Base):
    __tablename__ = "audio_samples"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    path = Column(String, nullable=False)
    split = Column(String, nullable=False)
    language = Column(String, nullable=False)
    source = Column(String, nullable=True)
    speaker = Column(String, nullable=True)
    gender = Column(String, nullable=True)
    emotion = Column(String, nullable=True)
