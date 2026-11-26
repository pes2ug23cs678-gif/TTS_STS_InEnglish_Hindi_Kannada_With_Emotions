"""Text-to-speech engine package: base interface, engines, and the dispatcher."""
# tts/__init__.py
from .base import BaseTTSEngine
from .edge_engine import EdgeTTSEngine
from .elevenlabs_engine import ElevenLabsTTSEngine
from .dispatcher import TTSDispatcher

__all__ = ["BaseTTSEngine", "EdgeTTSEngine", "ElevenLabsTTSEngine", "TTSDispatcher"]
