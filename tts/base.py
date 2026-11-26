"base.py"
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
class BaseTTSEngine(ABC):
    """
    Abstract Strategy interface for all Speech Synthesis engines.
    """
    @abstractmethod
    async def synthesize(self, text: str, lang_code: str,
    emotion_meta: Optional[Dict[str, Any]] = None,
    voice: Optional[str] = None) -> bytes:
        "Sees for the synthesis case over here!!!"
    @abstractmethod
    def is_language_supported(self, lang_code: str) -> bool:
        "Sees if the given language is supported or not"
