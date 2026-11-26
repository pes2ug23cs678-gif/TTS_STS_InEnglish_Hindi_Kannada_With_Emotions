"""Applies target-language surface replacement rules loaded from rules.json."""
import json
import re
from pathlib import Path
from typing import Dict, Any
class RulesEngine:
    """Loads rules.json once and applies its surface/source replacement rules."""
    def __init__(self, rules_path: str = "rules.json"):
        self.rules_path = Path(rules_path)
        self.rules: Dict[str, Any] = self._load_rules()
    def _load_rules(self) -> Dict[str, Any]:
        if not self.rules_path.exists():
            return {}
        with open(self.rules_path, "r", encoding="utf-8") as f:
            return json.load(f)
    def apply_conversational_surface_rules(self, text: str, tgt_code: str) -> str:
        """
        Applies target-language surface replacements defined in rules.json
        """
        if not text:
            return ""
        surface_rules = (
            self.rules.get("surface_rules", {})
            .get(tgt_code, {})
            .get("replacements", [])
        )
        refined = text
        for rule in surface_rules:
            if isinstance(rule, list) and len(rule) >= 2:
                pattern, repl = rule[0], rule[1]
                refined = re.sub(pattern, repl, refined)
        return re.sub(r"\s+", " ", refined).strip()

    def apply_source_rules(self, text: str) -> str:
        """Applies source-language replacements defined in rules.json."""
        for rule in self.rules.get("source_rules", {}).get("replacements", []):
            if isinstance(rule, list) and len(rule) >= 2:
                text = re.sub(rule[0], rule[1], text)
        return text


# Backward compatibility alias if other scripts import the function directly:
_default_engine = None
def apply_conversational_surface_rules(text: str, tgt_code: str, rules_path: str = "rules.json") -> str:
    """Module-level helper that lazily builds and reuses one default RulesEngine."""
    global _default_engine  # pylint: disable=global-statement
    if _default_engine is None:
        _default_engine = RulesEngine(rules_path)
    return _default_engine.apply_conversational_surface_rules(text, tgt_code)
