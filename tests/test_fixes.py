"""Starter tests for today's fixes. Run from the project folder with: pytest tests/
These tests load only the functions they check. Heavy imports (torch, soundfile,
indic libraries, config) are replaced with small stubs, so no models are needed.
"""
import ast
import importlib.util
import logging
import pathlib
import sys
import types
from typing import Any, Dict, Optional

ROOT = pathlib.Path(__file__).resolve().parent.parent

def _raise(*args, **kwargs):
    raise RuntimeError("stub: not available in tests")

def _stub(name, **attrs):
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        sys.modules[name] = mod

class _StubTranslit:
    transliterate = staticmethod(_raise)

_stub("soundfile")
_stub("indicnlp")
_stub("indicnlp.transliterate")
_stub("indicnlp.transliterate.unicode_transliterate",
      UnicodeIndicTransliterator=_StubTranslit)
_stub("indic_transliteration", transliterate=_raise,
      sanscript=types.SimpleNamespace(DEVANAGARI="x", KANNADA="x", ITRANS="x"))
_stub("indic_transliteration.sanscript", DEVANAGARI="x", KANNADA="x",
      ITRANS="x", transliterate=_raise)
_stub("config", NO_TAG_LABELS={"disgust"})

def _load(filename):
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""),
                                                  ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

verifier = _load("verifier.py")
entity_masker = _load("entity_masker.py")

def _load_guard():
    """Extract _apply_sensitivity_guard from pipeline.py without importing the whole pipeline."""
    tree = ast.parse((ROOT / "pipeline.py").read_text(encoding="utf-8"))
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "_apply_sensitivity_guard")
    namespace = {"Optional": Optional, "Dict": Dict, "Any": Any,
                 "logger": logging.getLogger("test_guard"),
                 "NO_TAG_LABELS": {"disgust"}}
    # Compile just the one guard function in isolation, so no heavy pipeline import is needed.
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "pipeline_guard", "exec"), namespace)  # pylint: disable=exec-used
    return namespace["_apply_sensitivity_guard"]

# --- verifier.py: the stutter fix must keep spaces between words ---

def test_devanagari_names_keep_space():
    out = verifier.verify_and_clean_output(
        "डेविड डेनियल के पसंदीदा फुटबॉल खिलाड़ी हैं ।", "hi")
    assert "डेविड डेनियल" in out

def test_kannada_possessive_keeps_space():
    out = verifier.verify_and_clean_output("ಡೇವಿಡ್ ಡೇನಿಯಲ್ನ ನೆಚ್ಚಿನ ಆಟಗಾರ", "kn")
    assert "ಡೇನಿಯಲ್ನ ನೆಚ್ಚಿನ" in out

def test_real_stutter_still_collapses():
    out = verifier.verify_and_clean_output("मैं मैं सच में जाना चाहता हूँ", "hi")
    assert out == "मैं सच में जाना चाहता हूँ"

# --- entity_masker.py: commas only before a conjunction ---

def test_no_comma_between_adjacent_names():
    out = entity_masker.unmask_and_restore_entities(
        "डेक्लान राइस लिकिथ के पसंदीदा खिलाड़ी हैं।", {}, "hi")
    assert out == "डेक्लान राइस लिकिथ के पसंदीदा खिलाड़ी हैं।"

def test_comma_before_conjunction_in_list():
    out = entity_masker.unmask_and_restore_entities(
        "लिकिथ और रिंगो स्टार मिलेंगे।", {}, "hi")
    assert out == "लिकिथ, और रिंगो स्टार मिलेंगे।"

def test_likith_spelling_is_aspirated():
    assert entity_masker.STATIC_ENTITY_MAP["Likith"]["hi"] == "लिकिथ"
    assert entity_masker.STATIC_ENTITY_MAP["Likith"]["kn"] == "ಲಿಕಿಥ್"

# --- pipeline.py: sensitivity guard ---

def test_guard_suppresses_tags_for_disgust():
    guard = _load_guard()
    assert guard({"detected_emo": "disgust"})["suppress_tags"] is True

def test_guard_leaves_other_labels_alone():
    guard = _load_guard()
    assert "suppress_tags" not in guard({"label": "happy"})

def test_guard_does_not_change_its_input():
    guard = _load_guard()
    original = {"label": "disgust"}
    guard(original)
    assert "suppress_tags" not in original
