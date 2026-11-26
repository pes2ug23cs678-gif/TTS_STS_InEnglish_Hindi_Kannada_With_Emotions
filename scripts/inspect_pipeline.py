#!/usr/bin/env python3
"""Find the pipeline entry point and every place the speech engine is called.
Reads the Python files as text (it does not import them, so no models are loaded).
Usage:
  python inspect_pipeline.py                 # scan the current folder
  python inspect_pipeline.py path/to/project # scan another folder
It reports:
  1. classes whose name contains Pipeline or Dispatcher, with their methods and signatures
  2. async and sync functions that look like entry points (process, run, handle, synthesize)
  3. every call to a speech engine (synthesize, tts, dispatch) and to the translator and
     emotion functions, with file, line, the enclosing function, and the source line
"""
import ast
import pathlib
import sys

SKIP_DIRS = {"__pycache__", ".git", "venv", ".venv"}
ENTRY_NAMES = {"process", "run", "handle", "synthesize", "process_text", "process_audio"}
CALL_HINTS = ("synthesize", "tts", "dispatch", "translator_fn", "emotion_fn", "translate_text")


def py_files(root):
    for path in sorted(root.rglob("*.py")):
        if any(part in SKIP_DIRS or part.startswith("env") or part.endswith("_env") for part in path.parts):
            continue
        yield path


def signature(node):
    args = [a.arg for a in node.args.args]
    prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    return f"{prefix}{node.name}({', '.join(args)})"


def call_name(call):
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def report_file(path, root):
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        print(f"\n{path.relative_to(root)}: could not parse ({exc})")
        return

    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and ("Pipeline" in node.name or "Dispatcher" in node.name):
            found.append(f"  class {node.name}")
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    marker = "  <- entry point" if item.name in ENTRY_NAMES else ""
                    found.append(f"      {signature(item)} [line {item.lineno}]{marker}")

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in ENTRY_NAMES:
            if not any(f"{signature(node)} [line" in f for f in found):
                found.append(f"  top-level {signature(node)} [line {node.lineno}]")

    # Map each call to the function it sits in, so the report says where it is called from.
    enclosing = {}
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(fn):
                if isinstance(child, ast.Call):
                    enclosing.setdefault(id(child), fn.name)

    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = call_name(node)
            if any(hint in name for hint in CALL_HINTS) or name in ("translator_fn", "emotion_fn"):
                where = enclosing.get(id(node), "<module>")
                full = ast.get_source_segment(text, node) or lines[node.lineno - 1].strip()
                full = " ".join(part.strip() for part in full.splitlines())
                calls.append(f"  line {node.lineno:>4}  in {where}(): {full}")

    if not found and not calls:
        return
    print(f"\n== {path.relative_to(root)}")
    for line in found:
        print(line)
    if calls:
        print("  speech/translation/emotion calls:")
        for line in sorted(set(calls), key=lambda s: int(s.split()[1])):
            print(line)


def main():
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    if not root.exists():
        print(f"Folder not found: {root}")
        return 1
    print(f"Scanning {root}")
    for path in py_files(root):
        report_file(path, root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
