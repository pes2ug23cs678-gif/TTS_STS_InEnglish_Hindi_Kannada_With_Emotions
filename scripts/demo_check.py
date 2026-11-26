#!/usr/bin/env python3
"""Demo rehearsal: runs the demo sentences through the backend and saves the results.
For each sentence, in Kannada and in Hindi, it calls POST /synthesize and checks:
  - the request succeeds,
  - the output is in the right script,
  - the audio is returned,
  - the time taken is within the slow-run threshold.
Setup (once):
  1. Write the demo sentences in demo_sentences.txt, one per line. Lines starting with # are skipped.
  2. Set the login details in the terminal (they are not stored in any file):
       export STS_EMAIL=test2@example.com
       export STS_PASSWORD='your password'
Run with the backend running, from the combined_app folder:
  python demo_check.py
Writes to demo_check_output/: one MP3 per clip, and demo_check_results.txt.
Exits with status 1 if any clip fails, so the check can be used in a script.
"""
import base64
import os
import pathlib
import re
import sys
import time
import httpx

BASE = pathlib.Path(__file__).resolve().parent
API_URL = os.getenv("STS_API_URL", "http://127.0.0.1:8000")
SENTENCES_FILE = BASE / "demo_sentences.txt"
OUT_DIR = BASE / "demo_check_output"
SLOW_SECONDS = 15.0
TIMEOUT = httpx.Timeout(180.0, connect=10.0)

LANGS = {
    "kannada": ("kn", r"[ಀ-೿]"),   # Kannada block
    "hindi": ("hi", r"[ऀ-ॿ]"),     # Devanagari block
}

def read_sentences(path):
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines

def script_share(text, pattern):
    letters = re.findall(r"\w", text)
    if not letters:
        return 0.0
    return len(re.findall(pattern, text)) / len(letters)

def login(client):
    email = os.getenv("STS_EMAIL")
    password = os.getenv("STS_PASSWORD")
    if not email or not password:
        sys.exit("Set STS_EMAIL and STS_PASSWORD in the terminal first (see the top of this file).")
    resp = client.post("/login", data={"username": email, "password": password})
    if resp.status_code != 200:
        sys.exit(f"Login failed: {resp.text}")
    return resp.json()["access_token"]

def main():
    if not SENTENCES_FILE.exists():
        sys.exit(f"Create {SENTENCES_FILE.name} first, with one demo sentence per line.")
    sentences = read_sentences(SENTENCES_FILE)
    if not sentences:
        sys.exit(f"{SENTENCES_FILE.name} has no sentences.")

    OUT_DIR.mkdir(exist_ok=True)
    results = []
    failures = 0

    with httpx.Client(base_url=API_URL, timeout=TIMEOUT) as client:
        token = login(client)
        client.headers["Authorization"] = f"Bearer {token}"

        for number, sentence in enumerate(sentences, start=1):
            for lang_name, (code, pattern) in LANGS.items():
                tag = f"{number:02d}_{lang_name}"
                started = time.perf_counter()
                problems = []
                output_text = ""
                try:
                    resp = client.post("/synthesize", data={
                        "text": sentence, "output_lang": lang_name, "gender": "female"})
                    seconds = time.perf_counter() - started
                    if resp.status_code != 200:
                        problems.append(f"HTTP {resp.status_code}: {resp.text[:120]}")
                    else:
                        data = resp.json()
                        output_text = data.get("output_text", "")
                        audio = base64.b64decode(data.get("audio_base64", ""))
                        if not output_text.strip():
                            problems.append("empty translation")
                        elif script_share(output_text, pattern) < 0.5:
                            problems.append("wrong script")
                        if not audio:
                            problems.append("no audio")
                        else:
                            (OUT_DIR / f"clip_{tag}.mp3").write_bytes(audio)
                        if seconds > SLOW_SECONDS:
                            problems.append(f"slow ({seconds:.1f}s)")
                except Exception as err:  # noqa: BLE001 - record and carry on with the next clip
                    seconds = time.perf_counter() - started
                    problems.append(f"{type(err).__name__}: {err}")

                status = "OK" if not any(p for p in problems if not p.startswith("slow")) else "FAIL"
                if status == "FAIL":
                    failures += 1
                note = "; ".join(problems) if problems else "ok"
                print(f"{status:<4} {tag}  {seconds:6.1f}s  {note}")
                results.append(f"{status}\t{tag}\t{seconds:.1f}s\t{note}\n"
                               f"    input:  {sentence}\n    output: {output_text}")

    report = OUT_DIR.parent / "demo_check_results.txt"
    report.write_text("\n".join(results) + "\n", encoding="utf-8")
    print(f"\n{len(sentences) * len(LANGS) - failures} of {len(sentences) * len(LANGS)} clips OK."
          f" Audio in {OUT_DIR.name}/, details in {report.name}.")
    return 1 if failures else 0

if __name__ == "__main__":
    sys.exit(main())
