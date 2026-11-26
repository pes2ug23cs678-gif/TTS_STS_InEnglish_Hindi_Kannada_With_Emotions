#!/usr/bin/env python3
"""Makes backup MP3 clips for the demo, using the same Edge voices as the live system.
Input: a text file with one line per clip, in the form
    kn | ಶುಭೋದಯ.
    hi | सुप्रभात।
The language is kn (Kannada) or hi (Hindi). Blank lines and lines starting with # are skipped.
Output: backup_audio/clip_01_kn.mp3, clip_02_hi.mp3, ... and backup_audio/clips.txt, which lists
each clip with its text, so the files can be checked later.
Run from the project folder:
  python make_backup_audio.py demo_lines.txt
  python make_backup_audio.py demo_lines.txt --gender male
Requires internet access, because Edge TTS runs on Microsoft's service.
"""
import argparse
import asyncio
import pathlib
import sys

import edge_tts

VOICES = {
    ("kn", "female"): "kn-IN-SapnaNeural",
    ("kn", "male"): "kn-IN-GaganNeural",
    ("hi", "female"): "hi-IN-SwaraNeural",
    ("hi", "male"): "hi-IN-MadhurNeural",
}
TIMEOUT_SEC = 60

def read_lines(path):
    clips = []
    for raw in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "|" not in line:
            raise ValueError(f"Line must look like 'kn | text': {line}")
        lang, text = (part.strip() for part in line.split("|", 1))
        if lang not in ("kn", "hi"):
            raise ValueError(f"Language must be kn or hi: {line}")
        clips.append((lang, text))
    return clips

async def make_clip(text, voice, out_path):
    communicator = edge_tts.Communicate(text=text, voice=voice)
    async def collect():
        parts = []
        async for chunk in communicator.stream():
            if chunk["type"] == "audio":
                parts.append(chunk["data"])
        return parts
    parts = await asyncio.wait_for(collect(), timeout=TIMEOUT_SEC)
    if not parts:
        raise RuntimeError("Edge returned no audio")
    out_path.write_bytes(b"".join(parts))

async def main_async(args):
    clips = read_lines(args.lines)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(exist_ok=True)
    index = []
    failed = 0
    for number, (lang, text) in enumerate(clips, start=1):
        name = f"clip_{number:02d}_{lang}.mp3"
        voice = VOICES[(lang, args.gender)]
        try:
            await make_clip(text, voice, out_dir / name)
            print(f"OK    {name}  {text[:50]}")
            index.append(f"{name} | {text}")
        except Exception as err:  # noqa: BLE001 - report each failed clip and carry on
            failed += 1
            print(f"FAIL  {name}  {type(err).__name__}: {err}")
    (out_dir / "clips.txt").write_text("\n".join(index) + "\n", encoding="utf-8")
    print(f"\n{len(index)} clip(s) written to {out_dir}, {failed} failed.")
    return 0 if failed == 0 else 1

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lines", help="text file with one 'kn | text' or 'hi | text' line per clip")
    ap.add_argument("--gender", choices=["female", "male"], default="female")
    ap.add_argument("--out", default="backup_audio")
    args = ap.parse_args()
    return asyncio.run(main_async(args))

if __name__ == "__main__":
    sys.exit(main())
