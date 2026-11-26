"""
Diagnose why Hindi gender normalization is not firing.
Run from ~/Python312_Project:
    python ~/scratchpad/diag_gender.py
"""
import sys, os, re as _rg
sys.path.insert(0, os.path.expanduser("~/Python312_Project"))

SAMPLE = "पिता, मैं अपने बीमार दादा की देखभाल करूँगा। बस मुझे पाँच दिन दें।"

# ── 1. Unicode breakdown of the ूX cluster in the sample ──────────────────
print("=== 1. Unicode codepoints in sample text ===")
for i, ch in enumerate(SAMPLE):
    if 0x0900 <= ord(ch) <= 0x097F:
        print(f"  pos {i:3d}: U+{ord(ch):04X}  {ch!r}")

# ── 2. Does the regex in parler_engine.py actually match? ─────────────────
print("\n=== 2. Regex match test ===")
pat_an = r"ूंगा\b"   # anusvara   U+0902
pat_ch = r"ूँगा\b"   # chandrabindu U+0901

m1 = _rg.search(pat_an, SAMPLE)
m2 = _rg.search(pat_ch, SAMPLE)
print(f"  anusvara   pattern matches: {bool(m1)}")
print(f"  chandrabindu pattern matches: {bool(m2)}")

if m2:
    after = _rg.sub(pat_ch, "ूँगी", SAMPLE)
    print(f"  → after female fix: {after}")

# ── 3. What codepoints are in the REGEX PATTERNS as saved in parler_engine.py ──
P = "tts/parler_engine.py"
try:
    src = open(P, encoding="utf-8").read()
    # Find the normalization block
    idx = src.find("ूंगा")
    idx2 = src.find("ूँगा")
    print(f"\n=== 3. parler_engine.py pattern codepoints ===")
    if idx >= 0:
        ctx = src[idx:idx+6]
        print(f"  anusvara block found at byte {idx}:")
        for ch in ctx:
            print(f"    U+{ord(ch):04X}  {ch!r}")
    else:
        print("  anusvara pattern NOT found in file")
    if idx2 >= 0:
        ctx2 = src[idx2:idx2+6]
        print(f"  chandrabindu block found at byte {idx2}:")
        for ch in ctx2:
            print(f"    U+{ord(ch):04X}  {ch!r}")
    else:
        print("  chandrabindu pattern NOT found in file")
except FileNotFoundError:
    print("  parler_engine.py not found — run from ~/Python312_Project")

# ── 4. Actual MT output codepoints ───────────────────────────────────────
print("\n=== 4. Actual MT output for 'Father, I will take care...' ===")
try:
    from mt.indic_trans_engine import IndicTransEngine
    e = IndicTransEngine()
    result = e.translate(
        "Father, I will take care of my ailing grandfather. Just give me five days.",
        src="en", tgt="hi"
    )
    print(f"  MT output: {result!r}")
    for i, ch in enumerate(result):
        if 0x0900 <= ord(ch) <= 0x097F:
            print(f"  pos {i:3d}: U+{ord(ch):04X}  {ch!r}")
    # Try normalization
    out = _rg.sub(r"ूंगा\b", "ूंगी", result)
    out = _rg.sub(r"ूँगा\b", "ूँगी", out)
    print(f"  After female fix: {out!r}")
    if out == result:
        print("  >>> PATTERN DID NOT MATCH MT OUTPUT — codepoint mismatch confirmed")
    else:
        print("  >>> Pattern matched OK")
except Exception as ex:
    print(f"  MT import failed: {ex}")

# ── 5. Check lang_code path ───────────────────────────────────────────────
print("\n=== 5. lang_code check test ===")
for lc in ["hi", "hin", "hi_IN", "HI", "Hindi"]:
    fires = str(lc).lower() == "hi"
    print(f"  lang_code={lc!r:12s} -> normalization fires: {fires}")
print()
print("If 'hin' or 'hi_IN' is what the pipeline passes, normalization is skipped entirely.")

