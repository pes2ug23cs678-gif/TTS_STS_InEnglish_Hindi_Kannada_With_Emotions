import py_compile, shutil

P = "asr/whisper_engine.py"
s = open(P, encoding="utf-8").read()
orig = s

# ============================================================
# 1. Raise script-fraction floor 0.15 → 0.20
#    Real Kannada/Hindi speech well exceeds 20% target-script.
#    Hallucinated English output on Kannada input won't.
# ============================================================
OLD1 = "_HG_MIN_TARGET_FRAC = 0.15"
NEW1 = "_HG_MIN_TARGET_FRAC = 0.20"
assert s.count(OLD1) == 1, f"script-frac anchor not found ({s.count(OLD1)})"
s = s.replace(OLD1, NEW1, 1)
print("1. Script fraction 0.15 → 0.20 OK")

# ============================================================
# 2. Tighter Kannada decode thresholds (compression_ratio 2.4→2.0,
#    log_prob -1.0→-0.7) to catch loops that slip under the default.
# ============================================================
OLD2 = (
    "            initial_prompt=_initial_prompt(language),\n"
    "        )\n"
    "        if self._should_chunk(_model, audio_arr):"
)
NEW2 = (
    "            initial_prompt=_initial_prompt(language),\n"
    "        )\n"
    "        # Kannada is lower-resource in Whisper's training data; tighten thresholds\n"
    "        # to catch hallucination loops that slip under the universal defaults.\n"
    "        if str(language or \"\").lower().startswith(\"kn\"):\n"
    "            _decode_kw[\"compression_ratio_threshold\"] = 2.0\n"
    "            _decode_kw[\"log_prob_threshold\"] = -0.7\n"
    "        if self._should_chunk(_model, audio_arr):"
)
assert s.count(OLD2) == 1, f"kn-threshold anchor not found ({s.count(OLD2)})"
s = s.replace(OLD2, NEW2, 1)
print("2. Kannada decode thresholds OK")

# ============================================================
# 3. Make hallucination guard active: blank the transcript when
#    the guard fires so downstream NMT/TTS never sees bad text.
# ============================================================
OLD3 = (
    "                q[\"hallucination\"] = True\n"
    "        if q:"
)
NEW3 = (
    "                q[\"hallucination\"] = True\n"
    "                transcript = \"\"  # don't pass hallucinated text to NMT/TTS\n"
    "        if q:"
)
assert s.count(OLD3) == 1, f"hallucination-blank anchor not found ({s.count(OLD3)})"
s = s.replace(OLD3, NEW3, 1)
print("3. Hallucination active-blank OK")

# ============================================================
# Write + compile
# ============================================================
if s == orig:
    print(">>> No change (already patched).")
else:
    shutil.copy(P, P + ".bak_knhall")
    open(P, "w", encoding="utf-8").write(s)
    try:
        py_compile.compile(P, doraise=True)
        print(">>> asr/whisper_engine.py patched OK. Backup:", P + ".bak_knhall")
    except Exception as e:
        shutil.copy(P + ".bak_knhall", P)
        print(">>> Compile FAILED, reverted:", e)
