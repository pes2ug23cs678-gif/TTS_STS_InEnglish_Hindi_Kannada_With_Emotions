import py_compile, shutil

# --- 1) pipeline.py: add emotion debug log after sensitivity guard ---
PP = "pipeline.py"
ps = open(PP, encoding="utf-8").read()
PIPE_ANCHOR = "        emotion_meta = _apply_sensitivity_guard(emotion_meta)"
LOG_LINE = (
    "\n"
    "        logger.info(\"[EMO] det=%s text=%s audio=%s arousal=%.2f conf=%.2f\",\n"
    "                    emotion_meta.get(\"detected_emo\", \"?\"),\n"
    "                    emotion_meta.get(\"text_label\", \"?\"),\n"
    "                    emotion_meta.get(\"audio_label\", \"?\"),\n"
    "                    float((emotion_meta.get(\"avd\") or {}).get(\"arousal\", 0.0)),\n"
    "                    float(emotion_meta.get(\"confidence\", 0.0)))"
)
assert ps.count(PIPE_ANCHOR) == 1, "pipeline anchor not unique/found"
if '[EMO]' not in ps:
    shutil.copy(PP, PP + ".bak_emolog")
    ps2 = ps.replace(PIPE_ANCHOR, PIPE_ANCHOR + LOG_LINE, 1)
    open(PP, "w", encoding="utf-8").write(ps2)
    try:
        py_compile.compile(PP, doraise=True)
        print("pipeline.py patched OK")
    except Exception as e:
        shutil.copy(PP + ".bak_emolog", PP)
        print("pipeline.py FAILED, reverted:", e)
else:
    print("pipeline.py already patched")

# --- 2) verifier.py: fix "X. to [infinitive]" reported-speech split ---
VP = "verifier.py"
vs = open(VP, encoding="utf-8").read()
VER_ANCHOR = (
    "        # Negative-motion: Kannada ಯಾರು + neg verb is existential (no one), not interrogative.\n"
    "        out = re.sub(\n"
    "            r\"(?i)\\bWho (?:doesn't|didn't|won't|wouldn't) \"\n"
    "            r\"(?:move[sd]?|stir(?:red)?|budge[d]?|step(?:ped)?)[.!?]?\",\n"
    "            \"No one moved.\", out)"
)
SPLIT_FIX = (
    "\n"
    "        # Reported-speech split: IndicTrans2 sometimes breaks 'X. to do Y' at the\n"
    "        # subordinate clause boundary (Kannada ಎಂದು ending). Rejoin as 'X, to do Y'.\n"
    "        out = re.sub(r'(?<=[a-z])\\.\\s+(to\\s+[a-z])',\n"
    "                     lambda m: ', ' + m.group(1), out)\n"
    "        # Capitalize first letter of the whole string when MT left it lowercase.\n"
    "        if out and out[0].islower():\n"
    "            out = out[0].upper() + out[1:]"
)
assert vs.count(VER_ANCHOR) == 1, "verifier anchor not unique/found"
if 'Reported-speech split' not in vs:
    shutil.copy(VP, VP + ".bak_splitfix")
    vs2 = vs.replace(VER_ANCHOR, VER_ANCHOR + SPLIT_FIX, 1)
    open(VP, "w", encoding="utf-8").write(vs2)
    try:
        py_compile.compile(VP, doraise=True)
        print("verifier.py patched OK")
    except Exception as e:
        shutil.copy(VP + ".bak_splitfix", VP)
        print("verifier.py FAILED, reverted:", e)
else:
    print("verifier.py already patched")
