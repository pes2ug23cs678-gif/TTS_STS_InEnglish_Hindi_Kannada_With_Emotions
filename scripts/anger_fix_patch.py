import py_compile, shutil

P = "tts/parler_engine.py"
s = open(P, encoding="utf-8").read()
orig = s

# ============================================================
# 1. Strengthen EMOTION_STYLE for anger (and angry alias)
# ============================================================
OLD_ANGER = '"anger": "forceful, intense and emotionally charged",'
NEW_ANGER = '"anger": "speaks with a loud, sharp, and highly agitated voice, with emphatic stress and clear frustration",'
assert s.count(OLD_ANGER) == 1, f"anger anchor not found ({s.count(OLD_ANGER)})"
s = s.replace(OLD_ANGER, NEW_ANGER, 1)

OLD_ANGRY = '"angry": "forceful, intense and emotionally charged",'
NEW_ANGRY = '"angry": "speaks with a loud, sharp, and highly agitated voice, with emphatic stress and clear frustration",'
assert s.count(OLD_ANGRY) == 1, f"angry anchor not found ({s.count(OLD_ANGRY)})"
s = s.replace(OLD_ANGRY, NEW_ANGRY, 1)

# ============================================================
# 2. Per-emotion arousal floor (inject right after arousal is read from avd)
# ============================================================
AROUSAL_ANCHOR = '        arousal = float(avd.get("arousal", 0.0) or 0.0)'
AROUSAL_FLOOR = (
    "\n"
    "        # Per-emotion arousal floor: emotion2vec blended 50/50 with neutral text\n"
    "        # yields low arousal (e.g. 0.46 for anger). These emotions intrinsically\n"
    "        # have high arousal, so we enforce a minimum so pace/pitch stay energetic.\n"
    "        _AROUSAL_FLOORS = {\"anger\": 0.72, \"angry\": 0.72, \"excited\": 0.78,\n"
    "                           \"fear\": 0.65, \"surprise\": 0.65}\n"
    "        if not suppress and label in _AROUSAL_FLOORS:\n"
    "            arousal = max(arousal, _AROUSAL_FLOORS[label])"
)
assert s.count(AROUSAL_ANCHOR) == 1, f"arousal anchor not unique ({s.count(AROUSAL_ANCHOR)})"
s = s.replace(AROUSAL_ANCHOR, AROUSAL_ANCHOR + AROUSAL_FLOOR, 1)

# ============================================================
# 3. Add a stronger pace tier for very high arousal (> 0.65)
# ============================================================
OLD_PACE = (
    "            if arousal > 0.38:\n"
    "                    pace = \"a fast pace and high pitch\"\n"
    "                elif arousal < -0.25:\n"
    "                    pace = \"a slow pace and low pitch\"\n"
    "                else:\n"
    "                    pace = \"a moderate speed and pitch\""
)
# Try both indentation styles (the sed output uses 20-space indent inside if not pace:)
OLD_PACE2 = (
    "                if arousal > 0.38:\n"
    "                    pace = \"a fast pace and high pitch\"\n"
    "                elif arousal < -0.25:\n"
    "                    pace = \"a slow pace and low pitch\"\n"
    "                else:\n"
    "                    pace = \"a moderate speed and pitch\""
)
NEW_PACE2 = (
    "                if arousal > 0.65:\n"
    "                    pace = \"a rapid, urgent pace with elevated pitch\"\n"
    "                elif arousal > 0.38:\n"
    "                    pace = \"a fast pace and high pitch\"\n"
    "                elif arousal < -0.25:\n"
    "                    pace = \"a slow pace and low pitch\"\n"
    "                else:\n"
    "                    pace = \"a moderate speed and pitch\""
)

if s.count(OLD_PACE2) == 1:
    s = s.replace(OLD_PACE2, NEW_PACE2, 1)
    print("pace tier: replaced (style 2)")
elif s.count(OLD_PACE) == 1:
    NEW_PACE = OLD_PACE.replace(
        "if arousal > 0.38:\n"
        "                    pace = \"a fast pace and high pitch\"",
        "if arousal > 0.65:\n"
        "                    pace = \"a rapid, urgent pace with elevated pitch\"\n"
        "                elif arousal > 0.38:\n"
        "                    pace = \"a fast pace and high pitch\""
    )
    s = s.replace(OLD_PACE, NEW_PACE, 1)
    print("pace tier: replaced (style 1)")
else:
    print("WARNING: pace anchor not found -- skipping pace tier (fix 1+2 still applied)")

# ============================================================
# Write + compile
# ============================================================
if s == orig:
    print(">>> No change (already patched).")
else:
    shutil.copy(P, P + ".bak_angerfix")
    open(P, "w", encoding="utf-8").write(s)
    try:
        py_compile.compile(P, doraise=True)
        print(">>> tts/parler_engine.py patched OK. Backup:", P + ".bak_angerfix")
    except Exception as e:
        shutil.copy(P + ".bak_angerfix", P)
        print(">>> Compile FAILED, reverted:", e)

