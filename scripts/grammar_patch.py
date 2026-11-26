import py_compile, shutil
P = "verifier.py"
s = open(P, encoding="utf-8").read()
orig = s

# ============================================================
# 1. KN_PATCHES: city-FROM postposition fix
#    "ನಾನು ದೆಹಲಿನೊಂದಿಗೆ ಮಾತನಾಡುತ್ತಿದ್ದೇನೆ" (talking WITH)
#    -> "ನಾನು ದೆಹಲಿಯಿಂದ ಮಾತನಾಡುತ್ತಿದ್ದೇನೆ" (calling FROM)
# ============================================================
KN_ANCHOR = "    (re.compile(r'ಅದು ಕಾಣಿಸಿಕೊಳ್ಳುವವರೆಗೆ ಅದನ್ನು'), 'ಅದು ಕಾಣಿಸುವವರೆಗೆ'),\n]"
KN_NEW = (
    "    (re.compile(r'ಅದು ಕಾಣಿಸಿಕೊಳ್ಳುವವರೆಗೆ ಅದನ್ನು'), 'ಅದು ಕಾಣಿಸುವವರೆಗೆ'),\n"
    "    # City-FROM: Hindi 'se bol raha hoon' (from) misread as talking-to; fix postposition.\n"
    + r"    (re.compile(r'(\S+?)(ನೊಂದಿಗೆ|ಯೊಂದಿಗೆ)(\s+ಮಾತನಾಡುತ್ತಿದ್ದೇನೆ)'), r'\1ಯಿಂದ\3')," + "\n"
    + "]"
)
assert s.count(KN_ANCHOR) == 1, f"KN_ANCHOR not unique ({s.count(KN_ANCHOR)})"
if r"ನೊಂದಿಗೆ|ಯೊಂದಿಗೆ" not in s:
    s = s.replace(KN_ANCHOR, KN_NEW, 1)

# ============================================================
# 2. HI_PATCHES: who-doesn't-move fix
#    "कौन हिलता नहीं है" -> "कोई नहीं हिला"
# ============================================================
HI_ANCHOR = r"    (re.compile(r'(^|\s)(तस्वीर|फोटो)\s+लें(?=\s|[?.!,।]|$)'), r'\1\2 लो')," + "\n]"
HI_NEW = (
    r"    (re.compile(r'(^|\s)(तस्वीर|फोटो)\s+लें(?=\s|[?.!,।]|$)'), r'\1\2 लो')," + "\n"
    "    # Negative-motion: Kannada ಯಾರು + neg verb is existential 'no one', not interrogative.\n"
    + r"    (re.compile(r'\bकौन हिलता नहीं है\b'), 'कोई नहीं हिला')," + "\n"
    + "]"
)
assert s.count(HI_ANCHOR) == 1, f"HI_ANCHOR not unique ({s.count(HI_ANCHOR)})"
if 'कौन हिलता' not in s:
    s = s.replace(HI_ANCHOR, HI_NEW, 1)

# ============================================================
# 3. English section: self-intro + city-from + who-neg
#    "I'm talking about Suresh" -> "This is Suresh speaking"
#    "I'm talking to Delhi"     -> "I'm calling from Delhi"
#    "Who doesn't move"         -> "No one stirred."
# ============================================================
EN_ANCHOR = (
    "        if _stripped != out and _stripped:\n"
    "            out = _stripped[0].upper() + _stripped[1:]"
)
EN_ADDITION = (
    "\n"
    "        # Self-introduction: IndicTrans2 reads \"I am X speaking\" (phone intro) as\n"
    "        # \"I'm talking about X\". Rewrite when X is the whole sentence object.\n"
    "        out = re.sub(r\"I'?m talking about ([A-Z][a-z]+)(?=[.,!?]|\\Z)\",\n"
    "                     lambda m: \"This is \" + m.group(1) + \" speaking\", out)\n"
    "        # City-FROM: Hindi 'X se bol raha hoon' (from X) misread as talking-to X.\n"
    "        _CALL_CITIES = (\n"
    "            r'Delhi|Mumbai|Bangalore|Bengaluru|Chennai|Hyderabad|'\n"
    "            r'Kolkata|Pune|Ahmedabad|Jaipur|Lucknow|Mysore|Mysuru|'\n"
    "            r'Mangalore|Mangaluru|Kochi|Chandigarh')\n"
    "        out = re.sub(rf\"(?i)\\bI'?m talking to ({_CALL_CITIES})\\b\",\n"
    "                     r\"I'm calling from \\1\", out)\n"
    "        # Negative-motion: Kannada ಯಾರು + neg verb is existential (no one), not interrogative.\n"
    "        out = re.sub(\n"
    "            r\"(?i)\\bWho (?:doesn't|didn't|won't|wouldn't) \"\n"
    "            r\"(?:move[sd]?|stir(?:red)?|budge[d]?|step(?:ped)?)[.!?]?\",\n"
    "            \"No one stirred.\", out)"
)
assert s.count(EN_ANCHOR) == 1, f"EN_ANCHOR not unique ({s.count(EN_ANCHOR)})"
if '"No one stirred."' not in s:
    s = s.replace(EN_ANCHOR, EN_ANCHOR + EN_ADDITION, 1)

# ============================================================
# Write + compile
# ============================================================
if s == orig:
    print(">>> No change (already patched).")
else:
    shutil.copy(P, P + ".bak_grammar")
    open(P, "w", encoding="utf-8").write(s)
    try:
        py_compile.compile(P, doraise=True)
        print(">>> Patched + compiles OK. Backup: " + P + ".bak_grammar")
    except Exception as e:
        shutil.copy(P + ".bak_grammar", P)
        print(">>> Compile FAILED, reverted: " + repr(e))
