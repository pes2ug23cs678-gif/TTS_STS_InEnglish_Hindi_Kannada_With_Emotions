"""Masks proper nouns before translation and restores them in the target script after."""
import re
import logging

logger = logging.getLogger(__name__)

# indic-nlp-library (module: `indicnlp`) is OPTIONAL here: it is only the first-choice
# transliterator in unmask_and_restore_entities(), which already falls back to
# indic_transliteration's sanscript when it is missing. Import it defensively so a machine
# without it doesn't crash the entire app at startup (ModuleNotFoundError: 'indicnlp').
# Install it for the best proper-noun transliteration: pip install indic-nlp-library
try:
    from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
except Exception as _err:  # noqa: BLE001 - optional dependency
    UnicodeIndicTransliterator = None
    logger.info("indic-nlp-library not available (%s); using sanscript transliteration "
                "fallback. Install with: pip install indic-nlp-library", _err)

# indic_transliteration is the sanscript fallback. Guard it too so neither transliteration
# library being absent can stop the module importing; restoration then keeps the Latin name.
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
except Exception as _err:  # noqa: BLE001 - optional dependency
    sanscript = None
    transliterate = None
    logger.info("indic_transliteration not available (%s); proper nouns without a static "
                "mapping will be left in Latin script.", _err)
STATIC_ENTITY_MAP = {
    "TTS": {"hi": "टीटीएस", "kn": "ಟಿಟಿಎಸ್"},
    "STS": {"hi": "एसटीएस", "kn": "ಎಸ್ಟಿಎಸ್"},
    "Likith": {"hi": "लिकिथ", "kn": "ಲಿಕಿಥ್"},
    "Dr Rajkumar": {"hi": "डॉ. राजकुमार", "kn": "ಡಾ. ರಾಜ್‌ಕುಮಾರ್"},
    "Dr. Rajkumar": {"hi": "डॉ. राजकुमार", "kn": "ಡಾ. ರಾಜ್‌ಕುಮಾರ್"},
    "Shiva Rajkumar": {"hi": "शिव राजकुमार", "kn": "ಶಿವರಾಜ್‌ಕುಮಾರ್"},
    "Puneeth Rajkumar": {"hi": "पुनीत राजकुमार", "kn": "ಪುನೀತ್ ರಾಜ್‌ಕುಮಾರ್"},
    "Puneet Rajkumar": {"hi": "पुनीत राजकुमार", "kn": "ಪುನೀತ್ ರಾಜ್‌ಕುಮಾರ್"},
    "Raghavendra Rajkumar": {"hi": "राघवेंद्र राजकुमार", "kn": "ರಾಘವೇಂದ್ರ ರಾಜ್‌ಕುಮಾರ್"},
    "Parvathamma Rajkumar": {"hi": "पार्वथम्मा राजकुमार", "kn": "ಪಾರ್ವತಮ್ಮ ರಾಜ್‌ಕುಮಾರ್"},
    "Rajinikanth": {"hi": "रजनीकांत", "kn": "ರಜನಿಕಾಂತ್"},
    "Muthu": {"hi": "मुथु", "kn": "ಮುತ್ತು"},
    "Priyadarshan": {"hi": "प्रियदर्शन", "kn": "ಪ್ರಿಯದರ್ಶನ್"},
    "Mohanlal": {"hi": "मोहनलाल", "kn": "ಮೋಹನ್‌ಲಾಲ್"},
    "Nedumudi Venu": {"hi": "नेदुमुदी वेणु", "kn": "ನೆಡುಮುಡಿ ವೇಣು"},
    "Connie Francis": {"hi": "कोनी फ्रांसिस", "kn": "ಕಾನಿ ಫ್ರಾನ್ಸಿಸ್"},
    "Declan Rice": {"hi": "डेक्लान राइस", "kn": "ಡೆಕ್ಲಾನ್ ರೈಸ್"},
    "Sydney Sweeney": {"hi": "सिडनी स्वीनी", "kn": "ಸಿಡ್ನಿ ಸ್ವೀನಿ"},
    "George Washington": {"hi": "जॉर्ज वाशिंगटन", "kn": "ಜಾರ್ಜ್ ವಾಷಿಂಗ್ಟನ್"},
    "Thomas Jefferson": {"hi": "थॉमस जेफरसन", "kn": "ಥಾಮಸ್ ಜೆಫರ್ಸನ್"},
    "Abraham Lincoln": {"hi": "अब्राहम लिंकन", "kn": "ಅಬ್ರಹಾಂ ಲಿಂಕನ್"},
    "Alexander Hamilton": {"hi": "अलेक्जेंडर हैमिल्टन", "kn": "ಅಲೆಕ್ಸಾಂಡರ್ ಹ್ಯಾಮಿಲ್ಟನ್"},
    "Andrew Jackson": {"hi": "एंड्रयू जैक्सन", "kn": "ಆಂಡ್ರ್ಯೂ ಜಾಕ್ಸನ್"},
    "Ulysses S Grant": {"hi": "यूलिसिस एस. ग्रांट", "kn": "ಯುಲಿಸೆಸ್ ಎಸ್. ಗ್ರಾಂಟ್"},
    "Ulysses S. Grant": {"hi": "यूलिसिस एस. ग्रांट", "kn": "ಯುಲಿಸೆಸ್ ಎಸ್. ಗ್ರಾಂಟ್"},
    "Benjamin Franklin": {"hi": "बेंजामिन फ्रैंकलिन", "kn": "ಬೆಂಜಮಿನ್ ಫ್ರಾಂಕ್ಲಿನ್"},
    "Varun": {"hi": "वरुण", "kn": "ವರುಣ್"},
    "Sujan": {"hi": "सुजन", "kn": "ಸುಜನ್"},
    "Freddie Mercury": {"kn": "ಫ್ರೆಡ್ಡಿ ಮರ್ಕ್ಯುರಿ", "hi": "फ़्रेडी मर्करी"},
    "Beatles": {"kn": "ಬೀಟಲ್ಸ್", "hi": "बीटल्स"},
    "The Beatles": {"kn": "ಬೀಟಲ್ಸ್", "hi": "बीटल्स"},
    "John Lennon": {"kn": "ಜಾನ್ ಲೆನನ್", "hi": "जॉन लेनन"},
    "George Harrison": {"kn": "ಜಾರ್ಜ್ ಹ್ಯಾರಿಸನ್", "hi": "जॉर्ज हैरिसन"},
    "Paul McCartney": {"kn": "ಪಾಲ್ ಮೆಕ್ಕಾರ್ಟ್ನಿ", "hi": "पॉल मेकार्टनी"},
    "Ringo Starr": {"kn": "ರಿಂಗೋ ಸ್ಟಾರ್", "hi": "रिंगो स्टार"},
    "Michael Jackson": {"hi": "माइकल जैक्सन", "kn": "ಮೈಕೆಲ್ ಜಾಕ್ಸನ್"},
    "Madonna": {"hi": "मैडोना", "kn": "ಮಡೋನಾ"},
    "Prince": {"hi": "प्रिंस", "kn": "ಪ್ರಿನ್ಸ್"},
    "Whitney Houston": {"hi": "व्हिटनी ह्यूस्टन", "kn": "ವಿಟ್ನಿ ಹೂಸ್ಟನ್"},
    "Cindy Lauper": {"hi": "सिंडी लॉपर", "kn": "ಸಿಂಡಿ ಲಾಫರ್"},
    "Cyndi Lauper": {"hi": "सिंडी लॉपर", "kn": "ಸಿಂಡಿ ಲಾಫರ್"},
    "George Michael": {"hi": "जॉर्ज माइकल", "kn": "ಜಾರ್ಜ್ ಮೈಕೆಲ್"},
    "Rambha": {"hi": "रंभा", "kn": "ರಂಭೆ"},
}
# Common Indian place names. Merged into STATIC_ENTITY_MAP so they are protected in BOTH
# directions: English->Indic uses the curated spelling, and Indic->English restores the
# proper English name instead of IndicTrans2's raw romanisation (e.g. "बैंगलोर" -> "Bangalore"
# rather than "Bang Lur").
PLACE_NAMES = {
    "Bangalore": {"hi": "बैंगलोर", "kn": "ಬೆಂಗಳೂರು"},
    "Bengaluru": {"hi": "बेंगलुरु", "kn": "ಬೆಂಗಳೂರು"},
    "Mysore": {"hi": "मैसूर", "kn": "ಮೈಸೂರು"},
    "Mysuru": {"hi": "मैसूरु", "kn": "ಮೈಸೂರು"},
    "Mangalore": {"hi": "मंगलौर", "kn": "ಮಂಗಳೂರು"},
    "Hubli": {"hi": "हुबली", "kn": "ಹುಬ್ಬಳ್ಳಿ"},
    "Belgaum": {"hi": "बेलगाम", "kn": "ಬೆಳಗಾವಿ"},
    "Delhi": {"hi": "दिल्ली", "kn": "ದೆಹಲಿ"},
    "New Delhi": {"hi": "नई दिल्ली", "kn": "ಹೊಸ ದೆಹಲಿ"},
    "Mumbai": {"hi": "मुंबई", "kn": "ಮುಂಬೈ"},
    "Chennai": {"hi": "चेन्नई", "kn": "ಚೆನ್ನೈ"},
    "Hyderabad": {"hi": "हैदराबाद", "kn": "ಹೈದರಾಬಾದ್"},
    "Pune": {"hi": "पुणे", "kn": "ಪುಣೆ"},
    "Kolkata": {"hi": "कोलकाता", "kn": "ಕೋಲ್ಕತಾ"},
    "Bhopal": {"hi": "भोपाल", "kn": "ಭೋಪಾಲ್"},
    "Jaipur": {"hi": "जयपुर", "kn": "ಜೈಪುರ"},
    "Lucknow": {"hi": "लखनऊ", "kn": "ಲಕ್ನೋ"},
    "Ahmedabad": {"hi": "अहमदाबाद", "kn": "ಅಹಮದಾಬಾದ್"},
    "Kerala": {"hi": "केरल", "kn": "ಕೇರಳ"},
    "Karnataka": {"hi": "कर्नाटक", "kn": "ಕರ್ನಾಟಕ"},
    # NOTE: "India" (भारत / ಭಾರತ) is deliberately NOT masked. IndicTrans2 already translates it
    # (and the adjective भारतीय -> "Indian") correctly, and masking it broke on the ADJECTIVE: the
    # masker substring-replaced भारत inside भारतीय with the surrogate "David", which IndicTrans2
    # inflected to "Davidic" -- a form the English un-masker could not restore. Common country
    # names gain nothing from masking and only risk this surrogate-leak, so leave India unmasked.
}
STATIC_ENTITY_MAP.update(PLACE_NAMES)
STATIC_ENTITY_MAP_LOWER = {k.lower(): v for k, v in STATIC_ENTITY_MAP.items()}
# Extra Indic-script spellings that ASR/NMT may emit for a known entity, mapped to the
# canonical English name. Used only for Indic->(English/other) detection, not for forward
# restoration. Add common variants so the entity is caught however it was transcribed.
INDIC_ENTITY_ALIASES = {
    # Bangalore / Bengaluru variants
    "बेंगलूर": "Bangalore", "बैंगलुरु": "Bangalore", "बंगलौर": "Bangalore",
    "ಬೆಂಗಳೂರ": "Bangalore", "ಬ್ಯಾಂಗಲೋರ್": "Bangalore",
    # Varun variants (व/ब confusion is common in ASR)
    "बरुण": "Varun", "वरूण": "Varun", "ವರುಣ": "Varun",
    # Sujan (ASR may drop the final virama: ಸುಜನ್ -> ಸುಜನ)
    "ಸುಜನ": "Sujan", "सुजन": "Sujan",
    # Mysore
    "मैसूरु": "Mysore",
}
# Reverse index: any known Indic surface form -> canonical English entity name.
# Built from the static map's hi/kn spellings plus the alias table above, longest first
# so multi-word names match before their fragments.
_INDIC_TO_ENTITY = {}
for _canon, _forms in STATIC_ENTITY_MAP.items():
    for _code in ("hi", "kn"):
        _form = _forms.get(_code)
        if _form:
            _INDIC_TO_ENTITY.setdefault(_form, _canon)
for _form, _canon in INDIC_ENTITY_ALIASES.items():
    _INDIC_TO_ENTITY.setdefault(_form, _canon)
_INDIC_FORMS_SORTED = sorted(_INDIC_TO_ENTITY, key=len, reverse=True)
# In-vocabulary natural surrogates with all common transliteration variants
# Robust 16-Surrogate Pool
SURROGATES = [
    {"en": "David", "kn": ["ಡೇವಿಡ್"], "hi": ["डेविड", "डैविड"]},
    {"en": "Daniel", "kn": ["ಡೇನಿಯಲ್"], "hi": ["डेनियल", "डैनियल", "दानियल"]},
    {"en": "Samuel", "kn": ["ಸ್ಯಾಮ್ಯುಯೆಲ್", "ಸಾಮ್ಯುಯೆಲ್"], "hi": ["सैमुअल", "सेमुअल", "सैम्युअल"]},
    {"en": "Joseph", "kn": ["ಜೋಸೆಫ್"], "hi": ["जोसेफ"]},
    {"en": "Thomas", "kn": ["ಥಾಮಸ್"], "hi": ["थॉमस", "टॉमस"]},
    {"en": "Gabriel", "kn": ["ಗೇಬ್ರಿಯಲ್"], "hi": ["गेब्रियल", "गैब्रियल"]},
    {"en": "Lucas", "kn": ["ಲ್ಯೂಕಾಸ್"], "hi": ["ल्यूकस", "लूकस", "ल्यूकास"]},
    {"en": "Oliver", "kn": ["ಆಲಿವರ್"], "hi": ["ओलिवर", "ऑलिवर"]},
    {"en": "Arthur", "kn": ["ಆರ್ಥರ್"], "hi": ["आर्थर"]},
    {"en": "Victor", "kn": ["ವಿಕ್ಟರ್"], "hi": ["विक्टर"]},
    {"en": "Henry", "kn": ["ಹೆನ್ರಿ"], "hi": ["हेनरी"]},
    {"en": "Edward", "kn": ["ಎಡ್ವರ್ಡ್"], "hi": ["एडवर्ड"]},
    {"en": "Albert", "kn": ["ಆಲ್ಬರ್ಟ್"], "hi": ["अल्बर्ट", "आल्बर्ट"]},
    {"en": "Walter", "kn": ["ವಾಲ್ಟರ್"], "hi": ["वॉल्टर", "वाल्टर"]},
    {"en": "Philip", "kn": ["ಫಿಲಿಪ್"], "hi": ["फिलिप"]},
    {"en": "Oscar", "kn": ["ಆಸ್ಕರ್"], "hi": ["ऑस्कर", "ओस्कर"]}
]
COMMON_NON_ENTITIES = {"see", "look", "listen", "hear", "read", "write", "the",
"this", "that", "these", "those", "please", "word", "math", "science",
"tell", "say", "speak", "talk", "show", "give", "take"}
def mask_named_entities(text: str) -> tuple[str, dict]:
    """Substitutes proper nouns with natural in-vocab names to avoid BPE token crashes."""
    ledger = {}
    masked_text = text
    used_surrogates = 0
    # 0. Indic-script sweep: when the source is Hindi/Kannada, the Latin patterns below
    # never match, so known proper nouns (names, cities) reach IndicTrans2 unprotected and
    # come back as mangled romanisations ("वरुण" -> "Barun", "बैंगलोर" -> "Bang Lur"). Here we
    # swap each known Indic surface form for an in-vocabulary English surrogate that survives
    # translation; unmask_and_restore_entities() then writes the canonical name in the target
    # script. The ledger format is identical to the Latin sweeps below.
    if re.search(r'[ऀ-ॿಀ-೿]', masked_text):
        for indic_form in _INDIC_FORMS_SORTED:
            if used_surrogates >= len(SURROGATES):
                break
            if indic_form and indic_form in masked_text:
                canonical = _INDIC_TO_ENTITY[indic_form]
                surrogate = SURROGATES[used_surrogates]
                masked_text = masked_text.replace(indic_form, surrogate["en"])
                ledger[surrogate["en"]] = (canonical, surrogate)
                used_surrogates += 1
    # 1. First sweep: Statically mapped entities (sorted descending by length)
    sorted_entities = sorted(STATIC_ENTITY_MAP.keys(), key=len, reverse=True)
    for ent in sorted_entities:
        if ent.strip().lower() in COMMON_NON_ENTITIES:
            continue
        pattern = rf"\b{re.escape(ent)}\b"
        if re.search(pattern, masked_text, flags=re.IGNORECASE):
            if used_surrogates < len(SURROGATES):
                surrogate = SURROGATES[used_surrogates]
                masked_text = re.sub(pattern, surrogate["en"], masked_text, flags=re.IGNORECASE)
                ledger[surrogate["en"]] = (ent, surrogate)
                used_surrogates += 1
    # 2. Second sweep: Unmapped Title-Cased multi-word entities
    title_pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"
    for match in re.finditer(title_pattern, masked_text):
        ent = match.group(1)
        # Verify none of the words are common imperative sentence starters or non-entities
        words_lower = [w.lower() for w in ent.split()]
        if any(w in COMMON_NON_ENTITIES for w in words_lower):
            continue
        if any(s["en"] == ent for s in SURROGATES):
            continue
        # Do NOT re-capture a surrogate an earlier sweep already inserted. Sweep 1 turns e.g.
        # "Varun" into "David"; without this guard, "Happy Birthday David" then looks like a new
        # Title-Case entity and gets nested under a second surrogate, which un-masking cannot
        # undo (the inner surrogate leaks into the output, e.g. "Happy Birthday David!").
        if any(w in ledger for w in ent.split()):
            continue
        if used_surrogates < len(SURROGATES):
            surrogate = SURROGATES[used_surrogates]
            masked_text = re.sub(rf"\b{re.escape(ent)}\b", surrogate["en"], masked_text)
            ledger[surrogate["en"]] = (ent, surrogate)
            used_surrogates += 1
    return masked_text, ledger
def unmask_and_restore_entities(translated_text: str, ledger: dict, tgt_code: str) -> str:
    """Restores the translated surrogate back to the actual target transliteration."""
    restored = translated_text
    # Normalize only the prime character (′, U+2032) IndicTrans2 occasionally emits between
    # list items. Do NOT also replace the apostrophe "'": that destroyed every English
    # contraction and possessive (I'm -> "I, m", don't -> "don, t", it's -> "it, s"), which
    # is a far more common and visible failure than the rare list-prime artifact.
    restored = restored.replace("′", ", ")
    # Normalize the target code to a 2-letter form ('eng_Latn' -> 'en', etc.)
    tgt_code = {"eng_latn": "en", "hin_deva": "hi", "kan_knda": "kn"}.get(
        str(tgt_code).lower(), str(tgt_code).lower()[:2])
    # English target keeps the Latin script: restore each surrogate to the ORIGINAL
    # entity name instead of transliterating it into an Indic script.
    if tgt_code == "en":
        for surrogate_en, (original_ent, _surrogate_dict) in ledger.items():
            restored = re.sub(rf"(?<![A-Za-z]){re.escape(surrogate_en)}(?![A-Za-z])",
                              original_ent, restored)
        restored = re.sub(r"\s+", " ", restored)
        restored = re.sub(r"\s+([.,!?])", r"\1", restored)
        return restored.strip()
    for surrogate_en, (original_ent, surrogate_dict) in ledger.items():
        ent_lower = original_ent.strip().lower()
        if ent_lower in STATIC_ENTITY_MAP_LOWER and tgt_code in STATIC_ENTITY_MAP_LOWER[ent_lower]:
            target_translit = STATIC_ENTITY_MAP_LOWER[ent_lower][tgt_code]
        else:
            try:
                if UnicodeIndicTransliterator is None:
                    raise RuntimeError("indic-nlp-library not installed")
                src_script = "latn"
                tgt_script = "knda" if tgt_code == "kn" else "deva"
                target_translit = UnicodeIndicTransliterator.transliterate(original_ent,
                src_script, tgt_script)
            except Exception:
                try:
                    if transliterate is None or sanscript is None:
                        raise RuntimeError("indic_transliteration not installed")
                    scheme = sanscript.KANNADA if tgt_code == "kn" else sanscript.DEVANAGARI
                    target_translit = transliterate(original_ent.lower(), sanscript.ITRANS, scheme)
                except Exception:
                    # No transliterator available: keep the original (Latin) name rather
                    # than crash. Static-map entities are unaffected (handled above).
                    target_translit = original_ent
        # Retrieve surrogate variants for the target language
        surrogate_val = surrogate_dict.get(tgt_code, surrogate_en)
        surrogate_variants = surrogate_val if isinstance(surrogate_val, list) else [surrogate_val]
        # Match against all known script variants (e.g., both 'डेनियल' and 'डैनियल')
        for surrogate_variant in surrogate_variants:
            pattern = rf"(?<![^\s,।.\-!?]){re.escape(surrogate_variant)}(\S*?)(?![^\s,।.\-!?])"
            restored = re.sub(pattern, rf"{target_translit}\1", restored)
        # The translator sometimes copies the English stand-in (e.g. "David") instead of
        # writing it in the target script, so also restore the Latin form.
        restored = re.sub(rf"(?<![A-Za-z]){re.escape(surrogate_en)}(?![A-Za-z])",
                          target_translit, restored)
    # Restore natural comma separation between adjacent entities in naked lists
    # Only add a comma when a name is followed by a conjunction (a list such as 'A और B'),
    # not when it is followed by a verb, a particle such as 'ने', or another name.
    list_follow = r"(?=(?:और|या|ಅಥವಾ)\s)"
    for _, meta in STATIC_ENTITY_MAP.items():
        name_tgt = meta.get(tgt_code)
        if name_tgt and name_tgt in restored:
            restored = re.sub(rf"({re.escape(name_tgt)})\s+{list_follow}", r"\1, ", restored)
    # Clean punctuation artifacts & orphan commas before Hindi auxiliary verbs
    restored = re.sub(r",\s*(हूँ|हैं|है|हो|ಆಗಿದ್ದಾರೆ|ಇದ್ದಾರೆ|ಇದ್ದರು)", r" \1", restored)
    restored = re.sub(r"\s+", " ", restored)
    restored = re.sub(r"\s*,\s*,+", ", ", restored)
    restored = re.sub(r",+\s*([।.,!?])", r"\1", restored)
    restored = re.sub(r"\s+([।.,!?])", r"\1", restored)
    restored = re.sub(r",+\s*$", "", restored)
    return restored.strip()

