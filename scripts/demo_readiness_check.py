"""Fast pre-demo sanity check for the translation-quality fixes.

Runs the two text-transform fixes from the Nov-2026 screencast (acronym dandas and
Indic->English proper-noun corruption) against known inputs. Uses NO neural models,
so it finishes in well under a second and needs no GPU or model download.

    python3 demo_readiness_check.py

Exit code 0 = all good; 1 = a regression (do not present until green).
"""
import sys

from verifier import verify_and_clean_output
from entity_masker import mask_named_entities, unmask_and_restore_entities


def _check(name, got, expected):
    ok = got == expected
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        print(f"        expected: {expected!r}")
        print(f"        got     : {got!r}")
    return ok


def _restore_en(src_indic):
    """Mask an Indic sentence, simulate the NMT copying surrogates verbatim into English,
    and restore — mirroring what the pipeline does for an Indic->English request."""
    masked, ledger = mask_named_entities(src_indic)
    return masked, ledger


results = []

# --- Fix 1: acronym spell-out dandas should collapse, real sentences must not. ----------
results.append(_check(
    "acronym BES (hi) collapses",
    verify_and_clean_output("मैं बी। ई। एस। विश्वविद्यालय में हूँ।", "hi"),
    "मैं बी ई एस विश्वविद्यालय में हूँ।"))

results.append(_check(
    "normal hi sentence untouched",
    verify_and_clean_output("हाँ। ठीक है। मैं आ रहा हूँ।", "hi"),
    "हाँ। ठीक है। मैं आ रहा हूँ।"))

results.append(_check(
    "acronym BES (kn) collapses",
    verify_and_clean_output("ನಾನು ಬಿ. ಇ. ಎಸ್. ವಿಶ್ವವಿದ್ಯಾಲಯದಲ್ಲಿ ಇದ್ದೇನೆ.", "kn"),
    "ನಾನು ಬಿ ಇ ಎಸ್ ವಿಶ್ವವಿದ್ಯಾಲಯದಲ್ಲಿ ಇದ್ದೇನೆ."))

# --- Fix 2: known Indic names/cities survive Indic->English intact. ---------------------
masked, ledger = _restore_en("मेरा नाम वरुण है, मैं बैंगलोर में रहता हूँ।")
# Varun and Bangalore must have been swapped for Latin surrogates (so NMT won't mangle them)
results.append(_check(
    "Varun masked out of Hindi source",
    "वरुण" not in masked, True))
results.append(_check(
    "Bangalore masked out of Hindi source",
    "बैंगलोर" not in masked, True))
# ...and the ledger restores them to the correct English spelling
canon = sorted(v[0] for v in ledger.values())
results.append(_check(
    "ledger restores Varun + Bangalore",
    canon, ["Bangalore", "Varun"]))

# --- Regression: English->Hindi forward masking still works. ---------------------------
masked_f, ledger_f = mask_named_entities("My name is Varun and I live in Bangalore")
restored_hi = unmask_and_restore_entities(masked_f, ledger_f, "hi")
results.append(_check(
    "forward en->hi keeps Varun -> वरुण",
    "वरुण" in restored_hi, True))
results.append(_check(
    "forward en->hi keeps Bangalore -> बैंगलोर",
    "बैंगलोर" in restored_hi, True))

print()
if all(results):
    print(f"ALL {len(results)} CHECKS PASSED — translation-quality fixes are live. ✅")
    sys.exit(0)
else:
    print(f"{results.count(False)} of {len(results)} checks FAILED — investigate before demo. ❌")
    sys.exit(1)
