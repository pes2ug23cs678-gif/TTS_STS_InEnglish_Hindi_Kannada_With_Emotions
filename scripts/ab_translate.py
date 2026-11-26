"""A/B benchmark harness comparing IndicTrans2 model sizes, beam counts, and glossaries."""
import argparse
import csv
import gc
import json
import os
import re
import statistics
import sys
import time
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
try:
    from IndicTransToolkit.processor import IndicProcessor
except ImportError:  # same fallback idea as the app
    IndicProcessor = None
MODEL_ALIASES = {
    "200m": "ai4bharat/indictrans2-en-indic-dist-200M",
    "1b": "ai4bharat/indictrans2-en-indic-1B",
}
LANG_CODES = {"hi": "hin_Deva", "kn": "kan_Knda"}
DEFAULT_SENTENCES = [
    "Rambha is a celestial nymph", "The meeting is at three o'clock.",
    "I am absolutely furious, this is unacceptable!", "She felt lonely and heartbroken after the farewell.",
    "The temple was built by the king in the ancient kingdom.", "Please open the window, it is very hot in here.",
    "Where is the nearest bus stop?", "My mother cooks delicious food every Sunday.",
    "The doctor told him to rest for two weeks.", "We are thrilled to announce the winners of this year's competition!",
    "He was terrified when he heard footsteps behind him.", "The river flows through the forest and joins the sea.",
    "The old king ruled the land with kindness and wisdom.", "Can you send me the report by tomorrow evening?",
    "I'm sorry, I did not mean to hurt your feelings.", "The students were surprised by the sudden announcement.",
    "Mythology describes nymphs as heavenly dancers in the court of the gods.",
    "The train to Bengaluru leaves at 6:45 in the morning.",
    "She won the first prize, and everyone clapped for her.", "It is disgusting how the street has been left unclean.",
]
DEFAULT_SOURCE_RULES = [
    [r"(?i)\ba (?:celestial )?nymph\b", "an apsara"],   # keep article agreement
    [r"(?i)\bcelestial nymph(s?)\b", r"apsara\1"],
    [r"(?i)\bnymph(s?)\b", r"apsara\1"],
]
# --------------------------------------------------------------------------- helpers
def app_preprocess(text: str) -> str:
    """Same punctuation padding as _translate_single_clause in models.py."""
    t = re.sub(r"([!?,।])", r" \1 ", text.strip())
    return re.sub(r"\s+", " ", t).strip()


def apply_glossary(text: str, rules) -> str:
    """Applies a list of [pattern, replacement] source-side glossary rules."""
    for rule in rules:
        if isinstance(rule, (list, tuple)) and len(rule) >= 2:
            text = re.sub(rule[0], rule[1], text)
    return text


def quality_flags(out: str, lang: str) -> list:
    """Flags heuristic issues: leftover Latin text, wrong script, or missing end punctuation."""
    if not out:
        return ["empty"]
    flags = []
    if re.search(r"[A-Za-z]", out):
        flags.append("latin")
    if lang == "kn" and re.search(r"[\u0900-\u097F]", out):
        flags.append("devanagari")
    if lang == "hi" and re.search(r"[\u0C80-\u0CFF]", out):
        flags.append("kannada_script")
    if out[-1] not in ".।?!":
        flags.append("no_end_punct")
    return flags


def load_model(model_id: str, device: str, quant: bool):
    """Loads a tokenizer + seq2seq model, optionally int8-quantized on CPU."""
    print(f"\n>>> Loading {model_id} (device={device}, int8={quant and device == 'cpu'}) ...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    dtype = torch.float16 if device == "cuda" else torch.float32
    mod = AutoModelForSeq2SeqLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    mod.generation_config.max_length = None
    if device == "cpu" and quant:
        mod = torch.ao.quantization.quantize_dynamic(mod, {torch.nn.Linear}, dtype=torch.qint8)
    mod.eval()
    print(f"    loaded in {time.time() - t0:.1f}s", flush=True)
    return tok, mod


def translate(tok, mod, proc, text, tgt_code, beams, device):
    """Runs one preprocess -> generate -> postprocess translation and returns the text."""
    norm = app_preprocess(text)
    if proc:
        batch = proc.preprocess_batch([norm], src_lang="eng_Latn", tgt_lang=tgt_code)
    else:
        batch = [f"eng_Latn {tgt_code} {norm}"]
    inputs = tok(batch, truncation=True, padding="longest",
                 return_tensors="pt", return_attention_mask=True).to(device)
    max_new = min(int(inputs["input_ids"].shape[1] * 2.0) + 24, 512)

    def _gen(use_cache):
        with torch.inference_mode():
            return mod.generate(
                **inputs, max_new_tokens=max_new, min_new_tokens=1, num_beams=beams,
                repetition_penalty=1.15, do_sample=False, use_cache=use_cache)

    try:
        generated = _gen(True)
    except Exception:
        generated = _gen(False)  # some transformers versions break IndicTrans2's cache path

    decoded = tok.batch_decode(generated, skip_special_tokens=True)
    if proc:
        try:
            decoded = proc.postprocess_batch(decoded, lang=tgt_code)
        except Exception:
            pass
    return decoded[0].strip()


def pct(values, q):
    """Returns the q-quantile (0..1) of a numeric list."""
    values = sorted(values)
    if not values:
        return 0.0
    idx = min(len(values) - 1, int(round(q * (len(values) - 1))))
    return values[idx]


# --------------------------------------------------------------------------- main
def main(argv=None):
    """Parses CLI args and runs the full benchmark sweep, writing CSVs and a summary."""
    # Single CLI entry point: argument parsing plus the full benchmark loop.
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="200m,1b", help="comma list of aliases (200m, 1b) or HF ids")
    ap.add_argument("--beams", default="1,4", help="comma list of beam sizes")
    ap.add_argument("--langs", default="hi,kn")
    ap.add_argument("--limit", type=int, default=0, help="only use the first N sentences")
    ap.add_argument("--sentences-file", default="", help="text file, one English sentence per line")
    ap.add_argument("--rules", default="", help="rules.json to read source_rules.replacements from")
    ap.add_argument("--no-gloss", action="store_true", help="skip the glossary variant")
    ap.add_argument("--no-quant", action="store_true", help="CPU fp32 without int8 quantization")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--threads", type=int, default=0, help="torch threads (default: min(cores, 4) like the app)")
    ap.add_argument("--out", default="ab_results.csv")
    args = ap.parse_args(argv)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; using CPU.")
        device = "cpu"
    threads = args.threads or max(1, min(os.cpu_count() or 4, 4))
    torch.set_num_threads(threads)
    print(f"torch threads: {threads} | cores: {os.cpu_count()} | device: {device}")

    # sentences
    if args.sentences_file:
        with open(args.sentences_file, encoding="utf-8") as f:
            sentences = [ln.strip() for ln in f if ln.strip()]
    else:
        sentences = list(DEFAULT_SENTENCES)
    if args.limit:
        sentences = sentences[: args.limit]

    # glossary
    rules = DEFAULT_SOURCE_RULES
    if args.rules and os.path.exists(args.rules):
        with open(args.rules, encoding="utf-8") as f:
            loaded = json.load(f).get("source_rules", {}).get("replacements", [])
        if loaded:
            rules = loaded

    variants = []  # (variant, original_sentence, text_to_translate)
    for s in sentences:
        variants.append(("raw", s, s))
        if not args.no_gloss:
            g = apply_glossary(s, rules)
            if g != s:
                variants.append(("gloss", s, g))

    model_ids = [MODEL_ALIASES.get(m.strip().lower(), m.strip()) for m in args.models.split(",") if m.strip()]
    beam_list = [int(b) for b in args.beams.split(",") if b.strip()]
    langs = [l.strip() for l in args.langs.split(",") if l.strip()]

    total = len(model_ids) * len(beam_list) * len(langs) * len(variants)
    print(f"{len(sentences)} sentences -> {len(variants)} inputs | {total} generations planned")

    rows = []
    review = defaultdict(dict)  # (lang, variant, sentence, input_used) -> {config: output}
    proc = IndicProcessor(inference=True) if IndicProcessor else None
    if not proc:
        print("WARNING: IndicTransToolkit not installed; using plain tag formatting.")

    for model_id in model_ids:
        short = model_id.split("/")[-1].replace("indictrans2-en-indic-", "")
        tok, mod = load_model(model_id, device, not args.no_quant)

        # warm-up (not timed, not recorded)
        try:
            translate(tok, mod, proc, "Hello, how are you today?", LANG_CODES[langs[0]], 1, device)
        except Exception as e:
            print(f"    warm-up failed: {e}")

        for beams in beam_list:
            for lang in langs:
                tgt = LANG_CODES[lang]
                config = f"{short}|b{beams}|{lang}"
                print(f"\n== {config} ==", flush=True)
                for variant, original, used in variants:
                    t0 = time.perf_counter()
                    try:
                        out = translate(tok, mod, proc, used, tgt, beams, device)
                        err = ""
                    except Exception as e:  # keep going; record the failure
                        out, err = "", f"{type(e).__name__}: {e}"
                    dt = time.perf_counter() - t0
                    flags = quality_flags(out, lang) if not err else ["error"]
                    rows.append({
                        "model": short, "int8": (device == "cpu" and not args.no_quant),
                        "beams": beams, "lang": lang, "variant": variant,
                        "sentence": original, "input_used": used, "output": out or err,
                        "seconds": round(dt, 2), "flags": ";".join(flags),
                    })
                    review[(lang, variant, original, used)][f"{short}|b{beams}"] = out or err
                    tag = f" [{';'.join(flags)}]" if flags else ""
                    print(f"  ({variant}) {used}\n     -> {out or err}   {dt:.1f}s{tag}", flush=True)

        del mod, tok
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # ---- write CSVs
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["model"])
        w.writeheader()
        w.writerows(rows)

    review_path = re.sub(r"\.csv$", "", args.out) + "_review.csv"
    configs = sorted({c for d in review.values() for c in d})
    with open(review_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["lang", "variant", "sentence", "input_used"] + configs + ["reviewer_best", "reviewer_notes"])
        for (lang, variant, original, used), outs in sorted(review.items()):
            w.writerow([lang, variant, original, used] + [outs.get(c, "") for c in configs] + ["", ""])

    # ---- summary
    print("\n" + "=" * 78)
    print("SUMMARY (per-sentence latency in seconds, warmed up)")
    print("=" * 78)
    groups = defaultdict(list)
    flagged = defaultdict(int)
    for r in rows:
        key = (r["model"], r["beams"], r["lang"])
        groups[key].append(r["seconds"])
        if r["flags"]:
            flagged[key] += 1
    print(f"{'model':<10}{'beams':>6}{'lang':>6}{'n':>5}{'median':>9}{'mean':>8}{'p90':>8}{'flagged':>9}")
    for key in sorted(groups):
        v = groups[key]
        print(f"{key[0]:<10}{key[1]:>6}{key[2]:>6}{len(v):>5}"
              f"{statistics.median(v):>9.2f}{statistics.mean(v):>8.2f}{pct(v, 0.9):>8.2f}{flagged[key]:>9}")
    print(f"\nWrote: {args.out}\n       {review_path}  (side-by-side sheet for a native-speaker review)")
    print("Note: 'flagged' counts heuristic issues only (untranslated Latin text, wrong script, no end punctuation).")


if __name__ == "__main__":
    sys.exit(main())
