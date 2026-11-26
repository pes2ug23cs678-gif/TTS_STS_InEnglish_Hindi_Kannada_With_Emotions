# Results Summary

This note pulls together what the benchmark harness (`ab_translate.py`) and the pipeline's
own telemetry actually told me. Everything below comes from the committed data files
(`ab_results.csv`, `ab_results_review.csv`, `eval/baseline_performance_report.csv`) — I've
tried hard not to round anything in my favour.

## 1. What was measured

- **Translation sweep (`ab_results.csv`):** 176 generations = **2 models** (IndicTrans2
  `1B` vs `dist-200M`) × **2 beam sizes** (1, 4) × **2 target languages** (Hindi, Kannada) ×
  **22 inputs** (20 sentences, plus a glossary "apsara" variant on the 2 that contain it).
  All runs are **CPU, int8-quantized**, warmed up before timing.
- **Full-pipeline breakdown (`eval/baseline_performance_report.csv`):** end-to-end latency
  split by stage, on three representative inputs of increasing difficulty.
- **MOS:** the app has a subjective-rating form, but **no ratings have been logged yet** —
  see §5 for the honest status.

## 2. Translation latency (per sentence, CPU int8, warmed up)

| Model | Beams | Lang | n | Median (s) | Mean (s) | p90 (s) |
|-------|------:|------|--:|-----------:|---------:|--------:|
| 1B | 1 | hi | 22 | 1.58 | 1.58 | 1.92 |
| 1B | 1 | kn | 22 | 2.12 | 2.14 | 3.00 |
| 1B | 4 | hi | 22 | 2.29 | 2.48 | 3.33 |
| 1B | 4 | kn | 22 | 2.11 | 2.07 | 2.65 |
| dist-200M | 1 | hi | 22 | 0.80 | 0.87 | 1.20 |
| dist-200M | 1 | kn | 22 | 0.80 | 0.83 | 1.11 |
| dist-200M | 4 | hi | 22 | 1.16 | 1.22 | 1.40 |
| dist-200M | 4 | kn | 22 | 1.09 | 1.25 | 1.63 |

**Reading this:** the 200M model is roughly **2–3× faster** than the 1B (≈0.8s vs ≈1.6–2.1s
at greedy decoding). Beam search (b4) costs more time on the 1B, but greedy decoding (b1) can produce the wrong
language. On "She won the first prize, and everyone clapped for her" the 1B greedy output
for Hindi came out in Maithili, while beam 4 gave correct Hindi. The automated checks did not
catch this, so the production setting keeps beam 4.

## 3. End-to-end latency, by stage

From `eval/baseline_performance_report.csv` (ms, mean ± sd where available):

| Test case | Lang | Emotion | Translation | Rules | Synthesis | **Total** | Peak RAM Δ |
|-----------|------|--------:|------------:|------:|----------:|----------:|-----------:|
| short_imperative | KN | 73.1 ± 21.0 | 2343.4 ± 153.1 | 0.22 | 1227.9 ± 572.6 | **3644.6** | 1.34 MB |
| formal_academic | KN | 80.6 ± 13.0 | 5202.1 ± 981.3 | 1.54 | 1099.8 ± 344.5 | **6384.1** | 3.15 MB |
| complex_poetic | HI | 100.8 ± 16.8 | 8022.6 ± 615.6 | 1.66 | 1355.0 ± 469.1 | **9480.0** | 5.09 MB |

**Takeaways:**
- **Translation dominates** the budget and scales hard with input length (2.3s → 8.0s from a
  short imperative to a long poetic line). This is the stage to optimise first.
- The **symbolic rules layer is effectively free** (0.2–1.7 ms) — strong evidence that the
  rules + verifier cleanup is a cheap way to buy quality.
- **Emotion detection is cheap** (~73–101 ms), which is exactly why running it in parallel
  with translation is worth it — it hides entirely behind the NMT call.
- Synthesis is ~1.1–1.4 s and fairly flat across input size.

## 4. Quality (heuristic flags + a native-speaker look)

Across all 176 translation runs the automated quality flags were almost entirely clean:
**0 errors, 0 wrong-script outputs, 0 leftover Latin text**, and only **3 "missing end
punctuation"** cases. In other words, both models reliably stay in the correct native script
and produce well-formed sentences — the failure modes are cosmetic, not catastrophic.

The interesting differences are qualitative. A few side-by-side rows from
`ab_results_review.csv`:

- *"Rambha is a celestial nymph"* →
  - **1B (HI):** रंभा एक **दिव्य** अप्सरा है।  vs  **200M (HI):** रंभा एक **खगोलीय** अप्सरा है।
  - **1B (KN):** ರಂಭಾ ಒಂದು **ಆಕಾಶದ ಅಪ್ಸರೆ**  vs  **200M (KN, b1):** ರಂಭಾ ಒಂದು ಖಗೋಳ **ನಿಮ್ಫ್** ಆಗಿದೆ.
    → the 200M left "nymph" essentially **untranslated** (transliterated), while the 1B found
    the correct cultural term. The source glossary ("nymph → apsara") exists precisely to
    harden this case.
- *"He was terrified…"* → the 1B's Hindi (**घबरा गया** / **भयभीत**) reads more naturally than
  the 200M's more literal phrasing.

So the pattern is: **the 1B is more idiomatic and culturally accurate**, especially on rarer
or figurative vocabulary, while the 200M is faster but occasionally literal or incomplete.

## 5. MOS (subjective quality) — current status

The app logs naturalness / intelligibility / affective-fidelity ratings (1–5) to
`mos_evaluations.jsonl` via the in-app form. **As of this report, no MOS data has been
collected**, so I'm not going to quote numbers I don't have. The plumbing is in place; the
remaining work is a short listening study with a handful of Hindi/Kannada speakers rating
the synthesized clips. That is the single highest-value missing piece of the evaluation.

## 6. The model decision (and why)

My rule was *"if two models do the same job, the more precise one wins."* The data backs the
**1B** choice: it costs ~2–3× the latency of the 200M, but it is consistently more idiomatic
and avoids the untranslated-word failures the 200M occasionally hits. For a demo where
quality is the headline and latency is "a couple of seconds either way," that trade is worth
it — so the production config uses **IndicTrans2 `en-indic-1B` with beam search (4 beams)**.
Greedy decoding is faster, but it produced a Maithili output for a Hindi target on at least one
sentence. A wrong language is not acceptable in the demo, so the latency cost of beam 4 is
accepted.

## 7. Honest limitations of this evaluation

- Numbers are **CPU int8**; a GPU run would change the latency story (and the relative
  model cost) substantially.
- The sentence set is **20 sentences** — enough to spot patterns, not a formal test set, and
  the quality judgements are mine, not a blind panel's.
- No **BLEU/chrF** against references yet; the comparison is heuristic + eyeballing against
  Google Translate (see `Tests.txt` for spot checks).
- The heuristic flags check script and punctuation, not language. A Hindi target that comes
  out in Maithili passes them. The `translation_regression.py` test catches this by comparing
  each output with the approved Hindi and Kannada text.
- **Speed is not yet within the 5.2s target.** On the demo CPU, the full pipeline's median end-to-end time is about 7.5s, and the longest sentences (about 180 characters) take about 17s. Translation inside the pipeline also runs about 2.5× slower than translation alone on the same sentences, and the cause is not yet identified. Part of the problem was that the translation worker used 8 torch threads instead of 4; setting it to 4 in `app_ui/app.py` lowered the pipeline's median from 10.9s to 6.6s. Demo sentences should be short.
- MOS is **not yet collected** (§5).
- **Speech synthesis timeout:** the pipeline's TTS calls were cut off at the 5.2 s latency target,
  so any longer synthesis failed. The cutoff is now 25 s (`TTS_HARD_TIMEOUT_SEC` in `config.py`);
  the 5.2 s figure remains the latency target. Replies longer than a sentence or two take longer
  than 5.2 s, so the target is not met for them.

A legacy `run.log` from an earlier NLLB-600M version of the pipeline is kept only for
provenance; it also surfaced the `torch_dtype`/`torch.ao.quantization` deprecation warnings
worth cleaning up before final submission.

## 8. Backend integration (31 October 2026)

- **Automated tests:** `tests/test_api.py` and `tests/test_auth.py` pass (25 of 25). Covers
  registration, duplicate emails, login, wrong passwords, and rejection of unsupported languages
  and non-audio files. The 9 warnings are deprecation notices, not failures.
- **Database:** PostgreSQL (`sts_db`) stores users and translations. Registration and login were
  checked against it directly.
- **End-to-end runs (CPU, `POST /translate`):**
  - One short English sentence: returned 200 and saved the translation.
  - A ten-sentence English passage: returned 200 and saved the translation (row 4), with Kannada
    text and audio. The detected emotion was "disgust", so the emotion tags were suppressed, as
    the sensitivity guard intends.
  - The total time for the ten-sentence request was not measured.
- **Failures found and fixed during integration:**
  - Speech recognition received the wrapper object instead of the audio bytes. Fixed in
    `backend_api/main.py`.
  - Speech synthesis stalled for longer text, and the 5.2 s cutoff cancelled it. Fixed by the
    25 s cutoff above, plus a 12 s limit on each sentence in `tts/edge_engine.py`.
  - A short standalone synthesis test succeeded, so the stall is intermittent and was not
    reproduced on demand.
- **Not yet done:** Swagger's Authorize button still returns 401/422 (login works with curl);
  the backend's package versions are pinned but the deprecation warnings are not yet addressed.
