# Architecture Note

A walkthrough of how the Autonomous Trilingual Emotion STS Engine is put together, why
it is built this way, and the trade-offs I made along the way. The aim was a pipeline
that is *modular* (each stage swappable), *resilient* (one failing service should not
crash the whole app), and *emotion-faithful* (the output should carry the feeling of the
input, not just its meaning).

## 1. The big picture

At a high level the system is a chain of specialists. English comes in (typed or spoken),
and Hindi/Kannada speech comes out. Instead of one giant model doing everything, I split
the job into stages and let each stage do one thing well:

```
            ┌─────────────┐
 speech ───►│ ASR (Whisper)│──┐
            └─────────────┘  │
 text ───────────────────────┤
                             ▼
                    ┌──────────────────┐
                    │  entity masking  │   (protect proper nouns)
                    └──────────────────┘
                             │
             ┌───────────────┴───────────────┐   run in parallel
             ▼                               ▼
     ┌───────────────┐              ┌──────────────────┐
     │ IndicTrans2 NMT│              │ DistilRoBERTa emo │
     └───────────────┘              └──────────────────┘
             │                               │
             ▼                               │
     ┌───────────────┐                       │
     │ unmask +      │                       │
     │ transliterate │                       │
     └───────────────┘                       │
             ▼                               │
     ┌───────────────┐                       │
     │ rules+verifier│                       │
     └───────────────┘                       │
             ▼                               ▼
          ┌──────────────────────────────────────┐
          │ AVD vector  ─►  TTS (emotion→prosody)  │
          └──────────────────────────────────────┘
                             │
                             ▼
                  Hindi / Kannada audio
```

The orchestration lives in `pipeline.py` (`STSPipeline.process`), the UI and glue live in
`app.py`, and everything else is a self-contained module behind a simple function or class.

## 2. Component breakdown

**`app.py` — UI + orchestration.** A Streamlit front end. It builds the pipeline once
(cached), takes the user's input + target language + gender, calls the pipeline, and
renders the translation, the inferred emotion, and the audio. It also hosts a MOS feedback
form for subjective evaluation.

**`pipeline.py` — the conductor.** `STSPipeline.process()` runs the stages in order and
records per-stage timing in a `telemetry` dict. It does *not* know how any individual
model works — it only talks to injected functions (`translator_fn`, `emotion_fn`) and to
the TTS dispatcher. This dependency-injection style is what makes the pipeline testable and
the models swappable.

**`models.py` — the neural heavy-lifting.** Loads IndicTrans2 (translation), DistilRoBERTa
(emotion), and Whisper (ASR), warms them up, and exposes `translate_text` and
`detect_text_emotion`. Loading is wrapped in Streamlit's `@st.cache_resource` so the weights
load once per session, not once per click.

**`affect.py` — emotion as numbers.** Converts the emotion probability distribution into an
Arousal–Valence–Dominance (AVD) vector in `[-1, 1]³`. This is the bridge between "the model
said 70% joy" and "so the voice should be higher, faster, and a bit louder."

**`entity_masker.py` — protecting names.** Proper nouns (people, places) tend to get
mangled by sub-word tokenizers, so before translation they are swapped for safe in-vocabulary
surrogate names, and after translation they are restored in the correct target script via a
static map or transliteration.

**`rules_engine.py` + `verifier.py` — the cleanup crew.** NMT output is good but not perfect.
`rules_engine` applies declarative surface rules from `rules.json`, and `verifier` applies
deterministic regex fixes (killing stutter loops, repairing known sub-word truncations,
normalising punctuation). Both are sub-millisecond and purely rule-based — no extra model.

**`tts/` — the voice.** A strategy/dispatcher pattern: a common `BaseTTSEngine` interface,
three concrete engines (ElevenLabs, Edge, and Parler, which is disabled for now), and a `TTSDispatcher` that picks one and
falls back on failure.

**`asr/whisper_engine.py` — speech in.** faster-whisper with built-in voice-activity
detection, plus a transcript cleaner that strips fillers and stutters before the text hits
the NMT stage.

## 3. The design decisions I actually had to think about

**Why run NMT and emotion detection in parallel?** They are independent — translation does
not need the emotion label and vice-versa — so running them concurrently with
`asyncio.gather` hides the latency of the slower one. For a user-facing demo, shaving those
seconds matters.

**Why mask entities at all?** Early on, names like "Declan Rice" or "Puneeth Rajkumar" would
come out as garbled sub-words or get half-translated. Masking them with ordinary names the
model *does* know, then restoring them afterwards, was the cleanest fix that did not require
retraining anything.

**Why go through an AVD vector instead of just using the emotion label?** A label is a single
bucket ("sad"), but prosody is continuous. The AVD vector lets me scale *how* sad — a 90%-sad
line gets a bigger pitch/rate drop than a 55%-sad one. It also gives a sensible fallback when
the label itself has no direct prosody mapping.

**Why more than one TTS engine?** This is really about resilience, not redundancy. ElevenLabs v3
(with emotion tags) is the primary because it is the most expressive. Edge-TTS is the network
fallback. Indic-Parler-TTS was meant to be the fully-offline last resort, but it is disabled
for now, so the chain currently ends at Edge-TTS when ElevenLabs is unavailable. The dispatcher enforces a latency ceiling and degrades gracefully down this
chain instead of hanging or erroring out.

**Why a separate rules + verifier layer?** Because the failure modes of NMT are *predictable*
and *repeatable* (the same awkward phrase, the same fused word). Fixing those with cheap
deterministic rules is far more reliable than hoping a bigger model gets it right, and it
keeps the corrections auditable — I can read exactly what was changed and why.

## 4. Emotion → prosody

This is the heart of the "emotion-aware" claim. The emotion model produces a probability
distribution; `affect.py` turns it into an AVD vector; and each TTS engine maps that vector
to its own controls:

- **Edge-TTS** derives pitch, rate and volume directly from arousal and valence (e.g. high
  arousal → faster and higher; negative valence + low arousal → slower, lower, quieter).
  The coefficients were widened deliberately because the neural voice sounded flat/robotic
  when the swings were too small.
- **ElevenLabs** picks an emotion audio tag (`[happy]`, `[sad]`, …) and a stability setting
  from the same AVD signals.
- **Parler** (currently disabled) builds a natural-language style description ("… spoken at a fast pace and high
  pitch …") that conditions the local model.

Keeping each engine's mapping inside that engine was a conscious choice — it keeps them
self-contained and swappable, rather than coupling them all to one shared prosody module.

## 5. Caching and the "fast path"

Two levels of caching keep the app responsive:

1. **Model weights** are cached for the whole session (`@st.cache_resource`), so you pay the
   load cost once.
2. **Results** are cached per input in `st.session_state`. If you only change the *voice
   gender* but keep the same text, the app skips translation + emotion entirely and
   re-synthesises audio only — a genuine "fast path."

The pipeline exposes its UI-facing shape through a single `PipelineResult.ui_meta` property,
so the UI never has to re-derive status/engine/emotion fields by hand — it has one source of
truth to read from.

## 6. Resilience and failure handling

The guiding rule was *the app should never hard-crash on a flaky service*. So:

- The TTS call is wrapped in a timeout; if the primary engine stalls, the dispatcher falls
  back down the engine chain.
- A quick socket pre-flight check reroutes away from Edge-TTS if its endpoint is unreachable,
  instead of waiting for a timeout.
- Rules/verifier/transliteration steps are wrapped so that if a cleanup step fails, the
  pipeline still returns the un-cleaned (but valid) translation rather than nothing.
- Whisper input is resampled to 16 kHz regardless of the source rate, so audio from any
  microphone/file transcribes correctly.

## 7. Known limitations and future work

- Only **Hindi** and **Kannada** targets are wired up today; adding a language is mostly a
  matter of voice IDs, rules, and transliteration entries.
- The surface-rules layer is hand-curated, so it improves exactly the cases it has seen —
  it is a safety net, not a general corrector.
- There are **no automated tests** yet; the obvious first targets are `affect.compute_avd`,
  `verifier.verify_and_clean_output`, and the entity mask/unmask round-trip.
- Evaluation is currently offline/manual (the `ab_translate` harness + MOS logging);
  folding those into a reported metric is the natural next step.
