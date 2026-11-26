import asyncio, sys, time, types
# Stub the heavy modules so pipeline.py can be imported on its own
torch = types.ModuleType("torch"); torch.cuda = types.SimpleNamespace(is_available=lambda: False, ipc_collect=lambda: None)
sys.modules["torch"] = torch
em = types.ModuleType("entity_masker")
em.mask_named_entities = lambda t: (t, {})
em.unmask_and_restore_entities = lambda t, l, g: t
sys.modules["entity_masker"] = em
re_ = types.ModuleType("rules_engine")
class RulesEngine:
    def __init__(self, p): pass
    def apply_conversational_surface_rules(self, t, g): return t
re_.RulesEngine = RulesEngine
sys.modules["rules_engine"] = re_
tts = types.ModuleType("tts")
class Disp:
    primary_engine = "edge"
    def __init__(self, primary_engine="edge"): self.primary_engine = primary_engine
    async def synthesize(self, **kw): return b"audio", {"engine_used": "edge"}
tts.TTSDispatcher = Disp
sys.modules["tts"] = tts
cfg = types.ModuleType("config")
cfg.ROUTING_CONFIG = {}; cfg.NO_TAG_LABELS = set(); cfg.TTS_HARD_TIMEOUT_SEC = 25.0
sys.modules["config"] = cfg
sys.path.insert(0, ".")
import pipeline as P
P._is_edge_reachable = lambda *a, **k: True

active = {"n": 0, "max": 0}
def slow_translator(text, src, tgt):
    active["n"] += 1; active["max"] = max(active["max"], active["n"])
    time.sleep(1.0)            # stands in for a 1-second translation on the CPU
    active["n"] -= 1
    return text
def fast_emotion(text):
    return {"detected_emo": "neutral"}

async def main():
    pipe = P.STSPipeline(slow_translator, fast_emotion, rules_path="x")
    # Test 1: two requests at once -> translations must not overlap
    r = await asyncio.gather(pipe.process("one", "female", tgt_lang="kn"),
                             pipe.process("two", "female", tgt_lang="kn"))
    print("TEST 1 max concurrent translations:", active["max"], "(expected 1)")
    # Test 2: a request is dropped mid-translation; the next one must still wait for the thread to end
    active["max"] = 0
    t1 = asyncio.create_task(pipe.process("dropped", "female", tgt_lang="kn"))
    await asyncio.sleep(0.3)
    t1.cancel()
    try: await t1
    except asyncio.CancelledError: pass
    t0 = time.perf_counter()
    await pipe.process("next", "female", tgt_lang="kn")
    print(f"TEST 2 max concurrent after a dropped request: {active['max']} (expected 1); "
          f"next request waited {time.perf_counter()-t0:.2f}s")
    # Test 3: queue wait limit -> PipelineBusy instead of an endless wait
    P.NMT_QUEUE_WAIT_SEC = 0.2
    a = asyncio.create_task(pipe.process("holds the lock", "female", tgt_lang="kn"))
    await asyncio.sleep(0.05)
    try:
        await pipe.process("waits too long", "female", tgt_lang="kn")
        print("TEST 3 FAILED: no busy error")
    except P.PipelineBusy as e:
        print("TEST 3 PipelineBusy raised as intended:", str(e)[:50])
    await a
asyncio.run(main())
