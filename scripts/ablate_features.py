"""
Cost/quality ablation harness for NPChat's RAG features.

Goal: for each per-query feature, measure (a) its TIME cost and (b) its QUALITY
impact when disabled — judged objectively by whether the same SOURCE URLs surface.

Design decisions (see investigation notes):
- ONE chain instance is built with everything ON, then feature booleans are
  toggled as attributes between runs. No reconstruction, no embedder reload.
- Baseline = the REAL persisted runtime config (data/app_settings.json), which
  has HyDE + multi-query + expansion + reranking + confidence-eval all ON.
- Each config runs N times; we report MEDIAN wall time and discard the first
  (warm-up) run. Variance is itself a finding (single-slot server queueing).
- The OpenAI client is monkeypatched to (1) count LLM calls and (2) inject
  `cache_prompt: false` so llama-server prefix-caching can't fake fast times.
  We run the whole matrix BOTH with caching disabled (true feature cost) and
  enabled (realistic user experience) — controlled by NPCHAT_DISABLE_PREFIX_CACHE.
- Feedback-based reordering and query logging are neutralized so they can't
  shift sources or add time mid-ablation.

This does NOT change chunking, the embedding model, or ranking logic. It only
toggles existing feature booleans and adds instrumentation.

Usage:
    python -m scripts.ablate_features
"""
import os
import sys
import json
import time
import statistics
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Quiet the httpx INFO logs so the breakdown is readable.
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)

DISABLE_PREFIX_CACHE = os.environ.get("NPCHAT_DISABLE_PREFIX_CACHE", "1") == "1"
N_REPS = int(os.environ.get("NPCHAT_REPS", "3"))
# Incremental results dumped here after EACH config so a kill mid-run loses nothing.
RESULTS_JSON = Path(__file__).resolve().parent.parent / "data" / "ablation_results.json"

# Varied test set: easy (direct keyword), hard-recall (paraphrase / synonym gap
# where expansion & multi-query are supposed to earn their keep), and
# unanswerable (should retrieve weakly / answer "I don't know").
TEST_QUESTIONS = [
    ("easy",        "vocês fazem outsystems?"),
    ("hard_recall", "como é que ajudam empresas a modernizar sistemas antigos?"),
    ("hard_recall", "do you help with customer relationship management tooling?"),
    ("unanswerable","qual é a capital da Austrália?"),
]

LLM_CALL_COUNT = {"n": 0}


def _patch_llm_client():
    """Count LLM calls and force cache_prompt:false when configured."""
    from openai.resources.chat import completions as _completions_mod
    original_create = _completions_mod.Completions.create

    def counted_create(self, *args, **kwargs):
        LLM_CALL_COUNT["n"] += 1
        if DISABLE_PREFIX_CACHE:
            extra = dict(kwargs.get("extra_body") or {})
            extra["cache_prompt"] = False
            kwargs["extra_body"] = extra
        return original_create(self, *args, **kwargs)

    _completions_mod.Completions.create = counted_create


def _source_urls(result) -> tuple:
    """Sorted tuple of source URLs — the objective quality signal."""
    return tuple(sorted(s.get("url", "") for s in result.sources if s.get("url")))


def run_config(chain, label, flags, questions):
    """
    Apply feature flags to the chain, run each question N_REPS times, and return
    a per-question record: median wall time, LLM call count, source URLs, answer.
    """
    # Toggle features as attributes (no reconstruction).
    chain.use_query_expansion = flags["expansion"]
    chain.use_multi_query = flags["multi_query"]
    chain.use_reranking = flags["reranking"]
    chain.use_hybrid_search = flags["hybrid"]

    print(f"\n{'=' * 72}\n  CONFIG: {label}\n  flags: {flags}  hyde={flags['hyde']} eval={flags['eval']}\n{'=' * 72}")

    records = []
    for kind, q in questions:
        times = []
        n_calls = None
        urls = None
        answer = None
        for rep in range(N_REPS + 1):  # +1 warm-up that we discard
            LLM_CALL_COUNT["n"] = 0
            t0 = time.time()
            result = chain.query(
                question=q,
                top_k=5,
                temperature=0.7,
                use_hyde=flags["hyde"],
                conversation_history=None,
                evaluate_confidence=flags["eval"],
            )
            wall = time.time() - t0
            if rep > 0:  # discard warm-up
                times.append(wall)
            n_calls = LLM_CALL_COUNT["n"]
            urls = _source_urls(result)
            answer = result.answer
        median = round(statistics.median(times), 1)
        lo, hi = round(min(times), 1), round(max(times), 1)
        print(f"  [{kind:11}] median={median:>5.1f}s  range={lo}-{hi}s  "
              f"llm_calls={n_calls}  sources={len(urls)}")
        records.append({
            "kind": kind, "q": q, "median": median, "range": (lo, hi),
            "llm_calls": n_calls, "urls": urls, "answer": answer,
        })
    return records


def main():
    _patch_llm_client()
    from src.generation.enhanced_rag_chain import EnhancedRAGChain

    print(f"prefix_cache_disabled={DISABLE_PREFIX_CACHE}  reps={N_REPS} (+1 warm-up discarded)")

    chain = EnhancedRAGChain(
        use_query_expansion=True,
        use_hybrid_search=True,
        use_logging=False,           # no DB writes
        use_reranking=True,
        use_multi_query=True,
    )
    # Neutralize feedback reordering so sources don't shift mid-ablation.
    chain.feedback_learner.apply_adjustments_to_results = lambda chunks: chunks

    # Warm up lazy FlashRank load so it isn't charged to the first config.
    print("\n[warmup] priming reranker + caches...")
    chain.query(question="warmup", top_k=5, temperature=0.7, use_hyde=False,
                conversation_history=None, evaluate_confidence=False)

    # Live ablation matrix. hyde/eval are passed per-call (not chain attrs).
    # BASELINE reflects the REAL persisted config: hyde ON, eval ON, multi ON, rerank ON.
    base = {"expansion": True, "multi_query": True, "reranking": True,
            "hybrid": True, "hyde": True, "eval": True}

    configs = [
        ("BASELINE (real config: hyde+multi+rerank+eval)", base),
        ("no eval (confidence off)",        {**base, "eval": False}),
        ("no hyde (no pre-retrieval LLM)",  {**base, "hyde": False, "expansion": False}),
        ("no multi-query",                  {**base, "multi_query": False}),
        ("no reranking",                    {**base, "reranking": False}),
        ("no hyde + no multi-query",        {**base, "hyde": False, "expansion": False, "multi_query": False}),
        ("LEAN (only hybrid+rerank+gen)",   {"expansion": False, "multi_query": False,
                                             "reranking": True, "hybrid": True,
                                             "hyde": False, "eval": False}),
        ("MINIMAL (hybrid+gen only)",       {"expansion": False, "multi_query": False,
                                             "reranking": False, "hybrid": True,
                                             "hyde": False, "eval": False}),
    ]

    all_results = {}
    for label, flags in configs:
        all_results[label] = run_config(chain, label, flags, TEST_QUESTIONS)
        # Persist after EVERY config so an interrupted run keeps its partial matrix.
        # urls are tuples → lists for JSON; drop the long answer text to keep it small.
        dump = {
            lbl: [{k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in r.items() if k != "answer"} for r in recs]
            for lbl, recs in all_results.items()
        }
        RESULTS_JSON.write_text(json.dumps(dump, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  [saved {len(all_results)}/{len(configs)} configs -> {RESULTS_JSON.name}]")

    # Quality table: compare source URLs to BASELINE per question.
    print(f"\n\n{'#' * 72}\n#  QUALITY: source-URL overlap vs BASELINE (per question)\n{'#' * 72}")
    baseline = {r["q"]: r for r in all_results["BASELINE (real config: hyde+multi+rerank+eval)"]}
    for label, recs in all_results.items():
        print(f"\n  {label}")
        for r in recs:
            b = baseline[r["q"]]
            same = "SAME " if r["urls"] == b["urls"] else "DIFF "
            shared = len(set(r["urls"]) & set(b["urls"]))
            print(f"    [{r['kind']:11}] {same} shared={shared}/{len(b['urls'])}  "
                  f"median={r['median']}s  calls={r['llm_calls']}")

    print(f"\n\n{'#' * 72}\n#  TIME SUMMARY: median wall by config (averaged over questions)\n{'#' * 72}")
    for label, recs in all_results.items():
        med = round(statistics.median([r["median"] for r in recs]), 1)
        print(f"  {med:>5.1f}s   {label}")


if __name__ == "__main__":
    main()
