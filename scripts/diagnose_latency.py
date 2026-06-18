"""
Latency diagnostic harness for a single RAG query.

Runs the REAL EnhancedRAGChain pipeline in a fresh process (no Streamlit), so we
can cleanly separate one-time construction cost (model loads) from per-query cost.

It does NOT change retrieval logic, chunking, or ranking. It only wraps the
OpenAI client and the Embedder with timers/counters (monkeypatch) to capture:

  - chain construction time (embedder load + BM25 load + reranker is lazy)
  - per-LLM-call CLIENT-SIDE wall time (to compare against server-reported time)
  - per-embedding-call wall time and total embedding count
  - each pipeline phase the chain already prints (expansion, retrieval, rerank, gen)
  - the UNACCOUNTED residual = total - sum(measured phases)

The query is run TWICE: run #1 is cold (first query in the process, pays any
lazy loads like the FlashRank reranker); run #2 is warm (steady state, what a
user actually experiences after the first query).

Usage:
    python -m scripts.diagnose_latency "vocês fazem outsystems?"
    python scripts/diagnose_latency.py            # uses default question
"""
import sys
import time
from pathlib import Path

# Force UTF-8 stdout/stderr. The chain prints non-ASCII (e.g. the '->' arrow and
# Portuguese accents); on Windows the default cp1252 console codec raises
# UnicodeEncodeError. This affects only console output, not the pipeline.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Make `src` importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_QUESTION = "vocês fazem outsystems?"

# ---------------------------------------------------------------------------
# Global counters/log populated by the monkeypatched wrappers.
# ---------------------------------------------------------------------------
LLM_CALLS = []     # list of dicts: {label, client_wall_s, prompt_tokens, completion_tokens}
EMBED_CALLS = []   # list of dicts: {client_wall_s, n_texts, chars}


def _patch_llm_client():
    """
    Wrap OpenAI().chat.completions.create to record client-side wall time and,
    when the server returns usage, the prompt/completion token counts.

    We patch at the openai.resources layer so every call site (llm.py,
    query_expansion.py) is captured without touching their code.
    """
    from openai.resources.chat import completions as _completions_mod

    original_create = _completions_mod.Completions.create

    def timed_create(self, *args, **kwargs):
        t0 = time.time()
        result = original_create(self, *args, **kwargs)
        wall = time.time() - t0

        prompt_tokens = completion_tokens = None
        usage = getattr(result, "usage", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)

        LLM_CALLS.append({
            "client_wall_s": round(wall, 2),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        })
        print(
            f"    [LLM call] client wall={wall:.2f}s  "
            f"prompt_tokens={prompt_tokens}  completion_tokens={completion_tokens}"
        )
        return result

    _completions_mod.Completions.create = timed_create


def _patch_embedder():
    """
    Wrap Embedder.embed_query / embed_text to count calls and time each one.
    This reveals the multi-query fan-out (N embeds per user question).
    """
    from src.ingestion import embedder as _embedder_mod

    original_embed_text = _embedder_mod.Embedder.embed_text

    def timed_embed_text(self, text):
        t0 = time.time()
        result = original_embed_text(self, text)
        wall = time.time() - t0
        EMBED_CALLS.append({
            "client_wall_s": round(wall, 3),
            "chars": len(text),
        })
        print(f"    [embed] wall={wall:.3f}s  chars={len(text)}")
        return result

    _embedder_mod.Embedder.embed_text = timed_embed_text


def _phase_sum(timings: dict) -> float:
    """Sum the per-phase timings the chain records, excluding the 'total' key."""
    return round(sum(v for k, v in timings.items() if k != "total"), 2)


def run_once(rag_chain, question: str, run_label: str):
    """Run a single query and print a full breakdown including the residual."""
    print("\n" + "=" * 70)
    print(f"  {run_label}: querying  ->  {question!r}")
    print("=" * 70)

    # Reset per-run call logs so each run's fan-out is reported independently.
    LLM_CALLS.clear()
    EMBED_CALLS.clear()

    t0 = time.time()
    result = rag_chain.query(
        question=question,
        top_k=3,
        temperature=0.7,
        use_hyde=False,
        conversation_history=None,
        evaluate_confidence=False,
    )
    wall_total = time.time() - t0

    timings = dict(result.timings)
    phase_total = _phase_sum(timings)
    residual = round(wall_total - phase_total, 2)

    print("\n  --- phase breakdown (chain-reported) ---")
    for k, v in timings.items():
        if k == "total":
            continue
        print(f"    {k:<18} {v:>7.2f}s")
    print(f"    {'-' * 26}")
    print(f"    {'phase sum':<18} {phase_total:>7.2f}s")
    print(f"    {'chain total':<18} {timings.get('total', 0):>7.2f}s")
    print(f"    {'wall (harness)':<18} {wall_total:>7.2f}s")
    print(f"    {'UNACCOUNTED':<18} {residual:>7.2f}s   <-- residual not in any phase")

    print("\n  --- LLM calls this run (client-side wall time) ---")
    llm_wall = 0.0
    for i, c in enumerate(LLM_CALLS):
        llm_wall += c["client_wall_s"]
        print(
            f"    call {i}: client_wall={c['client_wall_s']:>6.2f}s  "
            f"prompt_tokens={c['prompt_tokens']}  completion_tokens={c['completion_tokens']}"
        )
    print(f"    LLM calls: {len(LLM_CALLS)}   total client wall: {llm_wall:.2f}s")

    print("\n  --- embedding calls this run ---")
    embed_wall = sum(c["client_wall_s"] for c in EMBED_CALLS)
    print(f"    embed calls: {len(EMBED_CALLS)}   total wall: {embed_wall:.2f}s")

    return wall_total


def main():
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION

    # Patch BEFORE constructing the chain so even construction-time calls (if any)
    # are captured.
    _patch_llm_client()
    _patch_embedder()

    from src.generation.enhanced_rag_chain import EnhancedRAGChain

    print("\n" + "#" * 70)
    print("#  CONSTRUCTION (one-time): EnhancedRAGChain(...)")
    print("#  This pays embedder model load + BM25 index load. Reranker is lazy.")
    print("#" * 70)
    t0 = time.time()
    rag_chain = EnhancedRAGChain(
        use_query_expansion=True,
        use_hybrid_search=True,
        use_logging=False,          # skip DB writes so logging time doesn't pollute query timing
        use_reranking=True,
        use_multi_query=True,
    )
    construct_wall = time.time() - t0
    print(f"\n  >>> construction wall: {construct_wall:.2f}s")

    # Run #1 = cold (pays lazy reranker load on first rerank call).
    cold_wall = run_once(rag_chain, question, "RUN #1 (COLD)")

    # Run #2 = warm (steady state — what a user sees after the first query).
    warm_wall = run_once(rag_chain, question, "RUN #2 (WARM)")

    print("\n" + "#" * 70)
    print("#  SUMMARY")
    print("#" * 70)
    print(f"  construction (one-time)   : {construct_wall:6.2f}s")
    print(f"  run #1 wall (cold)        : {cold_wall:6.2f}s")
    print(f"  run #2 wall (warm)        : {warm_wall:6.2f}s")
    print(f"  cold-vs-warm delta        : {cold_wall - warm_wall:6.2f}s   (lazy loads paid on run #1)")
    print("#" * 70)


if __name__ == "__main__":
    main()
