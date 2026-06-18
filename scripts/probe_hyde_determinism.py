"""
Probe: does HyDE introduce retrieval NON-determinism?

HyDE generates a hypothetical answer doc at temperature 0.5, embeds it, and
retrieves on that embedding. If the generated doc varies run-to-run, the
retrieved source URLs vary too — for the SAME question. This probe runs the
same question several times with HyDE ON and several times with HyDE OFF
(plain query embedding) and reports how stable the retrieved source set is.

Stable retrieval (HyDE off) should return identical URLs every run.
"""
import sys
from pathlib import Path
from collections import Counter

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
logging.getLogger("httpx").setLevel(logging.WARNING)

from src.generation.enhanced_rag_chain import EnhancedRAGChain

QUESTIONS = [
    "como é que ajudam empresas a modernizar sistemas antigos?",
    "vocês fazem outsystems?",
]
RUNS = 4


def source_urls(result):
    return tuple(sorted(s.get("url", "") for s in result.sources if s.get("url")))


def main():
    chain = EnhancedRAGChain(
        use_query_expansion=True, use_hybrid_search=True, use_logging=False,
        use_reranking=True, use_multi_query=False,  # isolate HyDE; multi-query off
    )
    chain.feedback_learner.apply_adjustments_to_results = lambda chunks: chunks
    # warm up reranker
    chain.query(question="warmup", top_k=5, temperature=0.7, use_hyde=False)

    for q in QUESTIONS:
        print(f"\n{'=' * 70}\n  QUESTION: {q}\n{'=' * 70}")
        for hyde in (True, False):
            print(f"\n  --- HyDE={'ON' if hyde else 'OFF'} ({RUNS} runs) ---")
            seen = Counter()
            results = []
            for i in range(RUNS):
                r = chain.query(question=q, top_k=5, temperature=0.7, use_hyde=hyde)
                urls = source_urls(r)
                results.append(urls)
                seen[urls] += 1
                print(f"    run {i}: {len(urls)} sources  {[u.split('/')[-2] for u in urls]}")
            distinct = len(set(results))
            print(f"    => {distinct} DISTINCT source-set(s) across {RUNS} runs "
                  f"({'STABLE' if distinct == 1 else 'NON-DETERMINISTIC'})")


if __name__ == "__main__":
    main()
