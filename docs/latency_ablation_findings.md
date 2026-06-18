# NPChat RAG — Latency & Feature Ablation Findings

**Date:** 2026-06-18
**Hardware:** llama.cpp (`llama-server`, Qwen2.5-7B-Instruct-Q4_K_M, Vulkan, single slot, ctx 8192) on Intel Arc 140T laptop. Decode ≈ 12 tok/s. Embeddings `mixedbread-ai/mxbai-embed-large-v1` (1024-dim) in-process via sentence-transformers.
**Scope:** latency instrumentation + cost/quality ablation of the 8 RAG features. No changes to chunking, embedding model, or retrieval ranking.

---

## TL;DR

- A query's wall-clock time is **100% LLM-call time**. There is no hidden overhead: instrumentation showed the unaccounted residual = **0.00s** every run. Embeddings (~0.4s total) and ChromaDB are negligible; the embedding model is **not** reloaded per query.
- In the real persisted config there are **3 LLM calls** per query, each ≈ 7–12s on this ~12 tok/s single slot: **HyDE → multi-query → generation**.
- The cheapest high-impact change is to drop the two pre-retrieval LLM calls (HyDE + multi-query). **Reranking is the highest-value feature and must stay.**
- **Auto-Quality Eval is broken** — it makes no LLM call and silently returns `None`.
- Target met: a **LEAN config (hybrid + reranking + generation, 1 LLM call)** lands a typical query at **~18–25s** with equal-or-better answers and correct source links.

---

## 1. Where the time goes (root cause)

Instrumented via `scripts/diagnose_latency.py` (fresh process, times construction separately, runs the query twice, prints per-phase + per-LLM-call **client-side** time and the unaccounted residual).

| Phase | Cost | Notes |
|---|---|---|
| Chain construction (model loads) | ~6s | One-time; amortized by Streamlit `@st.cache_resource` after first query |
| Embeddings (4× per query) | ~0.4s total | ~0.09s each; model loaded once, reused — **not** reloaded per query |
| ChromaDB search | negligible | |
| **LLM calls (3×)** | **~28–37s warm** | 100% of query time |
| **Unaccounted residual** | **0.00s** | No hidden time anywhere |

### The 3 LLM calls (maps to llama-server tasks 0 / 55 / 94)

| Server task | Feature | What it is | Typical cost |
|---|---|---|---|
| task 0 | **HyDE** | generates a hypothetical answer doc, embeds it (HyDE wins over expansion — same `if` branch in `enhanced_rag_chain.py`) | ~8–12s |
| task 55 | **Multi-query** | generates 4 query variants → 4× embed + 4× hybrid retrieve | ~3–4s |
| task 94 | **Generation** | the actual answer (~1500-token prompt) | ~14–22s |

### On the "~60s every query" claim
Isolated **warm** queries are ~30s. The 60s figure comes from **single-slot contention** — a small call queued behind another (one multi-query call spiked 3s → 25s). The variance itself is a finding: a single server slot means no concurrency, so anything that adds calls also adds queueing risk. Reducing to 1 LLM call shrinks both the mean and the variance.

---

## 2. Per-feature cost/quality ablation

Method (`scripts/ablate_features.py`): one chain instance, feature booleans toggled as attributes (no reload), ≥2 reps per config (median, first run discarded), feedback-reordering and logging neutralized. **Quality signal = whether the same source URLs surface vs baseline** (objective, cheap). Test set: easy / two hard-recall / one unanswerable. Server prefix-caching controllable via `cache_prompt:false`.

### Median wall time by config (prefix-cache on, 2 reps)

| Config | LLM calls | Median | Quality vs baseline |
|---|---|---|---|
| BASELINE (hyde+multi+rerank+eval) | 3 | 30.3s | reference |
| no eval | 3 | 27.6s | identical retrieval (eval is a no-op) |
| no HyDE | 2 | 24.0s | core sources kept; loses only a marginal boundary source |
| no multi-query | 2 | 28.5s | **same sources** on easy + CRM queries |
| **no reranking** | 3 | **45.1s** | ⚠️ **breaks** — 6–10 sources, top-k precision destroyed (overlap 0–2/N) |
| no HyDE + no multi-query | 1 | 17.9s | core sources kept |
| **LEAN** (hybrid+rerank+gen) | 1 | ~18–25s | core sources kept; **answer equal-or-better** |
| MINIMAL (no rerank) | 1 | ~19s | worse sources (no reranker) |

> Note: some MINIMAL/LEAN unanswerable timings (5–7s) were prefix-cache hits on repeated prompts. The honest single-call figure is **~18–20s** (from the clean "no HyDE + no multi-query" measurement).

---

## 3. Three findings that shaped the recommendation

1. **Auto-Quality Eval (#8) is broken and free.** With `evaluate_confidence=True` it makes **no LLM call** — it references `config.lmstudio_base_url` (a nonexistent attribute, `enhanced_rag_chain.py:260`), throws, and the `try/except` swallows it → returns `None`. Costs ~0s and never produces a confidence score. (This refutes the prior hypothesis that it was a large dispensable cost.)

2. **Reranking is the highest-value feature.** Removing it *worsened* both time (45s — bigger generation prompt) and quality (sources ballooned to 6–10; the precise top results vanished, overlap with baseline collapsed to 0–2/N). It costs only ~1.5–2s of CPU (FlashRank). **Keep it.**

3. **HyDE and multi-query buy almost no recall here.**
   - Multi-query returned *identical* sources to baseline on the easy and CRM queries.
   - HyDE determinism probe (`scripts/probe_hyde_determinism.py`): with HyDE **off**, the easy query is perfectly stable (3/3 identical runs); with HyDE **on**, it's noisy (3-or-4 sources run-to-run) for **no gain** — the relevant sources are present either way. HyDE is the single biggest non-generation cost (~8–12s) for marginal, *noisier* results.
   - Answer-text spot check: LEAN's answers were equal-or-better than BASELINE (the OutSystems answer was actually richer), and the unanswerable query was still correctly declined.

---

## 4. Recommendation (smallest-change-first)

| # | Feature | Verdict | Reasoning / quality lost |
|---|---|---|---|
| 6 | Cross-encoder reranking | **KEEP** | Highest value; ~2s; removing it destroys top-k precision |
| 2 | Hybrid search | **KEEP** | Cheap, no LLM call, feeds the reranker |
| 7 | Response caching | **KEEP** | Free; helps repeat queries |
| 4 | Contextual retrieval | **KEEP** (ingest-only) | Zero query-time cost |
| 5 | HyDE | **DISABLE** (optional toggle, default off) | Saves ~8–12s. Lose: nothing measurable; it added variance, not recall |
| 3 | Multi-query | **DISABLE** (optional toggle, default off) | Saves ~3–4s + 4× embed. Lose: nothing on tested queries |
| 1 | Query expansion | **DISABLE** (optional toggle, default off) | Same class as HyDE; one pre-retrieval LLM call for marginal recall |
| 8 | Auto-quality eval | **FIX or remove** | Broken (wrong base_url). Point it at `config.llm_base_url` or drop the toggle |

### Net result
Default config becomes **LEAN = hybrid + reranking + generation** → **1 LLM call**, typical query **~18–25s** (vs ~30–60s) — meeting the ~20–25s target, near the ~19s server floor for a 2k-context query. Answer quality equal-or-better; source links correct. Keep HyDE / multi-query / expansion as UI toggles (default off) for the rare hard-recall query where the extra seconds are worth it.

---

## 5. Constraints respected

- No changes to chunking, embedding model, or retrieval ranking logic (correctness already validated).
- A feature being slow was **not** treated as sufficient reason to remove it — each disable is justified by a quality-when-disabled judgment, and the one slow-but-valuable feature (reranking) is kept.
- Quality judged on multiple varied queries (answerable + unanswerable), not a single case.
- Ablation bypassed the response cache so cached answers couldn't fake fast times.

## 6. Artifacts (instrumentation only — safe to keep or delete)

- `scripts/diagnose_latency.py` — per-phase + per-LLM-call timing, unaccounted residual, cold vs warm.
- `scripts/ablate_features.py` — feature toggle matrix, median timing, source-URL quality, incremental JSON.
- `scripts/probe_hyde_determinism.py` — HyDE on/off retrieval stability.
- `data/ablation_results.json`, `data/ablation_cached.txt`, `data/hyde_probe.txt` — raw results.

## 7. Bug found (separate from tuning)

`src/generation/enhanced_rag_chain.py:260` references `config.lmstudio_base_url`, which does not exist in `Config`. The auto-quality evaluation path therefore always fails silently. Fix: use `config.llm_base_url` (and drop the unused `api_key="lm-studio"` legacy value), or remove the feature.
