# NPChat - Near Partner RAG Chatbot

A production-ready RAG (Retrieval-Augmented Generation) chatbot built with Python, llama.cpp, ChromaDB, and Streamlit. Designed to answer any question about Near Partner — its services, values, culture, and team — using content scraped from nearpartner.com. Runs entirely on local infrastructure: no API keys, no data leaves the machine.

## Features

### Core RAG Pipeline
- **Web Scraper**: Incremental blog post and company page scraping from nearpartner.com
- **Document Ingestion**: Chunking (~1200 chars), embedding, and storage in ChromaDB
- **Semantic Search**: Vector similarity search using in-process `mxbai-embed-large-v1` embeddings
- **Cross-Encoder Reranking**: FlashRank reranks the top candidates for higher precision (on by default)
- **LLM Generation**: Response generation via a local llama.cpp server (OpenAI-compatible API, no API keys)

### Advanced ML Features
- **Hybrid Search**: Combines semantic (embedding) search with BM25 keyword search using Reciprocal Rank Fusion (RRF)
- **Multi-Query Retrieval**: Generates query variants to improve recall (on by default)
- **HyDE / Query Expansion**: Hypothetical Document Embedding and term expansion for abstract questions (optional, off by default)
- **Parent-Child Chunking**: Embed small paragraphs, serve larger parent chunks to the LLM (optional; requires re-ingest)
- **Contextual Retrieval**: Prepend LLM-generated context to each chunk before embedding (optional; requires re-ingest)
- **Response Caching**: SQLite-based cache to avoid redundant LLM calls
- **Multi-turn Conversation**: Maintains conversation history for follow-up questions
- **Language Mirroring**: Answers in the language of the question (Portuguese or English)
- **Auto-Quality Evaluation**: LLM self-evaluates response confidence (0.0–1.0); warns on low-confidence answers
- **RAGAS Evaluation Harness**: Reference-free pipeline scoring (faithfulness, answer relevancy, context precision/relevance)

### Automatic Feedback Learning
- **Cache Invalidation**: Removes cached responses on negative feedback
- **Chunk Boosting/Penalizing**: Adjusts retrieval scores based on user feedback (👍 +0.1, 👎 −0.15)
- **Auto-Flagging**: Flags queries with repeated negative feedback for review
- **Query Mapping**: Learns successful query-chunk mappings from positive feedback

### Analytics & Monitoring
- **Query Logging**: Tracks all queries with retrieval scores and response times
- **Feedback Analytics**: Monitors positive/negative feedback trends
- **Knowledge Gap Detection**: Identifies low-score queries indicating missing content
- **Learning Statistics**: Shows chunk adjustments and flagged queries
- **Weekly Reports**: Auto-generated JSON reports saved to `data/reports/`

### Operational Features
- **Rate Limiting**: 30 requests/minute per IP on chat endpoints
- **Prompt Injection Protection**: Input sanitization and system prompt hardening
- **Settings Persistence**: ML feature toggles and parameters persist across page refreshes
- **Automatic Scheduling**: Weekly scrape+ingest, daily cache cleanup (APScheduler)
- **Data Export**: Full export of knowledge base, logs, feedback, and cache
- **Security**: Localhost-only API binding, parameterized SQL, no secrets stored

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | llama.cpp server (Qwen2.5-7B-Instruct-Q4_K_M, OpenAI-compatible API) |
| Embeddings | `mxbai-embed-large-v1` (1024-dim) via sentence-transformers, in-process |
| Reranker | FlashRank cross-encoder (CPU) |
| Vector Store | ChromaDB |
| Keyword Search | BM25 (rank-bm25) + RRF |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Databases | SQLite (analytics, cache, feedback, learning) |
| Scheduler | APScheduler |
| Evaluation | RAGAS |
| Tests | pytest |

## Installation

### Prerequisites
- Python 3.12+
- [llama.cpp](https://github.com/ggml-org/llama.cpp) (`llama-server.exe`) and a GGUF model — by default `Qwen2.5-7B-Instruct-Q4_K_M.gguf`
- Embeddings download automatically on first run from Hugging Face (`mixedbread-ai/mxbai-embed-large-v1`) — no separate server needed

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/nelsonandreproton/NPChat.git
cd NPChat
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Start the local LLM server**

The repo includes `start_llm.ps1`, which launches `llama-server` with the expected `--alias` (must match `llm_model` in `src/config.py`):
```powershell
./start_llm.ps1
```
This starts llama.cpp on `http://localhost:8080/v1` serving Qwen2.5-7B-Instruct (alias `qwen2.5-7b-instruct`). Adjust the model path/flags in the script for your machine.

5. **Scrape Near Partner content**
```bash
python scraper.py                 # Blog posts
python scrape_company_pages.py    # Company/product pages
```

6. **Ingest to knowledge base**
```bash
python scripts/ingest_blogs.py
```

## Usage

### Run the Main App
```bash
streamlit run app/main_app.py
```

This opens a unified interface with 4 tabs:
- **Chat**: Ask questions about Near Partner in Portuguese or English (answers mirror the question's language)
- **Analytics**: View query logs, feedback, and performance metrics
- **ChromaDB**: Browse and search the knowledge base
- **Settings**: Configure ML features and manage the knowledge base (settings persist across refreshes)

### Run the API Server
```bash
uvicorn src.api.main:app --reload
```

### API Endpoints

```
POST /api/v1/chat
{
  "message": "Quais são os serviços da Near Partner?",
  "top_k": 5,
  "temperature": 0.7,
  "conversation_history": []
}

POST /api/v1/feedback
GET  /api/v1/health
GET  /api/v1/stats
GET  /api/v1/sources
```

### Evaluate the Pipeline (RAGAS)
```bash
python scripts/evaluate_rag.py                     # built-in Near Partner question set
python scripts/evaluate_rag.py --questions q.json  # custom questions
python scripts/evaluate_rag.py --dry-run           # print questions only, no LLM calls
```
Reference-free metrics (faithfulness, answer relevancy, context precision/relevance) are scored using the local llama.cpp server. Results are written to `data/eval_results/<timestamp>.json` and printed as a summary table.

### Export Data
```bash
python scripts/export_data.py
```
Creates a timestamped export in `data/exports/` with the knowledge base, all SQLite tables, and a manifest.

### Run Tests
```bash
pytest tests/
```

## Configuration

Edit `src/config.py`:
```python
# LLM server (llama.cpp, OpenAI-compatible API)
llm_base_url: str = "http://localhost:8080/v1"
llm_model: str = "qwen2.5-7b-instruct"     # must match --alias on llama-server
embedding_model: str = "mxbai-embed-large-v1"  # in-process via sentence-transformers

# Chunking (characters; ~1200 chars ≈ 300-400 tokens)
chunk_size: int = 1200
chunk_overlap: int = 200

# Retrieval
top_k: int = 3                          # Chunks to retrieve (fewer = faster)
use_reranking: bool = True              # FlashRank cross-encoder reranking
rerank_top_k_candidates: int = 20       # Candidates fetched before reranking
use_multi_query: bool = True            # Generate query variants (UI default overrides to off — see below)
use_contextual_retrieval: bool = False  # Prepend LLM context to chunks (requires re-ingest)
use_parent_child_chunking: bool = False # Embed paragraphs, serve parents (requires re-ingest)

# API
api_host: str = "127.0.0.1"             # Localhost only (change to 0.0.0.0 for external)
api_port: int = 8000
```

> **Note**: `config.py` holds the dataclass defaults, but the running Streamlit app's effective settings come from `app/main_app.py` (`_DEFAULT_SETTINGS`) and the persisted `data/app_settings.json`, which override the dataclass. The effective runtime default is the **LEAN** config below.

### Default configuration & benchmarking

A latency + feature ablation study ([`docs/latency_ablation_findings.md`](docs/latency_ablation_findings.md)) drove the default feature set. Key result: **a query's wall-clock time is 100% LLM-call time** on the single-slot llama.cpp server (~12 tok/s). Every pre-retrieval feature (HyDE, multi-query, query expansion) adds a full LLM call (~3–12s) for little-to-no recall gain on this hardware, so they default **off**. Cross-encoder reranking (~2s CPU) is the highest-value retrieval step — removing it explodes results to 6–10 sources and destroys top-k precision — so it stays **on**.

The shipped **LEAN** default = hybrid search + reranking + generation = **1 LLM call**, ~18–25s/query (vs ~30–60s with everything on), with equal-or-better answers.

| Feature | Default | Verdict from benchmarking |
|---------|---------|---------------------------|
| Hybrid search | **on** | Cheap, no LLM call, feeds the reranker |
| Cross-encoder reranking | **on** | Highest value; ~2s; removing it destroys top-k precision |
| Response caching | **on** | Free; speeds up repeat queries |
| Contextual retrieval | **on** | Ingest-time only — zero query-time cost |
| HyDE | off | Saves ~8–12s; added variance, not recall |
| Multi-query | off | Saves ~3–4s + 4× embed; no recall gain on tested queries |
| Query expansion | off | One pre-retrieval LLM call for marginal recall |
| Evaluate confidence | off | Opt-in; one extra LLM call |

Toggle any of these per-session in the **Settings** tab; changes persist to `data/app_settings.json`.

### Settings (via UI)

| Setting | Description |
|---------|-------------|
| Query Expansion | Expand queries with related terms |
| Hybrid Search | Combine semantic + BM25 keyword search |
| Multi-Query | Generate query variants to improve recall |
| HyDE | Use hypothetical document embedding |
| Reranking | Cross-encoder reranking of top candidates |
| Response Caching | Cache responses to reduce LLM calls |
| Evaluate Confidence | Auto-score response quality (0–1) |
| Show Confidence | Display confidence score in chat |
| top_k | Number of chunks to retrieve |
| Temperature | LLM creativity (0.0–1.0) |

## Project Structure

```
NPChat/
├── app/
│   └── main_app.py              # Unified Streamlit app (4 tabs)
├── scripts/
│   ├── ingest_blogs.py          # Ingestion script
│   ├── update_knowledge.py      # Scrape + re-ingest pipeline
│   ├── evaluate_rag.py          # RAGAS evaluation runner
│   ├── export_data.py           # Data export tool
│   ├── ablate_features.py       # Feature ablation experiments
│   ├── diagnose_latency.py      # Latency breakdown profiler
│   └── probe_hyde_determinism.py
├── src/
│   ├── analytics/
│   │   ├── query_logger.py      # Query logging (SQLite)
│   │   └── response_cache.py    # Response caching (SQLite)
│   ├── api/
│   │   ├── main.py              # FastAPI app + rate limiting middleware
│   │   ├── routes.py            # API routes (EnhancedRAGChain)
│   │   └── schemas.py           # Pydantic schemas (with conversation history)
│   ├── evaluation/
│   │   └── ragas_evaluator.py   # RAGAS reference-free evaluation harness
│   ├── feedback/
│   │   ├── feedback_learner.py  # Automatic learning from feedback
│   │   ├── learner.py           # Learning helpers
│   │   ├── models.py            # Feedback models
│   │   └── store.py             # Feedback storage
│   ├── generation/
│   │   ├── enhanced_rag_chain.py  # Main RAG pipeline + confidence eval
│   │   ├── rag_chain.py           # Base RAG chain
│   │   ├── llm.py                 # llama.cpp client (OpenAI-compatible)
│   │   └── prompts.py             # PT/EN prompts + injection protection
│   ├── ingestion/
│   │   ├── chunker.py           # Text chunking (character-based)
│   │   ├── contextualizer.py    # LLM context prepending (contextual retrieval)
│   │   ├── embedder.py          # In-process sentence-transformers embeddings
│   │   └── ingest.py            # Ingestion pipeline
│   ├── retrieval/
│   │   ├── hybrid_retriever.py  # Hybrid search (semantic + BM25 + RRF)
│   │   ├── query_expansion.py   # Query expansion + HyDE + multi-query
│   │   ├── reranker.py          # FlashRank cross-encoder reranker
│   │   ├── retriever.py         # Base semantic retriever
│   │   └── vector_store.py      # ChromaDB wrapper
│   ├── scheduler.py             # APScheduler background jobs
│   └── config.py                # Central configuration
├── tests/                       # pytest suite (chunker, embedder, reranker, RAGAS, …)
├── scraper.py                   # Blog post scraper
├── scrape_company_pages.py      # Company page scraper (13 pages)
├── start_llm.ps1                # Launch local llama.cpp server
└── requirements.txt
```

## Feedback Learning System

The app automatically learns from user feedback:

| Feedback | Actions Taken |
|----------|---------------|
| **👍 Positive** | Boost chunk scores (+0.1), learn query mapping, keep cache |
| **👎 Negative** | Invalidate cache, penalize chunks (-0.15), track for flagging |
| **2+ 👎 on same query** | Auto-flag query for human review |

View learning statistics in the **Analytics > Learning** tab.

## Automatic Scheduling

When running the Streamlit app, a background scheduler starts automatically:

| Schedule | Job |
|----------|-----|
| Monday 02:00 | Full update: scrape blog + company pages, re-ingest |
| Daily 03:00 | Clear expired cache entries |
| Sunday 23:00 | Generate weekly analytics report |

Disable in Settings or by not starting the scheduler.

## Security Considerations

- **API Binding**: Binds to `127.0.0.1` by default. Change to `0.0.0.0` in `src/config.py` for external access.
- **CORS**: Configured for localhost Streamlit. Update `src/api/main.py` for production domains.
- **Rate Limiting**: 30 requests/minute per IP on chat endpoints (in-memory sliding window).
- **Prompt Injection**: Input sanitized (max 1000 chars, null bytes removed) + system prompt instructs the model to ignore manipulation attempts.
- **No Secrets**: Uses a local llama.cpp server and in-process embeddings — no API keys. If adding external APIs, use `.env` with `python-dotenv`.
- **SQL**: All queries use parameterized statements (no SQL injection).

## License

MIT License

## Acknowledgments

- [llama.cpp](https://github.com/ggml-org/llama.cpp) for local LLM inference
- [mixedbread-ai/mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1) for embeddings
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) for cross-encoder reranking
- [Streamlit](https://streamlit.io/) for the UI
- [LangChain](https://langchain.com/) for RAG utilities
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) for keyword search
- [RAGAS](https://github.com/explodinggradients/ragas) for evaluation
