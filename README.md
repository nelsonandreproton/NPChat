# NPChat - Near Partner RAG Chatbot

A production-ready RAG (Retrieval-Augmented Generation) chatbot built with Python, Ollama, ChromaDB, and Streamlit. Designed to answer any question about Near Partner — its services, values, culture, and team — using content scraped from nearpartner.com.

## Features

### Core RAG Pipeline
- **Web Scraper**: Incremental blog post and company page scraping from nearpartner.com
- **Document Ingestion**: Chunking (~1200 chars), embedding, and storage in ChromaDB
- **Semantic Search**: Vector similarity search using Ollama embeddings
- **LLM Generation**: Response generation using local Ollama models (no API keys)

### Advanced ML Features
- **Hybrid Search**: Combines semantic (embedding) search with BM25 keyword search using Reciprocal Rank Fusion (RRF)
- **Query Expansion**: Automatically expands queries with related terms for broader retrieval
- **HyDE**: Hypothetical Document Embedding for improved retrieval on abstract questions
- **Response Caching**: SQLite-based cache to avoid redundant LLM calls
- **Multi-turn Conversation**: Maintains conversation history for follow-up questions
- **Auto-Quality Evaluation**: LLM self-evaluates response confidence (0.0–1.0); warns on low-confidence answers

### Automatic Feedback Learning
- **Cache Invalidation**: Removes cached responses on negative feedback
- **Chunk Boosting/Penalizing**: Adjusts retrieval scores based on user feedback
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
- **Automatic Scheduling**: Weekly scrape+ingest, daily cache cleanup (APScheduler)
- **Data Export**: Full export of knowledge base, logs, feedback, and cache
- **Security**: Localhost-only API binding, parameterized SQL, no secrets stored

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Ollama (gemma2:2b, mistral:7b, …) |
| Embeddings | nomic-embed-text via Ollama |
| Vector Store | ChromaDB |
| Keyword Search | BM25 (rank-bm25) |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Databases | SQLite (analytics, cache, feedback, learning) |
| Scheduler | APScheduler |
| Tests | pytest |

## Installation

### Prerequisites
- Python 3.12+
- [Ollama](https://ollama.ai) installed and running

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

4. **Pull Ollama models**
```bash
ollama pull nomic-embed-text
ollama pull gemma2:2b  # or mistral:7b for better quality
```

5. **Scrape Near Partner content**
```bash
python scraper.py                 # Blog posts
python scrape_company_pages.py   # Company/product pages
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
- **Chat**: Ask questions about Near Partner in Portuguese or English
- **Analytics**: View query logs, feedback, and performance metrics
- **ChromaDB**: Browse and search the knowledge base
- **Settings**: Configure ML features and manage the knowledge base

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
llm_model: str = "gemma2:2b"          # LLM model
embedding_model: str = "nomic-embed-text"
top_k: int = 3                         # Chunks to retrieve
chunk_size: int = 1200                 # Characters per chunk (~300-400 tokens)
chunk_overlap: int = 200               # Overlap between chunks
api_host: str = "127.0.0.1"           # Localhost only (change to 0.0.0.0 for external)
api_port: int = 8000
```

### Settings (via UI)

| Setting | Description |
|---------|-------------|
| Query Expansion | Expand queries with related terms |
| Hybrid Search | Combine semantic + BM25 keyword search |
| HyDE | Use hypothetical document embedding |
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
│   ├── export_data.py           # Data export tool
│   └── setup_ollama.sh          # Ollama setup script
├── src/
│   ├── analytics/
│   │   ├── query_logger.py      # Query logging (SQLite)
│   │   └── response_cache.py    # Response caching (SQLite)
│   ├── api/
│   │   ├── main.py              # FastAPI app + rate limiting middleware
│   │   ├── routes.py            # API routes (EnhancedRAGChain)
│   │   └── schemas.py           # Pydantic schemas (with conversation history)
│   ├── feedback/
│   │   ├── feedback_learner.py  # Automatic learning from feedback
│   │   ├── models.py            # Feedback models
│   │   └── store.py             # Feedback storage
│   ├── generation/
│   │   ├── enhanced_rag_chain.py  # Main RAG pipeline + confidence eval
│   │   ├── llm.py                 # Ollama wrapper
│   │   └── prompts.py             # Portuguese prompts + injection protection
│   ├── ingestion/
│   │   ├── chunker.py           # Text chunking (character-based)
│   │   ├── embedder.py          # Embedding generation
│   │   └── ingest.py            # Ingestion pipeline
│   ├── retrieval/
│   │   ├── hybrid_retriever.py  # Hybrid search (semantic + BM25 + RRF)
│   │   ├── query_expansion.py   # Query expansion + HyDE
│   │   ├── retriever.py         # Base semantic retriever
│   │   └── vector_store.py      # ChromaDB wrapper
│   ├── scheduler.py             # APScheduler background jobs
│   └── config.py                # Central configuration
├── tests/
│   ├── test_chunker.py          # Chunker unit tests
│   ├── test_prompts.py          # Prompt builder tests
│   ├── test_feedback_learner.py # Feedback learning tests
│   └── test_response_cache.py  # Cache tests
├── scraper.py                   # Blog post scraper
├── scrape_company_pages.py      # Company page scraper (13 pages)
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
- **No Secrets**: Uses local Ollama — no API keys. If adding external APIs, use `.env` with `python-dotenv`.
- **SQL**: All queries use parameterized statements (no SQL injection).

## License

MIT License

## Acknowledgments

- [Ollama](https://ollama.ai) for local LLM inference
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Streamlit](https://streamlit.io/) for the UI
- [LangChain](https://langchain.com/) for RAG utilities
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) for keyword search
