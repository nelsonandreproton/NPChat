# NPChat - Near Partner RAG Chatbot

A production-ready RAG (Retrieval-Augmented Generation) chatbot built with Python, Ollama, ChromaDB, and Streamlit. Features advanced ML capabilities including hybrid search, query expansion, response caching, and automatic feedback learning.

## Features

### Core RAG Pipeline
- **Web Scraper**: Incremental blog post scraping from nearpartner.com
- **Document Ingestion**: Chunking, embedding, and storage in ChromaDB
- **Semantic Search**: Vector similarity search using Ollama embeddings
- **LLM Generation**: Response generation using local Ollama models

### Advanced ML Features
- **Hybrid Search**: Combines semantic (embedding) search with BM25 keyword search
- **Query Expansion**: Automatically expands queries with related terms
- **HyDE**: Hypothetical Document Embedding for improved retrieval
- **Response Caching**: SQLite-based cache to reduce LLM calls

### Automatic Feedback Learning
- **Cache Invalidation**: Removes cached responses on negative feedback
- **Chunk Boosting/Penalizing**: Adjusts retrieval scores based on feedback
- **Auto-Flagging**: Flags queries with repeated negative feedback for review
- **Query Mapping**: Learns successful query-chunk mappings from positive feedback

### Analytics & Monitoring
- **Query Logging**: Tracks all queries with retrieval scores and response times
- **Feedback Analytics**: Monitors positive/negative feedback trends
- **Knowledge Gap Detection**: Identifies low-score queries indicating missing content
- **Learning Statistics**: Shows chunk adjustments and flagged queries

## Tech Stack

- **LLM**: Ollama (gemma2:2b, mistral:7b, or other models)
- **Embeddings**: nomic-embed-text via Ollama
- **Vector Store**: ChromaDB
- **Backend**: FastAPI
- **Frontend**: Streamlit
- **Database**: SQLite (for analytics, cache, feedback)

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

5. **Ingest content** (if you have scraped data)
```bash
python scripts/ingest_blogs.py
```

## Usage

### Run the Main App
```bash
streamlit run app/main_app.py
```

This opens a unified interface with 4 tabs:
- **Chat**: Ask questions about Near Partner
- **Analytics**: View query logs, feedback, and performance metrics
- **ChromaDB**: Browse and search the knowledge base
- **Settings**: Configure ML features and manage knowledge base

### Configuration

Edit `src/config.py` to change:
```python
llm_model: str = "gemma2:2b"        # LLM model
embedding_model: str = "nomic-embed-text"  # Embedding model
top_k: int = 3                       # Chunks to retrieve
chunk_size: int = 600                # Tokens per chunk
```

### Settings (via UI)

| Setting | Description |
|---------|-------------|
| Query Expansion | Expand queries with related terms |
| Hybrid Search | Combine semantic + keyword search |
| HyDE | Use hypothetical document embedding |
| Response Caching | Cache responses to reduce LLM calls |
| top_k | Number of chunks to retrieve |
| Temperature | LLM creativity (0-1) |

## Project Structure

```
NPChat/
├── app/
│   └── main_app.py          # Unified Streamlit app
├── scripts/
│   ├── ingest_blogs.py      # Ingestion script
│   └── setup_ollama.sh      # Ollama setup script
├── src/
│   ├── analytics/
│   │   ├── query_logger.py      # Query logging
│   │   └── response_cache.py    # Response caching
│   ├── api/
│   │   ├── main.py              # FastAPI app
│   │   ├── routes.py            # API routes
│   │   └── schemas.py           # Pydantic schemas
│   ├── feedback/
│   │   ├── feedback_learner.py  # Automatic learning
│   │   ├── models.py            # Feedback models
│   │   └── store.py             # Feedback storage
│   ├── generation/
│   │   ├── enhanced_rag_chain.py  # Main RAG pipeline
│   │   ├── llm.py                 # Ollama wrapper
│   │   └── prompts.py             # Prompt templates
│   ├── ingestion/
│   │   ├── chunker.py           # Text chunking
│   │   ├── embedder.py          # Embedding generation
│   │   └── ingest.py            # Ingestion pipeline
│   ├── retrieval/
│   │   ├── hybrid_retriever.py  # Hybrid search
│   │   ├── query_expansion.py   # Query expansion
│   │   ├── retriever.py         # Base retriever
│   │   └── vector_store.py      # ChromaDB wrapper
│   └── config.py                # Configuration
├── scraper.py                   # Blog scraper
├── scrape_company_pages.py      # Company page scraper
└── requirements.txt
```

## Feedback Learning System

The app automatically learns from user feedback:

| Feedback | Actions Taken |
|----------|---------------|
| **👍 Positive** | Boost chunk scores (+0.1), learn query mapping |
| **👎 Negative** | Invalidate cache, penalize chunks (-0.15), track for flagging |
| **2+ 👎** | Auto-flag query for review |

View learning statistics in **Analytics > Learning** tab.

## API Usage

Start the API server:
```bash
uvicorn src.api.main:app --reload
```

### Endpoints

```
POST /api/query
{
  "question": "What services does Near Partner offer?",
  "top_k": 5
}

GET /api/health
GET /api/stats
```

## Development

### Adding New Content

1. **Scrape new posts**:
```bash
python scraper.py
```

2. **Ingest to knowledge base**:
```bash
python scripts/ingest_blogs.py
```

Or use the **Settings** tab in the app for one-click updates.

### Customizing Prompts

Edit `src/generation/prompts.py`:
```python
SYSTEM_PROMPT = """Your custom system prompt..."""
RAG_PROMPT_TEMPLATE = """Your custom RAG template..."""
```

## Security Considerations

- **API Binding**: By default, the API binds to `127.0.0.1` (localhost only). Change to `0.0.0.0` in `src/config.py` if external access is needed.
- **CORS**: Configured for localhost Streamlit access. Update `src/api/main.py` with your domain for production.
- **No Secrets**: Uses local Ollama - no API keys stored. If adding external APIs, use environment variables via `.env`.
- **Input Validation**: All API inputs validated via Pydantic schemas.
- **SQL**: All database queries use parameterized statements (no SQL injection).

## License

MIT License

## Acknowledgments

- [Ollama](https://ollama.ai) for local LLM inference
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Streamlit](https://streamlit.io/) for the UI
- [LangChain](https://langchain.com/) for RAG utilities
