"""
Central configuration for the Near Partner RAG Chatbot.
"""
import os
from pathlib import Path
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()  # read .env if present (gitignored); env vars still win

# --------------------------------------------------------------------------- #
# LLM backend profiles                                                         #
# --------------------------------------------------------------------------- #
# Both backends speak the OpenAI-compatible API, so only base_url + model id
# change between them. Switch with LLM_BACKEND=local|runpod in .env (or the
# environment). Defaults below keep behavior identical to before (local).
_LLM_BACKENDS = {
    # Local llama.cpp server (start_llm.ps1). Model id = the --alias.
    "local": {
        "base_url": "http://localhost:8080/v1",
        "model": "qwen2.5-7b-instruct",
    },
    # RunPod vLLM (OpenAI server). Model id = vLLM's served id, NOT the HF repo
    # path — verify with: curl <base_url>/models  (the "id" field).
    "runpod": {
        "base_url": "https://wetcua2lntx03a-8000.proxy.runpod.net/v1",
        "model": "qwen3-coder",
    },
}


@dataclass
class Config:
    """Application configuration."""

    # Paths
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Path = base_dir / "data"
    chroma_db_path: Path = data_dir / "chroma_db"
    feedback_db_path: Path = data_dir / "feedback.db"
    blog_posts_path: Path = base_dir / "nearpartner_blog_posts.json"
    company_pages_path: Path = base_dir / "nearpartner_company_pages.json"

    # LLM server (OpenAI-compatible API). Resolved from LLM_BACKEND in
    # __post_init__; the defaults here are the "local" profile.
    llm_backend: str = "local"
    llm_base_url: str = "http://localhost:8080/v1"
    llm_model: str = "qwen2.5-7b-instruct"        # must match --alias on llama-server
    # Embeddings run in-process via sentence-transformers (no server needed)
    embedding_model: str = "mxbai-embed-large-v1"  # HF: mixedbread-ai/mxbai-embed-large-v1

    # Chunking settings (in characters; ~1200 chars ≈ 300-400 tokens)
    chunk_size: int = 1200
    chunk_overlap: int = 200

    # Retrieval settings
    top_k: int = 3  # Number of chunks to retrieve (less = faster)
    similarity_threshold: float = 0.0  # Disabled - let LLM decide relevance
    use_reranking: bool = True  # Cross-encoder reranking via FlashRank
    rerank_top_k_candidates: int = 20  # Candidates fetched before reranking
    use_multi_query: bool = True  # Generate query variants to improve recall
    use_contextual_retrieval: bool = False  # Prepend LLM context to each chunk before embedding (requires full re-ingest)
    use_parent_child_chunking: bool = False  # Embed paragraphs, serve parent chunks to LLM (requires full re-ingest)

    # ChromaDB collection name
    collection_name: str = "nearpartner_knowledge"

    # API settings
    # Use 127.0.0.1 for local-only access (more secure)
    # Change to 0.0.0.0 if you need external network access
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    def __post_init__(self):
        """Resolve the LLM backend and ensure directories exist."""
        self._resolve_llm_backend()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_db_path.mkdir(parents=True, exist_ok=True)

    def _resolve_llm_backend(self):
        """
        Pick the LLM backend from LLM_BACKEND (default 'local') and set
        llm_base_url / llm_model accordingly. A pod id can change without a
        code edit via the per-backend env overrides below.
        """
        backend = os.getenv("LLM_BACKEND", "local").strip().lower()
        profile = _LLM_BACKENDS.get(backend)
        if profile is None:
            raise ValueError(
                f"Unknown LLM_BACKEND={backend!r}. "
                f"Expected one of: {', '.join(_LLM_BACKENDS)}"
            )

        prefix = backend.upper()  # LOCAL_* / RUNPOD_*
        self.llm_backend = backend
        self.llm_base_url = os.getenv(f"{prefix}_LLM_BASE_URL", profile["base_url"])
        self.llm_model = os.getenv(f"{prefix}_LLM_MODEL", profile["model"])


# Global config instance
config = Config()
