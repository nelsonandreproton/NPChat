"""
Central configuration for the Near Partner RAG Chatbot.
"""
from pathlib import Path
from dataclasses import dataclass


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

    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "gemma2:2b"  # Faster on CPU
    embedding_model: str = "nomic-embed-text"

    # Chunking settings (in characters; ~1200 chars ≈ 300-400 tokens)
    chunk_size: int = 1200
    chunk_overlap: int = 200

    # Retrieval settings
    top_k: int = 3  # Number of chunks to retrieve (less = faster)
    similarity_threshold: float = 0.0  # Disabled - let LLM decide relevance

    # ChromaDB collection name
    collection_name: str = "nearpartner_knowledge"

    # API settings
    # Use 127.0.0.1 for local-only access (more secure)
    # Change to 0.0.0.0 if you need external network access
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    def __post_init__(self):
        """Ensure directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_db_path.mkdir(parents=True, exist_ok=True)


# Global config instance
config = Config()
