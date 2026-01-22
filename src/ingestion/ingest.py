"""
Main ingestion pipeline for processing content into the vector store.
Handles both blog posts and company pages.
"""
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from .chunker import TextChunker, Chunk
from .embedder import Embedder
from ..retrieval.vector_store import VectorStore
from ..config import config


class IngestPipeline:
    """
    Pipeline for ingesting content (blog posts and company pages) into the vector store.
    """

    def __init__(
        self,
        chunker: Optional[TextChunker] = None,
        embedder: Optional[Embedder] = None,
        vector_store: Optional[VectorStore] = None
    ):
        """
        Initialize the ingestion pipeline.

        Args:
            chunker: TextChunker instance
            embedder: Embedder instance
            vector_store: VectorStore instance
        """
        self.chunker = chunker or TextChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or VectorStore()

    def load_blog_posts(self, json_path: Optional[Path] = None) -> List[Dict[str, Any]]:
        """
        Load blog posts from JSON file.

        Args:
            json_path: Path to the JSON file

        Returns:
            List of blog post dicts
        """
        json_path = json_path or config.blog_posts_path

        if not json_path.exists():
            print(f"Blog posts file not found: {json_path}")
            return []

        with open(json_path, "r", encoding="utf-8") as f:
            posts = json.load(f)

        # Mark as blog posts
        for post in posts:
            post["page_type"] = post.get("page_type", "blog_post")

        print(f"Loaded {len(posts)} blog posts from {json_path}")
        return posts

    def load_company_pages(self, json_path: Optional[Path] = None) -> List[Dict[str, Any]]:
        """
        Load company pages from JSON file.

        Args:
            json_path: Path to the JSON file

        Returns:
            List of company page dicts
        """
        json_path = json_path or config.company_pages_path

        if not json_path.exists():
            print(f"Company pages file not found: {json_path}")
            return []

        with open(json_path, "r", encoding="utf-8") as f:
            pages = json.load(f)

        # Mark as company pages
        for page in pages:
            page["page_type"] = page.get("page_type", "company_page")

        print(f"Loaded {len(pages)} company pages from {json_path}")
        return pages

    def load_all_content(self) -> List[Dict[str, Any]]:
        """
        Load all content (blog posts and company pages).

        Returns:
            Combined list of all content
        """
        all_content = []

        # Load blog posts
        blog_posts = self.load_blog_posts()
        all_content.extend(blog_posts)

        # Load company pages
        company_pages = self.load_company_pages()
        all_content.extend(company_pages)

        print(f"Total content loaded: {len(all_content)} items")
        return all_content

    def get_new_posts(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter out posts that are already in the vector store.

        Args:
            posts: List of all blog posts

        Returns:
            List of posts not yet in the vector store
        """
        existing_urls = set(self.vector_store.get_all_urls())
        new_posts = [p for p in posts if p.get("url") not in existing_urls]

        print(f"Found {len(new_posts)} new posts (out of {len(posts)} total)")
        return new_posts

    def ingest_posts(
        self,
        posts: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> Dict[str, int]:
        """
        Ingest blog posts into the vector store.

        Args:
            posts: List of blog post dicts
            show_progress: Whether to print progress

        Returns:
            Stats dict with counts
        """
        if not posts:
            return {"posts": 0, "chunks": 0}

        # Chunk all posts
        if show_progress:
            print(f"Chunking {len(posts)} posts...")

        all_chunks = self.chunker.chunk_all_posts(posts)
        if show_progress:
            print(f"Created {len(all_chunks)} chunks")

        if not all_chunks:
            return {"posts": len(posts), "chunks": 0}

        # Generate embeddings
        if show_progress:
            print(f"Generating embeddings for {len(all_chunks)} chunks...")

        texts = [chunk.text for chunk in all_chunks]
        embeddings = []

        # Process in batches to show progress
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embedder.embed_texts(batch)
            embeddings.extend(batch_embeddings)

            if show_progress:
                progress = min(i + batch_size, len(texts))
                print(f"  Embedded {progress}/{len(texts)} chunks")

        # Add to vector store
        if show_progress:
            print("Adding chunks to vector store...")

        added = self.vector_store.add_chunks(all_chunks, embeddings)

        if show_progress:
            print(f"Added {added} chunks to vector store")
            print(f"Total chunks in store: {self.vector_store.count()}")

        return {"posts": len(posts), "chunks": added}

    def ingest_from_file(
        self,
        json_path: Optional[Path] = None,
        incremental: bool = True
    ) -> Dict[str, int]:
        """
        Full ingestion pipeline from JSON file.

        Args:
            json_path: Path to the JSON file
            incremental: If True, only ingest new posts

        Returns:
            Stats dict with counts
        """
        # Load posts
        posts = self.load_blog_posts(json_path)

        # Filter to new posts if incremental
        if incremental:
            posts = self.get_new_posts(posts)

        if not posts:
            print("No new posts to ingest.")
            return {"posts": 0, "chunks": 0}

        # Ingest
        return self.ingest_posts(posts)

    def reingest_all(self, json_path: Optional[Path] = None) -> Dict[str, int]:
        """
        Clear the vector store and reingest all content (blog posts + company pages).

        Args:
            json_path: Path to a specific JSON file (optional, uses all if not specified)

        Returns:
            Stats dict with counts
        """
        print("Clearing vector store...")
        self.vector_store.clear()

        if json_path:
            # Ingest specific file
            return self.ingest_from_file(json_path, incremental=False)
        else:
            # Ingest all content
            all_content = self.load_all_content()

            if not all_content:
                print("No content to ingest.")
                return {"posts": 0, "chunks": 0}

            return self.ingest_posts(all_content)
