"""
Main ingestion pipeline for processing content into the vector store.
Handles both blog posts and company pages.
"""
import hashlib
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from .chunker import TextChunker, Chunk, _content_hash
from .embedder import Embedder
from ..retrieval.vector_store import VectorStore
from ..config import config


def _post_content_hash(post: Dict[str, Any]) -> str:
    """Stable fingerprint of a post's content (title + body). Used to detect edits."""
    raw = (post.get("title", "") + post.get("content", "")).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


class IngestPipeline:
    """
    Pipeline for ingesting content (blog posts and company pages) into the vector store.
    """

    def __init__(
        self,
        chunker: Optional[TextChunker] = None,
        embedder: Optional[Embedder] = None,
        vector_store: Optional[VectorStore] = None,
        contextualizer=None,
    ):
        self.chunker = chunker or TextChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or VectorStore()
        # Contextualizer is instantiated lazily or injected (avoids import cost
        # when contextual retrieval is disabled).
        self._contextualizer = contextualizer

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

    def _get_contextualizer(self):
        if self._contextualizer is None:
            from .contextualizer import ChunkContextualizer
            self._contextualizer = ChunkContextualizer()
        return self._contextualizer

    def _apply_contextual_retrieval(
        self,
        chunks: List[Chunk],
        posts: List[Dict[str, Any]],
    ) -> List[Chunk]:
        """
        Prepend an LLM-generated context sentence to each chunk's text.

        - source_hash stays unchanged (plain-text fingerprint, used as dedup key).
        - content_hash is recomputed over the contextualized text (embedding-cache key).
        - The original plain text is preserved in metadata["plain_text"].
        - metadata["context_prefix"] stores the generated prefix.
        """
        doc_by_url: Dict[str, str] = {
            p.get("url", ""): (p.get("title", "") + "\n\n" + p.get("content", "")).strip()
            for p in posts
        }

        contextualizer = self._get_contextualizer()
        contextualized: List[Chunk] = []
        total = len(chunks)

        for i, chunk in enumerate(chunks, 1):
            url = chunk.metadata.get("url", "")
            document = doc_by_url.get(url, "")

            prefix = contextualizer.get_context(document, chunk.text) if document else ""

            if prefix:
                new_text = f"{prefix}\n\n{chunk.text}"
            else:
                new_text = chunk.text

            new_metadata = {
                **chunk.metadata,
                "plain_text": chunk.text,
                "context_prefix": prefix,
                # source_hash stays as the plain-text hash (already set by chunker)
                "content_hash": _content_hash(new_text),
            }
            contextualized.append(Chunk(
                text=new_text,
                metadata=new_metadata,
                chunk_index=chunk.chunk_index,
            ))

            if i % 10 == 0 or i == total:
                print(f"  Contextualized {i}/{total} chunks")

        return contextualized

    def _chunk_post(self, post: Dict[str, Any]) -> List[Chunk]:
        """Chunk a single post using the active chunking strategy."""
        if config.use_parent_child_chunking:
            return self.chunker.chunk_blog_post_with_children(post)
        return self.chunker.chunk_blog_post(post)

    def _chunk_all_posts(self, posts: List[Dict[str, Any]]) -> List[Chunk]:
        """Chunk all posts using the active chunking strategy."""
        if config.use_parent_child_chunking:
            return self.chunker.chunk_all_posts_with_children(posts)
        return self.chunker.chunk_all_posts(posts)

    def get_new_posts(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Return posts that are new or have changed since last ingest.

        A post is considered unchanged only if all of its chunks are already in
        the store (matched by source_hash). If any chunk is missing or the
        content changed, the whole post is returned for re-ingestion (stale
        chunks for that URL are deleted before re-upserting).
        """
        existing_urls = set(self.vector_store.get_all_urls())
        changed = []

        for post in posts:
            url = post.get("url", "")
            if url not in existing_urls:
                changed.append(post)
                continue

            # Post URL is known — check whether plain-text content changed.
            # Compare source_hash (plain-text) so the check is stable regardless
            # of whether contextual retrieval is on or off.
            chunks = self._chunk_post(post)
            existing_hashes = self.vector_store.get_existing_source_hashes_for_url(url)
            new_hashes = {c.metadata["source_hash"] for c in chunks}

            if not new_hashes.issubset(existing_hashes):
                changed.append(post)

        unchanged = len(posts) - len(changed)
        print(f"Found {len(changed)} new/changed posts, {unchanged} unchanged (out of {len(posts)} total)")
        return changed

    def ingest_posts(
        self,
        posts: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> Dict[str, int]:
        """
        Ingest blog posts into the vector store.

        For each post:
        - Stale chunks (old content_hash values for the same URL) are deleted
          before upserting so edited posts don't leave orphan chunks.
        - Chunks whose content_hash is already present are skipped (no
          re-embedding needed), making incremental ingestion cheap.

        Returns:
            Stats dict with counts
        """
        if not posts:
            return {"posts": 0, "chunks": 0}

        if show_progress:
            print(f"Chunking {len(posts)} posts...")

        all_chunks = self._chunk_all_posts(posts)
        if show_progress:
            print(f"Created {len(all_chunks)} chunks")

        if not all_chunks:
            return {"posts": len(posts), "chunks": 0}

        # Pass 1: per-URL — delete stale chunks (content removed/edited) so
        # orphans don't accumulate. Dedup on source_hash (plain-text fingerprint)
        # so the check is stable regardless of whether contextual retrieval is on.
        url_new_source_hashes: Dict[str, set] = {}
        for chunk in all_chunks:
            url = chunk.metadata.get("url", "")
            if url not in url_new_source_hashes:
                url_new_source_hashes[url] = set()
            url_new_source_hashes[url].add(chunk.metadata.get("source_hash", ""))

        url_existing_source: Dict[str, set] = {}
        for url, new_hashes in url_new_source_hashes.items():
            existing = self.vector_store.get_existing_source_hashes_for_url(url)
            stale = existing - new_hashes
            if stale:
                self.vector_store.delete_by_url(url)
                url_existing_source[url] = set()  # all gone — re-embed everything
            else:
                url_existing_source[url] = existing

        # Pass 2: per-chunk — skip chunks whose source_hash is already in the store
        chunks_to_embed = [
            c for c in all_chunks
            if c.metadata.get("source_hash") not in url_existing_source.get(c.metadata.get("url", ""), set())
        ]

        skipped = len(all_chunks) - len(chunks_to_embed)
        if show_progress and skipped:
            print(f"Skipping {skipped} unchanged chunks (already in store)")

        if not chunks_to_embed:
            if show_progress:
                print("All chunks already up to date.")
            return {"posts": len(posts), "chunks": 0}

        # Optional: prepend LLM-generated context prefix to surviving chunks only.
        # Running this after dedup means we only pay LLM costs for new/changed chunks.
        if config.use_contextual_retrieval:
            if show_progress:
                print("Applying contextual retrieval (LLM context per chunk)...")
            chunks_to_embed = self._apply_contextual_retrieval(chunks_to_embed, posts)

        # Generate embeddings only for new/changed chunks
        if show_progress:
            print(f"Generating embeddings for {len(chunks_to_embed)} chunks...")

        texts = [chunk.text for chunk in chunks_to_embed]
        embeddings = []

        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embedder.embed_texts(batch)
            embeddings.extend(batch_embeddings)

            if show_progress:
                progress = min(i + batch_size, len(texts))
                print(f"  Embedded {progress}/{len(texts)} chunks")

        if show_progress:
            print("Upserting chunks into vector store...")

        added = self.vector_store.add_chunks(chunks_to_embed, embeddings)

        if show_progress:
            print(f"Upserted {added} chunks to vector store")
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
