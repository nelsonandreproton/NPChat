"""
Smart text chunking for blog posts.
Respects paragraph and heading boundaries for better semantic coherence.
"""
import hashlib
import re
from typing import List, Dict, Any
from dataclasses import dataclass


def _content_hash(text: str) -> str:
    """SHA-256 of text, first 16 hex chars — used as a stable chunk fingerprint."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


@dataclass
class Chunk:
    """A text chunk with metadata."""
    text: str
    metadata: Dict[str, Any]
    chunk_index: int


class TextChunker:
    """
    Chunks text intelligently by respecting semantic boundaries.
    """

    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 200):
        """
        Initialize chunker.

        Args:
            chunk_size: Target chunk size in characters (approximate). Note: the
                config.chunk_size is labelled in tokens but this chunker works in
                characters. 1200 chars ≈ 300-400 tokens for typical English text.
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_blog_post(self, post: Dict[str, Any]) -> List[Chunk]:
        """
        Chunk a single blog post into semantic chunks.

        Args:
            post: Blog post dict with 'content', 'title', 'url', etc.

        Returns:
            List of Chunk objects with metadata
        """
        content = post.get("content", "")
        if not content:
            return []

        # Extract metadata for all chunks from this post
        base_metadata = {
            "url": post.get("url", ""),
            "title": post.get("title", ""),
            "author": post.get("author", ""),
            "published_date": post.get("published_date", ""),
            "categories": post.get("categories", []),
        }

        # Split into paragraphs first
        paragraphs = self._split_into_paragraphs(content)

        # Group paragraphs into chunks
        chunks = self._group_paragraphs(paragraphs, base_metadata)

        return chunks

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs, keeping headings with their content."""
        # Split on double newlines (paragraph breaks)
        raw_paragraphs = re.split(r'\n\n+', text)

        paragraphs = []
        current_heading = None

        for para in raw_paragraphs:
            para = para.strip()
            if not para:
                continue

            # Check if this looks like a heading (short, no punctuation at end)
            is_heading = (
                len(para) < 100 and
                not para.endswith(('.', '!', '?', ':')) and
                not para.startswith(('-', '*', '•'))
            )

            if is_heading:
                # Store heading to prepend to next paragraph
                current_heading = para
            else:
                # Prepend heading if we have one
                if current_heading:
                    para = f"{current_heading}\n\n{para}"
                    current_heading = None
                paragraphs.append(para)

        # Don't lose a trailing heading
        if current_heading:
            paragraphs.append(current_heading)

        return paragraphs

    def _group_paragraphs(
        self,
        paragraphs: List[str],
        base_metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """Group paragraphs into chunks of approximately chunk_size."""
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_index = 0

        for para in paragraphs:
            para_length = len(para)

            # If adding this paragraph exceeds chunk_size, finalize current chunk
            if current_length + para_length > self.chunk_size and current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                source_hash = _content_hash(chunk_text)
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata={
                        **base_metadata,
                        "chunk_index": chunk_index,
                        "source_hash": source_hash,
                        "content_hash": source_hash,  # overwritten by contextualizer if enabled
                    },
                    chunk_index=chunk_index
                ))
                chunk_index += 1

                # Start new chunk with overlap (last paragraph if it's not too long)
                if len(current_chunk[-1]) < self.chunk_overlap * 2:
                    current_chunk = [current_chunk[-1], para]
                    current_length = len(current_chunk[-1]) + para_length
                else:
                    current_chunk = [para]
                    current_length = para_length
            else:
                current_chunk.append(para)
                current_length += para_length

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            source_hash = _content_hash(chunk_text)
            chunks.append(Chunk(
                text=chunk_text,
                metadata={
                    **base_metadata,
                    "chunk_index": chunk_index,
                    "source_hash": source_hash,
                    "content_hash": source_hash,  # overwritten by contextualizer if enabled
                },
                chunk_index=chunk_index
            ))

        return chunks

    def chunk_blog_post_with_children(self, post: Dict[str, Any]) -> List[Chunk]:
        """
        Chunk a blog post into paragraph-level child chunks, each carrying its
        parent chunk text in metadata.

        Parent granularity = current 1200-char chunks (same as chunk_blog_post).
        Child granularity = individual paragraphs produced by _split_into_paragraphs.

        Each child Chunk has:
          - text: the paragraph text (small, precise, what gets embedded)
          - metadata["parent_id"]: stable hash of the parent chunk text
          - metadata["parent_text"]: the full parent chunk text (served to the LLM)
          - metadata["source_hash"] / metadata["content_hash"]: hash of child text

        Old path (chunk_blog_post) is untouched.
        """
        content = post.get("content", "")
        if not content:
            return []

        base_metadata = {
            "url": post.get("url", ""),
            "title": post.get("title", ""),
            "author": post.get("author", ""),
            "published_date": post.get("published_date", ""),
            "categories": post.get("categories", []),
        }

        paragraphs = self._split_into_paragraphs(content)
        parent_chunks = self._group_paragraphs(paragraphs, base_metadata)

        children: List[Chunk] = []
        child_index = 0

        for parent in parent_chunks:
            parent_id = parent.metadata["source_hash"]
            parent_text = parent.text

            # Split parent back into paragraphs — the inverse of _group_paragraphs'
            # "\n\n".join(current_chunk). This is exact and avoids the overlap
            # paragraph ambiguity that a substring scan would create.
            parent_paras = parent_text.split("\n\n") if parent_text else [parent_text]

            for para in parent_paras:
                source_hash = _content_hash(para)
                children.append(Chunk(
                    text=para,
                    metadata={
                        **base_metadata,
                        "chunk_index": child_index,
                        "parent_id": parent_id,
                        "parent_text": parent_text,
                        "source_hash": source_hash,
                        "content_hash": source_hash,
                    },
                    chunk_index=child_index,
                ))
                child_index += 1

        return children

    def chunk_all_posts(self, posts: List[Dict[str, Any]]) -> List[Chunk]:
        """
        Chunk all blog posts.

        Args:
            posts: List of blog post dicts

        Returns:
            List of all chunks from all posts
        """
        all_chunks = []
        for post in posts:
            chunks = self.chunk_blog_post(post)
            all_chunks.extend(chunks)
        return all_chunks

    def chunk_all_posts_with_children(self, posts: List[Dict[str, Any]]) -> List[Chunk]:
        """Chunk all posts using parent-child granularity."""
        all_chunks = []
        for post in posts:
            chunks = self.chunk_blog_post_with_children(post)
            all_chunks.extend(chunks)
        return all_chunks
