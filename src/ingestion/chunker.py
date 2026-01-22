"""
Smart text chunking for blog posts.
Respects paragraph and heading boundaries for better semantic coherence.
"""
import re
from typing import List, Dict, Any
from dataclasses import dataclass


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

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        """
        Initialize chunker.

        Args:
            chunk_size: Target chunk size in characters (approximate)
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
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata={**base_metadata, "chunk_index": chunk_index},
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
            chunks.append(Chunk(
                text=chunk_text,
                metadata={**base_metadata, "chunk_index": chunk_index},
                chunk_index=chunk_index
            ))

        return chunks

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
