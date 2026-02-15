"""Tests for the text chunker."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.ingestion.chunker import TextChunker, Chunk


@pytest.fixture
def chunker():
    return TextChunker(chunk_size=500, chunk_overlap=50)


def make_post(content: str, title: str = "Test Post", url: str = "http://example.com") -> dict:
    return {"content": content, "title": title, "url": url, "author": "Test Author", "published_date": "2024-01"}


class TestTextChunker:
    def test_empty_content_returns_empty(self, chunker):
        chunks = chunker.chunk_blog_post(make_post(""))
        assert chunks == []

    def test_short_content_returns_single_chunk(self, chunker):
        post = make_post("This is a short blog post with some content.")
        chunks = chunker.chunk_blog_post(post)
        assert len(chunks) == 1
        assert chunks[0].text == "This is a short blog post with some content."

    def test_chunk_has_metadata(self, chunker):
        post = make_post("Hello world.", url="http://test.com", title="My Title")
        chunks = chunker.chunk_blog_post(post)
        assert len(chunks) >= 1
        assert chunks[0].metadata["url"] == "http://test.com"
        assert chunks[0].metadata["title"] == "My Title"
        assert chunks[0].metadata["author"] == "Test Author"

    def test_long_content_is_split(self, chunker):
        # Create content that should be split into multiple chunks
        long_para = "This is a paragraph with enough content. " * 10
        content = "\n\n".join([long_para] * 5)
        chunks = chunker.chunk_blog_post(make_post(content))
        assert len(chunks) > 1

    def test_chunk_indices_are_sequential(self, chunker):
        long_para = "This is a paragraph. " * 20
        content = "\n\n".join([long_para] * 4)
        chunks = chunker.chunk_blog_post(make_post(content))
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_heading_is_kept_with_content(self, chunker):
        content = "Introduction\n\nThis is the first paragraph under introduction."
        chunks = chunker.chunk_blog_post(make_post(content))
        assert len(chunks) >= 1
        # Heading should be prepended to content
        assert "Introduction" in chunks[0].text

    def test_chunk_all_posts(self, chunker):
        posts = [
            make_post("Post one content.", url="http://one.com"),
            make_post("Post two content.", url="http://two.com"),
        ]
        all_chunks = chunker.chunk_all_posts(posts)
        assert len(all_chunks) == 2
        urls = [c.metadata["url"] for c in all_chunks]
        assert "http://one.com" in urls
        assert "http://two.com" in urls

    def test_no_missing_content_between_chunks(self, chunker):
        """Verify all text from a post appears in some chunk."""
        long_para = "Unique word here. " * 15
        content = "\n\n".join([f"Section {i}\n\n{long_para}" for i in range(4)])
        chunks = chunker.chunk_blog_post(make_post(content))
        combined = " ".join(c.text for c in chunks)
        # Each section heading should appear somewhere
        for i in range(4):
            assert f"Section {i}" in combined
