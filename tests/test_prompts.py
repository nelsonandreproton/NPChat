"""Tests for prompt templates and sanitization."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.generation.prompts import (
    format_context,
    build_rag_prompt,
    sanitize_user_input,
    SYSTEM_PROMPT,
    PromptTemplates,
)


class TestFormatContext:
    def test_empty_chunks_returns_no_context_message(self):
        result = format_context([])
        assert "Não foi encontrado" in result

    def test_single_chunk_formatted(self):
        chunks = [{"text": "Hello world", "metadata": {"title": "My Post", "author": "John"}}]
        result = format_context(chunks)
        assert "My Post" in result
        assert "John" in result
        assert "Hello world" in result

    def test_multiple_chunks_separated(self):
        chunks = [
            {"text": "First chunk", "metadata": {"title": "Post 1", "author": "A"}},
            {"text": "Second chunk", "metadata": {"title": "Post 2", "author": "B"}},
        ]
        result = format_context(chunks)
        assert "First chunk" in result
        assert "Second chunk" in result
        assert "---" in result  # separator

    def test_missing_metadata_uses_defaults(self):
        chunks = [{"text": "Content only", "metadata": {}}]
        result = format_context(chunks)
        assert "Content only" in result
        assert "Desconhecido" in result or "Near Partner" in result


class TestBuildRagPrompt:
    def test_basic_prompt_contains_question(self):
        chunks = [{"text": "Some context", "metadata": {"title": "T", "author": "A"}}]
        prompt = build_rag_prompt("What is Near Partner?", chunks)
        assert "What is Near Partner?" in prompt
        assert "Some context" in prompt

    def test_conversation_history_included(self):
        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"}
        ]
        chunks = [{"text": "Context", "metadata": {"title": "T", "author": "A"}}]
        prompt = build_rag_prompt("Follow-up question", chunks, history)
        assert "Previous question" in prompt
        assert "Previous answer" in prompt

    def test_empty_history_no_history_section(self):
        chunks = [{"text": "Context", "metadata": {"title": "T", "author": "A"}}]
        prompt = build_rag_prompt("Question", chunks, [])
        assert "HISTÓRICO" not in prompt

    def test_history_truncated_to_last_6_messages(self):
        history = [{"role": "user", "content": f"Message {i}"} for i in range(20)]
        chunks = [{"text": "Context", "metadata": {"title": "T", "author": "A"}}]
        prompt = build_rag_prompt("Q", chunks, history)
        # Only last 6 messages should appear
        assert "Message 19" in prompt
        assert "Message 0" not in prompt


class TestSanitizeInput:
    def test_normal_text_unchanged(self):
        assert sanitize_user_input("Hello Near Partner") == "Hello Near Partner"

    def test_strips_whitespace(self):
        assert sanitize_user_input("  hello  ") == "hello"

    def test_removes_null_bytes(self):
        assert sanitize_user_input("hello\x00world") == "helloworld"

    def test_truncates_to_1000_chars(self):
        long_text = "a" * 2000
        result = sanitize_user_input(long_text)
        assert len(result) == 1000

    def test_empty_string_returns_empty(self):
        assert sanitize_user_input("") == ""

    def test_none_returns_none(self):
        assert sanitize_user_input(None) is None


class TestSystemPrompt:
    def test_system_prompt_has_portuguese(self):
        assert "Near Partner" in SYSTEM_PROMPT
        assert "Português" in SYSTEM_PROMPT or "português" in SYSTEM_PROMPT.lower()

    def test_system_prompt_has_values(self):
        assert "Integridade" in SYSTEM_PROMPT
        assert "Compromisso" in SYSTEM_PROMPT

    def test_system_prompt_has_services(self):
        assert "Salesforce" in SYSTEM_PROMPT
        assert "Low-Code" in SYSTEM_PROMPT


class TestPromptTemplates:
    def test_static_methods_work(self):
        pt = PromptTemplates()
        chunks = [{"text": "text", "metadata": {"title": "t", "author": "a"}}]
        result = pt.build_rag_prompt("q", chunks)
        assert "q" in result

    def test_sanitize_input(self):
        pt = PromptTemplates()
        assert pt.sanitize_input("hello") == "hello"
