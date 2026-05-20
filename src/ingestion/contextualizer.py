"""
Contextual Retrieval (Anthropic 2024): prepend LLM-generated context to each chunk.

For each chunk, a short sentence is generated describing where the chunk sits
within its source document. That prefix is prepended to the chunk text before
embedding, improving retrieval accuracy for out-of-context chunks.

Reference: https://www.anthropic.com/news/contextual-retrieval
"""
from typing import Optional
from openai import OpenAI
from ..config import config

_CONTEXT_PROMPT = """\
Here is the full document:
<document>
{document}
</document>

Here is a chunk from that document:
<chunk>
{chunk}
</chunk>

Respond with a single concise sentence (max 80 words) that situates this chunk \
within the document — what topic it covers and how it relates to the document's \
overall subject. Do not repeat the chunk text. Do not add commentary."""

# XML delimiter tags used in the prompt — strip them from scraped content so
# a page containing </document> or </chunk> cannot escape the delimiter and
# inject arbitrary instructions into the prompt (prompt injection via CMS).
_PROMPT_DELIMITER_TAGS = ("<document>", "</document>", "<chunk>", "</chunk>")


def _sanitize_for_prompt(text: str) -> str:
    for tag in _PROMPT_DELIMITER_TAGS:
        text = text.replace(tag, "")
    return text


class ChunkContextualizer:
    """
    Generates a context prefix for each chunk using an LLM call.

    Usage:
        contextualizer = ChunkContextualizer()
        prefix = contextualizer.get_context(document_text, chunk_text)
        # Returns e.g. "This chunk discusses pricing tiers for Near Partner's..."
    """

    def __init__(self, client: Optional[OpenAI] = None):
        self._client = client or OpenAI(
            base_url=config.lmstudio_base_url,
            api_key="lm-studio",
        )

    def get_context(self, document: str, chunk: str) -> str:
        """
        Generate a context sentence for *chunk* within *document*.

        Returns an empty string on failure so the caller can fall back gracefully.
        """
        prompt = _CONTEXT_PROMPT.format(
            document=_sanitize_for_prompt(document),
            chunk=_sanitize_for_prompt(chunk),
        )
        try:
            response = self._client.chat.completions.create(
                model=config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=80,
            )
            content = response.choices[0].message.content
            return content.strip() if content else ""
        except Exception as e:
            print(f"[ChunkContextualizer] LLM call failed, using plain chunk: {e}")
            return ""
