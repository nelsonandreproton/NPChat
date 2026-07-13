"""
Query expansion techniques to improve retrieval.
"""
from typing import Dict, List, Optional
from openai import OpenAI
from ..config import config


EXPANSION_PROMPT = """You are a search query optimizer. Given a user's question, generate an expanded version that includes:
1. The original question
2. Related keywords and synonyms
3. Alternative phrasings

The expanded query should help find relevant documents about Near Partner (a technology consulting company specializing in digital transformation, AI, and software development).

Keep the expansion concise (under 100 words). Output ONLY the expanded query, nothing else.

User question: {query}

Expanded query:"""


HYDE_PROMPT = """You are an expert on Near Partner, a technology consulting company specializing in digital transformation, software development, and AI solutions.

Given this question, write a short paragraph (50-100 words) that would be a good answer. This will be used to find similar content.

Question: {query}

Answer:"""

CONDENSE_PROMPT = """Given a conversation history and a follow-up question, rewrite the follow-up question as a standalone question that includes any context needed from the history (resolve pronouns, ellipsis, and implicit references).

Rules:
- Do NOT answer the question.
- Do NOT add information that isn't implied by the history or the question.
- Keep the same language as the follow-up question.
- If the follow-up question is already standalone, return it unchanged.
- Output ONLY the rewritten question, nothing else.

Conversation history:
{history}

Follow-up question: {query}

Standalone question:"""


class QueryExpander:
    """
    Expands user queries to improve retrieval performance.
    """

    def __init__(self, model: str = None):
        self.model = model or config.llm_model
        self._client = OpenAI(
            base_url=config.llm_base_url,
            api_key="not-needed",
        )

    def _complete(self, prompt: str, temperature: float = 0.3, max_tokens: int = 150) -> str:
        """Single chat completion call, returns content string."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    def expand_query(self, query: str) -> str:
        if len(query.split()) <= 2:
            return query

        try:
            expanded = self._complete(
                EXPANSION_PROMPT.format(query=query),
                temperature=0.3,
                max_tokens=150,
            )
            return f"{query} {expanded}"
        except Exception as e:
            print(f"[QueryExpander] Expansion failed: {e}")
            return query

    def generate_hyde(self, query: str) -> str:
        if len(query.split()) <= 2:
            return query

        try:
            return self._complete(
                HYDE_PROMPT.format(query=query),
                temperature=0.5,
                max_tokens=150,
            )
        except Exception as e:
            print(f"[QueryExpander] HyDE generation failed: {e}")
            return query

    def condense_query(self, query: str, conversation_history: Optional[List[Dict]] = None) -> str:
        """
        Rewrite a follow-up question into a standalone one using conversation
        history, so retrieval (embedding search / BM25 / reranking) isn't blind
        to context carried over from earlier turns (e.g. "e quanto custa isso?").

        Returns `query` unchanged when there is no history, or on failure.
        """
        history_str = self._format_history(conversation_history)
        if not history_str:
            return query

        try:
            rewritten = self._complete(
                CONDENSE_PROMPT.format(history=history_str, query=query),
                temperature=0.0,
                max_tokens=100,
            )
            return rewritten or query
        except Exception as e:
            print(f"[QueryExpander] Query condensation failed: {e}")
            return query

    def _format_history(self, conversation_history: Optional[List[Dict]], max_turns: int = 3, max_chars: int = 300) -> str:
        """Format the last `max_turns` exchanges as 'Role: content' lines."""
        if not conversation_history:
            return ""
        recent = [m for m in conversation_history if m.get("role") in ("user", "assistant")][-max_turns * 2:]
        lines = []
        for msg in recent:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = (msg.get("content") or "")[:max_chars]
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def multi_query(self, query: str, num_variants: int = 3) -> List[str]:
        prompt = f"""Generate {num_variants} different ways to ask this question.
Each variant should capture a slightly different aspect or use different words.
Output each variant on a new line, numbered 1-{num_variants}.

Original question: {query}

Variants:"""

        try:
            raw = self._complete(prompt, temperature=0.7, max_tokens=200)

            variants = [query]  # always include original
            for line in raw.split("\n"):
                clean = line.strip()
                if clean and clean[0].isdigit():
                    clean = clean.lstrip("0123456789.):- ")
                if clean and clean != query:
                    variants.append(clean)

            return variants[:num_variants + 1]

        except Exception as e:
            print(f"[QueryExpander] Multi-query generation failed: {e}")
            return [query]
