"""
Query expansion techniques to improve retrieval.
"""
from typing import List
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


class QueryExpander:
    """
    Expands user queries to improve retrieval performance.
    """

    def __init__(self, model: str = None):
        self.model = model or config.llm_model
        self._client = OpenAI(
            base_url=config.lmstudio_base_url,
            api_key="lm-studio",
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
