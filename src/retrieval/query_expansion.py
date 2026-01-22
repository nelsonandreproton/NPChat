"""
Query expansion techniques to improve retrieval.
"""
from typing import List, Optional
import ollama
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
        self._client = ollama.Client(host=config.ollama_base_url)

    def expand_query(self, query: str) -> str:
        """
        Expand a query with related terms and synonyms.

        Args:
            query: Original user query

        Returns:
            Expanded query string
        """
        # Skip expansion for very short queries (greetings)
        if len(query.split()) <= 2:
            return query

        try:
            response = self._client.generate(
                model=self.model,
                prompt=EXPANSION_PROMPT.format(query=query),
                options={"temperature": 0.3, "num_predict": 150}
            )
            expanded = response["response"].strip()

            # Combine original + expanded for best results
            return f"{query} {expanded}"

        except Exception as e:
            print(f"[QueryExpander] Expansion failed: {e}")
            return query

    def generate_hyde(self, query: str) -> str:
        """
        Generate a Hypothetical Document Embedding (HyDE).

        Instead of embedding the question, we generate a hypothetical answer
        and embed that - often finds better matches.

        Args:
            query: Original user query

        Returns:
            Hypothetical answer to embed
        """
        # Skip for greetings
        if len(query.split()) <= 2:
            return query

        try:
            response = self._client.generate(
                model=self.model,
                prompt=HYDE_PROMPT.format(query=query),
                options={"temperature": 0.5, "num_predict": 150}
            )
            return response["response"].strip()

        except Exception as e:
            print(f"[QueryExpander] HyDE generation failed: {e}")
            return query

    def multi_query(self, query: str, num_variants: int = 3) -> List[str]:
        """
        Generate multiple query variants for multi-query retrieval.

        Args:
            query: Original user query
            num_variants: Number of variants to generate

        Returns:
            List of query variants including original
        """
        prompt = f"""Generate {num_variants} different ways to ask this question.
Each variant should capture a slightly different aspect or use different words.
Output each variant on a new line, numbered 1-{num_variants}.

Original question: {query}

Variants:"""

        try:
            response = self._client.generate(
                model=self.model,
                prompt=prompt,
                options={"temperature": 0.7, "num_predict": 200}
            )

            # Parse variants
            variants = [query]  # Always include original
            lines = response["response"].strip().split("\n")

            for line in lines:
                # Remove numbering like "1.", "1)", etc.
                clean = line.strip()
                if clean and clean[0].isdigit():
                    clean = clean.lstrip("0123456789.):- ")
                if clean and clean != query:
                    variants.append(clean)

            return variants[:num_variants + 1]  # Original + variants

        except Exception as e:
            print(f"[QueryExpander] Multi-query generation failed: {e}")
            return [query]
