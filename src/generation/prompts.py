"""
Prompt templates for the RAG chatbot.
"""

SYSTEM_PROMPT = """You are a helpful assistant for Near Partner, a technology consulting company specializing in digital transformation, software development, and AI solutions.

**Near Partner's Core Values (embody these in every response):**
- Integrity and Honesty: Always provide truthful, accurate information.
- Commitment: Show dedication to helping users.
- Teamwork Spirit: Encourage collaboration.
- Respect: Treat all questions with respect.
- Professionalism: Maintain a professional yet approachable tone.

**Guidelines:**
1. For greetings (hello, hi, etc.): Respond warmly and introduce yourself as Near Partner's assistant
2. For questions about Near Partner: Use the provided context to answer accurately
3. Do not mention source numbers or citations in your response
4. If no relevant context is found, offer to help with topics like: services, technology insights, AI, software development, low-code, Salesforce
5. Be conversational, concise, and helpful"""

RAG_PROMPT_TEMPLATE = """Based on the following context from Near Partner's blog posts, please answer the user's question.

CONTEXT:
{context}

USER QUESTION: {question}

Please provide a helpful and accurate response based on the context above. Do not mention source numbers or citations in your response."""


def format_context(retrieved_chunks: list) -> str:
    """
    Format retrieved chunks into a context string.

    Args:
        retrieved_chunks: List of retrieved chunk dicts

    Returns:
        Formatted context string
    """
    if not retrieved_chunks:
        return "No relevant context found."

    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, 1):
        title = chunk.get("metadata", {}).get("title", "Unknown")
        author = chunk.get("metadata", {}).get("author", "Unknown")
        url = chunk.get("metadata", {}).get("url", "")
        text = chunk.get("text", "")

        context_parts.append(
            f"[Source {i}: \"{title}\" by {author}]\n{text}\n"
        )

    return "\n---\n".join(context_parts)


def build_rag_prompt(question: str, retrieved_chunks: list) -> str:
    """
    Build the full RAG prompt with context.

    Args:
        question: User's question
        retrieved_chunks: List of retrieved chunk dicts

    Returns:
        Complete prompt string
    """
    context = format_context(retrieved_chunks)

    return RAG_PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )


class PromptTemplates:
    """Container for prompt templates and utilities."""

    system_prompt = SYSTEM_PROMPT
    rag_template = RAG_PROMPT_TEMPLATE

    @staticmethod
    def format_context(chunks: list) -> str:
        return format_context(chunks)

    @staticmethod
    def build_rag_prompt(question: str, chunks: list) -> str:
        return build_rag_prompt(question, chunks)
