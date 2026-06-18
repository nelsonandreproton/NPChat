"""
LLM wrapper for generating responses via llama.cpp (OpenAI-compatible API).
"""
from typing import Generator, Optional
from openai import OpenAI
from ..config import config


class OllamaLLM:
    """
    LLM client backed by llama.cpp's OpenAI-compatible local server.
    The class name is preserved for backwards compatibility with existing callers.
    """

    def __init__(self, model: str = None):
        self.model = model or config.llm_model
        self._client = OpenAI(
            base_url=config.llm_base_url,
            api_key="not-needed",
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Generator[str, None, None]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        stream = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
