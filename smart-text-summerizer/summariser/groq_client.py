import os
"""
INFO SECTION: Groq LLM Client
----------------------------
PURPOSE:
This client handles communication with Groq, an ultra-fast AI provider. It allows
the summarizer to get high-speed responses using Llama 3 models.

HOW IT WORKS:
1. It initializes an `OpenAI` client but points the 'base_url' to Groq's high-speed servers.
2. It pulls the Groq API key and default model from the global `config`.
3. `generate_response`: Sends a single request and returns the full summary.
4. `stream_response`: Sends a request and 'yields' text chunks as they arrive for real-time display.

TERMINOLOGY:
- Groq: A company that built special chips (LPUs) to make AI incredibly fast.
- LPU (Language Processing Unit): The hardware 'brain' that processes AI text faster than regular chips.
- OpenAI-Compatible: Groq uses the same code format as OpenAI, making it easy to swap them.
- Chat Completions: The standard way of asking an AI a question and getting an answer.
"""

from .base import BaseLLMClient
from config import config
from openai import OpenAI
from typing import Iterator

class GroqClient(BaseLLMClient):
    def __init__(self, model: str = None):
        self.api_key = config.GROQ_API_KEY
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set in configuration or environment")
        
        self.model = model or config.DEFAULT_MODEL_GROQ
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=self.api_key
        )

    def generate_response(self, prompt: str, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=kwargs.get("max_tokens", config.DEFAULT_MAX_TOKENS),
            temperature=kwargs.get("temperature", 0.7),
            stream=False,
        )
        return response.choices[0].message.content

    def stream_response(self, prompt: str, **kwargs) -> Iterator[str]:
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=kwargs.get("max_tokens", config.DEFAULT_MAX_TOKENS),
            temperature=kwargs.get("temperature", 0.7),
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
