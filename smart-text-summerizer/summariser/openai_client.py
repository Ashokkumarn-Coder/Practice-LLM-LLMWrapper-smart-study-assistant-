"""
INFO SECTION: OpenAI LLM Client
------------------------------
PURPOSE:
This client handles communication with OpenAI, the creators of ChatGPT. It provides 
access to industry-standard models like GPT-4o for high-quality summarization.

HOW IT WORKS:
1. It uses the official `openai` library to connect to their cloud servers.
2. It authenticates using the `OPENAI_API_KEY` stored in your `.env` file.
3. It sends your prompt to the AI and manages either a full response or a real-time stream.

TERMINOLOGY:
- OpenAI: The organization known for pioneering modern LLM technology.
- GPT (Generative Pre-trained Transformer): The underlying AI family used by OpenAI.
- API Key: Your personal 'access pass' to use OpenAI's paid services.
- Token: A small piece of text (roughly 3/4 of a word) used by the AI to calculate costs and length.
"""

from .base import BaseLLMClient
from config import config
import os
from openai import OpenAI
from typing import Iterator

class OpenAIClient(BaseLLMClient):
    def __init__(self, model: str = None):
        self.api_key = config.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set in configuration or environment")
        
        self.model = model or config.DEFAULT_MODEL_OPENAI
        self.client = OpenAI(api_key=self.api_key)

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