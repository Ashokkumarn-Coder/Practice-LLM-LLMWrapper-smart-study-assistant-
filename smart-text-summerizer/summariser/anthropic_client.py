"""
INFO SECTION: Anthropic LLM Client
---------------------------------
PURPOSE:
This client connects to Anthropic, the makers of Claude. Claude models are known 
for being very helpful, safe, and writing in a natural-sounding tone.

HOW IT WORKS:
1. It uses the `anthropic` Python library to communicate with Claude models.
2. Unlike others, Anthropic uses a specific 'Messages' format for its requests.
3. It handles both full summaries and character-by-character streaming for a responsive feel.

TERMINOLOGY:
- Anthropic: The AI safety and research company behind Claude.
- Claude: The name of the AI model family (Anthropic's version of ChatGPT).
- Messages API: The specific technical format required to send data to Claude.
- Stream Context Manager: A Python pattern (`with...as`) used here to cleanly open and close the streaming connection.
"""

from .base import BaseLLMClient
from config import config
import os
from anthropic import Anthropic
from typing import Iterator

class AnthropicClient(BaseLLMClient):
    def __init__(self, model: str = None):
        self.api_key = config.ANTHROPIC_API_KEY
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set in configuration or environment")
        
        self.model = model or config.DEFAULT_MODEL_ANTHROPIC
        self.client = Anthropic(api_key=self.api_key)

    def generate_response(self, prompt: str, **kwargs) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", config.DEFAULT_MAX_TOKENS),
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.7),
        )
        return response.content[0].text

    def stream_response(self, prompt: str, **kwargs) -> Iterator[str]:
        with self.client.messages.stream(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", config.DEFAULT_MAX_TOKENS),
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.7),
        ) as stream:
            for text in stream.text_stream:
                yield text