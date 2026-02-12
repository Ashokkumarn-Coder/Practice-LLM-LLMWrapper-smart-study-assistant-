"""
INFO SECTION: LM Studio (Local AI) Client
-----------------------------------------
PURPOSE:
This client is special! It allows you to run AI completely 'Locally' on your own 
computer using LM Studio. This means no internet is required and your data stays private.

HOW IT WORKS:
1. It assumes you have LM Studio running and its 'Local Server' turned on.
2. It uses the `OpenAI` library but points it to your own computer (`localhost`).
3. It doesn't need a real API key (it uses a placeholder) because you own the server.

TERMINOLOGY:
- LM Studio: An app you download to run AI models (like Llama) on your own PC/Mac.
- Local Inference: Running AI on your local hardware instead of a company's cloud.
- Localhost (127.0.0.1): The technical name for 'this very computer'.
- Base URL: The 'address' in your browser/code where the service is listening.
"""

from .base import BaseLLMClient
from config import config
import os
from openai import OpenAI
from typing import Iterator

class LMStudioClient(BaseLLMClient):
    def __init__(self, model: str = None):
        self.base_url = config.LMSTUDIO_BASE_URL
        self.model = model or config.DEFAULT_MODEL_LMSTUDIO
        # LM Studio usually doesn't require an API key, but the OpenAI client needs one
        self.client = OpenAI(base_url=self.base_url, api_key="not-needed")

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
