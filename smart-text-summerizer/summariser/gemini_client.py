"""
INFO SECTION: Google Gemini LLM Client
-------------------------------------
PURPOSE:
This client allows the summarizer to use Google's Gemini models. Gemini is powerful 
and often faster or more accessible for users within the Google ecosystem.

HOW IT WORKS:
1. It use the new `google-genai` library (the latest official way to talk to Gemini).
2. It wraps the prompt into a 'Contents' object, which is Gemini's unique way of receiving text.
3. It supports both standard generation and 'Live' streaming chunks.

TERMINOLOGY:
- Gemini: Google's flagship multimodal AI (can understand text, images, and audio).
- Google-Genai: The Python tool (SDK) specifically built to integrate Gemini into apps.
- GenerateContent: The internal Google command used to trigger a summary response.
- SDK (Software Development Kit): A collection of tools that makes it easy for us to talk to Google's service.
"""

from .base import BaseLLMClient
from config import config
import os
from google import genai
from google.genai import types
from typing import Iterator

class GeminiClient(BaseLLMClient):
    def __init__(self, model: str = None):
        self.api_key = config.GEMINI_API_KEY
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set in configuration or environment")
        
        self.model_name = model or config.DEFAULT_MODEL_GEMINI
        self.client = genai.Client(api_key=self.api_key)

    def generate_response(self, prompt: str, **kwargs) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=kwargs.get("max_tokens", config.DEFAULT_MAX_TOKENS),
                temperature=kwargs.get("temperature", 0.7),
            )
        )
        return response.text

    def stream_response(self, prompt: str, **kwargs) -> Iterator[str]:
        response = self.client.models.generate_content_stream(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=kwargs.get("max_tokens", config.DEFAULT_MAX_TOKENS),
                temperature=kwargs.get("temperature", 0.7),
            )
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text
