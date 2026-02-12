"""
INFO SECTION: Base LLM Client Interface
---------------------------------------
PURPOSE:
This file defines the 'Rules of the Road' for all AI providers. It ensures that 
no matter which AI you use (Groq, OpenAI, Gemini), they all follow the same pattern.

HOW IT WORKS:
1. It defines an 'Abstract Base Class' (ABC). Think of this as a 'Contract' or 'Blueprint'.
2. Any client we build MUST have a `generate_response` and a `stream_response` method.
3. This 'Consistency' allows `main.py` to switch providers without breaking.

TERMINOLOGY:
- Interface/ABC: A blueprint that defines what actions a provider must support.
- generate_response: A 'wait-and-get' method that gives you the full summary at once.
- stream_response: A 'chunk-by-chunk' method that gives you pieces of text as they are built.
- Iterator: A Python tool that allows us to 'loop' through the AI's streaming chunks.
"""

from abc import ABC, abstractmethod
from typing import Iterator

class BaseLLMClient(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        pass

    @abstractmethod
    def stream_response(self, prompt: str, **kwargs) -> Iterator[str]:
        pass