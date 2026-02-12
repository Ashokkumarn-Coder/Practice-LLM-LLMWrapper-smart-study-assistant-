from ..core import LLMProvider
from ..models import LLMRequest, LLMResponse, LLMResponseChunk
from typing import Iterator, AsyncIterator

class GeminiProvider(LLMProvider):
    def generate(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError("GeminiProvider not implemented yet")

    def stream(self, request: LLMRequest) -> Iterator[LLMResponseChunk]:
        raise NotImplementedError("GeminiProvider not implemented yet")

    async def generate_async(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError("GeminiProvider not implemented yet")

    async def stream_async(self, request: LLMRequest) -> AsyncIterator[LLMResponseChunk]:
        raise NotImplementedError("GeminiProvider not implemented yet")
