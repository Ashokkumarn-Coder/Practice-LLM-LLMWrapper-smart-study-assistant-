from ..core import LLMProvider
from ..models import LLMRequest, LLMResponse, LLMResponseChunk
from typing import Iterator, AsyncIterator

class OpenAIProvider(LLMProvider):
    def generate(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError("OpenAIProvider not implemented yet")

    def stream(self, request: LLMRequest) -> Iterator[LLMResponseChunk]:
        raise NotImplementedError("OpenAIProvider not implemented yet")

    async def generate_async(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError("OpenAIProvider not implemented yet")

    async def stream_async(self, request: LLMRequest) -> AsyncIterator[LLMResponseChunk]:
        raise NotImplementedError("OpenAIProvider not implemented yet")
