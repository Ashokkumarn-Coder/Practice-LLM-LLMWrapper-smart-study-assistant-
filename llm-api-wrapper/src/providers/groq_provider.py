import os
from typing import Iterator, AsyncIterator, List, Optional
from openai import OpenAI, AsyncOpenAI
from ..core import LLMProvider
from ..models import LLMRequest, LLMResponse, LLMResponseChunk, TokenUsage, Message

class GroqProvider(LLMProvider):
    def __init__(self, api_key: str = None, model: str = "llama-3.1-8b-instant"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set")
        self.model = model
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=self.api_key
        )
        self.async_client = AsyncOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=self.api_key
        )

    def _convert_messages(self, messages: List[Message]) -> List[dict]:
        return [{"role": m.role.value, "content": m.content} for m in messages]

    def generate(self, request: LLMRequest) -> LLMResponse:
        messages = self._convert_messages(request.messages)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=request.temperature,
            stream=False,
        )
        
        content = response.choices[0].message.content
        usage = None
        if response.usage:
            usage = TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens
            )
        
        return LLMResponse(
            content=content,
            usage=usage,
            provider="groq",
            model_name=self.model,
        )

    def stream(self, request: LLMRequest) -> Iterator[LLMResponseChunk]:
        messages = self._convert_messages(request.messages)
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=request.temperature,
            stream=True,
        )
        
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield LLMResponseChunk(
                    content_delta=chunk.choices[0].delta.content,
                    provider="groq",
                    model_name=self.model,
                )

    async def generate_async(self, request: LLMRequest) -> LLMResponse:
        messages = self._convert_messages(request.messages)
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=request.temperature,
            stream=False,
        )
        
        content = response.choices[0].message.content
        usage = None
        if response.usage:
             usage = TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens
            )
        
        return LLMResponse(
            content=content,
            usage=usage,
            provider="groq",
            model_name=self.model,
        )

    async def stream_async(self, request: LLMRequest) -> AsyncIterator[LLMResponseChunk]:
        messages = self._convert_messages(request.messages)
        stream = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=request.temperature,
            stream=True,
        )
        
        async for chunk in stream:
             if chunk.choices and chunk.choices[0].delta.content:
                yield LLMResponseChunk(
                    content_delta=chunk.choices[0].delta.content,
                    provider="groq",
                    model_name=self.model,
                )
