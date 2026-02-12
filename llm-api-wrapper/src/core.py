"""
Core Abstract Base Class for LLM Providers

This file defines the interface (contract) that all LLM providers must implement.
It uses Python's ABC (Abstract Base Class) to enforce this contract.

PURPOSE: Ensure all providers (Groq, OpenAI, Anthropic, etc.) have the same methods
WHY: Allows the wrapper to use any provider interchangeably without knowing the details
"""

# ============================================================================
# SECTION 1: IMPORTS
# ============================================================================
from abc import ABC, abstractmethod           # For creating abstract base classes
from typing import Iterator, AsyncIterator    # For type hints
from .models import LLMRequest, LLMResponse, LLMResponseChunk  # Data models

# ============================================================================
# SECTION 2: ABSTRACT BASE CLASS
# ============================================================================
class LLMProvider(ABC):
    """
    Abstract base class that defines the interface for all LLM providers.
    
    PURPOSE: Create a standard interface that all providers must follow
    
    WHAT IS AN ABSTRACT CLASS?
        - It's like a template or contract
        - You can't create instances of it directly
        - All subclasses MUST implement the abstract methods
    
    WHY USE THIS?
        - Ensures consistency across providers
        - Allows the wrapper to treat all providers the same way
        - Makes it easy to add new providers
    
    PROVIDERS THAT IMPLEMENT THIS:
        - GroqProvider (fully implemented)
        - OpenAIProvider (placeholder)
        - AnthropicProvider (placeholder)
        - GeminiProvider (placeholder)
    """

    # ========================================================================
    # METHOD 1: SYNCHRONOUS GENERATION (Non-Streaming)
    # ========================================================================
    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate a complete response from the LLM (synchronous, non-streaming).
        
        PURPOSE: Get a full response all at once (wait until complete)
        
        WHEN TO USE:
            - When you want the full response before proceeding
            - When streaming is not needed
            - For batch processing
        
        FLOW:
            1. Send request to LLM API
            2. Wait for complete response
            3. Return full response as LLMResponse object
        
        Args:
            request (LLMRequest): The unified request object containing:
                - messages: conversation history
                - temperature: randomness setting
                - max_tokens: response length limit
                
        Returns:
            LLMResponse: The complete response including:
                - content: AI's response text
                - usage: token consumption stats
                - provider: which provider was used
                - model_name: which model was used
        
        EXAMPLE:
            request = LLMRequest(messages=[Message(role=Role.USER, content="Hello")])
            response = provider.generate(request)
            print(response.content)  # "Hello! How can I help you?"
        """
        pass

    # ========================================================================
    # METHOD 2: SYNCHRONOUS STREAMING
    # ========================================================================
    @abstractmethod
    def stream(self, request: LLMRequest) -> Iterator[LLMResponseChunk]:
        """
        Stream the response from the LLM word-by-word (synchronous).
        
        PURPOSE: Get response incrementally as it's generated (like ChatGPT typing)
        
        WHEN TO USE:
            - When you want to show progress to the user
            - For better user experience (no waiting)
            - When processing long responses
        
        FLOW:
            1. Send request to LLM API
            2. Receive response in chunks (word by word)
            3. Yield each chunk as it arrives
            4. Continue until response is complete
        
        Args:
            request (LLMRequest): The unified request object
            
        Returns:
            Iterator[LLMResponseChunk]: An iterator that yields response chunks
                Each chunk contains:
                - content_delta: new text in this chunk
                - provider: which provider is streaming
                - model_name: which model is being used
        
        EXAMPLE:
            request = LLMRequest(messages=[Message(role=Role.USER, content="Hello")])
            for chunk in provider.stream(request):
                print(chunk.content_delta, end="")  # Prints word by word
        """
        pass
    
    # ========================================================================
    # METHOD 3: ASYNCHRONOUS GENERATION (Non-Streaming)
    # ========================================================================
    @abstractmethod
    async def generate_async(self, request: LLMRequest) -> LLMResponse:
        """
        Asynchronously generate a complete response from the LLM.
        
        PURPOSE: Get a full response without blocking other operations
        
        WHAT IS ASYNC?
            - Allows other code to run while waiting for API response
            - More efficient for handling multiple requests
            - Required for Streamlit and modern web apps
        
        WHEN TO USE:
            - In async contexts (Streamlit, FastAPI, etc.)
            - When handling multiple requests concurrently
            - When you want non-blocking I/O
        
        FLOW:
            1. Send request to LLM API (async)
            2. Await complete response (other code can run meanwhile)
            3. Return full response as LLMResponse object
        
        Args:
            request (LLMRequest): The unified request object
            
        Returns:
            LLMResponse: The complete response object
        
        EXAMPLE:
            request = LLMRequest(messages=[Message(role=Role.USER, content="Hello")])
            response = await provider.generate_async(request)
            print(response.content)
        """
        pass

    # ========================================================================
    # METHOD 4: ASYNCHRONOUS STREAMING
    # ========================================================================
    @abstractmethod
    async def stream_async(self, request: LLMRequest) -> AsyncIterator[LLMResponseChunk]:
        """
        Asynchronously stream the response from the LLM.
        
        PURPOSE: Stream response word-by-word without blocking other operations
        
        WHEN TO USE:
            - In async contexts (Streamlit, FastAPI, etc.)
            - When you want streaming AND non-blocking I/O
            - For the best user experience in web apps
        
        FLOW:
            1. Send request to LLM API (async)
            2. Receive chunks asynchronously
            3. Yield each chunk as it arrives
            4. Other code can run between chunks
        
        Args:
            request (LLMRequest): The unified request object
            
        Returns:
            AsyncIterator[LLMResponseChunk]: An async iterator yielding response chunks
        
        EXAMPLE:
            request = LLMRequest(messages=[Message(role=Role.USER, content="Hello")])
            async for chunk in provider.stream_async(request):
                print(chunk.content_delta, end="")  # Prints word by word
        """
        pass

# ============================================================================
# SUMMARY: THE FOUR METHODS
# ============================================================================
# 
# 1. generate()        - Sync, full response at once
# 2. stream()          - Sync, word-by-word streaming
# 3. generate_async()  - Async, full response at once
# 4. stream_async()    - Async, word-by-word streaming
#
# MOST COMMONLY USED:
#   - CLI: stream_async() for interactive chat
#   - Streamlit GUI: stream_async() for live updates
#   - Batch processing: generate() for simplicity
# ============================================================================