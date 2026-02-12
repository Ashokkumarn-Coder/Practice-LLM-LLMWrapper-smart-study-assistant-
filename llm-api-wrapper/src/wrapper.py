"""
Unified LLM Wrapper with Automatic Fallback

This file implements the main UnifiedLLM class that provides a single interface
to multiple LLM providers with automatic fallback if one fails.

PURPOSE: Simplify LLM usage by handling provider selection and fallback automatically
KEY FEATURE: If one provider fails, automatically tries the next one
"""

# ============================================================================
# SECTION 1: IMPORTS
# ============================================================================
import logging                                 # For logging errors and info
from typing import List, Optional, Iterator, AsyncIterator  # Type hints
from .core import LLMProvider                  # Base class interface
from .models import LLMRequest, LLMResponse, LLMResponseChunk  # Data models
from .providers.openai_provider import OpenAIProvider      # OpenAI implementation
from .providers.anthropic_provider import AnthropicProvider  # Anthropic implementation
from .providers.gemini_provider import GeminiProvider      # Gemini implementation
from .providers.groq_provider import GroqProvider          # Groq implementation

# ============================================================================
# SECTION 2: LOGGING SETUP
# ============================================================================
# Configure logging to track which providers are being used and any errors
logger = logging.getLogger(__name__)

# ============================================================================
# SECTION 3: UNIFIED LLM CLASS
# ============================================================================
class UnifiedLLM(LLMProvider):
    """
    Main wrapper class that manages multiple LLM providers with automatic fallback.
    
    PURPOSE: Provide a single, simple interface to use any LLM provider
    
    KEY FEATURES:
        1. Auto-initialization: Automatically detects which providers have API keys
        2. Fallback logic: If one provider fails, tries the next one
        3. Provider selection: Can specify which provider to use
        4. Unified interface: Same methods work for all providers
    
    USAGE EXAMPLE:
        llm = UnifiedLLM()  # Auto-detects available providers
        request = LLMRequest(messages=[Message(role=Role.USER, content="Hello")])
        response = await llm.generate_async(request)  # Uses first available provider
    """

    # ========================================================================
    # SUBSECTION 3.1: INITIALIZATION
    # ========================================================================
    def __init__(self, providers: List[LLMProvider] = None, retry_attempts: int = 1):
        """
        Initialize the UnifiedLLM wrapper.
        
        PURPOSE: Set up the wrapper with available providers
        
        INITIALIZATION LOGIC:
            1. If providers list is given, use it
            2. Otherwise, try to initialize all standard providers
            3. Only add providers that have valid API keys
            4. Log warning if no providers are available
        
        Args:
            providers (List[LLMProvider]): Optional list of pre-initialized providers
                                           If None, auto-detects based on API keys
            retry_attempts (int): Number of times to retry on failure (not used yet)
        
        EXAMPLE:
            # Auto-detect providers
            llm = UnifiedLLM()
            
            # Or specify providers manually
            llm = UnifiedLLM(providers=[GroqProvider()])
        """
        self.providers = providers or []
        self.retry_attempts = retry_attempts
        
        # ====================================================================
        # AUTO-INITIALIZATION: Try to initialize all standard providers
        # ====================================================================
        # If no providers given, try to initialize all standard ones if keys exist
        if not self.providers:
            # Try Groq (currently the only fully implemented provider)
            try:
                self.providers.append(GroqProvider())
            except ValueError:
                # ValueError is raised if GROQ_API_KEY is not set
                pass

            # Try OpenAI (placeholder - not implemented yet)
            try:
                self.providers.append(OpenAIProvider())
            except ValueError:
                pass
            
            # Try Anthropic (placeholder - not implemented yet)
            try:
                self.providers.append(AnthropicProvider())
            except ValueError:
                pass
                
            # Try Gemini (placeholder - not implemented yet)
            try:
                self.providers.append(GeminiProvider())
            except ValueError:
                pass
                
        # Warn if no providers could be initialized
        if not self.providers:
            logger.warning("No providers could be initialized. Please check API keys.")

    # ========================================================================
    # SUBSECTION 3.2: PROVIDER LOOKUP
    # ========================================================================
    def get_provider(self, name: str) -> Optional[LLMProvider]:
        """
        Get a specific provider by name.
        
        PURPOSE: Find and return a provider instance by its name
        
        Args:
            name (str): Provider name ("groq", "openai", "anthropic", "gemini")
        
        Returns:
            Optional[LLMProvider]: The provider instance, or None if not found
        
        EXAMPLE:
            groq = llm.get_provider("groq")
            if groq:
                response = groq.generate(request)
        """
        for p in self.providers:
            if isinstance(p, GroqProvider) and name == "groq":
                return p
            if isinstance(p, OpenAIProvider) and name == "openai":
                return p
            if isinstance(p, AnthropicProvider) and name == "anthropic":
                return p
            if isinstance(p, GeminiProvider) and name == "gemini":
                return p
        return None

    # ========================================================================
    # SUBSECTION 3.3: PROVIDER CHAIN BUILDER
    # ========================================================================
    def _get_providers_chain(self, provider_name: str = None) -> List[LLMProvider]:
        """
        Get the list of providers to try, prioritizing the requested one if valid.
        
        PURPOSE: Build the fallback chain - which providers to try and in what order
        
        LOGIC:
            1. If no provider specified, return all providers in order
            2. If provider specified and valid, put it first, then others
            3. If provider specified but not found, log warning and use all
        
        Args:
            provider_name (str): Optional provider to prioritize
        
        Returns:
            List[LLMProvider]: Ordered list of providers to try
        
        EXAMPLE:
            # No preference - tries all in order
            chain = llm._get_providers_chain()  # [Groq, OpenAI, Anthropic, Gemini]
            
            # Prefer Groq - tries Groq first, then others
            chain = llm._get_providers_chain("groq")  # [Groq, OpenAI, Anthropic, Gemini]
        """
        if not provider_name:
            return self.providers
            
        specific_provider = self.get_provider(provider_name)
        if not specific_provider:
            logger.warning(f"Requested provider '{provider_name}' not available (not initialized). Falling back to other active providers.")
            return self.providers
            
        # Prioritize specific provider, then others
        return [specific_provider] + [p for p in self.providers if p != specific_provider]

    # ========================================================================
    # SUBSECTION 3.4: SYNCHRONOUS GENERATION (Non-Streaming)
    # ========================================================================
    def generate(self, request: LLMRequest, provider_name: str = None) -> LLMResponse:
        """
        Generate response with automatic fallback logic.
        
        PURPOSE: Get a complete response, trying multiple providers if needed
        
        FALLBACK LOGIC:
            1. Get the provider chain (ordered list of providers to try)
            2. Try each provider in order
            3. If one fails, log error and try next
            4. If all fail, raise RuntimeError
        
        Args:
            request (LLMRequest): The request to send
            provider_name (str): Optional provider to prefer
        
        Returns:
            LLMResponse: The complete response from the first successful provider
        
        Raises:
            RuntimeError: If all providers fail
        
        EXAMPLE:
            request = LLMRequest(messages=[Message(role=Role.USER, content="Hello")])
            response = llm.generate(request)  # Tries providers until one succeeds
        """
        providers_to_try = self._get_providers_chain(provider_name)
        
        last_error = None
        
        # Try each provider in the chain
        for provider in providers_to_try:
            try:
                logger.info(f"Attempting generation with provider: {provider.__class__.__name__}")
                return provider.generate(request)
            except Exception as e:
                logger.error(f"Provider {provider.__class__.__name__} failed: {e}")
                last_error = e
                continue  # Try next provider
        
        # All providers failed
        raise RuntimeError("All providers failed to generate response.") from last_error

    # ========================================================================
    # SUBSECTION 3.5: SYNCHRONOUS STREAMING
    # ========================================================================
    def stream(self, request: LLMRequest, provider_name: str = None) -> Iterator[LLMResponseChunk]:
        """
        Stream response with automatic fallback logic.
        
        PURPOSE: Stream response word-by-word, trying multiple providers if needed
        
        FALLBACK LOGIC:
            Same as generate(), but yields chunks instead of returning full response
        
        Args:
            request (LLMRequest): The request to send
            provider_name (str): Optional provider to prefer
        
        Yields:
            LLMResponseChunk: Response chunks from the first successful provider
        
        Raises:
            RuntimeError: If all providers fail
        """
        providers_to_try = self._get_providers_chain(provider_name)

        last_error = None
        
        for provider in providers_to_try:
            try:
                logger.info(f"Attempting streaming with provider: {provider.__class__.__name__}")
                yield from provider.stream(request)
                return  # Success - exit after first successful provider
            except Exception as e:
                logger.error(f"Provider {provider.__class__.__name__} failed: {e}")
                last_error = e
                continue
        
        raise RuntimeError("All providers failed to stream response.") from last_error

    # ========================================================================
    # SUBSECTION 3.6: ASYNCHRONOUS GENERATION (Non-Streaming)
    # ========================================================================
    async def generate_async(self, request: LLMRequest, provider_name: str = None) -> LLMResponse:
        """
        Asynchronously generate response with automatic fallback logic.
        
        PURPOSE: Get a complete response asynchronously, trying multiple providers
        
        USED BY:
            - Streamlit GUI (non-streaming mode)
            - CLI (non-streaming mode)
            - Any async application
        
        Args:
            request (LLMRequest): The request to send
            provider_name (str): Optional provider to prefer
        
        Returns:
            LLMResponse: The complete response from the first successful provider
        
        Raises:
            RuntimeError: If all providers fail
        """
        providers_to_try = self._get_providers_chain(provider_name)

        last_error = None
        
        for provider in providers_to_try:
            try:
                logger.info(f"Attempting async generation with provider: {provider.__class__.__name__}")
                return await provider.generate_async(request)
            except Exception as e:
                logger.error(f"Provider {provider.__class__.__name__} failed: {e}")
                last_error = e
                continue
        
        raise RuntimeError("All providers failed to generate response.") from last_error

    # ========================================================================
    # SUBSECTION 3.7: ASYNCHRONOUS STREAMING
    # ========================================================================
    async def stream_async(self, request: LLMRequest, provider_name: str = None) -> AsyncIterator[LLMResponseChunk]:
        """
        Asynchronously stream response with automatic fallback logic.
        
        PURPOSE: Stream response word-by-word asynchronously, trying multiple providers
        
        USED BY:
            - Streamlit GUI (streaming mode) - MOST COMMON
            - CLI (streaming mode)
            - Any async application that wants live updates
        
        FLOW:
            1. Get provider chain
            2. Try each provider
            3. Yield chunks as they arrive
            4. If provider fails, try next one
            5. If all fail, raise error
        
        Args:
            request (LLMRequest): The request to send
            provider_name (str): Optional provider to prefer
        
        Yields:
            LLMResponseChunk: Response chunks from the first successful provider
        
        Raises:
            RuntimeError: If all providers fail
        
        EXAMPLE:
            request = LLMRequest(messages=[Message(role=Role.USER, content="Hello")])
            async for chunk in llm.stream_async(request):
                print(chunk.content_delta, end="")  # Prints word by word
        """
        providers_to_try = self._get_providers_chain(provider_name)

        last_error = None
        
        for provider in providers_to_try:
            try:
                logger.info(f"Attempting async streaming with provider: {provider.__class__.__name__}")
                async for chunk in provider.stream_async(request):
                    yield chunk
                return  # Success - exit after first successful provider
            except Exception as e:
                logger.error(f"Provider {provider.__class__.__name__} failed: {e}")
                last_error = e
                continue
        
        raise RuntimeError("All providers failed to stream response.") from last_error

# ============================================================================
# SUMMARY: HOW THE WRAPPER WORKS
# ============================================================================
#
# 1. INITIALIZATION:
#    - Auto-detects which providers have API keys
#    - Creates a list of available providers
#
# 2. REQUEST HANDLING:
#    - User calls generate_async() or stream_async()
#    - Wrapper tries providers in order (preferred first, then others)
#    - Returns result from first successful provider
#
# 3. FALLBACK:
#    - If a provider fails, logs error and tries next
#    - If all fail, raises RuntimeError
#
# 4. PROVIDER SELECTION:
#    - Can specify provider: llm.generate_async(request, provider_name="groq")
#    - Or let it auto-select: llm.generate_async(request)
#
# ============================================================================