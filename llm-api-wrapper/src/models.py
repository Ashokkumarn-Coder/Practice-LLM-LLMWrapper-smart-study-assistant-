"""
Data Models for LLM API Wrapper

This file defines all the data structures (models) used throughout the LLM wrapper system.
These models ensure type safety and data validation using Pydantic.

PURPOSE: Standardize the format of requests and responses across different LLM providers
"""

# ============================================================================
# SECTION 1: IMPORTS
# ============================================================================
from enum import Enum                    # For creating enumeration types
from typing import List, Optional, Any   # For type hints
from pydantic import BaseModel, Field    # For data validation and serialization

# ============================================================================
# SECTION 2: ROLE ENUMERATION
# ============================================================================
class Role(str, Enum):
    """
    Defines the possible roles in a conversation.
    
    PURPOSE: Standardize message roles across all LLM providers
    
    USAGE:
        - SYSTEM: Instructions for the AI (e.g., "You are a helpful assistant")
        - USER: Messages from the human user
        - ASSISTANT: Responses from the AI
    
    EXAMPLE:
        Message(role=Role.USER, content="Hello!")
    """
    SYSTEM = "system"        # System instructions/prompts
    USER = "user"            # User messages
    ASSISTANT = "assistant"  # AI responses

# ============================================================================
# SECTION 3: MESSAGE MODEL
# ============================================================================
class Message(BaseModel):
    """
    Represents a single message in a conversation.
    
    PURPOSE: Store one message with its role and content
    
    ATTRIBUTES:
        role (Role): Who sent this message (system/user/assistant)
        content (str): The actual text of the message
    
    EXAMPLE:
        msg = Message(role=Role.USER, content="What's the weather?")
    """
    role: Role      # Who sent this message
    content: str    # The message text

# ============================================================================
# SECTION 4: LLM REQUEST MODEL
# ============================================================================
class LLMRequest(BaseModel):
    """
    Represents a request to an LLM provider.
    
    PURPOSE: Package all the information needed to make an API call to any LLM
    
    ATTRIBUTES:
        messages (List[Message]): Full conversation history
        temperature (float): Randomness (0.0 = deterministic, 1.0 = creative)
        max_tokens (int): Maximum response length (None = provider default)
        stream (bool): Whether to stream response word-by-word
        stop_sequences (List[str]): Strings that stop generation when encountered
    
    EXAMPLE:
        request = LLMRequest(
            messages=[Message(role=Role.USER, content="Hello")],
            temperature=0.7,
            stream=True
        )
    """
    messages: List[Message]                    # Full conversation history
    temperature: float = 0.7                   # Default: balanced creativity
    max_tokens: Optional[int] = None           # Default: provider decides
    stream: bool = False                       # Default: wait for full response
    stop_sequences: Optional[List[str]] = None # Default: no stop sequences

# ============================================================================
# SECTION 5: TOKEN USAGE MODEL
# ============================================================================
class TokenUsage(BaseModel):
    """
    Tracks token consumption for billing and monitoring.
    
    PURPOSE: Record how many tokens were used in a request/response
    
    ATTRIBUTES:
        input_tokens (int): Tokens in the request (your messages)
        output_tokens (int): Tokens in the response (AI's message)
        total_tokens (int): Sum of input + output tokens
    
    WHY THIS MATTERS:
        - LLM APIs charge per token
        - Helps track costs
        - Useful for optimization
    
    EXAMPLE:
        usage = TokenUsage(input_tokens=50, output_tokens=100, total_tokens=150)
    """
    input_tokens: int   # Tokens sent to the API
    output_tokens: int  # Tokens received from the API
    total_tokens: int   # Total tokens used (for billing)

# ============================================================================
# SECTION 6: LLM RESPONSE MODEL
# ============================================================================
class LLMResponse(BaseModel):
    """
    Represents a complete response from an LLM provider.
    
    PURPOSE: Standardize the format of responses from different providers
    
    ATTRIBUTES:
        content (str): The AI's response text
        usage (TokenUsage): Token consumption info (if available)
        provider (str): Which provider generated this (e.g., "groq", "openai")
        model_name (str): Which model was used (e.g., "llama-3.1-8b-instant")
        raw_response (Any): Original API response (excluded from serialization)
    
    USAGE:
        This is returned by generate() and generate_async() methods
    
    EXAMPLE:
        response = LLMResponse(
            content="Hello! How can I help?",
            provider="groq",
            model_name="llama-3.1-8b-instant",
            usage=TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30)
        )
    """
    content: str                               # The AI's response
    usage: Optional[TokenUsage] = None         # Token usage (if available)
    provider: str                              # Which provider was used
    model_name: str                            # Which model was used
    raw_response: Any = Field(default=None, exclude=True)  # Original API response

# ============================================================================
# SECTION 7: LLM RESPONSE CHUNK MODEL
# ============================================================================
class LLMResponseChunk(BaseModel):
    """
    Represents one piece of a streaming response.
    
    PURPOSE: Handle word-by-word streaming responses from LLMs
    
    ATTRIBUTES:
        content_delta (str): The new text in this chunk (e.g., one word)
        provider (str): Which provider is streaming
        model_name (str): Which model is being used
        usage (TokenUsage): Token usage (usually only in final chunk)
    
    USAGE:
        This is yielded by stream() and stream_async() methods
        Multiple chunks combine to form the complete response
    
    EXAMPLE:
        chunk1 = LLMResponseChunk(content_delta="Hello", provider="groq", model_name="llama-3.1-8b")
        chunk2 = LLMResponseChunk(content_delta=" there", provider="groq", model_name="llama-3.1-8b")
        # Full response: "Hello there"
    """
    content_delta: str                    # New text in this chunk
    provider: str                         # Which provider is streaming
    model_name: str                       # Which model is being used
    usage: Optional[TokenUsage] = None    # Token usage (if available)