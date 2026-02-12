"""
INFO SECTION: Configuration Manager
----------------------------------
PURPOSE: 
This file acts as the central 'brain' for settings and secrets. It loads environment 
variables (API keys) and defines global defaults used across the entire summarizer.

HOW IT WORKS:
1. It uses `load_dotenv()` to read the `.env` file from the project root.
2. The `Config` class then pulls these values into Python variables using `os.getenv`.
3. It also sets default model names and a global `DEFAULT_MAX_TOKENS` limit.
4. Finally, it creates a single `config` object that other files can import.

TERMINOLOGY:
- .env: A secret file on your computer that stores private API keys safely.
- API Key: A 'password' for your code to talk to AI services like OpenAI or Groq.
- Default: A pre-set value used if the user doesn't choose something else.
- Max Tokens: The maximum length allowed for the AI's response (1 token ≈ 0.75 words).
"""

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    # LM Studio defaults
    LMSTUDIO_BASE_URL = os.getenv("lmstudio_base_url", "http://172.31.64.1:1234/v1")
    
    # Defaults
    DEFAULT_MODEL_OPENAI = "gpt-4-turbo" # Or gpt-3.5-turbo
    DEFAULT_MODEL_ANTHROPIC = "claude-sonnet-4-5" # Example
    DEFAULT_MODEL_GEMINI = "gemini-pro"
    DEFAULT_MODEL_LMSTUDIO = "local-model" # Placeholder
    DEFAULT_MODEL_GROQ = "llama-3.3-70b-versatile"
    DEFAULT_MAX_TOKENS = 4096

config = Config()
