# LLM API Wrapper

A unified Python generic wrapper for interacting with multiple Large Language Model (LLM) providers (OpenAI, Anthropic, Google Gemini) through a single, consistent interface.

## Features

- **Multi-Provider Support**: Seamlessly switch between OpenAI, Anthropic, and Gemini.
- **Unified Interface**: Use a single `LLMRequest` and `LLMResponse` model across all providers.
- **Fault Tolerance**: Automatic fallback to other available providers if one fails.
- **Streaming Support**: Unified streaming interface for real-time responses.
- **Streaming Toggle**: Option to enable or disable streaming directly in the UI.
- **Token Tracking**: Standardized token usage reporting (Total, Input, Output) for all providers.
- **User Interfaces**: Includes a Command Line Interface (CLI) and a Streamlit Web App.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd llm-api-wrapper
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1.  Copy the example environment file:
    ```bash
    cp .env.example .env
    ```

2.  Open `.env` and add your API keys:
    ```env
    OPENAI_API_KEY=sk-...
    ANTHROPIC_API_KEY=sk-ant-...
    GEMINI_API_KEY=...
    ```

## Usage

### 1. Command Line Interface (CLI)

Run the CLI for quick testing in your terminal:

```bash
python -m src.cli
```

Options:
- `--provider`: Force a specific provider (e.g., `--provider openai`).
- `--no-stream`: Disable streaming.
- `--temperature`: Set temperature (default 0.7).

### 2. Streamlit Web App

Launch the web-based chat interface:

```bash
streamlit run src/app.py
```

**Features:**
- **Sidebar Configuration**: Choose specific provider or "Auto" (Fallback Mode).
- **Settings**: Adjust temperature, max tokens.
- **Streaming Toggle**: Enable/Disable real-time streaming updates.
- **Metadata**: View Provider, Model, and Token Usage for every response.

### 3. Library Usage

You can use the wrapper in your own Python projects:

```python
import asyncio
from src.wrapper import UnifiedLLM
from src.models import LLMRequest, Message, Role

async def main():
    # Initialize wrapper (automatically loads providers from .env)
    llm = UnifiedLLM()

    # Create a request
    request = LLMRequest(
        messages=[
            Message(role=Role.USER, content="Explain quantum computing in one sentence.")
        ],
        temperature=0.7
    )

    # Generate response (async)
    response = await llm.generate_async(request)
    print(f"Provider: {response.provider}")
    print(f"Response: {response.content}")
    print(f"Tokens: {response.usage}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Project Structure

- `src/core.py`: Abstract Base Class for providers.
- `src/models.py`: Unified Pydantic models.
- `src/wrapper.py`: Main `UnifiedLLM` class with fallback logic.
- `src/providers/`: Individual provider implementations.
- `src/cli.py`: CLI implementation.
- `src/app.py`: Streamlit app implementation.

## Fault Tolerance

The `UnifiedLLM` class attempts to use providers in the order they were initialized. If the primary provider fails (e.g., API error, network issue), it automatically tries the next available provider in the list, ensuring high availability for your application.
