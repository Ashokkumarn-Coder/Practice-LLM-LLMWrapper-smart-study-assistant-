# Smart Text Summariser

A Python-based multi-LLM summarisation engine supporting Groq, OpenAI, Anthropic, Gemini, and LM Studio.

## Setup

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure API Keys:**
    Copy `.env.example` to `.env` and fill in your API keys.
    ```bash
    cp .env.example .env
    ```

## Usage

Run the summariser via CLI:

```bash
# Summarize a file using Groq (default)
python main.py --input-file path/to/article.txt

# Summarize raw text using Anthropic with 'executive' tone
python main.py --provider anthropic --text "Your long text here..." --tone executive

# Summarize using local LM Studio
python main.py --provider lmstudio --input-file path/to/article.txt
```

## Options

-   `--provider`: `groq` (default), `openai`, `anthropic`, `gemini`, `lmstudio`
-   `--input-file`: Path to text file to summarize
-   `--text`: Raw text string to summarize
-   `--tone`: Tone of the summary (neutral, simple, executive, etc.)
-   `--stream`: Stream the response from the LLM

## Terminology & Glossary

| Term | Explanation |
| :--- | :--- |
| **LLM** | **Large Language Model** - The AI "brain" (like Llama 3 or GPT-4) that generates the summary. |
| **Provider** | The service hosting the AI (e.g., Groq, OpenAI, LM Studio). |
| **Streaming** | A mode where text is displayed instantly as it's generated, character-by-character. |
| **Tone** | The stylistic format of the summary (e.g., *Executive* for business). |
| **CLI** | **Command Line Interface** - The terminal-based interface used to run this program. |
