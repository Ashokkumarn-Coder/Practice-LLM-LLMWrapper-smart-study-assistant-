"""
INFO SECTION: Main Application Entry Point
------------------------------------------
PURPOSE:
This script is the 'User Interface' of the program. It takes your commands from the 
terminal, reads the text you want to summarize, and sends it to the chosen AI provider.

HOW IT WORKS:
1. ARGUMENT PARSING: It uses `argparse` to understand commands like `--provider` or `--tone`.
2. INPUT LOADING: It reads text either directly from the command line or from a `.txt` file.
3. PROMPT TEMPLATING: It uses `Jinja2` to 'bake' your text into a specific instruction (e.g., 'Summary in an Executive tone').
4. CLIENT SELECTION: It picks the right AI client (e.g., GroqClient) based on your choice.
5. EXECUTION: It calls the client to get the summary and prints it to your screen.

TERMINOLOGY:
- CLI (Command Line Interface): The text-based way you run this program (the terminal).
- Argparse: A Python tool that helps the program understand your '--' flags.
- Jinja2: A template engine (like a 'Fill-in-the-blanks' system) for AI prompts.
- Provider: The AI company (OpenAI, Anthropic, etc.) or local software (LM Studio) doing the work.
- Streaming: Watching the AI write in real-time, one piece at a time.
"""

import argparse
import sys
import os
from config import config
from jinja2 import Environment, FileSystemLoader
from summariser.groq_client import GroqClient
from summariser.openai_client import OpenAIClient
from summariser.anthropic_client import AnthropicClient
from summariser.gemini_client import GeminiClient
from summariser.lmstudio_client import LMStudioClient

def main():
    # Setup our CLI 'User Interface' - this defines how users interact with the tool via the terminal
    parser = argparse.ArgumentParser(description="Smart Text Summarizer - A multi-provider AI tool")
    parser.add_argument("--text", type=str, help="Raw text string to summarize")
    parser.add_argument("--input-file", type=str, help="Path to text file to summarize")
    parser.add_argument("--provider", type=str, default="groq", 
                        choices=["groq", "openai", "anthropic", "gemini", "lmstudio"], 
                        help="LLM provider (default: groq)")
    parser.add_argument("--tone", type=str, default="neutral", help="Tone of the summary (e.g., simple, executive)")
    parser.add_argument("--stream", action="store_true", help="Stream the response from the LLM")
    parser.add_argument("--max-tokens", type=int, default=config.DEFAULT_MAX_TOKENS, help="Maximum number of tokens for the response")
    
    args = parser.parse_args()

    # Determine input text
    input_text = ""
    if args.text:
        input_text = args.text
    elif args.input_file:
        if not os.path.exists(args.input_file):
            print(f"Error: File not found: {args.input_file}")
            return
        with open(args.input_file, 'r', encoding='utf-8') as f:
            input_text = f.read().strip()
    else:
        print("Error: Please provide either --text or --input-file")
        return

    if not input_text:
        print(f"Error: The input source (text or file) is empty.")
        return

    # Initialize Jinja2 environment
    template_dir = os.path.join(os.path.dirname(__file__), "prompts")
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("summarise.j2")
    
    # Render prompt
    prompt = template.render(text=input_text, tone=args.tone)

    # Client Selection
    clients = {
        "groq": GroqClient,
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
        "gemini": GeminiClient,
        "lmstudio": LMStudioClient
    }
    
    client_class = clients.get(args.provider)
    if not client_class:
        print(f"Error: Provider {args.provider} is not supported")
        return
        
    client = client_class()

    print(f"\n--- Summarizing via {args.provider.upper()} (Tone: {args.tone}) ---\n")

    if args.stream:
        for chunk in client.stream_response(prompt, max_tokens=args.max_tokens):
            print(chunk, end="", flush=True)
        print()
    else:
        response = client.generate_response(prompt, max_tokens=args.max_tokens)
        print(response)

if __name__ == "__main__":
    main()