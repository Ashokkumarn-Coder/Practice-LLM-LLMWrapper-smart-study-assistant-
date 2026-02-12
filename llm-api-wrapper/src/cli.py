"""
Command-Line Interface (CLI) for LLM API Wrapper

This file provides an interactive terminal-based chat interface for the LLM wrapper.
It uses the Rich library for beautiful terminal formatting.

PURPOSE: Allow users to chat with LLMs directly from the command line
HOW TO RUN: python -m src.cli --provider groq --stream
"""

# ============================================================================
# SECTION 1: IMPORTS
# ============================================================================
import asyncio                # For running async functions
import argparse               # For parsing command-line arguments
import sys                    # For system operations
from dotenv import load_dotenv  # For loading API keys from .env

# Load environment variables BEFORE importing other modules
# This ensures API keys are available when providers are initialized
load_dotenv()

# Rich library imports for beautiful terminal output
from rich.console import Console    # Main console for printing
from rich.markdown import Markdown  # For rendering markdown in terminal
from rich.panel import Panel        # For creating bordered panels
from rich.prompt import Prompt      # For getting user input
from rich.live import Live          # For live-updating content (streaming)

# Import our LLM wrapper components
from .wrapper import UnifiedLLM                    # Main wrapper class
from .models import LLMRequest, Message, Role      # Data models

# ============================================================================
# SECTION 2: CONSOLE SETUP
# ============================================================================
# Create a Rich console for all terminal output
# This provides colors, formatting, and markdown rendering
console = Console()

# ============================================================================
# SECTION 3: MAIN CLI FUNCTION
# ============================================================================
async def main():
    """
    Main function that runs the interactive CLI chat loop.
    
    PURPOSE: Provide an interactive terminal chat experience
    
    FLOW:
        1. Parse command-line arguments
        2. Initialize the LLM wrapper
        3. Display welcome message
        4. Enter chat loop:
           - Get user input
           - Send to LLM
           - Display response (streaming or non-streaming)
           - Repeat until user types 'exit' or 'quit'
    
    CLI STRUCTURE:
        - Header: Blue panel with title
        - Active Providers: Green text showing available providers
        - Chat Loop:
            - "You" prompt (green) for user input
            - "Assistant" label (purple) for AI responses
            - Metadata (dim italic) showing provider and token usage
    """
    
    # ========================================================================
    # SUBSECTION 3.1: ARGUMENT PARSING
    # ========================================================================
    # Parse command-line arguments to configure the CLI
    parser = argparse.ArgumentParser(description="LLM API Wrapper CLI")
    parser.add_argument("--provider", type=str, help="Specific provider to use (openai, anthropic, gemini, groq)", default=None)
    parser.add_argument("--stream", action="store_true", help="Enable streaming", default=True)
    parser.add_argument("--temperature", type=float, help="Temperature", default=0.7)
    args = parser.parse_args()

    # ========================================================================
    # SUBSECTION 3.2: DISPLAY HEADER
    # ========================================================================
    # Print a nice header panel
    # CLI OUTPUT: ╭─────────────────────╮
    #             │ LLM API Wrapper CLI │
    #             ╰─────────────────────╯
    console.print(Panel.fit("[bold blue]LLM API Wrapper CLI[/bold blue]", border_style="blue"))

    # ========================================================================
    # SUBSECTION 3.3: INITIALIZE LLM WRAPPER
    # ========================================================================
    # Create the UnifiedLLM instance
    # This auto-detects which providers have API keys
    try:
        llm = UnifiedLLM()
    except Exception as e:
        console.print(f"[bold red]Error initializing LLM Wrapper:[/bold red] {e}")
        return

    # Check if any providers are available
    if not llm.providers:
        console.print("[bold red]No providers available. Please check your .env file.[/bold red]")
        return
    
    # Display which providers are active
    # CLI OUTPUT: Active Providers: GroqProvider, OpenAIProvider
    active_provider_names = [p.__class__.__name__ for p in llm.providers]
    console.print(f"[green]Active Providers:[/green] {', '.join(active_provider_names)}")

    # ========================================================================
    # SUBSECTION 3.4: INITIALIZE CONVERSATION HISTORY
    # ========================================================================
    # Store all messages in the conversation
    # This allows the LLM to remember context
    messages = []

    # ========================================================================
    # SUBSECTION 3.5: MAIN CHAT LOOP
    # ========================================================================
    # Keep chatting until user types 'exit' or 'quit'
    while True:
        try:
            # ================================================================
            # STEP 1: GET USER INPUT
            # ================================================================
            # Prompt user for input with green "You" label
            # CLI OUTPUT: You: [cursor here]
            user_input = Prompt.ask("\n[bold green]You[/bold green]").strip()
            
            # Skip empty inputs
            if not user_input:
                continue
            
            # Check for exit commands
            if user_input.lower() in ('exit', 'quit'):
                break
            
            # ================================================================
            # STEP 2: ADD USER MESSAGE TO HISTORY
            # ================================================================
            # Add the user's message to the conversation history
            messages.append(Message(role=Role.USER, content=user_input))
            
            # Create the API request with full conversation history
            request = LLMRequest(
                messages=messages,          # Full conversation for context
                temperature=args.temperature,  # Randomness setting
                stream=args.stream          # Streaming mode
            )

            # ================================================================
            # STEP 3: DISPLAY "ASSISTANT" LABEL
            # ================================================================
            # Print purple "Assistant" label before the response
            # CLI OUTPUT: Assistant
            console.print("\n[bold purple]Assistant[/bold purple]")
            
            # Variables to track response metadata
            full_response = ""
            provider_used = ""
            model_used = ""
            usage_info = ""
            
            # ================================================================
            # STEP 4A: STREAMING MODE
            # ================================================================
            # If streaming is enabled, show response word-by-word
            if args.stream:
                # Use Rich's Live display for updating content in place
                # This creates a live-updating markdown display
                with Live(Markdown(""), refresh_per_second=10) as live:
                    # Get chunks from the LLM asynchronously
                    async for chunk in llm.stream_async(request, provider_name=args.provider):
                        # Add new text to the full response
                        full_response += chunk.content_delta
                        
                        # Update the live display with new content
                        # CLI OUTPUT: Words appear one by one
                        live.update(Markdown(full_response))
                        
                        # Track metadata from chunks
                        provider_used = chunk.provider
                        model_used = chunk.model_name
                        if chunk.usage:
                            usage_info = f" | Total Tokens: {chunk.usage.total_tokens} (In: {chunk.usage.input_tokens}, Out: {chunk.usage.output_tokens})"
                
                # Display metadata after response is complete
                # CLI OUTPUT: Generated by groq (llama-3.1-8b-instant) | Total Tokens: 150 (In: 50, Out: 100)
                console.print(f"\n[dim italic]Generated by {provider_used} ({model_used}){usage_info}[/dim italic]")

            # ================================================================
            # STEP 4B: NON-STREAMING MODE
            # ================================================================
            # If streaming is disabled, wait for full response
            else:
                # Show a spinner while waiting
                with console.status("[bold green]Generating...[/bold green]"):
                    # Get complete response
                    response = await llm.generate_async(request, provider_name=args.provider)
                    full_response = response.content
                    
                    # Display the full response as markdown
                    console.print(Markdown(full_response))
                    
                    # Build usage info string
                    usage_info = ""
                    if response.usage:
                        usage_info = f" | Total Tokens: {response.usage.total_tokens} (In: {response.usage.input_tokens}, Out: {response.usage.output_tokens})"
                    
                    # Display metadata
                    console.print(f"\n[dim italic]Generated by {response.provider} ({response.model_name}){usage_info}[/dim italic]")
            
            # ================================================================
            # STEP 5: ADD ASSISTANT RESPONSE TO HISTORY
            # ================================================================
            # Add the AI's response to conversation history
            # This allows the AI to remember what it said
            messages.append(Message(role=Role.ASSISTANT, content=full_response))

        # ====================================================================
        # ERROR HANDLING
        # ====================================================================
        except KeyboardInterrupt:
            # User pressed Ctrl+C
            console.print("\n[yellow]Exiting...[/yellow]")
            break
        except Exception as e:
            # Any other error
            import traceback
            console.print(f"\n[bold red]Error:[/bold red] {e}")
            console.print(traceback.format_exc())

# ============================================================================
# SECTION 4: ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    """
    Entry point when running the CLI directly.
    
    USAGE:
        python -m src.cli
        python -m src.cli --provider groq
        python -m src.cli --provider groq --stream --temperature 0.9
    
    FLOW:
        1. Run the async main() function
        2. Handle Ctrl+C gracefully
    """
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass  # Exit silently on Ctrl+C

# ============================================================================
# SUMMARY: CLI FEATURES
# ============================================================================
#
# 1. INTERACTIVE CHAT:
#    - Type messages and get responses
#    - Full conversation history maintained
#    - Type 'exit' or 'quit' to quit
#
# 2. BEAUTIFUL FORMATTING:
#    - Colored output (green for user, purple for assistant)
#    - Markdown rendering for AI responses
#    - Live streaming updates
#    - Bordered panels and spinners
#
# 3. CONFIGURATION:
#    - --provider: Choose specific provider
#    - --stream: Enable/disable streaming
#    - --temperature: Control randomness
#
# 4. METADATA DISPLAY:
#    - Shows which provider was used
#    - Shows which model was used
#    - Shows token usage (for cost tracking)
#
# ============================================================================