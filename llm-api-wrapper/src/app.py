"""
Streamlit GUI for LLM API Wrapper

This file creates a web-based chat interface using Streamlit that allows users to interact
with various LLM providers (Groq, OpenAI, Anthropic, Gemini) through a unified interface.

HOW TO RUN:
- Method 1: python -m src.app
- Method 2: python -m src.app --provider groq --stream
- Method 3: streamlit run src/app.py
"""

# ============================================================================
# SECTION 1: IMPORTS
# ============================================================================
# Import required libraries for the Streamlit app

import streamlit as st      # Main library for creating the web UI
import asyncio              # For handling async operations (LLM API calls)
import sys                  # For system-level operations (paths, arguments)
import os                   # For file system operations
import argparse             # For parsing command-line arguments
from dotenv import load_dotenv  # For loading API keys from .env file

# ============================================================================
# SECTION 2: PATH SETUP
# ============================================================================
# Add the parent directory to sys.path to allow relative imports when run directly
# This ensures the app can find the 'src' module regardless of how it's run
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================================
# SECTION 3: IMPORT LLM WRAPPER COMPONENTS
# ============================================================================
# Try to import using relative imports (when run as a module)
# If that fails, use absolute imports (when run directly)
try:
    from .wrapper import UnifiedLLM      # Main LLM wrapper class
    from .models import LLMRequest, Message, Role  # Data models
except ImportError:
    # Fallback for when not run as a module
    from src.wrapper import UnifiedLLM
    from src.models import LLMRequest, Message, Role

# ============================================================================
# SECTION 4: LOAD ENVIRONMENT VARIABLES
# ============================================================================
# Load API keys and other secrets from .env file
# This includes GROQ_API_KEY, OPENAI_API_KEY, etc.
load_dotenv()

# ============================================================================
# SECTION 5: COMMAND-LINE ARGUMENT PARSER
# ============================================================================
def parse_args():
    """
    Parse command-line arguments to pre-configure the UI.
    
    UI IMPACT: Sets default values for provider selection and streaming toggle
    
    USAGE EXAMPLES:
    - python -m src.app --provider groq --stream
    - python -m src.app --provider openai --no-stream
    
    Returns:
        Namespace: Parsed arguments with provider and stream settings
    """
    parser = argparse.ArgumentParser(description="LLM API Wrapper - Streamlit App")
    parser.add_argument("--provider", type=str, help="Provider to use", default=None)
    parser.add_argument("--stream", action="store_true", help="Enable streaming", default=None)
    parser.add_argument("--no-stream", action="store_false", dest="stream", help="Disable streaming")
    
    # When running via streamlit run, the arguments are passed after '--'
    if "--" in sys.argv:
        args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    else:
        # Try to parse all if not run via streamlit (though this is mainly for the wrapper)
        args, _ = parser.parse_known_args()
    return args

# ============================================================================
# SECTION 6: MAIN APP FUNCTION
# ============================================================================
async def run_app():
    """
    Main application function that creates and runs the Streamlit UI.
    
    UI STRUCTURE:
    1. Page header with title and description
    2. Sidebar with configuration options (provider, streaming, temperature)
    3. Chat message history display
    4. Chat input box at the bottom
    5. Real-time response generation (streaming or non-streaming)
    """
    
    # ------------------------------------------------------------------------
    # SUBSECTION 6.1: PAGE CONFIGURATION
    # ------------------------------------------------------------------------
    # UI IMPACT: Sets browser tab title, layout, and icon
    st.set_page_config(page_title="LLM API Wrapper", layout="centered", page_icon="🤖")
    
    # ------------------------------------------------------------------------
    # SUBSECTION 6.2: PAGE HEADER
    # ------------------------------------------------------------------------
    # UI IMPACT: Displays the main title and subtitle at the top of the page
    st.title("🤖 LLM API Wrapper")
    st.markdown("A unified interface for various LLM providers.")

    # ------------------------------------------------------------------------
    # SUBSECTION 6.3: PARSE CLI ARGUMENTS
    # ------------------------------------------------------------------------
    # Get command-line arguments to set initial UI defaults
    cli_args = parse_args()

    # ------------------------------------------------------------------------
    # SUBSECTION 6.4: SESSION STATE INITIALIZATION
    # ------------------------------------------------------------------------
    # UI IMPACT: Initializes the LLM wrapper and message history
    # Session state persists across page reruns (when user interacts with UI)
    
    # Initialize LLM wrapper (only once per session)
    if "llm" not in st.session_state:
        try:
            st.session_state.llm = UnifiedLLM()  # Creates wrapper with all available providers
        except Exception as e:
            st.error(f"Error initializing LLM Wrapper: {e}")
            return

    # Initialize message history (only once per session)
    if "messages" not in st.session_state:
        st.session_state.messages = []  # Stores all chat messages (user + assistant)

    # ------------------------------------------------------------------------
    # SUBSECTION 6.5: SIDEBAR - CONFIGURATION PANEL
    # ------------------------------------------------------------------------
    # UI IMPACT: Creates a sidebar on the left with all configuration options
    st.sidebar.title("Configuration")
    
    # Get list of active providers (those with valid API keys)
    # UI IMPACT: Populates the provider dropdown menu
    active_providers = [p.__class__.__name__.replace("Provider", "").lower() for p in st.session_state.llm.providers]
    available_options = ["None (Auto)"] + active_providers
    
    # Determine default provider from CLI arguments
    # UI IMPACT: Pre-selects the provider if specified via command line
    default_provider_idx = 0
    if cli_args.provider:
        p_name = cli_args.provider.lower()
        if p_name in active_providers:
            default_provider_idx = active_providers.index(p_name) + 1
        else:
            st.sidebar.warning(f"Provider '{p_name}' not available. Using Auto.")

    # Provider selection dropdown
    # UI IMPACT: Dropdown menu to select which LLM provider to use
    provider_name = st.sidebar.selectbox(
        "Select Provider",
        options=available_options,
        index=default_provider_idx
    )
    
    # Convert "None (Auto)" to None for the API
    if provider_name == "None (Auto)":
        provider_name = None

    # Streaming toggle checkbox
    # UI IMPACT: Checkbox to enable/disable streaming responses
    # Streaming shows words appearing one by one (like ChatGPT)
    # Non-streaming waits for complete response before displaying
    default_stream = True if cli_args.stream is None else cli_args.stream
    streaming_enabled = st.sidebar.checkbox("Enable Streaming", value=default_stream)
    
    # Temperature slider
    # UI IMPACT: Slider to control response randomness (0.0 = deterministic, 1.0 = creative)
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)

    # Clear chat button
    # UI IMPACT: Button to reset the conversation and clear all messages
    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()  # Refresh the page to show empty chat

    # ------------------------------------------------------------------------
    # SUBSECTION 6.6: DISPLAY CHAT HISTORY
    # ------------------------------------------------------------------------
    # UI IMPACT: Shows all previous messages in the conversation
    # Each message is displayed in a chat bubble (user or assistant)
    for msg in st.session_state.messages:
        with st.chat_message(msg.role.value):  # Creates a chat bubble
            st.markdown(msg.content)  # Displays the message text

    # ------------------------------------------------------------------------
    # SUBSECTION 6.7: CHAT INPUT AND RESPONSE GENERATION
    # ------------------------------------------------------------------------
    # UI IMPACT: Creates the chat input box at the bottom of the page
    # When user types and presses Enter, this code executes
    if prompt := st.chat_input("What's on your mind?"):
        
        # Add user message to history and display it
        # UI IMPACT: Shows user's message in a chat bubble
        user_msg = Message(role=Role.USER, content=prompt)
        st.session_state.messages.append(user_msg)
        with st.chat_message("user"):
            st.markdown(prompt)

        # Prepare the API request with all messages and settings
        request = LLMRequest(
            messages=st.session_state.messages,  # Full conversation history
            temperature=temperature,              # Randomness setting
            stream=streaming_enabled              # Streaming mode
        )

        # Generate and display the assistant's response
        # UI IMPACT: Shows assistant's response in a chat bubble
        with st.chat_message("assistant"):
            response_placeholder = st.empty()  # Placeholder for updating response
            full_response = ""
            
            try:
                # STREAMING MODE: Show response word-by-word as it's generated
                if streaming_enabled:
                    # UI IMPACT: Words appear one by one with a cursor (▌)
                    async for chunk in st.session_state.llm.stream_async(request, provider_name=provider_name):
                        full_response += chunk.content_delta
                        response_placeholder.markdown(full_response + "▌")  # Show cursor
                    response_placeholder.markdown(full_response)  # Remove cursor when done
                
                # NON-STREAMING MODE: Wait for complete response, then show all at once
                else:
                    # UI IMPACT: Shows a spinner while waiting, then displays full response
                    with st.spinner("Generating..."):
                        response = await st.session_state.llm.generate_async(request, provider_name=provider_name)
                        full_response = response.content
                        response_placeholder.markdown(full_response)
                
                # Add assistant's response to conversation history
                st.session_state.messages.append(Message(role=Role.ASSISTANT, content=full_response))
            
            # Handle any errors during generation
            # UI IMPACT: Shows error message and details in an expandable section
            except Exception as e:
                st.error(f"Error generating response: {e}")
                import traceback
                st.expander("Details").code(traceback.format_exc())

# ============================================================================
# SECTION 7: STREAMLIT LAUNCHER FUNCTION
# ============================================================================
def start_streamlit():
    """
    Launch Streamlit server when running via 'python -m src.app'.
    
    This function is called when the script is run directly (not via 'streamlit run').
    It starts a subprocess that runs 'streamlit run' with this file.
    
    UI IMPACT: Opens the Streamlit app in the default web browser
    """
    import subprocess
    app_path = os.path.abspath(__file__)
    cmd = [sys.executable, "-m", "streamlit", "run", app_path]
    
    # Pass through any command-line arguments
    if len(sys.argv) > 1:
        cmd.append("--")
        cmd.extend(sys.argv[1:])
    
    print(f"Launching Streamlit GUI...")
    subprocess.run(cmd)

# ============================================================================
# SECTION 8: ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    """
    Entry point when running the script directly.
    
    LOGIC:
    1. Check if we're already running inside Streamlit
    2. If yes: Run the app function
    3. If no: Launch Streamlit server
    
    This prevents infinite recursion when launching via 'python -m src.app'
    """
    # Check if we are running inside streamlit
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is not None:
            # We're inside Streamlit, run the app
            asyncio.run(run_app())
        else:
            # We're not inside Streamlit, launch it
            start_streamlit()
    except ImportError:
        # Streamlit not available, try to launch it
        start_streamlit()
