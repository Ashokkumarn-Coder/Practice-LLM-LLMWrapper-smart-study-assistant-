# 🧠 Smart Study Assistant (Simple)

A lightweight, AI-powered study companion designed to help you learn new topics and quiz your knowledge. This project provides both a Command Line Interface (CLI) and a Web Interface (Streamlit) to interact with an LLM (Large Language Model) as a personalized tutor.

## ✨ Features

- **Tutor Mode**: Get detailed explanations for any topic, tailored to your expertise level (Beginner, Intermediate, Advanced).
- **Quiz Mode**: Test your knowledge with generated quizzes on specific topics.
- **Dual Interfaces**:
  - **CLI**: Fast and scriptable access via terminal.
  - **Web UI**: Interactive and user-friendly interface using Streamlit.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- An OpenAI API Key

### Installation

1.  **Clone the repository** (if applicable) or navigate to the project directory.

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment**:
    - Copy the example environment file:
      ```bash
      cp .env.example .env
      ```
    - Open `.env` and add your OpenAI API Key:
      ```env
      OPENAI_API_KEY=sk-your_api_key_here
      ```

## 📖 Usage

### Command Line Interface (CLI)

The CLI tool `cli.py` allows you to interact with the assistant directly from your terminal.

**Ask for a Tutor:**
```bash
# Default level is 'beginner'
python cli.py tutor "Quantum Physics"

# Specify a difficulty level
python cli.py tutor "Machine Learning" --level "advanced"
```

**Generate a Quiz:**
```bash
python cli.py quiz "World History"
```

### Web Interface (Streamlit)

Launch the interactive web application to use the assistant in your browser.

```bash
streamlit run basic_streamlit.py
```
This will open the app in your default web browser (usually at `http://localhost:8501`).

## 📂 Project Structure

- **`basic_core.py`**: Contains the core application logic connecting the LLM to business functions (`tutor`, `quiz`).
- **`cli.py`**: Implementation of the Command Line Interface using `typer`.
- **`basic_streamlit.py`**: Implementation of the Web User Interface using `streamlit`.
- **`llm.py`**: Wrapper for OpenAI API calls.
- **`prompts.py`**: Contains prompt templates for the LLM to ensure consistent responses.
- **`requirements.txt`**: List of Python dependencies.

## 🛠️ Configuration

- **LLM Model**: Currently configured to use `gpt-4o-mini` in `llm.py`.
- **Prompts**: You can adjust the behavior of the tutor or quiz generator by modifying `prompts.py`.
