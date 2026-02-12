import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def get_completion(prompt, model="llama-3.1-8b-instant"):
    """
    Get a completion from Groq.
    Safe for Streamlit.
    """
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return "Error: GROQ_API_KEY not set"

        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Groq API Error: {e}"
