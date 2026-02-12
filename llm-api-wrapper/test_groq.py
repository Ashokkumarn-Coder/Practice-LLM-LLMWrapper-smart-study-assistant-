import asyncio
import os
from dotenv import load_dotenv
from src.providers.groq_provider import GroqProvider
from src.models import LLMRequest, Message, Role

load_dotenv()

async def main():
    api_key = os.getenv("GROQ_API_KEY")
    print(f"API Key present: {bool(api_key)}")

    try:
        provider = GroqProvider()
        print("Provider initialized.")
        
        request = LLMRequest(
            messages=[Message(role=Role.USER, content="Hello, test streaming.")],
            temperature=0.7
        )
        
        print("Sending stream request...")
        full_content = ""
        async for chunk in provider.stream_async(request):
            print(f"Chunk: {chunk.content_delta}", end="|")
            full_content += chunk.content_delta
        
        print("\nFull response received:")
        print(full_content)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
