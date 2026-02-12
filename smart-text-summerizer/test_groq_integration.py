from summariser.groq_client import GroqClient
import sys

def test_groq():
    print("Testing GroqClient...")
    try:
        client = GroqClient()
        print(f"Model: {client.model}")
        
        print("\nTest 1: Non-streaming response")
        response = client.generate_response("Say 'Groq is working!'")
        print(f"Response: {response}")
        
        print("\nTest 2: Streaming response")
        print("Response: ", end="")
        for chunk in client.stream_response("Say 'Streaming works!'"):
            print(chunk, end="", flush=True)
        print("\n\nGroq integration test passed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_groq()
