from llm import get_completion
from prompts import get_tutor_prompt, get_quiz_prompt

def tutor(topic, level="beginner"):
    """
    Generate a tutor explanation for a topic.
    """
    prompt = get_tutor_prompt(topic, level)
    return get_completion(prompt)

def quiz(topic):
    """
    Generate a quiz for a topic.
    """
    prompt = get_quiz_prompt(topic)
    return get_completion(prompt)
