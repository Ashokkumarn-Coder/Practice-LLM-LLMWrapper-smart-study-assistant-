def get_tutor_prompt(topic, level):
    return f"You are a helpful tutor. Explain {topic} to a {level} student. Keep it concise."

def get_quiz_prompt(topic):
    return f"Create a short quiz with 3 questions about {topic}. Provide the answers at the end."
