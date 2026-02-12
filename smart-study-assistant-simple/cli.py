import typer
from basic_core import tutor, quiz

app = typer.Typer()

@app.command()
def tutor_cmd(topic: str, level: str = "beginner"):
    response = tutor(topic, level)
    print(response)

@app.command()
def quiz_cmd(topic: str):
    response = quiz(topic)
    print(response)

if __name__ == "__main__":
    app() 