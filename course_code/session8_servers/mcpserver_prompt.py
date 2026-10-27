"""An MCP server with three PROMPTS: ready-made instructions a user can pick in the client.

A prompt returns either:
  - a string: one message from the user, or
  - a list of Message objects: a short conversation (user and assistant turns).
(FastMCP 4 rejects a single Message on its own, so use a string or a list.)

Test it without Claude:       python test_client.py mcpserver_prompt.py
"""

from fastmcp import FastMCP
from fastmcp.prompts import Message

mcp = FastMCP(name="PromptServer")


# 1. A prompt that returns a string (the simplest kind)
@mcp.prompt
def explain_topic(topic: str) -> str:
    """Asks the model to explain a study topic."""
    return f"Explain the concept of '{topic}' in simple terms with a short example."


# 2. A prompt that returns a short conversation
@mcp.prompt
def generate_quiz(topic: str, difficulty: str = "easy") -> list[Message]:
    """Asks the model to create a few quiz questions for a topic."""
    return [
        Message(f"I'm studying {topic}. Can you quiz me?"),
        Message("Of course! What difficulty would you like?", role="assistant"),
        Message(f"{difficulty}. Please give 3 questions, each with 4 options, and say which answer is correct."),
    ]


# 3. Another string prompt, with the user's notes inside it
@mcp.prompt
def summarize_notes(notes: str) -> str:
    """Summarizes long notes into a short recap."""
    return f"Summarize the following notes into 3 bullet points for quick revision:\n\n{notes}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
