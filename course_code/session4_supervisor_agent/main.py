# 1. import
import os

from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI

from tools.travel import search_travel_guide
from tools.weather import get_weather_forecast

load_dotenv()  # loads API keys from .env


def text_of(message):
    """Get plain text from an agent message (Gemini sometimes returns a list of content parts)."""
    content = message.content
    if isinstance(content, list):
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return content


# 2. Add wrapper for python function to be used in the main.py file
@tool
def travel_guide(place: str) -> str:
    """Get travel guide information (sights, food, how to get there) for a city or place."""
    try:
        return search_travel_guide(place)
    except Exception as error:  # e.g. KeyError if Wikivoyage returns an error instead of data
        return f"The travel guide failed for {place} ({type(error).__name__}: {error})."


@tool
def weather_forecast(place: str, region: str = "") -> str:
    """Get the 7-day daily weather forecast (min/max temperature in C and rain in mm) for a place.
    Use `region` (state/province, e.g. "Punjab") to pick the right place when the name is ambiguous."""
    try:
        return get_weather_forecast(place, region)
    except Exception as error:  # e.g. KeyError 'daily' if Open-Meteo returns an error instead of a forecast
        return f"The weather service failed for {place} ({type(error).__name__}: {error})."


# initiate the LLM (shared by all three agents)
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.2,    # at most 1 request every 5 seconds (stays under free-tier limits)
    check_every_n_seconds=0.1,
    max_bucket_size=1,
)

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite",
                             google_api_key=os.environ["GOOGLE_API_KEY"],
                             rate_limiter=rate_limiter)


# create the agents
# --- Weather agent: only knows about weather ---
weather_agent = create_agent(
    model=llm,
    tools=[weather_forecast],
    system_prompt=(
        "You are a weather specialist. Use the weather_forecast tool to answer questions about "
        "the weather for a place. Summarise the forecast clearly: temperatures, rain, and what to pack."
    ),
)

# --- Travel agent: only knows about places to visit ---
travel_agent = create_agent(
    model=llm,
    tools=[travel_guide],
    system_prompt=(
        "You are a travel guide specialist. Use the travel_guide tool to answer questions about "
        "a place: top sights, food, how to get there and get around. Keep it short and practical."
    ),
)


# --- Wrap the specialist agents as tools so the supervisor can call them ---
@tool
def ask_weather_agent(request: str) -> str:
    """Ask the weather agent about the weather forecast for a place.
    Pass a full, clear request, e.g. 'What is the weather in Manali this week?'"""
    result = weather_agent.invoke({"messages": [{"role": "user", "content": request}]})
    return text_of(result["messages"][-1])


@tool
def ask_travel_agent(request: str) -> str:
    """Ask the travel agent about sights, food, and getting around in a place.
    Pass a full, clear request, e.g. 'What are the top things to see in Jaipur?'"""
    result = travel_agent.invoke({"messages": [{"role": "user", "content": request}]})
    return text_of(result["messages"][-1])


# --- Supervisor agent: decides which specialist(s) to ask and combines the answers ---
supervisor_agent = create_agent(
    model=llm,
    tools=[ask_weather_agent, ask_travel_agent],
    system_prompt=(
        "You are a travel assistant supervisor. You do not answer from your own knowledge. "
        "For weather questions, ask the weather agent. For questions about places, sights, food "
        "or getting around, ask the travel agent. If the user asks about both, ask both agents. "
        "Then combine their answers into one friendly, natural reply for the user."
    ),
)


# ask the question to the agent and get the answer
def ask(question, history):
    """Send a question to the supervisor (with earlier chat for context) and return its answer."""
    history.append({"role": "user", "content": question})
    result = supervisor_agent.invoke({"messages": history})
    answer = text_of(result["messages"][-1])
    history.append({"role": "assistant", "content": answer})
    return answer


if __name__ == "__main__":
    print("Travel assistant ready! Ask about any place's weather or sights. Type 'exit' to quit.\n")
    history = []  # remembers the conversation, so follow-ups like "what about Shimla?" work

    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit", "bye"):
            print("Assistant: Happy travels!")
            break
        if not question:
            continue
        try:
            print(f"\nAssistant: {ask(question, history)}\n")
        except Exception as error:
            history.pop()  # drop the question that failed
            print(f"\nAssistant: Sorry, something went wrong ({type(error).__name__}: {error})\n")