"""The two specialist agents, and how they talk to each other (Sessions 1, 4 and 5).

- The flight agent finds flights that suit the couple.
- The destination agent covers weather, places, hotels and the first day. When it needs to know
  when the couple lands, it asks the flight agent directly: two agents talking (Session 5).

Each agent is one create_agent(...) call: a model, its tools, and its instructions from prompts.yaml.

Try it:   python agents.py
"""

from datetime import date, timedelta

from langchain.agents import create_agent

import memory
import tools
from config import PROMPTS, llm

# What the agents share during one message: whose trip it is, and what the flight agent found
trip = {"couple": "", "flight_report": ""}


def say(sender, receiver, text):
    """Print one message between members of the team, so everyone can see who talks to whom."""
    short = text.replace("\n", " ")
    if len(short) > 150:
        short = short[:150] + "..."
    print(f"   [{sender} -> {receiver}] {short}")


def run_agent(agent, task, name):
    """Run an agent on a task, print each tool it used, and return its final answer."""
    result = agent.invoke({"messages": [{"role": "user", "content": task}]}, config={"recursion_limit": 25})
    for message in result["messages"]:
        if message.type == "ai":
            for call in message.tool_calls:
                print(f"      [{name}] used {call['name']}({call['args']})")
    return result["messages"][-1].text


# ---------------- tools that need to know whose trip it is ----------------

def recall_memories(question: str) -> str:
    """Recall what DilSeTrips knows about this couple that matters for a question.

    Args:
        question: What you want to know, for example 'flight preferences: stops, timing, seats'.
    """
    return memory.recall(trip["couple"], question)


def ask_flight_agent(question: str) -> str:
    """Ask the flight agent on your team a question about the couple's flight, for example when they land.

    Args:
        question: Your question for the flight agent.
    """
    say("destination agent", "flight agent", question)
    if trip["flight_report"] == "":
        answer = "There is no flight plan for this trip yet."
    else:
        answer = llm.invoke(PROMPTS["peer_answer"].format(question=question, flight_report=trip["flight_report"])).text
    say("flight agent", "destination agent", answer)
    return answer


def book_flight(flight_numbers: str) -> str:
    """Try to book a flight for the couple.

    Args:
        flight_numbers: The flight numbers to book, for example 6E 6393.
    """
    return "Sorry, I can't book flights at this time. The couple can book directly with the airline."


def book_hotel(hotel_name: str) -> str:
    """Try to book a hotel for the couple.

    Args:
        hotel_name: The name of the hotel to book.
    """
    return "Sorry, I can't book hotels at this time. The couple can book directly with the hotel."


# ---------------- the two agents ----------------

flight_agent = create_agent(
    model=llm,
    tools=[recall_memories, tools.find_airport, tools.search_flights, book_flight],
    system_prompt=PROMPTS["flight_agent"],
)

destination_agent = create_agent(
    model=llm,
    tools=[recall_memories, tools.get_weather_forecast, tools.search_travel_guide, ask_flight_agent,
           tools.find_hotels, book_hotel],
    system_prompt=PROMPTS["destination_agent"],
)


if __name__ == "__main__":
    trip["couple"] = "prasad"
    travel_day = (date.today() + timedelta(days=45)).isoformat()
    report = run_agent(flight_agent, f"Find a flight for two from Mumbai to Chandigarh on {travel_day}.", "flight agent")
    print("\nFLIGHT AGENT REPORT:\n" + report)
