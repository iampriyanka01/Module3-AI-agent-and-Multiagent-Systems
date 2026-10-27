"""Design pattern 3: FEDERATED TEAMS / SPECIALIST AGENTS  (Session 7: centralized coordination of teams)

A project manager oversees specialist teams and gives each part of the work to the right specialist:
  Research team:  a researcher and a data analyst
  Marketing team: a content strategist
In CrewAI, that's Process.hierarchical with our own manager_agent.

Two rules make it work:
  1. Leave the task UNASSIGNED (no agent=...). The old version assigned it to the manager itself,
     so the manager's only "coworker" was itself and the specialists were never used.
  2. A manager agent has NO tools. Its only job is to delegate.

Three C's in this file:
  Communication  the manager sends work to specialists and receives their results
  Coordination   the manager decides which team handles each part (task allocation)
  Cooperation    research, analysis and writing combine into one report

Run:   python federated_teams_specialist_agents.py
"""

import os
from getpass import getpass

import requests
from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import tool
from dotenv import load_dotenv

os.environ["CREWAI_TRACING_ENABLED"] = "false"
load_dotenv(".env")
if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass("Paste your Gemini API key: ")

llm = LLM(model="gemini/gemini-3.5-flash-lite", api_key=os.environ["GOOGLE_API_KEY"])


@tool("Wikipedia search")
def search_wikipedia(topic: str) -> str:
    """Search Wikipedia for a topic and return the opening of the best-matching article."""
    headers = {"User-Agent": "GenAI-course/1.0 (educational use)"}
    found = requests.get("https://en.wikipedia.org/w/api.php", headers=headers,
                         params={"action": "query", "list": "search", "srsearch": topic, "format": "json"}).json()
    results = found["query"]["search"]
    if len(results) == 0:
        return f"No Wikipedia article found for '{topic}'."
    title = results[0]["title"]
    page = requests.get("https://en.wikipedia.org/w/api.php", headers=headers,
                        params={"action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1,
                                "titles": title, "format": "json"}).json()
    for page_id in page["query"]["pages"]:
        return f"{title}: " + page["query"]["pages"][page_id].get("extract", "")[:2500]
    return "No text found."


# ----- Research team -----
researcher = Agent(
    role="Lead researcher",
    goal="Find well-documented information about robotics trends, using Wikipedia.",
    backstory="You research carefully and only report what your sources say.",
    tools=[search_wikipedia],
    llm=llm,
)

analyst = Agent(
    role="Data analyst",
    goal="Turn research into clear insights about market impact.",
    backstory="You spot patterns and explain why they matter for businesses.",
    llm=llm,
)

# ----- Marketing team -----
writer = Agent(
    role="Content strategist",
    goal="Turn findings into an engaging summary for executives.",
    backstory="You write clear executive summaries from the facts you're given.",
    llm=llm,
)

# ----- The manager: no tools, its job is to delegate -----
project_manager = Agent(
    role="Project manager",
    goal="Deliver a high-quality report by giving each part of the work to the right specialist.",
    backstory="You run research and marketing teams, and you know who is best for each job.",
    llm=llm,
)

report_task = Task(
    description="Create a report on three important robotics trends: (1) a short overview of each trend, "
                "(2) its likely market impact, (3) a summary for executives.",
    expected_output="A well-structured report in Markdown, about 300 words.",
    # no agent= here: the manager decides who does each part
)

crew = Crew(
    agents=[researcher, analyst, writer],     # the manager is NOT in this list
    tasks=[report_task],
    process=Process.hierarchical,
    manager_agent=project_manager,
    max_rpm=4,
    verbose=True,                             # watch the manager delegate to each team
)

if __name__ == "__main__":
    result = crew.kickoff()
    print("\n=== FINAL REPORT ===\n")
    print(result.raw)
