"""Design pattern 4: SEQUENTIAL PIPELINE  (Session 7: rule-based coordination)

Tasks run in a fixed order. Each task's result becomes the next task's input, like a CI/CD pipeline:
research -> analysis -> report. In CrewAI, that's Process.sequential plus context=[...].

Three C's in this file:
  Communication  context=[research_task] hands the research to the analyst (a direct hand-off)
  Coordination   the order is fixed by us: research must finish before analysis starts
  Cooperation    three specialists build one report together

Run:   python sequential_pipeline.py
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


researcher = Agent(
    role="AI research analyst",
    goal="Find well-documented facts about AI in software development, using Wikipedia.",
    backstory="You never state a fact you haven't found in your research.",
    tools=[search_wikipedia],
    llm=llm,
)

analyst = Agent(
    role="Data analyst",
    goal="Pick the most promising idea from the research and explain why.",
    backstory="You turn raw research into a clear recommendation.",
    llm=llm,
)

report_writer = Agent(
    role="Executive summary writer",
    goal="Write a short, polished summary for a business audience.",
    backstory="You write clearly and briefly, and you only use facts you were given.",
    llm=llm,
)

research_task = Task(
    description="Research three important uses of AI in software development: large language models, "
                "automation and MLOps. Use Wikipedia.",
    expected_output="A bullet list of three uses, two sentences each, based on the research.",
    agent=researcher,
)

analysis_task = Task(
    description="From the research, choose the most promising use for a new software product, and explain why.",
    expected_output="The chosen use, and three bullet-point reasons.",
    agent=analyst,
    context=[research_task],                  # communication: the analyst reads the research
)

report_task = Task(
    description="Write a one-page summary titled 'AI in Software Development': the chosen use, the reasons, "
                "and a conclusion.",
    expected_output="A summary of about 200 words.",
    agent=report_writer,
    context=[analysis_task],
)

crew = Crew(
    agents=[researcher, analyst, report_writer],
    tasks=[research_task, analysis_task, report_task],
    process=Process.sequential,               # coordination: one after another, in this order
    max_rpm=4,
    verbose=True,                             # True or False (the old file's verbose=2 is not allowed)
)

if __name__ == "__main__":
    result = crew.kickoff()
    print("\n=== FINAL SUMMARY ===\n")
    print(result.raw)
