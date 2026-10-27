"""All three COMMUNICATION PATTERNS in one small system  (Session 7, section 1)

A researcher, a writer and an editor produce a short article. Each hand-off shows one pattern:

  1. Shared memory (blackboard idea)  The researcher's findings are stored as a task result that later
                                      tasks can read. The writer reads them through context=[...],
                                      without asking the researcher anything.
  2. Direct (request-response idea)   The writer's draft goes straight to the editor, one-to-one,
                                      through context=[writing_task].
  3. Broadcast (publish-subscribe)    The editor ends with an announcement written for everyone,
                                      not for one agent: a one-to-many message.

CrewAI implements all three with task results and context. The patterns are about WHO receives
the information and HOW: one reader, a shared store, or everyone.

Run:   python crewai_comm_pattern.py
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


researcher = Agent(role="Researcher", goal="Find accurate facts using Wikipedia.",
                   backstory="You only report what your sources say.", tools=[search_wikipedia], llm=llm)
writer = Agent(role="Writer", goal="Write a short, clear article from the research.",
               backstory="You write simply and only use the facts you're given.", llm=llm)
editor = Agent(role="Editor", goal="Polish the article and announce it to the whole team.",
               backstory="You fix mistakes, tighten the writing, and keep everyone informed.", llm=llm)

# 1. SHARED MEMORY: the research result is stored for later tasks to read
research_task = Task(
    description="Find 4 facts about {topic} using Wikipedia.",
    expected_output="4 bullet points of facts.",
    agent=researcher,
)

# The writer reads the stored research (shared memory), without messaging the researcher
writing_task = Task(
    description="Write a 120-word article about {topic} using the research.",
    expected_output="A 120-word article.",
    agent=writer,
    context=[research_task],
)

# 2. DIRECT: the draft goes one-to-one from the writer to the editor
# 3. BROADCAST: the editor's output ends with an announcement for everyone
editing_task = Task(
    description="Edit the article for clarity. Then add a final line that starts with 'ANNOUNCEMENT TO ALL TEAMS:' "
                "telling everyone the article is ready and what it's about.",
    expected_output="The edited article, then the announcement line.",
    agent=editor,
    context=[writing_task],
)

crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, writing_task, editing_task],
    process=Process.sequential,
    max_rpm=4,
    verbose=True,
)

if __name__ == "__main__":
    result = crew.kickoff(inputs={"topic": "electric vehicles"})
    for task_output in result.tasks_output:
        print(f"\n=== {task_output.agent} ===\n{task_output.raw}")
