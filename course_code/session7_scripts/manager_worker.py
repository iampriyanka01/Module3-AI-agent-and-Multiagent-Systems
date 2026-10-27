"""Design pattern 1: MANAGER / WORKER  (Session 7: centralized coordination)

A manager receives the tasks and decides which worker does each one. In CrewAI, that's
Process.hierarchical: CrewAI creates the manager for you from manager_llm.

The one rule that makes this real manager/worker: DON'T assign the tasks to agents.
If a task says agent=..., the manager can only pass it to that agent, so it has no choice.

Three C's in this file:
  Communication  task results flow to later tasks through context=[...]
  Coordination   the manager decides who does what (task allocation)
  Cooperation    analyst, optimizer and writer together produce one report

Run:   python manager_worker.py
"""

import os
from getpass import getpass

from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import tool
from dotenv import load_dotenv

os.environ["CREWAI_TRACING_ENABLED"] = "false"   # stops CrewAI asking a yes/no question on its first run
load_dotenv(".env")                              # reads GOOGLE_API_KEY from a .env file, if there is one
if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass("Paste your Gemini API key: ")

llm = LLM(model="gemini/gemini-3.5-flash-lite", api_key=os.environ["GOOGLE_API_KEY"])


@tool("Customer support data")
def get_support_data(query: str) -> str:
    """Get last quarter's customer support summary: ticket counts, resolution times and feedback."""
    # PRETEND DATA for this exercise. A real company would read this from its support system.
    return ("Last quarter's support data:\n"
            "- 50 tickets: login issues (average 48 hours to resolve)\n"
            "- 30 tickets: billing errors (average 12 hours to resolve)\n"
            "- 20 tickets: feature requests (often closed without a fix)\n"
            "- Feedback: the password reset screen is confusing\n"
            "- Many calls about account verification")


# ----- The workers -----
data_analyst = Agent(
    role="Customer support data analyst",
    goal="Find the biggest problems in the support data, with numbers.",
    backstory="You read support data carefully and always quote the figures.",
    tools=[get_support_data],
    llm=llm,
)

process_optimizer = Agent(
    role="Process improvement specialist",
    goal="Turn the analysis into practical improvements.",
    backstory="You suggest specific, realistic fixes, not vague advice.",
    llm=llm,
)

report_writer = Agent(
    role="Executive report writer",
    goal="Write a short, clear report for the company's COO.",
    backstory="Your reports are brief, specific and easy to act on.",
    llm=llm,
)

# ----- The tasks: no agent= on purpose, so the manager chooses who does each one -----
analysis_task = Task(
    description="Analyse last quarter's customer support data and find the main problems.",
    expected_output="The main problems, with ticket counts and resolution times.",
)

improvement_task = Task(
    description="Based on the analysis, suggest three concrete improvements to the support process.",
    expected_output="Three improvements, each one or two sentences.",
    context=[analysis_task],                  # communication: reads the analysis
)

report_task = Task(
    description="Write a short report for the COO: a summary, the problems, and the improvements.",
    expected_output="A report of about 200 words.",
    context=[analysis_task, improvement_task],
)

crew = Crew(
    agents=[data_analyst, process_optimizer, report_writer],
    tasks=[analysis_task, improvement_task, report_task],
    process=Process.hierarchical,             # coordination: a manager decides
    manager_llm=llm,                          # CrewAI builds the manager agent from this
    max_rpm=4,                                # at most 4 model calls a minute (free tier)
    verbose=True,                             # watch the manager delegate
)

if __name__ == "__main__":
    result = crew.kickoff()
    print("\n=== FINAL REPORT ===\n")
    print(result.raw)
