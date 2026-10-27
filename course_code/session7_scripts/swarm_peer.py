"""Design pattern 2: PEER / SWARM  (Session 7: distributed coordination)

Three equals (a developer, a tester and a writer) take turns on one feature, with no manager.
Each one can ALSO ask the others for help directly, because of allow_delegation=True.
That direct help is the peer-to-peer part.

CrewAI has no true "swarm" process, so, as in the session plan, the turns run in order with
Process.sequential. The peers asking each other is what makes it more than a plain pipeline.

Three C's in this file:
  Communication  context=[...] passes the code and the test report along; peers can also ask each other
  Coordination   no boss: the turn order is agreed in advance, and peers sort out questions among themselves
  Cooperation    the tester finds problems, the developer's work improves, the writer documents the result

Run:   python swarm_peer.py
"""

import os
from getpass import getpass

from crewai import LLM, Agent, Crew, Process, Task
from dotenv import load_dotenv

os.environ["CREWAI_TRACING_ENABLED"] = "false"
load_dotenv(".env")
if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass("Paste your Gemini API key: ")

llm = LLM(model="gemini/gemini-3.5-flash-lite", api_key=os.environ["GOOGLE_API_KEY"])

developer = Agent(
    role="Software developer",
    goal="Write clean, correct, well-commented Python code.",
    backstory="You write careful code and welcome questions from teammates.",
    llm=llm,
    allow_delegation=True,                    # can ask the other peers for help
)

tester = Agent(
    role="QA engineer",
    goal="Find bugs and edge cases before users do.",
    backstory="You think about what could go wrong. If something is unclear, you ask the developer.",
    llm=llm,
    allow_delegation=True,
)

documentation_writer = Agent(
    role="Technical writer",
    goal="Write clear, accurate documentation for other developers.",
    backstory="You explain code simply, and you check details with the developer when unsure.",
    llm=llm,
    allow_delegation=True,
)

coding_task = Task(
    description="Write a Python function called calculate_average that takes a list of numbers and returns "
                "their average. Handle an empty list sensibly. Add comments.",
    expected_output="The complete Python function.",
    agent=developer,
)

testing_task = Task(
    description="Review calculate_average. Think through test cases: positive numbers, negative numbers, and "
                "an empty list. Report any bugs you find.",
    expected_output="The test cases, the expected result of each, and any bugs found.",
    agent=tester,
    context=[coding_task],                    # communication: the tester reads the code
)

documentation_task = Task(
    description="Write documentation for calculate_average: what it does, its parameter, what it returns, "
                "what happens with an empty list, and two usage examples.",
    expected_output="Short documentation in Markdown.",
    agent=documentation_writer,
    context=[coding_task, testing_task],      # reads both the code and the test report
)

crew = Crew(
    agents=[developer, tester, documentation_writer],
    tasks=[coding_task, testing_task, documentation_task],
    process=Process.sequential,
    max_rpm=4,
    verbose=True,                             # look for peers asking each other questions
)

if __name__ == "__main__":
    result = crew.kickoff()
    print("\n=== FINAL DOCUMENTATION ===\n")
    print(result.raw)
