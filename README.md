# Agents Course Lab: eight interactive Streamlit apps, exam practice and the course code

| Session | Folder | What it teaches |
|---|---|---|
| 1 | `session1_foundations/` | What an agent is, the agent loop, tools, robustness, workflow patterns |
| 2 | `session2_building_blocks/` | Embeddings, ReAct vs plan, PEAS and agent types, safety/approval gates, limits, evaluation |
| 3 | `session3_memory/` | Short- and long-term memory, forgetting, CoALA kinds, risks |
| 4 | `session4_architecture/` | Architectures and patterns |
| 5 | `session5_multi_agent/` | Multi-agent systems: why a team, centralized / decentralized / hybrid, four design patterns, what a team costs (with the notebook's diagrams) |
| 6 | `session6_building_mas/` | Building multi-agent systems: how LangChain works, the notebook 6A desks (agents as tools, step limits, router and peer questions), how CrewAI works (prompts recorded from the real library), crews, processes and allow_delegation |
| 7 | `session7_three_cs/` | The Three C's: communication patterns (request-response, pub/sub, blackboard), task/resource/conflict problems, four coordination mechanisms, the four patterns in CrewAI, a crew debugger with CrewAI's real errors, and multi-agent reinforcement learning (post-read) |
| 8 | `session8_protocols/` | Protocols: MCP host/client/server, live JSON-RPC to a mini MCP server, tools/resources/prompts, sampling/roots/elicitation (course vs spec), transports, FastMCP, a Claude Desktop config checker, MCP in CrewAI, and A2A (Agent Cards, task states, tracing) |

Also in this folder:

| Folder / file | What it is |
|---|---|
| `exam_bank.py` | Scenario questions in the style of the weekly assignments, with the course answer and a real-library note (hub → *Exam practice*) |
| `course_code/` | Every course script, server and notebook (outputs cleared), browsable and downloadable in the hub (→ *Code library*) |

## Run

```bash
pip install -r requirements.txt
streamlit run app.py                      # the hub: one app that connects all eight sessions
streamlit run session1_foundations/app.py # or any session on its own
```

The hub gives a course map with a concept index (deep links into pages), keeps each
session's state separate, and shares the mode and API key across sessions.

## Demo vs Live
- **Demo** (default): no key, no network. Scripted "models", pretend data and a toy embedding. Good for learning how the ideas work.
- **Live**: paste a Gemini API key in the sidebar (or put it in `.streamlit/secrets.toml`, see the `.example` file). It falls back to Demo on any error. Live mode has not been tested end to end.

Each session has the same structure: explanation pages, a hands-on lab per concept, and a "code peek".

Sessions 5 to 8 cover their PDFs, post-reads and notebooks completely, so the apps can be used instead of reading them.

## Checked against the real libraries
The Session 7 crew debugger and the Session 8 labs were checked against **CrewAI 1.15.30** and **FastMCP 4.1.4** (with a stub model, no API calls).
Where a course quiz answer differs from what the library does, the app shows both. The main differences:

- A CrewAI task with **no** `context` parameter receives **all** earlier task outputs; only `context=[]` gives it nothing.
- A task whose `context` points to a **later** task stops the crew at `Crew(...)` with a validation error.
- `crew.kickoff()` with `output_file` returns a `CrewOutput` whose `.raw` is the full text (the course answer says a confirmation message).
- MCP's *sampling*, *roots* and *elicitation* follow the course's restaurant analogy in the quizzes, but the specification defines them differently (page 4 of Session 8).
