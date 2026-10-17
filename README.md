# Agents Course Lab: six interactive Streamlit apps

| Session | Folder | What it teaches |
|---|---|---|
| 1 | `session1_foundations/` | What an agent is, the agent loop, tools, robustness, workflow patterns |
| 2 | `session2_building_blocks/` | Embeddings, ReAct vs plan, PEAS and agent types, safety/approval gates, limits, evaluation |
| 3 | `session3_memory/` | Short- and long-term memory, forgetting, CoALA kinds, risks |
| 4 | `session4_architecture/` | Architectures and patterns |
| 5 | `session5_multi_agent/` | Multi-agent systems: why a team, centralized / decentralized / hybrid, four design patterns, what a team costs |
| 6 | `session6_building_mas/` | Building multi-agent systems: a ReAct agent with tool and memory, centralized / decentralized / hybrid support desks, LangChain vs CrewAI (crew, agent, task, sequential and hierarchical process) |

## Run

```bash
pip install -r requirements.txt
streamlit run app.py                      # the hub: one app that connects all six sessions
streamlit run session1_foundations/app.py # or any session on its own
```

The hub gives a course map with a concept index (deep links into pages), keeps each
session's state separate, and shares the mode and API key across sessions.

## Demo vs Live
- **Demo** (default): no key, no network. Scripted "models", pretend data and a toy embedding. Good for learning how the ideas work.
- **Live**: paste a Gemini API key in the sidebar (or put it in `.streamlit/secrets.toml`, see the `.example` file). It falls back to Demo on any error. Live mode has not been tested end to end.

Each session has the same structure: explanation pages, a hands-on lab per concept, and a "code peek".
