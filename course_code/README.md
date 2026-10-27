# Course code library

| Folder | Session | What it is |
|---|---|---|
| `session4_supervisor_agent/` | 4 | LangChain supervisor that calls a weather agent and a travel agent as tools (`main.py`), and a one-agent version (`main1.py`) |
| `session6_dilsetrips/` | 6 | The DilSeTrips concierge project: agents, tools, memory, prompts, coordinator |
| `session7_scripts/` | 7 | The five CrewAI demos: communication patterns, manager/worker, pipeline, peer/swarm, federated teams |
| `session8_servers/` | 8 | FastMCP servers (tools, resources, prompts), `test_client.py`, `print_claude_config.py` |
| `notebooks/` | 3–8 | The session notebooks with outputs cleared |

Each project needs a Gemini key in a `.env` file (`GOOGLE_API_KEY=...`), except the Session 8 servers, which need none.
Session 8 Part 1 (FastMCP 4) and CrewAI need **separate virtual environments**: they require different versions of the `mcp` library.
The `.venv` and `.env` from the original Session 4 folder were left out on purpose.
