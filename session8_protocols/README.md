# Session 8: AI Communication Protocols, Building an MCP Server, and A2A

```bash
streamlit run app.py        # or open it from the hub: streamlit run ../app.py
```

Covers the session PDF, both notebooks (Part 1: FastMCP and Claude Desktop; Part 2: MCP with CrewAI and A2A), the `servers/` folder, and the A2A post-read.

| Page | Lab |
|---|---|
| 2 | Send raw JSON-RPC requests to a working mini MCP server (the course's three servers) |
| 3 | Paste a FastMCP file: see what it registers and what will go wrong (static check, never runs your code) |
| 4 | Sampling, roots, elicitation: the course's restaurant mapping AND the specification |
| 5 | STDIO vs Streamable HTTP |
| 6 | FastMCP step by step, and a host whose LLM decides when to call your tools (Live-capable) |
| 7 | A checker for `claude_desktop_config.json` (JSON errors, Windows paths, transport mismatch) |
| 8 | MCP in CrewAI (`mcps=[...]`) and CrewAI 1.15's exact tool-name rule (why names get cut) |
| 9–10 | A2A: Agent Cards, schema-checked messages, task states, patterns, MCP vs A2A, tracing with correlation IDs |

`protocols_engine.py` is plain Python (MiniMCP mirrors FastMCP 4.1.4's behaviour). No key, no network needed.
