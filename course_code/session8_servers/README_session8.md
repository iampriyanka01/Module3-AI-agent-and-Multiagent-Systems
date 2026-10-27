# Session 8: AI Communication Protocols and Building an MCP Server

- `Agents_Session_8_MCP_Building_an_MCP_Server.ipynb` (**Part 1**): MCP concepts, building servers with FastMCP, and Claude Desktop. No API key needed.
- `Agents_Session_8_Part2_MCP_with_CrewAI_and_A2A.ipynb` (**Part 2**): giving CrewAI agents tools from an MCP server, an introduction to A2A, and the session's conclusions. Needs your Gemini key.

Both work in **Google Colab** and in **VS Code**. **They need different environments:** FastMCP 4 (Part 1) and CrewAI (Part 2) require different versions of the same library, so they can't be installed together.
- `servers/`: the three course servers, plus two helpers:

| File | What it is |
|---|---|
| `servers/mcpserver_tools.py` | Three tools (add, multiply, greet) and one resource |
| `servers/mcpserver_resources.py` | Two resources (a greeting and a JSON config) |
| `servers/mcpserver_prompt.py` | Three prompts (explain a topic, a quiz, summarise notes) |
| `servers/test_client.py` | An MCP client that tests any server without Claude Desktop |
| `servers/print_claude_config.py` | Prints the Claude Desktop configuration for your computer |

## In Google Colab

Upload a notebook and run the cells from the top. **Open Part 2 in a new Colab session**, not the one you used for Part 1. Everything works except connecting to Claude Desktop, which needs your own computer.

For Part 2, save your Gemini key once in Colab Secrets: the **key icon** in the left sidebar, a secret named `GOOGLE_API_KEY`, with **Notebook access** switched on.

## In VS Code (needed for Claude Desktop)

1. Install Python 3.10 or newer, VS Code, and the **Python** and **Jupyter** extensions.
2. **Part 1:** in a terminal in this folder, create a **new** virtual environment (not the CrewAI one from Sessions 6 and 7):

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Mac or Linux
pip install -r requirements_part1.txt
```

Open the Part 1 notebook, choose the `.venv` kernel (top right), and run the cells.

3. **Part 2:** use your **Session 6 and 7 environment** (it already has CrewAI), or make a second one:

```bash
python -m venv .venv-part2
.venv-part2\Scripts\activate    # Windows
source .venv-part2/bin/activate # Mac or Linux
pip install -r requirements_part2.txt
```

Copy `.env.example` to `.env` and add your Gemini key. Open the Part 2 notebook and choose that environment's kernel.

## Testing the servers from a terminal

```bash
cd servers
python test_client.py mcpserver_tools.py
python test_client.py mcpserver_resources.py
python test_client.py mcpserver_prompt.py
```

## Connecting to Claude Desktop

1. Install Claude Desktop from [claude.ai/download](https://claude.ai/download) and sign in.
2. With your virtual environment active, run `python servers/print_claude_config.py`. It prints the configuration with the right paths for your computer.
3. In Claude Desktop: **Settings > Developer > Edit Config**, paste the configuration, and save.
4. **Quit Claude Desktop completely and reopen it.** Your servers should show as **running** in Settings > Developer.
5. Ask Claude: "Use the math server to multiply 12.5 by 8."

If a server doesn't appear, check the paths, check FastMCP is installed in that Python, restart Claude Desktop fully, and read its MCP logs (macOS: `~/Library/Logs/Claude/`; Windows: `%APPDATA%\Claude\logs\`).
