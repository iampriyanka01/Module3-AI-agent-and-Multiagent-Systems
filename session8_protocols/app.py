"""
Protocols Lab: an interactive companion to "Agents, Session 8: AI Communication Protocols & Building an MCP Server"
(the session PDF, both notebooks, the servers/ folder) and the post-read "Agent-to-Agent (A2A) Communication Protocols".

Run:   streamlit run app.py
Works with NO API key and NO network (Demo mode). In Live mode, Gemini plays the host's LLM on page 6.
"""
import inspect
import json
import os

import pandas as pd
import streamlit as st

import protocols_engine as pe

st.set_page_config(page_title="Protocols Lab: MCP & A2A", page_icon="🔌", layout="wide")

PAGES = [
    "🏠 Start here",
    "1 · Why protocols?",
    "2 · MCP: host, client, server & messages",
    "3 · Server features: tools, resources, prompts",
    "4 · Client features: sampling, roots, elicitation",
    "5 · Transports: STDIO vs HTTP",
    "6 · Build an MCP server with FastMCP",
    "7 · Connect it to Claude Desktop",
    "8 · MCP with CrewAI",
    "9 · A2A: agents talking to agents",
    "10 · MCP vs A2A, challenges & tracing",
    "11 · Review, quiz & cheat sheet",
]

try:
    STRETCH = {"width": "stretch"} if inspect.signature(st.dataframe).parameters["width"].default == "stretch" else {"use_container_width": True}
except Exception:
    STRETCH = {"use_container_width": True}

NODE = 'node [shape=box, style="rounded,filled", fillcolor="#eef2ff", fontname="Helvetica"];'


def _default_key() -> str:
    key = os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("GOOGLE_API_KEY", "")
        except Exception:
            key = ""
    return key


def get_backend() -> pe.Backend:
    mode = st.session_state.get("mode", "Demo (no API key)")
    key = st.session_state.get("api_key", "") if mode.startswith("Live") else ""
    cfg = (key, st.session_state.get("chat_model", "gemini-3.5-flash-lite"), st.session_state.get("rate_limit", True))
    if st.session_state.get("_cfg") != cfg or "backend" not in st.session_state:
        st.session_state.backend = pe.Backend(*cfg)
        st.session_state._cfg = cfg
    return st.session_state.backend


def goto(page: str):
    st.session_state.nav = page


def badge(b: pe.Backend):
    if b.last_source == "gemini":
        st.caption("🟢 Decided by Gemini")
    elif "fallback" in b.last_source:
        st.caption("🟠 " + (b.error or "Live call failed; showing the demo decision."))
    else:
        st.caption("🔵 Demo: a scripted host LLM (no API key needed). The MCP messages are real JSON-RPC handled by the MiniMCP server.")


def plain(text: str):
    st.info("**In plain words:** " + text)


def story(text: str):
    st.warning("**🎫 " + text + "**")


def predict(text: str):
    st.caption("🎲 **Before you run it:** " + text)


def notice(text: str):
    st.caption("🔎 **What to notice.** " + text)


def exam(text: str):
    st.success("**📝 Exam tip:** " + text)


def reality(text: str):
    st.caption("🧪 **Course answer vs the real specification / library.** " + text)


def df_show(rows, **kw):
    """Short tables: an interactive dataframe. Text-heavy tables (long sentences): a static table that wraps, so nothing is cut off."""
    df = pd.DataFrame(rows)
    wordy = len(df) and any(len(str(v)) > 70 for v in df.astype(str).values.flatten())
    if wordy and len(df) <= 25 and not kw:
        st.table(df.style.hide(axis="index"))
    else:
        st.dataframe(df, hide_index=True, **STRETCH, **kw)


def show_code(title: str, code: str, lang: str = "python"):
    with st.expander("🔍 " + title):
        st.code(code.strip("\n"), language=lang)


def dot(src: str):
    st.graphviz_chart(src)


def server() -> pe.MiniMCP:
    if "mini" not in st.session_state:
        st.session_state.mini = pe.course_server()
    return st.session_state.mini


# ----------------------------------------------------------------------------------------
def page_home():
    st.title("🔌 Protocols Lab: one tool, every AI app; one language, every agent")
    st.caption("An interactive companion to *Agents, Session 8: AI Communication Protocols & Building an MCP Server*, its two notebooks and servers/ folder, "
               "and the post-read *Agent-to-Agent (A2A) Communication Protocols*.")
    story("Prasad wrote a great weather tool for DilSeTrips' CrewAI agents. Then the team wanted it in Claude Desktop, and in an IDE, and in a partner's app. "
          "Rewriting the tool for every app doesn't scale. And when DilSeTrips' planner needed an airline's booking agent, it couldn't import someone else's code at all.")
    st.success("**Two protocols, two jobs.** **MCP** connects an agent to its **tools and data** (write the tool once, use it from any MCP client). "
               "**A2A** connects **agents to other agents**, across teams and companies. *MCP inside agents, A2A between agents.*")
    dot('digraph { rankdir=LR; ' + NODE + ' subgraph cluster_a { label="Your agent (host)"; LLM [fillcolor="#fff3bf"]; C1 [label="MCP client"]; C2 [label="MCP client"]; } '
        'S1 [label="MCP server\\nweather tools", fillcolor="#d3f9d8"]; S2 [label="MCP server\\nfiles / DB", fillcolor="#d3f9d8"]; '
        'B [label="Airline\'s agent\\n(another company)", fillcolor="#ffe3e3"]; LLM -> C1; LLM -> C2; C1 -> S1 [label="MCP"]; C2 -> S2 [label="MCP"]; LLM -> B [label="A2A", dir=both, color="#c92a2a"]; }')
    st.subheader("Your path")
    for page, desc in [
        (PAGES[1], "Protocols as traffic rules; function calling vs tool calling vs MCP."),
        (PAGES[2], "Host, client, server; send real JSON-RPC requests to a working mini MCP server."),
        (PAGES[3], "Tools (model), resources (app), prompts (user). Paste server code and see what gets registered."),
        (PAGES[4], "Sampling, roots, elicitation: the course's restaurant analogy AND the real definitions."),
        (PAGES[5], "STDIO vs Streamable HTTP, and when each fits."),
        (PAGES[6], "FastMCP step by step, a test client, and a host that decides when to call your tools."),
        (PAGES[7], "Check your claude_desktop_config.json before Claude Desktop does."),
        (PAGES[8], "Give CrewAI agents MCP tools with mcps=[...]; why tool names get cut."),
        (PAGES[9], "Agent Cards, structured messages, task states and A2A patterns."),
        (PAGES[10], "MCP vs A2A, common A2A problems, and tracing a failure with correlation IDs."),
        (PAGES[11], "Quiz, cheat sheet, exercises."),
    ]:
        a, b, c = st.columns([3, 6, 1.2])
        a.markdown("**" + page + "**")
        b.write(desc)
        c.button("Open", key="home_" + page, on_click=goto, args=(page,))
    reality("The session PDF explains a few MCP ideas in ways that differ from the official specification, and the course quizzes follow the PDF. "
            "This app shows both wherever they differ, so you can answer the exam AND build real servers. (Checked against FastMCP 4.1.4 and CrewAI 1.15.30.)")


def page_why():
    st.header("1 · Why protocols?")
    plain("A communication protocol is a shared rulebook: how messages are formatted, how they're sent, received and acknowledged, and what happens when something fails. "
          "Like traffic rules: who goes first, which side to drive on, what to do at a red light.")
    df_show(pe.PROTOCOL_WHY)
    st.subheader("Without and with MCP")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Without MCP:** every app needs custom code for every tool (M × N connections).")
        dot('digraph { rankdir=LR; ' + NODE + ' A1 [label="Claude Desktop"]; A2 [label="IDE"]; A3 [label="CrewAI agent"]; T1 [label="Weather"]; T2 [label="Files"]; T3 [label="Database"]; '
            'A1 -> T1; A1 -> T2; A1 -> T3; A2 -> T1; A2 -> T2; A2 -> T3; A3 -> T1; A3 -> T2; A3 -> T3; }')
    with c2:
        st.markdown("**With MCP:** each tool sits behind one server; any MCP client can use it (M + N).")
        dot('digraph { rankdir=LR; ' + NODE + ' A1 [label="Claude Desktop"]; A2 [label="IDE"]; A3 [label="CrewAI agent"]; P [label="MCP", shape=circle, fillcolor="#fff3bf"]; '
            'T1 [label="Weather server"]; T2 [label="Files server"]; T3 [label="DB server"]; A1 -> P; A2 -> P; A3 -> P; P -> T1; P -> T2; P -> T3; }')
    apps = st.slider("AI apps", 1, 10, 3, key="p1_apps")
    tools = st.slider("Tools", 1, 20, 3, key="p1_tools")
    m = st.columns(2)
    m[0].metric("Custom integrations without MCP", apps * tools)
    m[1].metric("Pieces to build with MCP (clients + servers)", apps + tools)
    st.subheader("Function calling, tool calling and MCP")
    df_show(pe.CALLING_VS_MCP)
    notice("Function calling and tool calling are about how a model ASKS for an action. MCP is about how tools are PACKAGED and SHARED so any AI app can find and use them. Think USB-C.")
    st.subheader("MCP's core rules")
    df_show(pe.MCP_PRINCIPLES)
    st.markdown("**Why MCP over plain API calls?** Tool-awareness (models invoke tools, not just return text) · structured responses · multi-turn context · deterministic, safer calls.")


def page_mcp():
    st.header("2 · MCP: host, client, server & messages")
    plain("The HOST is the app you use (Claude Desktop, an IDE, your own app); it holds the LLM and the conversation. Inside it, one MCP CLIENT per server keeps one connection. "
          "The SERVER is your code: it offers tools, resources and prompts. Restaurant: host = restaurant, client = waiter, server = kitchen.")
    dot('digraph { rankdir=LR; ' + NODE + ' subgraph cluster_h { label="Host (Claude Desktop)"; style=rounded; LLM [fillcolor="#fff3bf"]; C1 [label="Client 1"]; C2 [label="Client 2"]; LLM -> C1; LLM -> C2; } '
        'S1 [label="Server: math\\n(tools)", fillcolor="#d3f9d8"]; S2 [label="Server: data\\n(resources, prompts)", fillcolor="#d3f9d8"]; C1 -> S1 [dir=both]; C2 -> S2 [dir=both]; }')
    df_show([
        {"Role": "Host", "What it does": "The app the user sees; contains the LLM; creates one client per server", "Restaurant": "The restaurant"},
        {"Role": "MCP client", "What it does": "Sends structured requests to ONE server, receives responses, can track session context", "Restaurant": "The waiter"},
        {"Role": "MCP server", "What it does": "Offers tools, resources and prompts; validates requests; returns structured results or error codes", "Restaurant": "The kitchen"},
    ])
    exam("The **Session Manager** belongs to the MCP **client**, not the server. 'Which are essential server components?' → tools, prompts, resources.")
    st.subheader("🧪 Lab: talk to an MCP server in raw JSON-RPC")
    st.markdown("This is a working mini server holding the course's three servers (math tools, data resources, study prompts) plus a `divide_numbers` tool for showing failures. "
                "Pick a request, or edit it, and send it.")
    pick = st.selectbox("Request", list(pe.REQUEST_EXAMPLES), key="p2_req")
    raw = st.text_area("Request JSON (client → server)", json.dumps(pe.REQUEST_EXAMPLES[pick], indent=2), height=170, key="p2_raw_" + pick)
    resp = server().handle(raw)
    st.markdown("**Response (server → client)**")
    st.code(json.dumps(resp, indent=2, ensure_ascii=False), language="json")
    if "error" in resp:
        st.error("A protocol-level error: code %s. Standard codes: -32700 parse error, -32600 invalid request, -32601 method not found, -32602 invalid params." % resp["error"]["code"])
    elif resp.get("result", {}).get("isError"):
        st.warning("The call reached the tool, but it failed. FastMCP returns that as a normal result flagged **isError: true**, so the model can read the problem and recover.")
    notice("The same **id** links each reply to its request. Methods follow one pattern: tools/list, tools/call, resources/read, prompts/get. FastMCP writes and reads all of this for you.")
    reality("The PDF says each message has 'type, payload and metadata'. Real MCP uses **JSON-RPC 2.0**: jsonrpc, id, method, params → result or error. "
            "The PDF also says MCP lets models talk to each other; MCP connects apps to tools and data. Agent-to-agent is A2A's job.")


def page_server():
    st.header("3 · Server features: tools, resources, prompts")
    plain("Tools DO things, resources SHOW information, prompts GUIDE the model. And each has a different boss: the model, the application, or the user.")
    df_show(pe.SERVER_FEATURES)
    exam("Control mapping: **Tools → Model**, **Resources → Application**, **Prompts → User**. The model acts, the app supplies, the user chooses.")
    st.subheader("🧪 Lab: what will this server register?")
    st.markdown("Paste a FastMCP server file (or pick one). The app **reads** the code (with Python's `ast`) and never runs it.")
    pick = st.selectbox("Start from", list(pe.SERVER_SNIPPETS), key="p3_pick")
    src = st.text_area("server.py", pe.SERVER_SNIPPETS[pick], height=320, key="p3_src_" + pick)
    predict("for the exam's weather server: how many tools will an MCP client find?")
    r = pe.inspect_server_code(src)
    if not r["ok"]:
        st.error(r["error"])
        return
    m = st.columns(4)
    m[0].metric("Tools", r["counts"]["tool"])
    m[1].metric("Resources", r["counts"]["resource"])
    m[2].metric("Prompts", r["counts"]["prompt"])
    m[3].metric("Transport", r["transport"])
    if r["items"]:
        df_show(r["items"])
    else:
        st.error("An MCP client would receive an **empty list**: nothing is registered.")
    for level, text in r["findings"]:
        {"❌": st.error, "⚠️": st.warning, "ℹ️": st.info, "✅": st.success}[level](level + " " + text)
    exam("**No decorator, no tool.** In the exam's weather server, get_current_temp has no @mcp.tool and _log_to_database is a private helper, so the client finds an **empty list of tools**. "
         "The underscore is only a naming convention: FastMCP ignores it and raises no error.")


def page_client():
    st.header("4 · Client features: sampling, roots, elicitation")
    plain("Usually the client asks and the server answers. These three features let the SERVER ask the CLIENT for something.")
    df_show(pe.CLIENT_FEATURES)
    st.error("**Two versions of the same words.** The course's mapping quizzes use the restaurant analogy (Sampling = 'look at the menu', Roots = 'choose the menu', Elicitation = 'place your order'). "
             "The PDF's own table and the MCP specification use the definitions in the middle column. Answer the exam with the analogy; build servers with the specification.")
    t1, t2 = st.tabs(["Exam: the restaurant mapping", "Specification: who asks whom"])
    with t1:
        story("An AI assistant analyses financial data through an MCP server. Match each step.")
        steps = {"Sampling": "b. Pings the server to learn which tools it exposes (list_files, read_document, summarize_text)",
                 "Roots": "c. Confines all work to /corporate_filings/2024/Q4/",
                 "Elicitation": "a. Sends the final structured command to run generate_profit_loss_statement"}
        opts = list(steps.values())
        picks = {k: st.selectbox(k, opts, index=None, key="p4_m_" + k) for k in steps}
        if st.button("Check", key="p4_check1"):
            score = sum(picks[k] == v for k, v in steps.items())
            (st.success if score == 3 else st.error)("%d / 3. Course answer: 1 → b, 2 → c, 3 → a. Memory trick: **discover → scope → request**." % score)
            st.caption("Specification view: (b) is really tool discovery (tools/list), (a) is a tool call (tools/call), and only (c) matches the real meaning of roots.")
    with t2:
        dot('digraph { rankdir=LR; ' + NODE + ' C [label="Client\\n(host + user)"]; S [label="Server"]; '
            'C -> S [label="tools / resources / prompts\\n(server features)"]; S -> C [label="sampling / roots / elicitation\\n(client features)", color="#c92a2a", fontcolor="#c92a2a"]; }')
        picks = []
        for i, (text, ans) in enumerate(pe.CLIENT_SCENARIOS_SPEC):
            picks.append(st.radio(text, ["Sampling", "Roots", "Elicitation"], index=None, horizontal=True, key="p4_s_%d" % i))
        if st.button("Check", key="p4_check2"):
            score = 0
            for (text, ans), p in zip(pe.CLIENT_SCENARIOS_SPEC, picks):
                ok = p == ans
                score += ok
                (st.success if ok else st.error)("%s → **%s**" % (text, ans))
            st.metric("Score", "%d / %d" % (score, len(pe.CLIENT_SCENARIOS_SPEC)))
    reality("The session notes add that the MCP specification released on 28 July 2026 deprecates Sampling and Roots; Elicitation remains. "
            "The PDF also describes sampling as 'how the model selects tokens' in one place: that is a different idea (temperature/top-p), not the MCP feature.")


def page_transport():
    st.header("5 · Transports: STDIO vs HTTP")
    plain("MCP defines what messages look like; the transport is the road they travel on. MCP itself is transport-agnostic.")
    df_show(pe.TRANSPORTS)
    dot('digraph { rankdir=LR; ' + NODE + ' subgraph cluster_s { label="STDIO: same machine"; CD [label="Claude Desktop"]; P [label="python server.py\\n(child process)", fillcolor="#d3f9d8"]; CD -> P [label="stdin"]; P -> CD [label="stdout"]; } '
        'subgraph cluster_h { label="Streamable HTTP: over a network"; W [label="Web app"]; M [label="Mobile app"]; D [label="Desktop app"]; SRV [label="https://.../mcp", fillcolor="#d3f9d8"]; W -> SRV; M -> SRV; D -> SRV; } }')
    st.subheader("🧪 Which transport?")
    qs = [("Testing your first server on your laptop with Claude Desktop", "STDIO"), ("A team of 200 people using one shared company server", "Streamable HTTP"),
          ("A secure, air-gapped research machine with no network", "STDIO"), ("A server hosted in the cloud for web and mobile clients", "Streamable HTTP")]
    picks = [st.radio(q, ["STDIO", "Streamable HTTP"], index=None, horizontal=True, key="p5_%d" % i) for i, (q, a) in enumerate(qs)]
    if st.button("Check", key="p5_check"):
        for (q, a), p in zip(qs, picks):
            (st.success if p == a else st.error)("%s → **%s**" % (q, a))
    reality("The PDF also lists WebSockets and TCP. The specification defines two standard transports: **stdio** and **Streamable HTTP**.")


def page_build():
    st.header("6 · Build an MCP server with FastMCP")
    plain("FastMCP wraps the protocol in plain Python: create a server, decorate functions, call run(). It writes the JSON schemas from your type hints and the descriptions from your docstrings.")
    steps = st.tabs(["1 · Tools", "2 · Test client", "3 · Resources", "4 · Prompts", "5 · Run over stdio", "🔴 Watch a host use it"])
    with steps[0]:
        st.code('''from fastmcp import FastMCP

mcp = FastMCP("Simple Math Server")          # a server: a collection of tools, resources and prompts


@mcp.tool()                                  # @mcp.tool and @mcp.tool() both work
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together.

    Args:
        a: First number
        b: Second number
    """
    return a + b''', language="python")
        st.markdown("**What the client sees** (tools/list, from your type hints and docstring):")
        st.code(json.dumps(server().handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"][0], indent=2), language="json")
    with steps[1]:
        st.markdown("`servers/test_client.py` tests any server **without Claude Desktop**, exactly as Claude Desktop connects (stdio).")
        st.code('''import asyncio, sys
from pathlib import Path
from fastmcp import Client

async def test(server_file):
    async with Client(Path(server_file)) as client:      # starts the server and connects over stdio
        for tool in await client.list_tools():
            print(tool.name, "-", tool.description.splitlines()[0])
        result = await client.call_tool("add_numbers", {"a": 2, "b": 3})
        print("add_numbers(2, 3) ->", result.data)
        for resource in await client.list_resources():
            content = await client.read_resource(str(resource.uri))
            print(resource.uri, "->", content[0].text)

asyncio.run(test(sys.argv[1]))          # python test_client.py mcpserver_tools.py''', language="python")
        st.caption("FastMCP 4: use Client(Path('server.py')) (a plain string is deprecated) and tool.input_schema (not inputSchema).")
    with steps[2]:
        st.code('''@mcp.resource("resource://greeting")        # every resource has an address (a URI)
def get_greeting() -> str:
    """Provides a simple greeting message."""
    return "Hello from FastMCP Resources!"


@mcp.resource("data://config")              # a dict is sent as JSON
def get_config() -> dict:
    """Provides application configuration as JSON."""
    return {"theme": "dark", "version": "1.2.0", "features": ["tools", "resources"]}''', language="python")
        st.code(json.dumps(server().handle({"jsonrpc": "2.0", "id": 4, "method": "resources/read", "params": {"uri": "data://config"}}), indent=2), language="json")
    with steps[3]:
        st.code('''from fastmcp.prompts import Message

@mcp.prompt
def explain_topic(topic: str) -> str:                  # a string = one user message
    """Asks the model to explain a study topic."""
    return f"Explain the concept of '{topic}' in simple terms with a short example."


@mcp.prompt
def generate_quiz(topic: str, difficulty: str = "easy") -> list[Message]:   # a short conversation
    """Asks the model to create a few quiz questions for a topic."""
    return [
        Message(f"I'm studying {topic}. Can you quiz me?"),
        Message("Of course! What difficulty would you like?", role="assistant"),
        Message(f"{difficulty}. Please give 3 questions, each with 4 options."),
    ]''', language="python")
        st.caption("FastMCP 4 rejects a prompt that returns a single Message: return a str or a list. The old import `fastmcp.prompts.prompt` no longer exists.")
    with steps[4]:
        st.code('''if __name__ == "__main__":
    mcp.run()                            # default: stdio. The client starts this file and talks through stdin/stdout
    # mcp.run(transport="http", port=8000)   # remote access over Streamable HTTP instead''', language="python")
        st.warning("In a stdio server, **stdout is the message channel**. A stray `print()` corrupts the JSON-RPC stream. Log to stderr instead.")
        st.markdown("FastMCP 4 and CrewAI need different versions of the `mcp` library, so the course uses **two virtual environments**: Part 1 (FastMCP 4) and Part 2 (CrewAI).")
    with steps[5]:
        st.markdown("Ask the host something. Its LLM sees the tool list, decides whether to call a tool, and the client sends a real `tools/call` to the mini server.")
        b = get_backend()
        msg = st.text_input("You say to Claude Desktop", "Use the math server to multiply 12.5 by 8", key="p6_msg")
        st.caption("Try: 'add 7 and 35', 'divide 1 by 0', 'greet Priyanka', 'what is MCP?'")
        if st.button("Send", type="primary", key="p6_send"):
            r = pe.host_turn(msg, server(), b)
            for who, payload in r["steps"]:
                st.markdown("**%s**" % who)
                st.code(json.dumps(payload, indent=2, ensure_ascii=False), language="json")
            st.success("**Host's answer:** " + r["answer"])
            badge(b)
        notice("The server never decides anything. The host's LLM chooses the tool (that's why tools are 'model-controlled'); the server just runs it and returns a structured result.")


def page_desktop():
    st.header("7 · Connect it to Claude Desktop")
    plain("Claude Desktop is an MCP client. You tell it which program to start (your venv's Python) and which file to run (your server). It then talks to your server over stdio.")
    st.markdown("""
1. Install Claude Desktop from claude.ai/download and sign in.
2. With your virtual environment active, run `python servers/print_claude_config.py`: it prints the config with the right paths for your computer.
3. **Settings → Developer → Edit Config**, paste it, save.
4. **Quit Claude Desktop completely** (Cmd+Q on macOS, or from the system tray on Windows) and reopen it. Closing the window isn't enough.
5. Settings → Developer: each server should show as **running**. Then ask: *"Use the math server to multiply 12.5 by 8."*
""")
    st.subheader("🧪 Lab: check your claude_desktop_config.json")
    a, b = st.columns([3, 1])
    pick = a.selectbox("Start from", list(pe.CONFIG_EXAMPLES), key="p7_pick")
    plat = b.radio("Your computer", ["macOS / Linux", "Windows"], index=1 if "Windows" in pick else 0, key="p7_plat_" + pick)
    text = st.text_area("claude_desktop_config.json", pe.CONFIG_EXAMPLES[pick], height=220, key="p7_txt_" + pick)
    r = pe.check_claude_config(text, plat)
    for level, msg in r["findings"]:
        {"❌": st.error, "⚠️": st.warning, "ℹ️": st.info, "✅": st.success}[level](level + " " + msg)
    if r["servers"]:
        df_show(r["servers"])
    notice("Fix one ❌ at a time: a JSON syntax error hides every problem behind it.")
    exam("The 'Protocol Mismatch Paradox': the script correctly uses mcp.run() (stdio), but args add `--transport http`. The answer: the args force HTTP mode, a **transport mismatch** "
         "with a command-based (stdio) client. Splitting args into separate strings is normal, and Claude Desktop passes args correctly.")
    show_code("servers/print_claude_config.py", '''import json, os, sys

folder = os.path.dirname(os.path.abspath(__file__))
config = {"mcpServers": {
    "math-server": {"command": sys.executable, "args": [os.path.join(folder, "mcpserver_tools.py")]},
    "data-server": {"command": sys.executable, "args": [os.path.join(folder, "mcpserver_resources.py")]},
    "prompt-server": {"command": sys.executable, "args": [os.path.join(folder, "mcpserver_prompt.py")]},
}}
print(json.dumps(config, indent=2))     # json.dumps doubles Windows backslashes for you''')
    st.markdown("**If a server doesn't appear:** check both paths; check FastMCP is installed in THAT Python; restart Claude Desktop fully; read the MCP logs "
                "(macOS `~/Library/Logs/Claude/`, Windows `%APPDATA%\\Claude\\logs\\`).")
    reality("The PDF says Claude Desktop offers 'local processing'. The Claude model runs on Anthropic's servers; what runs locally is your MCP server.")


def page_crewai():
    st.header("8 · MCP with CrewAI")
    plain("In Sessions 6 and 7, tools were @tool functions inside one script. With MCP, the tools live in a server, and a CrewAI agent becomes just another MCP client: one line, mcps=[...].")
    st.code('''from crewai import Agent, Task, Crew, LLM
from crewai.mcp import MCPServerStdio

travel_server = MCPServerStdio(command=python_command, args=["travel_server.py"])   # same two things as Claude Desktop's config

travel_assistant = Agent(
    role="Travel assistant",
    goal="Help couples plan trips using real weather data.",
    backstory="You always check the forecast with your tools before advising.",
    llm=llm,
    mcps=[travel_server],        # all of this agent's tools come from the MCP server
    max_rpm=4, verbose=True,
)
weather_task = Task(description="A couple is visiting Manali this week. Check the forecast, then say what to pack.",
                    expected_output="Three sentences of packing advice.", agent=travel_assistant)
result = await Crew(agents=[travel_assistant], tasks=[weather_task], max_rpm=4).kickoff_async()''', language="python")
    df_show([
        {"Step": "1", "What CrewAI does": "Starts travel_server.py as a separate program (stdio, like Claude Desktop)"},
        {"Step": "2", "What CrewAI does": "Completes the MCP handshake and sends tools/list"},
        {"Step": "3", "What CrewAI does": "The agent reads tool descriptions and decides to call get_weather_forecast: a tools/call"},
        {"Step": "4", "What CrewAI does": "The server's answer comes back through MCP (a 'Tool Output' box in verbose mode)"},
    ])
    st.markdown("The same server works unchanged in Claude Desktop. **Write the tool once; use it from any MCP client.** "
                "Remote servers: `MCPServerHTTP(url=\"https://example.com/mcp\", headers={\"Authorization\": \"Bearer ...\"})`.")
    st.subheader("🧪 Lab: why do my tool names look cut off?")
    st.markdown("CrewAI names each MCP tool `<command>_<args>_<tool>`, cleans it to lowercase letters, digits and `_`, and limits it to **64 characters**. "
                "If it's longer, the end (your tool's real name!) is replaced with a hash. This is CrewAI 1.15.30's exact rule.")
    c = st.columns(3)
    cmd = c[0].text_input("command", "/Users/priyanka/Desktop/courses/agents/session8/.venv/bin/python3", key="p8_cmd")
    arg = c[1].text_input("args[0]", "/Users/priyanka/Desktop/courses/agents/session8/travel_server.py", key="p8_arg")
    tool = c[2].text_input("tool", "get_weather_forecast", key="p8_tool")
    r = pe.crewai_tool_name(cmd, [arg], tool)
    st.code(r["name"], language="text")
    (st.error if r["cut"] else st.success)("%d characters. %s" % (r["length"], "The tool's own name was cut and replaced by a hash." if r["cut"] else "Readable: the tool's name survives."))
    st.caption("Fix: use SHORT relative paths (the notebook uses os.path.relpath(sys.executable) and 'travel_server.py'). Even cut names still work, because the model picks tools by description.")
    st.warning("**Only connect agents to servers you trust.** A tool runs whatever its author wrote, and a tool's description is text the model reads, so a malicious server could try to steer your agent. Keep tokens in environment variables.")


def page_a2a():
    st.header("9 · A2A: agents talking to agents")
    plain("A2A (Agent2Agent) lets agents built by different teams, with different frameworks, on different servers, delegate tasks and share results through structured messages. "
          "Like microservices, but each one can reason.")
    df_show([
        {"Type": "H2A (human-to-agent)", "Who talks": "Human ↔ AI agent", "Purpose": "Questions, instructions", "Example": "You ask an assistant to summarise a document"},
        {"Type": "A2A (agent-to-agent)", "Who talks": "AI agent ↔ AI agent", "Purpose": "Delegate tasks, share data, collaborate", "Example": "A summariser requests facts from a research agent"},
    ])
    t = st.tabs(["Agent Cards & discovery", "Message flow", "Task states", "Patterns", "The real A2A spec"])
    with t[0]:
        st.markdown("An **Agent Card** is an agent's business card: who it is, what it can do, how to reach it, and its input/output schema.")
        intent = st.selectbox("I need an agent that can...", sorted({c for card in pe.AGENT_CARDS.values() for c in card["capabilities"]}), key="p9_intent")
        found = pe.discover(intent)
        st.success("Found by reading the cards: **%s**" % ", ".join(found))
        st.code(json.dumps(pe.AGENT_CARDS[found[0]], indent=2), language="json")
    with t[1]:
        st.markdown("**Request → Process → Response.** The sender builds the message from the receiver's card. The receiver checks it against its schema.")
        a, b, c = st.columns(3)
        receiver = a.selectbox("Receiver", list(pe.AGENT_CARDS), key="p9_recv")
        intent = b.selectbox("Intent", pe.AGENT_CARDS[receiver]["capabilities"] + ["book_flight"], key="p9_int_" + receiver)
        example = {k: ("Design Review" if k == "meeting_topic" else "Tomorrow 4 PM" if k == "preferred_time" else 2 if v == "integer" else "Delhi" if v == "string" else [])
                   for k, v in pe.AGENT_CARDS[receiver]["schemas"]["input"].items()}
        payload_txt = st.text_area("payload (try a typo or a wrong type)", json.dumps(example, indent=2), height=140, key="p9_pl_" + receiver)
        try:
            payload = json.loads(payload_txt)
            r = pe.send_a2a("planner_agent", receiver, intent, payload)
            st.markdown("**1 · Request**")
            st.code(json.dumps(r["message"], indent=2), language="json")
            for p in r["problems"]:
                st.error(p)
            st.markdown("**3 · Response**")
            st.code(json.dumps(r["response"], indent=2), language="json")
        except json.JSONDecodeError as exc:
            st.error("Not valid JSON: %s" % exc)
        notice("Strict schemas catch typos and wrong types BEFORE the receiver acts on them: that is how standardised schemas reduce 'hallucination drift'.")
    with t[2]:
        st.markdown("A request to another agent becomes a **task** that moves through states. Long tasks can be checked on, streamed, or reported back later.")
        df_show([{"State": k, "Meaning": v} for k, v in pe.TASK_STATES.items()])
        st.session_state.setdefault("p9_state", ["submitted"])
        hist = st.session_state["p9_state"]
        cur = hist[-1]
        st.markdown("Current state: **%s**  ·  history: %s" % (cur, " → ".join(hist)))
        nxt = pe.TRANSITIONS.get(cur, [])
        if cur in pe.TERMINAL:
            st.info("Terminal state: the task is finished. Start a new task to try again.")
        cols = st.columns(max(1, len(nxt)) + 1)
        for col, s in zip(cols, nxt):
            if col.button(s, key="p9_go_" + s):
                hist.append(s)
                st.rerun()
        if cols[-1].button("↺ new task", key="p9_reset"):
            st.session_state["p9_state"] = ["submitted"]
            st.rerun()
    with t[3]:
        df_show(pe.A2A_PATTERNS)
        dot('digraph { rankdir=LR; ' + NODE + ' subgraph cluster_h { label="Hierarchical"; M [label="Manager"]; W1 [label="Worker"]; W2 [label="Worker"]; M -> W1; M -> W2; } '
            'subgraph cluster_c { label="Collaborative"; A [label="Writer"]; B [label="Critic"]; C [label="Refiner"]; A -> B -> C -> A; } '
            'subgraph cluster_p { label="Pipeline"; E [label="Extract"]; Cl [label="Clean"]; S [label="Summarise"]; V [label="Visualise"]; E -> Cl -> S -> V; } }')
    with t[4]:
        st.markdown("The post-read describes A2A conceptually. The real open protocol (from the Part 2 notebook): introduced by Google in April 2025, donated to the Linux Foundation in June 2025, "
                    "v1.0 in March 2026, and part of the Agentic AI Foundation alongside MCP since August 2026.")
        df_show([
            {"Idea": "Agent Card", "In the spec": "JSON at /.well-known/agent-card.json: name, skills, interfaces, capabilities, auth (older tutorials: agent.json)"},
            {"Idea": "Task", "In the spec": "A unit of work with states: submitted, working, input-required, auth-required, completed, failed, canceled, rejected"},
            {"Idea": "Messages & parts", "In the spec": "Messages from 'user' or 'agent', made of parts: text, file, URL or JSON data"},
            {"Idea": "Artifacts", "In the spec": "The results of a task: a booking, a report, an image"},
            {"Idea": "Transport", "In the spec": "JSON-RPC 2.0 over HTTP (with SSE streaming), gRPC, HTTP/REST; push notifications for long tasks"},
            {"Idea": "Opaque agents", "In the spec": "An agent never reveals its prompts, memory or tools: only its card and its results"},
        ])
        st.code(json.dumps(pe.A2A_SPEC_CARD, indent=2), language="json")


def page_compare():
    st.header("10 · MCP vs A2A, challenges & tracing")
    plain("MCP is the inner layer (an agent reaching its tools). A2A is the outer layer (agents working with agents). Partners, not rivals.")
    df_show(pe.MCP_VS_A2A)
    story("Agent A (task manager) asks Agent B (data retriever) for the latest sales numbers: that's A2A. Agent B queries a database through MCP. Agent A never needs to know how.")
    dot('digraph { rankdir=TB; ' + NODE + ' A [label="Agent A\\n(task manager)"]; B [label="Agent B\\n(data retriever)"]; DB [label="Sales DB\\n(MCP server)", shape=cylinder, fillcolor="#d3f9d8"]; '
        'W [label="Web search\\n(MCP server)", fillcolor="#d3f9d8"]; A -> B [label="A2A: get sales numbers", color="#c92a2a", fontcolor="#c92a2a", dir=both]; B -> DB [label="MCP"]; B -> W [label="MCP"]; }')
    exam("'Why reject MCP for a manager-less drone swarm?' MCP's **client-server** model leans toward **centralized** coordination (a server answering clients), "
         "which doesn't fit **decentralized peer-to-peer** negotiation. Not the transport, not missing error handling.")
    st.subheader("Common A2A challenges")
    df_show(pe.A2A_CHALLENGES)
    exam("The 'Narrow Agent problem' is about agents becoming too BROAD. The fix is to keep them narrow: 'do one thing, and do it well'.")
    st.subheader("🧪 Lab: find the failed request")
    st.markdown("Three users' requests run at the same time. Each goes Coordinator → Flight agent → flights DB, Coordinator → Hotel agent → maps API, then back to the user. "
                "One hop times out for ONE user. Can you tell which request failed, and where?")
    a, b = st.columns(2)
    fail = a.selectbox("Where the timeout happens", ["flight_agent → flights_db", "coordinator → hotel_agent", "hotel_agent → maps_api"], key="p10_fail")
    ids = b.toggle("Put a correlation_id on every message", False, key="p10_ids")
    r = pe.trace_requests(3, fail, ids)
    rows = [{k: v for k, v in row.items() if k != "_user"} for row in r["log"]]
    if not ids:
        for row in rows:
            row.pop("correlation_id")
    df_show(rows)
    if ids:
        pick = st.selectbox("Filter the log by correlation_id", sorted({row["correlation_id"] for row in rows}), key="p10_pick")
        df_show([row for row in rows if row["correlation_id"] == pick])
        st.success("The failed request is **%s**: filter by it and its whole path is in front of you." % r["victim_id"])
    else:
        st.error("Without IDs the timeout is visible, but which user it hit, and what that user's request did next, is guesswork in an interleaved log.")
    notice("Like a parcel tracking number: one ID follows a request through every agent and tool.")


def page_review():
    st.header("11 · Review, quiz & cheat sheet")
    with st.form("quiz8"):
        answers = [st.radio("**%d. %s**" % (i + 1, q), opts, index=None, key="q8_%d" % i) for i, (q, opts, c, w) in enumerate(pe.QUIZ)]
        submitted = st.form_submit_button("Check my answers", type="primary")
    if submitted:
        score = 0
        for i, (q, opts, c, why) in enumerate(pe.QUIZ):
            ok = answers[i] == opts[c]
            score += ok
            (st.success if ok else st.error)("**Q%d:** %s. %s%s" % (i + 1, "Correct" if ok else "Not quite", "" if ok else "Answer: **%s**. " % opts[c], why))
        st.metric("Score", "%d / %d" % (score, len(pe.QUIZ)))
    st.subheader("⚡ Cheat sheet")
    df_show([
        {"Idea": "Protocol", "One line": "Shared rules for message format, delivery, errors, permissions"},
        {"Idea": "MCP", "One line": "Open standard (from Anthropic) connecting AI apps to tools, data and prompts: USB-C for AI"},
        {"Idea": "Host / client / server", "One line": "App with the LLM / one connection per server / your code offering features"},
        {"Idea": "Messages", "One line": "JSON-RPC 2.0: tools/list, tools/call, resources/read, prompts/get"},
        {"Idea": "Server features", "One line": "Tools (model), resources (application), prompts (user)"},
        {"Idea": "Client features (spec)", "One line": "Sampling (server asks client's LLM), roots (allowed folders), elicitation (server asks the user)"},
        {"Idea": "Client features (course analogy)", "One line": "Sampling = look at the menu, roots = choose the menu, elicitation = place the order"},
        {"Idea": "Transports", "One line": "STDIO (local, child process) and Streamable HTTP (remote, many clients)"},
        {"Idea": "FastMCP", "One line": "mcp = FastMCP('name'); @mcp.tool / @mcp.resource('uri') / @mcp.prompt; mcp.run()"},
        {"Idea": "No decorator", "One line": "No tool: the client never sees it"},
        {"Idea": "Claude Desktop", "One line": "Settings → Developer → Edit Config: command (venv Python) + args (server path); quit fully and reopen"},
        {"Idea": "CrewAI + MCP", "One line": "Agent(..., mcps=[MCPServerStdio(command=..., args=[...])]); MCPServerHTTP for remote"},
        {"Idea": "A2A", "One line": "Agents ↔ agents: Agent Cards, tasks with states, messages, artifacts; agents stay opaque"},
        {"Idea": "MCP vs A2A", "One line": "MCP inside agents (vertical), A2A between agents (horizontal)"},
        {"Idea": "A2A best practices", "One line": "Narrow cohesive agents, strict schemas, lean orchestration, correlation IDs"},
    ])
    st.subheader("📝 Notebook exercises")
    df_show([
        {"Level": "Easy", "Exercise": "Add a convert_inr_to_usd(amount) tool (fixed rate, say so in the docstring)", "Try it": "Page 3: write it in the checker; page 6 for the pattern"},
        {"Level": "Easy", "Exercise": "Re-list tools after adding one: does CrewAI pick it up with no other change?", "Try it": "Page 8"},
        {"Level": "Medium", "Exercise": "Give the travel assistant two servers: mcps=[travel_server, another_server]. How are tools kept apart?", "Try it": "Page 8: the server prefix in each name"},
        {"Level": "Medium", "Exercise": "Add travel_server.py to Claude Desktop: same server, two clients", "Try it": "Page 7"},
        {"Level": "Discussion", "Exercise": "Write an Agent Card for a destination agent with two skills. What stays private?", "Try it": "Page 9"},
    ])
    st.subheader("Conclusions")
    st.markdown("""
1. **Protocols are shared rules** for exchanging messages: consistent, reliable, interoperable and secure.
2. **MCP standardises how AI applications connect to tools and data.** A host contains the LLM; each client talks to one server; servers offer tools, resources and prompts.
3. **FastMCP makes servers easy:** decorate ordinary functions; run locally over stdio or remotely over HTTP.
4. **Write once, use anywhere:** the same server works in Claude Desktop and as tools for a CrewAI agent.
5. **A2A lets agents work with other agents** across teams, frameworks and companies. **MCP inside agents, A2A between agents.**
""")


def sidebar():
    st.sidebar.title("🔌 Protocols Lab")
    st.session_state.setdefault("nav", PAGES[0])
    st.sidebar.radio("Go to", PAGES, key="nav")
    st.sidebar.divider()
    st.sidebar.radio("Mode", ["Demo (no API key)", "Live (Gemini)"], key="mode")
    if st.session_state.get("mode", "").startswith("Live"):
        st.sidebar.text_input("Gemini API key", type="password", key="api_key", value=_default_key(),
                              help="Kept in this browser session only. Never commit keys: use .streamlit/secrets.toml or an env var.")
        with st.sidebar.expander("Model settings"):
            st.text_input("Chat model", "gemini-3.5-flash-lite", key="chat_model")
            st.checkbox("Free-tier rate limit (4 calls/min)", True, key="rate_limit")
    b = get_backend()
    if st.session_state.get("mode", "").startswith("Live"):
        if b.live:
            st.sidebar.success("Live: connected to Gemini (🔴 labs use it)")
        else:
            st.sidebar.warning(b.error or "Paste a key to go live. Demo mode meanwhile.")
    else:
        st.sidebar.info("Demo mode: scripted host, real MCP messages.")


def main():
    sidebar()
    page = st.session_state.get("nav", PAGES[0])
    {PAGES[0]: page_home, PAGES[1]: page_why, PAGES[2]: page_mcp, PAGES[3]: page_server, PAGES[4]: page_client, PAGES[5]: page_transport,
     PAGES[6]: page_build, PAGES[7]: page_desktop, PAGES[8]: page_crewai, PAGES[9]: page_a2a, PAGES[10]: page_compare, PAGES[11]: page_review}[page]()


main()
