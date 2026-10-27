"""
Agents Course Lab: ONE app that connects all eight sessions, plus exam practice and a library of the course code.

    streamlit run app.py

Session 1  Foundations of Agents          (session1_foundations/app.py)
Session 2  Building Blocks of Agents      (session2_building_blocks/app.py)
Session 3  Memory in AI Agents            (session3_memory/app.py)
Session 4  Agent Architectures & Patterns (session4_architecture/app.py)
Session 5  Intro to Multi-Agent Systems   (session5_multi_agent/app.py)
Session 6  Building Multi-Agent Systems   (session6_building_mas/app.py)
Session 7  The Three C's & learning teams (session7_three_cs/app.py)
Session 8  Protocols: MCP & A2A           (session8_protocols/app.py)

This file is a thin hub. It does not copy any session code: it runs the chosen session's own app.py, and
gives every session its own saved state, so switching sessions never mixes their pages, widgets or data.
Each session folder still runs on its own too:  streamlit run session1_foundations/app.py
"""
import ast
import os
import runpy
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import exam_bank as eb  # noqa: E402

st.set_page_config(page_title="Agents Course Lab", page_icon="🎓", layout="wide")

HOME = "🏠 Course map"
SESSIONS = {
    "S1": {"label": "🤖 Session 1 · Foundations of Agents", "folder": "session1_foundations",
           "question": "Why can an LLM talk but not act, and what turns it into an agent?",
           "story": "The concierge that could talk but not act. Tools, the agent loop, and the four workflow patterns.",
           "labs": "Reactive LLM · agent loop · evolution of agents · tools & the loop · reliability maths · 4 workflow patterns · Manali planner"},
    "S2": {"label": "🧱 Session 2 · Building Blocks of Agents", "folder": "session2_building_blocks",
           "question": "What is an agent made of, what is its job, and is it safe?",
           "story": "From demo to product: DilSeTrips. Four pillars, PEAS, five agent types, function calling vs agents, injection safety.",
           "labs": "Perception · memory · planning & ReAct · tools · PEAS · five agent types · injection & step limit · assembled agent & grader"},
    "S3": {"label": "🧠 Session 3 · Memory in AI Agents", "folder": "session3_memory",
           "question": "How does an agent remember, and how does it forget safely?",
           "story": "Five support tickets, five memory problems. Short-term and long-term memory, forgetting, and when memory goes wrong.",
           "labs": "Vector DB vs graph · context overflow · memory lifecycle (Pinecone) · forgetting · semantic/episodic/procedural · risks"},
    "S4": {"label": "🏗️ Session 4 · Agent Architectures & Patterns", "folder": "session4_architecture",
           "question": "How much thinking does each moment need, and who decides?",
           "story": "The night the agent thought too hard. Deliberative, reactive, BDI, layered and CoALA architectures; planning, ReAct, tool use, reflection.",
           "labs": "Plan checking · reflexes · BDI commitment · orchestrator · CoALA scoring · planning vs ReAct · reflection · capstone"},
    "S5": {"label": "🤝 Session 5 · Intro to Multi-Agent Systems", "folder": "session5_multi_agent",
           "question": "When one agent is not enough, who decides and who talks to whom?",
           "story": "The honeymoon request one agent could only skim. Why teams, centralized vs decentralized vs hybrid, and four design patterns, with the cost of each.",
           "labs": "Generalist vs specialists · three C's · coordinator down · stale status board · hybrid rebooking · manager's missing requirement · swarm races · federated report format · pipeline error propagation · call costs"},
    "S6": {"label": "🛠️ Session 6 · Building MAS (LangChain & CrewAI)", "folder": "session6_building_mas",
           "question": "How do you build the team you drew, and what does a framework do for you?",
           "story": "Prasad takes LangChain apart, builds his support desk three ways (notebook 6A), then describes a team in CrewAI and reads the exact prompts it writes (notebook 6B).",
           "labs": "How LangChain works (message list, tools from docstrings, step limit, old vs new code) · ReAct agent with tool and memory · centralized desk with agents as tools · decentralized 'is this mine?' and six approaches · hybrid router, peer question and refund in code · scoreboard · LangChain vs CrewAI · how CrewAI works (recorded prompts) · crew builder · sequential context · hierarchical manager (no tools) · allow_delegation"},
    "S7": {"label": "🔺 Session 7 · The Three C's & Learning Teams", "folder": "session7_three_cs",
           "question": "Talking is not teamwork: what must agents share, who does what, and how do they help each other, and can a team learn that by itself?",
           "story": "The team that messaged constantly and still duplicated work. Communication patterns, the three core coordination problems, four mechanisms, the four patterns in CrewAI, and multi-agent reinforcement learning (post-read).",
           "labs": "Failure cascade · request-response / pub-sub / blackboard (overwritten code) · task, resource and conflict labs · mechanism sorting · CrewAI patterns · crew debugger with real CrewAI errors · MARL corridor and helping games · self-play critic (RLPF) · learning a workflow · reward design"},
    "S8": {"label": "🔌 Session 8 · Protocols: MCP & A2A", "folder": "session8_protocols",
           "question": "How do you write a tool once and use it from every AI app, and how do agents from different companies work together?",
           "story": "One weather tool, many apps; one planner, someone else's agent. MCP's host, client and server; tools, resources and prompts; FastMCP; Claude Desktop; MCP in CrewAI; then A2A (post-read and Part 2).",
           "labs": "Live JSON-RPC to a mini MCP server · what will this server register? · sampling/roots/elicitation (course vs spec) · stdio vs HTTP · a host deciding tool calls · claude_desktop_config.json checker · CrewAI MCP tool names · Agent Cards, schema-checked messages, task states · tracing with correlation IDs"},
}
LABEL_TO_KEY = {v["label"]: k for k, v in SESSIONS.items()}
OPTIONS = [HOME] + [v["label"] for v in SESSIONS.values()]

# Settings that follow you from session to session (everything else is saved per session).
SHARED_KEYS = ("mode", "chat_model", "rate_limit", "rpm", "embed_model")


def read_pages(folder: str) -> list:
    """Read the PAGES list from a session's app.py without running it."""
    try:
        tree = ast.parse((ROOT / folder / "app.py").read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "PAGES":
                return ast.literal_eval(node.value)
    except Exception:
        pass
    return []


for _k, _v in SESSIONS.items():
    _v["pages"] = read_pages(_v["folder"])


# ----------------------------------------------------------------------------------------
# Navigation + per-session state isolation
# ----------------------------------------------------------------------------------------
def open_page(session_key: str, page_index: int = 0):
    """Callback: open a session at a given page (works whether or not that session is the active one)."""
    pages = SESSIONS[session_key]["pages"]
    target = pages[page_index] if pages and 0 <= page_index < len(pages) else None
    if st.session_state.get("_hub_active") == session_key:
        if target:
            st.session_state["nav"] = target
    elif target:
        st.session_state.setdefault("_hub_stash", {}).setdefault(session_key, {})["nav"] = target
    st.session_state["hub_session"] = SESSIONS[session_key]["label"]


def switch_state(selected_key: str):
    """Save the leaving session's state, clear it, and restore the entering session's saved state."""
    active = st.session_state.get("_hub_active")
    if active == selected_key:
        return
    stash = st.session_state.setdefault("_hub_stash", {})
    shared = st.session_state.setdefault("_hub_shared", {})
    is_hub_key = lambda k: k.startswith("hub_") or k.startswith("_hub")
    for k in SHARED_KEYS:
        if k in st.session_state:
            shared[k] = st.session_state[k]
    if active is not None:
        skip = BUTTON_KEYS | set(SHARED_KEYS)
        is_btn = lambda k: k in skip or k.startswith("FormSubmitter:")
        stash[active] = {k: v for k, v in st.session_state.items() if not is_hub_key(k) and not is_btn(k)}
    for k in [k for k in st.session_state.keys() if not is_hub_key(k)]:
        try:
            del st.session_state[k]
        except Exception:
            pass
    for k, v in stash.get(selected_key, {}).items():
        try:
            st.session_state[k] = v
        except Exception:
            pass            # button-like widgets are read-only: skip them, they start unclicked
    for k, v in shared.items():
        if k != "hub_key":
            try:
                st.session_state[k] = v
            except Exception:
                pass
    if shared.get("hub_key") and not st.session_state.get("api_key"):
        try:
            st.session_state["api_key"] = shared["hub_key"]  # the key typed once on the course map
        except Exception:
            pass
    st.session_state["_hub_active"] = selected_key


BUTTON_KEYS: set = sys.modules.setdefault("_hub_button_keys", type(sys)("_hub_button_keys")).__dict__.setdefault("keys", set())      # persists across reruns; keys of button widgets: their state cannot be restored by assignment


def _track_buttons():
    """Wrap the button widgets (also when called on columns/containers) so the hub knows which
    keys must never be stashed or restored: Streamlit forbids assigning them via session_state."""
    from streamlit.delta_generator import DeltaGenerator
    for name in ("button", "download_button", "form_submit_button", "link_button"):
        fn = getattr(DeltaGenerator, name, None)
        if fn is None or getattr(fn, "_hub_wrapped", False):
            continue
        def make(f):
            def wrapped(self, *a, **k):
                if k.get("key"):
                    BUTTON_KEYS.add(k["key"])
                return f(self, *a, **k)
            wrapped._hub_wrapped = True
            return wrapped
        setattr(DeltaGenerator, name, make(fn))
        setattr(st, name, getattr(st._main, name))


_track_buttons()


def run_session(key: str):
    """Run the chosen session's own app.py inside this page."""
    folder = ROOT / SESSIONS[key]["folder"]
    entry = folder / "app.py"
    if not entry.exists():
        st.error("Could not find %s. Keep the session folders next to this app.py." % entry)
        return
    own = [str(ROOT / v["folder"]) for v in SESSIONS.values()]
    sys.path[:] = [p for p in sys.path if p not in own]
    sys.path.insert(0, str(folder))
    original = st.set_page_config
    st.set_page_config = lambda *a, **k: None          # the hub already set the page config
    try:
        runpy.run_path(str(entry), run_name="__main__")
    finally:
        st.set_page_config = original


def remember_shared():
    shared = st.session_state.setdefault("_hub_shared", {})
    for k in SHARED_KEYS:
        if k in st.session_state:
            shared[k] = st.session_state[k]
    if st.session_state.get("hub_api_key_home"):
        shared["hub_key"] = st.session_state["hub_api_key_home"]
    elif st.session_state.get("api_key"):
        shared["hub_key"] = st.session_state["api_key"]        # a key typed inside a session is shared too


# ----------------------------------------------------------------------------------------
# Hub content
# ----------------------------------------------------------------------------------------
NODE = 'node [shape=box, style="rounded,filled", fillcolor="#eef2ff", fontname="Helvetica"];'

CONCEPTS = [
    ("The agent loop (perceive → reason → act → feedback)", "The same loop appears in every session; only the question changes: who decides the next step?",
     [("S1", 2, "What an agent is"), ("S1", 4, "Tools & the loop"), ("S2", 5, "Tools & orchestrator"), ("S4", 8, "Planning vs ReAct"), ("S6", 2, "A ReAct agent in LangChain")]),
    ("Tools and tool calling", "A tool is a function with a clear name, type hints and docstring. The model only requests; your code runs it.",
     [("S1", 4, "Giving the concierge hands"), ("S2", 5, "Tools & orchestrator"), ("S4", 9, "Tool use & reflection"), ("S6", 2, "A search tool in a LangChain agent"), ("S6", 1, "How a function becomes a tool")]),
    ("Planning", "Break a goal into steps, then replan when reality disagrees.",
     [("S1", 3, "Deliberative agents"), ("S2", 4, "Planning & ReAct"), ("S4", 2, "Deliberative architecture"), ("S4", 8, "Planning vs ReAct")]),
    ("Memory", "An LLM remembers nothing: memory is what your code writes down and puts back into the prompt.",
     [("S1", 3, "Memory as a message list"), ("S2", 3, "Short- and long-term memory"), ("S3", 3, "Short-term memory lab"), ("S3", 4, "Long-term lifecycle"), ("S4", 6, "CoALA memory"), ("S6", 2, "Chat memory: a follow-up fails without it"), ("S6", 1, "Memory = the message list")]),
    ("Learning from feedback", "The weights never change; lessons are stored and put back into the prompt.",
     [("S1", 3, "Learning agents"), ("S2", 7, "Learning agent"), ("S3", 6, "Procedural memory"), ("S4", 9, "Reflection"), ("S7", 8, "Self-play critic & rewards")]),
    ("Types and architectures of agents", "Reflex, model-based, goal, utility, learning; then deliberative, reactive, BDI, layered, CoALA.",
     [("S1", 3, "Evolution of agents"), ("S2", 7, "Five types"), ("S4", 3, "Reactive"), ("S4", 4, "BDI"), ("S4", 7, "Choosing an architecture"), ("S5", 7, "Choosing a MAS architecture")]),
    ("Workflow or agent?", "If you can picture the steps, use a workflow; if not, an agent. Then choose how much thinking each moment needs.",
     [("S1", 5, "Why agents, why not always"), ("S1", 6, "Four workflow patterns"), ("S1", 7, "Workflow or agent?"), ("S4", 7, "Choosing an architecture"), ("S5", 2, "Single agent or a team?")]),
    ("Safety, privacy and control", "Injection, approval gates, step limits, wrong memories and user isolation.",
     [("S2", 9, "Safety & meta-control"), ("S3", 7, "When memory goes wrong"), ("S4", 10, "Capstone"), ("S5", 11, "Checkers stop error propagation"), ("S6", 6, "What each design survives")]),
    ("Evaluating an agent", "Check output against a source or a rubric; remember that a judge is another model.",
     [("S1", 8, "Grounding check"), ("S2", 10, "Grading against PEAS"), ("S3", 7, "Memory audit"), ("S5", 8, "Check the plan's coverage"), ("S6", 11, "A manager that checks expected_output")]),
    ("Perception: language into structure", "Structured output turns messy language into a dictionary your code can route on.",
     [("S1", 3, "Rule-based vs LLM understanding"), ("S2", 2, "Perception & Hinglish")]),
    ("One agent or a team?", "A team buys specialisation, parallelism and robustness, and pays in calls, latency and coordination. Use one agent when the problem is small, centralised and stable.",
     [("S5", 1, "Generalist vs specialists"), ("S5", 2, "Why a team, and when not"), ("S5", 12, "What a team costs")]),
    ("Who decides? MAS architectures", "Centralized (one controller), decentralized (peers decide locally) or hybrid (central plan, local execution).",
     [("S5", 4, "Centralized"), ("S5", 5, "Decentralized"), ("S5", 6, "Hybrid"), ("S5", 7, "Choosing"), ("S6", 3, "Centralized desk, built"), ("S6", 4, "Decentralized desk, built"), ("S6", 5, "Hybrid desk, built"), ("S6", 6, "The three compared")]),
    ("Who talks to whom? MAS design patterns", "Manager/worker, peer/swarm, federated teams and sequential pipeline: reusable interaction blueprints, each with its own failure mode.",
     [("S5", 8, "Manager / worker"), ("S5", 9, "Peer / swarm"), ("S5", 10, "Federated teams"), ("S5", 11, "Pipeline"), ("S6", 6, "Patterns vs architectures"), ("S6", 10, "CrewAI sequential"), ("S6", 11, "CrewAI hierarchical"), ("S7", 5, "All four in CrewAI"), ("S7", 6, "Crew debugger")]),
    ("Communication, coordination, collaboration", "Agents need a shared message format, a way to avoid duplicate or clashing work, and a way to combine partial results.",
     [("S5", 3, "The three C's"), ("S5", 10, "A shared report format"), ("S7", 1, "The pyramid and failure cascade"), ("S7", 3, "Coordination vs cooperation")]),
    ("Communication patterns", "Request-response (1:1, blocking), publish-subscribe (1:n, no feedback), blackboard (n:n shared memory, needs conflict management).",
     [("S7", 2, "Three patterns, simulated"), ("S5", 5, "A stale status board"), ("S8", 2, "MCP's request-response in JSON-RPC")]),
    ("Coordination problems and mechanisms", "Task allocation, resource allocation, conflict resolution; solved by centralized, distributed, rule-based or learning-based coordination.",
     [("S7", 3, "Hospital: the three problems"), ("S7", 4, "Four mechanisms"), ("S5", 7, "Choosing an architecture")]),
    ("Learning from each other (MARL)", "Agents improve teamwork through rewards; other learners make the world non-stationary; shared rewards encourage helping.",
     [("S7", 7, "Corridor and helping games"), ("S7", 8, "Peer feedback, self-play, reward design")]),
    ("Protocols: MCP", "A standard way to package tools, resources and prompts in servers so any AI app (Claude Desktop, CrewAI, an IDE) can use them.",
     [("S8", 2, "Host, client, server, JSON-RPC"), ("S8", 3, "Tools, resources, prompts"), ("S8", 6, "Build with FastMCP"), ("S8", 7, "Claude Desktop"), ("S8", 8, "MCP in CrewAI")]),
    ("Agent-to-agent (A2A)", "Agents discover each other through Agent Cards and exchange structured messages and tasks. MCP inside agents, A2A between agents.",
     [("S8", 9, "Agent Cards, messages, task states"), ("S8", 10, "MCP vs A2A, tracing"), ("S6", 4, "Peers asking each other")]),
    ("Frameworks for agent teams (LangChain vs CrewAI)", "LangChain supplies the parts for one agent; CrewAI adds roles, tasks, hand-offs and processes for a team. A framework takes over the coordination code, not the responsibility for checking answers.",
     [("S6", 7, "Why a framework?"), ("S6", 9, "Crew, Agent, Task, Process"), ("S6", 10, "Sequential and context"), ("S6", 11, "Hierarchical and the manager"), ("S6", 1, "How LangChain works"), ("S6", 8, "How CrewAI works (recorded prompts)"), ("S6", 12, "allow_delegation")]),
]

GLOSSARY = [
    ("Agent", "Observes, decides and acts toward a goal in a loop, using feedback", "S1"),
    ("Tool call", "The model's request to run a function; nothing runs until your code runs it", "S1"),
    ("Workflow", "Predefined steps in a fixed order; your code decides the path", "S1"),
    ("Evaluator-optimizer / parallelization / chaining / routing", "The four workflow patterns", "S1"),
    ("Four pillars", "Perception, memory, planning, tools", "S2"),
    ("PEAS", "Performance measure, Environment, Actuators, Sensors: the agent's job description", "S2"),
    ("ReAct", "Reason, act, observe, repeat", "S2"),
    ("Utility function", "A visible weighted score that ranks the ways to reach a goal", "S2"),
    ("Prompt injection", "Malicious text, often inside a tool result, that tries to override instructions", "S2"),
    ("Human-in-the-loop", "A person approves a sensitive tool call before it runs", "S2"),
    ("Context window", "What the model sees on one call; the limit of short-term memory", "S3"),
    ("Embedding / vector DB", "Meaning as numbers; similarity search over them", "S3"),
    ("Semantic / episodic / procedural memory", "Facts / experiences / how-to rules", "S3"),
    ("Soft vs hard delete", "Hide a memory vs truly remove it (and its vector)", "S3"),
    ("Deliberative / reactive / BDI / layered / CoALA", "Five agent architectures: blueprints for how thinking is organised", "S4"),
    ("Design pattern", "A proven technique inside an architecture: planning, ReAct, tool use, reflection", "S4"),
    ("Orchestrator", "Routes each message to the right amount of thinking", "S4"),
    ("Reflection", "A critic loop that checks and improves the agent's own output", "S4"),
    ("Multi-agent system (MAS)", "Several autonomous agents that collaborate or compete; system behaviour emerges from their interactions", "S5"),
    ("Three C's", "Communication, coordination, collaboration", "S5"),
    ("Centralized / decentralized / hybrid", "One controller decides / peers decide locally / central plan with local execution", "S5"),
    ("Manager / worker", "A manager delegates tasks to specialised workers and integrates the results", "S5"),
    ("Peer / swarm", "Equal agents following simple local rules, coordinating through a shared board", "S5"),
    ("Federated teams", "Specialist teams working independently, joined by a shared report format", "S5"),
    ("Sequential pipeline", "Agents in a fixed order; each stage's output is the next stage's input", "S5"),
    ("Emergent behaviour", "System-level behaviour no single agent was programmed to produce", "S5"),
    ("LangChain", "A toolkit of parts for LLM apps: model wrappers, tools, memory, chains. You write the coordination between agents", "S6"),
    ("CrewAI", "An open-source Python framework for teams of role-based agents: Crew, Agent, Task, Process", "S6"),
    ("Role / goal / backstory", "The three fields that define a CrewAI agent: what it is, what it aims for, and its persona", "S6"),
    ("Task context", "In CrewAI, context=[task1] hands task1's output to another task: the hand-off between agents", "S6"),
    ("Sequential / hierarchical process", "Tasks run in a fixed order / a manager delegates and can check the work", "S6"),
    ("manager_llm vs manager_agent", "A model acting as manager (lighter) vs a full agent with role, instructions and tools (can verify work)", "S6"),
    ("Hop limit", "The maximum number of times a query may be forwarded between peers before it stops: it prevents endless loops", "S6"),
    ("create_agent", "LangChain 1.x: wires a model, tools and a system prompt into the agent loop; replaces initialize_agent", "S6"),
    ("Agents as tools", "Wrap an agent in a function and give it to another agent as a tool: how a LangChain boss calls its experts", "S6"),
    ("recursion_limit", "LangChain's step limit: one step per model call and per round of tool runs; the run stops with GraphRecursionError", "S6"),
    ("Structured output", "llm.with_structured_output(schema): the model fills a JSON shape and your code gets a dictionary (routing, decisions)", "S6"),
    ("kickoff / kickoff_async", "Start a CrewAI crew or agent; in a notebook use result = await crew.kickoff_async(inputs={...})", "S6"),
    ("allow_delegation", "CrewAI: gives an agent the tools 'Delegate work to coworker' and 'Ask question to coworker'", "S6"),
    ("max_rpm", "CrewAI's rate limit: at most this many model calls per minute", "S6"),
    ("Request-response / publish-subscribe / blackboard", "Direct 1:1 and blocking / broadcast 1:n with no feedback / shared memory n:n", "S7"),
    ("Task / resource allocation, conflict resolution", "The three core coordination problems: who does what / who gets the shared thing / settling clashes", "S7"),
    ("Rule-based vs learning-based coordination", "Predefined if-then protocols vs strategies learned from rewards", "S7"),
    ("MARL", "Multi-agent reinforcement learning: agents learn to communicate, coordinate and cooperate from rewards", "S7"),
    ("Non-stationarity", "Other agents keep learning, so the environment each agent learns in keeps changing", "S7"),
    ("RLPF", "Reinforcement learning from peer feedback: other agents supply the reward signal", "S7"),
    ("Shared (team) reward", "Every agent gets the team's reward, which encourages helping each other", "S7"),
    ("MCP", "Model Context Protocol: an open standard connecting AI apps to tools, data and prompts", "S8"),
    ("Host / MCP client / MCP server", "The app with the LLM / one connection to one server / your code offering features", "S8"),
    ("Tools / resources / prompts", "Server features controlled by the model / the application / the user", "S8"),
    ("Sampling / roots / elicitation", "Spec: server asks the client's LLM / allowed folders / server asks the user. Course analogy: menu / which menu / order", "S8"),
    ("JSON-RPC 2.0", "MCP's message format: jsonrpc, id, method, params → result or error", "S8"),
    ("STDIO / Streamable HTTP", "Local child-process transport / remote network transport", "S8"),
    ("FastMCP", "A Python library that turns decorated functions into an MCP server", "S8"),
    ("mcps=[...]", "CrewAI: give an agent the tools of MCP servers (MCPServerStdio, MCPServerHTTP)", "S8"),
    ("A2A", "Agent2Agent protocol: agents discover each other and delegate tasks across teams and companies", "S8"),
    ("Agent Card", "An agent's public description: name, skills, endpoint, input/output schema", "S8"),
    ("Correlation ID", "One ID carried by every message of a request, so it can be traced across agents", "S8"),
]


# ----------------------------------------------------------------------------------------
# Exam practice and the code library
# ----------------------------------------------------------------------------------------
LIBRARY = [
    {"title": "Session 4 · Supervisor travel agent (LangChain)", "folder": "course_code/session4_supervisor_agent", "session": "S4",
     "what": "A supervisor agent that calls a weather agent and a travel-guide agent as tools (agents as tools), with chat history for follow-ups. "
             "Real data: Open-Meteo forecasts and Wikivoyage guides, both free and keyless. main1.py is the simpler single-agent version.",
     "run": "pip install -r requirements.txt\n# put GOOGLE_API_KEY=... in a .env file next to main.py\npython main.py          # supervisor + two specialists\npython main1.py         # one agent with both tools",
     "learn": ("S6", 3, "Centralized desk with agents as tools")},
    {"title": "Session 6 · DilSeTrips project", "folder": "course_code/session6_dilsetrips", "session": "S6",
     "what": "The course's travel-concierge project split into agents, tools, memory, prompts (YAML) and a coordinator, with a couples demo.",
     "run": "pip install -r requirements.txt\ncp .env.example .env    # add your key\npython main.py\npython demo_couples.py",
     "learn": ("S6", 3, "Centralized desk, built")},
    {"title": "Session 7 · Three C's demos (CrewAI)", "folder": "course_code/session7_scripts", "session": "S7",
     "what": "The five notebook demos as scripts: all three communication patterns, manager/worker, sequential pipeline, peer/swarm, federated teams. Checked against CrewAI 1.15.",
     "run": "python -m venv .venv && source .venv/bin/activate   # Windows: .venv\\Scripts\\activate\npip install -r requirements.txt\npython manager_worker.py          # each script runs one demo (max_rpm=4: a few minutes each)",
     "learn": ("S7", 5, "The four design patterns in CrewAI")},
    {"title": "Session 8 · MCP servers (FastMCP 4)", "folder": "course_code/session8_servers", "session": "S8",
     "what": "Three servers (tools, resources, prompts), a test client that needs no Claude Desktop, and a script that prints your Claude Desktop config. "
             "Use a SEPARATE venv: FastMCP 4 and CrewAI need different versions of the mcp library.",
     "run": "python -m venv .venv && source .venv/bin/activate\npip install -r requirements_part1.txt\npython test_client.py mcpserver_tools.py\npython print_claude_config.py      # paste into Claude Desktop > Settings > Developer > Edit Config",
     "learn": ("S8", 6, "Build an MCP server with FastMCP")},
]
NOTEBOOKS = [
    ("Session3_Memory.ipynb", "S3"), ("Session4_Architectures_and_Patterns.ipynb", "S4"), ("Session5_Intro_to_MAS.ipynb", "S5"),
    ("Session6A_MAS_with_LangChain.ipynb", "S6"), ("Session6B_MAS_with_CrewAI.ipynb", "S6"), ("Session7_Three_Cs_and_Patterns_in_CrewAI.ipynb", "S7"),
    ("Session7_Class_Notes.ipynb", "S7"), ("Session8_Part1_Building_an_MCP_Server.ipynb", "S8"), ("Session8_Part2_MCP_with_CrewAI_and_A2A.ipynb", "S8"),
]


def exam_practice():
    st.write("Scenario questions in the style of the weekly assignments. Answer, then check: you get the course's answer and reasoning, "
             "plus a **real-library note** wherever CrewAI or the MCP specification behaves differently from the course answer.")
    c1, c2 = st.columns(2)
    sess = c1.multiselect("Sessions", ["S7", "S8"], ["S7", "S8"], key="hub_ex_sess")
    lvl = c2.multiselect("Level", eb.LEVELS, eb.LEVELS, key="hub_ex_lvl")
    items = [q for q in eb.EXAM if q["session"] in sess and q["level"] in lvl]
    st.caption("%d question(s)" % len(items))
    score, answered = 0, 0
    for i, q in enumerate(items):
        with st.container(border=True):
            st.markdown("**%s** · %s · %s" % (q["title"], SESSIONS[q["session"]]["label"], q["level"]))
            st.write(q["scenario"])
            st.markdown("**" + q["question"] + "**")
            key = "hub_ex_%s" % q["title"]
            if q["multi"]:
                picked = [j for j, o in enumerate(q["options"]) if st.checkbox(o, key="%s_%d" % (key, j))]
            else:
                choice = st.radio("Pick one", q["options"], index=None, key=key, label_visibility="collapsed")
                picked = [q["options"].index(choice)] if choice is not None else []
            a, b = st.columns([1, 1])
            shown = a.toggle("Check my answer", key=key + "_show")
            b.button("Open the lab that teaches this", key=key + "_lab", on_click=open_page, args=q["page"])
            if shown:
                answered += 1
                right = sorted(picked) == sorted(q["answer"])
                score += right
                correct = "; ".join(q["options"][j] for j in q["answer"])
                (st.success if right else st.error)(("Correct. " if right else "Not quite. Course answer: **%s**. " % correct) + q["why"])
                if q["real"]:
                    st.caption("🧪 **Real library / spec:** " + q["real"])
    if answered:
        st.metric("Score so far", "%d / %d checked" % (score, answered))


def code_library():
    st.write("Every script, server and notebook from the course, in one place. Read the code here, download it, and see how to run it. "
             "Notebooks are included with their outputs cleared (they keep their diagrams).")
    kind = st.radio("Show", ["Projects and scripts", "Notebooks"], horizontal=True, key="hub_lib_kind")
    if kind.startswith("Projects"):
        names = [p["title"] for p in LIBRARY]
        pick = st.selectbox("Project", names, key="hub_lib_proj")
        proj = LIBRARY[names.index(pick)]
        folder = ROOT / proj["folder"]
        st.info(proj["what"])
        st.markdown("**How to run**")
        st.code(proj["run"], language="bash")
        st.button("Open the lab for this session: " + proj["learn"][2], key="hub_lib_open", on_click=open_page, args=proj["learn"][:2])
        files = sorted(str(f.relative_to(folder)) for f in folder.rglob("*") if f.is_file() and "__pycache__" not in f.parts)
        if not files:
            st.warning("The folder %s is missing. Keep course_code/ next to app.py." % proj["folder"])
            return
        fname = st.selectbox("File", files, key="hub_lib_file_" + pick)
        path = folder / fname
        text = path.read_text(encoding="utf-8", errors="replace")
        lang = {".py": "python", ".md": "markdown", ".yaml": "yaml", ".txt": "text", ".json": "json"}.get(path.suffix, "text")
        st.download_button("⬇️ Download " + fname, text, file_name=path.name, key="hub_lib_dl")
        st.code(text, language=lang)
    else:
        nbdir = ROOT / "course_code" / "notebooks"
        pick = st.selectbox("Notebook", ["%s · %s" % (s, n) for n, s in NOTEBOOKS], key="hub_lib_nb")
        name = pick.split(" · ", 1)[1]
        path = nbdir / name
        if not path.exists():
            st.warning("Missing: %s" % path)
            return
        import json as _json
        nb = _json.loads(path.read_text(encoding="utf-8"))
        st.download_button("⬇️ Download %s (open in Colab or VS Code)" % name, path.read_bytes(), file_name=name, key="hub_lib_nbdl")
        heads = [l for c in nb["cells"] if c["cell_type"] == "markdown" for l in "".join(c["source"]).splitlines() if l.startswith("#")]
        with st.expander("Outline (%d headings)" % len(heads), expanded=True):
            st.markdown("\n".join("- " + h.lstrip("#").strip() if h.startswith("##") else "**" + h.lstrip("#").strip() + "**" for h in heads[:80]))
        only_code = st.toggle("Show code cells only", True, key="hub_lib_onlycode")
        for c in nb["cells"]:
            src = "".join(c["source"]).strip()
            if not src:
                continue
            if c["cell_type"] == "code":
                st.code(src, language="python")
            elif not only_code:
                st.markdown(src.replace("attachment:", "#"))



def hub_home():
    st.title("🎓 Agents Course Lab")
    st.caption("One app for all eight sessions of *Agents*, with exam practice and every course script and notebook. Pick a session on the left, or open one from here.")
    st.warning("**📖 The thread through all eight sessions.** Prasad's concierge could talk but not act (Session 1). So he gave it parts and a job description (Session 2), "
               "a memory it could trust (Session 3), and an architecture that decides how much to think (Session 4). When the work outgrew one agent, he designed a team (Session 5) and then built it, by hand and with a framework (Session 6). "
               "He made the team truly work together and even learn (Session 7), and finally connected it to tools and other companies' agents through shared protocols (Session 8).")
    st.success("**One question to hold on to:** who decides the next step, and how much thinking does this moment need? Every lab in the course is a variation on it.")
    tabs = st.tabs(["Course map", "Concept map", "Glossary", "Course quiz", "📝 Exam practice", "📚 Code library", "How to run"])

    with tabs[0]:
        dot = ('digraph { rankdir=LR; ' + NODE + ' S1 [label="Session 1\\nFoundations\\nWhy it must act", fillcolor="#d0ebff"]; S2 [label="Session 2\\nBuilding blocks\\nWhat it is made of", fillcolor="#d3f9d8"]; '
               'S3 [label="Session 3\\nMemory\\nWhat it remembers", fillcolor="#fff3bf"]; S4 [label="Session 4\\nArchitectures\\nHow it thinks", fillcolor="#ffe3e3"]; S5 [label="Session 5\\nMulti-agent\\nWho works together", fillcolor="#e5dbff"]; S6 [label="Session 6\\nBuilding MAS\\nHow to build the team", fillcolor="#c5f6fa"]; '
               'S7 [label="Session 7\\nThree C\'s\\nHow the team works", fillcolor="#ffd8a8"]; S8 [label="Session 8\\nMCP & A2A\\nHow it connects", fillcolor="#e9ecef"]; S1 -> S2 -> S3 -> S4 -> S5 -> S6 -> S7 -> S8; }')
        st.graphviz_chart(dot)
        for key, s in SESSIONS.items():
            with st.container(border=True):
                c1, c2 = st.columns([5, 1.3])
                c1.markdown("### " + s["label"])
                c1.markdown("**Question:** " + s["question"])
                c1.markdown(s["story"])
                c1.caption("Labs: " + s["labs"])
                c2.button("Open session", key="open_" + key, on_click=open_page, args=(key, 0), type="primary")
                with c2.popover("Pages"):
                    for i, p in enumerate(s["pages"]):
                        st.button(p, key="pg_%s_%d" % (key, i), on_click=open_page, args=(key, i))

    with tabs[1]:
        st.write("The same ideas return in deeper form. Pick a concept and jump straight to where each session treats it.")
        names = [c[0] for c in CONCEPTS]
        pick = st.selectbox("Concept", names, key="hub_concept")
        _, desc, links = CONCEPTS[names.index(pick)]
        st.info(desc)
        cols = st.columns(max(1, len(links)))
        for i, (col, (sk, idx, label)) in enumerate(zip(cols, links)):
            col.button("%s · %s" % (sk, label), key="hub_link_%d" % i, on_click=open_page, args=(sk, idx))
        st.markdown("**The whole map**")
        rows = []
        for name, desc, links in CONCEPTS:
            row = {"Concept": name}
            for sk in SESSIONS:
                row[sk] = "; ".join(label for s2, _, label in links if s2 == sk) or "—"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True)

    with tabs[2]:
        q = st.text_input("Filter", "", key="hub_gloss_q", placeholder="type a word, e.g. memory")
        rows = [{"Term": t, "Meaning": m, "Session": s} for t, m, s in GLOSSARY if q.lower() in (t + m).lower()]
        st.dataframe(pd.DataFrame(rows), hide_index=True)

    with tabs[3]:
        st.write("Twenty questions across the whole course. For exam-style scenario questions, open the **Exam practice** tab.")
        items = [
            ("Who decides the next step in a workflow? In an agent?", ["Workflow: the model. Agent: your code", "Workflow: your code. Agent: the model", "Both: the model"], 1, "Workflow: your code fixes the order. Agent: the model picks the next action."),
            ("A model returns a tool call. What has happened?", ["The function ran", "The model requested it; your code decides whether to run it", "The function was trained into the model"], 1, "Nothing runs until your code runs it (S1)."),
            ("Which four pillars do LLM agents organise around?", ["Perception, memory, planning, tools", "Prompt, model, output, loop", "Search, plan, act, reflect"], 0, "S2."),
            ("The performance measures in PEAS become...", ["Tool functions", "An evaluation rubric", "The system prompt"], 1, "S2: the rubric a judge scores against."),
            ("Where does an LLM's 'memory' of a conversation live?", ["In its weights", "In what your code sends back each call", "In the GPU"], 1, "S1 and S3: the model remembers nothing."),
            ("Which memory kind records 'We went to Manali in December'?", ["Semantic", "Episodic", "Procedural"], 1, "S3: an event, not a fact."),
            ("The most reliable defence against a prompt-injected booking?", ["A stern system prompt", "A human approval gate on the sensitive tool", "A bigger context window"], 1, "S2: never rely on the prompt alone."),
            ("An emergency message needs an instant answer. Which layer should handle it?", ["Deliberative planning", "A reactive reflex", "A reflection loop"], 1, "S4: match thinking to the moment."),
            ("If each step is 95% reliable, a 10-step agent run is right about...", ["95%", "80%", "60%"], 2, "S1: 0.95^10 ≈ 0.60."),
            ("A step limit exists to...", ["Speed up the agent", "Stop runaway loops, then escalate or return a partial answer", "Shorten answers"], 1, "S1 and S2."),
            ("A centralized desk's coordinator crashes while all specialists are healthy. What happens?", ["They carry on as before", "No query is answered: a single point of failure", "Only the cheapest are answered"], 1, "S5: that is why hybrid and decentralized designs exist."),
            ("A manager's plan says 'leave insurance aside' and a keyword check passes it. What should you check instead?", ["The word count", "That every requirement has a named owner", "Nothing"], 1, "S5: a word in a task is not an owner."),
            ("A decentralized desk's General agent forwards any question it is unsure about. A customer asks something nobody owns. What stops the endless loop?", ["A faster model", "A hop limit and a final fallback", "Nothing needs to"], 1, "S6: no one has the whole picture, so bound the loop and end every chain."),
            ("In CrewAI, task2 (the writer) has context=[] (an empty list). What happens?", ["An error is raised", "It runs, but has no research and may invent facts", "It waits for task1"], 1, "S6: a silent failure. (Leaving context out entirely is different: CrewAI 1.15 then passes all earlier outputs.)"),
            ("A CrewAI manager_agent is given tools to 'verify work', as in the PDF. What does current CrewAI do?", ["Uses the tools to check", "Fails on kickoff: 'Manager agent should not have tools'", "Gives the tools to the workers"], 1, "S6: checked on CrewAI 1.15. Managers only delegate; the workers hold the tools."),
            ("Agents know what the others are doing, but work is duplicated and dependencies are missed. Which C failed?", ["Communication", "Coordination", "Cooperation"], 1, "S7: informed but disorganised."),
            ("Two agents overwrite each other's file on a shared blackboard. Which limitations show?", ["Blocking and no broadcast", "Conflict management and consistency", "No feedback and ordering"], 1, "S7: the blackboard is asynchronous and many-to-many."),
            ("With a shared team reward, learning agents tend to...", ["Free-ride", "Help each other, even at a small cost to themselves", "Stop exploring"], 1, "S7: if one succeeds, all succeed."),
            ("A FastMCP function has no @mcp.tool decorator. What does the client see?", ["The tool, named after the function", "Nothing: it isn't registered", "An error"], 1, "S8: no decorator, no tool."),
            ("MCP vs A2A in one line?", ["Same thing, two names", "MCP inside agents (to tools), A2A between agents", "A2A inside agents, MCP between agents"], 1, "S8."),
        ]
        picks = [st.radio("**%d. %s**" % (i + 1, q), o, index=None, key="hub_quiz_%d" % i) for i, (q, o, _, _) in enumerate(items)]
        if st.button("Check my answers", key="hub_quiz_check", type="primary"):
            score = 0
            for i, (q, o, a, why) in enumerate(items):
                if picks[i] is None:
                    st.warning("%d. Not answered." % (i + 1))
                elif picks[i] == o[a]:
                    score += 1
                    st.success("%d. Correct. %s" % (i + 1, why))
                else:
                    st.error("%d. Not quite. Correct: **%s**. %s" % (i + 1, o[a], why))
            st.metric("Score", "%d / %d" % (score, len(items)))

    with tabs[4]:
        exam_practice()

    with tabs[5]:
        code_library()

    with tabs[6]:
        st.markdown("""
**Run everything from one place**

```bash
pip install -r requirements.txt
streamlit run app.py
```

**Or run one session on its own**

```bash
streamlit run session1_foundations/app.py
```

**Modes.** *Demo* (default) needs no key: scripted models, toy embeddings and pretend data. *Live* sends the labs to Gemini: paste a key in the sidebar,
or set `GOOGLE_API_KEY` as an environment variable, or copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`. Never commit `secrets.toml`.

**State.** Each session keeps its own pages, widgets and data while you move between sessions. The *mode* and model settings follow you.
""")
        st.caption("Gemini API key (optional, shared): paste it once here and every session's Live mode picks it up.")
        st.text_input("Gemini API key", type="password", key="hub_api_key_home", help="Kept in this browser session only, never written to disk.")
        if st.session_state.get("_hub_shared", {}).get("hub_key"):
            st.caption("✅ A key is stored for this browser session.")


def hub_sidebar() -> str:
    st.sidebar.title("🎓 Agents Course Lab")
    st.session_state.setdefault("hub_session", HOME)
    st.sidebar.radio("Session", OPTIONS, key="hub_session")
    st.sidebar.divider()
    return st.session_state["hub_session"]


def main():
    chosen = hub_sidebar()
    target = "HOME" if chosen == HOME else LABEL_TO_KEY[chosen]
    switch_state(target)
    if target == "HOME":
        hub_home()
    else:
        run_session(target)
    remember_shared()


main()
