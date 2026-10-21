"""
Agents Course Lab: ONE app that connects all six sessions.

    streamlit run app.py

Session 1  Foundations of Agents          (session1_foundations/app.py)
Session 2  Building Blocks of Agents      (session2_building_blocks/app.py)
Session 3  Memory in AI Agents            (session3_memory/app.py)
Session 4  Agent Architectures & Patterns (session4_architecture/app.py)
Session 5  Intro to Multi-Agent Systems   (session5_multi_agent/app.py)
Session 6  Building Multi-Agent Systems   (session6_building_mas/app.py)

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
        st.error("Could not find %s. Keep the six session folders next to this app.py." % entry)
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
     [("S1", 3, "Learning agents"), ("S2", 7, "Learning agent"), ("S3", 6, "Procedural memory"), ("S4", 9, "Reflection")]),
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
     [("S5", 8, "Manager / worker"), ("S5", 9, "Peer / swarm"), ("S5", 10, "Federated teams"), ("S5", 11, "Pipeline"), ("S6", 6, "Patterns vs architectures"), ("S6", 10, "CrewAI sequential"), ("S6", 11, "CrewAI hierarchical"), ("S6", 12, "Peer collaboration in CrewAI")]),
    ("Communication, coordination, collaboration", "Agents need a shared message format, a way to avoid duplicate or clashing work, and a way to combine partial results.",
     [("S5", 3, "The three C's"), ("S5", 10, "A shared report format")]),
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
]


def hub_home():
    st.title("🎓 Agents Course Lab")
    st.caption("One app for all six sessions of *Agents*. Pick a session on the left, or open one from here.")
    st.warning("**📖 The thread through all six sessions.** Prasad's concierge could talk but not act (Session 1). So he gave it parts and a job description (Session 2), "
               "a memory it could trust (Session 3), and an architecture that decides how much to think (Session 4). When the work outgrew one agent, he designed a team (Session 5) and then built it, by hand and with a framework (Session 6).")
    st.success("**One question to hold on to:** who decides the next step, and how much thinking does this moment need? Every lab in the course is a variation on it.")
    tabs = st.tabs(["Course map", "Concept map", "Glossary", "Course quiz", "How to run"])

    with tabs[0]:
        dot = ('digraph { rankdir=LR; ' + NODE + ' S1 [label="Session 1\\nFoundations\\nWhy it must act", fillcolor="#d0ebff"]; S2 [label="Session 2\\nBuilding blocks\\nWhat it is made of", fillcolor="#d3f9d8"]; '
               'S3 [label="Session 3\\nMemory\\nWhat it remembers", fillcolor="#fff3bf"]; S4 [label="Session 4\\nArchitectures\\nHow it thinks", fillcolor="#ffe3e3"]; S5 [label="Session 5\\nMulti-agent\\nWho works together", fillcolor="#e5dbff"]; S6 [label="Session 6\\nBuilding MAS\\nHow to build the team", fillcolor="#c5f6fa"]; S1 -> S2 -> S3 -> S4 -> S5 -> S6; }')
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
        st.write("Fifteen questions across the whole course.")
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
            ("In CrewAI, task2 (the writer) has no context=[task1]. What happens?", ["An error is raised", "It runs, but has no research and may invent facts", "It waits for task1"], 1, "S6: a silent failure; check what flows between tasks."),
            ("A CrewAI manager_agent is given tools to 'verify work', as in the PDF. What does current CrewAI do?", ["Uses the tools to check", "Fails on kickoff: 'Manager agent should not have tools'", "Gives the tools to the workers"], 1, "S6: checked on CrewAI 1.15. Managers only delegate; the workers hold the tools."),
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
