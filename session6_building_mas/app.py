"""
Building MAS Lab: an interactive companion to "Agents, Session 6: Building Multi-Agent Systems with LangChain and CrewAI"
(the session PDF, notebook 6A "Multi-Agent Systems with LangChain" and notebook 6B "Meet CrewAI").

Run:   streamlit run app.py
Works with NO API key and with neither LangChain nor CrewAI installed (Demo mode: scripted agents, pretend data).
Add a Gemini key for Live mode on the CrewAI-style labs (pages 10 and 11).
"""
import copy
import inspect
import os
from pathlib import Path

import pandas as pd
import streamlit as st

import crew_engine as ce
import lc_engine as le

st.set_page_config(page_title="Building MAS Lab", page_icon="🛠️", layout="wide")

PAGES = [
    "🏠 Start here",
    "1 · How LangChain works",
    "2 · One agent in LangChain: tool + memory",
    "3 · Centralized support desk",
    "4 · Decentralized support desk",
    "5 · Hybrid support desk",
    "6 · Three designs, one set of questions",
    "7 · Why a framework? LangChain vs CrewAI",
    "8 · How CrewAI works",
    "9 · CrewAI building blocks",
    "10 · Process: sequential",
    "11 · Process: hierarchical (a manager)",
    "12 · Agents asking each other",
    "13 · Review, quiz & cheat sheet",
]
ASSETS = Path(__file__).resolve().parent / "assets"

try:
    STRETCH = {"width": "stretch"} if inspect.signature(st.dataframe).parameters["width"].default == "stretch" else {"use_container_width": True}
except Exception:
    STRETCH = {"use_container_width": True}


# ----------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------
def _default_key() -> str:
    key = os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("GOOGLE_API_KEY", "")
        except Exception:
            key = ""
    return key


def get_backend() -> ce.Backend:
    mode = st.session_state.get("mode", "Demo (no API key)")
    key = st.session_state.get("api_key", "") if mode.startswith("Live") else ""
    cfg = (key, st.session_state.get("chat_model", "gemini-3.5-flash-lite"), st.session_state.get("rate_limit", True))
    if st.session_state.get("_cfg") != cfg or "backend" not in st.session_state:
        st.session_state.backend = ce.Backend(*cfg)
        st.session_state._cfg = cfg
    return st.session_state.backend


def goto(page: str):
    st.session_state.nav = page


def badge(b: ce.Backend):
    if b.last_source == "gemini":
        st.caption("🟢 Written by Gemini")
    elif "fallback" in b.last_source:
        st.caption("🟠 " + (b.error or "Live call failed; showing the demo text."))
    else:
        st.caption("🔵 Demo simulation (scripted agents, pretend data, no API key needed)")


def plain(text: str):
    st.info("**In plain words:** " + text)


def story(text: str):
    st.warning("**🎫 " + text + "**")


def predict(text: str):
    st.caption("🎲 **Before you run it:** " + text)


def notice(text: str):
    st.caption("🔎 **What to notice.** " + text)


def df_show(rows, **kw):
    st.dataframe(pd.DataFrame(rows), hide_index=True, **STRETCH, **kw)


def table_show(rows):
    """A table whose long cells wrap instead of being cut off (st.dataframe truncates them)."""
    df = pd.DataFrame(rows)
    st.table(df.set_index(df.columns[0]))


def show_code(title: str, code: str):
    with st.expander("🔍 " + title):
        st.code(code.strip("\n"), language="python")


def dot(src: str):
    st.graphviz_chart(src)


def picture(name: str, caption: str, width: int | None = None):
    """A diagram from the notebooks (kept in the assets folder next to this app)."""
    path = ASSETS / name
    if path.exists():
        if width:
            st.image(str(path), caption=caption, width=width)
        else:
            st.image(str(path), caption=caption)


try:
    _CODE_WRAP = "wrap_lines" in inspect.signature(st.code).parameters
except Exception:
    _CODE_WRAP = False


def code_block(text: str):
    """Long prompt text, wrapped when this Streamlit version supports it."""
    if _CODE_WRAP:
        st.code(text, language=None, wrap_lines=True)
    else:
        st.text(text)


def show_log(log: ce.Log, title: str = "Message log (who said what to whom)"):
    st.markdown("**" + title + "**")
    df_show(log.rows())


def metric_row(items):
    cols = st.columns(len(items))
    for c, (label, value) in zip(cols, items):
        c.metric(label, value)


def pick_query(key: str, default: int = 0, extra: list | None = None) -> str:
    options = ce.QUERIES + (extra or [])
    labels = options + ["✍️ Type my own"]
    choice = st.selectbox("Customer message", labels, index=default, key=key)
    if choice == labels[-1]:
        return st.text_input("Your message", "Does the Goa trip include insurance?", key=key + "_txt")
    return choice


def coverage_line(r: dict):
    needs = [ce.name_of(t) for t in r["needs"]] or ["(nothing the desk knows; needs a human)"]
    got = [ce.name_of(t) for t in r["covered"]] or ["none"]
    st.caption("**Topics in the question:** %s  ·  **Topics answered:** %s" % (", ".join(needs), ", ".join(got)))


def show_reply(r: dict):
    if r["complete"]:
        st.success("**Customer receives:** " + r["reply"])
    else:
        st.error("**Customer receives:** " + r["reply"])


NODE = 'node [shape=box, style="rounded,filled", fillcolor="#eef2ff", fontname="Helvetica"];'
ICON = {"Thought": "💭", "Action": "🔧", "Observation": "👀", "Final answer": "💬", "Memory": "🧠"}


def crew_cfg() -> dict:
    return copy.deepcopy(st.session_state.get("s6_cfg", ce.DEFAULT_CREW))


def crew_inputs() -> dict:
    return {"topic": st.session_state.get("s6_topic", "The future of electrical vehicles")}


def live_backend():
    b = get_backend()
    return b if b.live else None


# ----------------------------------------------------------------------------------------
# Page 0: Start
# ----------------------------------------------------------------------------------------
def page_home():
    st.title("🛠️ Building MAS Lab: from the design to a working team")
    st.caption("An interactive companion to *Agents, Session 6: Building Multi-Agent Systems* (the PDF, notebook 6A with LangChain, and notebook 6B with CrewAI). "
               "Everything in those three is explained here, so you can learn from the app alone.")
    story("Last session Prasad drew the team on a whiteboard: who decides, and who talks to whom. Now he has to build it. His support desk needs a Product agent, a Policy agent and a "
          "General agent. He builds it three ways, then wonders if a framework could do the plumbing for him.")
    st.success("**Two ideas to hold on to:** an *architecture* is how the system is structured (the blueprint); a *design pattern* is how the agents interact (how people use the building). "
               "And a framework such as CrewAI does not make agents smarter: it takes over the coordination code you would otherwise write by hand.")
    dot('digraph { rankdir=LR; ' + NODE + ' Customer [fillcolor="#fff3bf"]; Orchestrator [fillcolor="#ffe3e3"]; '
        'Customer -> Orchestrator; Orchestrator -> Product; Orchestrator -> Policy; Orchestrator -> General; Product -> Policy [style=dashed, dir=both, label="ask directly"]; }')
    st.subheader("Your path")
    st.markdown("**Part A · LangChain (notebook 6A):** the parts of one agent, then the same support desk built three ways by hand.  \n"
                "**Part B · CrewAI (notebook 6B):** what a framework does with a description of the team.")
    path = [
        ("1 · How LangChain works", PAGES[1], "The parts of LangChain, one agent's message list step by step, how a function becomes a tool, the step limit, old vs new code."),
        ("2 · One agent in LangChain", PAGES[2], "A ReAct agent with a search tool and a chat memory. Switch the memory off and watch a follow-up fail."),
        ("3 · Centralized support desk", PAGES[3], "An orchestrator routes each query. Notebook 6A: the experts are the boss's tools; a step limit stops it."),
        ("4 · Decentralized support desk", PAGES[4], "No orchestrator. Notebook 6A: each expert asks 'is this mine?'. The six ways to handle a multi-part message."),
        ("5 · Hybrid support desk", PAGES[5], "Route, then experts ask each other. Notebook 6A: a light router, a peer question, the refund done in code."),
        ("6 · Three designs compared", PAGES[6], "The same questions through every design, what each survives, and notebook 6A's scoreboard."),
        ("7 · Why a framework?", PAGES[7], "LangChain vs CrewAI as a checklist of what you wire by hand."),
        ("8 · How CrewAI works", PAGES[8], "The exact prompts real CrewAI builds from role, goal, backstory, task and context; kickoff, tools and settings."),
        ("9 · CrewAI building blocks", PAGES[9], "Crew, Agents, Tasks, Process. Edit role, goal and task text; the CrewAI code updates."),
        ("10 · Process: sequential", PAGES[10], "Research then write. Remove the context, reorder the tasks, mistype the input: see what breaks."),
        ("11 · Process: hierarchical", PAGES[11], "A manager decides. Why it cannot have tools, and what assigning tasks does to it."),
        ("12 · Agents asking each other", PAGES[12], "allow_delegation: a copywriter asks a fact checker. Session 5's patterns in CrewAI."),
        ("13 · Review", PAGES[13], "Quiz, homework from both notebooks, cheat sheet and the session's conclusions."),
    ]
    for title, page, desc in path:
        a, b, c = st.columns([3, 6, 1.4])
        a.markdown("**" + title + "**")
        b.write(desc)
        c.button("Open", key="home_" + page, on_click=goto, args=(page,))
    st.markdown("**Modes:** *Demo* (default) needs no key and installs neither LangChain nor CrewAI: the agents are scripted and the data is pretend. "
                "*Live* (sidebar) lets Gemini write the text on pages 10 and 11. Counting, routing and checking are always done by code.")
    st.caption("How the demos stay honest: the notebook's data, tools and refund arithmetic are copied exactly; LangChain's step counting and tool descriptions were "
               "measured on LangChain 1.4; the prompts on page 8 were recorded from the real CrewAI 1.15. The 'mini CrewAI' on pages 9 to 11 is a small teaching "
               "model, not the CrewAI library.")
    with st.expander("📍 Where each part of the PDF and the notebooks is covered"):
        table_show([
            {"Source": "PDF · Why LangChain for MAS (4 reasons)", "Page": "1 and 2"},
            {"Source": "PDF · Build a ReAct agent step by step (7 steps)", "Page": "2 (and 1 for the LangChain 1.x version)"},
            {"Source": "PDF · Centralized / decentralized / hybrid customer support", "Page": "3, 4, 5"},
            {"Source": "PDF · Relating patterns to architectures (summary table)", "Page": "6"},
            {"Source": "PDF · Limits of LangChain; why CrewAI; comparison table", "Page": "7"},
            {"Source": "PDF · CrewAI components; sequential and hierarchical; manager_llm vs manager_agent", "Page": "8 to 11"},
            {"Source": "PDF · Hands-on CrewAI (research + writer crew, kickoff)", "Page": "9 and 10"},
            {"Source": "PDF · Conclusions", "Page": "13"},
            {"Source": "Notebook 6A · setup, ask(), warm-up agent with tools and memory", "Page": "1"},
            {"Source": "Notebook 6A · centralized desk: agents as tools, step limit", "Page": "3 (Lab B)"},
            {"Source": "Notebook 6A · decentralized desk and the six approaches", "Page": "4 (Lab B)"},
            {"Source": "Notebook 6A · hybrid desk: router, peer question, calculate_refund", "Page": "5 (Lab B)"},
            {"Source": "Notebook 6A · the same message, three desks; patterns; homework", "Page": "6 (Lab B) and 13"},
            {"Source": "Notebook 6B · steps 1 to 3: one agent, a tool, tasks", "Page": "8 and 9"},
            {"Source": "Notebook 6B · step 4: pipeline with context and {topic}", "Page": "10"},
            {"Source": "Notebook 6B · step 5: a manager who decides", "Page": "11"},
            {"Source": "Notebook 6B · step 6: agents asking each other; patterns in CrewAI; memory", "Page": "12"},
            {"Source": "Notebook 6B · what CrewAI did and didn't; exercises; quick reference", "Page": "8 and 13"},
        ])


# ----------------------------------------------------------------------------------------
# Page 1: How LangChain works
# ----------------------------------------------------------------------------------------
def page_how_langchain():
    st.header("1 · How LangChain works")
    story("Before Prasad builds a team, he wants to know exactly what each part of LangChain does, so that nothing in the "
          "team code is magic. He takes one agent apart, message by message.")
    plain("**LangChain** is a Python toolkit for building apps on top of language models. It does not make the model smarter. "
          "It gives you ready-made parts: a wrapper to call the model, one standard **message** format, a way to turn your own Python "
          "functions into **tools** the model can request, and **create_agent**, which runs the loop *think → request a tool → "
          "read the result → answer*. In a team, who talks to whom is still **your own code** (pages 3 to 5). That is the gap CrewAI fills (page 8).")
    st.subheader("The parts, and the notebook line for each")
    table_show(le.PARTS)
    show_code("The notebook's setup cell (one model and one rate limiter, shared by every agent)", le.CODE["setup"])

    st.subheader("Lab 1 · Inside create_agent: one list of messages")
    st.markdown("An agent made with `create_agent` works on **one list of messages**. Each time the model is called it reads the **whole** list "
                "and adds one message. If that message asks for a tool, LangChain runs your Python function and adds the result. "
                "The loop stops when the model answers without asking for a tool. Step through the notebook's warm-up agent "
                "(two free tools: a weather forecast and the Wikivoyage travel guide).")
    dot('digraph { rankdir=LR; ' + NODE + ' you [label="Your question\\n(HumanMessage)", fillcolor="#fff3bf"]; '
        'model [label="Model call\\n(writes an AIMessage)", fillcolor="#ffe3e3"]; tools [label="LangChain runs\\nyour function\\n(ToolMessage)", fillcolor="#d3f9d8"]; '
        'answer [label="Answer\\n(AIMessage with\\nno tool_calls)", fillcolor="#d0ebff"]; '
        'you -> model; model -> tools [label="tool_calls"]; tools -> model [label="result"]; model -> answer [label="done"]; }')
    c1, c2 = st.columns([2, 3])
    mem = c1.radio("Between turn 1 and turn 2", ["Keep the conversation list (memory on)", "conversation = [ ] (memory off)"], key="s6_lc_mem")
    msgs = le.warmup_messages(mem.startswith("Keep"))
    n = c2.slider("Show the list up to message number", 1, len(msgs), len(msgs), key="s6_lc_n_" + mem[:4])
    shown = msgs[:n]
    metric_row([("Messages in the list", n), ("Model calls so far", sum(1 for m in shown if m[1] == "AIMessage")),
                ("Tool runs so far", sum(1 for m in shown if m[1] == "ToolMessage"))])
    df_show([{"#": i + 1, "Turn": t, "Message type": typ, "Content": content[:140], "What happened": note}
             for i, (t, typ, content, note) in enumerate(shown)])
    last = shown[-1]
    st.info("**Message %d (%s):** %s" % (n, last[1], last[3]))
    if not mem.startswith("Keep") and n == len(msgs):
        st.error("With the list emptied, turn 2 starts from nothing: 'there' has no meaning, so the agent cannot pick a tool. "
                 "The model itself never remembered turn 1: **memory is the list your code keeps and sends again.**")
    notice("Every AIMessage is one model call. The model never runs your function: it only writes a request (tool_calls). "
           "LangChain runs it and appends a ToolMessage. Count the calls: a question that needs one tool costs **two** model calls.")
    show_code("The notebook's warm-up agent (memory = a Python list)", le.CODE["warmup"])

    st.subheader("Lab 2 · How a Python function becomes a tool")
    st.markdown("The model never sees your code. It sees a short description that LangChain builds from the function's **name**, "
                "**type hints** and **docstring**. Pick one of the notebook's tools.")
    tool = st.selectbox("Tool", list(le.TOOLS), index=list(le.TOOLS).index("calculate_refund") if "calculate_refund" in le.TOOLS else 0, key="s6_lc_tool")
    c1, c2 = st.columns(2)
    c1.markdown("**What you write (notebook 6A)**")
    c1.code(le.TOOLS[tool]["code"], language="python")
    c2.markdown("**What the model receives** (generated by LangChain 1.4 from that function)")
    c2.json(le.TOOL_SCHEMAS[tool])
    notice("name = the function's name · description = the docstring's first line · each argument's type = its type hint · "
           "each argument's description = its line under *Args:*. That is **all** the model knows about your tool, so the docstring "
           "is an instruction to the model, not just a comment.")

    st.subheader("Lab 3 · The step limit (recursion_limit)")
    st.markdown("A runaway agent could call tools forever, so every run has a **step limit**. In LangChain 1.x it is "
                "`config={\"recursion_limit\": N}` and it counts **steps**: one per model call and one per round of tool runs.")
    c1, c2, c3 = st.columns(3)
    k = c1.slider("Tool calls the agent needs", 0, 5, 3, key="s6_lc_k")
    together = c2.radio("How the model asks for them", ["One tool per step", "All in one step"], key="s6_lc_together") == "All in one step"
    limit = c3.slider("recursion_limit", 2, 15, 4, key="s6_lc_limit")
    steps = 1 if k == 0 else (3 if together else 2 * k + 1)
    need = steps + 1
    metric_row([("Steps this run takes", steps), ("Smallest recursion_limit that works", need), ("Model calls", 1 if k == 0 else (2 if together else k + 1))])
    if limit >= need:
        st.success("The run finishes: %d steps fit inside recursion_limit=%d." % (steps, limit))
    else:
        st.error("GraphRecursionError: Recursion limit of %d reached without hitting a stop condition. "
                 "The work done so far is lost (but its model calls were still paid for)." % limit)
    df_show([{"Older code (session plan)": "max_iterations=2", "LangChain 1.x": "recursion_limit=4", "Why": "about two steps per iteration (model + tools)"},
             {"Older code (session plan)": "max_iterations=4", "LangChain 1.x": "recursion_limit=9", "Why": "4 iterations ≈ 8 steps, plus the final answer"}])
    st.caption("Measured on LangChain 1.4 with a stand-in model: three tool calls one per step = 7 steps, which fails at recursion_limit 7 and passes at 8; "
               "all three in one step = 3 steps, which passes at 4. Page 3 uses this on the notebook's three-part message.")

    st.subheader("Structured output: a dictionary instead of text")
    st.markdown("Routing and yes/no decisions need data your code can use, not prose. `llm.with_structured_output(schema)` makes the model "
                "fill in a JSON shape, and you get a plain Python **dictionary** back. It is one model call. The decentralized desk (page 4) uses it "
                "for each expert's *is this mine?* decision, and the hybrid desk (page 5) uses it to split a message.")
    show_code("The hybrid desk's router shape (notebook 6A)", le.CODE["router_schema"])

    st.subheader("Old LangChain code vs LangChain 1.x")
    st.markdown("The PDF and many tutorials use the older interface. The notebooks use LangChain 1.x. Here is what changed, checked on LangChain 1.4:")
    table_show(le.OLD_VS_NEW)

    st.subheader("From one agent to a team: what LangChain leaves to you")
    table_show([
        {"Desk": "Centralized (page 3)", "How notebook 6A builds it": "Agents as tools: each expert is wrapped in a function, and those functions are the boss's tools"},
        {"Desk": "Decentralized (page 4)", "How notebook 6A builds it": "A Python loop over the experts; each makes a structured yes/no decision"},
        {"Desk": "Hybrid (page 5)", "How notebook 6A builds it": "A structured-output router splits the message; an expert gets a tool that calls another expert directly"},
    ])
    st.success("**Takeaway.** LangChain = the parts for one agent: model, messages, tools from docstrings, the create_agent loop, structured output and a step limit. "
               "Every hand-off between agents is code you write. Pages 3 to 5 show that code; page 8 shows what CrewAI does instead.")
    st.button("Next: one agent with tools and memory →", on_click=goto, args=(PAGES[2],), type="primary")



# ----------------------------------------------------------------------------------------
# Page 2: one agent in LangChain
# ----------------------------------------------------------------------------------------
def page_react():
    st.header("2 · One agent in LangChain: tool + memory")
    story("Before building a team, build one good member. Prasad's support agent must answer 'what's the latest...' questions (it needs a search tool) and follow-ups like 'and the visa fee there?' (it needs memory).")
    plain("LangChain gives you the parts: a model wrapper, tools, memory, and a way to wire them into an agent. Each agent in a team is built from the same parts, so learn them on one agent first.")
    st.subheader("The seven steps in the PDF")
    table_show([{"Step": s, "What it does": w, "The line to look for": c} for s, w, c in ce.STEPS_OF_AN_AGENT])
    st.caption("These are the PDF's steps, written for the older LangChain. With LangChain 1.x (the notebooks), step 5 becomes 'keep the message list' and "
               "step 6 becomes create_agent: see the old-vs-new table on page 1. The ideas are the same.")
    st.caption("Why LangChain for multi-agent work? Native LLM integration, easy tool wrapping, plug-in memory, and a modular design where each agent is a chain you can swap.")

    st.subheader("Lab 1 · A three-message conversation, with and without memory")
    c1, c2, c3 = st.columns(3)
    mem = c1.radio("Chat memory", ["On (ConversationBufferMemory)", "Off"], key="s6_mem")
    maxr = c2.slider("Search results kept (max_results)", 1, 3, 2, key="s6_maxr", help="The notebook uses 2. More results = a longer prompt on the next call.")
    third = c3.text_input("Third message", ce.CHAT_SCRIPT[2], key="s6_third")
    predict("message 3 says 'there'. With memory off, what can the agent possibly search for?")
    turns = [ce.CHAT_SCRIPT[0], ce.CHAT_SCRIPT[1], third]
    res = ce.react_chat(turns, mem.startswith("On"), maxr)
    total = sum(r["calls"] for r in res)
    metric_row([("Model calls (3 messages)", total), ("Searches made", sum(r["searched"] for r in res)), ("Messages in memory", 2 * len(res) if mem.startswith("On") else 0)])
    for i, r in enumerate(res, 1):
        with st.expander("Message %d: %s" % (i, r["question"]), expanded=(i == 3)):
            for kind, text in r["steps"]:
                st.markdown("%s **%s:** %s" % (ICON.get(kind, "•"), kind, text))
            st.caption("Model calls this turn: %d" % r["calls"])
    if res[2]["clarify"]:
        st.error("With memory off, message 3 failed: the agent cannot see messages 1 and 2, so 'there' means nothing. Memory is not magic: it is **earlier messages put back into the prompt**.")
    else:
        st.success("With memory on, 'there' became 'Bali': the buffer put the earlier messages back into the prompt.")
    if mem.startswith("On"):
        st.markdown("**What the buffer holds after message 3** (this grows every turn, which is why memory costs tokens: see Session 3)")
        df_show([{"Role": role, "Message": txt[:110]} for q, a in [(r["question"], r["answer"]) for r in res] for role, txt in (("human", q), ("ai", a))])
    notice("Searching costs one extra model call (the answer is written after reading the result). Changing max_results does not change the number of calls, but it changes how much text each call must read.")
    st.caption("Everything above is scripted with pretend search results; there is no live web search in the demo. The model's temperature (0.3 in the notebook) cannot be shown here because a script does not vary: "
               "a lower temperature makes a real model's answers more focused and repeatable, a higher one more varied.")
    show_code("The same agent with LangChain 1.x (notebook 6A style)", """
from langchain.agents import create_agent

agent = create_agent(model=llm, tools=[search], system_prompt="You are a helpful travel assistant. Use search for current facts.")
conversation = []                                             # memory = the message list you keep
conversation.append({"role": "user", "content": user_input})
result = agent.invoke({"messages": conversation}, config={"recursion_limit": 9})
conversation = result["messages"]
print(result["messages"][-1].text)                            # the answer is the last message""")
    show_code("The PDF's agent (older LangChain), and what fails on LangChain 1.x", """
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key, temperature=0.3)
search = TavilySearchResults(tavily_api_key=tavily_api_key, max_results=2)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
agent = initialize_agent(tools=[search], llm=llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                         memory=memory, verbose=True)
response = agent.invoke(user_input)

# Checked on LangChain 1.4:
#  * 'from langchain.agents import initialize_agent' and 'from langchain.memory import ...' both fail: these older interfaces
#    moved to the separate langchain-classic package (pip install langchain-classic, then import from langchain_classic).
#  * The classic agent's invoke() returns a dictionary whose answer sits under 'output', so print(response['output']).
#    The PDF's print(response.content) raises an AttributeError.""")
    st.success("**Takeaway.** One LangChain agent = a model + tools + memory. A team is several of these, plus the coordination you have to write yourself.")
    st.button("Next: the centralized desk →", on_click=goto, args=(PAGES[3],), type="primary")


# ----------------------------------------------------------------------------------------
# Page 3: centralized
# ----------------------------------------------------------------------------------------
def page_central():
    st.header("3 · Centralized support desk")
    story("Prasad's first desk has a central **orchestrator** that receives every message and decides who answers: a Product agent, a Policy agent or a General agent. "
          "Then a customer asks about a package **and** its cancellation in one sentence.")
    plain("Pattern: **manager / worker**. Architecture: **centralized**. The orchestrator is the manager; the three agents are workers. All communication flows through the manager: "
          "customer asks, orchestrator analyses, routes to the right agent, the agent answers back through the orchestrator.")
    dot('digraph { rankdir=LR; ' + NODE + ' Customer [fillcolor="#fff3bf"]; Orchestrator [fillcolor="#ffe3e3"]; Customer -> Orchestrator; '
        'Orchestrator -> Product; Orchestrator -> Policy; Orchestrator -> General; Product -> Orchestrator; Policy -> Orchestrator; General -> Orchestrator; Orchestrator -> Customer; }')
    st.subheader("Lab A · Route a question (the PDF's desk)")
    q = pick_query("s6_cq", default=3, extra=[ce.ODD_QUERY])
    c1, c2 = st.columns(2)
    split = c1.checkbox("The orchestrator splits mixed questions across several agents", False, key="s6_csplit")
    down = c2.checkbox("The orchestrator is down", False, key="s6_cdown")
    predict("the question mixes a package (Goa) and a policy (cancellation). If the orchestrator picks ONE agent, which half is lost?")
    r = ce.centralized(q, split, ("orchestrator",) if down else ())
    metric_row([("Model calls", r["calls"]), ("Messages", r["messages"]), ("Fully answered?", "yes" if r["complete"] else "NO")])
    coverage_line(r)
    show_reply(r)
    show_log(r["log"])
    notice("Without splitting, the orchestrator sends the whole question to one worker, so a mixed question loses half. Splitting fixes that, and costs more calls (one to split, one per agent, one to combine). "
           "With the orchestrator down, nothing is answered even though all three workers are healthy: a single point of failure.")
    st.markdown("**Why teams like it:** clear control, easy debugging (one place to read), predictable. **Why they outgrow it:** every message waits for the manager, and the manager is a bottleneck and a single point of failure.")
    show_code("The shape of an orchestrator in LangChain (a sketch, not the notebook's exact code)", """
def orchestrator(query):
    route = router_llm.invoke("Which agent should answer: product, policy or general?\\n" + query).content.strip().lower()   # 1 model call
    agents = {"product": product_agent, "policy": policy_agent, "general": general_agent}
    return agents.get(route, general_agent).invoke(query)                                                                 # 1 more model call""")
    lab_6a_central()
    st.success("**Takeaway.** Centralized = control and simplicity. The price: one bottleneck, one point of failure, a model call at every hand-off, "
               "and experts who cannot fill each other's gaps.")
    st.button("Next: decentralized →", on_click=goto, args=(PAGES[4],), type="primary")


# ----------------------------------------------------------------------------------------
# Page 4: decentralized
# ----------------------------------------------------------------------------------------
def page_decentral():
    st.header("4 · Decentralized support desk")
    story("Version two has no orchestrator. A customer can message any agent. Each agent asks: 'Can I answer this? If not, who can?' and forwards the rest to a peer.")
    plain("Pattern: **peer / swarm**. Architecture: **decentralized**. No central controller; agents collaborate through local rules written in their prompts. Flexible and resilient, but harder to predict and debug.")
    dot('digraph { rankdir=LR; ' + NODE + ' Customer [fillcolor="#fff3bf"]; Customer -> General; Product -> Policy [dir=both]; Policy -> General [dir=both]; General -> Product [dir=both]; }')
    st.subheader("Lab A · Where does the customer land, and does the question reach an owner? (the PDF's desk)")
    q = pick_query("s6_dq", default=3, extra=[ce.ODD_QUERY])
    c1, c2, c3 = st.columns(3)
    entry = c1.selectbox("Customer messages this agent first", ce.DOMAINS, index=2, key="s6_dentry")
    gf = c2.checkbox("General agent forwards when unsure", False, key="s6_dgf", help="The notebook's General agent 'guides the customer when unsure'. Here it can also pass the question on.")
    hops = c3.slider("Hop limit (maximum forwards)", 1, 8, 4, key="s6_dhops")
    predict("pick the Himalayan-treks question, tick 'General agent forwards when unsure'. Nobody owns this topic. What happens?")
    r = ce.decentralized(q, entry, hops, gf)
    metric_row([("Model calls", r["calls"]), ("Forwards (hops)", r["hops"]), ("Fully answered?", "yes" if r["complete"] else "NO"), ("Wasted calls", r["wasted"])])
    coverage_line(r)
    show_reply(r)
    show_log(r["log"])
    notice("For a question some peer owns, the query finds its owner in one or two forwards. For a question **nobody** owns, forwarding can circle forever: product to policy to general to product. "
           "A hop limit turns an endless loop into a bounded one, but those calls are still spent and the customer still has no answer. Give every agent a final fallback instead of a polite 'try someone else'.")
    st.markdown("**Why teams like it:** no bottleneck, no single point of failure, easy to add another agent. **The catch:** nobody has the whole picture, so there is more chatter and loops are possible.")
    show_code("The local rule each peer follows (in its prompt)", """
PEER_RULE = '''You are the {name}. If the customer's question is something you can answer, answer it.
If not, reply with FORWARD:<agent> naming the peer best able to help. Never forward the same question twice.'''""")
    lab_6a_decentral()
    st.success("**Takeaway.** Decentralized = resilience and flexibility. The price: more messages, loops, parts nobody checks, and answers nobody can fully explain. "
               "Always set a hop limit and a final fallback.")
    st.button("Next: hybrid →", on_click=goto, args=(PAGES[5],), type="primary")


# ----------------------------------------------------------------------------------------
# Page 5: hybrid
# ----------------------------------------------------------------------------------------
def page_hybrid():
    st.header("5 · Hybrid support desk")
    story("Version three keeps one orchestrator for routing, but lets the agents talk to each other: the Product agent can ask the Policy agent about cancellation without going back to the boss.")
    plain("Pattern: **federated teams / specialists** (also fits a **pipeline**, for example Research → Write → Edit, which needs orchestration to enforce the order). Architecture: **hybrid**. "
          "Coordination comes from the orchestrator; flexibility from peers who can collaborate directly.")
    dot('digraph { rankdir=LR; ' + NODE + ' Customer [fillcolor="#fff3bf"]; Orchestrator [fillcolor="#ffe3e3"]; Customer -> Orchestrator; Orchestrator -> Product; Product -> Policy [dir=both, label="direct"]; Product -> General [dir=both, label="direct"]; Policy -> General [dir=both]; }')
    st.subheader("Lab A · Route once, then ask peers (the PDF's desk)")
    q = pick_query("s6_hq", default=3, extra=[ce.ODD_QUERY])
    way = st.radio("When an agent needs a peer's answer", ["Ask the peer directly", "Go back through the orchestrator"], key="s6_hway", horizontal=True)
    predict("the Product agent needs the cancellation policy. Does asking the Policy agent directly save model calls?")
    r = ce.hybrid(q, direct=way.startswith("Ask"))
    metric_row([("Model calls", r["calls"]), ("Messages", r["messages"]), ("Fully answered?", "yes" if r["complete"] else "NO")])
    coverage_line(r)
    show_reply(r)
    show_log(r["log"])
    other = ce.hybrid(q, direct=not way.startswith("Ask"))
    df_show([{"Peer questions": "direct", "Model calls": (r if way.startswith("Ask") else other)["calls"]},
             {"Peer questions": "via the orchestrator", "Model calls": (other if way.startswith("Ask") else r)["calls"]}])
    notice("Direct messages skip the orchestrator's extra routing call, but then the orchestrator no longer sees every message. Hybrid always trades **oversight** for **efficiency**; log the direct messages too.")
    st.markdown("**What hybrid promises:** coordination through the orchestrator, flexibility because agents collaborate without waiting for routing, and resilience because both styles coexist. "
                "Page 6 tests that last claim: it only holds if you build a fallback.")
    show_code("A specialist that can ask a peer", """
def product_agent(query):
    answer = product_llm.invoke(query)                         # the part it owns
    if "cancel" in query.lower():
        answer += " " + policy_agent(query)                    # a direct call to a peer: no orchestrator in between
    return answer""")
    lab_6a_hybrid()
    st.success("**Takeaway.** Hybrid = central plan and answer, direct messages in between, and arithmetic in code. The cleverness is in deciding **which** messages may skip the boss.")
    st.button("Next: compare the three →", on_click=goto, args=(PAGES[6],), type="primary")


# ----------------------------------------------------------------------------------------
# Page 6: compare
# ----------------------------------------------------------------------------------------
def page_compare():
    st.header("6 · Three designs, one set of questions")
    plain("The fair way to compare architectures is to give them the **same work** and count. Five customer questions go through every design below: three simple, two that mix a package with a policy.")
    c1, c2, c3, c4 = st.columns(4)
    down = c1.selectbox("Which agent is down?", ["nobody", "orchestrator", "product", "policy", "general"], key="s6_down")
    entry = c2.selectbox("Decentralized: customers start at", ce.DOMAINS, index=2, key="s6_centry")
    direct = c3.checkbox("Hybrid: peers ask directly", True, key="s6_cdirect")
    fallback = c4.checkbox("Hybrid: peers take over if the orchestrator is down", True, key="s6_cfallback")
    dn = () if down == "nobody" else (down,)
    predict("with the orchestrator down, which designs still answer anything? Then try 'product' down: does any design cope?")
    rows = ce.compare_designs(ce.QUERIES, entry, dn, direct, fallback)
    df_show([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    st.markdown("**Question by question** (✅ fully answered, ❌ not; the number is model calls)")
    grid = []
    for i, q in enumerate(ce.QUERIES):
        row = {"Question": q[:55]}
        for r in rows:
            x = r["_results"][i]
            row[r["_short"]] = ("✅ " if x["complete"] else "❌ ") + str(x["calls"])
        grid.append(row)
    df_show(grid)
    notice("Normal day: the plain centralized design loses the two mixed questions; splitting, forwarding and asking peers all answer everything, at different call costs. "
           "Orchestrator down: both centralized designs answer nothing; decentralized is unaffected; hybrid survives **only because a peer-to-peer fallback was built** (untick the box). "
           "A specialist down: **no** design helps, because only that one agent knows its topic. Decentralization spreads control, not knowledge: robustness needs a second owner for important topics.")
    st.subheader("What each design survives (every failure, side by side)")
    mat = []
    for d in ("nobody", "orchestrator", "product", "policy", "general"):
        rr = ce.compare_designs(ce.QUERIES, entry, () if d == "nobody" else (d,), direct, fallback)
        mat.append({"Down": d, **{x["_short"]: x["Fully answered"] for x in rr}})
    df_show(mat)
    st.subheader("Relating patterns to architectures")
    st.markdown("Design patterns define **how agents interact**; architectures define **how the system is structured**: like how people use a building versus its blueprint. "
                "Every desk above has both: the pattern says who talks to whom, the architecture says who decides.")
    table_show(ce.PATTERN_TABLE)
    lab_6a_scoreboard()
    st.success("**Takeaway.** No architecture wins everywhere. Centralized is simple and fragile at the centre; decentralized is robust at the centre and chatty; hybrid takes the best of both only if you build the fallback.")
    st.button("Next: why a framework? →", on_click=goto, args=(PAGES[7],), type="primary")


# ----------------------------------------------------------------------------------------
# Page 7: framework
# ----------------------------------------------------------------------------------------
def page_framework():
    st.header("7 · Why a framework? LangChain vs CrewAI")
    story("Prasad built three desks by hand. Each time he wrote the routing, the hand-offs and the message log himself. His next project has six agents. He starts to wonder whether somebody already wrote the plumbing.")
    st.markdown("**What LangChain by hand costs you:** manual orchestration of the agents, complicated agent-to-agent communication, and extra code for each custom tool or API. "
                "**What CrewAI is:** an open-source, Python-based framework for orchestrating teams of AI agents. Think of a virtual workplace: the **crew** is the organisation, each **agent** a team member with a role, "
                "the **tasks** are the jobs, and the **process** says how the work runs (step by step, or under a manager).")
    st.subheader("Lab · What does your project need?")
    needs = [r["Need"] for r in ce.FRAMEWORK_TABLE]
    chosen = st.multiselect("Tick everything your project needs", needs, default=needs[:4], key="s6_needs")
    predict("choose only 'One agent with a tool and a chat memory'. Is a framework worth it?")
    rows = [r for r in ce.FRAMEWORK_TABLE if r["Need"] in chosen]
    if rows:
        table_show([{k: v for k, v in r.items() if k != "by_hand"} for r in rows])
    hand = sum(r["by_hand"] for r in rows)
    metric_row([("Needs chosen", len(rows)), ("Things you would wire by hand in LangChain", hand)])
    if rows and hand == 0:
        st.success("LangChain alone is enough here, and a framework would be more machinery than you need.")
    elif hand >= 3:
        st.info("%d of your needs are things CrewAI provides as built-ins: this is where a framework starts to pay for itself." % hand)
    notice("The count comes from the notebook's comparison table: it shows where the manual work is, it is not a measurement of code size. For one agent, LangChain is the right tool; the gap opens as the number of agents and hand-offs grows.")
    st.subheader("A fair view of the trade-off (our addition to the notebook)")
    c1, c2 = st.columns(2)
    c1.markdown("**A framework gives you**\n- roles, tasks and hand-offs as ready-made objects\n- built-in memory and tool attachment\n- fast prototyping of a team\n- a common vocabulary for a team")
    c2.markdown("**A framework costs you**\n- some control: the loop is hidden inside the library\n- harder debugging when something goes wrong inside\n- another dependency whose interfaces change between versions\n- more model calls you did not write (see pages 11 and 12)")
    st.markdown("**The same comparison from notebook 6B, after building both:**")
    table_show(ce.CREWAI_VS_LANGCHAIN_6B)
    st.success("**Takeaway.** Use LangChain's parts for one agent. Reach for a team framework when the coordination code starts to be bigger than the agents themselves.")
    st.button("Next: how CrewAI works →", on_click=goto, args=(PAGES[8],), type="primary")


# ----------------------------------------------------------------------------------------
# Notebook 6A labs used on pages 3, 4, 5 and 6
# ----------------------------------------------------------------------------------------
def notebook_message(key: str) -> str:
    return st.selectbox("Customer message (notebook 6A)", le.MESSAGES, index=0, key=key,
                        help="The first one is the notebook's three-part message: a greeting, a price, and a refund.")


def show_6a_result(r: dict, msg: str):
    printed = r["run"].printed()
    st.markdown("**What the notebook prints**")
    st.code(printed or "(nothing printed)", language=None)
    if r["run"].error:
        st.error(r["run"].error)
        return
    sc = r["score"]
    if msg == le.MULTI_PART:
        line = "Greeting %s · Snow Valley price Rs 3,200 %s · Refund Rs 3,600 %s" % (
            "✅" if sc["greeting"] else "❌", "✅" if sc["price"] else "❌", "✅" if sc["refund"] else "❌")
        (st.success if all(sc.values()) else st.error)("**Customer receives:** " + r["answer"] + "\n\n" + line)
    else:
        st.success("**Customer receives:** " + r["answer"])
    with st.expander("Message log (who said what to whom)"):
        df_show([{"#": i + 1, **e} for i, e in enumerate(r["run"].events)])


def lab_6a_central():
    st.subheader("Lab B · Notebook 6A: the boss's tools are the experts")
    picture("a_central.png", "The centralized desk in notebook 6A: the orchestrator's only tools are the three experts.", 520)
    st.markdown("In notebook 6A the boss is itself a `create_agent` agent. Its tools are three small functions, `ask_stays_expert`, "
                "`ask_policy_expert` and `ask_general_expert`, and each one runs a whole expert agent. This trick is called **agents as tools**. "
                "The routing rules go in the boss's `system_prompt`.")
    msg = notebook_message("s6_c6a_msg")
    c1, c2, c3 = st.columns(3)
    limit = c1.slider("Step limit (recursion_limit)", 2, 15, 9, key="s6_c6a_limit")
    together = c2.checkbox("Gemini asks all the experts in one step", False, key="s6_c6a_tog",
                           help="Gemini sometimes requests several tools in one model call. Then the run needs far fewer steps.")
    boss_sum = c3.checkbox("The boss notices the missing price and does the sum itself", False, key="s6_c6a_sum")
    predict("set the step limit to 4 with the three-part message. Will the boss manage all three parts? Then tick 'all in one step'.")
    r = le.centralized(msg, limit, together, boss_sum)
    metric_row([("Model calls", r["run"].calls), ("Steps used / needed", "%d / %d" % (r["steps_run"], r["steps_needed"])),
                ("Smallest limit that works", r["steps_needed"] + 1)])
    show_6a_result(r, msg)
    if msg == le.MULTI_PART and not r["stopped"]:
        if r.get("boss_math"):
            st.warning("The boss got there, but only by asking the stays expert again (more steps, more calls) and then doing the "
                       "arithmetic **in its own head**. That number was not computed by code, so nothing guarantees it. The hybrid desk (page 5) does it properly.")
        else:
            notice("Look at the refund. The policy expert knows the rule (5 days before = 50% back), but not what the couple paid: only the stays expert "
                   "knows River Bend's price. In this design experts never talk to each other, so the answer stops at '50% of what you paid'.")
    st.caption("Even 'Hi, how are you?' costs 3 model calls here: the boss decides, the expert answers, the boss passes the answer on. Central control has a price.")
    with st.expander("What the boss's model sees for ask_stays_expert (generated by LangChain from the function)"):
        st.json(le.TOOL_SCHEMAS.get("ask_stays_expert", {}))
    show_code("The three workers (notebook 6A)", le.CODE["workers"])
    show_code("Agents as tools, the routing rules, and the boss (notebook 6A)", le.CODE["agents_as_tools"])


def lab_6a_decentral():
    st.subheader("Lab B · Notebook 6A: 'is this mine?'")
    picture("a_decentral.png", "The decentralized desk in notebook 6A: each expert asks itself 'is this mine?' and answers or passes it on.", 640)
    st.markdown("No boss. The message goes to the experts in a fixed order. Each one makes a small **structured** yes/no decision "
                "(`{\"handle_it\": true/false, \"reason\": ...}`, one model call). If yes, it answers; if not, it passes the message on. The general expert is the last resort. "
                "A `for` loop means the message can never go round in circles.")
    msg = notebook_message("s6_d6a_msg")
    approach = st.radio("How do the experts handle a message with several parts?",
                        ["As in the notebook: the first expert that says 'mine' answers",
                         "Approach 3 (homework): answer your own part, pass the rest on"], key="s6_d6a_app")
    predict("with the three-part message, which expert answers first, and what happens to the other two parts?")
    r = le.decentralized(msg, "relay" if approach.startswith("Approach 3") else "notebook")
    metric_row([("Model calls", r["run"].calls), ("Parts in the message", len(r["parts"]))])
    show_6a_result(r, msg)
    if msg == le.MULTI_PART:
        if approach.startswith("As in"):
            notice("The stays expert sees a part it can answer, says 'mine', and answers **only that part**. Nobody greets the couple and the refund "
                   "question is lost, because once an expert takes a message, nobody else sees it. Each agent has a **local view**, and nobody checks "
                   "that the **whole** message was answered. It is the cheapest desk, and the least complete.")
        else:
            notice("Now all three parts get an owner, with no boss. But the refund amount is **still** missing: the policy expert cannot get River Bend's "
                   "price, because in this design experts only pass messages along; they cannot ask each other. Decentralization spreads control, not knowledge.")
    st.markdown("**How could a team of equals handle a multi-part message?** The session plan's six approaches:")
    table_show(le.APPROACHES)
    show_code("The decentralized desk (notebook 6A)", le.CODE["decentral"])


def lab_6a_hybrid():
    st.subheader("Lab B · Notebook 6A: a light router, and experts who ask each other")
    picture("a_hybrid.png", "The hybrid desk in notebook 6A: the router only splits; the policy expert asks the stays expert for a price directly.", 520)
    st.markdown("Three steps: **split** (a router makes one structured-output call that labels each part with an expert, and answers nothing), "
                "**answer each part** (each expert works on its own part), **combine** (one call writes one warm reply). The policy expert gets two new tools: "
                "`ask_stays_peer` (a direct message to the stays expert, with no router in between) and `calculate_refund` (the maths, **in code**).")
    msg = notebook_message("s6_h6a_msg")
    peer = st.checkbox("The policy expert can ask the stays expert directly (ask_stays_peer)", True, key="s6_h6a_peer")
    predict("how does the policy expert find out what the couple paid for River Bend Homestay?")
    r = le.hybrid(msg, peer)
    metric_row([("Model calls", r["run"].calls), ("Parts found by the router", len(r["parts"]))])
    show_6a_result(r, msg)
    if msg == le.MULTI_PART and peer:
        notice("Read the printed lines from top to bottom: the router split the message three ways; the general expert greeted; the stays expert gave "
               "Snow Valley's price; and the policy expert, needing River Bend's price, **asked the stays expert directly**, then let code do the sum: "
               "Rs 2,400 × 3 nights = Rs 7,200, and 50% back is Rs 3,600. (Its own 'used' lines appear after its chat with the stays expert, because an "
               "agent's steps are printed when it finishes.) That direct message between experts is what makes this a hybrid.")
    st.markdown("**The refund, in code:** `calculate_refund(2400, 3, 5)` → `%s`" % le.calculate_refund(2400, 3, 5))
    show_code("The router's shape (structured output)", le.CODE["router_schema"])
    show_code("The peer tool, the policy expert with peers, and the hybrid desk (notebook 6A)", le.CODE["hybrid"])


def lab_6a_scoreboard():
    st.subheader("Lab B · Notebook 6A: the same three-part message, three desks")
    st.markdown("Which desk answered best, and at what cost? Each answer is checked for the two numbers that matter: **Rs 3,200** (Snow Valley's price) "
                "and **Rs 3,600** (the refund).")
    rows = le.scoreboard()
    df_show([{k: ("✅" if v is True else "❌" if v is False else v) for k, v in r.items() if not k.startswith("_")} for r in rows])
    with st.expander("Each desk's answer"):
        for r in rows:
            st.markdown("**%s:** %s" % (r["Desk"], r["_answer"]))
    notice("A typical run: the **decentralized** desk is cheapest but answers one part. The **centralized** desk answers more, but the refund amount depends "
           "on the boss noticing the missing price. The **hybrid** desk gets both numbers right, because the router gave every part an owner and the policy "
           "expert could ask its peer. It is not the cheapest: correctness costs a few more calls.")
    st.markdown("**What LangChain left us doing by hand:** the `ask_` wrapper functions, the peer list and its loop, the router shape, and the combining step. "
                "LangChain gave us good agents; **the teamwork between them was all our own code.** That is the gap CrewAI fills (pages 8 to 12).")


# ----------------------------------------------------------------------------------------
# Page 8: How CrewAI works
# ----------------------------------------------------------------------------------------
def page_how_crewai():
    st.header("8 · How CrewAI works")
    story("Prasad's LangChain desks worked, but most of his code was teamwork: wrappers, loops, a router, a combining step. "
          "CrewAI promises that he can describe the team instead of writing it. What does it actually do with his description?")
    plain("**CrewAI** is an open-source Python framework for teams of agents. Instead of writing the teamwork, you **describe** it: who is on the team "
          "(agents), what jobs need doing (tasks), and in what order (the process). Then CrewAI writes the prompts, runs each agent's loop, passes "
          "results between tasks, and lets agents delegate to each other. You focus on the **what**; CrewAI handles the **how**.")
    picture("b_components.png", "A Crew holds Agents and Tasks; the Process decides how the tasks run.", 760)
    table_show([
        {"Block": "Agent", "What it is": "A worker with a role, goal, backstory and tools", "Like...": "An employee"},
        {"Block": "Task", "What it is": "A job: what to do (description), what good looks like (expected_output), who does it (agent), what it reads first (context)", "Like...": "A ticket on a to-do board"},
        {"Block": "Crew", "What it is": "The team: its agents and its tasks, and kickoff() to start the work", "Like...": "A project team (or a publishing house)"},
        {"Block": "Process", "What it is": "How the tasks run: in order (sequential), or under a manager (hierarchical)", "Like...": "How the team is managed"},
    ])

    st.subheader("The same agent in LangChain and in CrewAI")
    c1, c2 = st.columns(2)
    c1.markdown("**LangChain (notebook 6A)**")
    c1.code('''agent = create_agent(
    model=llm,
    tools=[],
    system_prompt="You are a friendly travel buddy for couples "
                  "in India. You suggest short, specific ideas.",
)''', language="python")
    c2.markdown("**CrewAI (notebook 6B)**")
    c2.code('''travel_buddy = Agent(
    role="Travel buddy",
    goal="Suggest short, specific trip ideas for couples in India.",
    backstory="You are a friendly traveller who has explored "
              "every corner of India.",
    llm=llm,
)''', language="python")
    df_show([
        {"LangChain": "model=llm", "CrewAI": "llm=llm", "What it means": "Which model thinks for the agent"},
        {"LangChain": "tools=[...]", "CrewAI": "tools=[...]", "What it means": "What it can do"},
        {"LangChain": "system_prompt=\"...\"", "CrewAI": "role=\"...\"", "What it means": "Its job title"},
        {"LangChain": "", "CrewAI": "goal=\"...\"", "What it means": "What it is trying to achieve"},
        {"LangChain": "", "CrewAI": "backstory=\"...\"", "What it means": "Its personality and experience"},
    ])
    st.caption("CrewAI combines role, goal and backstory into the system prompt for you. Lab 1 shows exactly how.")

    st.subheader("Lab 1 · What CrewAI actually sends to the model")
    st.caption("These prompts were **recorded from the real CrewAI library (version 1.15.28)**, run with a stand-in model that saves every message and replies "
               "from a script. So every prompt below is CrewAI's own; only the replies are scripted. (With Gemini, CrewAI may pass the tool list through "
               "Gemini's native tool calling instead of inside the text. The role, goal, backstory, task and context parts are the same.)")
    scen = st.radio("Scenario", list(ce.RECORDED), format_func=lambda k: ce.RECORDED[k]["title"], key="s6_cr_scen")
    st.markdown("**The code that produced it (notebook 6B)**")
    st.code(ce.RECORDED[scen]["code"], language="python")
    calls = ce.recorded_calls(scen)
    metric_row([("Model calls", len(calls)), ("Agents involved", len({c["agent"] for c in calls}))])
    for c in calls:
        with st.expander("Model call %d · %s" % (c["n"], c["agent"]), expanded=(c["n"] == 1)):
            for m in c["messages"]:
                st.caption("**%s message**" % m["role"])
                code_block(m["content"])
            st.markdown("**The model's reply (scripted):**")
            code_block(c["reply"])
            if c["after"]:
                st.info("**What CrewAI did next:** " + c["after"])
    st.markdown("**Where each part of the prompt comes from**")
    table_show(ce.PROMPT_PARTS)
    notice("Nothing here is magic: an agent is a system prompt built from role, backstory and goal; a task is a user message built from the description, "
           "the expected_output and the context. The ReAct loop (Thought, Action, Observation, Final Answer) is the same one you built in LangChain; CrewAI writes its instructions for you.")

    st.subheader("Running it: kickoff, kickoff_async and the result")
    c1, c2 = st.columns(2)
    c1.markdown("**One agent on its own**")
    c1.code('''result = await travel_buddy.kickoff_async(
    "Suggest one weekend trip from Mumbai for a quiet couple.")
print(result.raw)            # the answer text''', language="python")
    c2.markdown("**A crew**")
    c2.code('''result = await blog_crew.kickoff_async(inputs={"topic": "..."})
print(result.raw)            # the last task's output
for t in result.tasks_output:   # one entry per task
    print(t.agent, t.raw)''', language="python")
    st.markdown("- **Why `await`?** Notebooks (Colab and VS Code) already run an event loop, and CrewAI must run inside it. `await` means *wait here until the answer is ready*. "
                "In a normal Python script, write `crew.kickoff(...)` instead.\n"
                "- **The boxes CrewAI prints** (for example 'LiteAgent Started') are CrewAI showing what it is doing. Your answer is printed below them.\n"
                "- **`inputs={...}`** fills every `{placeholder}` in goals and task descriptions, so one crew works for any topic.")

    st.subheader("Tools in CrewAI")
    st.code('''from crewai.tools import tool

@tool("Travel guide")                     # a decorator: "this function is a tool"
def search_travel_guide(place: str) -> str:
    """Read the Wikivoyage travel guide for a place, e.g. Old Manali. Returns what to see, do and eat there."""
    ...''', language="python")
    st.markdown("The **docstring** is what the agent reads to decide when to use the tool, exactly as in LangChain. In the recorded prompts above, notice that "
                "CrewAI writes the tool's name as **`travel_guide`**, not 'Travel guide'.")

    st.subheader("Settings worth knowing (checked on CrewAI 1.15)")
    table_show([
        {"Setting": "pip install \"crewai[google-genai]\"", "What it does": "The [google-genai] part installs CrewAI's connection to Gemini; without it CrewAI cannot use Gemini models"},
        {"Setting": "LLM(model=\"gemini/gemini-3.5-flash-lite\")", "What it does": "The model the agents use; 'gemini/' says which company makes it"},
        {"Setting": "os.environ[\"CREWAI_TRACING_ENABLED\"] = \"false\"", "What it does": "Answers CrewAI's first-run tracing question in advance, so a notebook does not wait forever"},
        {"Setting": "verbose=True", "What it does": "Prints the agent's thinking: Thought, Action, Observation"},
        {"Setting": "max_rpm=4", "What it does": "At most 4 model calls a minute (CrewAI's own rate limiter): stays inside the free tier"},
        {"Setting": "allow_delegation=True", "What it does": "Default False. Gives the agent two tools: delegate work to a coworker, ask a coworker a question (page 12)"},
        {"Setting": "max_iter (default 25)", "What it does": "The agent's step limit: like recursion_limit in LangChain"},
        {"Setting": "memory=True on a Crew", "What it does": "Memory across runs. Needs an embedding model (OpenAI's by default, so with only a Gemini key set embedder= to Gemini's) and makes extra calls"},
    ])

    st.subheader("What CrewAI did for us, and what it didn't")
    table_show(ce.CREWAI_VS_LANGCHAIN_6B)
    st.warning("CrewAI removed most of the teamwork **code**. It did not remove a single **decision**: you still choose the roles, write the goals and backstories "
               "(which are prompts), decide which tasks to assign, and pick the process. And it adds model calls you did not write (pages 11 and 12).")
    show_code("Notebook 6B's setup cell", '''%pip install -q "crewai[google-genai]>=1.15" requests python-dotenv

import os
os.environ["CREWAI_TRACING_ENABLED"] = "false"
from crewai import Agent, Task, Crew, Process, LLM

llm = LLM(model="gemini/gemini-3.5-flash-lite", api_key=os.environ["GOOGLE_API_KEY"])''')
    st.success("**Takeaway.** A CrewAI agent is a prompt built from role, goal and backstory. A task is a message built from description, expected_output and "
               "context. The crew runs them, in order or under a manager. Next: build one yourself.")
    st.button("Next: CrewAI building blocks →", on_click=goto, args=(PAGES[9],), type="primary")



# ----------------------------------------------------------------------------------------
# Page 9: CrewAI blocks
# ----------------------------------------------------------------------------------------
def page_blocks():
    st.header("9 · CrewAI building blocks")
    story("Prasad rewrites his research-and-blog team in CrewAI. He finds he is filling in four kinds of object instead of writing coordination code.")
    df_show([
        {"Block": "Crew", "What it is": "The container and orchestrator: which agents, which tasks, how they are connected", "Analogy": "The publishing house managing the whole project"},
        {"Block": "Agent", "What it is": "An individual AI worker with a role, a persona, and access to tools or memory", "Analogy": "An employee who knows their job"},
        {"Block": "Task", "What it is": "A unit of work: a goal (description), an expected output, and an assigned agent", "Analogy": "A ticket on a Jira board"},
        {"Block": "Process", "What it is": "How tasks run: sequentially, or hierarchically under a manager", "Analogy": "The workflow the team follows"},
    ])
    st.subheader("Lab · Write the crew and watch the code")
    cfg = copy.deepcopy(ce.DEFAULT_CREW)
    cols = st.columns(2)
    for col, (k, a) in zip(cols, cfg["agents"].items()):
        with col:
            st.markdown("**Agent: %s**" % k)
            a["role"] = st.text_input("role: what the agent *is*", a["role"], key="s6_%s_role" % k)
            a["goal"] = st.text_area("goal: what it is trying to achieve ({topic} is a placeholder)", a["goal"], key="s6_%s_goal" % k, height=70)
            a["backstory"] = st.text_area("backstory: helps it keep a persona", a["backstory"], key="s6_%s_back" % k, height=70)
    cols = st.columns(2)
    for col, t in zip(cols, cfg["tasks"]):
        with col:
            st.markdown("**Task: %s** (agent: %s%s)" % (t["id"], t["agent"], ", context: " + ", ".join(t["context"]) if t["context"] else ""))
            t["description"] = st.text_area("description: the instruction", t["description"], key="s6_%s_desc" % t["id"], height=90)
            t["expected_output"] = st.text_input("expected_output: what success looks like", t["expected_output"], key="s6_%s_exp" % t["id"])
    st.session_state["s6_cfg"] = cfg
    topic = st.text_input("Topic for kickoff(inputs={'topic': ...})", "The future of electrical vehicles", key="s6_topic")
    inputs = {"topic": topic}
    st.markdown("**What the model will actually read** (placeholders filled by kickoff)")
    df_show([{"Where": "researcher goal", "Text": ce.fill(cfg["agents"]["researcher"]["goal"], inputs)},
             {"Where": "task1 description", "Text": ce.fill(cfg["tasks"][0]["description"], inputs)},
             {"Where": "task2 description", "Text": ce.fill(cfg["tasks"][1]["description"], inputs)}])
    miss = ce.missing_inputs(cfg, inputs)
    if miss:
        st.warning("Your text uses {%s}, but kickoff only supplies 'topic'. Add the value to inputs, or remove the placeholder." % "}, {".join(miss))
    st.markdown("**The CrewAI code this configuration corresponds to**")
    st.code(ce.crew_code(cfg, inputs), language="python")
    notice("Notice what is **not** here: no loop, no hand-off code, no message passing. `context=[task1]` is the entire 'the writer uses the research' instruction. "
           "Running real CrewAI also needs `pip install \"crewai[google-genai]\"` and a Gemini key; this page only builds the code for you. "
           "The code above was checked to build a real Crew with CrewAI 1.15.")
    st.markdown("**Notebook 6B's first steps, in order** (each adds one idea):")
    table_show([
        {"Step": "1 · One agent", "Code": "await travel_buddy.kickoff_async(\"Suggest one weekend trip...\")", "The idea": "An agent is role + goal + backstory instead of a system prompt. Try a grumpy backstory: same question, different answer"},
        {"Step": "2 · A tool", "Code": "@tool(\"Travel guide\") above a function; tools=[search_travel_guide]; verbose=True; max_rpm=4", "The idea": "The docstring tells the agent when to use the tool; verbose shows Thought / Action / Observation"},
        {"Step": "3 · A task", "Code": "Task(description=..., expected_output=\"Exactly three bullet points...\", agent=local_guide)", "The idea": "Separate what to do from who does it; expected_output shapes the answer. Even one agent runs inside a Crew"},
        {"Step": "4 · Two agents in a row", "Code": "context=[research_task], {topic}, Process.sequential", "The idea": "Page 10"},
    ])
    st.success("**Takeaway.** In CrewAI you describe **what** (roles, tasks, who feeds whom) and the crew handles **how**.")
    st.button("Next: run it, sequentially →", on_click=goto, args=(PAGES[10],), type="primary")


# ----------------------------------------------------------------------------------------
# Page 10: sequential
# ----------------------------------------------------------------------------------------
def page_sequential():
    b = get_backend()
    st.header("10 · Process: sequential")
    plain("Sequential means tasks run in a fixed order, and one task's output feeds the next: research → write → edit. Best when later work depends on earlier results. "
          "In the notebook the writer receives the research through `context=[task1]`.")
    cfg = crew_cfg()
    inputs = crew_inputs()
    picture("crewai_sequential.jpg", "From the session plan: research, then writing, then editing, each task feeding the next. "
            "The lab below runs the first two steps; adding the editor is notebook 6B's first exercise.")
    st.caption("Using the agents and tasks you wrote on page 9 (or the notebook's defaults if you skipped it). Topic: **%s**" % inputs["topic"])
    st.subheader("Lab · kickoff() and break it")
    c1, c2, c3 = st.columns(3)
    use_ctx = c1.checkbox("Task 2 uses context=[task1]", True, key="s6_ctx", help="Unticked = context=[] (an explicit empty list). Leaving context out entirely is different: in CrewAI 1.15 a sequential task with no context parameter receives ALL earlier outputs.")
    wrong = c2.checkbox("List the writer's task FIRST", False, key="s6_wrongorder")
    typo = c3.checkbox("Mistype the input key ('topics')", False, key="s6_typo")
    predict("untick the context box. The writer still runs. What does it write about?")
    if not use_ctx:
        cfg["tasks"][1]["context"] = []
    if wrong:
        cfg["tasks"] = [cfg["tasks"][1], cfg["tasks"][0]]
        cfg["tasks"][0]["context"] = ["task1"] if use_ctx else []
    run_inputs = {"topics": inputs["topic"]} if typo else inputs
    r = ce.run_crew(cfg, run_inputs, "sequential", None, False, True, live_backend())
    if r["error"]:
        st.error("**kickoff() would stop here:** " + r["error"])
    else:
        metric_row([("Model calls", r["calls"]), ("Tasks run", len(r["outputs"])), ("Manager?", "none (fixed order)")])
        for t in cfg["tasks"]:
            ok, why = ce.meets_expected(t, r["outputs"][t["id"]])
            with st.expander("%s · %s · expected: %s" % (t["id"], cfg["agents"][t["agent"]]["role"], t["expected_output"]), expanded=True):
                st.text(r["outputs"][t["id"]])
                st.caption(("✅ " if ok else "❌ ") + "expected_output check: " + why)
        if not use_ctx:
            st.error("The writer ran with **no research**. It still produced something: in this demo a refusal, but a real model would likely invent facts and sound confident. "
                     "The order was right; the hand-off was missing. That is what `context` is for.")
            st.caption("🧪 Checked on CrewAI 1.15.30: this happens with `context=[]`. If you simply leave `context` out of a sequential task, CrewAI passes it ALL earlier outputs instead.")
        else:
            st.success("The writer built its summary from the researcher's bullets.")
        badge(b)
        show_log(r["log"])
    notice("Two of the three mistakes are caught by the (mini) framework before any model is called: a missing input, and a task that needs another task's output before it exists. "
           "The third, removing the context, is silent: nothing errors, the answer is just worse. Silent failures are the dangerous ones.")
    st.markdown("**Workflow summary:** the crew receives a topic → the research agent collects 3 to 5 facts → the writer turns them into a blog summary → the crew manages the dependency and runs them in order.")
    show_code("The sequential crew (notebook 6B, step 4)", """
blog_crew = Crew(agents=[researcher, writer], tasks=[research_task, writing_task], process=Process.sequential, max_rpm=4)
result = await blog_crew.kickoff_async(inputs={"topic": "a quiet weekend in Old Manali"})   # in a script: blog_crew.kickoff(...)

for task_output in result.tasks_output:      # one entry per task: see what each agent produced
    print(f"=== {task_output.agent} ===")
    print(task_output.raw)""")
    st.caption("Want to see the exact prompt the writer receives, with the researcher's output pasted in as context? Page 8, Lab 1, scenario 'A two-task pipeline'.")
    st.success("**Takeaway.** Sequential is the pipeline pattern with the plumbing done for you. Check **what flows between tasks**, not just that every task ran.")
    st.button("Next: add a manager →", on_click=goto, args=(PAGES[11],), type="primary")


# ----------------------------------------------------------------------------------------
# Page 11: hierarchical
# ----------------------------------------------------------------------------------------
def page_hierarchical():
    b = get_backend()
    st.header("11 · Process: hierarchical (a manager)")
    plain("In a hierarchical process a **manager** decides who does each task and can check the work: CrewAI's version of the **manager / worker** pattern. "
          "You supply either a **manager_llm** (CrewAI builds its own 'Crew Manager' agent around that model: the lighter way) or a **manager_agent** "
          "(your own Agent with a role, goal and backstory). Two rules from notebook 6B: **leave the tasks' `agent` empty** so the manager can choose, and "
          "**a manager cannot have tools**: its only tools are the two delegation tools CrewAI gives it.")
    st.warning("**The PDF's manager_agent example gives the manager `tools=[search_tool, code_executor]` 'to verify work'. Current CrewAI refuses that:** "
               "checked on CrewAI 1.15, `kickoff()` fails with *'Manager agent should not have tools'*. A manager checks work through its **instructions** "
               "(goal and backstory) and by asking coworkers; the tools belong to the workers.")
    cfg = crew_cfg()
    inputs = crew_inputs()
    st.subheader("Lab · The manager that checks")
    c1, c2, c3 = st.columns(3)
    kind = c1.radio("Manager", ["manager_llm (CrewAI's default manager)", "manager_agent told to check expected_output",
                                "manager_agent given tools (the PDF's example)", "No manager"], key="s6_mgr2")
    assigned = c2.checkbox("Assign each task to an agent (agent=...)", False, key="s6_assigned",
                           help="Notebook 6B, exercise 3: if a task names its agent, the manager has no choice. It can only pass the task on.")
    sloppy = c3.checkbox("The researcher is sloppy (returns only 2 facts)", True, key="s6_sloppy")
    predict("the task asks for '3-5 facts' but the researcher returns 2. Which manager notices? Then pick the PDF's manager with tools.")
    manager = (None if kind.startswith("No") else {"kind": "llm", "assigned": assigned} if kind.startswith("manager_llm")
               else {"kind": "agent", "checks": True, "assigned": assigned} if "told to check" in kind
               else {"kind": "agent", "has_tools": True, "assigned": assigned})
    r = ce.run_crew(cfg, inputs, "hierarchical", manager, sloppy, True, live_backend(), assigned=assigned)
    if r["error"]:
        st.error("**kickoff() would stop here:** " + r["error"])
    else:
        facts = len(ce.bullets(r["outputs"][cfg["tasks"][0]["id"]]))
        metric_row([("Model calls", r["calls"]), ("Facts the writer received", facts), ("Tasks redone", len(r["reworked"]))])
        if r["checks"]:
            df_show([{"Review": c["task"], "Passed": "✅" if c["ok"] else "❌", "Detail": c["why"]} for c in r["checks"]])
        if sloppy and not r["reworked"]:
            st.error("The writer built a blog post on **%d** facts and nobody objected: the manager did not check the work, so the task's expected_output was never enforced." % facts)
        elif r["reworked"]:
            st.success("The manager rejected the short research, sent it back, and the writer received %d facts." % facts)
        else:
            st.success("The research was fine, so there was nothing to catch.")
        with st.expander("Final output"):
            st.text(r["final"])
        badge(b)
        show_log(r["log"])
    st.subheader("What each process costs (same two tasks, same sloppy researcher)")
    df_show(ce.process_costs(cfg, inputs, sloppy))
    notice("Sequential used 2 calls. A manager adds planning and final assembly. A manager that **checks** adds a review per task, and more when it sends work back. "
           "That is the price of the hierarchy: the check bought a correct 4-fact result for 8 calls instead of an unchecked 2-fact result for 4. Whether that is worth it is a per-feature decision.")
    if assigned and not r["error"]:
        st.info("With every task assigned, the log shows 'delegate (no choice)': the manager can only pass each task to the agent you named. "
                "You get a pipeline with a manager in the middle, and its extra model calls. Leave tasks unassigned when you want the manager to choose.")
    st.markdown("**manager_llm or manager_agent?** With `manager_llm`, CrewAI builds a standard manager for you (its prompt is recorded on page 8: "
                "'You are Crew Manager...'). With `manager_agent` you write the manager's role, goal and backstory yourself, so you can tell it **what to check** "
                "(for example 'send back research with fewer than 3 facts'). Either way, the manager gets no tools of its own.")
    st.markdown("**In the log of a real run**, look for **`delegate_work_to_coworker`** (or 'Delegate work to coworker'): that is the manager handing part of the job "
                "to a worker. Nobody told it the order; it decided.")
    st.markdown("**The CrewAI code for this crew**")
    code_mgr = dict(manager or {"kind": "llm"})
    code_mgr.pop("has_tools", None)
    st.code(ce.crew_code(cfg, inputs, "hierarchical", code_mgr), language="python")
    st.caption("In this demo the 'check' is a simple rule (count the bullets). In real CrewAI the manager is itself a model, which can miss problems or raise false alarms. "
               "The code above was checked to build a real Crew with CrewAI 1.15 (without tools on the manager).")
    st.success("**Takeaway.** A manager only helps if it can **check** something. A manager that accepts everything is just extra calls.")
    st.button("Next: agents asking each other →", on_click=goto, args=(PAGES[12],), type="primary")


# ----------------------------------------------------------------------------------------
# Page 12: Agents asking each other (allow_delegation)
# ----------------------------------------------------------------------------------------
def page_delegation():
    st.header("12 · Agents asking each other")
    story("DilSeTrips' copywriter writes homepage blurbs. It is good with words but has no tools; the fact checker can read the travel guide. "
          "Prasad lets the copywriter ask.")
    plain("Give an agent **`allow_delegation=True`** and CrewAI hands it two extra tools: **Delegate work to coworker** and **Ask question to coworker** "
          "(in the prompt they appear as `delegate_work_to_coworker` and `ask_question_to_coworker`). Two agents then collaborate **directly**, with no manager "
          "in between: CrewAI's version of the peer collaboration in the hybrid desk.")
    dot('digraph { rankdir=LR; ' + NODE + ' Task [fillcolor="#fff3bf", label="blurb_task"]; Copywriter [fillcolor="#ffe3e3"]; Fact_checker [label="Fact checker"]; '
        'guide [shape=note, label="travel guide tool"]; Task -> Copywriter; Copywriter -> Fact_checker [label="Ask question\\nto coworker", dir=both, color="#2f9e44", penwidth=2]; '
        'Fact_checker -> guide [dir=both]; }')
    st.subheader("Lab · Will the copywriter check its facts?")
    c1, c2 = st.columns(2)
    allow = c1.checkbox("allow_delegation=True on the copywriter", True, key="s6_dg_allow")
    confident = c2.checkbox("The copywriter feels confident (the model skips the question)", False, key="s6_dg_conf", disabled=not allow)
    predict("untick allow_delegation. The copywriter has no tools. Where do its two 'specific places' come from?")
    r = ce.delegation(allow, confident)
    metric_row([("Model calls", r["calls"]), ("Asked the fact checker?", "yes" if r["checked"] else "no")])
    (st.success if r["checked"] else st.error)("**Blurb:** " + r["blurb"])
    if not r["checked"]:
        st.caption("The places in this blurb are invented by the demo script to make the point: with no check, a confident model can name places that do not exist, "
                   "and nothing in the output tells you it guessed.")
    show_log(r["log"])
    notice("allow_delegation=True makes asking **possible**, not **required**. The model decides, and a confident agent may skip the check. In the log of a real run, "
           "look for `ask_question_to_coworker` (or 'Ask question to coworker'). How would you know if a check was skipped? Only by reading the log, or by a rule "
           "that refuses a blurb without a recorded check.")
    st.caption("Want to see the exact prompts for this crew? Page 8, Lab 1, scenario 'A worker asks a coworker' shows all four recorded model calls.")

    st.subheader("Session 5's patterns, in CrewAI")
    picture("b_patterns.png", "Each design pattern from Session 5 has a natural CrewAI form.", 620)
    table_show(ce.PATTERNS_IN_CREWAI)
    st.markdown("**Federated teams (exercise 5):** CrewAI has no 'team of crews' object. Build one crew per team, run each, and combine their results in plain Python. "
                "Ask both teams for exactly three lines starting with '- ', so combining is easy:")
    st.code('''destination = await destination_crew.kickoff_async(inputs={"place": "Old Manali"})
packing = await packing_crew.kickoff_async(inputs={"place": "Old Manali"})
lines = [l for l in (destination.raw + "\\n" + packing.raw).splitlines() if l.startswith("- ")]
print("\\n".join(lines))        # integration is your code, as in Session 5''', language="python")

    st.subheader("A word on memory")
    st.markdown("Inside one run, results already flow between tasks through `context`. For memory **across** runs, CrewAI has `memory=True` on a crew. "
                "Two things to know first: it needs an **embedding model** (by default OpenAI's, so with only a Gemini key you must set `embedder=` to Gemini's), "
                "and it makes **extra model calls** to decide what to remember. On the free tier, try it last.")
    st.success("**Takeaway.** allow_delegation turns a crew of workers into colleagues who can ask each other, at the cost of calls you did not write and checks the model may skip.")
    st.button("Next: review →", on_click=goto, args=(PAGES[13],), type="primary")

# ----------------------------------------------------------------------------------------
# Page 13: review
# ----------------------------------------------------------------------------------------
QUIZ = [
    ("Message 3 says 'and the visa fee there?' and the agent's memory is off. What happens in the lab?",
     ["It searches for 'there'", "It asks which place you mean", "It answers from the web anyway", "It crashes"], 1,
     "Memory is earlier messages put back into the prompt; without it 'there' has no meaning."),
    ("In the centralized desk, which agents are the workers?",
     ["The orchestrator", "Product, Policy and General", "The customer", "Only the General agent"], 1,
     "The orchestrator is the manager; the three specialists are the workers."),
    ("A customer asks about the Goa package and its cancellation in one sentence. Why did the plain centralized desk give half an answer?",
     ["The model was slow", "The orchestrator sent the whole question to one agent", "Cancellation is not in the data", "The Policy agent was down"], 1,
     "Routing to one worker loses the other half. Splitting fixes it at the cost of more calls."),
    ("What protects a decentralized desk from forwarding a question around forever?",
     ["A faster model", "A hop limit and a final fallback", "More agents", "A longer prompt"], 1,
     "No one owns the whole picture, so the loop must be bounded and every chain must end somewhere."),
    ("The orchestrator crashes. Which design keeps answering with no extra work?",
     ["Centralized with splitting", "Centralized, plain", "Decentralized", "None"], 2,
     "Decentralized has no orchestrator. Hybrid also survives, but only if you build the peer fallback."),
    ("The Product agent is down. What does the comparison show?",
     ["Decentralized copes best", "Hybrid copes best", "No design copes: only that agent knows its topic", "Centralized with splitting copes best"], 2,
     "Spreading control is not spreading knowledge. Important topics need a second owner."),
    ("Why does asking a peer directly use fewer model calls in the hybrid desk?",
     ["Peers use a cheaper model", "It skips the orchestrator's extra routing call", "It skips the answer", "It skips the customer"], 1,
     "The price is oversight: the orchestrator no longer sees every message."),
    ("Which statement about patterns and architectures is right?",
     ["They are the same thing", "Patterns say how agents interact; architectures say how the system is structured", "Patterns are for LangChain only", "Architectures are for CrewAI only"], 1,
     "Like how people use a building versus its blueprint."),
    ("In CrewAI, what does context=[task1] on task2 do?",
     ["Runs task2 first", "Gives task2 the output of task1", "Deletes task1", "Makes task1 optional"], 1,
     "It is the hand-off. Remove it and the writer may invent facts without any error."),
    ("A crew has process hierarchical and no manager. What is needed?",
     ["Nothing", "A manager_llm or a manager_agent", "More tasks", "A second crew"], 1,
     "A hierarchical process must have a manager."),
    ("The model's reply contains tool_calls=[search_stays(...)]. Who runs search_stays?",
     ["The model, inside Gemini", "LangChain runs your Python function and adds a ToolMessage", "Nobody: it is just text", "CrewAI"], 1,
     "The model only requests a tool. Your code (through create_agent) runs it and puts the result back in the message list."),
    ("The boss must call three experts, one per step. With recursion_limit=4, what happens?",
     ["It finishes", "GraphRecursionError: the run needs 7 steps (limit at least 8)", "It answers only the greeting", "It calls them in parallel automatically"], 1,
     "One step per model call and one per round of tool runs: 4 model calls + 3 tool rounds = 7 steps."),
    ("Why does only the hybrid desk get the Rs 3,600 refund right in notebook 6A?",
     ["It uses a bigger model", "The policy expert asks the stays expert for the price directly, and calculate_refund does the maths in code", "It skips the greeting", "The router computes it"], 1,
     "Only that design lets one expert get another's knowledge, and the sum is code, not the model's arithmetic."),
    ("In CrewAI, where does an agent's backstory end up?",
     ["In the task description", "In the system prompt, right after 'You are {role}.'", "In a separate memory store", "Nowhere: it is documentation"], 1,
     "Recorded from CrewAI 1.15: 'You are {role}. {backstory}  Your personal goal is: {goal}'."),
    ("You give a CrewAI manager_agent tools=[search_tool]. What happens on kickoff()?",
     ["The manager uses the tool to check work", "It fails: 'Manager agent should not have tools'", "The tool is given to the workers", "Nothing changes"], 1,
     "Checked on CrewAI 1.15. The PDF's example no longer works: managers only delegate."),
    ("What does allow_delegation=True do?",
     ["Forces the agent to ask coworkers", "Gives the agent two tools: delegate work to a coworker, and ask a coworker a question", "Makes it the manager", "Turns on memory"], 1,
     "It makes asking possible, not required: the model decides."),
]


def page_review():
    st.header("13 · Review, quiz & cheat sheet")
    st.subheader("🧠 Quiz")
    with st.form("quiz6"):
        answers = [st.radio("**%d. %s**" % (i + 1, q), opts, index=None, key="q6_%d" % i) for i, (q, opts, c, w) in enumerate(QUIZ)]
        submitted = st.form_submit_button("Check my answers", type="primary")
    if submitted:
        score = 0
        for i, (q, opts, c, why) in enumerate(QUIZ):
            ok = answers[i] == opts[c]
            score += ok
            (st.success if ok else st.error)("**Q%d:** %s. %s" % (i + 1, "Correct" if ok else "Not quite", why))
        st.metric("Score", "%d / %d" % (score, len(QUIZ)))
    st.subheader("📝 Homework map")
    table_show([
        {"#": "6A HW1", "Level": "Easy", "Task": "A fourth worker: add a weather expert (get_weather_forecast) to the centralized desk. What changes: the worker, its ask_ function, the routing rules? Did any other worker change?", "Try it": "Page 3, Lab B (agents as tools)"},
        {"#": "6A HW2", "Level": "Easy", "Task": "Step limits: run the multi-part message with limits 4, 6, 9 and 12. When did Gemini call the experts together, and when one at a time?", "Try it": "Page 3, Lab B; page 1, Lab 3"},
        {"#": "6A HW3", "Level": "Medium", "Task": "Cheaper decisions: could a keyword check skip the obvious 'is this mine?' calls? Careful: 'stays' also matches 'homestays' in a cancellation question.", "Try it": "Page 4, Lab B (try 'cancellation policy for homestays')"},
        {"#": "6A HW4", "Level": "Medium", "Task": "Approach 3: each expert answers only its part and passes the rest on. Does it answer all three parts? How many calls versus the hybrid desk?", "Try it": "Page 4, Lab B (relay)"},
        {"#": "6B Ex1", "Level": "Easy", "Task": "Add the editor: a third agent and task that polishes the blog post. Which context does it need?", "Try it": "Page 9 (code) and page 10"},
        {"#": "6B Ex2", "Level": "Easy", "Task": "Change a personality: give the writer a different backstory (a poet? a stand-up comic?) and compare", "Try it": "Page 9 (backstory field)"},
        {"#": "6B Ex3", "Level": "Medium", "Task": "Assigned vs unassigned: add agent=writer to the welcome task. What changed in the log? Was the manager worth its calls?", "Try it": "Page 11 (assign checkbox)"},
        {"#": "6B Ex4", "Level": "Medium", "Task": "Is the coworker question worth it? Set allow_delegation=False. Does the blurb change? How would you know a check was skipped?", "Try it": "Page 12"},
        {"#": "6B Ex5", "Level": "Hard", "Task": "Federated teams: two crews (destination; packing), each returning three '- ' lines, combined in plain Python", "Try it": "Page 12 (federated code)"},
        {"#": "PDF HW", "Level": "Medium", "Task": "Make the decentralized desk loop-proof and give the hybrid desk a fallback for a dead orchestrator. What new risks appear?", "Try it": "Pages 4 and 6 (Lab A)"},
    ])
    st.subheader("⚡ Quick reference")
    table_show([
        {"Idea": "Architecture", "One line": "How the system is structured: centralized, decentralized, hybrid"},
        {"Idea": "Design pattern", "One line": "How agents interact: manager/worker, peer/swarm, pipeline, federated teams"},
        {"Idea": "Centralized desk", "One line": "Orchestrator routes; simple and easy to debug; single point of failure; mixed questions need splitting"},
        {"Idea": "Decentralized desk", "One line": "Peers forward; resilient; more chatter; loops need a hop limit and a fallback"},
        {"Idea": "Hybrid desk", "One line": "Route once, then peers ask each other; efficient; survives the boss only if you build the fallback"},
        {"Idea": "LangChain", "One line": "The parts: models, tools, memory, chains. You do the coordination."},
        {"Idea": "One agent (LangChain 1.x)", "One line": "create_agent(model=llm, tools=[...], system_prompt=\"...\"); answer = result[\"messages\"][-1].text"},
        {"Idea": "Memory (LangChain 1.x)", "One line": "Keep the messages in a list and send the whole list each turn"},
        {"Idea": "Agents as tools", "One line": "A function that calls ask(other_agent, question), given to the boss as a tool"},
        {"Idea": "Step limit", "One line": "config={\"recursion_limit\": N}: one step per model call and per round of tool runs; stops with an error"},
        {"Idea": "Structured output", "One line": "llm.with_structured_output(schema).invoke(prompt) returns a dictionary"},
        {"Idea": "Money in code", "One line": "A tool like calculate_refund, never the model's arithmetic"},
        {"Idea": "CrewAI", "One line": "A team framework: Crew, Agent (role, goal, backstory), Task (description, expected_output, context), Process"},
        {"Idea": "Sequential", "One line": "Fixed order; each task's output feeds the next through context"},
        {"Idea": "Hierarchical", "One line": "A manager delegates and (if it has a check) verifies; needs manager_llm or manager_agent"},
        {"Idea": "Run a crew (notebook)", "One line": "result = await crew.kickoff_async(inputs={\"topic\": \"...\"}); result.raw; result.tasks_output"},
        {"Idea": "CrewAI tool", "One line": "@tool(\"Name\") above a function with a docstring"},
        {"Idea": "Agents asking each other", "One line": "allow_delegation=True (adds 'Delegate work to coworker' and 'Ask question to coworker')"},
        {"Idea": "CrewAI safety settings", "One line": "max_rpm=4; CREWAI_TRACING_ENABLED=\"false\"; pip install \"crewai[google-genai]\""},
        {"Idea": "Cost", "One line": "Every extra agent, manager or check is more model calls"},
    ])
    st.subheader("What the session concludes")
    st.markdown("""
1. **Architecture determines interaction style.** Centralized, decentralized and hybrid shape how agents communicate; each supports particular patterns.
2. **Frameworks simplify MAS development.** LangChain provides the parts for single agents (memory, tools, chains); CrewAI adds role management, task sequencing and collaboration, speeding up prototypes.
3. **Design patterns map to real implementations.** Manager/worker, pipeline, federated teams and peer/swarm are implementable, and help structure interactions, task flows and responsibility.
4. **Effective MAS must address systemic challenges.** Scalability, uncertainty, credit assignment and emergent behaviour: decentralization, probabilistic reasoning, reward structuring and simulation help.
5. **MAS design mirrors human organisations.** Role specialisation, communication flow and incentive alignment inform better design.
""")
    st.info("A team framework changes who writes the coordination code. It does not change who is responsible for checking the answer.")


# ----------------------------------------------------------------------------------------
# Sidebar + router
# ----------------------------------------------------------------------------------------
def sidebar():
    st.sidebar.title("🛠️ Building MAS Lab")
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
            st.sidebar.success("Live: connected to Gemini (pages 10 and 11 use it)")
        else:
            st.sidebar.warning(b.error or "Paste a key to go live. Demo mode meanwhile.")
    else:
        st.sidebar.info("Demo mode: scripted agents, pretend data.")


def main():
    sidebar()
    page = st.session_state.get("nav", PAGES[0])
    {PAGES[0]: page_home, PAGES[1]: page_how_langchain, PAGES[2]: page_react, PAGES[3]: page_central, PAGES[4]: page_decentral,
     PAGES[5]: page_hybrid, PAGES[6]: page_compare, PAGES[7]: page_framework, PAGES[8]: page_how_crewai, PAGES[9]: page_blocks,
     PAGES[10]: page_sequential, PAGES[11]: page_hierarchical, PAGES[12]: page_delegation, PAGES[13]: page_review}[page]()


main()
