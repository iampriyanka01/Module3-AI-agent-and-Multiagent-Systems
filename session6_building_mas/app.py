"""
Building MAS Lab: an interactive companion to "Agents, Session 6 (cont.): Building Multi-Agent Systems with LangChain and CrewAI".

Run:   streamlit run app.py
Works with NO API key and with neither LangChain nor CrewAI installed (Demo mode: scripted agents, pretend data).
Add a Gemini key for Live mode on the CrewAI-style labs (pages 8 and 9).
"""
import copy
import inspect
import os

import pandas as pd
import streamlit as st

import crew_engine as ce

st.set_page_config(page_title="Building MAS Lab", page_icon="🛠️", layout="wide")

PAGES = [
    "🏠 Start here",
    "1 · One agent in LangChain: tool + memory",
    "2 · Centralized support desk",
    "3 · Decentralized support desk",
    "4 · Hybrid support desk",
    "5 · Three designs, one set of questions",
    "6 · Why a framework? LangChain vs CrewAI",
    "7 · CrewAI building blocks",
    "8 · Process: sequential",
    "9 · Process: hierarchical (a manager)",
    "10 · Review, quiz & cheat sheet",
]

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
    st.caption("An interactive companion to *Agents, Session 6 (cont.): Building Multi-Agent Systems* with LangChain and CrewAI.")
    story("Last session Prasad drew the team on a whiteboard: who decides, and who talks to whom. Now he has to build it. His support desk needs a Product agent, a Policy agent and a "
          "General agent. He builds it three ways, then wonders if a framework could do the plumbing for him.")
    st.success("**Two ideas to hold on to:** an *architecture* is how the system is structured (the blueprint); a *design pattern* is how the agents interact (how people use the building). "
               "And a framework such as CrewAI does not make agents smarter: it takes over the coordination code you would otherwise write by hand.")
    dot('digraph { rankdir=LR; ' + NODE + ' Customer [fillcolor="#fff3bf"]; Orchestrator [fillcolor="#ffe3e3"]; '
        'Customer -> Orchestrator; Orchestrator -> Product; Orchestrator -> Policy; Orchestrator -> General; Product -> Policy [style=dashed, dir=both, label="ask directly"]; }')
    st.subheader("Your path")
    path = [
        ("1 · One agent in LangChain", PAGES[1], "A ReAct agent with a search tool and a chat memory. Switch the memory off and watch a follow-up fail."),
        ("2 · Centralized support desk", PAGES[2], "An orchestrator routes each query to one worker. A mixed question gets half an answer. Switch the orchestrator off."),
        ("3 · Decentralized support desk", PAGES[3], "No orchestrator: each agent answers its part and forwards the rest. Add a loop, then a hop limit."),
        ("4 · Hybrid support desk", PAGES[4], "The orchestrator routes, but agents ask each other directly. Direct versus via the boss."),
        ("5 · Three designs, one set of questions", PAGES[5], "The same five questions through every design: calls, messages, who is fully answered, and what each survives."),
        ("6 · Why a framework?", PAGES[6], "The notebook's LangChain vs CrewAI table as a checklist of what you wire by hand."),
        ("7 · CrewAI building blocks", PAGES[7], "Crew, Agents, Tasks, Process. Edit role, goal and task text; the CrewAI code updates."),
        ("8 · Process: sequential", PAGES[8], "Research then write. Remove the context, reorder the tasks, mistype the input: see what breaks."),
        ("9 · Process: hierarchical", PAGES[9], "manager_llm versus manager_agent: the manager that checks costs calls and catches short research."),
        ("10 · Review", PAGES[10], "Quiz, homework map, cheat sheet and the session's conclusions."),
    ]
    for title, page, desc in path:
        a, b, c = st.columns([3, 6, 1.4])
        a.markdown("**" + title + "**")
        b.write(desc)
        c.button("Open", key="home_" + page, on_click=goto, args=(page,))
    st.markdown("**Modes:** *Demo* (default) needs no key and installs neither LangChain nor CrewAI: the agents are scripted and the data is pretend. "
                "*Live* (sidebar) lets Gemini write the text on pages 8 and 9. Counting, routing and checking are always done by code.")
    st.caption("About the 'mini CrewAI' on pages 7 to 9: it is a small model written for this lab to show what a framework does for you. It is not the CrewAI library, "
               "and the real library has more options and may behave differently in details. The notebook's real code is shown on each page.")


# ----------------------------------------------------------------------------------------
# Page 1: one agent in LangChain
# ----------------------------------------------------------------------------------------
def page_react():
    st.header("1 · One agent in LangChain: tool + memory")
    story("Before building a team, build one good member. Prasad's support agent must answer 'what's the latest...' questions (it needs a search tool) and follow-ups like 'and the visa fee there?' (it needs memory).")
    plain("LangChain gives you the parts: a model wrapper, tools, memory, and a way to wire them into an agent. Each agent in a team is built from the same parts, so learn them on one agent first.")
    st.subheader("The seven steps of the notebook")
    table_show([{"Step": s, "What it does": w, "The line to look for": c} for s, w, c in ce.STEPS_OF_AN_AGENT])
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
    show_code("The notebook's agent, step by step (and a note about versions)", """
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key, temperature=0.3)
search = TavilySearchResults(tavily_api_key=tavily_api_key, max_results=2)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
agent = initialize_agent(tools=[search], llm=llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                         memory=memory, verbose=True)
response = agent.invoke(user_input)

# Version notes to check on YOUR install (LangChain changes quickly):
#  * initialize_agent, AgentType and ConversationBufferMemory are the older 'classic' interfaces. In recent LangChain
#    releases they may live in a separate langchain-classic package or have newer replacements: read the docs for your version.
#  * The classic agent's invoke() normally returns a dictionary whose answer sits under 'output', so print(response['output']).
#    The notebook's print(response.content) can raise an AttributeError for that reason.""")
    st.success("**Takeaway.** One LangChain agent = a model + tools + memory. A team is several of these, plus the coordination you have to write yourself.")
    st.button("Next: the centralized desk →", on_click=goto, args=(PAGES[2],), type="primary")


# ----------------------------------------------------------------------------------------
# Page 2: centralized
# ----------------------------------------------------------------------------------------
def page_central():
    st.header("2 · Centralized support desk")
    story("Prasad's first desk has a central **orchestrator** that receives every message and decides who answers: a Product agent, a Policy agent or a General agent. "
          "Then a customer asks about a package **and** its cancellation in one sentence.")
    plain("Pattern: **manager / worker**. Architecture: **centralized**. The orchestrator is the manager; the three agents are workers. All communication flows through the manager: "
          "customer asks, orchestrator analyses, routes to the right agent, the agent answers back through the orchestrator.")
    dot('digraph { rankdir=LR; ' + NODE + ' Customer [fillcolor="#fff3bf"]; Orchestrator [fillcolor="#ffe3e3"]; Customer -> Orchestrator; '
        'Orchestrator -> Product; Orchestrator -> Policy; Orchestrator -> General; Product -> Orchestrator; Policy -> Orchestrator; General -> Orchestrator; Orchestrator -> Customer; }')
    st.subheader("Lab 2 · Route a question")
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
    st.success("**Takeaway.** Centralized = control and simplicity. The price: one bottleneck, one point of failure, and routing mistakes lose half a mixed question.")
    st.button("Next: decentralized →", on_click=goto, args=(PAGES[3],), type="primary")


# ----------------------------------------------------------------------------------------
# Page 3: decentralized
# ----------------------------------------------------------------------------------------
def page_decentral():
    st.header("3 · Decentralized support desk")
    story("Version two has no orchestrator. A customer can message any agent. Each agent asks: 'Can I answer this? If not, who can?' and forwards the rest to a peer.")
    plain("Pattern: **peer / swarm**. Architecture: **decentralized**. No central controller; agents collaborate through local rules written in their prompts. Flexible and resilient, but harder to predict and debug.")
    dot('digraph { rankdir=LR; ' + NODE + ' Customer [fillcolor="#fff3bf"]; Customer -> General; Product -> Policy [dir=both]; Policy -> General [dir=both]; General -> Product [dir=both]; }')
    st.subheader("Lab 3 · Where does the customer land, and does the question reach an owner?")
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
    st.success("**Takeaway.** Decentralized = resilience and flexibility. The price: more messages, loops, and answers nobody can fully explain. Always set a hop limit and a final fallback.")
    st.button("Next: hybrid →", on_click=goto, args=(PAGES[4],), type="primary")


# ----------------------------------------------------------------------------------------
# Page 4: hybrid
# ----------------------------------------------------------------------------------------
def page_hybrid():
    st.header("4 · Hybrid support desk")
    story("Version three keeps one orchestrator for routing, but lets the agents talk to each other: the Product agent can ask the Policy agent about cancellation without going back to the boss.")
    plain("Pattern: **federated teams / specialists** (also fits a **pipeline**, for example Research → Write → Edit, which needs orchestration to enforce the order). Architecture: **hybrid**. "
          "Coordination comes from the orchestrator; flexibility from peers who can collaborate directly.")
    dot('digraph { rankdir=LR; ' + NODE + ' Customer [fillcolor="#fff3bf"]; Orchestrator [fillcolor="#ffe3e3"]; Customer -> Orchestrator; Orchestrator -> Product; Product -> Policy [dir=both, label="direct"]; Product -> General [dir=both, label="direct"]; Policy -> General [dir=both]; }')
    st.subheader("Lab 4 · Route once, then ask peers")
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
                "Page 5 tests that last claim: it only holds if you build a fallback.")
    show_code("A specialist that can ask a peer", """
def product_agent(query):
    answer = product_llm.invoke(query)                         # the part it owns
    if "cancel" in query.lower():
        answer += " " + policy_agent(query)                    # a direct call to a peer: no orchestrator in between
    return answer""")
    st.success("**Takeaway.** Hybrid = central plan and answer, direct messages in between. The cleverness is in deciding **which** messages may skip the boss.")
    st.button("Next: compare the three →", on_click=goto, args=(PAGES[5],), type="primary")


# ----------------------------------------------------------------------------------------
# Page 5: compare
# ----------------------------------------------------------------------------------------
def page_compare():
    st.header("5 · Three designs, one set of questions")
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
    st.success("**Takeaway.** No architecture wins everywhere. Centralized is simple and fragile at the centre; decentralized is robust at the centre and chatty; hybrid takes the best of both only if you build the fallback.")
    st.button("Next: why a framework? →", on_click=goto, args=(PAGES[6],), type="primary")


# ----------------------------------------------------------------------------------------
# Page 6: framework
# ----------------------------------------------------------------------------------------
def page_framework():
    st.header("6 · Why a framework? LangChain vs CrewAI")
    story("Prasad built three desks by hand. Each time he wrote the routing, the hand-offs and the message log himself. His next project has six agents. He starts to wonder whether somebody already wrote the plumbing.")
    st.markdown("**What LangChain by hand costs you:** manual orchestration of the agents, complicated agent-to-agent communication, and extra code for each custom tool or API. "
                "**What CrewAI is:** an open-source, Python-based framework for orchestrating teams of AI agents. Think of a virtual workplace: the **crew** is the organisation, each **agent** a team member with a role, "
                "the **tasks** are the jobs, and the **process** says how the work runs (step by step, or under a manager).")
    st.subheader("Lab 6 · What does your project need?")
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
    c2.markdown("**A framework costs you**\n- some control: the loop is hidden inside the library\n- harder debugging when something goes wrong inside\n- another dependency whose interfaces change between versions\n- more model calls you did not write (see page 9)")
    st.success("**Takeaway.** Use LangChain's parts for one agent. Reach for a team framework when the coordination code starts to be bigger than the agents themselves.")
    st.button("Next: CrewAI building blocks →", on_click=goto, args=(PAGES[7],), type="primary")


# ----------------------------------------------------------------------------------------
# Page 7: CrewAI blocks
# ----------------------------------------------------------------------------------------
def page_blocks():
    st.header("7 · CrewAI building blocks")
    story("Prasad rewrites his research-and-blog team in CrewAI. He finds he is filling in four kinds of object instead of writing coordination code.")
    df_show([
        {"Block": "Crew", "What it is": "The container and orchestrator: which agents, which tasks, how they are connected", "Analogy": "The publishing house managing the whole project"},
        {"Block": "Agent", "What it is": "An individual AI worker with a role, a persona, and access to tools or memory", "Analogy": "An employee who knows their job"},
        {"Block": "Task", "What it is": "A unit of work: a goal (description), an expected output, and an assigned agent", "Analogy": "A ticket on a Jira board"},
        {"Block": "Process", "What it is": "How tasks run: sequentially, or hierarchically under a manager", "Analogy": "The workflow the team follows"},
    ])
    st.subheader("Lab 7 · Write the crew and watch the code")
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
           "The notebook gives the same example with the same fields. Running real CrewAI also needs `pip install crewai` and a model key; this page only builds the code for you.")
    st.success("**Takeaway.** In CrewAI you describe **what** (roles, tasks, who feeds whom) and the crew handles **how**.")
    st.button("Next: run it, sequentially →", on_click=goto, args=(PAGES[8],), type="primary")


# ----------------------------------------------------------------------------------------
# Page 8: sequential
# ----------------------------------------------------------------------------------------
def page_sequential():
    b = get_backend()
    st.header("8 · Process: sequential")
    plain("Sequential means tasks run in a fixed order, and one task's output feeds the next: research → write → edit. Best when later work depends on earlier results. "
          "In the notebook the writer receives the research through `context=[task1]`.")
    cfg = crew_cfg()
    inputs = crew_inputs()
    st.caption("Using the agents and tasks you wrote on page 7 (or the notebook's defaults if you skipped it). Topic: **%s**" % inputs["topic"])
    st.subheader("Lab 8 · kickoff() and break it")
    c1, c2, c3 = st.columns(3)
    use_ctx = c1.checkbox("Task 2 uses context=[task1]", True, key="s6_ctx")
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
        else:
            st.success("The writer built its summary from the researcher's bullets.")
        badge(b)
        show_log(r["log"])
    notice("Two of the three mistakes are caught by the (mini) framework before any model is called: a missing input, and a task that needs another task's output before it exists. "
           "The third, removing the context, is silent: nothing errors, the answer is just worse. Silent failures are the dangerous ones.")
    st.markdown("**Workflow summary:** the crew receives a topic → the research agent collects 3 to 5 facts → the writer turns them into a blog summary → the crew manages the dependency and runs them in order.")
    show_code("The notebook's sequential crew", """
crew = Crew(agents=[research_agent, writer_agent], tasks=[task1, task2], process=Process.sequential, verbose=True)
crew.kickoff(inputs={"topic": "The future of electrical vehicles"})""")
    st.success("**Takeaway.** Sequential is the pipeline pattern with the plumbing done for you. Check **what flows between tasks**, not just that every task ran.")
    st.button("Next: add a manager →", on_click=goto, args=(PAGES[9],), type="primary")


# ----------------------------------------------------------------------------------------
# Page 9: hierarchical
# ----------------------------------------------------------------------------------------
def page_hierarchical():
    b = get_backend()
    st.header("9 · Process: hierarchical (a manager)")
    plain("In a hierarchical process a manager delegates tasks to the agents and checks their work: CrewAI's version of the **manager / worker** pattern. "
          "You must supply either a **manager_llm** (a model acting as manager, the lighter way) or a **manager_agent** (a full agent with a role, instructions and tools).")
    cfg = crew_cfg()
    inputs = crew_inputs()
    st.subheader("Lab 9 · The manager that checks")
    c1, c2, c3 = st.columns(3)
    kind = c1.radio("Manager", ["manager_llm (lighter)", "manager_agent (role + tools)", "No manager"], key="s6_mgr")
    tools = c2.checkbox("Give the manager_agent a check tool", True, key="s6_mgrtools", disabled=not kind.startswith("manager_agent"),
                        help="In the notebook the manager agent has tools=[search_tool, code_executor] 'to verify work'.")
    sloppy = c3.checkbox("The researcher is sloppy (returns only 2 facts)", True, key="s6_sloppy")
    predict("the task asks for '3-5 facts' but the researcher returns 2. Which manager notices?")
    manager = None if kind.startswith("No") else ({"kind": "llm"} if kind.startswith("manager_llm") else {"kind": "agent", "has_tools": tools})
    r = ce.run_crew(cfg, inputs, "hierarchical", manager, sloppy, True, live_backend())
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
    st.markdown("**manager_llm or manager_agent?** An LLM manager is quick to set up but has no instructions of its own and no tools. An agent manager can be given a role, rules and tools to verify work: more power, more to maintain.")
    st.markdown("**The CrewAI code for this crew**")
    st.code(ce.crew_code(cfg, inputs, "hierarchical", manager or {"kind": "llm"}), language="python")
    st.caption("In this demo the 'check' is a simple rule (count the bullets). In real CrewAI the manager is itself a model, which can miss problems or raise false alarms.")
    st.success("**Takeaway.** A manager only helps if it can **check** something. A manager that accepts everything is just extra calls.")
    st.button("Next: review →", on_click=goto, args=(PAGES[10],), type="primary")


# ----------------------------------------------------------------------------------------
# Page 10: review
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
]


def page_review():
    st.header("10 · Review, quiz & cheat sheet")
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
        {"#": "HW1", "Level": "Easy", "Task": "Add a 'reschedule' question to the desk: which agent owns it, and do the three designs still answer it?", "Try it": "Pages 2 to 5 (the policy facts include a reschedule rule)"},
        {"#": "HW2", "Level": "Medium", "Task": "Make the decentralized desk loop-proof: what do you change besides the hop limit?", "Try it": "Page 3 (General agent forwards + hop limit)"},
        {"#": "HW3", "Level": "Medium", "Task": "Build the fallback that lets the hybrid desk survive a dead orchestrator. What new risk does it add?", "Try it": "Page 5 (untick the fallback)"},
        {"#": "HW4", "Level": "Easy", "Task": "Rewrite the crew for a trip planner: a researcher and an itinerary writer", "Try it": "Page 7 (edit roles, goals, tasks)"},
        {"#": "HW5", "Level": "Medium", "Task": "Explain why removing context=[task1] causes no error, and how you would detect it", "Try it": "Page 8"},
        {"#": "HW6", "Level": "Hard", "Task": "Add a reviewer task and a manager_agent with a check tool: where does the extra call count come from?", "Try it": "Page 9 (cost table)"},
    ])
    st.subheader("⚡ Quick reference")
    table_show([
        {"Idea": "Architecture", "One line": "How the system is structured: centralized, decentralized, hybrid"},
        {"Idea": "Design pattern", "One line": "How agents interact: manager/worker, peer/swarm, pipeline, federated teams"},
        {"Idea": "Centralized desk", "One line": "Orchestrator routes; simple and easy to debug; single point of failure; mixed questions need splitting"},
        {"Idea": "Decentralized desk", "One line": "Peers forward; resilient; more chatter; loops need a hop limit and a fallback"},
        {"Idea": "Hybrid desk", "One line": "Route once, then peers ask each other; efficient; survives the boss only if you build the fallback"},
        {"Idea": "LangChain", "One line": "The parts: models, tools, memory, chains. You do the coordination."},
        {"Idea": "CrewAI", "One line": "A team framework: Crew, Agent (role, goal, backstory), Task (description, expected_output, context), Process"},
        {"Idea": "Sequential", "One line": "Fixed order; each task's output feeds the next through context"},
        {"Idea": "Hierarchical", "One line": "A manager delegates and (if it has a check) verifies; needs manager_llm or manager_agent"},
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
            st.sidebar.success("Live: connected to Gemini (pages 8 and 9 use it)")
        else:
            st.sidebar.warning(b.error or "Paste a key to go live. Demo mode meanwhile.")
    else:
        st.sidebar.info("Demo mode: scripted agents, pretend data.")


def main():
    sidebar()
    page = st.session_state.get("nav", PAGES[0])
    {PAGES[0]: page_home, PAGES[1]: page_react, PAGES[2]: page_central, PAGES[3]: page_decentral, PAGES[4]: page_hybrid,
     PAGES[5]: page_compare, PAGES[6]: page_framework, PAGES[7]: page_blocks, PAGES[8]: page_sequential, PAGES[9]: page_hierarchical,
     PAGES[10]: page_review}[page]()


main()
