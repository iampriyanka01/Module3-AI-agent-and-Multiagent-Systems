"""
Multi-Agent Systems Lab: an interactive companion to "Agents, Session 5: Introduction to Multi-Agent Systems (MAS)".

Run:   streamlit run app.py
Works with NO API key (Demo mode: scripted agents and pretend data). Add a Gemini key for Live mode on the labs marked live-capable.
"""
import inspect
import os

import pandas as pd
import streamlit as st

import mas_engine as me

st.set_page_config(page_title="MAS Lab: multi-agent systems", page_icon="🤝", layout="wide")

PAGES = [
    "🏠 Start here",
    "1 · One agent, too many jobs",
    "2 · Why a team, and when not",
    "3 · Building blocks and the three C's",
    "4 · Centralized: one boss",
    "5 · Decentralized: no boss",
    "6 · Hybrid: a plan and free hands",
    "7 · Choosing an architecture",
    "8 · Pattern: manager / worker",
    "9 · Pattern: peer / swarm",
    "10 · Pattern: federated teams",
    "11 · Pattern: sequential pipeline",
    "12 · What a team costs",
    "13 · Review, quiz & cheat sheet",
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


def get_backend() -> me.Backend:
    mode = st.session_state.get("mode", "Demo (no API key)")
    key = st.session_state.get("api_key", "") if mode.startswith("Live") else ""
    cfg = (key, st.session_state.get("chat_model", "gemini-3.5-flash-lite"), st.session_state.get("rate_limit", True))
    if st.session_state.get("_cfg") != cfg or "backend" not in st.session_state:
        st.session_state.backend = me.Backend(*cfg)
        st.session_state._cfg = cfg
    return st.session_state.backend


def goto(page: str):
    st.session_state.nav = page


def badge(b: me.Backend):
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


def show_code(title: str, code: str):
    with st.expander("🔍 " + title):
        st.code(code.strip("\n"), language="python")


def dot(src: str):
    st.graphviz_chart(src)


def record(label: str, calls: int):
    """Remember how many model calls a design used, for the cost table on page 12."""
    st.session_state.setdefault("s5_ledger", {})[label] = calls


def show_log(log: me.Log, title: str = "Message log (who said what to whom)"):
    st.markdown("**" + title + "**")
    df_show(log.rows())


def metric_row(items):
    cols = st.columns(len(items))
    for c, (label, value) in zip(cols, items):
        c.metric(label, value)


NODE = 'node [shape=box, style="rounded,filled", fillcolor="#eef2ff", fontname="Helvetica"];'


# ----------------------------------------------------------------------------------------
# Page 0: Start
# ----------------------------------------------------------------------------------------
def page_home():
    st.title("🤝 MAS Lab: when one agent is not enough")
    st.caption("An interactive companion to *Agents, Session 5: Introduction to Multi-Agent Systems (MAS)*.")
    story("One agent, too many jobs. DilSeTrips' concierge now has 12 tools and a 3,000-word system prompt. Then a couple writes: "
          "\"We're getting married in February! Plan our honeymoon in Bali: visas from India, travel insurance, a budget under Rs 2 lakh, and a 6-day vegetarian itinerary.\" "
          "The answer is fluent and covers everything, a little. Prasad cannot be the developer, the accountant and the lawyer at once. Neither can one agent.")
    st.success("**Two questions to hold on to:** who decides, and who talks to whom? Architectures (centralized, decentralized, hybrid) answer the first. "
               "Design patterns (manager/worker, peer/swarm, federated teams, pipeline) answer the second. And every extra agent costs messages and model calls: a team is never free.")
    dot('digraph { rankdir=LR; ' + NODE + ' User [fillcolor="#fff3bf"]; Coordinator [fillcolor="#ffe3e3"]; '
        'User -> Coordinator; Coordinator -> Visa; Coordinator -> Insurance; Coordinator -> Itinerary; Visa -> Coordinator; Insurance -> Coordinator; Itinerary -> Coordinator; }')
    st.subheader("Your path")
    path = [
        ("1 · One agent, too many jobs", PAGES[1], "A generalist and a team of specialists answer the same request; one of them fails."),
        ("2 · Why a team, and when not", PAGES[2], "Complexity, distribution, dynamics. Sort six scenarios into single agent or MAS."),
        ("3 · Building blocks and the three C's", PAGES[3], "Agents, environment, perception, action, reasoning, learning; messages as data."),
        ("4 · Centralized", PAGES[4], "A coordinator routes everything. Switch it off."),
        ("5 · Decentralized", PAGES[5], "Peers decide locally from a stale status board."),
        ("6 · Hybrid", PAGES[6], "Central plan, peer-to-peer middle, central answer. The refund is arithmetic in code."),
        ("7 · Choosing an architecture", PAGES[7], "Side-by-side table and four scenarios to decide."),
        ("8 · Manager / worker", PAGES[8], "A manager that forgets a requirement, and the checks that catch it."),
        ("9 · Peer / swarm", PAGES[9], "Three equals share a board. Races, locks, priorities."),
        ("10 · Federated teams", PAGES[10], "A shared report format makes integration plain code."),
        ("11 · Sequential pipeline", PAGES[11], "One wrong number flows through every stage. Where does a checker earn its cost?"),
        ("12 · What a team costs", PAGES[12], "Model calls per design, and real-world MAS."),
        ("13 · Review", PAGES[13], "Quiz, homework map, cheat sheet."),
    ]
    for title, page, desc in path:
        a, b, c = st.columns([3, 6, 1.4])
        a.markdown("**" + title + "**")
        b.write(desc)
        c.button("Open", key="home_" + page, on_click=goto, args=(page,))
    st.markdown("**Modes:** *Demo* (default) needs no key: scripted agents and pretend data. *Live* (sidebar) asks Gemini to write the text on the labs marked 🔴 live-capable "
                "(pages 1, 6 and 8). Counting, routing, refunds and budgets are always done by code.")


# ----------------------------------------------------------------------------------------
# Page 1: One agent, too many jobs
# ----------------------------------------------------------------------------------------
def page_one_agent():
    b = get_backend()
    st.header("1 · One agent, too many jobs")
    story("A single concierge answers the whole honeymoon request. Visa rules get one line, insurance gets 'consider it', and the budget never adds up.")
    st.markdown("A single agent has four parts: **perception** (how it receives input), **memory**, **planning**, and **tools or actions**. "
                "That is enough for a contained job such as a to-do assistant. But it runs into four limits:")
    df_show([
        {"Limit": "Knowledge overload", "What happens": "One agent trying to master medicine, finance and travel becomes shallow or error-prone"},
        {"Limit": "Performance bottleneck", "What happens": "One reasoning cycle slows down as tasks multiply"},
        {"Limit": "No specialisation", "What happens": "It cannot be tuned for each domain"},
        {"Limit": "Failure fragility", "What happens": "If that one agent breaks or errs, everything fails"},
    ])
    st.markdown("**The request:** " + me.HONEYMOON_REQUEST)

    st.subheader("Lab 1 · How much does each answer say about each topic?")
    predict("the generalist writes about 250 words across four topics. How many words does visas get?")
    depth = me.depth_by_topic()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**🤖 One generalist (whole request)**")
        for t in me.TOPICS:
            st.markdown("**%s:** %s" % (t.title(), me.GENERALIST[t]))
    with c2:
        st.markdown("**🧑‍⚕️ Four specialists (one topic each)**")
        for t in me.TOPICS:
            st.markdown("**%s:** %s" % (t.title(), me.SPECIALIST[t]))
    chart = pd.DataFrame({"topic": me.TOPICS * 2, "words": [depth["generalist"][t] for t in me.TOPICS] + [depth["specialists"][t] for t in me.TOPICS],
                          "who": ["generalist"] * 4 + ["specialists"] * 4})
    try:
        import altair as alt
        st.altair_chart(alt.Chart(chart).mark_bar().encode(x=alt.X("topic:N", title="topic (X axis)"), y=alt.Y("words:Q", title="words spent on it (Y axis)"),
                                                           xOffset="who:N", color=alt.Color("who:N", legend=alt.Legend(orient="bottom", title=None))).properties(height=260), **STRETCH)
    except Exception:
        st.bar_chart(chart.pivot(index="topic", columns="who", values="words"))
    st.caption("**How to read it.** X axis: the four parts of the request. Y axis: how many words the answer spends on that topic. The generalist spreads a small budget thinly; each specialist spends its whole answer on one topic.")
    plain("Both 'agents' are the **same model**. The specialist is not smarter: it has a narrower role, a focused prompt and all of its answer budget on one problem. "
          "In real systems specialists also get **different tools and data** (a visa-rules database, an insurance quote API), which is where specialisation pays most. "
          "And neither answer is checked against an official source: specialisation improves **focus, not truth**.")
    if b.live and st.button("🔴 Ask Gemini live: generalist vs visa specialist", key="s5_live1"):
        gen = b.ask("You are DilSeTrips, a travel concierge that handles every kind of request. Answer in under 250 words.", me.HONEYMOON_REQUEST, me.GENERALIST["visa"])
        spec = b.ask("You are a visa and travel-documents specialist for Indian travellers. You only handle visas and documents. Answer in under 150 words, and say what to verify on official sites.",
                     me.HONEYMOON_REQUEST, me.SPECIALIST["visa"])
        st.session_state["s5_live1_out"] = (gen, spec)
    if st.session_state.get("s5_live1_out") and b.live:
        gen, spec = st.session_state["s5_live1_out"]
        a, c = st.columns(2)
        a.markdown("**Generalist (live)**")
        a.write(gen)
        c.markdown("**Visa specialist (live)**")
        c.write(spec)
        badge(b)

    st.subheader("Lab 2 · What happens when one agent fails?")
    fail = st.radio("Which agent breaks?", ["Nobody", "The generalist", "The visa specialist"], horizontal=True, key="s5_fail")
    gen_down = fail == "The generalist"
    spec_down = "visa" if fail == "The visa specialist" else ""
    d = me.depth_by_topic(gen_down, spec_down)
    gen_cov = sum(1 for t in me.TOPICS if d["generalist"][t] > 0)
    team_cov = sum(1 for t in me.TOPICS if d["specialists"][t] > 0)
    metric_row([("Topics answered by the generalist", "%d of 4" % gen_cov), ("Topics answered by the team", "%d of 4" % team_cov)])
    if fail == "The generalist":
        st.error("One broken agent, zero answers: that is **failure fragility**.")
    elif fail == "The visa specialist":
        st.success("The team still answers insurance, budget and itinerary. Visas need a second try, but the other three parts are not lost.")
    else:
        st.caption("Pick a failure above.")
    show_code("How depth is measured", """
def word_count(text): return len(re.findall(r"\\w+", text))
depth = {topic: word_count(answer[topic]) for topic in topics}     # words spent on each topic""")
    st.success("**Takeaway.** One agent for everything is shallow and fragile. Several focused agents are deeper and degrade gracefully, if you can afford to coordinate them. Page 2 asks when you should.")
    st.button("Next: why a team, and when not →", on_click=goto, args=(PAGES[2],), type="primary")


# ----------------------------------------------------------------------------------------
# Page 2: Why a team, and when not
# ----------------------------------------------------------------------------------------
def page_why():
    st.header("2 · Why a team, and when not")
    plain("A **multi-agent system (MAS)** is several autonomous agents, each with its own knowledge, abilities and decisions, that collaborate or compete. "
          "The behaviour of the whole **emerges** from their interactions. Think of a city: no central brain, yet drivers, traffic lights and navigation apps keep it moving.")
    df_show([
        {"Reason": "Complexity and scale", "The problem": "Too big for one agent's compute, memory and foresight", "Example": "Global supply chains: port, trucks, weather, one coordinator"},
        {"Reason": "Distribution", "The problem": "Data and decisions are spread over places and owners", "Example": "A smart power grid: no control room sees every node in real time"},
        {"Reason": "Dynamics", "The problem": "The world changes faster than a central model updates", "Example": "City traffic: accidents, roadworks and festival crowds"},
    ])
    st.subheader("Lab · single agent or MAS?")
    st.caption("For each scenario, tick which of the three properties it has. Two or more is the notebook's rule of thumb for a team.")
    h = st.columns([4, 1, 1, 1, 2])
    for col, text in zip(h, ["Scenario", "Complex?", "Distributed?", "Dynamic?", "Your verdict"]):
        col.markdown("**" + text + "**")
    agree = 0
    for i, s in enumerate(me.SCENARIOS):
        r = st.columns([4, 1, 1, 1, 2])
        r[0].write(s["name"])
        flags = [r[1].checkbox("c%d" % i, key="s5_sc_%d_c" % i, label_visibility="collapsed"),
                 r[2].checkbox("d%d" % i, key="s5_sc_%d_d" % i, label_visibility="collapsed"),
                 r[3].checkbox("y%d" % i, key="s5_sc_%d_y" % i, label_visibility="collapsed")]
        mine = me.mas_verdict(*flags)
        truth = me.mas_verdict(s["complex"], s["distributed"], s["dynamic"])
        r[4].write(mine)
        agree += mine == truth
    if st.toggle("Show the textbook answers", key="s5_sc_reveal"):
        df_show([{"Scenario": s["name"], "Complex": s["complex"], "Distributed": s["distributed"], "Dynamic": s["dynamic"],
                  "Textbook verdict": me.mas_verdict(s["complex"], s["distributed"], s["dynamic"]), "Why": s["why"]} for s in me.SCENARIOS])
        st.metric("Your verdicts that match", "%d / %d" % (agree, len(me.SCENARIOS)))
    st.subheader("Single agent versus MAS")
    df_show([
        {"Aspect": "Problem size", "Single agent": "Small, manageable", "Multi-agent system": "Large, complex, beyond one agent"},
        {"Aspect": "Knowledge / data", "Single agent": "Centralised", "Multi-agent system": "Distributed across many sources"},
        {"Aspect": "Environment", "Single agent": "Stable, predictable", "Multi-agent system": "Dynamic, continuously changing"},
        {"Aspect": "Strengths", "Single agent": "Simple, efficient for limited tasks", "Multi-agent system": "Scalable, flexible, robust"},
        {"Aspect": "Limitations", "Single agent": "Struggles with scale and change", "Multi-agent system": "Harder to design; more resources and model calls"},
        {"Aspect": "Example", "Single agent": "One chatbot answering FAQs", "Multi-agent system": "Several LLMs, each in a specialised domain"},
    ])
    st.success("**Takeaway.** Coordination, communication and conflict resolution cost something. Simple, stable and centralised: one agent. Large, distributed and ever-changing: call in a team.")
    st.button("Next: building blocks and the three C's →", on_click=goto, args=(PAGES[3],), type="primary")


# ----------------------------------------------------------------------------------------
# Page 3: Building blocks and the three C's
# ----------------------------------------------------------------------------------------
def page_blocks():
    st.header("3 · Building blocks and the three C's")
    st.markdown("Back to the airport: 50 cleaning robots, each cleaning a zone, sharing where the spills are.")
    df_show([
        {"Building block": "Agents", "In the airport": "Each cleaning robot", "In an LLM system": "A support bot, a visa specialist, a coordinator"},
        {"Building block": "Environment", "In the airport": "Terminals, gates, passengers", "In an LLM system": "A database, external APIs, **and the other agents**"},
        {"Building block": "Perception", "In the airport": "Camera sees a spill", "In an LLM system": "A user message, a tool result"},
        {"Building block": "Actions / tools", "In the airport": "Move, clean, alert others", "In an LLM system": "Call a tool, send a message, write an answer"},
        {"Building block": "Reasoning", "In the airport": "Which spill first?", "In an LLM system": "Understand, evaluate options, decide (local and global goals)"},
        {"Building block": "Learning", "In the airport": "Clean busy areas more efficiently", "In an LLM system": "Usually writing lessons to memory (Session 3), not changing the model"},
    ])
    with st.expander("Four kinds of learning in a MAS"):
        df_show([
            {"Kind": "Reinforcement (trial and error)", "Example": "Traffic lights adjust green time from congestion"},
            {"Kind": "Supervised (from examples)", "Example": "Navigation apps predict congestion from past data"},
            {"Kind": "Unsupervised (patterns)", "Example": "Cars notice zones with frequent stops"},
            {"Kind": "Collaborative", "Example": "Cars share accident-prone zones; lights share timing strategies"},
        ])
    st.subheader("The three C's")
    dot('digraph { rankdir=LR; ' + NODE + ' Communication [fillcolor="#d0ebff"]; Coordination [fillcolor="#d3f9d8"]; Collaboration [fillcolor="#fff3bf"]; Communication -> Coordination -> Collaboration; }')
    df_show([
        {"C": "Communication", "Meaning": "Exchanging information: what, between whom, in what format", "A failure looks like": "A message nobody can route"},
        {"C": "Coordination", "Meaning": "Organising who does what and when", "A failure looks like": "Two bots answer the same customer"},
        {"C": "Collaboration", "Meaning": "Combining partial results toward one goal", "A failure looks like": "Four specialists' findings that never become one plan"},
    ])
    st.subheader("Lab A · Communication: a message is data")
    presets = {
        "Complete JSON message": '{"from": "booking_agent", "to": "cancellation_agent", "kind": "question", "task_id": "H1", "content": "How much refund credit can I use?"}',
        "A field is missing": '{"from": "booking_agent", "kind": "question", "content": "How much refund credit can I use?"}',
        "Free text": "hey, can you tell me how much refund credit I can use?",
    }
    pick = st.selectbox("Pick a message", list(presets), key="s5_msg_pick")
    text = st.text_area("Edit it", presets[pick], height=110, key="s5_msg_text_" + pick)
    res = me.validate_message(text)
    (st.success if res["ok"] else st.error)(res["why"])
    st.caption("Agents need a shared language. For LLM agents that is usually a structured message (who, to whom, which task, what) so that other agents and code can read it reliably.")
    st.subheader("Lab B · Coordination: two bots, one queue")
    claims = st.checkbox("Each bot posts an 'I take it' claim before answering", key="s5_claims")
    d = me.duplicate_work(claims)
    metric_row([("Queries", d["queries"]), ("Answers sent", d["answers_sent"]), ("Duplicate answers", d["duplicates"]), ("Claim messages", d["claim_messages"])])
    if d["duplicates"]:
        st.error("Both bots saw each query at the same moment and both answered: confused customers and wasted calls.")
    else:
        st.success("No duplicates. The price: one extra message per query. Coordination is never free.")
    show_code("The message format used in every lab", """
message = {"from": sender, "to": recipient, "kind": kind, "task_id": task_id, "content": content}
message_log.append(message)          # one list = a record of who said what to whom (needed for debugging)""")
    st.success("**Takeaway.** The blocks are individual; the **interactions** are what make a MAS. Communication makes coordination possible, and coordination makes collaboration possible.")
    st.button("Next: centralized →", on_click=goto, args=(PAGES[4],), type="primary")


# ----------------------------------------------------------------------------------------
# Page 4: Centralized
# ----------------------------------------------------------------------------------------
def page_centralized():
    st.header("4 · Centralized: one boss")
    story("DilSeTrips' support desk: a coordinator receives every query and decides which specialist chatbot answers it.")
    dot('digraph { ' + NODE + ' Coordinator [fillcolor="#ffe3e3"]; Customer -> Coordinator; Coordinator -> bookings_bot; Coordinator -> refunds_bot; Coordinator -> advice_bot; }')
    st.markdown("**How it works:** agents report to the controller → the controller decides using its global view → it sends instructions back → agents execute. "
                "Like an orchestra with one conductor.")
    c1, c2 = st.columns(2)
    c1.markdown("**Strengths**\n- Simple to coordinate, monitor and debug (one place to look)\n- A global view lets it resolve conflicts\n- Uniform rules when priorities are critical")
    c2.markdown("**Limits**\n- Single point of failure\n- The controller is a bottleneck as load grows\n- Reduced autonomy for the specialists\n- Heavy message traffic and delayed decisions")
    st.subheader("Lab 1 · Switch the coordinator off")
    up = st.checkbox("The coordinator is up", True, key="s5_c_up")
    fallback = False
    if not up:
        fallback = st.checkbox("HW1 · Fallback: specialists answer in turn when the coordinator is down", key="s5_c_fb")
    predict("the three specialists are all healthy. With the coordinator down, how many queries get answered?")
    r = me.centralized_desk(up, fallback)
    metric_row([("Queries answered", "%d of %d" % (r["answered"], r["total"])), ("Misrouted", r["misrouted"]), ("Model calls", r["calls"])])
    if up:
        st.success("Everything went through one coordinator, so the log is easy to follow and every query reached the right specialist.")
    elif fallback:
        st.warning("Every query got an answer, but the fallback has no one deciding **who should answer**: some go to the wrong specialist. And if two bots take the same query, you get duplicates (page 3, lab B). "
                   "A fallback needs a rule and a claim message, which is the start of decentralization.")
    else:
        st.error("None answered. That is the **single point of failure**: no specialist knows which query is theirs, even though all three are healthy.")
    record("3.1 Centralized desk (coordinator up)" if up else "3.1 Centralized desk (coordinator down)", r["calls"])
    show_log(r["log"])
    st.subheader("Lab 2 · The bottleneck as load grows")
    route = st.slider("Queries the coordinator can route per minute", 2, 20, 6, key="s5_c_route")
    xs = [10, 25, 50, 100, 150, 200]
    rows = [{"queries": n, "centralized": me.desk_load(n, True, route)["minutes_to_clear"], "decentralized": me.desk_load(n, False)["minutes_to_clear"]} for n in xs]
    st.line_chart(pd.DataFrame(rows).set_index("queries"))
    st.caption("**How to read it.** X axis: how many queries arrive together. Y axis: minutes until the last one is answered (a toy model). With a slow coordinator, the centralized line climbs faster than the decentralized one: routing becomes the bottleneck, not answering.")
    show_code("The coordinator is just a router with one job", """
def centralized_desk(coordinator_up):
    if not coordinator_up:
        return 0                                  # nobody routes, nobody answers
    for query in queries:
        bot = coordinator_route(query)            # in the notebook: an LLM call with structured output
        answer = ask(specialist_roles[bot], query)""")
    st.success("**Takeaway.** Centralized is the easiest to control and debug, and the easiest to break. Remove the boss and the question becomes: can the agents decide for themselves?")
    st.button("Next: decentralized →", on_click=goto, args=(PAGES[5],), type="primary")


# ----------------------------------------------------------------------------------------
# Page 5: Decentralized
# ----------------------------------------------------------------------------------------
def page_decentralized():
    st.header("5 · Decentralized: no boss")
    plain("Like contributors to an open-source project: no manager assigns tasks. Each agent decides from **local observations** and messages from peers, and coordinates directly. "
          "Robust (one agent failing does not stop the others) and scalable, but with its own problems.")
    dot('digraph { ' + NODE + ' edge [dir=both]; bot_A -> bot_B; bot_B -> bot_C; bot_C -> bot_D; bot_D -> bot_A; bot_A -> bot_C; bot_B -> bot_D; Customer [fillcolor="#fff3bf"]; Customer -> bot_B [dir=forward]; }')
    st.markdown("Each bot loops: **observe** (new messages, what peers seem to be doing) → **decide locally** (handle it, or hand it to a peer) → **tell peers** → **act** → **learn** → repeat.")
    st.subheader("Lab 1 · A VIP query meets a stale status board")
    st.markdown("Bot B has 4 queries and receives a VIP query. It only knows the **status board**, which peers update when they can. Bot C's real queue is 3 but the board says 0.")
    board_kind = st.radio("The board is…", ["Stale (C shows 0, really 3)", "Accurate"], horizontal=True, key="s5_board")
    board = me.STALE_BOARD if board_kind.startswith("Stale") else me.ACTUAL
    df_show([{"bot": k, "real queue": me.ACTUAL[k], "board says": board[k]} for k in me.ACTUAL])
    predict("how many hand-overs and model calls will this take with the stale board? With the accurate one?")
    r = me.handover(me.ACTUAL, board)
    metric_row([("Attempts", r["attempts"]), ("Handled by", r["handled_by"]), ("Model calls", r["calls"]), ("Messages", len(r["log"].messages))])
    record("3.2 Decentralized hand-over (%s board)" % ("stale" if board_kind.startswith("Stale") else "accurate"), r["calls"])
    if r["refused"]:
        st.warning("Bot B trusted the board, handed the VIP query to **%s**, was refused, and had to decide again. Nobody failed, but the stale board cost an extra round of messages and an extra model call, "
                   "and the VIP waited longer." % ", ".join(r["refused"]))
    else:
        st.success("With an accurate board, one decision and one hand-over. Accurate knowledge costs messages to keep up to date: see lab 2.")
    show_log(r["log"])
    st.subheader("Lab 2 · HW2 · The cost of keeping everyone informed")
    n = st.slider("Number of agents", 2, 100, 20, key="s5_n")
    cost = me.broadcast_cost(n)
    metric_row([(k, "{:,}".format(v)) for k, v in cost.items()])
    xs = list(range(2, 101, 2))
    st.line_chart(pd.DataFrame({"everyone tells everyone": [me.broadcast_cost(x)["everyone tells everyone"] for x in xs],
                                "shared board": [me.broadcast_cost(x)["shared board (post once)"] for x in xs]}, index=xs))
    st.caption("**How to read it.** X axis: number of agents. Y axis: messages caused by ONE change in ONE agent's queue. 'Everyone tells everyone' is n × (n−1): it grows with the square of the team. "
               "That is why decentralized systems use a shared board, as the swarm will, or a hybrid coordinator.")
    st.subheader("Lab 3 · How much does staleness cost?")
    stale = st.slider("Boards are out of date by up to this many queries", 0, 4, 2, key="s5_stale")
    rows = [{"staleness": s, "4 bots": me.handover_study(4, s)["avg_attempts"], "20 bots": me.handover_study(20, s)["avg_attempts"]} for s in range(0, 5)]
    st.line_chart(pd.DataFrame(rows).set_index("staleness"))
    one = me.handover_study(4, stale)
    metric_row([("Avg hand-overs per VIP query (4 bots)", one["avg_attempts"]), ("Avg messages per VIP query", one["avg_messages"])])
    st.caption("**How to read it.** X axis: how wrong the board is. Y axis: average hand-over attempts per urgent query over 300 random situations. At zero staleness it is exactly 1; every bit of staleness adds hops. "
               "(Toy simulation: an average is not a guarantee for any single query.)")
    st.markdown("**More limits of decentralized systems:**")
    df_show([
        {"Limit": "Communication overhead", "Example": "20 chatbots exchanging 'who is free?' messages congest the network"},
        {"Limit": "Inconsistent global knowledge", "Example": "A thinks B is free but B is overloaded"},
        {"Limit": "Coordination complexity", "Example": "Two chatbots answer the same customer (page 3, lab B)"},
        {"Limit": "Hard to debug", "Example": "A delay across several hand-overs needs a log from every agent"},
        {"Limit": "Slower learning", "Example": "Each agent learns from partial experience"},
    ])
    show_code("The local decision, with a hard attempt limit", """
while not handled and attempts < 3:                    # agents must never loop forever
    decision = ask_structured(delegation_schema, "You are bot_B... decide locally", prompt_with_status_board)
    if decision["delegate_to"] in asked_already: continue      # never ask a peer that already refused""")
    st.success("**Takeaway.** Decentralized removes the single point of failure and pays in messages and in out-of-date knowledge. Can we keep the strengths of both? That is hybrid.")
    st.button("Next: hybrid →", on_click=goto, args=(PAGES[6],), type="primary")


# ----------------------------------------------------------------------------------------
# Page 6: Hybrid
# ----------------------------------------------------------------------------------------
def page_hybrid():
    b = get_backend()
    st.header("6 · Hybrid: a plan and free hands")
    plain("A **hybrid** architecture combines central oversight (global objectives, conflict resolution, monitoring) with autonomous agents that decide locally and talk to each other. "
          "The coordinator does **not** micromanage every agent, which avoids the bottleneck.")
    dot('digraph { ' + NODE + ' Manager [fillcolor="#ffe3e3"]; User -> Manager; Manager -> cancellation_agent; Manager -> booking_agent; '
        'booking_agent -> cancellation_agent [color="#2f9e44", penwidth=2, label="direct", dir=both]; cancellation_agent -> Manager [style=dashed]; booking_agent -> Manager [style=dashed]; Manager -> User; }')
    st.markdown("**The eight stages** (centralized at both ends, decentralized in the middle):")
    df_show([{"Stage": s, "Who handles it": w, "What happens": d} for s, w, d in me.HYBRID_STAGES])
    st.subheader("Lab · cancel one stay and rebook another using the refund")
    st.markdown("*\"Please cancel our Goa homestay for 14-17 Feb and book something quiet in Coorg instead, using the refund.\"* The booking data is **pretend**, written for this exercise.")
    c1, c2 = st.columns(2)
    fee = c1.slider("Cancellation fee", 0, 50, 20, 5, key="s5_fee", format="%d%%")
    direct = c2.checkbox("Agents talk to each other directly (the hybrid way)", True, key="s5_direct")
    predict("the booking agent needs to know the refund. How many more messages and model calls if it must ask through the manager?")
    r = me.hybrid_rebooking(fee / 100, direct, b if b.live else None)
    m = r["money"]
    st.markdown("**Refund arithmetic, done by code, not by an LLM:** Rs %s paid − %d%% fee (Rs %s) = **Rs %s**" % ("{:,}".format(m["paid"]), fee, "{:,}".format(m["fee"]), "{:,}".format(m["refund"])))
    st.success(r["final"])
    badge(b)
    metric_row([("Model calls", r["calls"]), ("Messages", len(r["log"].messages)), ("Direct agent-to-agent", r["direct_messages"])])
    record("3.3 Hybrid rebooking (%s)" % ("direct messages" if direct else "everything via manager"), r["calls"])
    if direct:
        notice("The one exchange that did **not** go through the manager is marked 'peer to peer'. Untick the box and the manager has to relay it: more messages and more model calls for the same answer.")
    else:
        st.warning("The manager is now relaying every detail between agents: the bottleneck we wanted to avoid.")
    show_log(r["log"])
    st.info("**The refund is arithmetic.** Rs 12,000 minus a 20%% fee is Rs 9,600, every time. LLMs write the sentences; code does the sums. Move the fee slider: the answer changes exactly, never approximately.")
    st.markdown("**Strengths:** balance of control and flexibility · add agents without breaking the system · fault tolerant (the coordinator can reassign) · efficient direct collaboration · "
                "better user experience (one structured answer, not a runaround).")
    st.markdown("**To build one:** define the coordinator → design specialised agents with roles and tools → agree a message protocol (JSON with task ID, sender, content) → implement routing.")
    st.success("**Takeaway.** Hybrid is where most LLM applications land: a central plan and answer, local autonomy and direct messages in between.")
    st.button("Next: choosing an architecture →", on_click=goto, args=(PAGES[7],), type="primary")


# ----------------------------------------------------------------------------------------
# Page 7: Choosing an architecture
# ----------------------------------------------------------------------------------------
def page_choose():
    st.header("7 · Choosing an architecture")
    st.markdown("**The three architectures side by side**")
    df_show(me.ARCH_COMPARE)
    st.subheader("Lab · HW7 · choose a design (and name one risk)")
    st.caption("For each scenario pick an architecture and a pattern, then reveal one defensible answer. More than one design can be reasonable: the point is to justify it and name the risk.")
    archs = ["(choose)", "Centralized", "Decentralized", "Hybrid"]
    patterns = ["(choose)", "Manager / worker", "Peer / swarm", "Federated teams", "Sequential pipeline"]
    right = 0
    for i, s in enumerate(me.DESIGN_SCENARIOS):
        with st.container(border=True):
            st.markdown("**%d. %s**" % (i + 1, s["name"]))
            c1, c2, c3 = st.columns([2, 2, 1])
            a = c1.selectbox("Architecture", archs, key="s5_arch_%d" % i)
            p = c2.selectbox("Pattern", patterns, key="s5_pat_%d" % i)
            show = c3.toggle("Reveal", key="s5_rev_%d" % i)
            if show:
                ok_a, ok_p = a == s["arch"], p == s["pattern"]
                right += ok_a + ok_p
                st.markdown("**One defensible answer:** %s with %s" % (s["arch"], s["pattern"]))
                st.markdown("*Why:* " + s["why"])
                st.markdown("*Risk to name:* " + s["risk"])
                if a != "(choose)" or p != "(choose)":
                    st.caption("Your choice: %s / %s → %s" % (a, p, "matches" if ok_a and ok_p else "differs, so check your reasoning rather than the label"))
    st.success("**Takeaway.** Choose per feature, not once for the whole product: tight control where outcomes must be uniform, autonomy where scale and change dominate.")
    st.button("Next: manager / worker →", on_click=goto, args=(PAGES[8],), type="primary")


# ----------------------------------------------------------------------------------------
# Page 8: Manager / worker
# ----------------------------------------------------------------------------------------
def page_manager():
    st.header("8 · Pattern: manager / worker")
    plain("A **manager** breaks the request into tasks, gives each to a specialised **worker**, and integrates the results. It suits centralized and hybrid architectures. "
          "The weak spot: **the manager's plan is the only map.** A requirement that is not on it simply does not get done.")
    dot('digraph { ' + NODE + ' Manager [fillcolor="#ffe3e3"]; Manager -> visa_worker; Manager -> budget_worker; Manager -> itinerary_worker; visa_worker -> Manager [style=dashed]; budget_worker -> Manager [style=dashed]; itinerary_worker -> Manager [style=dashed]; }')
    c1, c2 = st.columns(2)
    c1.markdown("**Advantages**\n- Clear delegation, no duplicated effort\n- Specialists used well\n- Central oversight, can reassign overloaded workers\n- Simplifies multi-step work")
    c2.markdown("**Limitations**\n- Single point of failure\n- Manager can be a bottleneck\n- Workers have little autonomy\n- Constant reporting costs bandwidth")
    st.subheader("Lab · the manager's plan forgets something")
    plan_name = st.radio("What does the manager's plan look like?", list(me.PLANS), key="s5_plan")
    strict = st.checkbox("HW3 · Stricter check: the manager must say which worker covers each requirement, and code rejects any that has none", key="s5_strict")
    st.markdown("**The tasks the manager wrote**")
    df_show([{"worker": w, "task": t} for w, t in me.PLANS[plan_name]["tasks"]])
    predict("will the check catch the plan that mentions insurance but assigns it to nobody?")
    r = me.manager_worker(plan_name, strict)
    which = "stricter coverage check" if strict else "keyword check (does the WORD appear in some task?)"
    if r["flagged"]:
        st.success("✅ The %s caught it: **%s** has no one assigned." % (which, ", ".join(r["flagged"])))
    elif r["missing_in_answer"]:
        st.error("❌ The %s found nothing wrong, but **%s** is still uncovered. Nobody will do it, and the final answer silently leaves it out." % (which, ", ".join(r["missing_in_answer"])))
    else:
        st.success("Every requirement is covered by a named worker.")
    metric_row([("Model calls", r["calls"]), ("Requirements covered", "%d of 4" % len(r["covered"])), ("Messages", len(r["log"].messages))])
    record("4.1 Manager / worker", r["calls"])
    with st.expander("What the workers reported"):
        for w, text in r["results"].items():
            st.markdown("**%s:** %s" % (w, text))
    show_log(r["log"])
    notice("Pick the third plan. A check on **words** passes because the word 'insurance' appears (in 'leave insurance aside'). A check on **ownership** does not. Check the thing you actually care about.")
    st.caption("Compare with page 1: a generalist used 1 model call; this team used %d. Each section is deeper, and you paid for it." % r["calls"])
    show_code("Two checks", """
# Keyword check: weak (a word in a task is not an owner)
for requirement in requirements:
    if requirement not in all_task_text: print("WARNING: no task mentions", requirement)

# HW3: the manager returns a coverage map, code rejects gaps
uncovered = [r for r in requirements if not plan["coverage"].get(r)]
if uncovered: raise ValueError("Plan leaves requirements uncovered: " + ", ".join(uncovered))""")
    st.success("**Takeaway.** Manager/worker gives clear delegation and oversight, and all of its safety depends on the plan. Validate the plan with code before anything is delegated.")
    st.button("Next: peer / swarm →", on_click=goto, args=(PAGES[9],), type="primary")


# ----------------------------------------------------------------------------------------
# Page 9: Peer / swarm
# ----------------------------------------------------------------------------------------
def page_swarm():
    st.header("9 · Pattern: peer / swarm")
    plain("**All agents are equals**, with no coordinator. Each peer decides for itself; **swarm behaviour** emerges when many peers follow simple local rules. "
          "Peers coordinate through a **shared board**, which is part of the environment.")
    dot('digraph { ' + NODE + ' board [shape=cylinder, fillcolor="#fff3bf"]; peer_1 -> board; peer_2 -> board; peer_3 -> board; board -> peer_1 [style=dashed]; board -> peer_2 [style=dashed]; board -> peer_3 [style=dashed]; }')
    st.markdown("**Two local rules:** (1) claim the first unclaimed section on the board and post **two new insights**, skipping anything already posted. (2) If nothing is left to claim, write the summary from the board. "
                "The sections come from a **pretend** Manali guide.")
    st.subheader("Lab · three peers, a board, and no boss")
    c1, c2, c3 = st.columns(3)
    mode_label = c1.radio("How do the peers run?", ["One at a time (like the notebook)", "Together, no lock (race)", "Together, with a claim lock"], key="s5_swarm_mode")
    peers = c2.slider("Number of peers", 2, 6, 3, key="s5_peers")
    prio = c3.checkbox("HW4 · A local priority rule: sections that matter to a quiet vegetarian couple go first (Eat, See, Do, Get around)", key="s5_prio")
    mode = {"One at a time (like the notebook)": "turns", "Together, no lock (race)": "race", "Together, with a claim lock": "lock"}[mode_label]
    predict("two peers claim at the same instant. What happens to the board and to the model calls?")
    r = me.swarm(mode, peers, prio)
    metric_row([("Rounds", r["rounds"]), ("Model calls", r["calls"]), ("Duplicate insights on the board", r["duplicates"]), ("Wasted calls", r["wasted_calls"])])
    record("4.2 Peer / swarm (%s)" % mode, r["calls"])
    st.markdown("**Order the sections were claimed:** " + " → ".join(r["order"][:8]))
    if mode == "race":
        st.error("Peers read the board at the same moment and claimed the **same** section: duplicate insights and wasted calls. This is the coordination problem of a real concurrent swarm.")
    elif mode == "lock":
        st.success("An atomic claim means nobody gets a section someone else holds, and the work finishes in fewer rounds with no waste.")
    else:
        st.info("One peer at a time keeps things tidy and slow. Real peers run concurrently.")
    if prio:
        st.caption("With the priority rule, 'Eat' and 'See' are claimed before 'Get around'. That is a **local rule**, not a coordinator: each peer applies it itself. If two peers apply it at once, they still need the claim lock.")
    else:
        st.caption("Without a priority rule every section gets equal effort, whatever the couple needs. That is the **no global oversight** limitation in miniature.")
    st.markdown("**The board**")
    df_show([{"#": i + 1, "insight": x} for i, x in enumerate(r["board"])])
    st.success(r["summary"][:300])
    show_log(r["log"])
    c1, c2 = st.columns(2)
    c1.markdown("**Advantages**\n- Robust: if some peers fail, others continue\n- Scales: no central node to overwhelm\n- High autonomy\n- Emergent intelligence from simple rules")
    c2.markdown("**Limitations**\n- No global oversight (uneven results)\n- Unpredictable emergent behaviour\n- Communication overhead\n- Slower to reach agreement")
    show_code("Rule 1 and rule 2 (the whole 'program')", """
while not summary_written and turn < 10:                   # a hard limit on turns
    peer = peers[turn % len(peers)]
    if board["unclaimed"]:
        section = board["unclaimed"].pop(0)                # rule 1: claim (this pop must be atomic in a real swarm)
        board["insights"] += ask(peer_role, section_text + already_on_board)
    else:
        summary = ask(peer_role, "\\n".join(board["insights"]))   # rule 2: write the summary""")
    st.success("**Takeaway.** A swarm can divide work, avoid duplicates and produce a summary with no boss, from two rules and a shared board. It cannot judge what matters unless you put that into a rule.")
    st.button("Next: federated teams →", on_click=goto, args=(PAGES[10],), type="primary")


# ----------------------------------------------------------------------------------------
# Page 10: Federated teams
# ----------------------------------------------------------------------------------------
def page_federated():
    st.header("10 · Pattern: federated teams")
    plain("Agents are **grouped by expertise**. Each team works on its own and collaborates across teams only when a task needs it. Unlike a swarm, agents are not equals; "
          "unlike manager/worker, there may be no single manager. The key to making it work is a **shared report format**.")
    dot('digraph { ' + NODE + ' subgraph cluster_a { label="documents team"; visa_agent; passport_agent; } subgraph cluster_b { label="money team"; flights_agent; insurance_agent; } '
        'money_team [shape=plaintext, fillcolor=white, style=""]; flights_agent -> visa_agent [label="asks across teams", color="#2f9e44", penwidth=2]; }')
    st.subheader("Lab 1 · integrate standard reports with plain code")
    c1, c2 = st.columns(2)
    add_food = c1.checkbox("HW6 · Add a third team (food)", key="s5_food")
    drop_food = False
    if add_food:
        drop_food = c2.checkbox("The money team stops counting food (no overlap)", key="s5_dropfood")
    reports = [me.DOCUMENTS_REPORT, me.MONEY_REPORT]
    if drop_food:
        reports = [me.DOCUMENTS_REPORT, dict(me.MONEY_REPORT, costs=[c for c in me.MONEY_REPORT["costs"] if c["category"] != "food"])]
    if add_food:
        reports = reports + [me.FOOD_REPORT]
    res = me.integrate(reports)
    df_show(res["rows"])
    metric_row([("Total (both people)", "Rs {:,}".format(res["total"])), ("Budget", "Rs {:,}".format(me.BUDGET)), ("Left over", "Rs {:,}".format(res["left"]))])
    st.progress(min(1.0, res["total"] / me.BUDGET))
    if res["overlaps"]:
        for o in res["overlaps"]:
            st.error("⚠️ Overlap: " + o + ". The total is inflated.")
        st.caption("A shared format made the integration **zero extra lines of code** (it loops over a list of reports), but it did not stop two teams costing the same thing. "
                   "A shared format needs clear ownership of each category as well.")
    elif add_food:
        st.success("A third team joined and the integration code did not change. That is what a standard format buys.")
    st.markdown("**Open questions from every team, together** (often the most useful part of a federated report: what still needs a human):")
    for q in res["questions"]:
        st.write("• " + q)
    st.subheader("Lab 2 · the same teams WITHOUT a shared format")
    st.caption("Without a standard, each team writes in its own style and someone (or another LLM call) has to translate.")
    total_naive = 0
    for team, text in me.FREE_TEXT.items():
        found = me.naive_parse(text)
        total_naive += sum(found)
        st.markdown("**%s:** %s" % (team, text))
        st.caption("A simple 'Rs 12,345' parser finds: %s" % (found or "nothing"))
    st.error("The simple parser totals Rs {:,}: it missed '~36k', '1.5k', 'twenty four thousand', '12k' and '4,000' (no 'Rs'). The real total is about Rs 139,500. Integration is where federated systems struggle."
             .format(total_naive))
    show_code("The standard report every team returns", """
team_report_schema = {"team": str, "findings": [str],
                      "costs": [{"item": str, "amount_inr": int, "category": str}],      # category added here to catch overlaps
                      "open_questions": [str]}
total = sum(c["amount_inr"] for report in reports for c in report["costs"])       # plain code, because the format is shared""")
    st.success("**Takeaway.** Federated teams give deep expertise in parallel. Their weak point is integration: force a shared format, and give each topic one owner.")
    st.button("Next: sequential pipeline →", on_click=goto, args=(PAGES[11],), type="primary")


# ----------------------------------------------------------------------------------------
# Page 11: Sequential pipeline
# ----------------------------------------------------------------------------------------
def page_pipeline():
    st.header("11 · Pattern: sequential pipeline")
    plain("Agents form a fixed sequence; each does one sub-task and passes its result on, like a CI/CD pipeline (lint → test → deploy). Ordered, modular and predictable, "
          "but **errors propagate**: a mistake early on flows into every later stage.")
    dot('digraph { rankdir=LR; ' + NODE + ' research -> planner -> budgeter -> writer -> Couple; Checker [fillcolor="#ffe3e3"]; research -> Checker [style=dashed, label="check here?"]; }')
    st.subheader("Lab · a flight price that is far too low")
    c1, c2, c3 = st.columns(3)
    price = c1.slider("Return flight Mumbai-Bali, per person (Rs), from the research stage", 3000, 60000, 4000, 1000, key="s5_price",
                      help="Research output here is pretend tool data. 4,000 is the deliberate error from the notebook.")
    checker = c2.radio("Where is the checker?", ["No checker", "After research", "After the budget"], key="s5_checker")
    ck = {"No checker": "none", "After research": "research", "After the budget": "budget"}[checker]
    minp = c3.slider("Checker: lowest plausible price (Rs)", 5000, 25000, 15000, 1000, key="s5_minp", disabled=(ck != "research"))
    predict("the budgeter adds the numbers faithfully. What does the writer tell the couple with the Rs 4,000 flight and no checker?")
    r = me.pipeline(price, ck, minp)
    metric_row([("Model calls", r["calls"]), ("Halted?", "yes, at " + r["halted_at"] if r["halted"] else "no"), ("Total cost for two", "Rs {:,}".format(r["total"]) if r["total"] else "n/a")])
    record("4.4 Pipeline (%s)" % checker.lower(), r["calls"])
    if r["halted"]:
        st.success("The checker stopped the pipeline before the couple saw a wrong number.")
    else:
        text = r["message"]
        if price < minp:
            st.error("**What the couple receives:** " + text + "  \nEvery agent did its own job correctly, and the system is still wrong, because nothing questioned the input: **error propagation**.")
        else:
            st.success("**What the couple receives:** " + text)
    show_log(r["log"])
    st.markdown("**Where should the checker go?** (HW5) Same bad price, three placements:")
    rows = []
    for label, kind in (("No checker", "none"), ("After research", "research"), ("After the budget", "budget")):
        x = me.pipeline(price, kind, minp)
        rows.append({"Placement": label, "Model calls": x["calls"], "Caught the error?": "yes" if x["halted"] else ("n/a (price is plausible)" if price >= minp else "NO"),
                     "Stopped at": x["halted_at"] or "-", "Wrong message sent?": "yes" if (not x["halted"] and price < minp) else "no"})
    df_show(rows)
    notice("Checking early is cheapest: one call and no wasted work. Checking late still stops the message, but only after the planner and budgeter were paid for. A checker after **every** stage catches more and doubles the calls.")
    st.caption("A checker is itself an LLM in the notebook, and can miss things or raise false alarms. Here it is a simple rule so you can see where it sits. After a halt: ask a human, re-run research, or continue with a warning? That is a product decision.")
    c1, c2 = st.columns(2)
    c1.markdown("**Advantages**\n- Ordered execution\n- Clear dependencies\n- Modular: swap one agent\n- Predictable outcomes")
    c2.markdown("**Limitations**\n- One slow stage delays all\n- Less flexible, no parallelism\n- Errors propagate\n- Needs strict orchestration")
    show_code("A checker stage before anything is built on the input", """
def run_pipeline(first_input, stages, checker=None):
    if checker is not None and checker(first_input) != "ok":
        return None                                       # halt before spending calls on a wrong number
    current = first_input
    for name, role in stages:
        current = ask(role, current)""")
    st.success("**Takeaway.** In a pipeline, put a cheap check where errors **enter**, not just where they show up.")
    st.button("Next: what a team costs →", on_click=goto, args=(PAGES[12],), type="primary")


# ----------------------------------------------------------------------------------------
# Page 12: What a team costs
# ----------------------------------------------------------------------------------------
def page_costs():
    st.header("12 · What a team costs")
    plain("Every demo counts its model calls. Multi-agent systems buy **specialisation, parallelism and robustness**, and pay in **calls, latency and coordination**.")
    ref = me.reference_costs()
    st.markdown("**Reference runs (one run of each design)**")
    st.bar_chart(pd.DataFrame(ref).set_index("Design")["Model calls"], horizontal=True) if "horizontal" in inspect.signature(st.bar_chart).parameters else st.bar_chart(pd.DataFrame(ref).set_index("Design")["Model calls"])
    st.caption("**How to read it.** Each bar is one design; its length is the number of model calls it needed for its task. Compare designs of the same family: hybrid with direct messages (4) versus via the manager (6); "
               "swarm one at a time (5) versus racing (10); pipeline with a research checker that halts (1) versus none (3).")
    ledger = st.session_state.get("s5_ledger", {})
    if ledger:
        st.markdown("**Runs you made on earlier pages** (latest setting of each)")
        df_show([{"Design": k, "Model calls": v} for k, v in ledger.items()])
    else:
        st.caption("Visit pages 4 to 11 and change the settings: each run you make is recorded here.")
    st.info("A generalist answered the honeymoon request in **1** call; manager/worker used **5** for the same request. On a free tier, each call can also mean 15 seconds of waiting. "
            "Whether the trade is worth it is a design decision, made **per feature**.")
    st.subheader("Multi-agent systems in the real world")
    df_show([
        {"Domain": "Smart power grids", "The agents": "Generators, solar farms, wind turbines, household meters", "Why a MAS": "Highly distributed; no control room sees everything", "Usually": "Decentralized, with regional coordinators"},
        {"Domain": "City-wide traffic", "The agents": "Lights, sensors, connected cars, navigation apps", "Why a MAS": "Dynamic: accidents and crowds change by the minute", "Usually": "Decentralized or hybrid"},
        {"Domain": "Supply chains", "The agents": "Port monitors, routing, weather, a coordinator", "Why a MAS": "Complexity and scale", "Usually": "Hybrid"},
        {"Domain": "Specialised LLM agents", "The agents": "Coordinator plus healthcare, finance, travel, legal", "Why a MAS": "Each domain needs its own knowledge and tools", "Usually": "Manager / worker or hybrid"},
        {"Domain": "Swarm robotics, drones", "The agents": "Many simple robots", "Why a MAS": "Robustness and scale; local rules suffice", "Usually": "Decentralized (swarm)"},
    ])
    st.subheader("Emergent behaviour: the challenge underneath")
    st.markdown("The most distinctive feature of a MAS is also its biggest risk: **behaviour no single agent was programmed to produce.** Sometimes it is the point (the swarm divided the guide with no coordinator). "
                "Sometimes it is a problem: two chatbots answering one customer, drones clustering in one area, agents politely handing a VIP query around while the customer waits (page 5, lab 3). "
                "Because no single agent contains the behaviour, it is hard to predict and debug. So you need **a log of every message**, **hard limits on loops**, and **testing by simulation** before real users.")
    st.warning("**What a team does not fix.** Every specialist here is the same model in a different role. Splitting the work improves **focus**, not **truth**. A team can be confidently wrong together, as the pipeline showed. "
               "Checkers, shared formats, code for arithmetic and humans for final decisions are what make a team trustworthy.")
    st.button("Next: review →", on_click=goto, args=(PAGES[13],), type="primary")


# ----------------------------------------------------------------------------------------
# Page 13: Review
# ----------------------------------------------------------------------------------------
QUIZ = [
    ("Which situation most clearly calls for a multi-agent system?",
     ["A thermostat controlling one room", "A smart power grid with thousands of generators and homes", "An LLM answering basic factual questions", "A chatbot giving bus times for one city"], 1,
     "It is complex, distributed and dynamic: the three reasons for a MAS."),
    ("In the honeymoon lab, why does a specialist answer visas better than the generalist?",
     ["It is a smarter model", "It has a narrower role and spends its whole answer on one topic", "It never makes mistakes", "It uses fewer tokens"], 1,
     "Same model, focused role. Specialisation improves focus, not truth."),
    ("What are the three C's?",
     ["Code, compile, check", "Communication, coordination, collaboration", "Control, cost, capacity", "Create, copy, combine"], 1,
     "Communication makes coordination possible; coordination makes collaboration possible."),
    ("The coordinator of a centralized desk goes down. The three specialists are healthy. What happens?",
     ["They answer everything", "No query is answered", "Only refunds are answered", "They elect a new coordinator automatically"], 1,
     "That is the single point of failure."),
    ("Why did bot B's hand-over take two attempts with the stale board?",
     ["Bot C refused: the board said it was free but its real queue was full", "The model was slow", "Bot D was offline", "Messages were lost"], 0,
     "Each agent has only a local, possibly out-of-date view."),
    ("With 20 agents, 'everyone tells everyone' sends how many messages per change?",
     ["20", "40", "380", "400"], 2,
     "n × (n − 1) = 20 × 19. This is why boards or coordinators are used."),
    ("In the hybrid demo, what is computed by code rather than by an LLM?",
     ["The friendly final sentence", "The refund arithmetic", "The routing of the query", "The greeting"], 1,
     "Rs 12,000 minus a 20% fee is Rs 9,600, every time. LLMs write; code counts."),
    ("A manager's plan mentions 'leave insurance aside'. A keyword check passes it. Why is that a problem?",
     ["The check should be case sensitive", "A word in a task is not an owner: nobody covers insurance", "Keyword checks are slow", "It is not a problem"], 1,
     "Check ownership (coverage), not just words."),
    ("Two peers read the board at the same moment and claim the same section. What fixes it?",
     ["A bigger board", "An atomic claim (a lock)", "More peers", "A different model"], 1,
     "A claim must be atomic, or work is duplicated."),
    ("Where does a checker earn its cost in the pipeline demo?",
     ["After the last stage only", "Where the error enters (after research)", "Nowhere", "After every stage, always"], 1,
     "One cheap call stops three more calls being spent on a wrong number."),
]


def page_review():
    st.header("13 · Review, quiz & cheat sheet")
    st.subheader("🧠 Quiz")
    with st.form("quiz5"):
        answers = [st.radio("**%d. %s**" % (i + 1, q), opts, index=None, key="q5_%d" % i) for i, (q, opts, c, w) in enumerate(QUIZ)]
        submitted = st.form_submit_button("Check my answers", type="primary")
    if submitted:
        score = 0
        for i, (q, opts, c, why) in enumerate(QUIZ):
            ok = answers[i] == opts[c]
            score += ok
            (st.success if ok else st.error)("**Q%d:** %s. %s" % (i + 1, "Correct" if ok else "Not quite", why))
        st.metric("Score", "%d / %d" % (score, len(QUIZ)))
    st.subheader("📝 Homework map")
    df_show([
        {"#": "HW1", "Level": "Easy", "Task": "Design a fallback for the centralized desk; what new problem does it create?", "Try it": "Page 4, lab 1 (fallback box)"},
        {"#": "HW2", "Level": "Medium", "Task": "The cost of keeping everyone informed: formula for 4, 20, 100 bots", "Try it": "Page 5, lab 2"},
        {"#": "HW3", "Level": "Medium", "Task": "A manager that cannot forget a requirement", "Try it": "Page 8 (stricter check)"},
        {"#": "HW4", "Level": "Hard", "Task": "A swarm with priorities and no boss; what if two peers claim at once?", "Try it": "Page 9 (priority + race/lock)"},
        {"#": "HW5", "Level": "Medium", "Task": "Where should checkers go in the pipeline?", "Try it": "Page 11 (placement table)"},
        {"#": "HW6", "Level": "Easy", "Task": "Add a third federated team: how many lines change?", "Try it": "Page 10 (food team)"},
        {"#": "HW7", "Level": "Medium", "Task": "Choose an architecture and pattern for four scenarios", "Try it": "Page 7"},
    ])
    st.subheader("⚡ Quick reference")
    df_show([
        {"Idea": "Question of the session", "One line": "Who decides, and who talks to whom?"},
        {"Idea": "Why a MAS", "One line": "Too complex, too distributed or too dynamic for one agent"},
        {"Idea": "When not", "One line": "Small, centralised, stable problems"},
        {"Idea": "Three C's", "One line": "Communication, coordination, collaboration"},
        {"Idea": "Centralized", "One line": "One controller decides; easy to debug; single point of failure"},
        {"Idea": "Decentralized", "One line": "Agents decide locally; robust; overhead and stale knowledge"},
        {"Idea": "Hybrid", "One line": "Central plan and answer; direct messages in between"},
        {"Idea": "Manager / worker", "One line": "Manager delegates and integrates; the plan is the only map"},
        {"Idea": "Peer / swarm", "One line": "Equals, local rules, shared board; needs atomic claims"},
        {"Idea": "Federated teams", "One line": "Specialist teams; a shared report format; clear ownership"},
        {"Idea": "Pipeline", "One line": "Each stage feeds the next; errors propagate; check early"},
        {"Idea": "Cost", "One line": "More agents, more calls, more waiting; a team is never free"},
    ])
    st.subheader("The four patterns compared")
    df_show([
        {"Pattern": "Manager / worker", "Who talks to whom": "Manager ↔ each worker", "Fits": "Centralized, hybrid", "Weak spot": "Manager is a single point of failure; uncovered requirements"},
        {"Pattern": "Peer / swarm", "Who talks to whom": "Peers ↔ peers (or via a board)", "Fits": "Decentralized", "Weak spot": "No global oversight; races; slower agreement"},
        {"Pattern": "Federated teams", "Who talks to whom": "Freely inside a team; across teams when needed", "Fits": "Hybrid, decentralized", "Weak spot": "Integration; needs shared formats"},
        {"Pattern": "Sequential pipeline", "Who talks to whom": "Each stage → the next", "Fits": "Centralized (fixed order)", "Weak spot": "Delays and errors propagate"},
    ])
    st.info("Next session: you build these designs properly with LangChain and with CrewAI, a framework for teams of role-based agents.")


# ----------------------------------------------------------------------------------------
# Sidebar + router
# ----------------------------------------------------------------------------------------
def sidebar():
    st.sidebar.title("🤝 MAS Lab")
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
        st.sidebar.info("Demo mode: scripted agents, pretend data.")


def main():
    sidebar()
    page = st.session_state.get("nav", PAGES[0])
    {PAGES[0]: page_home, PAGES[1]: page_one_agent, PAGES[2]: page_why, PAGES[3]: page_blocks, PAGES[4]: page_centralized,
     PAGES[5]: page_decentralized, PAGES[6]: page_hybrid, PAGES[7]: page_choose, PAGES[8]: page_manager, PAGES[9]: page_swarm,
     PAGES[10]: page_federated, PAGES[11]: page_pipeline, PAGES[12]: page_costs, PAGES[13]: page_review}[page]()


main()
