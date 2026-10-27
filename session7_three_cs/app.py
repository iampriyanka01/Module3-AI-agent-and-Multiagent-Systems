"""
Three C's Lab: an interactive companion to "Agents, Session 7: The Three C's in Multi-Agent Systems"
(the session PDF, the notebook Agents_Session_7_Three_Cs_and_Design_Patterns_in_CrewAI.ipynb, the five scripts in scripts/)
and the post-read "How Agents Learn From Each Other" (multi-agent reinforcement learning).

Run:   streamlit run app.py
Works with NO API key (Demo mode). Add a Gemini key for Live mode on the labs marked 🔴 live-capable.
"""
import inspect
import os

import pandas as pd
import streamlit as st

import threecs_engine as te

st.set_page_config(page_title="Three C's Lab", page_icon="🔺", layout="wide")

PAGES = [
    "🏠 Start here",
    "1 · The Three C's pyramid",
    "2 · Communication patterns",
    "3 · Coordination, cooperation & the 3 core problems",
    "4 · Four coordination mechanisms",
    "5 · The four design patterns in CrewAI",
    "6 · Crew debugger (predict, then fix)",
    "7 · How agents learn from each other (MARL)",
    "8 · Learning loops: peer feedback, self-play, rewards",
    "9 · Review, quiz & cheat sheet",
]

try:
    STRETCH = {"width": "stretch"} if inspect.signature(st.dataframe).parameters["width"].default == "stretch" else {"use_container_width": True}
except Exception:
    STRETCH = {"use_container_width": True}

NODE = 'node [shape=box, style="rounded,filled", fillcolor="#eef2ff", fontname="Helvetica"];'


# ----------------------------------------------------------------------------------------
# Helpers (same look as Sessions 5 and 6)
# ----------------------------------------------------------------------------------------
def _default_key() -> str:
    key = os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("GOOGLE_API_KEY", "")
        except Exception:
            key = ""
    return key


def get_backend() -> te.Backend:
    mode = st.session_state.get("mode", "Demo (no API key)")
    key = st.session_state.get("api_key", "") if mode.startswith("Live") else ""
    cfg = (key, st.session_state.get("chat_model", "gemini-3.5-flash-lite"), st.session_state.get("rate_limit", True))
    if st.session_state.get("_cfg") != cfg or "backend" not in st.session_state:
        st.session_state.backend = te.Backend(*cfg)
        st.session_state._cfg = cfg
    return st.session_state.backend


def goto(page: str):
    st.session_state.nav = page


def badge(b: te.Backend):
    if b.last_source == "gemini":
        st.caption("🟢 Written by Gemini")
    elif "fallback" in b.last_source:
        st.caption("🟠 " + (b.error or "Live call failed; showing the demo text."))
    else:
        st.caption("🔵 Demo simulation (scripted, no API key needed)")


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
    st.caption("🧪 **Course answer vs real library.** " + text)


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


def classify_lab(key: str, scenarios: list, choices: list):
    """Sort scenarios into categories, then check."""
    picks = []
    for i, (text, answer, why) in enumerate(scenarios):
        picks.append(st.radio("**%d.** %s" % (i + 1, text), choices, index=None, key="%s_%d" % (key, i), horizontal=True))
    if st.button("Check my sorting", key=key + "_check", type="primary"):
        score = 0
        for i, (text, answer, why) in enumerate(scenarios):
            if picks[i] == answer:
                score += 1
                st.success("%d. Correct: **%s**. %s" % (i + 1, answer, why))
            elif picks[i] is None:
                st.warning("%d. Not answered. It's **%s**. %s" % (i + 1, answer, why))
            else:
                st.error("%d. Not quite: it's **%s**. %s" % (i + 1, answer, why))
        st.metric("Score", "%d / %d" % (score, len(scenarios)))


# ----------------------------------------------------------------------------------------
# Page 0
# ----------------------------------------------------------------------------------------
def page_home():
    st.title("🔺 Three C's Lab: from talking, to organising, to teamwork")
    st.caption("An interactive companion to *Agents, Session 7: The Three C's in Multi-Agent Systems*, its notebook and five scripts, and the post-read *How Agents Learn From Each Other*.")
    story("DilSeTrips now runs a team of agents. They message each other constantly, and still two of them answered the same ticket, "
          "a third waited for data that never came, and a fourth overwrote a teammate's work on the shared board. "
          "Talking is not teamwork. Session 7 is about the three layers that turn agents into a team, and how a team can learn to work better on its own.")
    st.success("**The question to hold on to:** what must agents share (communication), who does what and when (coordination), and how do they help each other succeed (cooperation)? "
               "Each layer needs the one below it.")
    dot('digraph { rankdir=BT; ' + NODE + ' C1 [label="1 · Communication\\n\\"What should we know?\\"", fillcolor="#d0ebff"]; '
        'C2 [label="2 · Coordination\\n\\"Who does what, and when?\\"", fillcolor="#d3f9d8"]; C3 [label="3 · Cooperation\\n\\"How can we help each other succeed?\\"", fillcolor="#fff3bf"]; '
        'L [label="+ Learning (MARL)\\nthe team improves its own 3 C\'s", fillcolor="#ffe3e3"]; C1 -> C2 -> C3 -> L; }')
    st.subheader("Your path")
    path = [
        (PAGES[1], "Switch each layer on and off and watch a construction crew (and an agent team) fall apart in a specific way."),
        (PAGES[2], "Request–response, publish–subscribe and blackboard as live simulations, including the overwritten-code incident."),
        (PAGES[3], "Coordination vs cooperation. Task allocation, resource allocation and conflict resolution in the smart hospital."),
        (PAGES[4], "Centralized, distributed, rule-based and learning-based: sort real scenarios, including the traffic-grid trap."),
        (PAGES[5], "Manager/worker, pipeline, peer/swarm and federated teams in CrewAI, with the notebook's code and the Three C's in each."),
        (PAGES[6], "Build a crew, predict what it does, and see CrewAI's real errors. Includes the exam's broken crews."),
        (PAGES[7], "Two learning games: drones that learn to share corridors, and agents that learn to help each other."),
        (PAGES[8], "Peer feedback (RLPF), a self-play critic, reward design, and a team that learns its own workflow."),
        (PAGES[9], "Quiz, cheat sheet and the notebook's exercises."),
    ]
    for page, desc in path:
        a, b, c = st.columns([3, 6, 1.2])
        a.markdown("**" + page + "**")
        b.write(desc)
        c.button("Open", key="home_" + page, on_click=goto, args=(page,))
    st.markdown("**Modes:** *Demo* (default) needs no key. *Live* (sidebar) lets Gemini play the writer and critic on page 8. "
                "Every crew prediction, simulation and learning game is plain Python, so it is the same in both modes.")
    reality("Some course quiz answers describe CrewAI differently from how CrewAI 1.15 actually behaves. Where they differ, this app shows both: the answer the course expects, "
            "and what the library really does (checked against CrewAI 1.15.30 with a stub model).")


# ----------------------------------------------------------------------------------------
# Page 1: the pyramid
# ----------------------------------------------------------------------------------------
def page_pyramid():
    st.header("1 · The Three C's pyramid")
    plain("Communication is sharing information. Coordination is organising actions based on that information. Cooperation is actively helping each other toward a shared goal. "
          "Each layer is built on the one below it.")
    df_show(te.PYRAMID)
    st.subheader("🧪 Lab: the failure cascade")
    predict("switch on cooperation but switch off communication. Can the crew cooperate?")
    c1, c2, c3 = st.columns(3)
    comm = c1.toggle("Communication", True, key="p1_comm")
    coord = c2.toggle("Coordination", True, key="p1_coord")
    coop = c3.toggle("Cooperation", True, key="p1_coop")
    r = te.cascade(comm, coord, coop)
    colors = {0: "error", 1: "warning", 2: "info", 3: "success"}
    getattr(st, colors[r["level"]])("**%s.** %s" % (r["verdict"], r["story"]))
    for n in r["notes"]:
        st.caption("⚠️ " + n)
    cols = st.columns(3)
    for col, (name, ok) in zip(cols, r["working"].items()):
        col.metric(name, "working" if ok else "not working")
    notice("Poor communication → failed coordination → impossible cooperation. A higher layer you switch on does nothing if a lower one is off.")
    st.subheader("The same three layers in a product team")
    df_show([
        {"C": "Coordination", "Product team": "Designers mock up, developers code, testers check; scheduling prevents overlaps and bottlenecks", "Agent team": "Who handles which request, in which order"},
        {"C": "Cooperation", "Product team": "Developers and designers refine features together and help solve each other's problems", "Agent team": "Agents combine and improve each other's partial results"},
        {"C": "Communication", "Product team": "Meetings, updates in project tools, clarified requirements", "Agent team": "Messages, shared context, status updates"},
    ])
    exam("If a question says agents 'know what the others are doing' but work is duplicated or dependencies are missed, the failed layer is **coordination**, not communication.")


# ----------------------------------------------------------------------------------------
# Page 2: communication patterns
# ----------------------------------------------------------------------------------------
def page_comm():
    st.header("2 · Communication patterns")
    plain("Communication types describe the direction of information (one-to-one, one-to-many, shared). Patterns describe the interaction style built on each type.")
    df_show(te.PATTERNS)
    dot('digraph { rankdir=LR; ' + NODE + ' subgraph cluster_rr { label="Request–response (1:1)"; A1 [label="Agent A"]; B1 [label="Agent B"]; A1 -> B1 [label="request"]; B1 -> A1 [label="response"]; } '
        'subgraph cluster_ps { label="Publish–subscribe (1:n)"; P [label="Publisher"]; T [label="Topic", shape=ellipse, fillcolor="#fff3bf"]; S1 [label="Sub 1"]; S2 [label="Sub 2"]; S3 [label="Sub 3"]; P -> T; T -> S1; T -> S2; T -> S3; } '
        'subgraph cluster_bb { label="Blackboard (n:n)"; BB [label="Blackboard\\n(shared memory)", fillcolor="#d3f9d8"]; R [label="Research"]; W [label="Writer"]; E [label="Editor"]; R -> BB; BB -> W; W -> BB; BB -> E; E -> BB; } }')
    tabs = st.tabs(["Request–response", "Publish–subscribe", "Blackboard", "All three in one crew (Demo 1)"])
    with tabs[0]:
        st.markdown("Like emailing IT support: one sends, one replies, and the sender **waits**.")
        a, b, c = st.columns(3)
        n = a.slider("Requests to send", 1, 10, 5, key="p2_n")
        s = b.slider("Seconds per reply", 1, 10, 4, key="p2_s")
        w = c.slider("Support agents answering", 1, 5, 1, key="p2_w")
        r = te.request_response(n, s, w)
        df_show(r["rows"])
        st.metric("Requester blocked for", "%s s" % r["total"])
        notice("With one responder the waiting time grows with every request: blocking and tightly coupled. Adding responders helps, but each pair is still one-to-one.")
    with tabs[1]:
        st.markdown("Like a company-wide announcement: the publisher posts to a **topic**, every subscriber gets it, and nobody replies.")
        topics = ["alerts", "billing", "itinerary"]
        subs = {}
        cols = st.columns(4)
        for col, agent, default in zip(cols, ["Monitoring agent", "Billing agent", "Itinerary agent", "Support agent"],
                                       [["alerts"], ["billing", "alerts"], ["itinerary"], ["alerts", "billing"]]):
            subs[agent] = set(col.multiselect(agent, topics, default, key="p2_sub_" + agent))
        topic = st.selectbox("Publish to topic", topics, key="p2_topic")
        msg = st.text_input("Message", "Server overload detected: payments are slow", key="p2_msg")
        r = te.publish(subs, topic, msg)
        df_show(r["deliveries"])
        st.caption("Delivered to %d agent(s). The publisher doesn't wait and never learns whether anyone acted." % len(r["receivers"]))
        notice("Scalable and decoupled: adding a subscriber needs no change to the publisher. The price is **no feedback loop**, possible overload, and ordering problems.")
    with tabs[2]:
        story("Two DeveloperAgents finish at nearly the same moment and both post to latest_code.py on the blackboard. The TestingAgent only tests one of them.")
        mode = st.radio("How the board is protected", list(te.BLACKBOARD_MODES), key="p2_mode")
        st.caption(te.BLACKBOARD_MODES[mode])
        gap = st.slider("Agent B finishes this many seconds after Agent A", 0.0, 5.0, 0.2, 0.1, key="p2_gap")
        r = te.blackboard(mode, gap)
        df_show(r["events"])
        if r["lost"]:
            st.error("Lost: %s. The board is now **inconsistent** with what the team built." % ", ".join(r["lost"]))
        else:
            st.success("Nothing lost.")
        st.markdown("**What the TestingAgent reads:**")
        st.code(r["tester_sees"], language="python")
        exam("This incident shows a blackboard's limitations: it **requires conflict management** and **can have consistency issues**. "
             "It is NOT 'synchronous and blocking' (it's asynchronous), and it DOES support one-to-many (anything posted is visible to all).")
    with tabs[3]:
        st.markdown("The notebook's Demo 1 (`scripts/crewai_comm_pattern.py`) shows all three patterns in one CrewAI crew. CrewAI implements all three with task results and `context`; "
                    "the difference is **who receives the information**.")
        df_show([
            {"Hand-off": "Researcher → stored result → Writer", "Pattern": "Shared memory (blackboard idea)", "How": "The writer reads the stored research through context=[research_task], without asking the researcher anything"},
            {"Hand-off": "Writer → Editor", "Pattern": "Direct (request–response idea)", "How": "The draft goes one-to-one through context=[writing_task]"},
            {"Hand-off": "Editor → everyone", "Pattern": "Broadcast (publish–subscribe idea)", "How": "The editor ends with 'ANNOUNCEMENT TO ALL TEAMS:', written for everyone"},
        ])
        show_code("Demo 1 code (scripts/crewai_comm_pattern.py, the key part)", '''
researcher = Agent(role="Researcher", goal="Find accurate facts using Wikipedia.",
                   backstory="You only report what your sources say.", tools=[search_wikipedia], llm=llm)
writer = Agent(role="Writer", goal="Write a short, clear article from the research.",
               backstory="You write simply and only use the facts you're given.", llm=llm)
editor = Agent(role="Editor", goal="Polish the article and announce it to the whole team.",
               backstory="You fix mistakes, tighten the writing, and keep everyone informed.", llm=llm)

# 1. SHARED MEMORY: the research result is stored for later tasks to read
research_task = Task(description="Find 4 facts about {topic} using Wikipedia.",
                     expected_output="4 bullet points of facts.", agent=researcher)
# The writer reads the stored research, without messaging the researcher
writing_task = Task(description="Write a 120-word article about {topic} using the research.",
                    expected_output="A 120-word article.", agent=writer, context=[research_task])
# 2. DIRECT: writer -> editor, one-to-one.   3. BROADCAST: an announcement for everyone
editing_task = Task(description="Edit the article for clarity. Then add a final line that starts with "
                                "'ANNOUNCEMENT TO ALL TEAMS:' telling everyone the article is ready.",
                    expected_output="The edited article, then the announcement line.",
                    agent=editor, context=[writing_task])

crew = Crew(agents=[researcher, writer, editor], tasks=[research_task, writing_task, editing_task],
            process=Process.sequential, max_rpm=4, verbose=True)
result = crew.kickoff(inputs={"topic": "electric vehicles"})       # notebook: await crew.kickoff_async(...)
for task_output in result.tasks_output:
    print(task_output.agent, task_output.raw)''')


# ----------------------------------------------------------------------------------------
# Page 3: coordination, cooperation and the three core problems
# ----------------------------------------------------------------------------------------
def page_problems():
    st.header("3 · Coordination, cooperation & the three core problems")
    plain("Communication = sharing information. Coordination = organising actions based on it (who does what and when). "
          "Cooperation = actively working together for mutual benefit. Coordination can even exist between competitors; cooperation always means aligned goals.")
    df_show(te.COORD_VS_COOP)
    st.markdown("**The smart hospital** (the session's running example): a Symptom Analysis agent, a Diagnosis agent, a Treatment agent and a Report agent.")
    dot('digraph { rankdir=LR; ' + NODE + ' In [label="Patient input", fillcolor="#fff3bf"]; S [label="Symptom\\nAnalysis"]; D [label="Diagnosis"]; T [label="Treatment"]; R [label="Report"]; '
        'API [label="Medical knowledge API\\n(shared, rate-limited)", shape=cylinder, fillcolor="#ffe3e3"]; In -> S -> D -> T -> R; D -> API [style=dashed]; T -> API [style=dashed]; }')
    df_show(te.CORE_PROBLEMS)
    tabs = st.tabs(["Sort the problems", "Resource allocation lab", "Conflict resolution lab"])
    with tabs[0]:
        classify_lab("p3_cls", te.PROBLEM_SCENARIOS, ["Task allocation", "Resource allocation", "Conflict resolution"])
    with tabs[1]:
        story("Four agents each need one call to the medical knowledge API, which can serve only a limited number of calls per second.")
        a, b = st.columns(2)
        strategy = a.radio("Strategy", ["No coordination (everyone fires)", "Queue (resource manager agent)", "Priority (Diagnosis first)", "Token / lock (one at a time)"], key="p3_strat")
        cap = b.slider("API capacity (calls per tick)", 1, 4, 2, key="p3_cap")
        agents = ["Diagnosis", "Treatment", "Report", "Symptom"]
        pri = {"Diagnosis": 3, "Symptom": 2, "Treatment": 1, "Report": 0}
        r = te.resource_schedule(agents, strategy, cap, pri)
        df_show(r["rows"])
        m = st.columns(3)
        m[0].metric("Ticks to finish", r["ticks"])
        m[1].metric("Rejected calls (errors)", r["errors"])
        m[2].metric("Diagnosis served at tick", r["done_at"].get("Diagnosis", "-"))
        notice("Without coordination, calls fail and get retried, so the API wastes work and the order is random. A queue removes the errors; priority also decides who goes first; "
               "a token is the simplest decentralized option (whoever holds it may call).")
    with tabs[2]:
        story("The Diagnosis agent says flu. The Symptom agent, with newer input, says allergy. Antivirals or antihistamines?")
        base = pd.DataFrame([
            {"Agent": "Diagnosis agent 1", "Answer": "flu", "Confidence": 0.70, "Minutes ago": 30},
            {"Agent": "Symptom agent", "Answer": "allergy", "Confidence": 0.65, "Minutes ago": 2},
            {"Agent": "Diagnosis agent 2", "Answer": "flu", "Confidence": 0.55, "Minutes ago": 25},
            {"Agent": "Diagnosis agent 3", "Answer": "allergy", "Confidence": 0.80, "Minutes ago": 5},
        ])
        edited = st.data_editor(base, hide_index=True, key="p3_ops", num_rows="dynamic", **STRETCH)
        rows = edited.to_dict("records")
        cols = st.columns(4)
        for col, method in zip(cols, ["Majority vote", "Confidence arbitration", "Most recent data", "Referee / mediator agent"]):
            res = te.resolve_conflict(rows, method)
            col.markdown("**%s**" % method)
            col.metric("Decision", res["decision"])
            col.caption(res["why"])
        notice("Different rules can pick different answers from the SAME opinions. Choose the rule deliberately, and for medical or financial decisions, end with a human.")


# ----------------------------------------------------------------------------------------
# Page 4: mechanisms
# ----------------------------------------------------------------------------------------
def page_mechanisms():
    st.header("4 · Four coordination mechanisms")
    plain("Mechanisms are the playbook (the organising rules). Cooperation strategies are the teamwork that happens inside that playbook. Two sides of one coin.")
    df_show(te.MECHANISMS)
    st.subheader("🧪 Lab: which mechanism is it?")
    classify_lab("p4_cls", te.MECH_SCENARIOS, te.MECH_CHOICES)
    exam("When a question asks for the 'most accurate and comprehensive' classification and the system has two features (e.g. peers with no central server AND reinforcement learning), "
         "pick the answer that names both. Options with **'purely'** usually leave something out.")
    exam("Why is a CrewAI `Process.sequential` crew with no manager 'distributed' rather than centralized? Because coordination comes from agents following a **predefined protocol (the task list)**, "
         "not from real-time instructions issued by a single orchestrator agent.")


# ----------------------------------------------------------------------------------------
# Page 5: design patterns in CrewAI
# ----------------------------------------------------------------------------------------
PATTERN_CODE = {
    "Manager / worker (Demo 2)": ("Process.hierarchical + manager_llm. Tasks are left UNASSIGNED so the manager allocates them.", "Centralized", '''
data_analyst = Agent(role="Customer support data analyst", goal="Find the biggest problems in the support data, with numbers.",
                     backstory="You read support data carefully and always quote the figures.", tools=[get_support_data], llm=llm)
process_optimizer = Agent(role="Process improvement specialist", goal="Turn the analysis into practical improvements.",
                          backstory="You suggest specific, realistic fixes, not vague advice.", llm=llm)
report_writer = Agent(role="Executive report writer", goal="Write a short, clear report for the company's COO.",
                      backstory="Your reports are brief, specific and easy to act on.", llm=llm)

# No agent= on purpose, so the manager chooses who does each one
analysis_task = Task(description="Analyse last quarter's customer support data and find the main problems.",
                     expected_output="The main problems, with ticket counts and resolution times.")
improvement_task = Task(description="Based on the analysis, suggest three concrete improvements.",
                        expected_output="Three improvements, each one or two sentences.", context=[analysis_task])
report_task = Task(description="Write a short report for the COO: a summary, the problems, and the improvements.",
                   expected_output="A report of about 200 words.", context=[analysis_task, improvement_task])

crew = Crew(agents=[data_analyst, process_optimizer, report_writer],
            tasks=[analysis_task, improvement_task, report_task],
            process=Process.hierarchical,     # coordination: a manager decides
            manager_llm=llm,                  # CrewAI builds the manager agent from this
            max_rpm=4, verbose=True)''',
        [("Communication", "Task results flow to later tasks through context=[...]"), ("Coordination", "The manager decides who does what (task allocation)"),
         ("Cooperation", "Analyst, optimizer and writer together produce one report")]),
    "Sequential pipeline (Demo 3)": ("Process.sequential + context. Fixed order, each output feeds the next.", "Rule-based", '''
researcher = Agent(role="AI research analyst", goal="Find well-documented facts about AI in software development, using Wikipedia.",
                   backstory="You never state a fact you haven't found in your research.", tools=[search_wikipedia], llm=llm)
analyst = Agent(role="Data analyst", goal="Pick the most promising idea from the research and explain why.",
                backstory="You turn raw research into a clear recommendation.", llm=llm)
report_writer = Agent(role="Executive summary writer", goal="Write a short, polished summary for a business audience.",
                      backstory="You write clearly and briefly, and you only use facts you were given.", llm=llm)

research_task = Task(description="Research three important uses of AI in software development. Use Wikipedia.",
                     expected_output="A bullet list of three uses.", agent=researcher)
analysis_task = Task(description="Choose the most promising use for a new software product, and explain why.",
                     expected_output="The chosen use, and three reasons.", agent=analyst, context=[research_task])
report_task = Task(description="Write a one-page summary titled 'AI in Software Development'.",
                   expected_output="About 200 words.", agent=report_writer, context=[analysis_task])

crew = Crew(agents=[researcher, analyst, report_writer], tasks=[research_task, analysis_task, report_task],
            process=Process.sequential,       # one after another, in THIS order
            max_rpm=4, verbose=True)          # True or False (the old file's verbose=2 is rejected)''',
        [("Communication", "context=[research_task] hands the research to the analyst"), ("Coordination", "The order is fixed: research must finish before analysis"),
         ("Cooperation", "Three specialists build one report together")]),
    "Peer / swarm (Demo 4)": ("Process.sequential turns + allow_delegation=True on every peer, so equals can ask each other. CrewAI has no true swarm process.", "Distributed", '''
developer = Agent(role="Software developer", goal="Write clean, correct, well-commented Python code.",
                  backstory="You write careful code and welcome questions from teammates.", llm=llm, allow_delegation=True)
tester = Agent(role="QA engineer", goal="Find bugs and edge cases before users do.",
               backstory="If something is unclear, you ask the developer.", llm=llm, allow_delegation=True)
documentation_writer = Agent(role="Technical writer", goal="Write clear, accurate documentation.",
                             backstory="You check details with the developer when unsure.", llm=llm, allow_delegation=True)

coding_task = Task(description="Write calculate_average(numbers). Handle an empty list sensibly.",
                   expected_output="The complete Python function.", agent=developer)
testing_task = Task(description="Test calculate_average: positive, negative and empty lists. Report bugs.",
                    expected_output="Test cases, expected results, bugs found.", agent=tester, context=[coding_task])
documentation_task = Task(description="Document calculate_average with two usage examples.",
                          expected_output="Short documentation in Markdown.", agent=documentation_writer,
                          context=[coding_task, testing_task])

crew = Crew(agents=[developer, tester, documentation_writer],
            tasks=[coding_task, testing_task, documentation_task],
            process=Process.sequential, max_rpm=4, verbose=True)   # look for peers asking each other''',
        [("Communication", "context passes the code and test report; peers can also ask each other"), ("Coordination", "No boss: the turn order is agreed in advance"),
         ("Cooperation", "The tester finds problems, the developer's work improves, the writer documents the result")]),
    "Federated teams (Demo 5)": ("Process.hierarchical + your own manager_agent (no tools, not in the agents list) routing work to specialist teams. The task is unassigned.", "Centralized (over teams)", '''
# Research team                                     # Marketing team
researcher = Agent(role="Lead researcher", goal="Find well-documented robotics trends.", backstory="...",
                   tools=[search_wikipedia], llm=llm)
analyst = Agent(role="Data analyst", goal="Turn research into insights about market impact.", backstory="...", llm=llm)
writer = Agent(role="Content strategist", goal="Turn findings into an executive summary.", backstory="...", llm=llm)

# The manager: NO tools, its only job is to delegate
project_manager = Agent(role="Project manager",
                        goal="Deliver a high-quality report by giving each part to the right specialist.",
                        backstory="You run research and marketing teams.", llm=llm)

report_task = Task(description="Create a report on three robotics trends: overview, market impact, executive summary.",
                   expected_output="A well-structured Markdown report, about 300 words.")   # no agent=

crew = Crew(agents=[researcher, analyst, writer],   # the manager is NOT in this list
            tasks=[report_task], process=Process.hierarchical, manager_agent=project_manager,
            max_rpm=4, verbose=True)''',
        [("Communication", "The manager sends work to specialists and receives their results"), ("Coordination", "The manager decides which team handles each part"),
         ("Cooperation", "Research, analysis and writing combine into one report")]),
}


def page_patterns():
    st.header("5 · The four design patterns in CrewAI")
    plain("Session 5 named four patterns; Session 6 introduced CrewAI. Here each pattern is mapped onto CrewAI's parts: Agent, Task, Crew and Process.")
    df_show([
        {"Pattern": "Manager / worker", "CrewAI setting": "Process.hierarchical + manager_llm; tasks unassigned", "Mechanism": "Centralized", "Weak spot": "Manager is a bottleneck / single point of failure"},
        {"Pattern": "Sequential pipeline", "CrewAI setting": "Process.sequential + context=[...]", "Mechanism": "Rule-based", "Weak spot": "Errors and delays propagate down the line"},
        {"Pattern": "Peer / swarm", "CrewAI setting": "Process.sequential + allow_delegation=True on peers", "Mechanism": "Distributed", "Weak spot": "No global oversight; peers may ping-pong"},
        {"Pattern": "Federated teams", "CrewAI setting": "Process.hierarchical + manager_agent over specialist teams", "Mechanism": "Centralized coordination of teams", "Weak spot": "Integration across teams"},
    ])
    pick = st.radio("Pattern", list(PATTERN_CODE), horizontal=True, key="p5_pick")
    desc, mech, code, cs = PATTERN_CODE[pick]
    st.info("**%s** · Mechanism: **%s**" % (desc, mech))
    st.code(code.strip("\n"), language="python")
    df_show([{"C": c, "In this crew": t} for c, t in cs])
    st.subheader("How each pattern works (from the session document)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Peer / swarm workflow**\n1. Observation & local decision\n2. Communication with peers (no central controller)\n3. Local collaboration to avoid conflicts\n4. **Emergent behaviour**: a global outcome without central planning")
        st.markdown("**Sequential pipeline workflow**\n1. Input reception\n2. Stepwise processing (each output is the next input)\n3. Intermediate validation\n4. Final output\n\n*Example:* idea extraction → drafting → editing → QA.")
    with c2:
        st.markdown("**Federated teams workflow**\n1. Task assignment by type to the right team\n2. Specialist processing inside each team\n3. Cross-team sharing of results\n4. Synthesis into one solution\n\n*Example:* market, news and sentiment teams → one market-insights report.")
        st.markdown("**Manager / worker** in CrewAI: the manager receives the tasks, decides which worker does each one, can ask questions, and integrates the results.")
    st.subheader("Three rules the notebook fixed in the original course files")
    df_show([
        {"Rule": "In hierarchical crews, leave tasks unassigned", "Why": "If a task says agent=..., the manager can only pass it to that agent: a pipeline with extra manager calls"},
        {"Rule": "A manager_agent has NO tools and is NOT in agents=[...]", "Why": "CrewAI refuses otherwise: 'Manager agent should not be included in agents list.' / 'Manager agent should not have tools'"},
        {"Rule": "verbose=True or False (not 2); no temperature for Gemini 3; max_rpm=4", "Why": "verbose=2 fails at creation; the free tier allows ~5 calls a minute, and max_rpm makes agents take turns: resource allocation you can see"},
    ])


# ----------------------------------------------------------------------------------------
# Page 6: crew debugger
# ----------------------------------------------------------------------------------------
def _load_preset():
    name = st.session_state.get("p6_preset")
    if name in te.CREW_PRESETS:
        import copy
        st.session_state["p6_cfg"] = copy.deepcopy(te.CREW_PRESETS[name])
        st.session_state["p6_rev"] = st.session_state.get("p6_rev", 0) + 1


def page_debugger():
    st.header("6 · Crew debugger: predict, then fix")
    plain("Pick a crew (the notebook's demos, or a broken crew from an exam scenario), change anything, and see what CrewAI would do: "
          "the error it raises, or which outputs each task receives, and what crew.kickoff() returns. No API calls: this is a model of CrewAI's rules, checked against CrewAI 1.15.30.")
    names = list(te.CREW_PRESETS)
    st.selectbox("Start from", names, key="p6_preset", on_change=_load_preset)
    if "p6_cfg" not in st.session_state:
        _load_preset()
    cfg = st.session_state["p6_cfg"]
    rev = st.session_state.get("p6_rev", 0)
    with st.expander("✏️ Edit the crew", expanded=False):
        a, b = st.columns(2)
        cfg["process"] = a.radio("process", ["sequential", "hierarchical"], index=["sequential", "hierarchical"].index(cfg["process"]), horizontal=True, key="p6_proc_%d" % rev)
        if cfg["process"] == "hierarchical":
            opts = ["(none)", "manager_llm", "manager_agent"]
            m = b.radio("manager", opts, index=opts.index(cfg.get("manager") or "(none)"), horizontal=True, key="p6_mgr_%d" % rev)
            cfg["manager"] = None if m == "(none)" else m
            if cfg["manager"] == "manager_agent":
                cfg["manager_in_agents"] = b.checkbox("manager also listed in agents=[...]", cfg.get("manager_in_agents", False), key="p6_mia_%d" % rev)
                cfg["manager_has_tools"] = b.checkbox("manager has tools", cfg.get("manager_has_tools", False), key="p6_mht_%d" % rev)
        st.markdown("**Agents**")
        adf = st.data_editor(pd.DataFrame(cfg["agents"]), hide_index=True, num_rows="dynamic", key="p6_agents_%d" % rev, **STRETCH)
        cfg["agents"] = [dict(name=str(r["name"]), tools=bool(r["tools"]), allow_delegation=bool(r["allow_delegation"])) for r in adf.to_dict("records") if str(r.get("name") or "").strip()]
        st.markdown("**Tasks, in the order of `tasks=[...]`** (context: leave blank = not set; type `[]` for an explicit empty list; or task ids separated by commas)")
        trows = [{"id": t["id"], "description": t["description"], "agent": t["agent"] or "", "context": "" if t["context"] is None else ("[]" if t["context"] == [] else ", ".join(t["context"])),
                  "output_file": t.get("output_file", "")} for t in cfg["tasks"]]
        tdf = st.data_editor(pd.DataFrame(trows), hide_index=True, num_rows="dynamic", key="p6_tasks_%d" % rev, **STRETCH)
        new = []
        for r in tdf.to_dict("records"):
            if not str(r.get("id") or "").strip():
                continue
            ctx_raw = str(r.get("context") or "").strip()
            ctx = None if ctx_raw == "" else ([] if ctx_raw == "[]" else [c.strip() for c in ctx_raw.strip("[]").split(",") if c.strip()])
            new.append({"id": str(r["id"]).strip(), "description": str(r.get("description") or r["id"]), "agent": (str(r.get("agent") or "").strip() or None),
                        "context": ctx, "output_file": str(r.get("output_file") or "").strip()})
        cfg["tasks"] = new
    predict("will this crew run? If it runs, what will each task actually see?")
    if st.button("Check the crew", type="primary", key="p6_go") or st.session_state.get("p6_auto"):
        st.session_state["p6_auto"] = True
        r = te.check_crew(cfg)
        if r["errors"]:
            for when, msg in r["errors"]:
                st.error("**Fails at `%s`:** %s" % (when, msg))
        else:
            st.success("✅ It runs.")
        st.markdown("**Pattern:** %s  \n**Mechanism:** %s" % (r["pattern"], r["mechanism"]))
        if r["runs"]:
            df_show(r["flow"])
            st.caption("↩️ " + r["returns"])
        for w in r["warnings"]:
            st.warning(w)
        for i in r["info"]:
            st.caption("ℹ️ " + i)
        show_code("The code for this crew", te.crew_code(cfg))
    st.subheader("What the exam scenarios expect, and what CrewAI really does")
    df_show([
        {"Scenario": "Midnight report: task list reordered (analysis before scan)", "Course answer": "The tasks list was reordered, so a task ran before its dependency ('ContextUnavailableError')",
         "CrewAI 1.15": "Same cause. The crew isn't even created: 'Task ... has a context dependency on a future task ..., which is not allowed.'"},
        {"Scenario": "Midnight report: context removed, order still wrong", "Course answer": "Runs without error; the analyst works on empty input; the briefing says 'no anomalies found'",
         "CrewAI 1.15": "Matches: analysis runs first, so there are no earlier outputs to pass. A loud error became a silent wrong answer"},
        {"Scenario": "Empty executive briefing: writer got no input", "Course answer": "The writing task was missing its context parameter",
         "CrewAI 1.15": "Careful: with context NOT set, CrewAI passes ALL earlier outputs. The writer only gets nothing with context=[] (or a wrong list)"},
        {"Scenario": "kickoff() with output_file on the final task", "Course answer": "Returns a confirmation message string",
         "CrewAI 1.15": "Returns a CrewOutput whose .raw is the full text; the file gets a copy. Check with print(type(result))"},
        {"Scenario": "allow_delegation=True on one peer in a swarm", "Course answer": "Conceptually flawed: delegation is the mechanic of a hierarchy; it pollutes a peer model of equals",
         "CrewAI 1.15": "Runs fine; that agent gains 'Delegate work to coworker' and 'Ask question to coworker' tools"},
    ])
    exam("Answer exam questions with the course answer. Use the right-hand column when you write real code.")


# ----------------------------------------------------------------------------------------
# Page 7: MARL
# ----------------------------------------------------------------------------------------
def _line(rows, x, ys, title):
    df = pd.DataFrame(rows).set_index(x)[ys]
    st.markdown("**" + title + "**")
    st.line_chart(df)


def page_marl():
    st.header("7 · How agents learn from each other (MARL)")
    plain("Static rules only go so far. In multi-agent reinforcement learning (MARL), agents are rewarded for good teamwork and penalised for conflicts, "
          "and through repeated interactions they discover how to communicate, divide work and cooperate, without anyone scripting it.")
    story("Friends learning a new co-op video game. Round 1: chaos. Round 2: one shares a power-up, the score rises (positive reward). "
          "Round 3: two fight over one item and both lose it (negative reward). Rounds 4 → N: they work out roles, sharing and combos. Nobody wrote the rules.")
    df_show([
        {"": "Environment", "Single-agent RL": "Stable: changes only when the agent acts", "Multi-agent RL": "Contains other LEARNING agents: non-stationary"},
        {"": "Example", "Single-agent RL": "One delivery robot learning routes in steady traffic; an LLM summariser tuned with RLHF", "Multi-agent RL": "Hundreds of robots learning at once: each change alters traffic for all; researcher + writer + reviewer learning together"},
        {"": "Result", "Single-agent RL": "A stable policy", "Multi-agent RL": "A co-evolutionary loop: co-learning, emergent teamwork"},
    ])
    tabs = st.tabs(["Game 1 · Learning to coordinate (corridors)", "Game 2 · Learning to cooperate (helping)", "The feedback loop"])
    with tabs[0]:
        st.markdown("Two drones each choose a corridor every round. **A** is short (worth 3), **B** is long (worth 1). "
                    "If both choose the same corridor they block each other and **both get 0**. Each drone only knows its own experience.")
        a, b, c, d = st.columns(4)
        eps = a.slider("Exploration ε at start", 0.0, 1.0, 0.5, 0.05, key="p7_eps")
        decay = b.slider("ε decay per round", 0.90, 1.0, 0.985, 0.005, key="p7_decay")
        alpha = c.slider("Learning rate α", 0.05, 0.8, 0.2, 0.05, key="p7_alpha")
        seed = d.number_input("Random seed", 0, 999, 1, key="p7_seed")
        predict("with lots of exploration early on, how many clashes happen in the first 50 rounds compared with the last 50?")
        g = te.corridor_game(300, eps, decay, alpha, "individual", int(seed))
        rows = g["rows"]
        for r in rows:
            r["Clashes (last 20)"] = sum(x["Clash"] for x in rows[max(0, r["Episode"] - 20):r["Episode"]])
        m = st.columns(3)
        m[0].metric("Clashes, first 50 rounds", sum(r["Clash"] for r in rows[:50]))
        m[1].metric("Clashes, last 50 rounds", sum(r["Clash"] for r in rows[-50:]))
        final = "D1 → %s · D2 → %s" % (max(g["q"][0], key=g["q"][0].get), max(g["q"][1], key=g["q"][1].get))
        m[2].metric("Corridors they settled on", final)
        _line(rows, "Episode", ["Clashes (last 20)"], "Clashes in the last 20 rounds (exploration volatility, then emergent coordination)")
        _line(rows, "Episode", ["Drone 1 value of A", "Drone 1 value of B"], "Drone 1's estimate of each corridor: it keeps moving because Drone 2 is learning too")
        notice("The 'volatility phase' from the traffic-grid scenario: early exploration causes clashes (one agent's experiment can starve a neighbour), "
               "then a division emerges that nobody programmed. Drone 1's value for A rises and falls as Drone 2 changes its mind: that is **non-stationarity**.")
    with tabs[1]:
        st.markdown("Each round, each agent either **works alone** (+1 for itself) or **helps** (only +0.6 for itself, but its teammate gets +1). "
                    "Helping costs you a little and gives the team a lot.")
        reward = st.radio("Reward each agent receives", ["individual", "shared"], horizontal=True, key="p7_rew",
                          format_func=lambda s: "Individual: its own score" if s == "individual" else "Shared: the team's average score")
        predict("with individual rewards, will the agents learn to help each other?")
        g = te.helping_game(400, 0.4, 0.99, 0.1, reward, 3)
        rows = g["rows"]
        for r in rows:
            win = rows[max(0, r["Episode"] - 25):r["Episode"]]
            r["Help rate (last 25)"] = round(sum(x["Helps this round"] for x in win) / (2.0 * len(win)), 3)
            r["Team score (last 25)"] = round(sum(x["Team score"] for x in win) / len(win), 3)
        m = st.columns(2)
        m[0].metric("Help rate, last 100 rounds", "%.0f%%" % (100 * sum(r["Helps this round"] for r in rows[-100:]) / 200.0))
        m[1].metric("Average team score, last 100", "%.2f" % (sum(r["Team score"] for r in rows[-100:]) / 100.0))
        _line(rows, "Episode", ["Help rate (last 25)"], "How often agents help")
        _line(rows, "Episode", ["Team score (last 25)"], "Team score (2.0 if nobody helps, 3.2 if both help)")
        notice("With individual rewards, not helping always pays you more, so helping dies out and the team scores 2.0. With a **shared reward**, your teammate's gain is your gain: "
               "agents learn to help even at a short-term cost, and the team reaches 3.2. 'If one succeeds, all succeed.'")
    with tabs[2]:
        dot('digraph { rankdir=LR; ' + NODE + ' I [label="Interaction"]; R [label="Reward /\\nfeedback"]; P [label="Policy\\nupdate"]; B [label="New\\nbehaviour"]; I -> R -> P -> B -> I; }')
        st.markdown("Each loop deepens collective intelligence until new patterns emerge: spontaneous hand-offs, efficient language conventions, adaptive turn-taking, implicit trust and specialisation.")
        df_show([
            {"C": "Communication", "What agents learn": "What to share, when, and with whom", "Example": "Send alerts only when a task is critically delayed, instead of a fixed pub/sub firehose"},
            {"C": "Coordination", "What agents learn": "Task and resource allocation strategies", "Example": "Research finishing before writing starts beats doing them in parallel"},
            {"C": "Cooperation", "What agents learn": "Helping each other maximises the shared reward", "Example": "Accepting a short-term cost so the team does better"},
        ])


# ----------------------------------------------------------------------------------------
# Page 8: learning loops
# ----------------------------------------------------------------------------------------
def page_loops():
    st.header("8 · Learning loops: peer feedback, self-play, rewards")
    plain("Agents learn from each other through communication (peer feedback), coordination (finding the best workflow) and cooperation (a shared reward). "
          "With LLM agents the weights usually don't change: the 'learning' is stored lessons, chosen workflows and better prompts.")
    tabs = st.tabs(["🔴 Self-play critic (RLPF)", "Learning a workflow", "Designing the reward"])
    with tabs[0]:
        st.markdown("**Reinforcement learning with peer feedback (RLPF)**: like RLHF, but another LLM agent supplies the score. Each Writer → Critic cycle is one loop.")
        b = get_backend()
        topic = st.text_input("Topic", "electric vehicles", key="p8_topic")
        rounds = st.slider("Rounds", 1, 3, 3, key="p8_rounds")
        if st.button("Run the writer and critic", type="primary", key="p8_go"):
            draft, history = "", []
            for i in range(rounds):
                demo = te.SELF_PLAY_DEMO[min(i, 2)]
                w_req = ("Write a 3-sentence factual paragraph about %s." % topic) + (("\nRevise using this critique:\n" + history[-1]["critique"] + "\nPrevious draft:\n" + draft) if history else "")
                draft = b.ask("You are a Writer agent. Be accurate and hedge uncertain claims.", w_req, demo["draft"])
                crit = b.ask("You are a Critique agent. Start with 'Score N/10.' then give 2 short, specific fixes.", "Critique this paragraph:\n" + draft, demo["critique"])
                history.append({"round": i + 1, "draft": draft, "critique": crit})
            for h in history:
                with st.container(border=True):
                    st.markdown("**Round %d · Writer**" % h["round"])
                    st.write(h["draft"])
                    st.markdown("**Critic**")
                    st.write(h["critique"])
            badge(b)
        notice("Over many cycles the writer anticipates the critique, the critic learns to give targeted feedback, and the team develops implicit quality standards: "
               "cooperation shifts from task sharing to mutual growth (the post-read's self-play reflection).")
        show_code("A self-play loop in a few lines (LangChain, like Session 5's ask() helper)", '''
def ask(instructions, request):
    return llm.invoke([SystemMessage(content=instructions), HumanMessage(content=request)]).text

draft = ask("You are a Writer agent.", "Write a 3-sentence paragraph about electric vehicles.")
for round in range(3):
    critique = ask("You are a Critique agent. Start with 'Score N/10.'", "Critique:\\n" + draft)
    if "Score 9" in critique or "Score 10" in critique:
        break                                        # good enough: stop the loop
    draft = ask("You are a Writer agent.", "Revise using this critique:\\n" + critique + "\\n\\nDraft:\\n" + draft)''')
    with tabs[1]:
        kind = st.radio("What is being learned?", list(te.WORKFLOWS), horizontal=True, key="p8_kind")
        st.caption(te.WORKFLOWS[kind]["story"])
        a, b = st.columns(2)
        eps = a.slider("Exploration rate", 0.0, 0.6, 0.15, 0.05, key="p8_eps")
        rounds = b.slider("Runs", 20, 300, 120, 10, key="p8_runs")
        r = te.learn_workflow(kind, rounds, eps)
        df_show(r["table"])
        st.success("Learned choice: **%s**" % r["best"])
        _line(r["rows"], "Round", ["Average so far"], "Average reviewer score over time")
        notice("This is learning-based coordination in miniature: try, get a reward, update, mostly pick the best so far. With no exploration it can lock onto a worse option it got lucky with; "
               "with too much it keeps wasting runs on bad options.")
    with tabs[2]:
        st.markdown("The reward defines what agents strive for, and therefore what they learn. Choose weights and see which behaviour the reward favours.")
        preset = st.selectbox("Preset", list(te.REWARD_PRESETS), key="p8_preset")
        w = te.REWARD_PRESETS[preset]
        c = st.columns(4)
        wt = c[0].slider("Team quality +", 0.0, 1.5, w[0], 0.1, key="p8_wt_" + preset)
        wv = c[1].slider("Verified truth +", 0.0, 1.5, w[1], 0.1, key="p8_wv_" + preset)
        wr = c[2].slider("Redundancy −", 0.0, 1.5, w[2], 0.1, key="p8_wr_" + preset)
        wo = c[3].slider("Overconfidence −", 0.0, 1.5, w[3], 0.1, key="p8_wo_" + preset)
        rows = te.score_behaviours(wt, wv, wr, wo)
        df_show(rows)
        st.success("The behaviour this reward teaches: **%s**" % rows[0]["Behaviour"])
        notice("Team reward alone can reward confident bluffing that looks good. Truth alone can reward working in isolation. The post-read's answer is a dual reward: "
               "shared team reward + individual reward for verifiable truth − penalties for redundancy and overconfidence. Agents then seek truth, not just consensus.")


# ----------------------------------------------------------------------------------------
# Page 9: review
# ----------------------------------------------------------------------------------------
def page_review():
    st.header("9 · Review, quiz & cheat sheet")
    st.subheader("🧠 Quiz")
    with st.form("quiz7"):
        answers = [st.radio("**%d. %s**" % (i + 1, q), opts, index=None, key="q7_%d" % i) for i, (q, opts, c, w) in enumerate(te.QUIZ)]
        submitted = st.form_submit_button("Check my answers", type="primary")
    if submitted:
        score = 0
        for i, (q, opts, c, why) in enumerate(te.QUIZ):
            ok = answers[i] == opts[c]
            score += ok
            (st.success if ok else st.error)("**Q%d:** %s. %s%s" % (i + 1, "Correct" if ok else "Not quite", "" if ok else "Answer: **%s**. " % opts[c], why))
        st.metric("Score", "%d / %d" % (score, len(te.QUIZ)))
    st.subheader("⚡ Cheat sheet")
    df_show([
        {"Idea": "Three C's", "One line": "Communication (share info) → Coordination (who does what, when) → Cooperation (help each other succeed)"},
        {"Idea": "Failure cascade", "One line": "Poor communication → failed coordination → impossible cooperation"},
        {"Idea": "Request–response", "One line": "Direct, 1:1, synchronous: simple and traceable; blocking, not scalable"},
        {"Idea": "Publish–subscribe", "One line": "Broadcast, 1:n, asynchronous: scalable, decoupled; no feedback, ordering, overload"},
        {"Idea": "Blackboard", "One line": "Shared memory, n:n, asynchronous: flexible; needs conflict management, consistency issues"},
        {"Idea": "Core problems", "One line": "Task allocation (who does what), resource allocation (who gets the shared thing), conflict resolution (disagreements and clashes)"},
        {"Idea": "Mechanisms", "One line": "Centralized (one boss), distributed (negotiate), rule-based (SOPs), learning-based (rewards)"},
        {"Idea": "CrewAI manager/worker", "One line": "Process.hierarchical + manager_llm; leave tasks unassigned"},
        {"Idea": "CrewAI pipeline", "One line": "Process.sequential + context=[...]; the tasks list order IS the run order"},
        {"Idea": "CrewAI peer/swarm", "One line": "Process.sequential + allow_delegation=True on peers"},
        {"Idea": "CrewAI federated", "One line": "manager_agent with no tools, not in agents=[...]"},
        {"Idea": "context in CrewAI 1.15", "One line": "Not set → all earlier outputs; [] → nothing; [t1] → only t1; a later task → error at Crew(...)"},
        {"Idea": "MARL", "One line": "Agents learn the three C's from rewards; other learners make the environment non-stationary"},
        {"Idea": "RLPF", "One line": "Reinforcement learning from peer feedback: other agents provide the reward signal"},
        {"Idea": "Good rewards", "One line": "Shared team reward + individual truthfulness − penalties for redundancy and overconfidence"},
    ])
    st.subheader("📝 Notebook exercises")
    df_show([
        {"Exercise": "Add a fourth agent to the pipeline (e.g. a fact-checker between analyst and writer)", "Try it": "Page 6: add a task and point the writer's context at it"},
        {"Exercise": "Turn the manager/worker crew into a pipeline: what changes?", "Try it": "Page 6: switch process and assign agents"},
        {"Exercise": "Which communication pattern does each demo use?", "Try it": "Page 2, Demo 1 tab"},
        {"Exercise": "Design a reward for your own project (post-read 5.3 and 5.4)", "Try it": "Page 8, reward tab"},
    ])
    st.subheader("Conclusions")
    st.markdown("""
1. **The Three C's form a hierarchy.** Communication enables information exchange, coordination organises actions, cooperation creates synergy; each depends on the layer below.
2. **Communication is the enabler, not the goal.** Its value appears only when it supports coordination and cooperation.
3. **Coordination solves core operational problems** (task allocation, resource sharing, conflict resolution) through centralized control, distributed negotiation, rules or learning.
4. **Cooperation lifts performance** through joint execution, resource sharing and feedback loops: adaptability and resilience.
5. **Integration is the key.** Excellence in one C is not enough; a balanced mix makes a scalable, collaborative MAS.
6. **And beyond rules, teams can learn** (MARL): from programmed behaviour to self-improving collaboration.
""")
    st.info("Next session: agents need shared rules to talk to tools and to each other across apps and companies: the MCP and A2A protocols.")


# ----------------------------------------------------------------------------------------
# Sidebar + router
# ----------------------------------------------------------------------------------------
def sidebar():
    st.sidebar.title("🔺 Three C's Lab")
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
    {PAGES[0]: page_home, PAGES[1]: page_pyramid, PAGES[2]: page_comm, PAGES[3]: page_problems, PAGES[4]: page_mechanisms,
     PAGES[5]: page_patterns, PAGES[6]: page_debugger, PAGES[7]: page_marl, PAGES[8]: page_loops, PAGES[9]: page_review}[page]()


main()
