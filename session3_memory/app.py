"""
Memory Lab: an interactive companion to "Agents, Session 3: Memory in AI Agents".

Run:   streamlit run app.py
Works with NO API key (Demo mode: simulated answers). Add a Gemini key in the sidebar for Live mode.
"""
import html
import os

import pandas as pd
import streamlit as st

import memory_engine as me

st.set_page_config(page_title="Memory Lab: AI agent memory", page_icon="🧠", layout="wide")

PAGES = [
    "🏠 Start here",
    "1 · Why memory matters",
    "2 · Vector DB vs knowledge graph",
    "3 · Short-term memory (Ticket 1)",
    "4 · Long-term lifecycle (Tickets 2 & 5)",
    "5 · Forgetting at scale",
    "6 · Semantic, episodic, procedural (Ticket 4)",
    "7 · When memory goes wrong (Ticket 3)",
    "8 · Review, quiz & cheat sheet",
]


# ----------------------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------------------
def _default_key() -> str:
    key = os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("GOOGLE_API_KEY", "")
        except Exception:
            key = ""
    return key


def _notify(msg: str):
    """Live status line in the sidebar, updated while a Gemini call is running."""
    ph = st.session_state.get("_ph")
    if ph is not None:
        try:
            ph.info(msg)
        except Exception:
            pass


def get_backend() -> me.Backend:
    mode = st.session_state.get("mode", "Demo (no API key)")
    key = st.session_state.get("api_key", "") if mode.startswith("Live") else ""
    cfg = (key, st.session_state.get("chat_model", "gemini-3.5-flash-lite"),
           st.session_state.get("embed_model", "models/gemini-embedding-001"), int(st.session_state.get("rpm", 10)))
    if st.session_state.get("_cfg") != cfg or "backend" not in st.session_state:
        st.session_state.backend = me.Backend(*cfg)
        st.session_state._cfg = cfg
        if "store" in st.session_state:
            st.session_state.store.b = st.session_state.backend
    st.session_state.backend.notify = _notify
    if "store" not in st.session_state:
        reset_store()
    return st.session_state.backend


def reset_store():
    """Reset the LOCAL list to the notebook's sample data. Anything already in Pinecone is left alone."""
    old = st.session_state.get("store")
    remote = old.remote if old is not None else None
    store = me.MemoryStore(st.session_state.backend)
    me.seed_store(store)
    store.remote = remote            # attached AFTER seeding, so the sample data is not pushed to Pinecone
    st.session_state.store = store


def goto(page: str):
    st.session_state.nav = page


def badge(b: me.Backend):
    if b.last_source == "gemini":
        st.caption("🟢 Answer written by Gemini")
    elif "fallback" in b.last_source:
        st.caption("🟠 " + (b.error or "Live call failed; showing the demo simulation."))
    else:
        st.caption("🔵 Demo simulation (no API key): the reply reacts to what is in the prompt, like a real model would.")


def plain(text: str):
    st.info("**In plain words:** " + text)


def ticket(num: int, who: str, text: str):
    st.warning(f"**🎫 Ticket {num} ({who}):** {text}")


def show_code(title: str, code: str):
    with st.expander("🔍 " + title):
        st.code(code.strip("\n"), language="python")


def df_show(rows, **kw):
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, **kw)


# ----------------------------------------------------------------------------------------
# Page 0: Start
# ----------------------------------------------------------------------------------------
def page_home():
    st.title("🧠 Memory Lab: how AI agents remember")
    st.caption("An interactive companion to *Agents, Session 3: Memory in AI Agents* (the DilSeTrips notebook).")
    st.success("**The idea to hold on to:** an LLM remembers *nothing*. Every memory is something your code chose to "
               "**write down** and chose to **put back into the prompt**. Each kind of memory is a different answer to "
               "two questions: *what do we write down?* and *what do we put back?*")

    st.subheader("Five support tickets, five memory problems")
    st.write("DilSeTrips has been live for three weeks. Prasad opens the support inbox and finds five tickets. "
             "Each needs a different fix. Click a lab to reproduce the problem and fix it.")
    tickets = [
        (1, "Arjun", "I told it TWICE my wife is vegetarian. Twenty messages later it suggested butter chicken.",
         "Short-term memory overflowed", PAGES[3]),
        (2, "Prasad", "Every time I open the app it asks where I live.",
         "No long-term memory across sessions", PAGES[4]),
        (3, "Kavya", "Why does it think I'm diabetic? I only said no sugar in my chai.",
         "An eager extractor stored a guess as a fact", PAGES[7]),
        (4, "Prasad", "It keeps suggesting Manali. We WENT there in December.",
         "Facts, but no record of the event (episodic memory)", PAGES[6]),
        (5, "Arjun", "Please delete everything you know about me.",
         "Memory without a real deletion path", PAGES[4]),
    ]
    for num, who, text, cause, page in tickets:
        c1, c2, c3 = st.columns([5, 3, 1.4])
        c1.markdown(f"**Ticket {num}** ({who}): _{text}_")
        c2.markdown(f"**Cause:** {cause}")
        c3.button("Open lab", key=f"home_t{num}", on_click=goto, args=(page,))

    st.subheader("How to use this app")
    st.markdown(
        "- Go through the pages in the sidebar from **1 to 8**. Each page has an explanation, a hands-on lab and a code peek.\n"
        "- **Demo mode** (default) needs no key: embeddings and answers are simulated so you can see the *mechanics*.\n"
        "- **Live mode**: choose it in the sidebar and paste a Gemini key to run the same labs on real Gemini calls.\n"
        "- Everything you store lives in your browser session. **Reset memory store** in page 4 puts the notebook's sample data back."
    )

    with st.expander("📖 Glossary (read this first)"):
        glossary_table()

    st.subheader("Learning path")
    st.markdown("`Why memory` → `Where memory lives` → `Short-term` → `Long-term lifecycle` → `Forgetting` → "
                "`Three kinds of long-term memory` → `Failure modes` → `Quiz`")


def glossary_table():
    df_show([
        {"Word": "LLM", "Meaning": "The AI model (here Gemini). It reads text and writes text. It remembers nothing by itself."},
        {"Word": "Token", "Meaning": "The chunk of text an LLM reads and is billed by (about three-quarters of a word)."},
        {"Word": "Embedding", "Meaning": "A list of numbers capturing the meaning of a text. Similar meaning gives similar numbers."},
        {"Word": "Vector store", "Meaning": "A database that saves text + embedding so we can search by meaning."},
        {"Word": "Metadata", "Meaning": "Labels on each memory: user, kind, importance, date, status, version."},
        {"Word": "Structured output", "Meaning": "A fixed shape (like a form) the LLM must fill in, so we get clean data."},
        {"Word": "Semantic / Episodic / Procedural", "Meaning": "Facts / experiences / how-to rules."},
        {"Word": "Short-term vs long-term", "Meaning": "The current chat (context window) vs saved facts that survive into future sessions."},
        {"Word": "LLM-as-judge", "Meaning": "Using an LLM to score or check something, e.g. how useful a memory was."},
    ])


# ----------------------------------------------------------------------------------------
# Page 1: Why memory matters
# ----------------------------------------------------------------------------------------
def page_why():
    b = get_backend()
    st.header("1 · Why agents need memory")
    st.markdown(
        "Think of an LLM agent as a **junior travel consultant** who has read thousands of guides but, without memory, treats "
        "every client as brand new. Memory lets an agent **retain past interactions, learn from experience, keep context and "
        "personalise**. It is what makes an agent *feel* smart.")

    st.subheader("🧪 Lab: same model, same question, different memory")
    plain("the model forgets everything between calls, so a 'memory' is just text we paste into the prompt. "
          "Edit the saved memories below and watch the answers change.")
    question = st.text_input("Question asked to all three", "Explain what an embedding is. Keep it under 80 words.")
    cols = st.columns(3)
    defaults = [("Riya (school teacher)", "Saved facts about this user: she is a school teacher; she likes simple analogies and short answers."),
                ("Dr. Sen (AI researcher)", "Saved facts about this user: he is an AI researcher; he wants precise, technical detail."),
                ("Nobody (no memory)", "")]
    memories = []
    for col, (name, text) in zip(cols, defaults):
        with col:
            st.markdown(f"**{name}**")
            memories.append((name, st.text_area("Saved memory pasted into the system prompt", text, key=f"why_{name}",
                                                height=120, disabled=(name.startswith("Nobody")))))
    if st.button("▶ Run all three", type="primary"):
        results = []
        for name, mem in memories:
            system = "You are a helpful assistant.\n" + mem
            answer = b.ask([{"role": "system", "content": system}, {"role": "user", "content": question}],
                           lambda s=system: me.demo_persona_answer(s))
            results.append((name, answer))
        st.session_state.why_results = results
    if "why_results" in st.session_state:
        for col, (name, answer) in zip(st.columns(3), st.session_state.why_results):
            with col:
                st.markdown(f"**{name}**")
                st.write(answer)
        badge(b)
    st.caption("That is the whole trick: **better performance** (no starting from scratch), **better experience** "
               "(the answer fits you) and **personalisation at scale**.")

    st.subheader("What memory gives an agent (HealthMate example)")
    t1, t2, t3 = st.tabs(["1 · Contextual continuity", "2 · Personalisation", "3 · Learning & adaptation"])
    with t1:
        st.markdown("Morning: *\"remind me of my 10 am medicine\"*. Afternoon: *\"also vitamins after lunch\"*. Evening: *\"show today's reminders\"*.")
        df_show([{"": "Without memory", "Result": "Three isolated requests. The evening question gets 'what reminders?'"},
                 {"": "With memory", "Result": "One schedule: 10 am blood-pressure medicine, 1 pm vitamins after lunch."}])
        st.caption("Continuity spans multi-turn dialogue, sessions across days, and task-level understanding ('after lunch' is a condition).")
    with t2:
        df_show([{"": "Without memory", "Suggestion": "Chicken protein bowl. Run 30 minutes daily."},
                 {"": "With memory", "Suggestion": "High-protein paneer stir fry. 'You did great yesterday! How about a light evening walk?'"}])
        st.caption("Layers: profile memory (static facts), preference memory, behavioural memory (patterns), tone memory.")
    with t3:
        df_show([{"Week": 1, "What happens": "User asks for meals under 500 calories."},
                 {"Week": 2, "What happens": "Agent suggests 650 calories. User corrects it."},
                 {"Week": 3, "What happens": "It never repeats the mistake and explains: 'I avoided high-sugar options.'"}])
        st.caption("Error-correction memory, constraint memory ('vegetarian only') and optimisation memory (skip weekend workouts).")
    st.warning("⚠️ The same memory that makes HealthMate caring is dangerous when it is **wrong** (Ticket 3: a guessed 'diabetic'). "
               "Page 7 shows how to prevent that.")

    st.subheader("Traditional AI memory vs agentic AI memory")
    df_show([
        {"": "What it is", "Traditional": "A fixed database or rule set filled by people", "Agentic": "Memories written and retrieved by the agent as it works"},
        {"": "How it changes", "Traditional": "Only when a developer edits it", "Agentic": "Continuously, from every interaction"},
        {"": "Personal?", "Traditional": "No: same for every user", "Agentic": "Yes: per user"},
        {"": "Strength", "Traditional": "Predictable, auditable", "Agentic": "Context-aware, personalised, proactive"},
        {"": "Weakness", "Traditional": "Rigid; forgets you between sessions", "Agentic": "Can store wrong things, leak private data, grow without limit"},
    ])
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Classical booking system, session 2**")
        st.code("You: Book me a flight to London.\nSystem: Here are 40 flights. Choose seat? Meal? Hotel?", language="text")
    with c2:
        st.markdown("**Agentic travel agent, session 2**")
        st.code("You: Book me a flight to London.\nAgent: Window seat, as last time? Hilton again? "
                "You declined insurance before; want to reconsider?", language="text")
    st.caption("Agentic memory makes the agent **context-aware**, **personalised** and **proactive**.")

    st.subheader("Memory inside the agent loop")
    st.graphviz_chart('digraph { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef2ff", fontname="Helvetica"]; '
                      'Perceive -> "Remember / recall" -> "Think / plan" -> "Act (tools)" -> "Observe outcome" -> "Update memory"; '
                      '"Update memory" -> Perceive [style=dashed]; }')
    st.markdown("A personal librarian who remembers Alex likes mysteries can **store** (keep track), **categorise** "
                "(tasks, preferences, context) and **recall** (right memory, right time). A vanilla LLM is a *reactive* librarian with no memory.")
    st.graphviz_chart('digraph { rankdir=TB; node [shape=box, style="rounded,filled", fontname="Helvetica"]; '
                      'Memory [fillcolor="#fff3bf"]; "Short-term (working)\\ncontext window" [fillcolor="#d0ebff"]; '
                      '"Long-term\\ndatabases" [fillcolor="#d3f9d8"]; Semantic [fillcolor="#d3f9d8"]; Episodic [fillcolor="#d3f9d8"]; '
                      'Procedural [fillcolor="#d3f9d8"]; Memory -> "Short-term (working)\\ncontext window"; Memory -> "Long-term\\ndatabases"; '
                      '"Long-term\\ndatabases" -> Semantic [label=facts]; "Long-term\\ndatabases" -> Episodic [label=experiences]; '
                      '"Long-term\\ndatabases" -> Procedural [label="how-to"]; }')


# ----------------------------------------------------------------------------------------
# Page 2: Vector DB vs knowledge graph
# ----------------------------------------------------------------------------------------
def page_storage():
    b = get_backend()
    st.header("2 · Where memory lives: vector databases and knowledge graphs")
    tab_v, tab_g, tab_c = st.tabs(["Vector database", "Knowledge graph (lab)", "Compare"])

    with tab_v:
        st.markdown("A **vector database** stores each memory as an **embedding** so it can be searched by **meaning**, not keywords. "
                    "'I prefer vegetarian meals' becomes numbers; later, 'book me a flight' finds related memories.")
        plain("each line below becomes a list of numbers. We compare the question's numbers with each line's numbers; "
              "the closest lines are the 'relevant memories'.")
        facts = st.text_area("Memories (one per line)", "Prasad prefers vegetarian meals.\nPrasad gets sick on long bus rides.\n"
                             "Prasad lives in Pune.\nPrasad prefers window seats on flights.\nMall Road Grand is a noisy hotel in Manali.\n"
                             "Prasad once asked about beach shacks in Goa.", height=160)
        query = st.text_input("Question", "Book me a flight to New York")
        if st.button("🔎 Search by meaning", type="primary"):
            lines = [x.strip() for x in facts.splitlines() if x.strip()]
            vecs = b.embed_many(lines + [query])
            qv = vecs[-1]
            st.session_state.vec_result = sorted(
                [(line, b.similarity(qv, v)) for line, v in zip(lines, vecs[:-1])], key=lambda x: x[1], reverse=True)
            st.session_state.vec_mode = b.embedder_name
        if "vec_result" in st.session_state:
            res = st.session_state.vec_result
            st.bar_chart(pd.DataFrame({"similarity": [r[1] for r in res]}, index=[r[0] for r in res]))
            st.caption(f"Embedding used: **{st.session_state.vec_mode}** "
                       "(toy = a small offline stand-in; gemini = real embeddings). Real embeddings understand far more than this toy.")
        show_code("Code behind it", '''
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=KEY)
vectors = embeddings.embed_documents(list_of_memories)     # one list of numbers per memory
query_vec = embeddings.embed_query("Book me a flight to New York")
# similarity = cosine(query_vec, memory_vec); keep the top matches
''')

    with tab_g:
        st.markdown("A **knowledge graph** stores **entities** as nodes and **relationships** as edges, so memory becomes something you "
                    "can *reason over*: `User → prefers → Hilton → located in → London` ⇒ recommend Hilton London first.")
        plain("we turn sentences into small `(thing) --relation--> (thing)` facts. A hotel that matches *every* want is the answer, "
              "and the path is the proof.")
        facts = st.text_area("Facts in plain English", me.SAMPLE_FACTS, height=150, key="kg_facts")
        normalise = st.checkbox("Normalise qualities (so 'quiet hotels' and 'quiet' become the same node)", value=True)
        people = st.text_input("Who is asking? (comma separated)", "Prasad, Prasad's partner")
        if st.button("🕸 Build graph and follow the paths", type="primary"):
            prompt = ("Extract every fact as a triple. Use the exact names of people and places as subjects.\n"
                      + ("Normalise qualities to exactly one of these objects: quiet, noisy, vegetarian food.\n" if normalise else "")
                      + "Use the relation 'wants' for what a person wants or needs.\n\n" + facts)
            out = b.structured(me.TRIPLES_SCHEMA, prompt, lambda: {"triples": me.demo_triples(facts, normalise)})
            st.session_state.graph = out["triples"]
            st.session_state.graph_people = [p.strip() for p in people.split(",") if p.strip()]
            st.session_state.graph_src = b.last_source
        if "graph" in st.session_state:
            graph, names = st.session_state.graph, st.session_state.graph_people
            st.markdown("**The graph**")
            st.graphviz_chart(me.graph_dot(graph, highlight=names))
            df_show(graph)
            paths = me.graph_paths(graph, names)
            st.markdown("**Paths that connect the people's wants to hotels**")
            if paths:
                for p in paths:
                    st.code(p["text"], language="text")
                hotels, count = me.best_hotels(paths)
                wants = {p["want"].lower() for p in paths}
                if count == len(wants) and len(hotels) == 1:
                    st.success(f"✅ **{hotels[0]}** is the only hotel on paths for all {count} wants, and the edges above prove it.")
                elif count == len(wants):
                    st.warning(f"Several hotels match all {count} wants: {', '.join(hotels)}.")
                else:
                    st.error("No hotel satisfies every want. Check the entity names: the paths could not meet.")
            else:
                st.error("No paths found. The graph has no shared nodes between the people and the hotels.")
            if not normalise:
                st.warning("**Entity resolution problem:** 'quiet hotels' ≠ 'quiet', so the paths never meet. "
                           "This is why graphs are powerful but laborious to build, and why most agents start with a vector store.")
        with st.expander("Multi-hop example from the session document"):
            st.markdown(
                "Question: *which part of New York City is the director of the film **Big Stone Gap** based in?*\n\n"
                "1. **Semantic retrieval** finds chunks about the film, its director and a distractor film (*I Love NY*). None says where the director is based.\n"
                "2. A small **knowledge graph** from those chunks shows the triple `(Big Stone Gap, directed by, Adriana Trigiani)` and a **missing** 'based in' edge.\n"
                "3. **Graph-guided expansion** fetches the missing fact: *based in Greenwich Village*.\n"
                "4. **KG-based organisation** keeps only the path film → director → place and drops the distractor.\n"
                "5. The **LLM** reads that evidence and answers: *Greenwich Village*.\n\n"
                "Embeddings find things that *look* related. A graph gives *precise connections you can follow*.")
        show_code("Code behind it", '''
triple_llm = llm.with_structured_output(triples_schema)       # Gemini must answer as a list of triples
graph = triple_llm.invoke("Extract every fact as a triple ...\\n" + facts)["triples"]

def subjects_with(object_text):                                # reverse lookup: who points at 'quiet'?
    return [t["subject"] for t in graph if t["object"].lower() == object_text.lower()]
''')

    with tab_c:
        df_show([
            {"": "Stores", "Vector database": "Text + embedding (numbers for meaning)", "Knowledge graph": "Entities (nodes) + relations (edges)"},
            {"": "Finds", "Vector database": "Things that *look* related", "Knowledge graph": "Precise connections you can follow"},
            {"": "Good at", "Vector database": "Fuzzy recall, easy to build, scales", "Knowledge graph": "Multi-hop reasoning, explainable paths"},
            {"": "Weak at", "Vector database": "Multi-hop questions, exact relations", "Knowledge graph": "Entity resolution; laborious to build"},
            {"": "Typical start", "Vector database": "Most LLM agents", "Knowledge graph": "Added when relationships matter"},
        ])


# ----------------------------------------------------------------------------------------
# Page 3: Short-term memory
# ----------------------------------------------------------------------------------------
try:
    STRETCH_ALT = {"width": "stretch"} if "width" in __import__("inspect").signature(st.altair_chart).parameters else {"use_container_width": True}
except Exception:
    STRETCH_ALT = {"use_container_width": True}


def current_conversation(allergy: bool):
    conv = [dict(m) for m in me.CONVERSATION]
    if allergy:
        conv[0]["content"] += " Neha also has a peanut allergy."
    return conv


def page_short():
    b = get_backend()
    st.header("3 · Short-term memory: the scratchpad")
    ticket(1, "Arjun", "I told it TWICE that my wife is vegetarian. Twenty messages later it suggested butter chicken in Amritsar.")
    st.markdown("Short-term memory is the **context window**: the limited scratchpad holding the most recent conversation. "
                "It is **volatile** (lasts one session), **fast** and **limited**. When it overflows, the oldest text is silently cut.")
    allergy = st.checkbox("Homework 2: add 'Neha also has a peanut allergy' to the first message", value=False)
    conv = current_conversation(allergy)

    t1, t2, t3, t4 = st.tabs(["Token growth", "Forgetting by overflow", "Five strategies", "Trade-offs"])

    with t1:
        st.markdown("**What this tab shows.** The model has no memory, so on *every* turn your code re-sends the whole chat so far. "
                    "The prompt grows with each message and you pay for all of it again each time. "
                    "That growth is why short-term memory needs a strategy.")
        tok = [me.approx_tokens(m["content"]) + 3 for m in conv]
        w_n = st.slider("Compare 'send everything' with a window of the last N messages", 2, 8, 4, key="tg_window")
        full, win, cum_full, cum_win = [], [], [], []
        for i in range(len(conv)):
            full.append(sum(tok[:i + 1]))
            win.append(sum(tok[max(0, i + 1 - w_n):i + 1]))
            cum_full.append(sum(full))
            cum_win.append(sum(win))
        labels = ["msg %02d" % (i + 1) for i in range(len(conv))]
        # ---- Two charts, one conversation. Both share the SAME x-axis and the SAME two lines. ----
        # X axis (both charts): the message number in the chat, msg 01 ... msg 12.
        # The two lines (both charts): "send everything" = re-send the whole chat every call;
        #                              "last N messages only" = re-send just the newest N messages (the window).
        # What differs is the Y axis: LEFT = tokens in ONE call, RIGHT = tokens summed over ALL calls so far.
        import altair as alt
        win_name = "last %d messages only" % w_n
        strategies = ["send everything", win_name]
        colours = ["#8ec5ff", "#0066cc"]

        def line_chart(values_full, values_win, y_title):
            df = pd.DataFrame({"msg": labels * 2, "tokens": values_full + values_win,
                               "strategy": [strategies[0]] * len(labels) + [strategies[1]] * len(labels)})
            return alt.Chart(df).mark_line(point=True).encode(
                x=alt.X("msg:N", sort=labels, title="message number in the chat", axis=alt.Axis(labelAngle=-90)),
                y=alt.Y("tokens:Q", title=y_title),
                color=alt.Color("strategy:N", scale=alt.Scale(domain=strategies, range=colours), legend=alt.Legend(orient="bottom", title=None)),
                tooltip=["msg", "strategy", "tokens"])

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Left · one call:** how big is the prompt on message N? (tokens re-sent)")
            # Y axis: tokens sent to the model on THIS one call.
            #   send everything -> sum of every message so far (grows each turn)
            #   window          -> sum of only the newest N messages (flat once the window is full)
            drop = pd.DataFrame({"msg": [labels[min(w_n, len(labels) - 1)]], "text": ["message 1 (\"wife is vegetarian\") drops out here"]})
            rule = alt.Chart(drop).mark_rule(color="#d9480f", strokeDash=[5, 4], strokeWidth=2).encode(x=alt.X("msg:N", sort=labels))
            note = alt.Chart(drop).mark_text(align="right", dx=-6, dy=-4, baseline="bottom", color="#d9480f", fontWeight="bold", y=0).encode(x=alt.X("msg:N", sort=labels), text="text:N")
            st.altair_chart((line_chart(full, win, "tokens in this one call") + rule + note).properties(height=320), **STRETCH_ALT)
            st.caption("**How to read it.** **X axis:** the message number. **Y axis:** how many tokens your code sends to the model on that one call. "
                       "Identical until the window fills. After the dashed line the window no longer contains message 1, so the model cannot see that Neha is vegetarian. "
                       "'Send everything' still climbs; the window stays flat. Cheaper, but forgetful.")
        with c2:
            st.markdown("**Right · all calls added up:** the running bill after message N (tokens)")
            # Y axis: running total = this call's size + every earlier call's size (what you have been billed so far).
            #   Adding up a growing list makes a curve (about the square of the chat length);
            #   adding up a flat list makes a straight line.
            st.altair_chart(line_chart(cum_full, cum_win, "total tokens billed so far").properties(height=320), **STRETCH_ALT)
            st.caption("**How to read it.** **X axis:** the same message number. **Y axis:** the total tokens you have paid for across all calls up to that message. "
                       "Each point is the left chart's values added up. 'Send everything' bends upward because each call re-pays for all earlier messages; "
                       "the window is nearly a straight line because every call costs about the same.")
        m1, m2, m3 = st.columns(3)
        m1.metric("Billed so far: send everything", cum_full[-1])
        m2.metric("Billed so far: window of %d" % w_n, cum_win[-1])
        m3.metric("Saved by the window", "%d%%" % round(100 * (1 - cum_win[-1] / cum_full[-1])))
        avg = sum(tok) / len(tok)
        proj = []
        for n in (12, 60, 120):
            window_total = avg * (sum(min(i, w_n) for i in range(1, n + 1)))
            proj.append({"chat length": "%d messages" % n, "prompt on the last call (everything)": round(avg * n),
                         "prompt on the last call (window)": round(avg * min(n, w_n)),
                         "total billed (everything)": round(avg * n * (n + 1) / 2), "total billed (window)": round(window_total)})
        st.markdown("**Same chat, kept going** (extrapolated at the same pace)")
        df_show(proj)
        st.caption("A 10x longer chat makes the last prompt 10x bigger but the total bill about 100x bigger. That is the case for windows, summaries and entity memory.")
        limit = st.slider("Pretend the model can only read this many tokens at once", 50, 1000, 200, 10, key="tg_limit")
        first_over = next((i + 1 for i, v in enumerate(full) if v > limit), None)
        if first_over:
            st.error("With this limit, 'send everything' stops fitting from message %d onwards. From then on something must be dropped, summarised or stored elsewhere: that is the overflow on the next tab." % first_over)
        else:
            st.success("With this limit the whole chat still fits. (Real models read far more, but cost, speed and distraction still grow with length.)")
        with st.expander("Show the conversation"):
            for m in conv:
                with st.chat_message(m["role"]):
                    st.write(m["content"])

    with t2:
        st.markdown("Forgetting here is **structural and sudden**: a token is 100% visible or 0%, and the model never says 'I forgot'.")
        n = st.slider("Window size (messages kept)", 2, 12, 4)
        window = me.trim_last(conv, n)
        cut = len(conv) - len(window)
        chips = ""
        for i, m in enumerate(conv):
            colour = "#ffe3e3" if i < cut else "#d3f9d8"
            chips += (f'<div style="display:inline-block;margin:2px;padding:4px 8px;border-radius:6px;background:{colour};'
                      f'color:#222;font-size:12px">{i + 1}. {m["role"][0].upper()}: {html.escape(m["content"][:42])}…</div>')
        st.markdown(chips, unsafe_allow_html=True)
        st.caption(f"🟩 kept ({len(window)})   🟥 forgotten ({cut})")
        probe = st.text_area("Probe question (needs two early facts)", me.PROBE, height=70)
        if st.button("▶ Ask with the window vs the whole chat", type="primary"):
            ans_w = b.ask(window + [{"role": "user", "content": probe}], lambda: me.demo_probe_answer(window))
            ans_f = b.ask(conv + [{"role": "user", "content": probe}], lambda: me.demo_probe_answer(conv))
            st.session_state.overflow = (ans_w, ans_f, n)
        if "overflow" in st.session_state:
            ans_w, ans_f, n_used = st.session_state.overflow
            c1, c2 = st.columns(2)
            c1.error(f"**Window of {n_used} messages**\n\n{ans_w}")
            c2.success(f"**Whole conversation**\n\n{ans_f}")
            badge(b)
            st.caption("Nothing was 'ignored': the facts were simply **not in the prompt**. That is Ticket 1.")
        show_code("Code behind it", '''
from langchain_core.messages import trim_messages
small_window = trim_messages(conversation, max_tokens=4, token_counter=len, strategy="last", start_on="human")
reply = llm.invoke(small_window + [HumanMessage(content=probe)])    # window only
''')

    with t3:
        st.markdown("LangChain used to ship five ready-made memory classes (now deprecated). The **ideas** matter, and each is a few lines of code.")
        c1, c2 = st.columns(2)
        win_n = c1.slider("Window size for 'Window'", 2, 8, 4, key="strat_n")
        protect = c2.checkbox("Homework 2 fix: always pass importance-5 facts verbatim (never summarise them)", value=False)
        plain("five ways to remember one chat: **Buffer** (everything), **Window** (last few), **Summary** (short recap), "
              "**Summary + buffer** (recap + last few) and **Entity** (facts per person or thing). Same test question for all.")
        if b.live:
            per = (60 // b.rpm) if b.rpm and b.rpm > 0 else 2
            st.caption("This lab makes about 8 Gemini calls (3 to build the memories, 5 to answer). At your limit of %s calls/min that is roughly %d seconds. Progress shows in the sidebar." % (b.rpm if b.rpm else "unlimited", 8 * per))
        if st.button("▶ Build all five and test them", type="primary"):
            with st.spinner("Building memories and asking the model (Live mode can take a minute on the free tier)..."):
                crit = "Neha is strictly vegetarian." + (" Neha has a peanut allergy." if allergy else "")
                suffix = (" CRITICAL FACTS (verbatim): " + crit) if protect else ""
                s_all = b.ask([{"role": "user", "content": "Summarise this conversation in under 50 words, for your own memory:\n" + me.transcript(conv)}],
                              lambda: me.demo_summary(False, allergy, False)) + suffix
                s_old = b.ask([{"role": "user", "content": "Summarise this conversation in under 50 words, for your own memory:\n" + me.transcript(conv[:-4])}],
                              lambda: me.demo_summary(True, allergy, False)) + suffix
                ents = b.structured(me.ENTITY_SCHEMA, "List the key facts about each person, place or thing in this conversation:\n" + me.transcript(conv),
                                    lambda: {"entities": me.demo_entities(allergy)})["entities"]
                strategies = me.build_strategies(conv, win_n, s_all, s_old, ents)
                rows, answers = [], {}
                for label, msgs in strategies.items():
                    ans = b.ask(msgs + [{"role": "user", "content": me.PROBE}], lambda m=msgs: me.demo_probe_answer(m))
                    low = ans.lower()
                    mem_text = " ".join(m["content"] for m in msgs).lower()
                    row = {"Strategy": label, "Tokens sent": me.messages_tokens(msgs),
                           "Vegetarian?": "✅" if ("vegetarian" in low or "veg " in low) else "❌",
                           "Rs 3,200?": "✅" if ("3,200" in low or "3200" in low) else "❌"}
                    if allergy:
                        row["Allergy in memory?"] = "✅" if "peanut" in mem_text else "❌ lost/softened"
                    rows.append(row)
                    answers[label] = (msgs, ans)
                st.session_state.strat = (rows, answers, b.last_source)
        if "strat" in st.session_state:
            rows, answers, src = st.session_state.strat
            df_show(rows)
            st.caption("Fewer tokens is cheaper, but does it still *remember*? (✅/❌ are simple text checks on the model's answer.)")
            for label, (msgs, ans) in answers.items():
                with st.expander(f"{label}: what it keeps and what it answered"):
                    st.text("\n".join(f"[{m['role']}] {m['content']}" for m in msgs))
                    st.markdown("**Answer:** " + ans)
        df_show([
            {"Strategy": "Buffer", "Good at": "Exact recall of everything", "Weak at": "Grows forever; slow and costly", "Use when": "Short chats"},
            {"Strategy": "Window", "Good at": "Light and fast", "Weak at": "Nothing older than the window", "Use when": "Only the last few turns matter"},
            {"Strategy": "Summary", "Good at": "Long continuity in few tokens", "Weak at": "Loses exact details (prices, allergies)", "Use when": "The gist is enough"},
            {"Strategy": "Summary + buffer", "Good at": "Gist of old + exact recent", "Weak at": "Still no facts per entity", "Use when": "Usual default for chat agents"},
            {"Strategy": "Entity", "Good at": "Facts about people and things", "Weak at": "Loses conversation flow", "Use when": "Personalisation; must-survive facts"},
        ])
        st.caption("None of these survives the end of the session. For that we need long-term memory (page 4).")
        show_code("Code behind it (Summary + buffer)", '''
older_summary = llm.invoke("Summarise this conversation in under 50 words:\\n" + transcript(conversation[:-4])).text
memory = [SystemMessage(content="Summary of earlier conversation: " + older_summary)] + small_window
''')

    with t4:
        df_show([
            {"": "Advantages", "Small window": "Cheaper, faster, less noise, easy to debug", "Large window": "Longer coherence, critical details less likely dropped"},
            {"": "Drawbacks", "Small window": "Forgets older relevant details; users repeat themselves", "Large window": "Every token costs; slower; irrelevant detail distracts ('lost in the middle')"},
            {"": "Example", "Small window": "Banking bot forgets account type mid-session", "Large window": "200 pages of logs does not guarantee a better answer"},
        ])
        st.markdown("**Backpack rule:** pack the essentials (short-term), keep the rest in checked luggage (long-term): a *hybrid*.")
        st.markdown("**Prioritising inside the window**")
        df_show([
            {"Idea": "Naive eviction (FIFO)", "How": "Drop the oldest first", "Risk": "May drop the account ID, keep 'hello'"},
            {"Idea": "Relevance-aware eviction", "How": "Score messages, keep important ones", "Risk": "Extra compute"},
            {"Idea": "Summarise instead of drop", "How": "Compress old dialogue into a gist", "Risk": "Distortion ('peanut allergy' → 'food allergies')"},
            {"Idea": "Score = recency + similarity + importance", "How": "Pin allergies, drop small talk", "Risk": "Needs good importance flags"},
        ])


# ----------------------------------------------------------------------------------------
# Page 4: Long-term lifecycle
# ----------------------------------------------------------------------------------------
def generate(b, user, message, mode):
    rules = me.CAREFUL_RULES if mode == "careful" else me.EAGER_RULES
    prompt = rules + f"\nThe user's name is {user.title()}.\nMessage: {message}"
    out = b.structured(me.MEMORY_SCHEMA, prompt, lambda: {"memories": me.demo_generate_memories(user.title(), message, mode)})
    return out["memories"]


GEN_PRESETS = {
    "Benign: Pune + getaway": ("prasad", "Hi again! We live in Pune, by the way. We'd love a relaxed weekend getaway we can drive to. Lovely weather today, isn't it?"),
    "Health-inference trap (Kavya, Ticket 3)": ("kavya", "Plan us a cosy weekend in Coorg. Oh, and no sugar in my chai at the homestay please, I've given it up."),
    "Privacy trap (card + OTP)": ("kavya", "Book the Coorg homestay on my card 4111 1111 1111 1111, OTP is 482913. My phone is 98200 12345. We prefer a room with a balcony."),
    "Custom": ("prasad", "Type any message here."),
}


def _pc_key() -> str:
    key = os.environ.get("PINECONE_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("PINECONE_API_KEY", "")
        except Exception:
            key = ""
    return key


def pinecone_panel(b, store):
    st.markdown("#### ☁ Pinecone: a persistent long-term store")
    st.markdown("Right now memories live in this browser session only (they vanish when the app restarts). Pinecone is a real vector database: "
                "memories survive restarts, and each **user gets their own namespace**, so one user's search can never touch another's data. "
                "Once connected, **add, update, delete and search** all go to Pinecone automatically.")
    if not b.live:
        st.info("Pinecone stores real embeddings, so it needs **Live (Gemini)** mode (the offline toy embedding has a different size and would not fit the index). Switch to Live in the sidebar first.")
        return
    with st.expander("Connection settings", expanded=store.remote is None):
        c1, c2 = st.columns(2)
        key = c1.text_input("Pinecone API key", type="password", value=_pc_key(), key="pc_key",
                            help="From app.pinecone.io > API keys. Or set the PINECONE_API_KEY environment variable / secrets file.")
        index_name = c2.text_input("Index name", "dilsetrips-memory", key="pc_index")
        c3, c4 = st.columns(2)
        cloud = c3.selectbox("Cloud", ["aws", "gcp", "azure"], key="pc_cloud")
        region = c4.text_input("Region", "us-east-1", key="pc_region", help="Free (Starter) plans are limited to aws us-east-1.")
        st.caption("If the index does not exist it is created for you (serverless, cosine metric, dimension detected from your Gemini embedding model).")
        if st.button("🔌 Connect (creates the index if missing)", key="pc_connect", type="primary"):
            try:
                with st.spinner("Detecting the embedding size and connecting to Pinecone (a new index takes about a minute)..."):
                    dim = len(b.embed("dimension probe"))
                    store.remote = me.PineconeSync(key.strip(), index_name.strip(), cloud, region.strip(), dim)
                    store.remote_error = ""
                st.success("Connected to index '%s' (dimension %d)%s." % (index_name, store.remote.dimension, ", newly created" if store.remote.created else ""))
            except ImportError:
                store.remote = None
                st.error("The Pinecone package is missing. Run:  pip install pinecone")
            except Exception as exc:
                store.remote = None
                st.error("Could not connect: " + str(exc))
    if store.remote is None:
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Index", store.remote.index_name)
    c2.metric("Dimension", store.remote.dimension)
    if c3.button("🔄 Refresh Pinecone counts", key="pc_refresh"):
        try:
            st.session_state.pc_counts = store.remote.namespace_counts()
        except Exception as exc:
            st.error("Could not read stats: " + str(exc))
    counts = st.session_state.get("pc_counts")
    if counts is not None:
        df_show([{"namespace (= user)": ns, "vectors in Pinecone": n} for ns, n in counts.items()] or [{"namespace (= user)": "(index is empty)", "vectors in Pinecone": 0}])
    b1, b2 = st.columns(2)
    if b1.button("⬆ Push all local memories to Pinecone", key="pc_push"):
        try:
            with st.spinner("Embedding and uploading..."):
                n = store.push_all()
            st.success("Uploaded %d memories (same ids are overwritten). Pinecone can take a few seconds to show them in searches." % n)
        except Exception as exc:
            st.error("Upload failed: " + str(exc))
    if b2.button("⬇ Load memories from Pinecone (replaces the local list)", key="pc_pull"):
        try:
            with st.spinner("Reading from Pinecone..."):
                n = store.pull_from_remote()
            st.success("Loaded %d memories from Pinecone." % n)
            st.rerun()
        except Exception as exc:
            st.error("Load failed: " + str(exc))
    st.caption("First time: click Push. Coming back later: click Load (not Push), so you do not overwrite what is already stored.")
    if store.remote_error:
        st.warning(store.remote_error)


def page_long():
    b = get_backend()
    store: me.MemoryStore = st.session_state.store
    st.header("4 · Long-term memory: the full lifecycle")
    ticket(2, "Prasad", "Every time I open the app it asks me where I live. I've told it four times.")
    ticket(5, "Arjun", "Please delete everything you know about me.")
    st.markdown("Long-term memory is persistent: the agent's notebook across sessions. It is a **smart filing system with six operations**.")
    st.graphviz_chart('digraph { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#d3f9d8", fontname="Helvetica"]; '
                      'Generate -> Store -> Retrieve -> Integrate -> Update -> Delete; Delete -> Generate [style=dashed, label="new session"]; }')
    c1, c2 = st.columns([4, 1])
    c1.caption("The store below is shared by every tab on this page and starts with the notebook's sample memories for **prasad** and **arjun**.")
    c2.button("♻ Reset local memory list", on_click=reset_store)
    if st.session_state.get("flash"):
        st.success(st.session_state.pop("flash"))
    if store.remote_error:
        st.warning(store.remote_error)

    tabs = st.tabs(["1 Generate", "2 Store", "3 Retrieve", "4 Integrate", "5 Update", "6 Delete"])

    # --- Generate -----------------------------------------------------------------------
    with tabs[0]:
        st.markdown("Decide what is worth writing down: **extract** explicit facts, **summarise**, **embed**. "
                    "Pitfalls: over-capture (noise), under-capture (forget what matters), no provenance (cannot check later).")
        plain("instead of saving whole chats, an extractor picks a few facts. Each carries a kind, importance 1-5, confidence 0-1 and "
              "the user's **exact quote** as proof. Two extractors compete below: run both on the same message.")
        df_show([
            {"": "Instruction", "Careful extractor": "Only what the user EXPLICITLY said. Never guess. Skip card numbers, OTPs, small talk.",
             "Eager extractor": "Build a rich profile: infer health, lifestyle and preferences from hints."},
            {"": "Typical result", "Careful extractor": "'Kavya has given up sugar in her chai.'", "Eager extractor": "'Kavya may be diabetic.' (a guess, confidence 0.55)"},
            {"": "Risk", "Careful extractor": "Misses some useful hints", "Eager extractor": "A guess is saved as a fact and repeated forever (Ticket 3); may store sensitive data"},
        ])
        choice = st.selectbox("Example message", list(GEN_PRESETS), key="gen_preset")
        d_user, d_msg = GEN_PRESETS[choice]
        c1, c2 = st.columns([1, 3])
        user = c1.text_input("User id", d_user, key="gen_user_" + choice).strip().lower() or "prasad"
        message = c2.text_area("User message", d_msg, height=110, key="gen_msg_" + choice)
        with st.expander("See the exact rules each extractor is given"):
            st.markdown("**Careful**")
            st.code(me.CAREFUL_RULES, language="text")
            st.markdown("**Eager**")
            st.code(me.EAGER_RULES, language="text")
        if st.button("⚖ Run careful and eager side by side", type="primary", key="gen_go"):
            st.session_state.gen_cmp = {"user": user, "careful": generate(b, user, message, "careful"),
                                        "eager": generate(b, user, message, "eager"), "sens": me.find_sensitive(message)}
            st.session_state.gen_src = b.last_source
        if "gen_cmp" in st.session_state:
            res = st.session_state.gen_cmp
            if res["sens"]:
                st.error("Sensitive data in this message: " + ", ".join("%s (%s)" % (lab, val) for lab, val in res["sens"]))
            left, right = st.columns(2)
            for col, mode in ((left, "careful"), (right, "eager")):
                with col:
                    mems = res[mode]
                    (st.success if mode == "careful" else st.warning)("**%s extractor** stored %d memories" % (mode.upper(), len(mems)))
                    rows = []
                    for m in mems:
                        audit = me.audit_memory(m["text"], m["source_quote"], res["user"])
                        rows.append({"memory": m["text"], "importance": m["importance"], "confidence": m["confidence"],
                                     "exact quote": m["source_quote"], "quote supports it?": "✅ yes" if audit["supported"] else "❌ a guess"})
                    if rows:
                        df_show(rows)
                    else:
                        st.info("Nothing worth remembering.")
            badge(b)
            if [m["text"] for m in res["careful"]] == [m["text"] for m in res["eager"]]:
                st.info("Both extractors returned the same memories for this message, because it gives nothing to guess from. "
                        "Pick the **Kavya** or **card + OTP** example above to see them diverge.")
            save_mode = st.radio("Save which extractor's memories?", ["careful", "eager"], horizontal=True, key="gen_save_mode")
            if st.button("💾 Save to the store", key="gen_save"):
                for m in res[save_mode]:
                    store.add(res["user"], m["text"], m["kind"], m["importance"], m["confidence"], m["source_quote"])
                st.success("Saved %d memories for '%s'. Look at the Store tab%s." % (len(res[save_mode]), res["user"], " (and Pinecone)" if store.remote_ready() else ""))
        show_code("Code behind it", '''
memory_llm = llm.with_structured_output(memory_schema)          # text, kind, importance, confidence, source_quote
def generate_memories(user_name, message, rules=careful_rules):
    return memory_llm.invoke(rules + f"\nThe user's name is {user_name}.\nMessage: {message}")["memories"]
''')

    # --- Store --------------------------------------------------------------------------
    with tabs[1]:
        st.markdown("Text + embedding go into a **vector database**; **metadata** (user, kind, importance, date, status, version) goes alongside. "
                    "Good practice: raw logs in cold storage, condensed memories in hot storage, **one namespace per user**, encrypt everything.")
        pinecone_panel(b, store)
        st.divider()
        st.markdown("#### The memories (local view)")
        users = ["(all)"] + store.users()
        pick = st.selectbox("Show memories of", users, key="store_user")
        rows = store.rows(None if pick == "(all)" else pick)
        m1, m2, m3 = st.columns(3)
        m1.metric("Memories shown", len(rows))
        m2.metric("Active", sum(1 for r in rows if r["status"] == "active"))
        m3.metric("Users in store", len(store.users()))
        df_show(rows)
        if store.remote is None:
            st.caption("This list lives in RAM only: restarting the app erases it. Connect Pinecone above to make it persistent.")
        with st.form("manual_add"):
            st.markdown("**Add a memory by hand**")
            a, b2, c = st.columns(3)
            u = a.text_input("User", "prasad")
            kind = b2.selectbox("Kind", ["semantic", "episodic", "procedural"])
            imp = c.slider("Importance", 1, 5, 3)
            text = st.text_input("Memory (third person)", "Prasad prefers aisle seats on trains.")
            quote = st.text_input("Exact words it came from", "I like the aisle seat")
            if st.form_submit_button("Add") and text.strip():
                store.add(u.strip().lower(), text.strip(), kind, imp, 1.0, quote)
                st.session_state.flash = "Memory added%s." % (" and written to Pinecone" if store.remote_ready() else "")
                st.rerun()
        show_code("Code behind it", '''
memory_store = InMemoryVectorStore(embedding=embeddings)          # local version
metadata = {"memory_id": ..., "user_id": ..., "kind": ..., "importance": ..., "confidence": ...,
            "source_quote": ..., "stored_on": ..., "status": "active", "version": 1, "supersedes": ""}
memory_store.add_texts([text], metadatas=[metadata], ids=[memory_id])     # embeds + saves

# Pinecone version: same data, one namespace per user
index.upsert(vectors=[{"id": memory_id, "values": embedding, "metadata": {**metadata, "text": text}}], namespace=user_id)
''')

    # --- Retrieve -----------------------------------------------------------------------
    with tabs[2]:
        st.markdown("Retrieval = **pre-filter** (this user, active only) → **similarity search** → **re-rank** → **top-k**. "
                    "Similarity alone favours whatever sounds closest, however old or trivial, so we blend three signals:")
        st.latex(r"\text{score} = w_1 \times \text{similarity} + w_2 \times \text{recency} + w_3 \times \text{importance}")
        st.caption("recency = 1 / (1 + days_old / 30): today = 1.0, a month ago = 0.5. importance = rating / 5.")
        plain("how the agent *recalls*. Move the weights (Homework 1) and watch which memory wins. The search runs once; moving a slider only re-scores, so it stays fast.")
        c1, c2 = st.columns([1, 2])
        r_user = c1.selectbox("Search as", store.users() or ["prasad"], key="ret_user")
        k = c1.slider("k (memories to return)", 1, 8, 3, key="ret_k")
        query = c2.text_input("Question", "Plan a relaxed weekend getaway for us", key="ret_q")
        w1, w2, w3 = st.columns(3)
        ws = w1.slider("w₁ similarity", 0.0, 1.0, 0.6, 0.05), w2.slider("w₂ recency", 0.0, 1.0, 0.3, 0.05), w3.slider("w₃ importance", 0.0, 1.0, 0.1, 0.05)
        ranked = store.retrieve(r_user, query, k=8, weights=ws)
        if ranked:
            returned, rest = ranked[:k], ranked[k:]
            st.markdown("**✅ Returned to the prompt (top %d)**" % len(returned))
            for rank, (col, r) in enumerate(zip(st.columns(len(returned)), returned), start=1):
                with col:
                    with st.container(border=True):
                        st.markdown("**#%d  ·  score %.3f**" % (rank, r["score"]))
                        st.write(r["item"]["text"])
                        parts = {"similarity": r["similarity"] * ws[0], "recency": r["recency"] * ws[1], "importance": r["importance"] * ws[2]}
                        st.progress(min(1.0, r["similarity"]), text="similarity %.2f × %.2f = %.2f" % (r["similarity"], ws[0], parts["similarity"]))
                        st.progress(min(1.0, r["recency"]), text="recency %.2f × %.2f = %.2f  (%d days old)" % (r["recency"], ws[1], parts["recency"], r["days_old"]))
                        st.progress(min(1.0, r["importance"]), text="importance %.2f × %.2f = %.2f" % (r["importance"], ws[2], parts["importance"]))
                        st.caption("Ranked here mostly because of **%s**." % max(parts, key=parts.get))
            if rest:
                st.markdown("**Considered but not returned**")
                bar = lambda label: st.column_config.ProgressColumn(label, min_value=0.0, max_value=1.0, format="%.2f")
                st.dataframe(pd.DataFrame([{"memory": r["item"]["text"], "age (days)": r["days_old"], "similarity": round(r["similarity"], 2),
                                            "recency": round(r["recency"], 2), "importance": round(r["importance"], 2),
                                            "score": round(min(1.0, r["score"]), 3)} for r in rest]),
                             use_container_width=True, hide_index=True,
                             column_config={"similarity": bar("similarity"), "recency": bar("recency"), "importance": bar("importance"), "score": bar("score")})
            with st.expander("Show the arithmetic for every memory"):
                df_show([{"rank": i + 1, "memory": r["item"]["text"], "working": r["working"]} for i, r in enumerate(ranked)])
            others = len([it for it in store.all() if it["meta"]["user_id"] != r_user])
            st.caption("🔒 %d memories belonging to other users were never considered: the pre-filter%s removed them before similarity was computed. Embedding in use: **%s**." % (
                others, " (a Pinecone namespace)" if store.remote_ready() else "", b.embedder_name))
        else:
            st.info("No active memories for this user.")
        show_code("Code behind it", '''
candidates = memory_store.similarity_search_with_score(query, k=8, filter=is_this_users_active_memory)
for document, similarity in candidates:
    recency = 1 / (1 + days_old / 30)
    importance = document.metadata["importance"] / 5
    score = 0.6 * similarity + 0.3 * recency + 0.1 * importance

# Pinecone version of the pre-filter + similarity step
index.query(vector=query_vec, top_k=8, namespace=user_id, filter={"status": {"$eq": "active"}}, include_metadata=True)
''')

    # --- Integrate ----------------------------------------------------------------------
    with tabs[3]:
        st.markdown("Two patterns: **context-append** (put retrieved memories in the prompt) and **template-driven** (fill a fixed template, safer in high-risk domains). "
                    "Watch the **token budget**, show **provenance**, and flag **low confidence** so the agent asks first.")
        plain("the payoff. Same question in a brand-new session, with and without recalled memories. This is Ticket 2.")
        c1, c2 = st.columns([1, 3])
        i_user = c1.selectbox("Who is asking?", store.users() or ["prasad"], key="int_user")
        i_k = c1.slider("Memories to include", 1, 6, 4, key="int_k")
        question = c2.text_input("New-session question", "Suggest a weekend getaway for us. Keep it to three sentences.", key="int_q")
        ranked = store.retrieve(i_user, question, k=i_k)
        context = me.memory_context(ranked)
        st.markdown("**🧩 Memories added to the prompt (%d)**" % len(ranked))
        if ranked:
            df_show([{"#": i + 1, "memory": r["item"]["text"], "similarity": round(r["similarity"], 2), "score": round(r["score"], 3),
                      "stored on": r["item"]["meta"]["stored_on"], "the user's exact words": r["item"]["meta"]["source_quote"],
                      "confidence": r["item"]["meta"]["confidence"],
                      "flag": "⚠ unconfirmed: ask first" if r["item"]["meta"]["confidence"] < 0.6 else ""} for i, r in enumerate(ranked)])
            st.caption("Only these memories are injected. Every other memory stays in the store and costs no tokens.")
        else:
            st.warning("No memories were retrieved for this user, so the agent will behave as if it has never met them.")
        system = ("You are DilSeTrips. What you remember about this user:\n" + context
                  + "Use these memories, and mention the one that most shaped your suggestion.")
        with st.expander("👀 The final system prompt (memories included) that will be sent", expanded=True):
            st.code(system, language="text")
        if st.button("▶ Answer with and without memory", type="primary", key="int_go"):
            without = b.ask([{"role": "user", "content": question}], lambda: me.demo_getaway_answer(""))
            with_mem = b.ask([{"role": "system", "content": system}, {"role": "user", "content": question}],
                             lambda: me.demo_getaway_answer(context))
            st.session_state.integ = (without, with_mem)
        if "integ" in st.session_state:
            c1, c2 = st.columns(2)
            c1.error("**WITHOUT long-term memory**\n\n" + st.session_state.integ[0])
            c2.success("**WITH long-term memory**\n\n" + st.session_state.integ[1])
            badge(b)
        st.caption("Provenance (\"from the user's words on …\") lets users catch wrong memories. Never send sensitive memory text to third-party APIs without redaction.")

    # --- Update -------------------------------------------------------------------------
    with tabs[4]:
        st.markdown("People change. Strategies: **append and version** (best for audits), **overwrite** (only trivial facts), "
                    "and **regenerate derived summaries**. Classic pitfall: updating a fact but not the summaries built from it.")
        plain("when Prasad moves to Bengaluru we do not erase 'lives in Pune'. We mark it `superseded` and save a new version that links back."
              + (" With Pinecone connected, both changes are written there too." if store.remote_ready() else ""))
        u_user = st.selectbox("User", store.users() or ["prasad"], key="upd_user")
        new_fact = st.text_input("New fact", "Prasad lives in Bengaluru.", key="upd_fact")
        new_quote = st.text_input("Exact words", "We just moved to Bengaluru", key="upd_quote")
        if st.button("🧭 Which memory does this replace?", key="upd_find"):
            active = ""
            for it in store.of_user(u_user, active_only=True):
                active += f"{it['meta']['memory_id']}: {it['text']}\n"
            prompt = (f"New fact: {new_fact}\nExisting memories:\n{active}\n"
                      "Which single existing memory does the new fact make out of date? Answer with its id, or 'none'.")
            st.session_state.upd_decision = (u_user, b.structured(me.REPLACE_SCHEMA, prompt, lambda: me.demo_replacement(store, u_user, new_fact)))
        if "upd_decision" in st.session_state:
            d_user, decision = st.session_state.upd_decision
            st.write("**Model's decision:**", decision)
            badge(b)
            active_items = store.of_user(d_user, active_only=True)
            options = {f"{it['meta']['memory_id']}: {it['text']}": it["meta"]["memory_id"] for it in active_items}
            labels = list(options)
            default = next((i for i, lab in enumerate(labels) if options[lab] == decision.get("replaces_memory_id")), 0)
            if labels:
                chosen = st.selectbox("Memory to replace (you confirm: the notebook trusts the model blindly, which is Homework 4)", labels, index=default, key="upd_choice")
                if chosen in options and st.button("✅ Confirm and apply update", key="upd_apply"):
                    new_id = store.supersede(options[chosen], new_fact, new_quote)
                    st.session_state.flash = "New version saved as %s; the old one is now 'superseded'%s." % (new_id, " (both written to Pinecone)" if store.remote_ready() else "")
                    del st.session_state["upd_decision"]
                    st.rerun()
        chain = [r for r in store.rows(u_user) if r["version"] > 1 or r["status"] == "superseded"]
        if chain:
            st.markdown("**Version chain** (both versions exist for audit; only the active one can be retrieved)")
            df_show(chain)
        show_code("Code behind it", '''
def update_memory(old_memory_id, new_text, source_quote):
    old_meta["status"] = "superseded"                       # keep history, retire the old one
    memory_store.add_texts([old_text], metadatas=[old_meta], ids=[old_memory_id])
    return store_memory(..., version=old_meta["version"] + 1, supersedes=old_memory_id)

# Pinecone version
index.update(id=old_memory_id, set_metadata={"status": "superseded"}, namespace=user_id)
index.upsert(vectors=[new_vector], namespace=user_id)
''')

    # --- Delete -------------------------------------------------------------------------
    with tabs[5]:
        st.markdown("**Soft delete** flags a memory (hidden, but text *and embedding* remain). **Hard delete** removes both. "
                    "Only a hard delete honours a 'delete everything' request. Also **cascade**: summaries built from deleted memories must be regenerated.")
        df_show([
            {"": "What changes", "Soft delete": "One label: status = 'deleted'", "Hard delete": "The record itself is removed"},
            {"": "Text and embedding", "Soft delete": "Still stored", "Hard delete": "Erased"},
            {"": "Can a search find it?", "Soft delete": "Hidden only while every query filters on status='active'. A query without that filter still returns it.", "Hard delete": "No: there is nothing left to find"},
            {"": "Reversible?", "Soft delete": "Yes: set the status back", "Hard delete": "No (unless you keep a backup)"},
            {"": "Use it for", "Soft delete": "Old superseded versions, a short undo window, audit trails", "Hard delete": "'Delete everything about me': privacy requests"},
            {"": "In Pinecone", "Soft delete": "index.update(id, set_metadata={'status': 'deleted'}). The vector stays in the index and still counts toward storage.",
             "Hard delete": "index.delete(ids=[...], namespace=user). Vectors and their metadata are removed (it can take a moment to show)."},
        ])
        plain("a user asks 'delete everything' (Ticket 5). Try soft delete first and see that the data is still findable.")
        users = store.users()
        if not users:
            st.info("The store is empty. Click 'Reset local memory list' above.")
        else:
            d_user = st.selectbox("User asking to be forgotten", users, index=users.index("arjun") if "arjun" in users else 0, key="del_user")
            c1, c2 = st.columns(2)
            if c1.button("🙈 Soft delete (flag only)", key="del_soft"):
                n = store.soft_delete_user(d_user)
                st.warning("%d memories flagged as deleted. They are still stored%s." % (n, " in Pinecone too" if store.remote_ready() else ""))
            if c2.button("🗑 Hard delete (text + embedding)", type="primary", key="del_hard"):
                ids = store.hard_delete_user(d_user)
                st.session_state.flash = "%d memories erased for good%s." % (len(ids), " (and deleted from Pinecone)" if store.remote_ready() else "")
                st.rerun()
            remaining = store.of_user(d_user)
            st.metric("Records still stored for '%s'" % d_user, len(remaining))
            if remaining:
                df_show(store.rows(d_user))
            found = store.search("vegetarian travel preferences", user_id=d_user, active_only=False)
            if remaining and found:
                st.error("Proof it is NOT erased: a search that ignores status still finds %d of this user's memories%s." % (len(found), " in Pinecone" if store.remote_ready() else ""))
            elif not remaining:
                st.success("Nothing left for this user: a search that ignores status finds %d memories." % len(found))
            if store.remote_ready():
                st.markdown("**☁ Ask Pinecone directly**")
                if st.button("🔎 Check what Pinecone holds for this user", key="del_check"):
                    try:
                        counts = store.remote.namespace_counts()
                        hits = store.remote.query(d_user, store.b.embed("vegetarian travel preferences"), 10, None, False)
                        st.session_state.del_check = (d_user, counts.get(d_user, 0), len(hits))
                    except Exception as exc:
                        st.error("Pinecone check failed: " + str(exc))
                if "del_check" in st.session_state:
                    u, vec_count, nhits = st.session_state.del_check
                    st.write("Namespace **%s** holds **%d** vectors; a status-blind query returned **%d**. (Pinecone can lag a few seconds behind a delete.)" % (u, vec_count, nhits))
        show_code("Code behind it", '''
# Soft delete: only a label changes, text and embedding are still stored
meta["status"] = "deleted"; memory_store.add_texts([text], metadatas=[meta], ids=[id])
# Hard delete: remove text AND embedding
memory_store.delete(ids=arjun_ids)
# Pinecone
index.update(id=memory_id, set_metadata={"status": "deleted"}, namespace="arjun")     # soft
index.delete(ids=arjun_ids, namespace="arjun")                                         # hard
''')

    st.subheader("The lifecycle at a glance")
    df_show([
        {"Stage": "1 Generate", "Simple idea": "Decide what to write down", "Best practice": "Structured + raw text; timestamp and provenance", "Pitfall": "Over/under-capture; no provenance"},
        {"Stage": "2 Store", "Simple idea": "File it in the right folder", "Best practice": "Cold raw, hot condensed; per-user namespaces; encrypt", "Pitfall": "Mixing users; unprotected personal data"},
        {"Stage": "3 Retrieve", "Simple idea": "Search your notes", "Best practice": "Balance k; similarity + recency + importance", "Pitfall": "Stale or irrelevant results"},
        {"Stage": "4 Integrate", "Simple idea": "Ground the answer in the notes", "Best practice": "Show provenance; manage tokens; ask when unsure", "Pitfall": "Overloaded context; trusting memory blindly"},
        {"Stage": "5 Update", "Simple idea": "Keep notes fresh", "Best practice": "Append-only versions; regenerate summaries", "Pitfall": "Overwriting history; stale summaries"},
        {"Stage": "6 Delete", "Simple idea": "Throw notes away when asked/expired", "Best practice": "Honour requests; cascade to summaries", "Pitfall": "Embedding left behind; soft delete only"},
    ])


# ----------------------------------------------------------------------------------------
# Page 5: Forgetting at scale
# ----------------------------------------------------------------------------------------
def page_forget():
    get_backend()
    st.header("5 · Forgetting at scale: three strategies")
    st.markdown("You cannot keep everything forever: memory costs storage, slows retrieval, and old data is a privacy risk. "
                "So big systems **deliberately forget**.")
    df_show([
        {"Strategy": "1 Periodical (time-based)", "Deletes when": "Usage this month dropped far below last month", "Weakness": "Looks only at recency/frequency, not usefulness"},
        {"Strategy": "2 History-based (utility)", "Deletes when": "After n uses, average usefulness is low", "Weakness": "Keeps old useful memories forever"},
        {"Strategy": "3 Combined", "Deletes when": "BOTH: gone stale AND not useful", "Weakness": "Slowest to delete; usually the practical choice"},
    ])
    plain("each memory has a usage log and usefulness ratings (0-1, from an **LLM judge** in production). The numbers below are "
          "illustrative stand-ins for a month of traffic. Edit them and move the thresholds (Homework 6).")
    base = []
    for text, s in me.DEFAULT_USAGE.items():
        base.append({"memory": text, "uses_last_month": s["uses_previous_month"], "uses_this_month": s["uses_this_month"],
                     "avg_usefulness": round(sum(s["usefulness"]) / len(s["usefulness"]), 2), "times_rated": len(s["usefulness"])})
    edited = st.data_editor(pd.DataFrame(base), use_container_width=True, hide_index=True, num_rows="dynamic", key="usage_editor")
    edited = edited.dropna(subset=["memory", "uses_last_month", "uses_this_month"]).fillna(0)
    c1, c2, c3 = st.columns(3)
    drop = c1.slider("Periodical: delete if this month's use is below this fraction of last month's", 0.0, 1.0, 0.3, 0.05)
    useful = c2.slider("History: delete if average usefulness is below", 0.0, 1.0, 0.4, 0.05)
    min_uses = c3.slider("...once rated at least this many times", 1, 10, 3)
    rows = []
    for _, r in edited.iterrows():
        if not r["memory"] or r["uses_last_month"] == 0:
            continue
        ratio = r["uses_this_month"] / r["uses_last_month"]
        periodical = ratio < drop
        history = r["times_rated"] >= min_uses and r["avg_usefulness"] < useful
        verdict = lambda x: "🗑 DELETE" if x else "keep"
        rows.append({"memory": r["memory"], "use ratio": round(ratio, 2), "avg usefulness": r["avg_usefulness"],
                     "periodical": verdict(periodical), "history": verdict(history), "COMBINED": verdict(periodical and history)})
    if rows:
        df_show(rows)
    st.info("**Window seat** shows why *combined* exists: usage dropped (Prasad has not flown), so periodical would delete it, "
            "yet it was useful every time. **Goa beach shacks** is stale *and* useless, so all three agree to delete it. "
            "Try setting the thresholds very high: a user who travels once a year loses useful memories.")
    st.markdown("**What does 'delete' mean in big consumer apps?** In Google Photos, YouTube, Spotify or Netflix, forgetting usually means "
                "*lowering surfacing priority*, not erasing data. True deletion often happens only when privacy law requires it. "
                "Decide which kind of forgetting you promise users, and keep that promise.")
    show_code("Code behind it", '''
periodical = use_ratio < drop_threshold
history    = len(stats["usefulness"]) >= minimum_uses and average_usefulness < usefulness_threshold
combined   = periodical and history          # delete only if BOTH agree (the safe rule)
''')


# ----------------------------------------------------------------------------------------
# Page 6: Semantic, episodic, procedural
# ----------------------------------------------------------------------------------------
def page_kinds():
    b = get_backend()
    store: me.MemoryStore = st.session_state.store
    st.header("6 · Three kinds of long-term memory")
    ticket(4, "Prasad", "It keeps suggesting Manali for our next trip. We WENT to Manali in December. It even helped plan it.")
    t1, t2, t3, t4 = st.tabs(["Semantic: facts", "Episodic: experiences (lab)", "Procedural: how-to (lab)", "Compare"])

    with t1:
        st.markdown("**Semantic memory** is the agent's encyclopedia: general, impersonal facts. Three stages: **encoding** (identify entities, attributes, relations), "
                    "**storage** (often a knowledge graph), **retrieval** (follow the structure: France → has capital → Paris).")
        st.markdown("You have already built it twice: the triples on page 2, and facts like *'Prasad's partner is vegetarian'* on page 4. "
                    "Much of an LLM's general knowledge is semantic memory in its weights; stored facts add knowledge **about this user**.")

    with t2:
        st.markdown("**Episodic memory** is the diary: specific events (*what*, *where*, *when*). Semantic memory says Paris is the capital; "
                    "episodic memory recalls *your* trip to Paris. Stages: **encoding** with context, **consolidation**, **retrieval** by cue.")
        df_show([
            {"Use": "Planning", "How episodes help": "Past sequences become templates"},
            {"Use": "Imagination & prediction", "How episodes help": "Similar episodes simulate likely futures"},
            {"Use": "Problem solving", "How episodes help": "Solutions that worked are adapted"},
            {"Use": "Decision making", "How episodes help": "Past outcomes estimate the value of choices"},
            {"Use": "Learning from experience", "How episodes help": "Episodes are re-read and refined"},
        ])
        plain("DilSeTrips knows *facts* about Prasad but has no record of the *event*. We turn the trip story into a structured episode "
              "(what worked, what did not), save it, then plan the next trip with and without it. This fixes Ticket 4.")
        debrief = st.text_area("Trip story", me.SAMPLE_DEBRIEF, height=100)
        if st.button("📔 Turn the story into an episode", type="primary"):
            ep = b.structured(me.EPISODE_SCHEMA, "Record this trip as an episode for future planning:\n" + debrief,
                              lambda: me.demo_episode(debrief))
            st.session_state.episode = ep
        if "episode" in st.session_state:
            ep = st.session_state.episode
            st.json(ep)
            badge(b)
            text = me.episode_text(ep)
            if st.button("💾 Save this episode to Prasad's memory"):
                if not any(it["text"] == text for it in store.of_user("prasad")):
                    store.add("prasad", text, "episodic", 4, 1.0, debrief[:60])
                st.success("Saved as an episodic memory.")
        next_trip = "Where should we go for our next short trip in February? One suggestion, three sentences, and say why."
        if st.button("🧭 Plan the next trip: facts only vs facts + episodes"):
            outs = []
            for kinds in (["semantic"], ["semantic", "episodic"]):
                hits = store.search(next_trip, user_id="prasad", kinds=kinds, k=6)
                remembered = "".join("- " + it["text"] + "\n" for it, _ in hits)
                outs.append(b.ask([{"role": "system", "content": "You are DilSeTrips. What you remember about this couple:\n" + remembered},
                                   {"role": "user", "content": next_trip}], lambda r=remembered: me.demo_next_trip(r)))
            st.session_state.plan_cmp = outs
        if "plan_cmp" in st.session_state:
            c1, c2 = st.columns(2)
            c1.error("**Semantic memories only**\n\n" + st.session_state.plan_cmp[0])
            c2.success("**Semantic + episodic**\n\n" + st.session_state.plan_cmp[1])
            badge(b)
            st.caption("Caution (overfitting): an episode should **inform** future choices, not dictate them. One bad taxi ride should not ban road trips forever. "
                       "Homework 8: let old episodes matter less over time, and let the couple say 'we'd happily go back'.")
        if "plan_cmp" in st.session_state and not any("Trip episode" in it["text"] for it in store.of_user("prasad")):
            st.warning("Save the episode first (button above), then run the comparison again to see the difference.")

    with t3:
        st.markdown("**Procedural memory** is muscle memory: *how* to do things. Phases: **acquisition**, **consolidation**, **execution**, **adaptation**. "
                    "In an LLM agent, part of it lives in the agent's own **instructions**, which feedback can rewrite.")
        plain("the agent's working rules are a list. When a user complains, the model rewrites the rules so the mistake does not repeat.")
        rules_text = st.text_area("Current rules (one per line)", "\n".join(me.DEFAULT_PROCEDURE), height=90)
        feedback = st.text_area("User feedback", me.SAMPLE_FEEDBACK, height=90)
        if st.button("🛠 Rewrite the rules", type="primary"):
            rules = [r.strip() for r in rules_text.splitlines() if r.strip()]
            current = "".join("- " + r + "\n" for r in rules)
            prompt = ("These are your current working rules:\n" + current + "\nA user gave this feedback:\n" + feedback
                      + "\nRewrite the rules so this never happens again. Keep rules that are still right; make each rule short and specific.")
            out = b.structured(me.RULES_SCHEMA, prompt, lambda: {"rules": me.demo_rewrite_rules(rules, feedback)})
            st.session_state.new_rules = (rules, out["rules"])
        if "new_rules" in st.session_state:
            old, new = st.session_state.new_rules
            c1, c2 = st.columns(2)
            c1.markdown("**Before**")
            for r in old:
                c1.markdown(f"- {r}")
            c2.markdown("**After**")
            for r in new:
                c2.markdown(f"- {'🆕 ' if r not in old else ''}{r}")
            badge(b)
        st.warning("Procedural memory is **about how the agent works**, not about one user, so a bad rule spreads to everyone. "
                   "In production, procedural updates usually go through human review.")

    with t4:
        df_show([
            {"Aspect": "Focus", "Semantic": "Facts, concepts", "Episodic": "Personal experiences, events", "Procedural": "Skills, routines, how-to"},
            {"Aspect": "Human analogy", "Semantic": "Encyclopedia", "Episodic": "Diary", "Procedural": "Muscle memory"},
            {"Aspect": "Data stored", "Semantic": "Rules, definitions, relationships", "Episodic": "Episodes with time, place, outcome", "Procedural": "Step-by-step methods"},
            {"Aspect": "Retrieval", "Semantic": "Explicit, when needed", "Episodic": "Cued by the situation", "Procedural": "Implicit, automatic"},
            {"Aspect": "DilSeTrips example", "Semantic": "'Her partner is vegetarian'", "Episodic": "The December Manali trip", "Procedural": "'Suggest at most three options'"},
            {"Aspect": "About", "Semantic": "Usually a user", "Episodic": "Usually a user", "Procedural": "The agent itself"},
        ])


# ----------------------------------------------------------------------------------------
# Page 7: When memory goes wrong
# ----------------------------------------------------------------------------------------
def page_risks():
    b = get_backend()
    store: me.MemoryStore = st.session_state.store
    st.header("7 · When memory goes wrong")
    ticket(3, "Kavya", "Why does it think I'm diabetic? I only said I don't take sugar in my chai. Now every suggestion is 'diabetic-friendly'.")
    t1, t2, t3, t4 = st.tabs(["Forgetting vs overfitting", "Hallucination amplification (lab)", "Privacy & isolation (lab)", "Scaling"])

    with t1:
        st.markdown("**Forgetting:** retrieval favours recent chats and older facts fade or get 'summarised away'. "
                    "**Overfitting:** the agent clings to one old fact and treats it as always true.")
        df_show([
            {"Failure": "Forgetting", "Career-counsellor example": "Forgets 'I only want remote work', keeps suggesting office jobs", "Cause": "Top-k favours recent; summaries erase nuance"},
            {"Failure": "Overfitting", "Career-counsellor example": "Keeps pushing data science after you moved to product management", "Cause": "No freshness check"},
        ])
        st.markdown("**Fixes you built:** layered memory, a score balancing similarity, recency and importance (page 4), versioned updates, "
                    "and occasionally **asking the user**: *'Do you still prefer remote work?'*")

    with t2:
        st.markdown("If a made-up 'fact' is saved, the agent repeats it as truth and builds new memories on it. "
                    "A photocopier copying a blurry page: each copy is worse, and nobody notices the original was wrong.")
        st.graphviz_chart('digraph { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#ffe3e3", fontname="Helvetica"]; '
                          '"A guess" -> "Stored as fact" -> "Retrieved into every prompt" -> "New memories built on it" -> "Stored as fact"; }')
        plain("an **eager** extractor guesses ('no sugar' → 'diabetic'). A **careful** one stores only what the user said, with the exact quote as proof.")
        message = st.text_area("Kavya's message", "Plan us a cosy weekend in Coorg. Oh, and no sugar in my chai at the homestay please, I've given it up.", height=80)
        if st.button("⚖ Run both extractors", type="primary"):
            st.session_state.halluc = {"eager": generate(b, "kavya", message, "eager"), "careful": generate(b, "kavya", message, "careful")}
            st.session_state.halluc_src = b.last_source
        if "halluc" in st.session_state:
            for mode, colour in (("eager", st.error), ("careful", st.success)):
                mems = st.session_state.halluc[mode]
                colour(f"**{mode.upper()} extractor**: stored {len(mems)} memories")
                rows = []
                for m in mems:
                    audit = me.audit_memory(m["text"], m["source_quote"], "Kavya")
                    rows.append({"memory": m["text"], "confidence": m["confidence"], "source quote": m["source_quote"],
                                 "audit: quote supports it?": "✅ yes" if audit["supported"] else "❌ NO", "word overlap": round(audit["overlap"], 2)})
                df_show(rows)
            badge(b)
            st.markdown("**Memory audit (Homework 7).** Look at the quotes: *'no sugar in my chai'* does not support *'may be diabetic'*. "
                        "That mismatch is exactly what an automated audit checks. (This app uses a crude word-overlap check; the homework asks you to use Gemini.)")
            guess = any("diabetic" in m["text"].lower() for m in st.session_state.halluc["eager"])
            if guess:
                st.code("System prompt next time:\n- Kavya may be diabetic. (unconfirmed)\n→ Every suggestion becomes 'diabetic-friendly'.  = Ticket 3", language="text")
        st.markdown("**Fixes:** store **provenance** (`source_quote`), flag **low-trust** info (`confidence`, 'ask before relying'), run **audits**, use **version control** to roll back.")

    with t3:
        st.markdown("Stored chats, embeddings and metadata can leak (text can sometimes be partly reconstructed from an embedding). "
                    "Fixes: **encrypt**, **control access**, honour the **right to be forgotten** (hard delete), **log access**. "
                    "Two we can demonstrate: never *extract* sensitive data, and never let users' memories mix.")
        st.subheader("Check 1: sensitive data")
        risky = st.text_area("A message full of personal data", "Book the Coorg homestay on my card 4111 1111 1111 1111, OTP is 482913. "
                             "My phone is 98200 12345. We prefer a room with a balcony.", height=80)
        found = me.find_sensitive(risky)
        if found:
            st.error("Sensitive items found: " + ", ".join(f"{label} ({value})" for label, value in found))
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Eager extractor stores:**")
            df_show([{"memory": m["text"]} for m in me.demo_generate_memories("Kavya", risky, "eager")] or [{"memory": "(nothing)"}])
        with c2:
            st.markdown("**Careful extractor stores:**")
            df_show([{"memory": m["text"]} for m in me.demo_generate_memories("Kavya", risky, "careful")] or [{"memory": "(nothing)"}])
        st.markdown("**Redact in code before it reaches any model or store:**")
        st.code(me.redact(risky), language="text")
        st.caption("The extractor's rule is a *defence*, not a guarantee. Sensitive data should also be redacted by ordinary code.")

        st.subheader("Check 2: cross-user isolation")
        if not any(it["meta"]["user_id"] == "kavya" for it in store.all()):
            store.add("kavya", "Kavya has given up sugar in her chai.", "semantic", 2, 1.0, "no sugar in my chai")
        q = "food and drink preferences"
        with_filter = store.search(q, user_id="prasad", k=10)
        without_filter = store.search(q, user_id=None, k=10)
        leaked = sum(1 for it, _ in without_filter if it["meta"]["user_id"] != "prasad")
        c1, c2 = st.columns(2)
        c1.success(f"**With the user filter:** other users' memories returned to Prasad = {sum(1 for it, _ in with_filter if it['meta']['user_id'] != 'prasad')}")
        c2.error(f"**Forgot the filter:** other users' memories returned to Prasad = {leaked}")
        df_show([{"user": it["meta"]["user_id"], "memory": it["text"], "similarity": round(s, 2)} for it, s in without_filter[:6]])
        st.caption("Here the filter is one line of Python. In production use a **namespace per user/tenant** (e.g. Pinecone) so isolation does not depend on every query remembering to filter.")

    with t4:
        st.markdown("Fixes that work for one agent strain at **10,000 agents and billions of memories**: a bookshelf vs a city library. "
                    "Think of a hotel: private keycards (private memory), a shared cafeteria (shared knowledge), and chaos if keys get mixed up.")
        df_show([
            {"Challenge": "Forgetting vs overfitting", "What happens": "Old facts fade, or one fact dominates", "Fixes": "Layered memory; balanced scoring; versioning; confirm with user"},
            {"Challenge": "Hallucination amplification", "What happens": "Guesses stored as facts, reinforced", "Fixes": "Provenance; confidence flags; audits; rollback"},
            {"Challenge": "Privacy & security", "What happens": "Stored chats/embeddings/metadata can leak", "Fixes": "Don't extract sensitive data; encrypt; access control; hard delete; audit logs"},
            {"Challenge": "Scaling", "What happens": "Billions of memories across thousands of agents", "Fixes": "Hot/cold storage; caching; namespaces; monitoring"},
        ])


# ----------------------------------------------------------------------------------------
# Page 8: Review
# ----------------------------------------------------------------------------------------
QUIZ = [
    ("What does an LLM remember between two separate API calls?",
     ["Everything from the previous call", "Only the last 4 messages", "Nothing: memory is whatever your code puts back in the prompt", "Whatever is stored in its weights about you"], 2,
     "The model is stateless. Every memory is a decision in your code about what to write down and what to put back."),
    ("Ticket 1 (vegetarian forgotten after 20 messages) is caused by...",
     ["Overflow forgetting in a small context window", "A bad embedding model", "A missing hard delete", "An episodic memory bug"], 0,
     "Old messages fell out of the window, so the fact was simply not in the prompt."),
    ("Which short-term strategy keeps the gist of old turns AND exact recent turns?",
     ["Buffer", "Window", "Summary + buffer", "Entity"], 2, "Summary + buffer is the usual default for chat agents."),
    ("In the retrieval score, why add recency and importance to similarity?",
     ["Similarity alone may return old or trivial memories that merely sound close",
      "Similarity cannot be computed on metadata", "It makes retrieval cheaper", "It removes the need for a user filter"], 0,
     "0.6 × similarity + 0.3 × recency + 0.1 × importance balances meaning, freshness and criticality."),
    ("Prasad moves from Pune to Bengaluru. What is the best update strategy for an audit-friendly store?",
     ["Overwrite the old text", "Delete all his memories", "Mark the old memory superseded and add a new version linked to it", "Do nothing, retrieval will sort it out"], 2,
     "Append-and-version keeps history; only the active version is retrievable."),
    ("A user says 'delete everything about me'. Which action honours the request?",
     ["Soft delete (status = deleted)", "Hard delete of text and embeddings (and regenerate dependent summaries)", "Lower retrieval priority", "Encrypt the memories"], 1,
     "Soft delete leaves text and embedding behind, so it is not real erasure."),
    ("Kavya said 'no sugar in my chai' and the agent stored 'may be diabetic'. What is this?",
     ["Forgetting", "Hallucination amplification", "Cross-user leakage", "Procedural memory"], 1,
     "A guess saved as a fact is repeated, reinforced and built upon. Fix with explicit-only extraction, source quotes and confidence."),
    ("Which kind of memory is 'about how the agent works' and therefore risky to auto-update?",
     ["Semantic", "Episodic", "Procedural", "Short-term"], 2,
     "A bad rule in procedural memory spreads to every user, so changes usually need human review."),
]


def page_review():
    get_backend()
    st.header("8 · Review, quiz and cheat sheet")
    st.subheader("The five tickets, closed")
    df_show([
        {"Ticket": "1 Vegetarian forgotten", "Cause": "Overflow forgetting in a small window", "Fix": "Summary + buffer, or entity memory keeping critical facts", "Page": "3"},
        {"Ticket": "2 Asks where I live", "Cause": "No long-term memory across sessions", "Fix": "Generate → store → retrieve → integrate", "Page": "4"},
        {"Ticket": "3 'Diabetic' from 'no sugar'", "Cause": "Eager extractor stored a guess", "Fix": "Explicit-only extraction + source quotes + confidence", "Page": "7"},
        {"Ticket": "4 Suggests Manali again", "Cause": "Facts but no record of the event", "Fix": "Episodic memory: what worked, what did not", "Page": "6"},
        {"Ticket": "5 Delete everything", "Cause": "No deletion path", "Fix": "Hard delete of text and embeddings (+ cascade)", "Page": "4"},
    ])
    st.success("Underneath all of them: **the model remembered nothing at any point.** Every fix was a decision in your code about what to write down, "
               "how to label it and what to put back into the prompt. No framework makes those decisions for you, and none checks whether a stored memory is true.")

    st.subheader("🧠 Quiz")
    with st.form("quiz"):
        answers = []
        for i, (q, opts, correct, why) in enumerate(QUIZ):
            answers.append(st.radio(f"**{i + 1}. {q}**", opts, index=None, key=f"quiz_{i}"))
        submitted = st.form_submit_button("Check my answers", type="primary")
    if submitted:
        score = 0
        for i, (q, opts, correct, why) in enumerate(QUIZ):
            ok = answers[i] == opts[correct]
            score += ok
            (st.success if ok else st.error)(f"**Q{i + 1}:** {'Correct' if ok else 'Not quite'}. {why}")
        st.metric("Score", f"{score} / {len(QUIZ)}")

    st.subheader("📝 Homework")
    df_show([
        {"#": "HW1", "Level": "Easy", "Task": "Tune the retrieval score weights", "Try it": "Page 4, Retrieve tab sliders"},
        {"#": "HW2", "Level": "Medium", "Task": "Protect critical facts from summaries", "Try it": "Page 3, Five strategies (allergy + verbatim toggles)"},
        {"#": "HW3", "Level": "Medium", "Task": "Move memory to Pinecone with namespaces", "Try it": "Code change in memory_engine.MemoryStore"},
        {"#": "HW4", "Level": "Medium", "Task": "Handle bad replacement decisions", "Try it": "Page 4, Update tab (you confirm the replacement)"},
        {"#": "HW5", "Level": "Hard", "Task": "Cascading deletes for summaries", "Try it": "Design exercise"},
        {"#": "HW6", "Level": "Easy", "Task": "Tune the forgetting thresholds", "Try it": "Page 5 sliders"},
        {"#": "HW7", "Level": "Medium", "Task": "Automated memory audit", "Try it": "Page 7, Hallucination lab (replace overlap check with Gemini)"},
        {"#": "HW8", "Level": "Hard", "Task": "Stop old episodes dominating", "Try it": "Page 6, Episodic lab (add decay + 'we'd go back')"},
    ])

    st.subheader("⚡ Quick reference")
    df_show([
        {"Idea": "The rule", "One line": "The model remembers nothing; your code writes and reads memory"},
        {"Idea": "Short-term memory", "One line": "The context window; forgets by overflow"},
        {"Idea": "Five strategies", "One line": "Buffer, window, summary, summary + buffer, entity"},
        {"Idea": "Long-term memory", "One line": "Persists across sessions: store with metadata"},
        {"Idea": "Generate", "One line": "Extract explicit facts with source and confidence"},
        {"Idea": "Retrieve", "One line": "Filter → similarity → re-rank by recency and importance"},
        {"Idea": "Integrate", "One line": "Put memories in the prompt with provenance"},
        {"Idea": "Update", "One line": "Version; mark the old one superseded"},
        {"Idea": "Delete", "One line": "Soft (flag) vs hard (text + embedding)"},
        {"Idea": "Forgetting strategies", "One line": "Periodical, history-based, combined"},
        {"Idea": "Semantic / Episodic / Procedural", "One line": "Facts / experiences / how the agent works"},
        {"Idea": "Knowledge graph", "One line": "Entities as nodes, relations as edges"},
        {"Idea": "Hallucination amplification", "One line": "A stored guess becomes a 'fact'"},
    ])
    with st.expander("📖 Glossary"):
        glossary_table()


# ----------------------------------------------------------------------------------------
# Sidebar + router
# ----------------------------------------------------------------------------------------
def sidebar():
    st.sidebar.title("🧠 Memory Lab")
    st.session_state.setdefault("nav", PAGES[0])
    st.sidebar.radio("Go to", PAGES, key="nav")
    st.sidebar.divider()
    st.sidebar.markdown("**Mode**")
    st.sidebar.radio("Mode", ["Demo (no API key)", "Live (Gemini)"], key="mode", label_visibility="collapsed")
    if st.session_state.get("mode", "").startswith("Live"):
        st.sidebar.text_input("Gemini API key", type="password", key="api_key", value=_default_key(),
                              help="Kept in this browser session only. Never commit keys to git: use .streamlit/secrets.toml or an env var.")
        with st.sidebar.expander("Model settings"):
            st.text_input("Chat model", "gemini-3.5-flash-lite", key="chat_model")
            st.text_input("Embedding model", "models/gemini-embedding-001", key="embed_model")
            st.number_input("Max Gemini calls per minute (0 = no limit)", min_value=0, max_value=60, value=10, key="rpm",
                            help="Spaces the chat calls out to respect the free tier. Lower = safer but slower: 4 means one call every 15 seconds.")
    st.session_state._ph = st.sidebar.empty()        # live status line, filled while Gemini is working
    b = get_backend()
    if st.session_state.get("mode", "").startswith("Live"):
        if b.live:
            st.sidebar.success("Live: connected to Gemini")
            st.sidebar.caption("Gemini calls this session: %d  |  last call: %.1f s" % (b.calls, b.last_seconds))
            if st.sidebar.button("🩺 Test Gemini connection", key="ping"):
                reply = b.ask([{"role": "user", "content": "Reply with the single word OK."}], "OK")
                st.sidebar.write("Reply: " + reply.strip()[:40] + "  (%.1f s)" % b.last_seconds)
        else:
            st.sidebar.warning(b.error or "Paste a key to go live. Using demo mode meanwhile.")
    else:
        st.sidebar.info("Demo mode: simulated answers, toy embeddings.")
    if b.error and st.session_state.get("mode", "").startswith("Live") and "Could not start" not in b.error:
        st.sidebar.warning(b.error)
    store = st.session_state.get("store")
    if store is not None and store.remote is not None:
        if store.remote_ready():
            st.sidebar.success("☁ Pinecone: index '%s' connected, mirroring is ON" % store.remote.index_name)
        else:
            st.sidebar.warning("☁ Pinecone connected but paused: it needs Live (Gemini) embeddings.")


def main():
    sidebar()
    page = st.session_state.get("nav", PAGES[0])
    {PAGES[0]: page_home, PAGES[1]: page_why, PAGES[2]: page_storage, PAGES[3]: page_short, PAGES[4]: page_long,
     PAGES[5]: page_forget, PAGES[6]: page_kinds, PAGES[7]: page_risks, PAGES[8]: page_review}[page]()


main()
