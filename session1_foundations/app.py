"""
Foundations Lab: an interactive companion to "Agents, Session 1: Foundations of Agents".

Run:   streamlit run app.py
Works with NO API key (Demo mode: scripted "models" and pretend data). Add a Gemini key in the sidebar for Live mode.
"""
import os

import pandas as pd
import streamlit as st

import foundations_engine as fe

import inspect as _inspect

# Newer Streamlit wants width="stretch"; older versions want use_container_width=True.
try:
    STRETCH = {"width": "stretch"} if _inspect.signature(st.dataframe).parameters["width"].default == "stretch" else {"use_container_width": True}
except Exception:
    STRETCH = {"use_container_width": True}

st.set_page_config(page_title="Foundations Lab: from LLMs to agents", page_icon="🤖", layout="wide")

PAGES = [
    "🏠 Start here",
    "1 · The limit: text in, text out",
    "2 · What an agent is",
    "3 · How agents evolved",
    "4 · Giving the concierge hands",
    "5 · Why agents, and why not always",
    "6 · Four workflow patterns",
    "7 · Workflow or agent? Frameworks",
    "8 · Capstone: the Manali planner",
    "9 · Review, quiz & cheat sheet",
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
    ph = st.session_state.get("_ph")
    if ph is not None:
        try:
            ph.info(msg)
        except Exception:
            pass


def get_backend() -> fe.Backend:
    mode = st.session_state.get("mode", "Demo (no API key)")
    key = st.session_state.get("api_key", "") if mode.startswith("Live") else ""
    cfg = (key, st.session_state.get("chat_model", fe.MODEL_DEFAULT), st.session_state.get("rate_limit", True))
    if st.session_state.get("_cfg") != cfg or "backend" not in st.session_state:
        st.session_state.backend = fe.Backend(*cfg)
        st.session_state._cfg = cfg
    st.session_state.backend.notify = _notify
    return st.session_state.backend


def goto(page: str):
    st.session_state.nav = page


def badge(b: fe.Backend):
    if b.last_source == "gemini":
        st.caption("🟢 Written by Gemini")
    elif "fallback" in b.last_source:
        st.caption("🟠 " + (b.error or "Live call failed; showing the demo simulation."))
    else:
        st.caption("🔵 Demo simulation (scripted model and pretend data, no API key needed)")


def plain(text: str):
    st.info("**In plain words:** " + text)


def story(text: str):
    st.warning("**📖 " + text + "**")


def predict(text: str):
    st.markdown("> 🤔 **Before you run it:** " + text)


def df_show(rows, **kw):
    st.dataframe(pd.DataFrame(rows), **STRETCH, hide_index=True, **kw)


def show_code(title: str, code: str):
    with st.expander("🔍 " + title):
        st.code(code.strip("\n"), language="python")


def dot(src: str):
    st.graphviz_chart(src)


NODE = 'node [shape=box, style="rounded,filled", fillcolor="#eef2ff", fontname="Helvetica"];'


def quiz(key: str, items: list):
    """items: (question, options, index_of_correct, explanation)."""
    picks = []
    for i, (q, opts, _, _) in enumerate(items):
        picks.append(st.radio("**%d. %s**" % (i + 1, q), opts, index=None, key="%s_%d" % (key, i)))
    if st.button("Check my answers", key=key + "_check", type="primary"):
        score = 0
        for i, (q, opts, ans, why) in enumerate(items):
            if picks[i] is None:
                st.warning("%d. Not answered." % (i + 1))
            elif picks[i] == opts[ans]:
                score += 1
                st.success("%d. Correct. %s" % (i + 1, why))
            else:
                st.error("%d. Not quite. Correct answer: **%s**. %s" % (i + 1, opts[ans], why))
        st.metric("Score", "%d / %d" % (score, len(items)))


def render_events(events: list[dict]):
    """Show an agent run as a step-by-step timeline."""
    by_step = {}
    for e in events:
        by_step.setdefault(e["step"], []).append(e)
    for step, group in by_step.items():
        with st.container(border=True):
            st.markdown("**Step %d**" % step)
            for e in group:
                k = e["kind"]
                if k == "thought":
                    st.markdown("💭 *Model's reasoning:* " + e["text"])
                elif k == "request":
                    st.markdown("🛠️ **Model requests** `%s(%s)`  — *nothing has run yet*" % (e["name"], ", ".join("%s=%r" % kv for kv in e["args"].items())))
                elif k == "result":
                    st.markdown("⚙️ **Your code ran it** → `%s`" % e["text"])
                elif k == "error":
                    st.markdown("⚠️ **Error sent back to the model** → `%s`" % e["text"])
                elif k == "final":
                    st.success("🏁 **No tool requested, so the agent is done.** Final answer:\n\n" + e["text"])
                elif k == "crash":
                    st.error("💥 " + e["text"])
                elif k == "limit":
                    st.error("⛔ " + e["text"])


def calendar_table(calendar):
    df = pd.DataFrame(calendar) if calendar else pd.DataFrame(columns=["date", "time", "title"])
    st.dataframe(df, **STRETCH, hide_index=True)


# ----------------------------------------------------------------------------------------
# Page 0: Start
# ----------------------------------------------------------------------------------------
def page_home():
    st.title("🤖 Foundations Lab: from an LLM that talks to an agent that acts")
    st.caption("An interactive companion to *Agents, Session 1: Foundations of Agents* (the DilSeVector concierge notebook).")
    story("Monday morning. Prasad types: \"Book a table for two at the pink café in Dadar this Saturday at 7 pm and add it to my calendar.\" "
          "The concierge replies politely. Nothing is booked. Nothing is in his calendar.")
    st.success("**The question to hold on to:** *who decides the next step?* In a plain LLM, nobody acts. In a workflow, **your code** fixes the order. "
               "In an agent, **the model** picks the next action and your code carries it out. Every idea in this session is a variation on that question.")

    st.subheader("Three things the concierge was missing")
    c1, c2, c3 = st.columns(3)
    c1.markdown("### 🖐️ Hands\nChanging a calendar needs code to run. We give the model **tools**: Python functions it can ask us to run.")
    c2.markdown("### 👀 Eyes on the present\nThe model does not know today's date, so \"this Saturday\" means nothing. A **tool** can tell it.")
    c3.markdown("### 🔁 A second look\nBook *only if Prasad is free* needs: check, decide on the result, then act. A single call cannot react to a result it has not seen. We need a **loop**.")

    dot('digraph { rankdir=LR; ' + NODE + ' P [label="Perceive", fillcolor="#d0ebff"]; R [label="Reason & plan", fillcolor="#fff3bf"]; '
        'A [label="Act", fillcolor="#d3f9d8"]; F [label="Feedback", fillcolor="#ffe3e3"]; P -> R -> A -> F -> P [label=" repeat until the goal is met"]; }')

    st.subheader("Your path")
    path = [
        ("1 · The limit", PAGES[1], "Send Prasad's requests to a bare LLM. Count the calendar events."),
        ("2 · What an agent is", PAGES[2], "Run the perceive, reason, act, feedback loop; passive vs proactive."),
        ("3 · How agents evolved", PAGES[3], "Six stages, each rebuilt with an LLM; search, learning and exploration labs."),
        ("4 · Giving the concierge hands", PAGES[4], "Tools, tool calls, the agent loop, step limits, FlyBot 2.0."),
        ("5 · Why agents, and why not always", PAGES[5], "Reliability compounds. Workflow or agent?"),
        ("6 · Four workflow patterns", PAGES[6], "Evaluator-optimizer, parallelization, chaining, routing."),
        ("7 · Workflow or agent? Frameworks", PAGES[7], "A decision guide and the framework landscape."),
        ("8 · Capstone: the Manali planner", PAGES[8], "GenAI vs agent vs four-role planner, grounding check, calendar export."),
        ("9 · Review", PAGES[9], "Quiz, homework map, cheat sheet."),
    ]
    for title, page, desc in path:
        a, b, c = st.columns([3, 6, 1.4])
        a.markdown("**" + title + "**")
        b.write(desc)
        c.button("Open", key="home_" + page, on_click=goto, args=(page,))

    st.markdown("**Modes.** *Demo* (default) needs no key: a scripted stand-in plays the model and all data is pretend. *Live* (sidebar) sends the "
                "labs to Gemini through LangChain, as in the notebook. A failed live call falls back to the demo answer and says why.")
    with st.expander("📖 Glossary"):
        df_show([
            {"Term": "LLM", "Meaning": "A model that turns a prompt into text. Reactive: it acts only when prompted."},
            {"Term": "Agent", "Meaning": "Observes, decides and acts toward a goal, in a loop, using feedback."},
            {"Term": "AI agent", "Meaning": "An agent whose perceive and decide steps are powered by an AI model."},
            {"Term": "Tool", "Meaning": "A normal function with a clear name, type hints and docstring that the model may request."},
            {"Term": "Tool call", "Meaning": "The model's *request* to run a tool. Nothing runs until your code runs it."},
            {"Term": "Workflow", "Meaning": "Predefined steps in a fixed order. Your code decides the path."},
            {"Term": "Structured output", "Meaning": "A reply forced into a schema, so code can read it as a dictionary."},
        ])


# ----------------------------------------------------------------------------------------
# Page 1: The limit
# ----------------------------------------------------------------------------------------
def page_limit():
    b = get_backend()
    st.title("1 · The limit: text in, text out")
    story("Both of the course's earlier pictures (prompting and RAG) end in the same place: text. Send Prasad's two requests to the model "
          "exactly as the concierge would, and check his calendar afterwards.")
    dot('digraph { rankdir=LR; ' + NODE + ' Prompt -> LLM -> "Text" -> "Calendar?" [style=dashed, color=red, label=" nothing runs"]; "Calendar?" [fillcolor="#ffe3e3"]; }')

    presets = {
        "Book a table + add to calendar": "Book a table for two at the pink café in Dadar this Saturday at 7 pm and add it to my calendar.",
        "Cheapest flight to Kullu": "Find the cheapest IndiGo flight from Mumbai to Kullu next Friday and book it for me.",
    }
    choice = st.radio("Prasad's request", list(presets) + ["Write my own"], horizontal=True, key="s1_limit_choice")
    request = presets[choice] if choice in presets else st.text_input("Your request", "Order me a coffee and put it on my card.", key="s1_limit_custom")
    over = st.checkbox("Simulate the *worse* failure: a model that sounds like it confirmed", value=False, key="s1_limit_over",
                       help="Sometimes the model says honestly that it cannot book; sometimes it writes something that sounds like a confirmation.")
    predict("after the reply, how many events will be in Prasad's calendar?")
    if st.button("Send to the bare LLM", type="primary", key="s1_limit_send"):
        reply = b.ask(request, demo=lambda: fe.bare_llm_reply(request, over))
        st.chat_message("user").write(request)
        st.chat_message("assistant").write(reply)
        badge(b)
        st.metric("Events in Prasad's calendar", 0)
        if over:
            st.error("This is the dangerous case: Prasad would turn up at the café expecting a table.")
        else:
            st.warning("Honest, but still nothing happened. The only thing an LLM produces is text.")

    st.subheader("Why: LLMs are reactive")
    st.markdown("- They **do not start anything** on their own.\n- They have **no goals** of their own.\n- They act **only when a prompt arrives**, and even then the output is text.\n\n"
                "A calculator is reactive in the same way: it never wakes up and decides to add 2 + 2. An LLM will never decide to book your flight; it predicts the next words of a reply.")
    plain("Think of Siri: you say \"set an alarm for 7 am\" and it is set. It understood, chose an action and carried it out. A bare LLM only does the first part.")
    st.subheader("The three missing pieces")
    df_show([
        {"Missing": "Hands", "Why": "Changing a calendar needs code to run", "Fix": "Tools (Python functions)", "Where": "Page 4"},
        {"Missing": "Eyes on the present", "Why": "The model does not know today's date", "Fix": "A get_today tool", "Where": "Page 4"},
        {"Missing": "A second look", "Why": "Book only if free: act depends on a result", "Fix": "A loop", "Where": "Page 4"},
    ])


# ----------------------------------------------------------------------------------------
# Page 2: What an agent is
# ----------------------------------------------------------------------------------------
def page_agent():
    b = get_backend()
    st.title("2 · What an agent is")
    st.write("An **agent** is anything that can **observe** what is happening around it, **make decisions**, and **take actions** to achieve a goal. "
             "What makes agents special is the combination of four abilities, arranged as a loop.")
    t1, t2, t3 = st.tabs(["The loop (thermostat lab)", "Goal-driven & proactive", "From agents to AI agents"])

    with t1:
        st.markdown("**Perceive** → **Reason & plan** → **Act** → **Feedback** → repeat. An agent is *not* one-and-done: it keeps going until the goal is met.")
        c1, c2, c3, c4 = st.columns(4)
        start = c1.slider("Starting temperature (°C)", 10, 35, 15, key="s1_th_start")
        low = c2.slider("Heater ON below (°C)", 15, 24, 20, key="s1_th_low")
        high = c3.slider("Heater OFF above (°C)", 22, 32, 25, key="s1_th_high")
        cycles = c4.slider("Cycles", 5, 30, 14, key="s1_th_cycles")
        temp, heater, rows = float(start), False, []
        for n in range(1, cycles + 1):
            perceived = temp
            if perceived < low:
                decision, heater = "too cold → heater ON", True
            elif perceived > high:
                decision, heater = "too warm → heater OFF", False
            else:
                decision = "comfortable → keep state (%s)" % ("ON" if heater else "OFF")
            temp = temp + (1.5 if heater else -0.7)
            rows.append({"cycle": n, "perceive (°C)": round(perceived, 1), "reason": decision, "act": "heater " + ("ON" if heater else "OFF"),
                         "feedback (new °C)": round(temp, 1)})
        df_show(rows)
        st.line_chart(pd.DataFrame({"temperature": [r["feedback (new °C)"] for r in rows]}))
        plain("Without the feedback column the thermostat would never learn that it overshot. Remove the OFF threshold (set it to 32) and watch the room overheat: "
              "an agent with a bad rule still loops, it just loops badly.")
        st.markdown("The same loop for a robot vacuum: **perceive** dirt and walls → **decide** left or right → **act** by moving and cleaning → **feedback** (it bumped a wall, so change direction).")

    with t2:
        st.markdown("**Goal-driven** means you state the end result, not the steps (like a GPS rerouting toward your destination). "
                    "**Proactive** means it takes initiative: it raises problems you have not asked about yet. With an LLM, much of this starts in the **system prompt**.")
        request = st.text_input("Trip request", "I want to plan a trip to Manali with my partner in the last week of December.", key="s1_pro_req")
        styles = {
            "Passive": "You are a travel assistant. Answer exactly what the user asks and nothing more. Keep it under 80 words.",
            "Proactive": ("You are a proactive travel assistant. Keep the user's end goal in mind. Before answering, think about what could go wrong or what they have "
                          "not asked about yet (season, weather, road conditions, bookings that sell out, documents, budget). Raise the two most important points as questions. "
                          "Keep it under 80 words."),
        }
        with st.expander("The only difference: the system prompt"):
            st.code(styles["Passive"], language="text")
            st.code(styles["Proactive"], language="text")
        predict("what will the proactive assistant raise that the passive one will not?")
        if st.button("Ask both assistants", type="primary", key="s1_pro_go"):
            cols = st.columns(2)
            for col, style in zip(cols, styles):
                reply = b.ask([{"role": "system", "content": styles[style]}, {"role": "user", "content": request}],
                              demo=lambda s=style: fe.assistant_reply(s, request))
                col.markdown("#### %s assistant" % style)
                col.write(reply)
            badge(b)
            st.caption("Notice that neither *did* anything. Proactive instructions shape what the model **says**. Tools and a loop (page 4) let it **act**.")

    with t3:
        st.markdown("An **AI agent** is an agent whose perceive and decide steps are powered by AI: it understands natural language, plans several steps and uses digital tools on its own.")
        df_show([
            {"": "Simple agent (student example)", "What it does": "Reminds the student of deadlines"},
            {"": "AI agent", "What it does": "Reads the chapters, summarises key points, builds a study schedule and sets a reminder for each task"},
            {"": "Fictional ideal", "What it does": "J.A.R.V.I.S.: told \"get my coffee ready\", it plans and delivers without step-by-step instructions"},
        ])
        st.caption("Dates: deep-RL agents like AlphaGo date from about 2015; agents that use a language model to choose tools became practical around 2022 (ReAct, Oct 2022; AutoGPT, Mar 2023).")


# ----------------------------------------------------------------------------------------
# Page 3: Evolution
# ----------------------------------------------------------------------------------------
def page_evolution():
    b = get_backend()
    st.title("3 · How agents evolved, rebuilt with an LLM")
    st.write("Each stage fixed a limit of the one before. The loop stayed the same; what changed was **who decides the next step, and how**.")
    dot('digraph { rankdir=LR; ' + NODE + ' "1 Rule-based\\n1950s-70s" -> "2 Reactive\\n80s-90s" -> "3 Deliberative\\n90s-2000s" -> "4 Learning" -> "5 Domain-specific" -> "6 Modern AI agents\\n(LLM + tools + memory)"; }')
    tabs = st.tabs(["1 Rule-based", "2 Reactive", "3 Deliberative", "4 Learning", "5 Domain-specific", "6 Modern"])

    # ---- 3.1 rule-based
    with tabs[0]:
        story("Before DilSeVector, the pink café ran a WhatsApp booking bot built on if-then rules. Edit the rules, then test three real customer messages.")
        rules_df = st.data_editor(pd.DataFrame([{"keyword": k, "reply": v} for k, v in fe.CAFE_RULES.items()]), num_rows="dynamic",
                                  **STRETCH, key="s1_rules_editor")
        rules = {str(r["keyword"]): str(r["reply"]) for _, r in rules_df.iterrows() if str(r["keyword"]).strip() not in ("", "nan", "None")}
        msgs = list(fe.CAFE_MESSAGES) + [st.text_input("Or type your own message", "Do you have oat milk?", key="s1_rule_custom")]
        predict("what intent will an LLM give the caffeine question? Which messages will the rule bot handle?")
        if st.button("Run both", type="primary", key="s1_rules_go"):
            text = "\n".join("- " + m for m in msgs)
            result = b.structured(fe.UNDERSTANDING_SCHEMA, "Understand each of these café customer messages:\n" + text,
                                  demo=lambda: {"messages": [fe.demo_understand(m) for m in msgs]})
            rows = []
            for m, item in zip(msgs, result["messages"]):
                reply, hit = fe.rule_bot_reply(rules, m)
                rows.append({"Customer message": m, "Rule bot says": reply, "Matched keyword": hit or "(none)", "LLM intent": item["intent"], "LLM details": item["details"]})
            df_show(rows)
            badge(b)
            plain("The rule bot only handled the message containing a keyword. The LLM turned **all** of them into structured data your code can route on "
                  "(booking → booking flow, menu_question → menu...). That is the modern *perceive* step in one cell.")
        show_code("The structured-output call", """
understanding_llm = llm.with_structured_output(understanding_schema)   # a JSON schema with an enum of intents
result = understanding_llm.invoke("Understand each of these café customer messages:\\n" + all_messages_text)
for item in result["messages"]:
    print(item["intent"], item["details"])
""")

    # ---- 3.2 reactive
    with tabs[1]:
        story("A bare LLM call is a reactive agent: it responds to the message in front of it and remembers nothing.")
        first = st.text_input("Call 1", "Hi, I'm Prasad. I run a dating app called DilSeVector and my favourite drink is kesar cold coffee.", key="s1_re_first")
        second = st.text_input("Call 2", "What's my name, and what's my favourite drink?", key="s1_re_second")
        send_history = st.toggle("Send the whole conversation along with call 2", value=False, key="s1_re_hist")
        predict("will the second call know his favourite drink?")
        if st.button("Run call 1, then call 2", type="primary", key="s1_re_go"):
            r1 = b.ask(first, demo="Nice to meet you, Prasad! Kesar cold coffee sounds delicious.")
            st.chat_message("user").write(first)
            st.chat_message("assistant").write(r1)
            sent = [{"role": "user", "content": first}, {"role": "assistant", "content": r1}, {"role": "user", "content": second}] if send_history \
                else [{"role": "user", "content": second}]
            r2 = b.ask(sent, demo=lambda: fe.memoryless_reply(sent[:-1], second))
            st.chat_message("user").write(second)
            st.chat_message("assistant").write(r2)
            st.metric("Messages sent in call 2", len(sent))
            badge(b)
            if send_history:
                st.success("Now it knows. The model did not change; the **prompt** did. Every chat app works this way.")
            else:
                st.error("Each call is independent: the model reacts to its input and forgets everything afterwards.")
        plain("\"Memory\" in every agent you build means **what you choose to send back to the model**. The fix is not a smarter model; it is a list you manage yourself.")

    # ---- 3.3 deliberative
    with tabs[2]:
        story("Prasad's goal: a dinner date at the pink café this weekend. A deliberative agent plans first, executes with monitoring, and **replans** when reality pushes back.")
        st.markdown("**World model** the agent is given:")
        st.code("Tools: get_today, check_calendar, call_cafe, add_calendar_event, send_whatsapp.\n"
                "Prasad is busy this Saturday 18:00-21:00 (office cricket match).\n"
                "The pink café in Dadar is open 8:00-23:00 and takes bookings by phone only.", language="text")
        if st.button("1) Make the plan (do not execute)", type="primary", key="s1_plan_go"):
            res = b.structured(fe.PLAN_SCHEMA, "Tools: " + ", ".join(fe.PLAN_TOOLS) + ". Prasad is busy Saturday 18:00-21:00. Goal: arrange a dinner date at the pink café this weekend. "
                               "Make a step-by-step plan. Do not carry anything out.", demo=lambda: {"steps": fe.demo_plan()})
            st.session_state.s1_plan = res["steps"]
            st.session_state.pop("s1_replan", None)
            st.session_state.s1_plan_src = b.last_source
        if "s1_plan" in st.session_state:
            df_show([{"#": i + 1, "action": s["action"], "tool": s["tool"]} for i, s in enumerate(st.session_state.s1_plan)])
            surprise = st.selectbox("2) Now the world changes. What happens during execution?", list(fe.SURPRISES), key="s1_surprise")
            st.info("Observation: " + fe.SURPRISES[surprise])
            predict("what will the new plan do about Saturday's cricket match?")
            if st.button("Replan from here", key="s1_replan_go"):
                old = "\n".join("- " + s["action"] for s in st.session_state.s1_plan)
                res = b.structured(fe.PLAN_SCHEMA, "Earlier plan:\n" + old + "\nWhile executing it, this happened: " + fe.SURPRISES[surprise]
                                   + "\nPrasad is busy Saturday 18:00-21:00. Café open 8:00-23:00. Make a revised plan from this point.",
                                   demo=lambda: {"steps": fe.demo_replan(surprise)})
                st.session_state.s1_replan = res["steps"]
            if "s1_replan" in st.session_state:
                st.markdown("**Revised plan**")
                df_show([{"#": i + 1, "action": s["action"], "tool": s["tool"]} for i, s in enumerate(st.session_state.s1_replan)])
                badge(b)
                plain("Plan → execute → monitor → replan. Its weakness is the robot's weakness: the plan is only as good as the **model of the world** it was given.")

        st.divider()
        st.subheader("Search-based planning: BFS vs DFS vs A*")
        st.write("A deliberative agent that must find a route explores possibilities. **BFS** explores every route of length 1, then 2... **DFS** follows one route as far as it goes before backing up. "
                 "**A\\*** explores the most promising routes first using an estimate of the distance left.")
        c1, c2 = st.columns([1, 2])
        gname = c1.selectbox("Office map", list(fe.GRIDS), key="s1_grid")
        algo = c1.radio("Algorithm", ["BFS", "DFS", "A*"], key="s1_algo")
        grid = fe.GRIDS[gname]
        res = fe.search_grid(grid, algo)
        upto = c1.slider("Animate: cells explored so far", 0, len(res["order"]), len(res["order"]), key="s1_upto")
        c2.markdown(fe.grid_html(grid, res, upto), unsafe_allow_html=True)
        c2.caption("🟨 start  🟥 goal  🟦 explored  🟩 final path  ⬛ wall")
        compare = []
        for a in ["BFS", "DFS", "A*"]:
            r = fe.search_grid(grid, a)
            compare.append({"algorithm": a, "cells explored": r["explored"], "path length": len(r["path"]) - 1 if r["path"] else "none"})
        df_show(compare)
        plain("BFS always finds a shortest path but explores widely. DFS may find a *long* path quickly. A* usually explores fewest cells while still finding a shortest one. "
              "(Other planning styles: rule-based planning and hierarchical planning, where a goal splits into sub-goals.)")
        st.markdown("**Reasoning:** *logic-based* (\"the lift is out of service, so any plan using it is infeasible\") and *utility-based* (score options by expected benefit, pick the best).")

    # ---- 3.4 learning
    with tabs[3]:
        story("Learning agents grow smarter with experience. But Gemini's weights never change while your code runs: an LLM agent \"learns\" by **storing lessons and putting them back into its prompt**.")
        df_show([
            {"Learning-agent part": "Learning element", "RL name": "Policy + learning algorithm", "Here": "A second LLM call that turns feedback into rules"},
            {"Learning-agent part": "Performance element", "RL name": "The agent's actions", "Here": "The message-drafting call"},
            {"Learning-agent part": "Critic", "RL name": "Reward signal", "Here": "Prasad's feedback"},
            {"Learning-agent part": "Problem generator", "RL name": "Exploration vs exploitation", "Here": "Trying something new (below)"},
        ])
        st.session_state.setdefault("s1_lessons", [])
        tasks = list(fe.LESSON_TASKS)
        sys_text = "You write short WhatsApp messages on behalf of Prasad, who runs DilSeVector."

        def draft(task_key):
            lessons = st.session_state.s1_lessons
            instr = sys_text + ("\nLessons from Prasad's earlier feedback. Always follow them:\n" + "\n".join("- " + x for x in lessons) if lessons else "")
            return b.ask([{"role": "system", "content": instr}, {"role": "user", "content": fe.LESSON_TASKS[task_key]}],
                         demo=lambda: fe.demo_draft(fe.LESSON_TASKS[task_key], lessons))
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Attempt 1**")
            if st.button("Draft: " + tasks[0], key="s1_l_a1"):
                st.session_state.s1_attempt1 = draft(tasks[0])
            if "s1_attempt1" in st.session_state:
                st.chat_message("assistant").write(st.session_state.s1_attempt1)
            fb = st.text_area("Critic: Prasad's feedback", fe.DEFAULT_FEEDBACK, key="s1_fb")
            if st.button("Learn from this feedback", key="s1_l_learn"):
                reply = b.ask("Turn this feedback into at most three short, general rules for writing future messages. Return one rule per line and nothing else.\nFeedback: " + fb,
                              demo=lambda: "\n".join(fe.demo_lessons(fb)))
                for line in reply.split("\n"):
                    cleaned = line.strip().lstrip("-*0123456789. ").strip()
                    if cleaned:
                        st.session_state.s1_lessons.append(cleaned)
            st.markdown("**Lessons stored in memory** (a plain Python list):")
            if st.session_state.s1_lessons:
                for x in st.session_state.s1_lessons:
                    st.markdown("- " + x)
            else:
                st.caption("(empty)")
            if st.button("Clear lessons (the agent forgets)", key="s1_l_clear"):
                st.session_state.s1_lessons = []
                st.rerun()
        with c2:
            st.markdown("**Attempt 2: a different task**")
            predict("the next task is a food walk, not a café. Will the lessons still shape the message?")
            if st.button("Draft: " + tasks[1], key="s1_l_a2"):
                st.session_state.s1_attempt2 = draft(tasks[1])
                st.session_state.s1_attempt2_n = len(st.session_state.s1_lessons)
            if "s1_attempt2" in st.session_state:
                st.chat_message("assistant").write(st.session_state.s1_attempt2)
                st.caption("Lessons used: %d. Clear them and re-run: the agent is back to the long, formal, chicken-suggesting draft." % st.session_state.s1_attempt2_n)
            badge(b)

        st.divider()
        st.subheader("Exploration vs exploitation (the problem generator)")
        st.write("Delivery robot, three routes, reward = minutes saved. It does not know which is best. **ε** is how often it tries a random route instead of the best-so-far.")
        c1, c2 = st.columns(2)
        eps = c1.slider("ε (exploration rate)", 0.0, 1.0, 0.1, 0.05, key="s1_eps")
        rounds = c2.slider("Deliveries", 20, 500, 200, 10, key="s1_rounds")
        main = fe.run_bandit(eps, rounds)
        cmp = pd.DataFrame({"ε = 0 (never explore)": fe.run_bandit(0.0, rounds)["cumulative"], "ε = %.2f (yours)" % eps: main["cumulative"],
                            "ε = 1 (always random)": fe.run_bandit(1.0, rounds)["cumulative"]})
        st.line_chart(cmp)
        df_show([{"route": n, "true average reward": main["true_means"][n], "times chosen": main["counts"][n], "agent's estimate": round(main["estimates"][n], 2)} for n in main["counts"]])
        plain("With ε = 0 the agent locks onto the first route that paid anything and never discovers the better risky shortcut. With ε = 1 it never uses what it learned. "
              "A little curiosity, then mostly using what works, is how Amazon- and Swiggy-style route optimisation learns.")

    # ---- 3.5 domain-specific
    with tabs[4]:
        story("Prasad's December trip starts with a flight Mumbai → Kullu via Delhi. FlyBot was built the classic way: a developer coded every rule by hand.")
        with st.expander("FlyBot's rules"):
            st.json(fe.FLYBOT_RULES)
        rows = []
        for limit, msg in fe.FLYBOT_MESSAGES.items():
            rows.append({"Limit it exposes": limit, "Prasad": msg, "FlyBot 1.0": fe.flybot_reply(msg)})
        df_show(rows)
        custom = st.text_input("Try your own message to FlyBot", "I want to change my booking", key="s1_fly_custom")
        st.write("FlyBot:", fe.flybot_reply(custom))
        df_show([
            {"Limit": "Adaptability", "What happens": "Spots the word 'canceled' and sends a generic link. No rule for cancellation + snowstorm + snowboard.",
             "A modern agent": "Recognises the real goal, checks weather, searches flights, filters by snowboard check-in time."},
            {"Limit": "Scope", "What happens": "A hotel for a stranded passenger is outside its predefined world, so it refuses.",
             "A modern agent": "Sees a natural extension of the trip and calls a hotel search."},
            {"Limit": "Knowledge acquisition", "What happens": "New 'Business Lite' fare needs new rules, tests and a deploy.",
             "A modern agent": "Reads the current fare document at question time."},
        ])
        st.caption("You will build FlyBot 2.0 on page 4.")

    # ---- 3.6/3.7 modern
    with tabs[5]:
        st.markdown("#### What changes when AI enters the picture")
        df_show([
            {"From": "Narrow: specialists only", "To": "General and flexible: can work across domains"},
            {"From": "Rule-heavy: thousands of hand-written rules", "To": "Learning-powered: patterns learned from data"},
            {"From": "Rigid commands: MOVE_FORWARD(10)", "To": "Natural language: \"move forward a little and stop near the lamp post\""},
            {"From": "Stuck in one box", "To": "Tool-using and resourceful: APIs, databases, other services"},
        ])
        st.markdown("#### Six capabilities of modern AI agents, and where you meet them")
        df_show([
            {"Capability": "Perception", "Means": "Understand meaning, intent, context", "In this lab": "3.1 structured understanding"},
            {"Capability": "Reasoning & planning", "Means": "Break a goal into ordered steps", "In this lab": "3.3 planning and replanning"},
            {"Capability": "Memory", "Means": "Keep context so you never repeat yourself", "In this lab": "3.2 conversation list"},
            {"Capability": "Learning & adaptation", "Means": "Adjust to preferences without reprogramming", "In this lab": "3.4 lessons in memory"},
            {"Capability": "Tool use & action", "Means": "Call APIs, run code, book tickets", "In this lab": "Page 4"},
            {"Capability": "Collaboration & autonomy", "Means": "Work with humans or agents until done", "In this lab": "Pages 4 and 8"},
        ])
        plain("A Stage 5 agent searches flights Delhi → London. A Stage 6 agent, told you need to be in London for meetings, also checks your visa, looks for calendar clashes and books a hotel near the venue.")


# ----------------------------------------------------------------------------------------
# Page 4: Giving the concierge hands (tools + the agent loop)
# ----------------------------------------------------------------------------------------
def concierge_instructions(honest: bool) -> str:
    text = ("You are DilSeVector's concierge. You can read and change Prasad's calendar with the tools provided. "
            "Always get today's date with a tool before working out a date like 'this Saturday'. "
            "Check the calendar for clashes before adding anything.")
    if honest:
        text += (" You cannot book tables at the café itself; if asked, say so plainly and only add the calendar entry.")
    return text


def reset_concierge():
    st.session_state.s1_cal = fe.fresh_calendar()
    st.session_state.pop("s1_hist", None)
    st.session_state.pop("s1_last", None)


def page_tools():
    b = get_backend()
    st.title("4 · Giving the concierge hands")
    st.write("Back to Monday's request. We give the model what it lacked: **tools** (hands), a way to know today's date (eyes) and a **loop** (a second look).")
    dot('digraph { rankdir=LR; ' + NODE + ' User [fillcolor="#d0ebff"]; Model [label="Model\\n(decides)", fillcolor="#fff3bf"]; Code [label="Your code\\n(runs the tool)", fillcolor="#d3f9d8"]; '
        'User -> Model [label=" goal"]; Model -> Code [label=" tool call (a request)"]; Code -> Model [label=" tool result (feedback)"]; Model -> User [label=" final answer"]; }')
    st.success("**The key idea of every agent framework:** the model **decides** the next step, and your code **carries it out**.")
    if "s1_cal" not in st.session_state:
        reset_concierge()
    cal = st.session_state.s1_cal
    tabs = st.tabs(["4.1–4.3 Tools & one decision", "4.5 The whole loop", "4.6 The short form", "4.7 FlyBot 2.0"])

    # ---------------- tools and one decision
    with tabs[0]:
        st.markdown("### 4.1 Tools are ordinary Python functions")
        st.write("A tool is a normal function with three things the model can read: a **clear name**, **type hints** and a **docstring**. The model never sees the function body, so the docstring is effectively a prompt.")
        st.markdown("**Prasad's calendar** (a plain Python list; it starts with one cricket match):")
        calendar_table(cal)
        tools = fe.make_calendar_tools(cal, validate=False, with_update=False)
        c1, c2 = st.columns([1, 2])
        tname = c1.selectbox("Pick a tool and run it yourself, no model involved", list(tools), key="s1_tool_pick")
        import inspect
        params = list(inspect.signature(tools[tname]).parameters)
        sat = fe.next_saturday().isoformat()
        defaults = {"date_text": sat, "time_text": "19:00", "title": "Dinner at the pink café"}
        values = {p: c1.text_input(p, defaults.get(p, ""), key="s1_arg_%s_%s" % (tname, p)) for p in params}
        if c1.button("Run tool", key="s1_tool_run"):
            c2.code(tools[tname](**values), language="text")
        c1.button("Reset calendar", key="s1_cal_reset1", on_click=reset_concierge)
        with c2:
            st.markdown("### 4.2 What the model actually sees")
            st.caption("`bind_tools` builds this from your function: the name, the docstring's first line, and the `Args:` lines.")
            st.json(fe.tool_schema(tools[tname]))
            st.info("Notice what is **not** there: nothing says \"check for clashes before adding\" or \"never invent a date\". Rules like that exist only if you write them in a docstring or the system prompt.")

        st.markdown("### 4.3 One decision: the model asks, it does not act")
        goal = "Book a table for two at the pink café in Dadar this Saturday at 7 pm and add it to my calendar."
        st.code(goal, language="text")
        predict("will this single call change the calendar?")
        if st.button("Ask the model for ONE decision", type="primary", key="s1_one"):
            msgs = [{"role": "system", "content": concierge_instructions(True)}, {"role": "user", "content": goal}]
            d = b.decide(msgs, tools, fe.concierge_policy())
            st.write("**Text:**", repr(d["text"]))
            st.write("**Tool calls requested:**")
            for c in d["tool_calls"]:
                st.code("%s  args=%s" % (c["name"], c["args"]))
            st.metric("Events in the calendar", len(cal), help="Still the same as before: the model only requests; your code decides whether to act.")
            badge(b)
            st.warning("**The model only requests; your code decides.** That is where your control and safety live: you can refuse a request, check its arguments, or ask a human to confirm.")

    # ---------------- whole loop
    with tabs[1]:
        st.markdown("### 4.5 The whole loop")
        st.write("Run a decision → run the tools it asked for → append each result with the matching id → decide again. It stops when the model replies **without** requesting a tool, "
                 "or at `max_steps`. **The step limit is not optional**: a confused model can keep requesting tools until your quota runs out.")
        goals = {
            "Question: Am I free this Saturday evening?": "Am I free this Saturday evening?",
            "Goal B: book, with a Sunday fallback": ("Book a table for two at the pink café in Dadar this Saturday at 7 pm and add it to my calendar. "
                                                      "If I'm busy then, use Sunday at 7 pm instead."),
            "Follow-up: Actually, make it 8 pm. (continues the last run)": "Actually, make it 8 pm.",
            "Write my own": None,
        }
        gk = st.radio("Goal", list(goals), key="s1_goal_pick")
        goal = goals[gk] if goals[gk] else st.text_input("Your goal", "Am I free this Sunday morning?", key="s1_goal_custom")
        predict("how many tool calls will each goal need? Write a guess down.")
        with st.expander("⚙️ Make it break (homework HW1, HW2, HW3 as switches)", expanded=False):
            c1, c2 = st.columns(2)
            honest = c1.checkbox("Honesty sentence in the system prompt", True, key="s1_honest",
                                 help="'You cannot book tables at the café itself; say so plainly.' Without it many runs imply a booking that never happened.")
            with_update = c1.checkbox("HW1: give the agent an update_calendar_event tool", False, key="s1_upd")
            validate = c1.checkbox("HW2: add_calendar_event rejects bad dates with a helpful error", True, key="s1_val")
            sloppy = c2.checkbox("Simulate: the model sends date_text='Sunday' (not YYYY-MM-DD)", False, key="s1_sloppy")
            typo = c2.checkbox("Simulate: the model asks for a tool that does not exist ('check_calender')", False, key="s1_typo")
            robust = c2.checkbox("HW3: the loop sends an error back instead of crashing", True, key="s1_robust")
            max_steps = st.slider("max_steps", 1, 10, 8, key="s1_maxsteps")
            if b.live:
                st.caption("Live mode: the simulate-switches only steer the demo model. Gemini decides for itself; the prompt, tools, validation and step limit still apply.")
        c1, c2 = st.columns([1, 1])
        run = c1.button("Run the agent loop", type="primary", key="s1_loop_run")
        c2.button("Reset calendar and conversation", key="s1_cal_reset2", on_click=reset_concierge)
        if run:
            tools = fe.make_calendar_tools(cal, validate=validate, with_update=with_update)
            use_hist = gk.startswith("Follow-up") and "s1_hist" in st.session_state
            if gk.startswith("Follow-up") and not use_hist:
                st.info("No earlier run to continue; running it as a fresh conversation.")
            res = fe.run_agent_loop(b, goal, concierge_instructions(honest), tools, fe.concierge_policy(honest, sloppy, typo, with_update),
                                    max_steps=max_steps, robust=robust, history=st.session_state.get("s1_hist") if use_hist else None)
            st.session_state.s1_hist = res["messages"] if not res["crashed"] else st.session_state.get("s1_hist")
            st.session_state.s1_last = res
        res = st.session_state.get("s1_last")
        if res:
            render_events(res["events"])
            badge(b)
            m1, m2, m3 = st.columns(3)
            model_calls = sum(1 for m in res["messages"] if m["role"] == "assistant")
            m1.metric("Model calls so far", model_calls)
            m2.metric("Messages resent on the last call", len(res["messages"]) - 1)
            m3.metric("Events in calendar", len(cal))
            st.markdown("**Prasad's calendar now:**")
            calendar_table(cal)
            bad = [e for e in cal if e["date"].count("-") != 2]
            if bad:
                st.error("An event was saved with the date '%s'. `check_calendar` can never match it: the silent failure HW2 is about." % bad[0]["date"])
            if res["final"] and ("booked a table" in res["final"].lower()) and not honest:
                st.error("The agent claims a table is booked, but it has no tool that can book one. **An agent is only as honest as its instructions and tools allow.**")
            with st.expander("What the model saw on its last call (the conversation list is the agent's memory for this run)"):
                for m in res["messages"]:
                    label = {"system": "SystemMessage", "user": "HumanMessage", "assistant": "AIMessage", "tool": "ToolMessage"}[m["role"]]
                    extra = " requests " + ", ".join(c["name"] for c in m.get("tool_calls", [])) if m.get("tool_calls") else ""
                    st.markdown("`%s`%s %s" % (label, extra, (m["content"] or "")[:160]))
        show_code("The loop in code (the notebook's run_agent)", """
def run_agent(goal, instructions, tools, max_steps=6):
    tools_by_name = {tool.__name__: tool for tool in tools}
    model_with_tools = llm.bind_tools(tools)
    messages = [SystemMessage(content=instructions), HumanMessage(content=goal)]
    for step in range(1, max_steps + 1):
        reply = model_with_tools.invoke(messages)
        messages.append(reply)
        if len(reply.tool_calls) == 0:                      # no tool requested: the agent is done
            return reply.text, messages
        for tool_call in reply.tool_calls:
            tool_function = tools_by_name[tool_call["name"]]    # HW3: what if this name does not exist?
            result = str(tool_function(**tool_call["args"]))     #       or the arguments are wrong?
            messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
    return "Stopped: step limit reached without a final answer.", messages
""")
        plain("Map it onto the loop: the tool result is **perception**, the model's next request is **reasoning and planning**, running the tool is **action**, and appending the result is **feedback**.")

    # ---------------- create_agent
    with tabs[2]:
        st.markdown("### 4.6 The short form: `create_agent`")
        st.write("You wrote the loop by hand so you know exactly what it does. In practice LangChain runs it for you. `create_agent` takes a model, a list of tools and a system prompt, "
                 "and returns an agent with the same loop inside (built on LangGraph, covered in a later session).")
        st.code("""
from langchain.agents import create_agent

concierge_agent = create_agent(model=llm, tools=calendar_tools, system_prompt=concierge_instructions)
result = concierge_agent.invoke({"messages": [HumanMessage(content=goal_b)]})
""", language="python")
        c1, c2 = st.columns(2)
        c1.metric("Lines in the hand-written loop", "≈ 14")
        c2.metric("Lines with create_agent", "3")
        plain("Same behaviour, a quarter of the code. The framework removed the **plumbing**; it did not remove any **decisions**. You still chose the tools, wrote the docstrings, "
              "and wrote the instructions that keep the agent honest.")

    # ---------------- FlyBot 2.0
    with tabs[3]:
        st.markdown("### 4.7 FlyBot, rebuilt as an agent")
        st.write("Prasad is stranded in Delhi. FlyBot 2.0 has four tools standing in for the airline's systems. **The data is pretend**; in a real app each function would call an API.")
        airline = fe.make_airline_tools()
        with st.expander("The four tools (what the model reads)"):
            for t in airline.values():
                st.json(fe.tool_schema(t))
        pick = st.radio("Message to send (they build on each other, as a conversation)", list(fe.FLYBOT_MESSAGES), horizontal=True, key="s1_fly_pick",
                        format_func=lambda k: {"Adaptability": "1 · Cancellation + snowboard", "Scope": "2 · Hotel near Delhi airport", "Knowledge": "3 · Business Lite baggage"}[k])
        st.markdown("> " + fe.FLYBOT_MESSAGES[pick])
        predict("three flights leave tomorrow. Which should the agent recommend, and why? (Two facts matter, from two different tools.)")
        c1, c2 = st.columns(2)
        if c1.button("Send to FlyBot 2.0", type="primary", key="s1_fly_go"):
            instr = ("You are FlyBot 2.0, an airline assistant. Use the tools to check facts before you answer. "
                     "Do not book anything yourself: propose the best option with your reasons and ask Prasad to confirm.")
            res = fe.run_agent_loop(b, fe.FLYBOT_MESSAGES[pick], instr, airline, fe.flybot_policy, max_steps=8, history=st.session_state.get("s1_fly_hist"))
            st.session_state.s1_fly_hist = res["messages"]
            st.session_state.s1_fly_last = res
        if c2.button("New conversation", key="s1_fly_reset"):
            st.session_state.pop("s1_fly_hist", None)
            st.session_state.pop("s1_fly_last", None)
            st.rerun()
        if "s1_fly_last" in st.session_state:
            render_events(st.session_state.s1_fly_last["events"])
            badge(b)
            st.caption("Messages in this conversation so far: %d (the whole history is passed back, so the agent remembers the context)." % len(st.session_state.s1_fly_hist))
        plain("A good run checks the weather alert (nothing before 10:00, so 6E 2071 is out), searches flights, reads the snowboard rule, and recommends 6E 2073 at 11:05. "
              "No rule was written for \"snowstorm plus snowboard\": the agent combined tool results. But **an agent cannot reach beyond the tools it is given**: ask about the fare and it has to admit it has no tool for that.")


# ----------------------------------------------------------------------------------------
# Page 5: Why agents, and why not always
# ----------------------------------------------------------------------------------------
def page_why():
    st.title("5 · Why agents, and why not always")
    st.write("Real life is not single questions and single answers. An agent can observe, plan, act with several tools, remember what it has done and adapt. That is why agents bridge **human intent** and **real-world execution**. But flexibility has costs.")
    t1, t2, t3 = st.tabs(["Calls & time", "Reliability compounds", "Workflow or agent? (sorting game)"])
    with t1:
        st.markdown("\"Am I free on Saturday?\" took **three model calls**. A calendar app answers it with **none**.")
        cal = fe.fresh_calendar()
        res = fe.run_agent_loop(fe.Backend(), "Am I free this Saturday evening?", "x", fe.make_calendar_tools(cal), fe.concierge_policy())
        sent, rows = [], []
        n = 0
        for i, m in enumerate(res["messages"]):
            if m["role"] == "assistant":
                n += 1
                sent.append(i)
                rows.append({"model call": n, "messages sent": i})
        c1, c2 = st.columns(2)
        c1.metric("Calendar app", "0 model calls")
        c2.metric("Agent", "%d model calls" % n)
        df_show(rows)
        st.bar_chart(pd.DataFrame({"messages sent": sent}, index=["call %d" % (i + 1) for i in range(len(sent))]))
        plain("Every call resends the whole conversation, so each step costs a little more than the last. Also: run the same goal several times and the traces differ, so you cannot write a test that expects an exact sequence.")
    with t2:
        st.markdown("If each step is right 95% of the time, a run is right only if **every** step is.")
        p = st.slider("Chance a single step is right", 0.80, 0.995, 0.95, 0.005, key="s1_rel_p")
        steps = list(range(1, 21))
        df = pd.DataFrame({"steps": steps, "your p": [fe.chance_all_right(p, s) * 100 for s in steps],
                           "p = 0.99": [fe.chance_all_right(0.99, s) * 100 for s in steps],
                           "p = 0.90": [fe.chance_all_right(0.90, s) * 100 for s in steps]}).set_index("steps")
        st.line_chart(df)
        df_show([{"Steps in a run": s, "Chance all steps are right (p = %.3f)" % p: "%.0f%%" % (fe.chance_all_right(p, s) * 100)} for s in (1, 3, 5, 10, 20)])
        half = next((s for s in range(1, 200) if fe.chance_all_right(p, s) < 0.5), None)
        st.metric("Steps until the run is more likely wrong than right", half)
        plain("A 10-step agent built from excellent 95% steps fails four times in ten. So before reaching for an agent ask: **do we always need an agent?** No. If the problem is predictable and can be solved by predefined steps, that is a **workflow**.")
    with t3:
        st.markdown("A **workflow** is a predefined series of steps executed in a set order, like submitting an expense report. Use one when the process is **predictable**, **reliability matters more than creativity**, or **efficiency** is key. "
                    "Use an **agent** when you cannot know in advance what task, or what task parameters, you will get.")
        items = []
        for i, (scn, ans, why) in enumerate(fe.WORKFLOW_SCENARIOS):
            items.append((scn, ["Workflow", "Agent"], 0 if ans == "Workflow" else 1, why))
        quiz("s1_sort", items)


# ----------------------------------------------------------------------------------------
# Page 6: Four workflow patterns
# ----------------------------------------------------------------------------------------
def page_patterns():
    b = get_backend()
    st.title("6 · Four workflow patterns")
    st.write("Battle-tested recipes engineers use again and again. They do not implement anything for you, but they show a path that works. In all four, **your code decided the order of steps before any request arrived**.")
    tabs = st.tabs(["6.1 Evaluator-optimizer", "6.2 Parallelization", "6.3 Chaining", "6.4 Routing", "6.5 Side by side"])

    with tabs[0]:
        dot('digraph { rankdir=LR; ' + NODE + ' Producer [fillcolor="#d0ebff"]; Grader [fillcolor="#fff3bf"]; Producer -> Grader [label=" submission"]; Grader -> Producer [label=" feedback (rejected)", color=red]; Grader -> Accepted [label=" accepted", color=green]; Accepted [fillcolor="#d3f9d8"]; }')
        st.write("A **producer** creates output; a **grader** checks it against criteria and returns feedback; the loop repeats until accepted or the round limit is hit. Example: a DilSeVector profile bio for Prasad.")
        active = st.multiselect("Active criteria (the grader's rubric)", list(fe.BIO_CRITERIA), default=list(fe.BIO_CRITERIA),
                                format_func=lambda k: fe.BIO_CRITERIA[k], key="s1_eo_active")
        max_rounds = st.slider("Round limit (essential, like the agent's step limit)", 1, 5, 3, key="s1_eo_rounds")
        if st.button("Run the loop", type="primary", key="s1_eo_go"):
            rounds = fe.run_eval_opt(active or list(fe.BIO_CRITERIA), max_rounds)
            for r in rounds:
                with st.container(border=True):
                    st.markdown("**Round %d**: %d words, accepted = **%s**" % (r["round"], r["words"], r["accepted"]))
                    st.write(r["bio"])
                    st.markdown("  ".join(("✅ " if r["results"][k] else "❌ ") + k for k in active) if active else "")
                    if not r["accepted"]:
                        st.caption("Grader feedback: " + r["feedback"])
            accepted = rounds[-1]["accepted"]
            n = len(rounds)
            st.metric("Model calls this would cost (producer + grader)", 2 * n if accepted else 2 * n + 1)
            if not accepted:
                st.error("Round limit reached without acceptance: the loop must stop. That is why the limit exists.")
        plain("The grader names what failed and the next draft fixes it. **Weakness:** the grader is another model with its own mistakes; if producer and grader are the same model they can share blind spots. "
              "For mechanical criteria (a word count) plain code is a cheaper and more reliable grader, which is exactly what this lab uses.")

    with tabs[1]:
        st.write("A materials app: the user uploads a part and gets the best material. One giant prompt with criteria for six materials makes the model juggle dozens of competing criteria. "
                 "Better: **split** the task, run one focused evaluation per material **in parallel**, then **aggregate** with a final call.")
        dot('digraph { rankdir=LR; ' + NODE + ' Part -> Metal; Part -> Polymer; Part -> Ceramic; Part -> Composite; Metal -> Aggregator; Polymer -> Aggregator; Ceramic -> Aggregator; Composite -> Aggregator; Aggregator -> Recommendation; Aggregator [fillcolor="#fff3bf"]; }')
        part = st.selectbox("Part", list(fe.PARTS), key="s1_par_part")
        mats = st.multiselect("Materials to evaluate", list(fe.MATERIAL_CRITERIA), default=["Metal", "Polymer", "Ceramic", "Composite"], key="s1_par_mats")
        data = fe.PARTS[part]
        st.markdown("**Part:** " + data["description"])
        with st.expander("One focused prompt (per material)"):
            if mats:
                st.code("You are a materials engineer. Judge ONLY whether %s suits this part, using these criteria: %s.\nPart: %s\nGive a suitability score from 1 to 10 and a reason in under 60 words." % (mats[0], fe.MATERIAL_CRITERIA[mats[0]], data["description"]), language="text")
        if st.button("Run %d focused evaluations, then aggregate" % len(mats), type="primary", key="s1_par_go") and mats:
            rows = sorted([{"Material": m, "Score": data["scores"][m][0], "Reason": data["scores"][m][1]} for m in mats], key=lambda r: -r["Score"])
            df_show(rows)
            st.bar_chart(pd.DataFrame(rows).set_index("Material")["Score"])
            primary, alt = rows[0], rows[1] if len(rows) > 1 else None
            st.success("**Aggregator:** primary = **%s**%s. Key limitation of the primary choice: %s" % (primary["Material"], (", alternative = **%s**" % alt["Material"]) if alt else "", primary["Reason"].split(";")[-1].strip()))
            c1, c2, c3 = st.columns(3)
            c1.metric("Model calls (N + 1)", len(mats) + 1)
            c2.metric("Free tier (one call per 15 s)", "≈ %d s" % ((len(mats) + 1) * 15))
            c3.metric("Paid tier (truly parallel)", "≈ 2 calls of latency")
            st.caption("Scores here are scripted stand-ins for the model's judgements (in Live mode the notebook runs `llm.batch`).")
        df_show([
            {"Benefit": "Focused attention", "Why": "Each call weighs one material's criteria properly"},
            {"Benefit": "Easier optimisation", "Why": "If the metal analysis is weak, fix only that prompt"},
            {"Benefit": "Scalability", "Why": "Adding elastomer and wood means adding two prompts, nothing else (tick them above)"},
            {"Benefit": "Reliability", "Why": "Less juggling, more consistent results"},
        ])
        st.caption("Use it when a decision splits into **independent** judgements. If the sub-tasks depend on each other, you want a chain.")

    with tabs[2]:
        st.write("**Chaining** breaks a large task into sequential subtasks, each building on the last. It solves the **long prompt problem**: even when every constraint is clear, some drafts break a few. "
                 "Accept an imperfect first draft, then send a second, *focused* request that only revises.")
        dot('digraph { rankdir=LR; ' + NODE + ' "Step 1\\nwrite the draft" -> "Step 2\\nrevise (remove AI mentions, emojis, cliches)" -> "Final article"; }')
        st.code("CONSTRAINTS: DO NOT mention being an AI. DO NOT use emojis! DO NOT use cliched or cringey writing! ... Keep it under 350 words.", language="text")
        if st.button("Run the chain", type="primary", key="s1_ch_go"):
            draft = b.ask("Write a comprehensive, high-quality article about 'Prompt Engineering for AI Language Models' for a technical blog. Constraints: do not mention being an AI; "
                          "no emojis; no cliched or cringey writing. Keep it under 350 words.", demo=fe.DRAFT_ARTICLE)
            revised = b.ask("Revise the article below. 1. Remove any place where the text identifies the author as an AI. 2. Remove all emojis. 3. Replace cringey writing with text a technical writer "
                            "would write. Return only the revised article.\n\n" + draft, demo=lambda: fe.revise_article(draft))
            c1, c2 = st.columns(2)
            c1.markdown("**Draft (step 1)**")
            c1.write(draft)
            c2.markdown("**Revised (step 2)**")
            c2.write(revised)
            before, after = fe.count_violations(draft), fe.count_violations(revised)
            df_show([{"Violation": k, "In draft": before[k], "After revision": after[k]} for k in before])
            badge(b)
        st.code("""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

write_chain = ChatPromptTemplate.from_template(article_template) | llm | StrOutputParser()
revise_chain = ChatPromptTemplate.from_template(revision_template) | llm | StrOutputParser()
revised = revise_chain.invoke({"article": write_chain.invoke({"article_subject": article_subject})})
""", language="python")
        st.caption("`LLMChain` and `SequentialChain` were removed in LangChain 1.x; if you meet them in older tutorials, translate to the pipe form above.")

    with tabs[3]:
        st.write("A **routing** workflow first **categorises** the input, then sends it to **one** specialised pipeline. \"Python functions\" and \"surfing\" should not get the same video script.")
        dot('digraph { rankdir=LR; ' + NODE + ' Topic -> Router [label=" categorise"]; Router [fillcolor="#fff3bf"]; Router -> Educational; Router -> Comedy; Router -> "Personal vlog"; Router -> Reviews; Router -> Storytelling; Router -> Entertainment; }')
        topics = ["What an embedding is, explained in 60 seconds", "How I met my partner at a pink café in Dadar, thanks to vector search",
                  "Honest review: is the new espresso machine worth it?", "Surfing in Goa: top 10 spots"]
        topic = st.selectbox("Topic", topics + ["Write my own"], key="s1_rt_pick")
        if topic == "Write my own":
            topic = st.text_input("Your topic", "Python functions for beginners", key="s1_rt_custom")
        predict("which genre will this topic get?")
        if st.button("Route it", type="primary", key="s1_rt_go"):
            demo = fe.demo_route(topic)
            route = b.structured(fe.ROUTER_SCHEMA, "Categorize the topic of a video into one of the listed categories.\n<topic>%s</topic>" % topic, demo=lambda: {"genre": demo["genre"]})
            genre = route["genre"]
            st.success("Routed to **%s**" % genre)
            if b.last_source != "gemini":
                st.caption("Keyword scores (demo router): " + ", ".join("%s=%d" % kv for kv in demo["scores"].items()))
            st.markdown("**Specialised prompt template:** " + fe.GENRE_PROMPTS[genre].replace("[TOPIC]", "**" + topic + "**"))
            script = b.ask(fe.GENRE_PROMPTS[genre].replace("[TOPIC]", topic) + " Keep it under 100 words.", demo=lambda: fe.demo_script(genre, topic))
            st.write(script)
            badge(b)
        with st.expander("All six genre templates"):
            df_show([{"Genre": g, "Script prompt": p} for g, p in fe.GENRE_PROMPTS.items()])
        st.caption("Why an `enum` in the schema: the router can only answer with one of your six genres. Each input goes to **only one** pipeline, so each pipeline can be tuned for its kind of input.")

    with tabs[4]:
        df_show([
            {"Pattern": "Evaluator-optimizer", "Shape": "Produce, grade, feed back, repeat", "Calls": "2 up to 2 × rounds", "Use it when": "You can state clear acceptance criteria", "Avoid when": "You cannot say what 'good' means"},
            {"Pattern": "Parallelization", "Shape": "Split, run independently, aggregate", "Calls": "N + 1", "Use it when": "One decision splits into independent judgements", "Avoid when": "Sub-tasks depend on each other"},
            {"Pattern": "Chaining", "Shape": "Step 1 → step 2 → step 3", "Calls": "One per step", "Use it when": "Several dependent parts, or constraints a long prompt keeps dropping", "Avoid when": "One prompt already does the job"},
            {"Pattern": "Routing", "Shape": "Classify, then one specialised path", "Calls": "1 + the path", "Use it when": "Inputs come in clearly different kinds", "Avoid when": "Categories overlap and the router guesses"},
        ])


# ----------------------------------------------------------------------------------------
# Page 7: Workflow or agent? Frameworks
# ----------------------------------------------------------------------------------------
def page_choose():
    st.title("7 · Workflow or agent? Frameworks")
    dot('digraph { rankdir=LR; ' + NODE + ' subgraph cluster_w { label="Workflow: your code fixes the order"; A1 [label="Step 1"]; A2 [label="Step 2"]; A3 [label="Step 3"]; A1 -> A2 -> A3; } '
        'subgraph cluster_a { label="Agent: the model picks the next action"; M [label="Model", fillcolor="#fff3bf"]; T1 [label="Tool X"]; T2 [label="Tool Y"]; M -> T1 -> M; M -> T2 -> M; } }')
    c1, c2 = st.columns(2)
    c1.markdown("#### 🧾 Workflow = a recipe\nYou know the ingredients and the steps. Predictable, testable, reliable.")
    c2.markdown("#### 🗺️ Agent = a new city with a phone, some cash and a few apps\nNo plan, but it finds food, navigates, adapts to situations nobody prepared it for.")
    df_show([
        {"": "Benefits", "Workflows": "LLM focuses on one subtask (higher accuracy); easy to test because you know each step; predictable; suited to well-defined problems",
         "Agents": "Flexible experience; combine tools in unexpected ways; handle unanticipated situations; can ask the user for more input"},
        {"": "Downsides", "Workflows": "Much less flexible; constrained user experience; more up-front design",
         "Agents": "Lower task-completion rate; harder to test and evaluate; less predictable"},
    ])
    st.subheader("Decision helper")
    st.write("Tick what is true for your task.")
    q = [
        ("I can picture the exact steps the LLM should go through", -2),
        ("My app's interface limits users to a fixed set of tasks", -1),
        ("A wrong or skipped step causes real damage (payments, orders)", -2),
        ("I cannot know in advance what task or task parameters I will get", 2),
        ("The right tool or action depends on what earlier results say", 2),
        ("Users will ask open-ended things and expect me to figure it out", 2),
    ]
    score = 0
    for i, (text, w) in enumerate(q):
        if st.checkbox(text, key="s1_dec_%d" % i):
            score += w
    if score <= -2:
        st.success("Lean **workflow**: predictable steps, reliability first. (Keep an agent for the one step that truly cannot be predicted.)")
    elif score >= 2:
        st.warning("Lean **agent**: but add a step limit, human approval for sensitive actions and a way to check the output.")
    else:
        st.info("Mixed: consider a **workflow whose first step is an agent**, the most common good design.")
    st.subheader("Tools for building them")
    st.markdown("**No-code tools** such as **n8n** and Langflow let you build automations by connecting boxes (n8n is open source and self-hostable). "
                "*This course builds agents in code*, with LangGraph and CrewAI in later sessions, because code gives you full control and the ability to fix things when they break.")
    df_show([{"Framework": n, "Known for": d} for n, d in fe.FRAMEWORKS])


# ----------------------------------------------------------------------------------------
# Page 8: Capstone: the Manali travel planner
# ----------------------------------------------------------------------------------------
def show_itinerary(days):
    rows = []
    for d in days:
        for i in d["items"]:
            rows.append({"Day": d["day"], "When": i["time"], "Place": i["place"], "What": i["text"], "In the guide?": "✅" if i["place"] in fe.GUIDE else "❌ unverified"})
    df_show(rows)


def page_capstone():
    b = get_backend()
    st.title("8 · Capstone: planning the first trip")
    story("Prasad's flight is sorted. Now he and his partner want a day-by-day Manali plan they can load into their calendars. We build it three ways and compare: "
          "**GenAI** (one prompt), an **AI agent** (one agent with a search tool), and the **agentic planner** (four roles in sequence).")
    st.code("Create a 3-day itinerary for Manali. Preferences: We are vegetarian, we prefer quiet places to crowded ones, and we like short walks.", language="text")
    st.caption("Demo data below is a small **pretend** travel guide modelled loosely on Wikivoyage. Live mode and the real-lookup button use the actual Wikivoyage API (needs internet).")
    tabs = st.tabs(["GenAI: one prompt", "AI agent: one tool", "Agentic planner: four roles", "Did it come from the research?", "Into the calendar"])
    prefs = "vegetarian, quiet places, short walks"

    with tabs[0]:
        if st.button("Ask one prompt", type="primary", key="s1_cap_genai"):
            live = b.ask("Create a 3-day itinerary for Manali. Preferences: We are vegetarian, we prefer quiet places to crowded ones, and we like short walks. Use the format 'Day 1: ...', 'Day 2: ...'.",
                         demo=None)
            st.session_state.s1_genai_live = live if (b.last_source == "gemini") else None
            st.session_state.s1_genai_done = True
        if st.session_state.get("s1_genai_done"):
            if st.session_state.get("s1_genai_live"):
                st.write(st.session_state.s1_genai_live)
            else:
                show_itinerary(fe.itinerary_genai())
            badge(b)
            st.error("Fluent and plausible, but **nothing was looked up**. Whether a café exists, whether a road is open in December, whether a place is quiet: it all comes from training data with a cut-off date, and the model cannot tell you which details are stale.")

    with tabs[1]:
        st.markdown("One agent, one tool (`search_travel_guide`). The model decides what to look up and when it has enough to write the plan.")
        dot('digraph { rankdir=LR; ' + NODE + ' Request -> Agent [fillcolor="#fff3bf"]; Agent -> "search_travel_guide(Manali)" -> Agent; Agent -> Itinerary; }')
        if st.button("Run the single agent", type="primary", key="s1_cap_agent"):
            st.session_state.s1_agent_done = True
        if st.session_state.get("s1_agent_done"):
            st.markdown("🛠️ **Model requests** `search_travel_guide(place='Manali')` → ⚙️ *your code runs it* → guide text (pretend):")
            df_show([{"Place": n, "Kind": v["kind"], "Quiet": "yes" if v["quiet"] else "no", "Note": v["note"]} for n, v in fe.GUIDE.items()])
            show_itinerary(fe.itinerary_single_agent())
            st.warning("Limits: it researches and plans everything itself so results can be shallow; it can still hallucinate; **nothing refines the plan; nothing checks it**. Note it put noisy Mall Road and Solang Valley in a plan for people who asked for quiet.")
        if st.button("Try the real Wikivoyage lookup (needs internet)", key="s1_cap_real"):
            try:
                st.code(fe.search_travel_guide("Manali")[:1500], language="text")
            except Exception as exc:
                st.error("Could not reach Wikivoyage from here: %s" % exc)

    with tabs[2]:
        c1, c2 = st.columns(2)
        hw4 = c1.checkbox("HW4: give the Planner the preferences", False, key="s1_hw4", help="Otherwise the Planner drafts a plan the Optimizer must rework.")
        hw5 = c2.checkbox("HW5: let QA see the research", False, key="s1_hw5", help="Otherwise QA only ever sees the itinerary, so it cannot tell a real place from an invented one.")
        dot('digraph { rankdir=LR; ' + NODE + ' R [label="Researcher\\n(an AGENT with a tool)", fillcolor="#fff3bf"]; P [label="Planner"]; O [label="Optimizer"]; Q [label="QA"]; R -> P -> O -> Q -> "Final plan"; }')
        out = fe.role_outputs(pass_prefs_to_planner=hw4, qa_sees_research=hw5, prefs=prefs)
        with st.expander("1 · Researcher: summarise only what the guide says", expanded=False):
            st.write(out["research"])
        with st.expander("2 · Planner: draft a day-by-day itinerary from the research" + (" (with preferences)" if hw4 else "")):
            show_itinerary(out["draft"])
        with st.expander("3 · Optimizer: refine for preferences and add 'small enhancements'"):
            show_itinerary(out["optimized"])
            st.caption("The instruction \"add small enhancements (hidden gems)\" asks a role with **no tool** to add things, which it can only do from memory.")
        with st.expander("4 · QA: verify and correct", expanded=True):
            show_itinerary(out["verified"])
            if not hw5:
                st.caption("QA is told to \"remove hallucinations\" but only sees the itinerary, so the unverified hidden gem sailed through.")
        plain("Who decided each step? The orchestrator, in a fixed order. Only the researcher made choices. So this \"agentic\" planner is a **chaining workflow whose first step is an agent**: "
              "a good, common design. Just do not believe a role called \"QA\" guarantees quality: it is a prompt, only as good as the information it receives.")
        st.session_state.s1_planner_out = out

    with tabs[3]:
        st.write("All three itineraries read well, so reading them will not tell you which to trust. An LLM can do a first check: list the places in the itinerary and whether each appears in the research. "
                 "This is **LLM-as-judge** (the RAGAS faithfulness idea). Here the check is plain code over a known place list.")
        out = fe.role_outputs(pass_prefs_to_planner=st.session_state.get("s1_hw4", False), qa_sees_research=st.session_state.get("s1_hw5", False))
        cands = {"GenAI (no tool)": fe.itinerary_genai(), "Single agent": fe.itinerary_single_agent(), "Agentic planner (QA output)": out["verified"]}
        rows = []
        for name, days in cands.items():
            g = fe.grounding_check(days)
            rows.append({"Itinerary": name, "Places found in the guide": len(g["found_in_research"]), "Not in the guide": len(g["not_in_research"]), "Unverified places": ", ".join(g["not_in_research"]) or "none"})
        df_show(rows)
        st.bar_chart(pd.DataFrame(rows).set_index("Itinerary")[["Places found in the guide", "Not in the guide"]])
        for name, why in fe.NOT_IN_GUIDE.items():
            st.markdown("- **%s**: %s" % (name, why))
        plain("\"Not in the research\" does **not** mean false; it means **unverified**. The judge can be wrong too, so treat its list as a place to start checking.")

    with tabs[4]:
        st.write("The original app exports the final plan as an `.ics` file, which Google Calendar, Outlook and Apple Calendar all import.")
        out = fe.role_outputs(pass_prefs_to_planner=st.session_state.get("s1_hw4", False), qa_sees_research=st.session_state.get("s1_hw5", False))
        days = out["verified"]
        text = fe.itinerary_text(days)
        start = st.date_input("Trip starts on", fe.date.today() + fe.timedelta(days=60), key="s1_ics_start")
        regex_bytes, n_regex = fe.generate_ics(text, start)
        fixed_bytes, n_fixed = fe.generate_ics(text, start, structured_days=days)
        c1, c2, c3 = st.columns(3)
        c1.metric("Days in the plan", len(days))
        c2.metric("Events (regex on text)", n_regex, delta=n_regex - len(days), delta_color="inverse")
        c3.metric("Events (structured plan)", n_fixed)
        if n_regex != len(days):
            st.error("HW7: Day 2's text says \"Save some energy for Day 3 because...\". The pattern `Day (\\d+)[:\\s]+(.*?)(?=Day \\d+|$)` treats that as the start of another day, so the calendar gets the wrong number of events. "
                     "Asking the planner for **structured output** (page 3, stage 1) removes the guesswork.")
        st.text_area("The plan text the regex sees", text, height=150, disabled=True, key="s1_ics_text")
        d1, d2 = st.columns(2)
        d1.download_button("Download .ics (regex version)", regex_bytes, file_name="manali_itinerary_regex.ics", mime="text/calendar")
        d2.download_button("Download .ics (structured version)", fixed_bytes, file_name="manali_itinerary.ics", mime="text/calendar")


# ----------------------------------------------------------------------------------------
# Page 9: Review
# ----------------------------------------------------------------------------------------
def page_review():
    st.title("9 · Review, quiz & cheat sheet")
    t1, t2, t3, t4 = st.tabs(["GenAI vs agents vs agentic AI", "Quiz", "Homework map", "Cheat sheet"])
    with t1:
        df_show([
            {"Feature": "Core ability", "GenAI (creative engine)": "Generates content", "AI agents (task executors)": "Understands goals, uses tools, takes actions", "Agentic AI (true collaborators)": "Plans, adapts, collaborates, learns"},
            {"Feature": "Trigger", "GenAI (creative engine)": "A prompt every time", "AI agents (task executors)": "A clear task", "Agentic AI (true collaborators)": "Self-manages broad, vague or long-term goals"},
            {"Feature": "Memory", "GenAI (creative engine)": "None", "AI agents (task executors)": "Limited, short-term", "Agentic AI (true collaborators)": "Long-term, evolving"},
            {"Feature": "Example", "GenAI (creative engine)": "Write a poem", "AI agents (task executors)": "Book a flight and add it to the calendar", "Agentic AI (true collaborators)": "Build, test and deploy a recipe website"},
            {"Feature": "Human role", "GenAI (creative engine)": "Instruct every step", "AI agents (task executors)": "Set a task, agent executes", "Agentic AI (true collaborators)": "Partner: co-create and refine"},
            {"Feature": "Analogy", "GenAI (creative engine)": "A talented writer waiting for prompts", "AI agents (task executors)": "A personal assistant", "Agentic AI (true collaborators)": "A project manager and co-worker"},
            {"Feature": "In this lab", "GenAI (creative engine)": "Page 1; 8 (one prompt)", "AI agents (task executors)": "Page 4 (concierge, FlyBot 2.0)", "Agentic AI (true collaborators)": "Page 8 (four-role planner)"},
        ])
        plain("The **same model** sits inside all three columns; the difference is the tools, loop and memory built around it. \"Learns from experience\" and \"long-term memory\" happen only if someone builds them.")
        st.success("**One fact no agent or framework changes:** nothing in an agent checks whether its answer is true. The model only requests actions; your code performs them; the only facts you can trust are those you can trace to a tool result.")
    with t2:
        quiz("s1_quiz", [
            ("What does a tool call from the model do?", ["Runs the function", "Asks your code to run the function", "Changes the model's weights"], 1, "The model only requests; your code decides whether to act."),
            ("Which stage added an internal model of the world and planning?", ["Rule-based", "Reactive", "Deliberative"], 2, "Deliberative agents think ahead using a world model."),
            ("A bare LLM 'remembers' earlier messages only if...", ["It was trained on them", "You send them back in the prompt", "You use a bigger model"], 1, "Memory is whatever you choose to send back."),
            ("If every step is 95% reliable, a 10-step run is right about...", ["95%", "80%", "60%"], 2, "0.95^10 ≈ 0.60."),
            ("The step limit in an agent loop exists to...", ["Make it faster", "Stop a confused model from looping forever", "Improve accuracy"], 1, "Runaway loops burn quota."),
            ("You know the exact steps every input needs. Use a...", ["Workflow", "Agent"], 0, "Predictable steps favour a workflow."),
            ("Which pattern sends each input down exactly one specialised path?", ["Chaining", "Routing", "Parallelization"], 1, "Routing categorises, then specialises."),
            ("In the four-role planner, who actually made decisions at run time?", ["All four roles", "Only the researcher (an agent)", "None"], 1, "The orchestrator fixed the order; only the researcher chose tool calls."),
            ("'Not in the research' in the grounding check means the place is...", ["False", "Unverified", "Safe"], 1, "Unverified, not necessarily false."),
        ])
    with t3:
        df_show([
            {"#": "HW1", "Level": "Easy", "Task": "'Actually, make it 8 pm': write the missing tool", "Try it": "Page 4 → whole loop → tick the update tool, run goal B, then the follow-up"},
            {"#": "HW2", "Level": "Medium", "Task": "Make add_calendar_event reject bad dates in a way the model can learn from", "Try it": "Page 4 → 'model sends Sunday' with validation on/off"},
            {"#": "HW3", "Level": "Medium", "Task": "Stop a wrong tool name or argument from crashing the agent", "Try it": "Page 4 → 'tool that does not exist' with the robust loop on/off"},
            {"#": "HW4", "Level": "Easy", "Task": "Give the planner the preferences", "Try it": "Page 8 → planner tab"},
            {"#": "HW5", "Level": "Medium", "Task": "Let the QA role see the research", "Try it": "Page 8 → planner tab"},
            {"#": "HW6", "Level": "Hard", "Task": "Turn QA into an evaluator-optimizer loop", "Try it": "Page 6.1 for the pattern; build it in the notebook"},
            {"#": "HW7", "Level": "Easy", "Task": "Fix the calendar export (ideally with structured output)", "Try it": "Page 8 → calendar tab"},
            {"#": "HW8", "Level": "Medium", "Task": "Give the concierge its coffee pages back (a search_coffee_pages tool over your Pinecone index)", "Try it": "Notebook; see Session 3 for the memory side"},
            {"#": "HW9", "Level": "Medium", "Task": "Parallelization from a photo", "Try it": "Notebook (needs an image-capable model)"},
            {"#": "HW10", "Level": "Easy", "Task": "Passive or proactive? Raise exactly one concern and ask exactly one question", "Try it": "Page 2 → Goal-driven & proactive (edit the prompt in Live mode)"},
        ])
        st.caption("Suggested order: HW10, then HW1-3 together (the agent loop), then the planner (HW4, HW7, HW5, HW6 last), then HW8 and HW9.")
    with t4:
        df_show([
            {"Idea": "Agent loop", "One line": "Perceive, reason and plan, act, check the result, repeat", "Code": "Page 4"},
            {"Idea": "The key question", "One line": "Who decides the next step? Workflow: your code. Agent: the model", "Code": "Pages 5 to 8"},
            {"Idea": "Structured output", "One line": "Get a dictionary back instead of free text", "Code": "llm.with_structured_output(schema)"},
            {"Idea": "Memory", "One line": "Whatever you send back to the model", "Code": "a list of messages; the lessons list"},
            {"Idea": "Planning", "One line": "Ask for steps first, act later, replan on surprises", "Code": "planner_llm"},
            {"Idea": "Tool", "One line": "A function with a clear name, type hints and a docstring", "Code": "get_today, check_calendar, ..."},
            {"Idea": "Tool call", "One line": "The model's request; nothing runs until your code runs it", "Code": "reply.tool_calls"},
            {"Idea": "Tool result", "One line": "Sent back with the matching id", "Code": "ToolMessage(content=..., tool_call_id=...)"},
            {"Idea": "Agent, long form", "One line": "The loop by hand, with a step limit", "Code": "run_agent(...)"},
            {"Idea": "Agent, short form", "One line": "The same loop from LangChain", "Code": "create_agent(model, tools, system_prompt)"},
            {"Idea": "Evaluator-optimizer", "One line": "Produce, grade, feed back, until accepted", "Code": "bio writer"},
            {"Idea": "Parallelization", "One line": "Independent sub-tasks, then aggregate", "Code": "llm.batch(...)"},
            {"Idea": "Chaining", "One line": "One focused step after another", "Code": "write_chain | revise_chain"},
            {"Idea": "Routing", "One line": "Classify, then one specialised path", "Code": "router_llm + genre_prompts"},
            {"Idea": "LLM-as-judge", "One line": "Use a model to check another model's output against a source", "Code": "grounding check"},
        ])


# ----------------------------------------------------------------------------------------
# Sidebar + main
# ----------------------------------------------------------------------------------------
def sidebar():
    st.sidebar.title("🤖 Foundations Lab")
    st.session_state.setdefault("nav", PAGES[0])
    st.sidebar.radio("Go to", PAGES, key="nav")
    st.sidebar.divider()
    st.sidebar.radio("Mode", ["Demo (no API key)", "Live (Gemini)"], key="mode")
    if st.session_state.get("mode", "").startswith("Live"):
        st.sidebar.text_input("Gemini API key", type="password", key="api_key", value=_default_key(),
                              help="Kept in this browser session only. Never commit keys: use .streamlit/secrets.toml or an env var.")
        with st.sidebar.expander("Model settings"):
            st.text_input("Chat model", fe.MODEL_DEFAULT, key="chat_model")
            st.checkbox("Free-tier rate limit (4 calls/min)", True, key="rate_limit")
    st.session_state._ph = st.sidebar.empty()
    b = get_backend()
    if st.session_state.get("mode", "").startswith("Live"):
        if b.live:
            st.sidebar.success("Live: connected to Gemini")
            st.sidebar.caption("Gemini calls this session: %d  |  last call: %.1f s" % (b.calls, b.last_seconds))
        else:
            st.sidebar.warning(b.error or "Paste a key to go live. Demo mode meanwhile.")
    else:
        st.sidebar.info("Demo mode: scripted model, pretend data.")


def main():
    sidebar()
    page = st.session_state.get("nav", PAGES[0])
    {PAGES[0]: page_home, PAGES[1]: page_limit, PAGES[2]: page_agent, PAGES[3]: page_evolution, PAGES[4]: page_tools,
     PAGES[5]: page_why, PAGES[6]: page_patterns, PAGES[7]: page_choose, PAGES[8]: page_capstone, PAGES[9]: page_review}[page]()


main()
