"""
Architecture Lab: an interactive companion to "Agents, Session 4: Agent Architectures and Design Patterns".

Run:   streamlit run app.py
Works with NO API key (Demo mode: simulated labs with pretend data). Add a Gemini key for Live mode on the live-capable labs.
"""
import os
import time

import pandas as pd
import streamlit as st

import arch_engine as ae

st.set_page_config(page_title="Architecture Lab: agent architectures & patterns", page_icon="🏗️", layout="wide")

PAGES = [
    "🏠 Start here",
    "1 · Why architecture matters",
    "2 · Deliberative: plan, then check",
    "3 · Reactive: reflexes & subsumption",
    "4 · BDI: commitment",
    "5 · Layered: the orchestrator",
    "6 · CoALA: memory, actions, decisions",
    "7 · Choosing an architecture",
    "8 · Planning vs ReAct",
    "9 · Tool use & Reflection",
    "10 · Capstone: all patterns together",
    "11 · Review, quiz & cheat sheet",
]


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


def get_backend() -> ae.Backend:
    mode = st.session_state.get("mode", "Demo (no API key)")
    key = st.session_state.get("api_key", "") if mode.startswith("Live") else ""
    cfg = (key, st.session_state.get("chat_model", "gemini-3.5-flash-lite"), st.session_state.get("rate_limit", True))
    if st.session_state.get("_cfg") != cfg or "backend" not in st.session_state:
        st.session_state.backend = ae.Backend(*cfg)
        st.session_state._cfg = cfg
    return st.session_state.backend


def goto(page: str):
    st.session_state.nav = page


def badge(b: ae.Backend):
    if b.last_source == "gemini":
        st.caption("🟢 Written by Gemini")
    elif "fallback" in b.last_source:
        st.caption("🟠 " + (b.error or "Live call failed; showing the demo simulation."))
    else:
        st.caption("🔵 Demo simulation (pretend data, no API key needed)")


def plain(text: str):
    st.info("**In plain words:** " + text)


def story(text: str):
    st.warning("**🎫 " + text + "**")


def df_show(rows, **kw):
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, **kw)


def show_code(title: str, code: str):
    with st.expander("🔍 " + title):
        st.code(code.strip("\n"), language="python")


def dot(src: str):
    st.graphviz_chart(src)


NODE = 'node [shape=box, style="rounded,filled", fillcolor="#eef2ff", fontname="Helvetica"];'


# ----------------------------------------------------------------------------------------
# Page 0: Start
# ----------------------------------------------------------------------------------------
def page_home():
    st.title("🏗️ Architecture Lab: how agents think and act")
    st.caption("An interactive companion to *Agents, Session 4: Agent Architectures and Design Patterns*.")
    story("The night the agent thought too hard. DilSeTrips now serves 2,000 couples. At 7:40 pm one wrote: "
          "\"Help, our car broke down on the road to Rohtang and it's getting dark.\" The agent retrieved memories, checked weather, "
          "searched the guide and, forty seconds later, produced a beautiful answer. The emergency number was in paragraph four.")
    st.success("**The question to hold on to:** how much thinking does this moment need, and who decides? "
               "Every architecture and pattern in this session is a way of choosing a point on that dial, or of switching between points.")
    dot('digraph { rankdir=LR; ' + NODE + ' Reflex [fillcolor="#ffe3e3"]; Routine [fillcolor="#fff3bf"]; Deliberation [fillcolor="#d3f9d8"]; '
        '"No thinking\\n(fast, fixed, safe)" -> Reflex -> Routine -> Deliberation -> "Full reasoning\\n(slow, flexible, costly)"; }')

    st.subheader("Two ideas, then five architectures, then four patterns")
    c1, c2 = st.columns(2)
    c1.markdown("**Architecture** = the blueprint of the whole house (the city plan).\n\n"
                "Deliberative, Reactive, BDI, Layered, CoALA.")
    c2.markdown("**Design pattern** = a proven building technique inside it (traffic intersections, parks).\n\n"
                "Planning, ReAct, Tool use, Reflection.")

    st.subheader("Your path")
    path = [
        ("1 · Why architecture matters", PAGES[1], "Scalability, efficiency, adaptability, and blueprint vs technique."),
        ("2 · Deliberative", PAGES[2], "Build a plan; code checks its preconditions."),
        ("3 · Reactive", PAGES[3], "Fix Saturday night with reflexes; see where keywords fail."),
        ("4 · BDI", PAGES[4], "Commit to an intention; small events patch, big events reconsider."),
        ("5 · Layered", PAGES[5], "An orchestrator routes each message to the right amount of thinking."),
        ("6 · CoALA", PAGES[6], "Propose, evaluate, select, with weights you can move."),
        ("7 · Choosing", PAGES[7], "Which architecture fits your problem?"),
        ("8 · Planning vs ReAct", PAGES[8], "Break a tool and watch which pattern survives."),
        ("9 · Tool use & Reflection", PAGES[9], "Selection, parameter hallucination, critic loops."),
        ("10 · Capstone", PAGES[10], "All patterns in one agent, including the reference-code bug."),
        ("11 · Review", PAGES[11], "Quiz, homework map, cheat sheet."),
    ]
    for title, page, desc in path:
        a, b, c = st.columns([3, 6, 1.4])
        a.markdown("**" + title + "**")
        b.write(desc)
        c.button("Open", key="home_" + page, on_click=goto, args=(page,))
    st.markdown("**Modes:** *Demo* (default) needs no key and uses pretend data. *Live* (sidebar) calls Gemini on the labs marked 🔴 live-capable "
                "(plan proposal, BDI, CoALA, reflection).")


# ----------------------------------------------------------------------------------------
# Page 1: Why architecture matters
# ----------------------------------------------------------------------------------------
def page_why():
    st.header("1 · Why agent architectures matter: the blueprint")
    st.markdown("An **agent architecture** is the fundamental structure of an agent: its components, how they connect and how information flows "
                "as it **perceives, reasons and acts**. It separates a toy from a tool in three ways.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 🏙 Scalability")
        st.markdown("A town with no plan gridlocks when thousands arrive; a grid grows into a metropolis. **Amazon** moved to microservices so "
                    "payments can scale on Black Friday without taking the site down.")
        st.caption("Agent: add skills, data and users without redesign.")
    with c2:
        st.markdown("### 👩‍🍳 Efficiency")
        st.markdown("A chef with *mise en place* cooks without wasted movement. **Google Maps** returns a route in seconds because its design fits "
                    "real-time data.")
        st.caption("Agent: do not spend 40 seconds of model calls on a message that needs none.")
    with c3:
        st.markdown("### 🧱 Adaptability")
        st.markdown("A glued model car cannot change; Lego can become a truck. A **news feed** adapts as your interests shift.")
        st.caption("Agent: modular parts you can swap are the base for learning agents.")

    st.subheader("🧪 Lab: sort these into architecture or design pattern")
    items = {"BDI": "Architecture", "Subsumption layers": "Architecture", "CoALA": "Architecture", "Layered (hybrid)": "Architecture",
             "Planning": "Design pattern", "ReAct": "Design pattern", "Tool use": "Design pattern", "Reflection": "Design pattern",
             "Singleton (software)": "Design pattern", "Microservices for a whole platform": "Architecture"}
    answers = {}
    cols = st.columns(2)
    for i, (item, correct) in enumerate(items.items()):
        answers[item] = cols[i % 2].radio(item, ["Architecture", "Design pattern"], index=None, horizontal=True, key="sort_" + item)
    if st.button("Check my sorting", type="primary", key="sort_check"):
        score = sum(1 for k, v in answers.items() if v == items[k])
        st.metric("Correct", str(score) + " / " + str(len(items)))
        wrong = [k for k, v in answers.items() if v is not None and v != items[k]]
        if wrong:
            st.error("Revisit: " + ", ".join(wrong))
        elif score == len(items):
            st.success("Perfect. Architectures are the whole-system plan; patterns are reusable techniques that cut across them.")
    df_show([
        {"": "Scope", "Architecture": "The whole system: components, relationships, principles", "Design pattern": "One recurring problem inside it"},
        {"": "House analogy", "Architecture": "Blueprint: colonial or modern", "Design pattern": "Kitchen work triangle, cross-ventilation"},
        {"": "Agent examples", "Architecture": "Subsumption, BDI, Layered, CoALA", "Design pattern": "Planning, ReAct, Tool use, Reflection"},
        {"": "Relationship", "Architecture": "Chosen first", "Design pattern": "Cuts across architectures"},
    ])
    st.markdown("**Why patterns matter more for LLM agents:** they bring *structure* (less randomness), *standardisation* (readable reasoning steps) "
                "and *modularity* (combine Planning + Tool use + Reflection).")


# ----------------------------------------------------------------------------------------
# Page 2: Deliberative
# ----------------------------------------------------------------------------------------
def page_deliberative():
    b = get_backend()
    st.header("2.1 · Deliberative architecture: the chess grandmaster")
    st.markdown("**Sense → Think → Act.** The grandmaster observes the board, thinks several moves ahead, then moves. An LLM deliberative agent "
                "does the same with the LLM as the brain.")
    df_show([
        {"Component": "World model", "Chess grandmaster": "Mental image of the board", "LLM agent": "Prompt, conversation history, reasoning scratchpad, external memory"},
        {"Component": "Goal", "Chess grandmaster": "Checkmate", "LLM agent": "Fulfil the user's request"},
        {"Component": "Actions", "Chess grandmaster": "Legal moves (Knight to F3)", "LLM agent": "Tools and APIs (search_web, write_file)"},
        {"Component": "Planner", "Chess grandmaster": "Evaluating move sequences", "LLM agent": "The LLM writing a step-by-step plan"},
    ])
    st.subheader("🧪 Lab: let the LLM propose the plan, let code check it")
    plain("each action has **preconditions** (what must be known first) and **effects** (what it adds). You assemble a plan; plain code "
          "checks every step. Try booking before asking the user.")
    st.markdown("**Actions and their rules**")
    df_show([{"action": a, "needs": ", ".join(r["needs"]) or "-", "gives": ", ".join(r["gives"])} for a, r in ae.ACTION_RULES.items()])
    goal = "Book a hotel in Manali for our trip and send us an itinerary."
    known = ["destination"]
    st.markdown("**Goal:** " + goal + "   **Known at start:** `destination`")
    plan = st.multiselect("Build the plan (steps run in the order you pick them)", list(ae.ACTION_RULES),
                          default=["book_hotel", "search_hotels", "write_itinerary", "send_summary"], key="plan_pick")
    if plan:
        rows, valid = ae.validate_plan(plan, known)
        df_show(rows)
        reached = ae.goal_reached(plan, known, ["booking", "summary sent"])
        if valid and reached:
            st.success("✅ Plan is valid and reaches the goal.")
        elif valid:
            st.warning("Plan is valid but does not reach the goal (needs a booking and a sent summary).")
        else:
            st.error("❌ Plan is INVALID: a step needs something no earlier step provides. Caught before anything ran.")
    if st.button("🤖 Let the agent propose a plan", key="plan_ai"):
        out = b.structured(ae.PLAN_SCHEMA,
                           "Goal: %s\nYou already know: %s\nAvailable actions:\n%s\nWrite the shortest plan that reaches the goal. Do not carry anything out."
                           % (goal, known, "\n".join("- %s: needs %s, gives %s" % (a, r["needs"], r["gives"]) for a, r in ae.ACTION_RULES.items())),
                           lambda: {"steps": [{"action": a, "why": "demo plan"} for a in ae.DEMO_PLAN]})
        steps = [s["action"] for s in out["steps"] if s["action"] in ae.ACTION_RULES]
        rows, valid = ae.validate_plan(steps, known)
        st.session_state.ai_plan = (rows, valid)
        st.session_state.ai_src = b.last_source
    if "ai_plan" in st.session_state:
        rows, valid = st.session_state.ai_plan
        df_show(rows)
        (st.success if valid else st.error)("Agent's plan valid: " + str(valid))
        badge(b)
    st.caption("This is the combination CoALA recommends later: **the LLM for flexible ideas, code for strict rules.** 🔴 live-capable.")
    c1, c2 = st.columns(2)
    c1.success("**Strengths:** flexible and general; human-like, explainable reasoning; can recover from surprises by re-planning.")
    c2.error("**Weaknesses:** hallucination (plan on a flawed world model); cost and latency; black-box reasoning; the frame problem in a new form (stale world model).")
    show_code("Code behind it", '''
for step in plan["steps"]:
    missing = [n for n in action_rules[step["action"]]["needs"] if n not in known]   # code checks the LLM's plan
    ...
    known += action_rules[step["action"]]["gives"]
''')


# ----------------------------------------------------------------------------------------
# Page 3: Reactive
# ----------------------------------------------------------------------------------------
def page_reactive():
    st.header("2.2 · Reactive architecture: the hand on the hot stove")
    st.markdown("**Sense → Act.** No planning, no world model. Behaviours are rules: `IF <sensor> THEN <action>`. Intelligent behaviour *emerges* from many "
                "simple rules running together. The cockroach senses a shadow and scurries.")
    story("Saturday night needed a reflex layer in front of the thinking.")
    st.subheader("🧪 Lab 1: a keyword reflex layer")
    plain("fixed rules, no LLM. Edit the emergency words and try messages. Reflexes are fast *because* they do not understand, which is also "
          "why they miss real emergencies and fire on harmless ones.")
    words_text = st.text_input("Emergency words (comma separated)", ", ".join(ae.EMERGENCY_WORDS), key="rx_words")
    words = [w.strip().lower() for w in words_text.split(",") if w.strip()]
    message = st.text_input("Message", ae.SATURDAY_MESSAGE, key="rx_msg")
    t0 = time.perf_counter()
    reply = ae.reactive_layer(message, words)
    ms = (time.perf_counter() - t0) * 1000
    c1, c2 = st.columns(2)
    c1.metric("Reactive layer (measured)", "%.3f ms" % ms)
    c2.metric("An LLM call (typical, illustrative)", "2 to 40 s")
    if reply:
        st.error("🚨 REFLEX FIRED\n\n" + reply)
    else:
        st.info("Passed down to the slow path (an LLM would think about this).")
    st.markdown("**Try the tricky ones**")
    rows = []
    for m in ae.TRICKY_MESSAGES:
        r = ae.reactive_layer(m, words)
        rows.append({"message": m, "reflex": "FIRED" if r else "passed down",
                     "verdict": {0: "MISSED a real emergency", 1: "false alarm", 2: "false alarm"}[ae.TRICKY_MESSAGES.index(m)] if r and ae.TRICKY_MESSAGES.index(m) > 0
                     else ("MISSED a real emergency" if ae.TRICKY_MESSAGES.index(m) == 0 else "correct")})
    df_show(rows)
    st.caption("Homework 1: would you rather miss an emergency or raise a false alarm? (Hint: safety layers usually accept false alarms.)")

    st.subheader("🧪 Lab 2: subsumption, a robot vacuum")
    plain("higher-priority layers suppress (subsume) lower ones. Flip the sensors and see which layer takes the wheels.")
    c1, c2, c3 = st.columns(3)
    bumper = c1.checkbox("Bumper pressed (about to hit a wall)", value=False, key="sub_bump")
    battery = c2.checkbox("Battery below 10%", value=False, key="sub_batt")
    clear = c3.checkbox("Path is clear", value=True, key="sub_clear")
    res = ae.subsumption(bumper, battery, clear)
    df_show(res["layers"])
    st.success("Robot does: " + ", ".join(res["actions"]))
    st.caption("If the Clean layer fails, the robot still wanders and avoids walls: layering makes it robust.")
    c1, c2 = st.columns(2)
    c1.success("**Strengths:** extremely fast; robust; cheap; simple rules give rich behaviour.")
    c2.error("**Weaknesses:** no long-term planning; no world model (cleans the same spot again); hard to scale to hundreds of rules.")
    st.subheader("Deliberative vs reactive")
    df_show([
        {"Aspect": "Speed", "Deliberative": "Slower: plans and reasons", "Reactive": "Very fast: stimulus to action"},
        {"Aspect": "Best for", "Deliberative": "Complex tasks needing foresight", "Reactive": "Dynamic, uncertain, safety-critical"},
        {"Aspect": "Example", "Deliberative": "Delivery route with traffic optimisation", "Reactive": "Car braking for a pedestrian"},
        {"Aspect": "Weakness", "Deliberative": "Paralysis by analysis", "Reactive": "No foresight"},
        {"Aspect": "Analogy", "Deliberative": "Grandmaster 10 moves ahead", "Reactive": "Hand away from fire"},
    ])


# ----------------------------------------------------------------------------------------
# Page 4: BDI
# ----------------------------------------------------------------------------------------
def page_bdi():
    b = get_backend()
    st.header("2.3 · BDI: Belief, Desire, Intention")
    st.markdown("A structured \"mind\" that guides action without a fresh plan for every step.")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Beliefs** (informational)\n\nWhat the agent thinks is true. Can be incomplete or wrong. `at(robot, room_A)`")
    c2.markdown("**Desires** (motivational)\n\nEverything it would like. May conflict. `rescue_victim`, `recharge_battery`")
    c3.markdown("**Intentions** (deliberative)\n\nThe desires it has **committed** to, with a plan. The sweet spot: stability.")
    dot('digraph { rankdir=LR; ' + NODE + ' Beliefs; Desires; Intention [fillcolor="#d3f9d8"]; Plan; "Execute step" [fillcolor="#fff3bf"]; '
        'Beliefs -> Intention; Desires -> Intention [label="deliberate"]; Intention -> Plan -> "Execute step"; '
        '"Execute step" -> Beliefs [label="sense", style=dashed]; "Execute step" -> Plan [label="small change: patch", style=dotted]; '
        '"Execute step" -> Intention [label="big change: reconsider", style=dotted, color="#d9480f"]; }')
    st.subheader("🧪 Lab: small events patch a step, big events reconsider the intention")
    plain("the agent forms beliefs, desires and ONE committed intention for Prasad's Manali trip. Then events arrive. Commitment means it does "
          "not scrap everything for every new fact.")
    context = st.text_area("Trip context", ae.TRIP_CONTEXT, height=110, key="bdi_ctx")
    if st.button("🧠 Form beliefs, desires and an intention", type="primary", key="bdi_form"):
        st.session_state.bdi = b.structured(ae.BDI_SCHEMA, "Form beliefs, desires, one committed intention and a short plan for a travel "
                                            "companion agent supporting this trip:\n" + context, ae.demo_bdi)
        st.session_state.bdi_events = []
    if "bdi" in st.session_state:
        bdi = st.session_state.bdi
        c1, c2, c3 = st.columns(3)
        c1.markdown("**Beliefs**")
        for x in bdi["beliefs"]:
            c1.markdown("- " + x)
        c2.markdown("**Desires**")
        for x in bdi["desires"]:
            c2.markdown("- " + x)
        c3.markdown("**Intention**")
        c3.success(bdi["intention"])
        c3.markdown("**Plan**")
        for x in bdi["plan"]:
            c3.markdown("1. " + x)
        badge(b)
        event = st.selectbox("An event arrives", ae.BDI_EVENTS + ["(write your own below)"], key="bdi_event")
        if event.startswith("(write"):
            event = st.text_input("Your event", "The cottage offers a free room upgrade.", key="bdi_custom")
        if st.button("⚡ Apply this event", key="bdi_apply"):
            res = b.structured(ae.RECONSIDER_SCHEMA,
                               "Current beliefs: %s\nDesires: %s\nCommitted intention: %s\nPlan: %s\nNew event: %s\n"
                               "Is this a small change (patch a step) or a big one (reconsider the intention)?"
                               % (bdi["beliefs"], bdi["desires"], bdi["intention"], bdi["plan"], event),
                               lambda: ae.demo_reconsider(bdi, event))
            st.session_state.bdi_result = (event, res)
        if "bdi_result" in st.session_state:
            ev, res = st.session_state.bdi_result
            st.markdown("**Event:** " + ev)
            (st.warning if res["change_size"].startswith("big") else st.success)("**" + res["change_size"] + "**: " + res["reason"])
            st.markdown("**Intention now:** " + res["intention"])
            for x in res["plan"]:
                st.markdown("- " + x)
            badge(b)
    st.caption("🔴 live-capable. How BDI blends the two: deliberation (desire to intention) is the planner; execution is reactive and handles local surprises.")
    st.subheader("Classical BDI vs LLM-based BDI")
    df_show([
        {"Feature": "Knowledge", "Classical": "Explicit symbolic rules and facts", "LLM-based": "Weights, current context, real-time data, memory"},
        {"Feature": "Reasoning", "Classical": "Rule-based deduction", "LLM-based": "Natural-language reasoning"},
        {"Feature": "Planning", "Classical": "Fixed plan library", "LLM-based": "Plans generated on the fly"},
        {"Feature": "Flexibility", "Classical": "Limited to programmed rules", "LLM-based": "Generalises to new tasks"},
    ])
    st.markdown("**Limits of BDI:** beliefs, desires and intentions blur in messy real-world cases, and it is inefficient when you need fast reflexes *and* long-term strategy at once. That gap leads to layered architectures.")


# ----------------------------------------------------------------------------------------
# Page 5: Layered
# ----------------------------------------------------------------------------------------
def page_layered():
    st.header("2.4 · Layered (hybrid) architectures: the orchestrator")
    st.markdown("Split control into layers, each at a different level of abstraction. In an LLM agent the layers are **modes of operation**, chosen by an "
                "**orchestration engine**: the code wrapped around the LLM.")
    df_show([
        {"Layer": "Reactive (instinct)", "Self-driving car": "Brake instantly when an object appears", "LLM agent": "Keyword rule, canned reply, no LLM"},
        {"Layer": "Sequencer (habit)", "Self-driving car": "Lane change: mirrors, signal, turn", "LLM agent": "A fixed script: get weather, then write a packing list"},
        {"Layer": "Deliberative (grandmaster)", "Self-driving car": "Plan the whole route", "LLM agent": "Full reasoning, tools, decompose a vacation"},
    ])
    dot('digraph { rankdir=TB; ' + NODE + ' Message; "Reactive layer\\nkeywords, no LLM" [fillcolor="#ffe3e3"]; Router [fillcolor="#fff3bf"]; '
        '"Sequencer\\nfixed routine" [fillcolor="#fff3bf"]; "Deliberative\\nplan + ReAct + tools" [fillcolor="#d3f9d8"]; '
        'Message -> "Reactive layer\\nkeywords, no LLM"; "Reactive layer\\nkeywords, no LLM" -> Reply [label="fired", color="#d9480f"]; '
        '"Reactive layer\\nkeywords, no LLM" -> Router [label="passed down"]; Router -> "Sequencer\\nfixed routine" [label="known routine"]; '
        'Router -> "Deliberative\\nplan + ReAct + tools" [label="open-ended"]; "Sequencer\\nfixed routine" -> Reply; "Deliberative\\nplan + ReAct + tools" -> Reply; }')
    st.markdown("**Bottom-up activation:** an urgent observation bypasses higher layers for an immediate reply. **Top-down delegation:** the deliberative layer sets the goal and hands steps down.")
    st.subheader("🧪 Lab: DilSeTrips orchestrator")
    plain("send a message and watch which layer answers, how many LLM calls it costs and (illustrative) how long it takes. Seconds are illustrative, to show relative cost.")
    presets = {"Saturday night (emergency)": ae.SATURDAY_MESSAGE, "Packing list": "What should we pack for Manali this week?",
               "Open-ended planning": "Plan a relaxed 2-day food trail in Amritsar for a vegetarian couple.",
               "Emergency without a keyword": "Our car died near Rohtang pass and there's no signal, what now?", "Custom": ""}
    choice = st.selectbox("Message", list(presets), key="orch_choice")
    message = st.text_input("Message text", presets[choice] or "Can you help us find a quiet cafe in Old Manali?", key="orch_msg_" + choice)
    safety = st.checkbox("Homework 2: add a cheap LLM safety check to the router (costs 1 call on every message that passes the keywords)", value=False, key="orch_safe")
    res = ae.orchestrate(message, safety_check=safety)
    c1, c2, c3 = st.columns(3)
    c1.metric("Answered by layer", res["layer"])
    c2.metric("LLM calls", res["calls"])
    c3.metric("Seconds (illustrative)", "%.3f" % res["seconds"])
    df_show(res["trace"])
    st.success(res["reply"]) if res["layer"].startswith("reactive") else st.info(res["reply"])
    if choice.startswith("Emergency without") and not safety:
        st.error("The emergency slipped past the keyword layer and went to full deliberation. Tick the safety check to catch it, and see the extra latency every other message pays.")
    show_code("Code behind it", '''
def orchestrator(message):
    reflex = reactive_layer(message)                 # bottom layer, checked first, no LLM
    if reflex is not None:
        return "reactive", reflex
    route = router_llm.invoke("Route this message to a DilSeTrips layer:\\n" + message)
    if route["layer"].startswith("routine") and route["destination"] != "":
        return "sequencer", packing_list_routine(message, route["destination"])
    return "deliberative", planning_agent.invoke(...)
''')


# ----------------------------------------------------------------------------------------
# Page 6: CoALA
# ----------------------------------------------------------------------------------------
def page_coala():
    b = get_backend()
    st.header("2.5 · CoALA: Cognitive Architectures for Language Agents")
    st.markdown("\"If a powerful LLM sits at the heart of an agent, how do we build the memory, actions and decision-making around it?\" CoALA is an architect's blueprint for an AI's mind, with three pillars.")
    dot('digraph { rankdir=LR; ' + NODE + ' User; LLM [fillcolor="#d3f9d8"]; "Working memory" [fillcolor="#d0ebff"]; Semantic; Episodic; Procedural; '
        '"Decision cycle\\npropose, evaluate, select" [fillcolor="#fff3bf"]; "External actions\\n(tools, grounding)" [fillcolor="#ffe3e3"]; Outcome; Human [shape=ellipse]; '
        'User -> LLM; LLM -> "Working memory"; Semantic -> "Working memory" [label=retrieval]; Episodic -> "Working memory" [label=retrieval]; '
        'Procedural -> "Working memory" [label=retrieval]; "Working memory" -> "Decision cycle\\npropose, evaluate, select"; '
        '"Decision cycle\\npropose, evaluate, select" -> "External actions\\n(tools, grounding)" -> Outcome; Outcome -> Human [style=dashed, label=review]; '
        '"Working memory" -> Episodic [label=learning, style=dotted]; }')
    c1, c2 = st.columns(2)
    c1.markdown("**External actions (grounding):** physical environments, dialogue with humans, digital tools and APIs.")
    c2.markdown("**Internal actions:** *retrieval* (read long-term memory into working memory), *reasoning* (think in working memory), *learning* (write to long-term memory).")
    st.subheader("🧪 Lab 1: walk a ShopFast ticket through CoALA")
    step = st.slider("Step", 1, len(ae.SHOPFAST_STEPS), 1, key="shop_step")
    for i, (stage, text, _) in enumerate(ae.SHOPFAST_STEPS[:step]):
        (st.success if i == step - 1 else st.info)("**" + stage + ":** " + text)
    st.subheader("🧪 Lab 2: propose, evaluate, select")
    plain("the LLM proposes and judges; visible code selects. Move the weights: when safety counts most, the risky candidate never wins.")
    st.markdown("**Situation:** " + ae.SITUATION)
    if "coala_scores" not in st.session_state:
        st.session_state.coala_scores = ae.DEMO_CANDIDATES
    if st.button("🤖 Propose and evaluate candidates", key="coala_go"):
        props = b.structured(ae.PROPOSE_SCHEMA, "Propose 3 different candidate next actions for a travel agent in this situation:\n" + ae.SITUATION,
                             lambda: {"candidates": [{"action": c["action"], "action_type": c["action_type"]} for c in ae.DEMO_CANDIDATES]})["candidates"]
        text = "".join("- " + c["action"] + "\n" for c in props)
        scores = b.structured(ae.EVALUATE_SCHEMA, "Situation: %s\nScore each candidate from 1 to 10 for safety, usefulness and cost (10 = cheapest). Copy each action exactly.\n%s"
                              % (ae.SITUATION, text), lambda: {"scores": [{k: c[k] for k in ("action", "safety", "usefulness", "cost")} for c in ae.DEMO_CANDIDATES]})["scores"]
        types = {c["action"]: c["action_type"] for c in props}
        st.session_state.coala_scores = [{**s, "action_type": types.get(s["action"], "")} for s in scores]
        badge(b)
    c1, c2, c3 = st.columns(3)
    ws = c1.slider("Weight: safety", 0.0, 1.0, 0.5, 0.05, key="w_safe")
    wu = c2.slider("Weight: usefulness", 0.0, 1.0, 0.3, 0.05, key="w_use")
    wc = c3.slider("Weight: cost (10 = cheapest)", 0.0, 1.0, 0.2, 0.05, key="w_cost")
    rows, best = ae.coala_select(st.session_state.coala_scores, (ws, wu, wc))
    df_show([{"candidate": r["action"], "type": r.get("action_type", ""), "safety": r["safety"], "usefulness": r["usefulness"], "cost": r["cost"],
              "working": r["working"], "total": r["total"]} for r in rows])
    st.bar_chart(pd.DataFrame({"total": [r["total"] for r in rows]}, index=[r["action"][:40] for r in rows]))
    st.success("SELECTED: " + best)
    st.caption("🔴 live-capable. Like the utility agent in Session 2, the LLM judges and visible code decides. Note the mix of action types in the proposals.")
    st.markdown("**Reasoning loop vs decision process:** the loop in working memory is local *thinking* (what could I do?); the overall decision process is global *doing* (what I will actually do).")
    st.markdown("**CoALA habits:** build with Lego (swappable memory/action modules), combine brains and brawn (LLM for ideas, code for rules), create true learners (write to memory).")


# ----------------------------------------------------------------------------------------
# Page 7: Choosing
# ----------------------------------------------------------------------------------------
def page_choose():
    st.header("2.6 · How the architectures relate, and when to choose each")
    dot('digraph { rankdir=LR; ' + NODE + ' "Deliberative\\nstrong planning,\\nweak real-time" -> "Reactive\\nfast,\\nno foresight" -> "BDI\\nstructured mental states,\\nrigid when unpredictable" '
        '-> "Layered\\nexplicit coordination" -> "CoALA\\nLLM-native: memory,\\nreasoning, tools"; }')
    st.caption("Each step on the ladder fixes a practical gap in the one before.")
    st.subheader("🧪 Lab: which architecture fits your problem?")
    plain("rate how much each need matters (0 to 5). The score is a heuristic built from the session's comparison table, not a law.")
    presets = {
        "Custom": None,
        "Warehouse robot arm (fixed steps)": {"speed": 3, "foresight": 5, "adaptability": 0, "predictability": 5, "open-endedness": 0, "simplicity": 3},
        "Obstacle-avoiding drone": {"speed": 5, "foresight": 1, "adaptability": 2, "predictability": 5, "open-endedness": 0, "simplicity": 4},
        "Customer support AI": {"speed": 4, "foresight": 3, "adaptability": 4, "predictability": 4, "open-endedness": 3, "simplicity": 1},
        "Research assistant (open-ended)": {"speed": 1, "foresight": 4, "adaptability": 5, "predictability": 2, "open-endedness": 5, "simplicity": 1},
    }
    pick = st.selectbox("Start from a scenario", list(presets), key="ch_pick")
    base = presets[pick] or {"speed": 3, "foresight": 3, "adaptability": 3, "predictability": 3, "open-endedness": 3, "simplicity": 3}
    cols = st.columns(3)
    imp = {}
    for i, k in enumerate(base):
        imp[k] = cols[i % 3].slider(k.capitalize() + " matters", 0, 5, base[k], key="ch_%s_%s" % (pick, k))
    rows = ae.choose_architecture(imp)
    st.bar_chart(pd.DataFrame({"fit %": [r["fit %"] for r in rows]}, index=[r["architecture"] for r in rows]))
    st.success("Best fit for this profile: **" + rows[0]["architecture"] + "**")
    st.caption("Real systems mix them: DilSeTrips is a **layered** system whose deliberative layer is **CoALA-shaped** and whose trip companion behaves like **BDI**.")
    st.subheader("Comparison table from the session")
    df_show([
        {"Architecture": "Deliberative", "Best for": "Structured, clear rules (chess, scheduling)", "Complexity": "High", "Performance": "Low to medium", "Adaptability": "Low"},
        {"Architecture": "Reactive", "Best for": "Real-time, safety-critical", "Complexity": "Low", "Performance": "High", "Adaptability": "Low"},
        {"Architecture": "BDI", "Best for": "Goal-driven yet flexible (rescue robots)", "Complexity": "Medium to high", "Performance": "Medium", "Adaptability": "Medium"},
        {"Architecture": "Layered", "Best for": "Reflexes and strategy (self-driving, support AI)", "Complexity": "High", "Performance": "High", "Adaptability": "High"},
        {"Architecture": "CoALA", "Best for": "Open-ended tasks with reasoning, memory, tools", "Complexity": "Variable", "Performance": "Medium to high", "Adaptability": "Very high"},
    ])


# ----------------------------------------------------------------------------------------
# Page 8: Planning vs ReAct
# ----------------------------------------------------------------------------------------
def page_plan_react():
    st.header("3.1 to 3.4 · Planning vs ReAct")
    c1, c2 = st.columns(2)
    c1.markdown("### 📝 Planning\nWrite the complete step-by-step plan *before* acting (a recipe before cooking). Components: world model, goals, actions with preconditions and effects, planner. "
                "**Strength:** structured, predictable. **Weakness:** rigid and brittle.")
    c2.markdown("### 🔄 ReAct\nInterleave **Thought → Action → Observation** in a loop. The Thought decomposes and self-corrects; the Action is a tool call the orchestrator runs; the Observation feeds the next thought. "
                "**Strength:** adaptive, transparent. **Weakness:** more calls, prompt-sensitive.")
    st.subheader("🧪 Lab 1: break a tool and watch both patterns")
    plain("same task, same tools, with pretend data. Switch the hotel search off (it returns an HTTP 503) and compare how each pattern copes.")
    task = st.text_input("Task", "Find somewhere quiet to stay in Old Manali this week and tell us what weather to expect.", key="pr_task")
    c1, c2 = st.columns(2)
    hotels_up = c1.checkbox("Hotel search is UP", value=False, key="pr_hot")
    homes_up = c2.checkbox("Homestay search is UP", value=True, key="pr_home")
    place = ae.find_place(task) or "Old Manali"
    plan_res = ae.planning_simulate(ae.needs_of(task), place, hotels_up, homes_up)
    react_steps = ae.react_simulate(task, hotels_up, homes_up)
    left, right = st.columns(2)
    with left:
        st.markdown("#### Planning (fixed plan, executed strictly)")
        st.code("PLAN: " + str(plan_res["plan"]), language="text")
        df_show(plan_res["log"])
        if plan_res["broke"]:
            st.error("PLAN BROKEN: a step failed, every later step is invalid. The plan has no way to adapt; a full replanning cycle would be needed.")
        else:
            st.success("Plan finished the task.")
        st.caption("LLM calls: 1 to write the plan, 1 for the answer.")
    with right:
        st.markdown("#### ReAct (one step at a time)")
        for i, s in enumerate(react_steps):
            st.markdown("**Cycle %d**" % (i + 1))
            st.markdown("💭 *Thought:* " + s["thought"])
            if "action" in s:
                st.markdown("🛠 *Action:* `" + s["action"] + "`")
                (st.error if s["observation"].startswith("ERROR") else st.info)("👁 *Observation:* " + s["observation"])
            else:
                st.success("✅ *Final answer:* " + s["final"])
        st.caption("LLM calls: one per cycle (%d here). Cost of adaptability." % len(react_steps))
    df_show([
        {"": "Core idea", "Planning": "Think first, then act", "ReAct": "Think, act, think again"},
        {"": "Execution", "Planning": "Fixed sequence; one failure invalidates it", "ReAct": "Dynamic loop; plan grows one step at a time"},
        {"": "Adaptability", "Planning": "Brittle; replanning is slow", "ReAct": "Robust to surprises"},
        {"": "Efficiency", "Planning": "Expensive up front", "ReAct": "Progress with every step"},
        {"": "Implementation", "Planning": "A*, STRIPS on a symbolic model", "ReAct": "LLM with ReAct prompt or native tool calls"},
        {"": "Output", "Planning": "A formal, verified plan", "ReAct": "Thought/Action/Observation log"},
    ])
    st.markdown("**Use Planning when** the environment is static, optimality matters (truck-fleet routes) and the problem is closed (Rubik's cube). "
                "**Use ReAct when** the environment is dynamic (live order database), the problem is open-ended (latest fusion research) or robustness beats optimality.")

    st.subheader("🧪 Lab 2: ReAct written by hand, and why text parsing is fragile")
    plain("the original ReAct paper has the model write `Action: tool[input]` as text and our code parses it. Type an action line the way a model might.")
    line = st.text_input("Model output line", "Action: get_weather_forecast[Shimla]", key="pr_parse")
    tool, arg = ae.parse_action(line)
    if tool:
        st.success("Parsed OK: tool = `%s`, input = `%s`" % (tool, arg))
        st.info("Observation: " + ae.run_tool(tool, arg))
    else:
        st.error("parse_action FAILED. Square brackets are required; `get_weather_forecast(Shimla)` with round brackets breaks the loop. "
                 "Native tool calling (structured JSON, as in create_agent) removes this fragility.")
    q = st.text_input("Question for a full ReAct run", "Is this a good week to visit Shimla, and what is one thing the travel guide says not to miss there?", key="pr_q")
    steps = ae.react_simulate(q)
    for i, s in enumerate(steps):
        if "action" in s:
            st.code("Thought: %s\nAction: %s\nObservation: %s" % (s["thought"], s["action"], s["observation"]), language="text")
        else:
            st.code("Thought: %s\nFinal Answer: %s" % (s["thought"], s["final"]), language="text")

    st.subheader("ReAct agents vs Reactive agents (the names are confusing)")
    df_show([
        {"Feature": "Core principle", "Reactive agent": "Reflexive: sense → act", "ReAct agent": "Iterative: thought → action → observation"},
        {"Feature": "Reasoning", "Reactive agent": "None", "ReAct agent": "High, through the LLM"},
        {"Feature": "Memory", "Reactive agent": "None (stateless)", "ReAct agent": "Short-term thought history; optional long-term"},
        {"Feature": "Environment", "Reactive agent": "Static, predictable", "ReAct agent": "Dynamic, open-ended"},
        {"Feature": "Example", "Reactive agent": "Thermostat; DilSeTrips' emergency layer", "ReAct agent": "Research agent; DilSeTrips' planning layer"},
    ])


# ----------------------------------------------------------------------------------------
# Page 9: Tool use and Reflection
# ----------------------------------------------------------------------------------------
def page_tools_reflect():
    b = get_backend()
    st.header("3.5 and 3.6 · Tool use and Reflection")
    t1, t2 = st.tabs(["🛠 Tool use", "🪞 Reflection"])
    with t1:
        st.markdown("Planning and ReAct are *brains in a jar*. **Tool use** gives them hands and eyes: deterministic modules that a non-deterministic LLM can rely on.")
        df_show([{"Kind": "Information retrieval", "Examples": "Web search, SQL query, weather forecast"},
                 {"Kind": "Action execution", "Examples": "Send email, add reminder, book a slot"},
                 {"Kind": "Computation / logic", "Examples": "Calculator, code interpreter, validator"}])
        st.markdown("**Three mechanisms:** *tool selection* (the hardest), *parameter generation* (watch for hallucinated arguments), *output interpretation* (do not miss an `ERROR:` line).")
        st.subheader("🧪 Lab: selection and parameters")
        df_show([{"tool": k, "description": v} for k, v in ae.TOOLS.items()])
        plain("type a request. The simulator picks a tool and fills parameters. Ask 'How much is 5,000 rupees?' and see parameter hallucination; then tick the fix.")
        fix = st.checkbox("Homework 4 fix: tool description tells the model to ask when the target currency is missing", value=False, key="tu_fix")
        req = st.selectbox("Request", ["What's the weather this week in the city where the Taj Mahal is?", "How much is 5,000 rupees?",
                                       "How much is 5,000 rupees in euros?", "Remind us to pack woollens tomorrow", "(your own)"], key="tu_req")
        if req.startswith("(your"):
            req = st.text_input("Your request", "Will it rain in Mumbai?", key="tu_custom")
        res = ae.demo_tool_call(req, fix)
        if res["tool"]:
            st.markdown("**Selected:** `%s`   **Parameters:** `%s`" % (res["tool"], res["params"]))
            if res["tool"] == "convert_currency" and res["params"].get("amount") is not None:
                p = res["params"]
                st.info("Observation: " + ae.convert_currency(p["amount"], p["from_currency"], p["to_currency"]))
            if res["tool"] == "get_weather_forecast" and res["params"]["place"]:
                st.info("Observation: " + ae.WEATHER.get(res["params"]["place"].lower(), "ERROR: no forecast"))
        (st.error if "HALLUCINATION" in res["note"] else st.success)(res["note"])
        st.caption("Tool use is the *Action* in ReAct and the *execution* of each step in Planning. In Planning, tool calls are a consequence of the plan (top-down); in ReAct they are the central mechanism (dynamic, adaptive).")

    with t2:
        st.markdown("**Reflection** is metacognition: the model critiques its own output or log, like an author putting on an editor's hat. It usually runs *after* an attempt, unlike ReAct's inner loop.")
        dot('digraph { rankdir=LR; ' + NODE + ' "1 Execution\\ntry the task, keep a log" -> "2 Evaluation\\ncritic reads the log" -> "3 Refinement\\nlessons / better plan" -> "4 Integration\\nsave to memory" -> "1 Execution\\ntry the task, keep a log" [style=dashed]; }')
        st.subheader("🧪 Lab 1: explain gravity to a five-year-old")
        plain("a deliberately technical first attempt, a critic, then a refined answer. One review step, same model, better answer.")
        if st.button("▶ Run the reflection cycle", type="primary", key="rf_go"):
            first = b.ask(ae.GRAVITY_TASK + " Answer like a physics textbook, in two sentences.", ae.GRAVITY_FIRST)
            crit = b.structured(ae.CRITIC_SCHEMA, "Task: %s\nAnswer: %s\nReview the answer strictly. Is it appropriate for the audience?" % (ae.GRAVITY_TASK, first),
                                ae.GRAVITY_CRITIQUE)
            refined = b.ask("Task: %s\nYour previous answer: %s\nAn editor said: %s\nLessons: %s\nWrite a better answer." % (ae.GRAVITY_TASK, first, crit["critique"], crit["lessons"]),
                            ae.GRAVITY_REFINED)
            st.session_state.reflect = (first, crit, refined)
        if "reflect" in st.session_state:
            first, crit, refined = st.session_state.reflect
            st.error("**1 Execution (first attempt)**\n\n" + first)
            st.warning("**2 Evaluation (critic)**\n\n" + crit["critique"] + "\n\n*Why:* " + crit["reasoning"])
            st.info("**3 Refinement (lessons)**\n\n" + "\n".join("- " + x for x in crit["lessons"]))
            st.success("**Refined answer**\n\n" + refined)
            badge(b)
        st.caption("🔴 live-capable.")
        st.subheader("🧪 Lab 2: reflect on a failed agent log, save the lessons, plan again")
        hot = ae.planning_simulate(["weather", "stay"], "Old Manali", hotels_up=False, homestays_up=True)
        log_text = "TASK: quiet stay + weather in Old Manali\nPLAN: %s\n" % hot["plan"] + "\n".join("STEP %d: %s -> %s" % (r["step"], r["call"], r["result"][:70]) for r in hot["log"]) + "\nPLAN ABANDONED: a step failed."
        st.code(log_text, language="text")
        if "lessons" not in st.session_state:
            st.session_state.lessons = []
        if st.button("🧐 Critique the log and save the lessons", key="rf_log"):
            st.session_state.lessons = list(ae.LOG_LESSONS)
        if st.session_state.lessons:
            st.markdown("**Lessons saved to memory (4 Integration)**")
            for l in st.session_state.lessons:
                st.markdown("- " + l)
            st.markdown("**Second plan, starting from the lessons:**")
            st.code("get_weather_forecast(Old Manali)\nsearch_homestays(Old Manali)   # fallback chosen because of the lesson", language="text")
            st.caption("The planner did not become adaptive like ReAct; it became *wiser* about this kind of task. Homework 7: do hotel lessons help with restaurants?")
        st.markdown("**Reflection as a booster in other architectures:** BDI's re-evaluation phase when an intention fails; the top deliberative layer reviewing lower-layer logs; Reflexion and Self-RAG in CoALA-style agents. "
                    "**Costs:** an extra call per critique, risk of over-analysis, and a critic that is itself an LLM can share the producer's blind spots.")
    st.subheader("The four patterns compared")
    df_show([
        {"Pattern": "Planning", "Core idea": "Plan before acting", "Strength": "Structured, predictable", "Weakness": "Rigid, brittle", "Use when": "Static environments; optimality matters"},
        {"Pattern": "ReAct", "Core idea": "Thought, Action, Observation loop", "Strength": "Adaptive, transparent", "Weakness": "Latency, cost, prompt-sensitive", "Use when": "Open-ended, uncertain problems"},
        {"Pattern": "Tool use", "Core idea": "Extend the LLM with APIs and functions", "Strength": "Fills LLM gaps", "Weakness": "Wrong tool or parameters cause errors", "Use when": "Need outside information or actions"},
        {"Pattern": "Reflection", "Core idea": "Critique your own output, then improve", "Strength": "Quality, learns from mistakes", "Weakness": "Extra calls; over-analysis", "Use when": "Accuracy and reliability matter"},
    ])


# ----------------------------------------------------------------------------------------
# Page 10: Capstone
# ----------------------------------------------------------------------------------------
def page_capstone():
    st.header("4 · Capstone: one agent, several patterns")
    st.markdown("The session's reference agent runs a planning call, one 'ReAct' call, a tool step, a final answer and a reflection call. Read it critically: it hides a bug.")
    st.subheader("🐞 The reference code's bug")
    st.code('''if "get_weather" in react_result.lower():
    # Simple demo extraction
    city = "Gurgaon"
    observation = get_weather.invoke({"city": city})''', language="python")
    st.error("Whatever city the user asked about, this always fetches **Gurgaon**. It is also not reactive (it plans and reflects), its 'ReAct' runs once (no observation fed back), and it passes temperature=0 to a model that ignores it.")
    st.subheader("🧪 Lab: run the buggy agent and the corrected agent")
    plain("the corrected agent has five parts: a **reactive** layer first, **planning**, **ReAct with tool use** (the city comes from the model's own tool call), and **reflection** that checks the answer against the observations.")
    preset = st.selectbox("Message", ["What's the weather in Mumbai this week, and should we carry umbrellas?", ae.SATURDAY_MESSAGE,
                                      "Will it be sunny in Agra this week?"], key="cap_msg")
    left, right = st.columns(2)
    for col, buggy, title in ((left, True, "Reference agent (buggy)"), (right, False, "Corrected agent")):
        with col:
            st.markdown("#### " + title)
            res = ae.capstone(preset, buggy=buggy)
            for t in res["trace"]:
                st.markdown("**" + t["stage"] + "**")
                st.code(t["detail"], language="text")
            if res.get("wrong_city"):
                st.error("FINAL: " + res["final"] + "\n\n⚠ This is another city's weather presented as the answer. Reflection could not catch it.")
            else:
                st.success("FINAL: " + res["final"])
            st.caption("LLM calls: %d" % res["llm_calls"])
    st.markdown("**Lesson:** reflection only checks what it can see. Nothing told the critic which city the data belonged to. Same agent, two very different amounts of thinking, chosen per message: the emergency never reaches an LLM.")
    show_code("Code behind the corrected agent", '''
def dilsetrips_agent(user_message):
    reflex = reactive_layer(user_message)                 # 1. reactive: no LLM
    if reflex is not None: return reflex
    plan = llm.with_structured_output(short_plan_schema).invoke(...)          # 2. planning
    executor = create_agent(model=llm, tools=[get_weather_forecast], system_prompt=plan_text)   # 3+4. ReAct + tools
    result = executor.invoke({"messages": [HumanMessage(content=user_message)]})
    check = llm.with_structured_output(answer_check_schema).invoke(...)       # 5. reflection vs observations
''')


# ----------------------------------------------------------------------------------------
# Page 11: Review
# ----------------------------------------------------------------------------------------
QUIZ = [
    ("Which statement best separates architecture from design pattern?",
     ["Architecture is the whole-system blueprint; a pattern is a reusable technique inside it", "They are the same thing", "Patterns are bigger than architectures", "Architectures only exist for robots"], 0,
     "Architecture is the city plan; patterns are the best designs for intersections and parks within it."),
    ("Why did the keyword reflex miss 'Our car died near Rohtang pass'?",
     ["The LLM was too slow", "Reflexes match fixed words and do not understand meaning", "The router was down", "It was a BDI failure"], 1,
     "Reflexes are fast precisely because they do not understand."),
    ("In a subsumption robot, the bumper fires while the robot is cleaning. What happens?",
     ["Clean layer wins", "Avoid-obstacles layer suppresses lower layers and takes the wheels", "The robot plans a new route", "Nothing"], 1,
     "Higher-priority layers subsume lower ones."),
    ("In BDI, an event changes a hotel check-in time by two hours. What should the agent do?",
     ["Reconsider the whole intention", "Patch one step of the plan", "Delete its beliefs", "Ask an LLM to start over"], 1,
     "Commitment: small changes patch a step; only major changes reconsider the intention."),
    ("What does the orchestrator in a layered LLM agent do?",
     ["Trains the model", "Decides which mode (reflex, routine, deliberation) handles each message", "Stores memories", "Runs reflection only"], 1,
     "It is code around the LLM that chooses how much thinking a moment gets."),
    ("In CoALA's decision cycle, who selects the final action in the notebook's example?",
     ["The LLM alone", "Visible code applying weights to the LLM's scores", "A human always", "The tool"], 1,
     "The LLM proposes and judges; plain arithmetic selects."),
    ("The hotel search is down. Which pattern copes best without extra machinery?",
     ["Strict planning", "ReAct", "A reactive keyword rule", "None of them"], 1,
     "ReAct observes the error and tries the homestay tool; a fixed plan breaks."),
    ("'How much is 5,000 rupees?' leads the model to convert to USD. What is this?",
     ["Tool selection error", "Parameter hallucination", "Output interpretation error", "Reflection"], 1,
     "Nobody named the target currency; the model invented one. Better descriptions or asking first can help."),
    ("Why did reflection fail to catch the Gurgaon bug?",
     ["Reflection is useless", "The critic only saw observation text and could not tell which city the data belonged to", "Gurgaon has no weather", "The critic was too strict"], 1,
     "Reflection checks only what it can see; it needs the request and provenance too."),
]


def page_review():
    st.header("Review, quiz and cheat sheet")
    st.subheader("Saturday night, fixed")
    df_show([
        {"Message": "'Help, our car broke down...'", "Layer": "Reactive", "LLM calls": 0, "Why": "Safety-critical: fast, fixed, reviewed reply"},
        {"Message": "'What should we pack for Manali?'", "Layer": "Sequencer", "LLM calls": 2, "Why": "Known routine: fixed steps, LLM fills one"},
        {"Message": "'Plan a 2-day food trail in Amritsar'", "Layer": "Deliberative", "LLM calls": "5+", "Why": "Open-ended: plan + ReAct + tools + reflection"},
    ])
    st.success("**No single best architecture.** Hybrids are where real systems end up. LLMs are a dynamic engine, at the cost of latency, money and unpredictability, which is why the reactive layer still matters.")
    st.subheader("🧠 Quiz")
    with st.form("quiz4"):
        answers = [st.radio("**%d. %s**" % (i + 1, q), opts, index=None, key="q4_%d" % i) for i, (q, opts, c, w) in enumerate(QUIZ)]
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
        {"#": "HW1", "Level": "Easy", "Task": "Find what keyword reflexes miss or wrongly flag; miss vs false alarm?", "Try it": "Page 3, Lab 1"},
        {"#": "HW2", "Level": "Medium", "Task": "Add an LLM emergency check to the router", "Try it": "Page 5 (checkbox)"},
        {"#": "HW3", "Level": "Medium", "Task": "Replan from the current state (max twice) instead of abandoning", "Try it": "Page 8 design exercise"},
        {"#": "HW4", "Level": "Easy", "Task": "Rewrite the currency tool description so it asks instead of guessing", "Try it": "Page 9 (checkbox)"},
        {"#": "HW5", "Level": "Medium", "Task": "What else should the reflection reviewer check?", "Try it": "Page 10"},
        {"#": "HW6", "Level": "Hard", "Task": "Keep beliefs in a vector memory and retrieve only relevant ones", "Try it": "Page 4 + Session 3"},
        {"#": "HW7", "Level": "Hard", "Task": "Store lessons in a vector store; do hotel lessons help restaurants?", "Try it": "Page 9, Reflection"},
    ])
    st.subheader("⚡ Quick reference")
    df_show([
        {"Idea": "Question of the session", "One line": "How much thinking does this moment need, and who decides?"},
        {"Idea": "Deliberative", "One line": "Sense, think, act. Plan first. LLM proposes, code checks."},
        {"Idea": "Reactive", "One line": "Sense, act. IF/THEN reflexes; subsumption layers suppress lower ones."},
        {"Idea": "BDI", "One line": "Beliefs, desires, committed intentions. Patch small changes, reconsider big ones."},
        {"Idea": "Layered", "One line": "Reflex, routine, strategy, chosen by an orchestrator."},
        {"Idea": "CoALA", "One line": "LLM at the core: memory modules, internal/external actions, propose-evaluate-select."},
        {"Idea": "Planning", "One line": "Complete plan before acting; brittle."},
        {"Idea": "ReAct", "One line": "Thought, Action, Observation loop; adaptive."},
        {"Idea": "Tool use", "One line": "Selection, parameter generation, output interpretation."},
        {"Idea": "Reflection", "One line": "Execute, evaluate, refine, integrate (save the lesson)."},
    ])


# ----------------------------------------------------------------------------------------
# Sidebar + router
# ----------------------------------------------------------------------------------------
def sidebar():
    st.sidebar.title("🏗️ Architecture Lab")
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
        st.sidebar.info("Demo mode: simulated labs, pretend data.")


def main():
    sidebar()
    page = st.session_state.get("nav", PAGES[0])
    {PAGES[0]: page_home, PAGES[1]: page_why, PAGES[2]: page_deliberative, PAGES[3]: page_reactive, PAGES[4]: page_bdi,
     PAGES[5]: page_layered, PAGES[6]: page_coala, PAGES[7]: page_choose, PAGES[8]: page_plan_react, PAGES[9]: page_tools_reflect,
     PAGES[10]: page_capstone, PAGES[11]: page_review}[page]()


main()
