"""
Engine for the Three C's lab (Agents, Session 7: The Three C's in Multi-Agent Systems,
plus the post-read "How Agents Learn From Each Other").

Everything here is plain Python, so every lab runs with NO API key:
  * communication patterns (request-response, publish-subscribe, blackboard) as small simulations
  * the three core coordination problems (task allocation, resource allocation, conflict resolution)
  * a CrewAI crew checker that predicts what a crew will do, including the exact errors CrewAI raises
    (checked against CrewAI 1.15.30 with a stub model, no API calls)
  * multi-agent reinforcement learning (MARL) as tiny, readable learning games

All data (tickets, diagnoses, payoffs) is PRETEND data written for this exercise.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata


# ----------------------------------------------------------------------------------------
# Optional live backend (Gemini). Demo mode never touches it.
# ----------------------------------------------------------------------------------------
class Backend:
    def __init__(self, api_key: str = "", chat_model: str = "gemini-3.5-flash-lite", rate_limit: bool = True):
        self.live, self.error, self.last_source, self.llm = False, "", "demo", None
        if api_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                kwargs = {}
                if rate_limit:
                    from langchain_core.rate_limiters import InMemoryRateLimiter
                    kwargs["rate_limiter"] = InMemoryRateLimiter(requests_per_second=4 / 60, check_every_n_seconds=0.5, max_bucket_size=1)
                self.llm = ChatGoogleGenerativeAI(model=chat_model, google_api_key=api_key, **kwargs)
                self.live = True
            except Exception as exc:
                self.error = "Could not start Gemini: " + str(exc)

    def ask(self, system: str, request: str, demo) -> str:
        """One agent turn. Live: Gemini writes it. Demo (or any error): the scripted text."""
        if self.live:
            try:
                from langchain_core.messages import HumanMessage, SystemMessage
                out = self.llm.invoke([SystemMessage(content=system), HumanMessage(content=request)])
                self.last_source = "gemini"
                text = getattr(out, "text", None)
                return text if isinstance(text, str) else str(out.content)
            except Exception as exc:
                self.error, self.last_source = "Gemini call failed, showing the demo text: " + str(exc), "demo (fallback)"
        else:
            self.last_source = "demo"
        return demo() if callable(demo) else demo


# ----------------------------------------------------------------------------------------
# 1. The pyramid: communication -> coordination -> cooperation
# ----------------------------------------------------------------------------------------
PYRAMID = [
    {"Layer": "3 · Cooperation (synergy)", "Question": "How can we help each other succeed?", "Adds": "Joint problem-solving, resource sharing, adaptive help, shared ownership",
     "Failure symptom": "Suboptimal results, missed opportunities", "Success measure": "Collective achievement, synergy"},
    {"Layer": "2 · Coordination (organisation)", "Question": "Who does what, and when?", "Adds": "Task allocation, scheduling, dependencies, conflict prevention",
     "Failure symptom": "Chaos, conflicts, duplication", "Success measure": "Efficiency, conflict avoidance"},
    {"Layer": "1 · Communication (foundation)", "Question": "What should we know?", "Adds": "Exchanging information, status updates",
     "Failure symptom": "Ignorance, isolation", "Success measure": "Accurate, timely information"},
]


def cascade(communication: bool, coordination: bool, cooperation: bool) -> dict:
    """What a construction crew (or an agent team) looks like with each layer switched on or off.
    The layers depend on the ones below: a higher layer cannot work if a lower one is missing."""
    effective_coord = coordination and communication
    effective_coop = cooperation and effective_coord
    if not communication:
        level, verdict = 0, "Chaos"
        story = ("Nobody knows what anyone else is doing. Concrete is poured where the bricks should go, two workers fetch the same window, "
                 "and the plumber arrives before there is a wall.")
    elif not effective_coord:
        level, verdict = 1, "Informed but disorganised"
        story = ("Everyone shouts updates (\"I'm pouring concrete!\"), so they're informed, but there is no plan. Windows go in before the walls are ready; "
                 "two people do the same job while another job waits.")
    elif not effective_coop:
        level, verdict = 2, "Organised, working alone"
        story = ("Blueprints are shared and the schedule is clear: foundation → walls → windows → roof. The work happens in the right order, "
                 "but each worker only does their own part. When the window installer struggles with a heavy frame, nobody helps.")
    else:
        level, verdict = 3, "A real team"
        story = ("Updates flow, the schedule holds, AND the brick layer helps lift the heavy frame, the concrete pourer shares spare materials, "
                 "and they jointly fix an unexpected plumbing problem. The team achieves more than its members could alone.")
    notes = []
    if coordination and not communication:
        notes.append("You switched coordination on, but without communication it cannot work: a schedule nobody hears about is no schedule.")
    if cooperation and not effective_coord:
        notes.append("You switched cooperation on, but it cannot emerge without communication AND coordination underneath it.")
    return {"level": level, "verdict": verdict, "story": story, "notes": notes,
            "working": {"Communication": communication, "Coordination": effective_coord, "Cooperation": effective_coop}}


# ----------------------------------------------------------------------------------------
# 2. Communication patterns
# ----------------------------------------------------------------------------------------
PATTERNS = [
    {"Pattern": "Request–response", "Communication type": "Direct (one-to-one)", "Style": "Synchronous: the sender waits",
     "Example": "Employee emails IT support", "Advantage": "Simple, reliable, easy to trace", "Limitation": "Blocking, tightly coupled, not scalable"},
    {"Pattern": "Publish–subscribe", "Communication type": "Broadcast (one-to-many)", "Style": "Asynchronous, event-driven",
     "Example": "Company-wide announcement, news alert", "Advantage": "Scalable, decoupled", "Limitation": "No feedback loop, possible overload, message-ordering issues"},
    {"Pattern": "Blackboard", "Communication type": "Shared memory (many-to-many)", "Style": "Asynchronous",
     "Example": "A shared project board", "Advantage": "Flexible, supports complex problem-solving", "Limitation": "Bottlenecks, needs conflict management, consistency issues"},
]


def request_response(n_requests: int, seconds_each: float, parallel_workers: int = 1) -> dict:
    """One requester waits for each reply before sending the next (blocking). Compare with a broadcast that doesn't wait."""
    rows, t = [], 0.0
    for i in range(n_requests):
        worker = i % parallel_workers
        start = (i // parallel_workers) * seconds_each
        rows.append({"Request": i + 1, "To": "Support agent %d" % (worker + 1), "Sent at (s)": round(start, 1),
                     "Reply at (s)": round(start + seconds_each, 1), "Requester": "waiting (blocked)"})
        t = max(t, start + seconds_each)
    return {"rows": rows, "total": round(t, 1), "waiting": round(t, 1)}


def publish(subscriptions: dict, topic: str, message: str) -> dict:
    """Publish one message to a topic. Every subscriber of that topic gets it; the publisher gets no reply."""
    deliveries = []
    for agent, topics in subscriptions.items():
        got = topic in topics
        deliveries.append({"Agent": agent, "Subscribed to": ", ".join(sorted(topics)) or "(nothing)",
                           "Receives it?": "✅ yes" if got else "—", "Reply to the publisher": "none (pub/sub has no feedback loop)" if got else ""})
    receivers = [d["Agent"] for d in deliveries if d["Receives it?"].startswith("✅")]
    return {"deliveries": deliveries, "receivers": receivers, "message": message, "topic": topic}


BLACKBOARD_MODES = {
    "No protection (shared file)": "Both agents write latest_code.py. The last write wins.",
    "Lock, then read before writing": "An agent must hold the lock to write, and re-reads the board before writing, so it merges instead of overwriting.",
    "Unique keys (one entry per feature)": "Each agent writes its own entry: feature_A.py and feature_B.py. Nothing to collide on.",
    "Versioning (compare-and-set)": "A write only succeeds if the board hasn't changed since you read it. Otherwise re-read, merge and try again.",
}


def blackboard(mode: str, gap_seconds: float) -> dict:
    """Two developer agents finish at nearly the same time and post their code. What does the TestingAgent read?
    gap_seconds: how long after Agent A finishes Agent B finishes (0 = the same moment)."""
    events, board = [], {}
    a_code, b_code = "def feature_a(): ...  # Agent A's search feature", "def feature_b(): ...  # Agent B's export feature"
    # Both read the board when they started work (version 0, empty)
    events.append({"t (s)": 0.0, "Agent": "A and B", "Action": "read the board (version 0)", "Board after": "{}"})
    version = 0
    if mode.startswith("No protection"):
        board["latest_code.py"] = a_code
        version += 1
        events.append({"t (s)": 10.0, "Agent": "Agent A", "Action": "writes latest_code.py", "Board after": "latest_code.py = A's code"})
        board["latest_code.py"] = b_code
        version += 1
        events.append({"t (s)": 10.0 + gap_seconds, "Agent": "Agent B", "Action": "writes latest_code.py (overwrites, it never saw A's write)", "Board after": "latest_code.py = B's code"})
        lost = ["Agent A's feature"]
    elif mode.startswith("Lock"):
        events.append({"t (s)": 10.0, "Agent": "Agent A", "Action": "takes the lock, writes, releases", "Board after": "latest_code.py = A's code"})
        board["latest_code.py"] = a_code
        if gap_seconds < 1.0:
            events.append({"t (s)": 10.0 + gap_seconds, "Agent": "Agent B", "Action": "asks for the lock: busy, so it WAITS", "Board after": "(unchanged)"})
        events.append({"t (s)": 11.0 + gap_seconds, "Agent": "Agent B", "Action": "takes the lock, RE-READS the board, merges, writes", "Board after": "latest_code.py = A's + B's code"})
        board["latest_code.py"] = a_code + "\n" + b_code
        lost = []
    elif mode.startswith("Unique"):
        board["feature_A.py"] = a_code
        board["feature_B.py"] = b_code
        events.append({"t (s)": 10.0, "Agent": "Agent A", "Action": "writes feature_A.py", "Board after": "feature_A.py"})
        events.append({"t (s)": 10.0 + gap_seconds, "Agent": "Agent B", "Action": "writes feature_B.py", "Board after": "feature_A.py, feature_B.py"})
        lost = []
    else:  # versioning
        board["latest_code.py"] = a_code
        events.append({"t (s)": 10.0, "Agent": "Agent A", "Action": "write if version is still 0 → accepted (now version 1)", "Board after": "latest_code.py = A's code (v1)"})
        events.append({"t (s)": 10.0 + gap_seconds, "Agent": "Agent B", "Action": "write if version is still 0 → REJECTED (it's v1 now)", "Board after": "(unchanged)"})
        events.append({"t (s)": 10.5 + gap_seconds, "Agent": "Agent B", "Action": "re-reads v1, merges, writes if still v1 → accepted (v2)", "Board after": "latest_code.py = A's + B's code (v2)"})
        board["latest_code.py"] = a_code + "\n" + b_code
        lost = []
    tester_sees = "\n\n".join("# %s\n%s" % (k, v) for k, v in board.items())
    return {"events": events, "board": board, "lost": lost, "tester_sees": tester_sees}


# ----------------------------------------------------------------------------------------
# 3. Coordination and cooperation: the three core problems (the smart-hospital example)
# ----------------------------------------------------------------------------------------
CORE_PROBLEMS = [
    {"Core problem": "Task allocation", "Question": "Who does what?", "Analogy": "A manager assigning project tasks",
     "Hospital example": "Treatment starts before Diagnosis has finished, with nothing to work on",
     "Common solutions": "A central task coordinator; a workflow protocol (wait for the previous signal); capability-based allocation; bidding"},
    {"Core problem": "Resource allocation", "Question": "Who gets the shared resource?", "Analogy": "Several employees, one printer",
     "Hospital example": "Diagnosis and Treatment hit the medical knowledge API at once and trip its rate limit",
     "Common solutions": "A resource manager that queues requests; priority scheduling; locks or tokens"},
    {"Core problem": "Conflict resolution", "Question": "What happens when agents disagree or clash?", "Analogy": "Two managers wanting the same employee",
     "Hospital example": "Diagnosis says flu, Symptom Analysis now says allergy: antivirals or antihistamines?",
     "Common solutions": "A referee or mediator agent; confidence-based arbitration; voting / consensus; a reflective reasoning agent"},
]

COORD_VS_COOP = [
    {"Aspect": "Purpose", "Coordination": "Manage dependencies and task allocation", "Cooperation": "Achieve mutual benefit through joint action"},
    {"Aspect": "Nature", "Coordination": "Can be competitive OR cooperative", "Cooperation": "Always aligned objectives; mutually beneficial"},
    {"Aspect": "Focus", "Coordination": "Who does what and when", "Cooperation": "How agents work together effectively"},
    {"Aspect": "Example", "Coordination": "A coordinator assigns tasks to avoid conflicts", "Cooperation": "Agents jointly write a report and improve it together"},
]

PROBLEM_SCENARIOS = [
    ("A new patient record arrives. The Diagnosis and Treatment agents both start at once, and Treatment has no diagnosis to work from.",
     "Task allocation", "It's about who works on what, in which order. Fix: a coordinator or a protocol (Treatment waits for Diagnosis's signal)."),
    ("Diagnosis and Treatment both query the same medical-knowledge API at the same moment and get rate-limit errors.",
     "Resource allocation", "One shared asset, two users. Fix: a resource manager that queues, priority access, or a lock/token."),
    ("The Diagnosis agent says 'flu'. The Symptom agent, with newer input, says 'allergy'. Treatment doesn't know what to prescribe.",
     "Conflict resolution", "Contradictory outputs. Fix: a referee, confidence or recency arbitration, or a vote."),
    ("Two warehouse drones reach opposite ends of the same one-way corridor at the same moment. They must agree which one yields.",
     "Conflict resolution", "Both want the same thing at the same time and must settle the clash. (The corridor is a shared resource too, but the course treats the simultaneous clash as conflict resolution.)"),
    ("Ten report-writing agents all need the single licensed PDF-export service, which handles one job at a time.",
     "Resource allocation", "Classic 'one printer, many employees'. Fix: a queue or a token."),
    ("A ticket arrives and two support agents both answer it, while a billing ticket waits unanswered.",
     "Task allocation", "Duplicated work and a dropped task: nobody decided who owns which ticket."),
]


def resource_schedule(agents: list, strategy: str, capacity_per_tick: int, priorities: dict | None = None, seed: int = 7) -> dict:
    """Agents each need ONE call to a shared API that serves `capacity_per_tick` calls per tick.
    No coordination: everyone fires every tick; the API serves `capacity` in random order and rejects the rest (errors).
    Queue: a resource manager serves arrivals in order. Priority: the manager serves the highest priority first.
    Token: only the agent holding the single token may call (capacity 1)."""
    rng = random.Random(seed)
    waiting = list(agents)
    rows, errors, done_at, tick = [], 0, {}, 0
    cap = 1 if strategy.startswith("Token") else max(1, capacity_per_tick)
    if strategy.startswith("Priority"):
        pr = priorities or {}
        waiting.sort(key=lambda a: -pr.get(a, 0))
    while waiting and tick < 50:
        tick += 1
        if strategy.startswith("No coordination"):
            tried = list(waiting)
            rng.shuffle(tried)
            served, rejected = tried[:cap], tried[cap:]
            errors += len(rejected)
        else:
            served, rejected = waiting[:cap], []
        for a in served:
            done_at[a] = tick
        waiting = [a for a in waiting if a not in served]
        rows.append({"Tick": tick, "Served": ", ".join(served) or "—",
                     "Rejected (429 Too Many Requests)": ", ".join(rejected) or "—",
                     "Still waiting": ", ".join(waiting) or "—"})
    return {"rows": rows, "errors": errors, "done_at": done_at, "ticks": tick}


def resolve_conflict(opinions: list, method: str) -> dict:
    """opinions: [{"Agent", "Answer", "Confidence", "Minutes ago"}]. Returns the decision and why."""
    ops = [o for o in opinions if str(o.get("Answer", "")).strip()]
    if not ops:
        return {"decision": "(no opinions)", "why": "Nothing to resolve."}
    if method.startswith("Majority"):
        counts = {}
        for o in ops:
            counts[o["Answer"]] = counts.get(o["Answer"], 0) + 1
        top = max(counts.values())
        winners = [a for a, c in counts.items() if c == top]
        if len(winners) > 1:
            return {"decision": "TIE: " + " vs ".join(winners), "why": "Votes: %s. A tie needs a tie-breaker (a referee, or confidence)." % counts}
        return {"decision": winners[0], "why": "Votes: %s. The majority wins, even if the majority is less sure." % counts}
    if method.startswith("Confidence"):
        best = max(ops, key=lambda o: float(o["Confidence"]))
        return {"decision": best["Answer"], "why": "%s is the most confident (%.2f). Risk: a confident agent is not always a correct one." % (best["Agent"], float(best["Confidence"]))}
    if method.startswith("Most recent"):
        best = min(ops, key=lambda o: float(o["Minutes ago"]))
        return {"decision": best["Answer"], "why": "%s has the newest data (%s min ago). Good when the situation changes; risky if new data is noisy." % (best["Agent"], best["Minutes ago"])}
    # Referee: weigh confidence by freshness, and show the working (a mediator agent explains its choice)
    scores = {}
    for o in ops:
        fresh = 1.0 / (1.0 + float(o["Minutes ago"]) / 30.0)
        scores.setdefault(o["Answer"], 0.0)
        scores[o["Answer"]] += float(o["Confidence"]) * fresh
    best = max(scores, key=scores.get)
    detail = ", ".join("%s %.2f" % (k, v) for k, v in sorted(scores.items(), key=lambda kv: -kv[1]))
    return {"decision": best, "why": "The referee scores each answer as confidence × freshness, summed over agents: %s. "
                                     "A real referee agent would also ask for the evidence and explain its choice to a human." % detail}


# ----------------------------------------------------------------------------------------
# 4. Coordination mechanisms
# ----------------------------------------------------------------------------------------
MECHANISMS = [
    {"Mechanism": "Centralized", "How": "One orchestrator decides who does what and when", "Analogy": "A project manager",
     "Solves": "Task allocation", "Cooperation it enables": "Task sharing / joint task execution",
     "Strengths": "Simple, clear accountability, easy to optimise", "Weaknesses": "Single point of failure; doesn't scale to huge systems", "CrewAI": "Process.hierarchical (manager/worker)"},
    {"Mechanism": "Distributed", "How": "No single controller; agents share capabilities and load, and negotiate", "Analogy": "Specialists dividing work without a manager",
     "Solves": "Resource allocation, conflict resolution", "Cooperation it enables": "Resource sharing, joint decision making",
     "Strengths": "Robust (no single point of failure), scalable, adaptive", "Weaknesses": "Complex; negotiation overhead", "CrewAI": "Peers with allow_delegation=True (peer/swarm)"},
    {"Mechanism": "Rule-based / protocol-driven", "How": "Predefined rules trigger actions, no negotiation", "Analogy": "Standard operating procedures (SOPs)",
     "Solves": "Task and resource allocation", "Cooperation it enables": "Feedback / iterative improvement loops",
     "Strengths": "Predictable, easy to debug", "Weaknesses": "Inflexible; poor with surprises", "CrewAI": "Process.sequential with context=[...] (pipeline)"},
    {"Mechanism": "Learning-based", "How": "Agents try strategies, get rewards, and self-organise over time", "Analogy": "A team that learns from past projects",
     "Solves": "All three (adaptively)", "Cooperation it enables": "Learning-based cooperation (cooperation that improves itself)",
     "Strengths": "Adaptive, scalable, can optimise dynamically", "Weaknesses": "Hard to build; needs training data; unpredictable early on", "CrewAI": "Not built in: you add a feedback loop (Session 7 post-read)"},
]

MECH_SCENARIOS = [
    ("A central controller assigns LLM1 to summarise patient reports, LLM2 to write follow-up questions and LLM3 to book an appointment.",
     "Centralized", "One controller holds the whole plan and hands out the parts."),
    ("Several LLM agents summarising a research paper negotiate among themselves who takes which section, and take turns on a shared reference database.",
     "Distributed", "No boss: they negotiate (joint decision making) and share a resource."),
    ("A rule says: if the QuestionAgent finds missing key points in a summary, the task goes back to the SummarizationAgent automatically.",
     "Rule-based", "An if-then rule triggers the hand-back. No negotiation, no learning."),
    ("Agents try different orders of research, summarising and question-writing, and over many runs keep the order that scores best.",
     "Learning-based", "Feedback and rewards shape the workflow over time."),
    ("A city replaces its central traffic computer with one RL agent per intersection. Agents act on local sensors with no central server, and improve through reward.",
     "Hybrid: learning-based inside a distributed structure", "Two features: peers with no central server (distributed structure) AND reinforcement learning (learning strategy). Beware options that say 'purely'."),
    ("A drone swarm with no control tower: two drones meeting at a corridor negotiate directly who yields.",
     "Distributed", "Peers, real-time negotiation, nobody in charge."),
    ("A CrewAI crew of three equal agents runs Developer → Tester → Writer with Process.sequential and no manager.",
     "Distributed", "Course answer: coordination comes from a pre-agreed protocol (the task list), not a live orchestrator, so no agent has authority over the others. (It is also rule-based in style: the fixed order is a rule.)"),
]
MECH_CHOICES = ["Centralized", "Distributed", "Rule-based", "Learning-based", "Hybrid: learning-based inside a distributed structure"]


# ----------------------------------------------------------------------------------------
# 5. A CrewAI crew checker (predicts what a crew does, with CrewAI 1.15's real error messages)
# ----------------------------------------------------------------------------------------
def check_crew(cfg: dict) -> dict:
    """cfg = {"process": "sequential"|"hierarchical", "manager": None|"manager_llm"|"manager_agent",
              "manager_in_agents": bool, "manager_has_tools": bool,
              "agents": [{"name", "tools": bool, "allow_delegation": bool}],
              "tasks": [{"id", "description", "agent": name|None, "context": None|[ids], "output_file": str}]}
    context None means "not set" (CrewAI then passes ALL earlier outputs); [] means "explicitly nothing"."""
    errors, warnings, info, flow = [], [], [], []
    tasks, agents = cfg["tasks"], cfg["agents"]
    names = [a["name"] for a in agents]
    ids = [t["id"] for t in tasks]
    proc = cfg["process"]

    # --- errors CrewAI raises when the Crew(...) is created ---
    if proc == "hierarchical" and not cfg.get("manager"):
        errors.append(("Crew(...)", "Attribute `manager_llm` or `manager_agent` is required when using hierarchical process."))
    if proc == "hierarchical" and cfg.get("manager") == "manager_agent" and cfg.get("manager_in_agents"):
        errors.append(("Crew(...)", "Manager agent should not be included in agents list."))
    if proc == "sequential":
        for t in tasks:
            if not t.get("agent"):
                errors.append(("Crew(...)", "Sequential process error: Agent is missing in the task with the following description: %s" % t["description"]))
    for i, t in enumerate(tasks):
        for c in t.get("context") or []:
            if c not in ids:
                errors.append(("your code", "%s uses context=[%s], but no task called %s is in the crew (Python would fail with NameError first)." % (t["id"], c, c)))
            elif ids.index(c) > i:
                other = tasks[ids.index(c)]
                errors.append(("Crew(...)", "Task '%s' has a context dependency on a future task '%s', which is not allowed." % (t["description"], other["description"])))
        if t.get("agent") and t["agent"] not in names:
            warnings.append("%s is assigned to '%s', who is not in the crew's agents list." % (t["id"], t["agent"]))
    # --- error raised later, at kickoff ---
    if proc == "hierarchical" and cfg.get("manager") == "manager_agent" and cfg.get("manager_has_tools"):
        errors.append(("crew.kickoff()", "Manager agent should not have tools"))

    # --- the flow, if it would run ---
    done = []
    for i, t in enumerate(tasks):
        ctx = t.get("context")
        if ctx is None:
            receives = ", ".join(done) if done else "nothing (it runs first)"
            how = "context not set → CrewAI passes ALL earlier outputs"
        elif ctx == []:
            receives = "nothing"
            how = "context=[] → explicitly no earlier outputs"
        else:
            ok = [c for c in ctx if c in done]
            receives = ", ".join(ok) if ok else "nothing (its context hasn't run yet)"
            how = "context=[%s]" % ", ".join(ctx)
        if proc == "hierarchical":
            who = ("manager may only delegate to %s" % t["agent"]) if t.get("agent") else "manager chooses a worker"
        else:
            who = t.get("agent") or "(no agent!)"
        flow.append({"Order": i + 1, "Task": t["id"], "Done by": who, "Receives": receives, "Why": how,
                     "Saves to file": t.get("output_file") or ""})
        done.append(t["id"])
        if ctx == [] and i > 0:
            warnings.append("%s has context=[]: it sees none of the earlier results. If it needs them, it will improvise (a silent failure, no error)." % t["id"])
        if ctx and any(c in ids and ids.index(c) < i for c in ctx) and len(ctx) < i:
            missing = [x for x in ids[:i] if x not in ctx]
            if missing:
                info.append("%s only reads %s. It will NOT see %s." % (t["id"], ", ".join(ctx), ", ".join(missing)))

    # --- what kind of crew is this? ---
    delegating = [a["name"] for a in agents if a.get("allow_delegation")]
    if proc == "hierarchical":
        assigned = [t for t in tasks if t.get("agent")]
        if assigned and len(assigned) == len(tasks):
            pattern = "Pipeline in disguise: every task is assigned, so the manager has no choice (extra manager calls for nothing)"
        elif cfg.get("manager") == "manager_agent":
            pattern = "Federated teams / specialist agents: your own manager_agent routes work to specialists"
        else:
            pattern = "Manager / worker: CrewAI builds a manager from manager_llm, and it allocates the tasks"
        mechanism = "Centralized coordination"
    else:
        if delegating:
            pattern = "Peer / swarm (CrewAI style): fixed turns, plus peers can ask each other (allow_delegation=True)"
            mechanism = "Distributed coordination (no boss; a pre-agreed turn order)"
        else:
            pattern = "Sequential pipeline: each task's output feeds the next"
            mechanism = "Rule-based / protocol-driven coordination"
    if delegating:
        info.append("allow_delegation=True gives %s two extra tools: 'Delegate work to coworker' and 'Ask question to coworker'." % ", ".join(delegating))
    if proc == "sequential" and len(delegating) == 1 and len(agents) > 1:
        warnings.append("Only %s can delegate. In a swarm of equals this quietly makes one agent a mini-manager (the 'Trojan horse' for hierarchy)." % delegating[0])
    outfile = [t for t in tasks if t.get("output_file")]
    last = tasks[-1]["id"] if tasks else "-"
    ret = ("result = crew.kickoff() returns a CrewOutput. result.raw is the LAST task's (%s) full text; result.tasks_output holds every task's output." % last)
    if outfile:
        ret += " output_file only saves a COPY to %s; the return value is unchanged." % ", ".join(t["output_file"] for t in outfile)
    return {"errors": errors, "warnings": warnings, "info": info, "flow": flow, "pattern": pattern, "mechanism": mechanism,
            "returns": ret, "runs": not errors}


def crew_code(cfg: dict) -> str:
    """The Python a learner would write for this crew (CrewAI 1.15 style)."""
    lines = ["from crewai import Agent, Task, Crew, Process, LLM", "", 'llm = LLM(model="gemini/gemini-3.5-flash-lite")', ""]
    for a in cfg["agents"]:
        extra = ""
        if a.get("tools"):
            extra += ", tools=[search_wikipedia]"
        if a.get("allow_delegation"):
            extra += ", allow_delegation=True"
        lines.append('%s = Agent(role="%s", goal="...", backstory="...", llm=llm%s)' % (a["name"], a["name"].replace("_", " ").title(), extra))
    if cfg["process"] == "hierarchical" and cfg.get("manager") == "manager_agent":
        lines.append('manager = Agent(role="Manager", goal="Delegate to the right specialist", backstory="...", llm=llm%s)' %
                     (", tools=[search_wikipedia]" if cfg.get("manager_has_tools") else ""))
    lines.append("")
    for t in cfg["tasks"]:
        parts = ['description="%s"' % t["description"], 'expected_output="..."']
        if t.get("agent"):
            parts.append("agent=%s" % t["agent"])
        if t.get("context") is not None:
            parts.append("context=[%s]" % ", ".join(t["context"]))
        if t.get("output_file"):
            parts.append('output_file="%s"' % t["output_file"])
        lines.append("%s = Task(%s)" % (t["id"], ", ".join(parts)))
    lines.append("")
    agent_list = [a["name"] for a in cfg["agents"]]
    if cfg["process"] == "hierarchical" and cfg.get("manager") == "manager_agent" and cfg.get("manager_in_agents"):
        agent_list.append("manager")
    crew = ["    agents=[%s]," % ", ".join(agent_list), "    tasks=[%s]," % ", ".join(t["id"] for t in cfg["tasks"]),
            "    process=Process.%s," % cfg["process"]]
    if cfg["process"] == "hierarchical":
        if cfg.get("manager") == "manager_llm":
            crew.append("    manager_llm=llm,")
        elif cfg.get("manager") == "manager_agent":
            crew.append("    manager_agent=manager,")
    crew += ["    max_rpm=4,", "    verbose=True,"]
    lines += ["crew = Crew("] + crew + [")", "", "result = crew.kickoff()          # in a notebook: result = await crew.kickoff_async()", "print(result.raw)"]
    return "\n".join(lines)


def _t(i, d, agent=None, context=None, output_file=""):
    return {"id": i, "description": d, "agent": agent, "context": context, "output_file": output_file}


CREW_PRESETS = {
    "Demo 2 · Manager / worker (notebook)": {
        "process": "hierarchical", "manager": "manager_llm", "manager_in_agents": False, "manager_has_tools": False,
        "agents": [{"name": "data_analyst", "tools": True, "allow_delegation": False}, {"name": "process_optimizer", "tools": False, "allow_delegation": False},
                   {"name": "report_writer", "tools": False, "allow_delegation": False}],
        "tasks": [_t("analysis_task", "Analyse last quarter's support data"), _t("improvement_task", "Suggest three improvements", context=["analysis_task"]),
                  _t("report_task", "Write a 200-word report for the COO", context=["analysis_task", "improvement_task"])]},
    "Demo 3 · Sequential pipeline (notebook)": {
        "process": "sequential", "manager": None, "manager_in_agents": False, "manager_has_tools": False,
        "agents": [{"name": "researcher", "tools": True, "allow_delegation": False}, {"name": "analyst", "tools": False, "allow_delegation": False},
                   {"name": "report_writer", "tools": False, "allow_delegation": False}],
        "tasks": [_t("research_task", "Research three uses of AI in software", "researcher"), _t("analysis_task", "Pick the most promising use", "analyst", ["research_task"]),
                  _t("report_task", "Write a one-page summary", "report_writer", ["analysis_task"])]},
    "Demo 4 · Peer / swarm (notebook)": {
        "process": "sequential", "manager": None, "manager_in_agents": False, "manager_has_tools": False,
        "agents": [{"name": "developer", "tools": False, "allow_delegation": True}, {"name": "tester", "tools": False, "allow_delegation": True},
                   {"name": "documentation_writer", "tools": False, "allow_delegation": True}],
        "tasks": [_t("coding_task", "Write calculate_average", "developer"), _t("testing_task", "Test calculate_average", "tester", ["coding_task"]),
                  _t("documentation_task", "Document calculate_average", "documentation_writer", ["coding_task", "testing_task"])]},
    "Demo 5 · Federated teams (notebook)": {
        "process": "hierarchical", "manager": "manager_agent", "manager_in_agents": False, "manager_has_tools": False,
        "agents": [{"name": "researcher", "tools": True, "allow_delegation": False}, {"name": "analyst", "tools": False, "allow_delegation": False},
                   {"name": "writer", "tools": False, "allow_delegation": False}],
        "tasks": [_t("report_task", "Create a report on three robotics trends")]},
    "🐞 Midnight report: task list 'tidied up'": {
        "process": "sequential", "manager": None, "manager_in_agents": False, "manager_has_tools": False,
        "agents": [{"name": "market_scanner", "tools": True, "allow_delegation": False}, {"name": "anomalies_analyst", "tools": False, "allow_delegation": False},
                   {"name": "executive_summarizer", "tools": False, "allow_delegation": False}],
        "tasks": [_t("analysis_task", "Find outliers in the scan", "anomalies_analyst", ["scan_task"]), _t("scan_task", "Fetch end-of-day data", "market_scanner"),
                  _t("summary_task", "Write the midnight briefing", "executive_summarizer", ["analysis_task"])]},
    "🐞 Midnight report: 'decoupled' (context removed, order still wrong)": {
        "process": "sequential", "manager": None, "manager_in_agents": False, "manager_has_tools": False,
        "agents": [{"name": "market_scanner", "tools": True, "allow_delegation": False}, {"name": "anomalies_analyst", "tools": False, "allow_delegation": False},
                   {"name": "executive_summarizer", "tools": False, "allow_delegation": False}],
        "tasks": [_t("analysis_task", "Find outliers in the scan", "anomalies_analyst"), _t("scan_task", "Fetch end-of-day data", "market_scanner"),
                  _t("summary_task", "Write the midnight briefing", "executive_summarizer", ["analysis_task"])]},
    "🐞 Empty executive briefing: writer told to read nothing": {
        "process": "sequential", "manager": None, "manager_in_agents": False, "manager_has_tools": False,
        "agents": [{"name": "data_analyst", "tools": True, "allow_delegation": False}, {"name": "strategy_optimizer", "tools": False, "allow_delegation": False},
                   {"name": "executive_writer", "tools": False, "allow_delegation": False}],
        "tasks": [_t("analysis_task", "Find key financial trends", "data_analyst"), _t("strategy_task", "Propose strategies", "strategy_optimizer", ["analysis_task"]),
                  _t("executive_writing_task", "Write the COO briefing", "executive_writer", [])]},
    "🐞 Manager agent with tools, listed as a worker": {
        "process": "hierarchical", "manager": "manager_agent", "manager_in_agents": True, "manager_has_tools": True,
        "agents": [{"name": "researcher", "tools": True, "allow_delegation": False}, {"name": "writer", "tools": False, "allow_delegation": False}],
        "tasks": [_t("report_task", "Create a report")]},
    "🐞 Hierarchical, but every task assigned": {
        "process": "hierarchical", "manager": "manager_llm", "manager_in_agents": False, "manager_has_tools": False,
        "agents": [{"name": "data_analyst", "tools": True, "allow_delegation": False}, {"name": "report_writer", "tools": False, "allow_delegation": False}],
        "tasks": [_t("analysis_task", "Analyse the data", "data_analyst"), _t("report_task", "Write the report", "report_writer", ["analysis_task"])]},
    "🐞 Swarm with one 'Trojan horse' delegator": {
        "process": "sequential", "manager": None, "manager_in_agents": False, "manager_has_tools": False,
        "agents": [{"name": "code_developer", "tools": False, "allow_delegation": False}, {"name": "security_tester", "tools": False, "allow_delegation": True},
                   {"name": "documentation_writer", "tools": False, "allow_delegation": False}],
        "tasks": [_t("coding_task", "Write the feature", "code_developer"), _t("security_task", "Audit for vulnerabilities", "security_tester", ["coding_task"]),
                  _t("docs_task", "Document the approved code", "documentation_writer", ["coding_task", "security_task"], "release_notes.md")]},
}


# ----------------------------------------------------------------------------------------
# 6. How agents learn from each other: tiny MARL games
# ----------------------------------------------------------------------------------------
def corridor_game(episodes: int = 300, epsilon: float = 0.3, decay: float = 0.99, alpha: float = 0.2,
                  reward: str = "individual", seed: int = 1) -> dict:
    """Two drones, two corridors. Corridor A is short (worth 3), B is long (worth 1). If both pick the same corridor,
    they block each other and BOTH get 0 (they 'fight over the same item and both lose it').
    Each drone learns its own value for A and B (independent Q-learning, no state), choosing at random with probability epsilon.
    Because the other drone is learning too, the value of each choice keeps moving: a NON-STATIONARY environment."""
    rng = random.Random(seed)
    pay = {"A": 3.0, "B": 1.0}
    q = [{"A": 0.0, "B": 0.0}, {"A": 0.0, "B": 0.0}]
    rows, eps = [], epsilon
    for ep in range(1, episodes + 1):
        acts = []
        for i in range(2):
            if rng.random() < eps:
                acts.append(rng.choice(["A", "B"]))
            else:
                acts.append(max(q[i], key=lambda k: (q[i][k], rng.random())))
        clash = acts[0] == acts[1]
        own = [0.0 if clash else pay[a] for a in acts]
        team = sum(own)
        rs = own if reward == "individual" else [team / 2.0, team / 2.0]
        for i in range(2):
            q[i][acts[i]] += alpha * (rs[i] - q[i][acts[i]])
        rows.append({"Episode": ep, "Drone 1": acts[0], "Drone 2": acts[1], "Clash": int(clash), "Team score": team,
                     "Exploration ε": round(eps, 3), "Drone 1 value of A": round(q[0]["A"], 3), "Drone 1 value of B": round(q[0]["B"], 3),
                     "Drone 2 value of A": round(q[1]["A"], 3)})
        eps *= decay
    return {"rows": rows, "q": q}


def helping_game(episodes: int = 400, epsilon: float = 0.3, decay: float = 0.99, alpha: float = 0.1,
                 reward: str = "individual", seed: int = 3) -> dict:
    """Cooperation: each round, each agent either WORKS ALONE (+1 for itself) or HELPS (only +0.6 for itself,
    but its teammate gets +1 extra). With individual rewards, not helping always pays more for you.
    With a SHARED team reward, helping pays more, because your teammate's gain is your gain."""
    rng = random.Random(seed)
    q = [{"alone": 0.0, "help": 0.0}, {"alone": 0.0, "help": 0.0}]
    rows, eps = [], epsilon
    for ep in range(1, episodes + 1):
        acts = []
        for i in range(2):
            if rng.random() < eps:
                acts.append(rng.choice(["alone", "help"]))
            else:
                acts.append(max(q[i], key=lambda k: (q[i][k], rng.random())))
        own = [(0.6 if acts[i] == "help" else 1.0) + (1.0 if acts[1 - i] == "help" else 0.0) for i in range(2)]
        team = sum(own)
        rs = own if reward == "individual" else [team / 2.0] * 2
        for i in range(2):
            q[i][acts[i]] += alpha * (rs[i] - q[i][acts[i]])
        rows.append({"Episode": ep, "Agent 1": acts[0], "Agent 2": acts[1], "Helps this round": acts.count("help"),
                     "Team score": round(team, 2), "Exploration ε": round(eps, 3)})
        eps *= decay
    return {"rows": rows, "q": q}


WORKFLOWS = {
    "Workflow order": {"arms": ["Research → Write → Edit", "Research ∥ Write (parallel) → Edit", "Write → Research → Edit"], "means": [0.82, 0.62, 0.45],
                       "story": "The team doesn't know which order gives the best report. Each run, it tries an order and a reviewer scores the report (0 to 1, with noise)."},
    "Manager delegation": {"arms": ["Give summaries to Agent A", "Give summaries to Agent B", "Give summaries to Agent C"], "means": [0.55, 0.78, 0.6],
                           "story": "A manager learns which worker produces the best summaries (Session 7 post-read, reflection 5.1): a reward signal from downstream quality."},
}


def learn_workflow(kind: str, rounds: int = 120, epsilon: float = 0.2, seed: int = 5) -> dict:
    """Learning-based coordination as an epsilon-greedy bandit: try, get a reward, update the estimate, mostly pick the best so far."""
    spec = WORKFLOWS[kind]
    rng = random.Random(seed)
    n = len(spec["arms"])
    est, count, rows, total = [0.0] * n, [0] * n, [], 0.0
    for r in range(1, rounds + 1):
        explore = rng.random() < epsilon or r <= n
        arm = (r - 1) % n if r <= n else (rng.randrange(n) if explore else max(range(n), key=lambda k: est[k]))
        score = min(1.0, max(0.0, rng.gauss(spec["means"][arm], 0.12)))
        count[arm] += 1
        est[arm] += (score - est[arm]) / count[arm]
        total += score
        rows.append({"Round": r, "Tried": spec["arms"][arm], "Explore?": "explore" if explore else "exploit",
                     "Reviewer score": round(score, 2), "Average so far": round(total / r, 3)})
    table = [{"Option": spec["arms"][k], "Times tried": count[k], "Learned value": round(est[k], 3), "True value (hidden)": spec["means"][k]} for k in range(n)]
    return {"rows": rows, "table": table, "best": spec["arms"][max(range(n), key=lambda k: est[k])]}


BEHAVIOURS = [
    {"Behaviour": "Agree with whatever the group says", "Team quality": 0.60, "Verified truth": 0.30, "Redundancy": 0.20, "Overconfidence": 0.10},
    {"Behaviour": "Cite evidence and flag uncertainty", "Team quality": 0.80, "Verified truth": 0.88, "Redundancy": 0.10, "Overconfidence": 0.00},
    {"Behaviour": "Copy a teammate's section", "Team quality": 0.70, "Verified truth": 0.50, "Redundancy": 0.90, "Overconfidence": 0.10},
    {"Behaviour": "Confident guess, no sources", "Team quality": 0.83, "Verified truth": 0.15, "Redundancy": 0.10, "Overconfidence": 0.90},
    {"Behaviour": "Work alone, ignore teammates", "Team quality": 0.40, "Verified truth": 0.92, "Redundancy": 0.30, "Overconfidence": 0.20},
]
REWARD_PRESETS = {
    "Team reward only": (1.0, 0.0, 0.0, 0.0),
    "Individual truthfulness only": (0.0, 1.0, 0.0, 0.0),
    "The post-read's dual reward (team + truth − penalties)": (1.0, 0.8, 0.5, 0.6),
}


def score_behaviours(w_team: float, w_truth: float, w_redundant: float, w_overconfident: float) -> list:
    out = []
    for b in BEHAVIOURS:
        s = w_team * b["Team quality"] + w_truth * b["Verified truth"] - w_redundant * b["Redundancy"] - w_overconfident * b["Overconfidence"]
        out.append(dict(b, Reward=round(s, 3)))
    out.sort(key=lambda r: -r["Reward"])
    return out


SELF_PLAY_DEMO = [
    {"draft": "Electric vehicles are good. They are better than petrol cars in every way and will replace them all by 2030.",
     "critique": "Score 4/10. Two unsupported claims ('every way', 'by 2030'). No numbers. Say what EVs are better AT, and drop the prediction or source it.", "score": 4},
    {"draft": "Electric vehicles produce no tailpipe emissions and usually cost less to run per km than petrol cars. Their main drawbacks are higher upfront prices and charging time.",
     "critique": "Score 7/10. Balanced and specific. Still vague on 'usually' and missing who benefits most. Add one concrete comparison and one sentence on the grid's role.", "score": 7},
    {"draft": "Electric vehicles produce no tailpipe emissions, and running costs per km are typically a fraction of petrol's. How clean they are overall depends on the electricity grid that charges them. Upfront prices and charging time remain the main drawbacks, so they suit drivers with home charging most.",
     "critique": "Score 9/10. Clear, hedged where it should be, and useful. Ready to publish.", "score": 9},
]


# ----------------------------------------------------------------------------------------
# 7. Quiz (exam style)
# ----------------------------------------------------------------------------------------
QUIZ = [
    ("In the Three C's hierarchy, which layer must work first?", ["Cooperation", "Coordination", "Communication"], 2,
     "Communication is the foundation. Coordination needs it, and cooperation needs both."),
    ("Agents know exactly what the others are doing, but work is duplicated and dependencies are missed. Which layer failed?", ["Communication", "Coordination", "Cooperation"], 1,
     "Good communication, failed coordination: 'chaos, conflicts, duplication'."),
    ("Which communication pattern is synchronous, one-to-one and blocking?", ["Publish–subscribe", "Request–response", "Blackboard"], 1,
     "The sender waits for the reply: simple and traceable, but blocking and not scalable."),
    ("A publisher broadcasts 'server overload detected' to a topic. What is a known limitation?", ["It is blocking", "There is no feedback loop to the publisher", "Only one agent can receive it"], 1,
     "Pub/sub is asynchronous and one-to-many, so the publisher never learns who acted. Ordering and overload are the other limitations."),
    ("Two agents write latest_code.py on a blackboard at the same moment and one feature is lost. Which TWO limitations does this show?",
     ["Blocking + no one-to-many", "Needs conflict management + consistency issues", "No feedback + ordering"], 1,
     "The blackboard is asynchronous and many-to-many (so not blocking). Concurrent writes need conflict management, and the board ended up inconsistent."),
    ("Diagnosis says flu, Symptom Analysis says allergy. Which core coordination problem is this?", ["Task allocation", "Resource allocation", "Conflict resolution"], 2,
     "Contradictory outputs: use a referee, confidence/recency arbitration, or a vote."),
    ("Which mechanism has a single point of failure?", ["Centralized", "Distributed", "Learning-based"], 0, "If the coordinator fails, the system stalls."),
    ("'If the reviewer finds a problem, send the task back to the writer.' Which mechanism and which cooperation strategy?",
     ["Rule-based; feedback / iterative improvement", "Distributed; joint decision making", "Learning-based; task sharing"], 0, "A predefined rule triggers a feedback loop."),
    ("Autonomous per-intersection traffic agents with no central server that improve through reward. Most complete classification?",
     ["Purely distributed", "Purely learning-based", "Learning-based coordination inside a distributed structure"], 2, "Two features, so the comprehensive answer names both. Distrust 'purely'."),
    ("In CrewAI, how do you make real manager/worker with Process.hierarchical?", ["Assign every task to an agent", "Leave the tasks unassigned so the manager chooses", "Give the manager tools"], 1,
     "If a task says agent=..., the manager can only pass it to that agent. And a manager_agent must have no tools."),
    ("A sequential crew's task list is reordered so analysis_task (context=[scan_task]) comes before scan_task. What does CrewAI 1.15 do?",
     ["Runs analysis on empty input", "Refuses to create the crew: a context dependency on a future task is not allowed", "Swaps the tasks back automatically"], 1,
     "Checked on CrewAI 1.15.30: a ValidationError at Crew(...). (The course calls it a 'ContextUnavailableError'; the cause is the same: the order.)"),
    ("In a sequential crew, a task has NO context parameter at all. What does it receive in CrewAI 1.15?", ["Nothing", "All earlier tasks' outputs", "Only the previous task's output"], 1,
     "Checked on CrewAI 1.15.30: an unset context means ALL earlier outputs. Only context=[] means nothing."),
    ("Why is a learning environment with several learning agents 'non-stationary'?", ["The data never changes", "Each agent's learning changes the world the others learn in", "Rewards are random"], 1,
     "Every policy update by one agent shifts what works for the others: a co-evolutionary loop."),
    ("Agents get a SHARED team reward. What behaviour does that encourage?", ["Free-riding", "Helping teammates, even at a short-term cost", "Working alone"], 1,
     "If one succeeds, all succeed: complementary behaviour and synergy."),
    ("What is RLPF?", ["Reinforcement learning from peer feedback: other agents provide the reward signal", "A routing protocol", "A vector database"], 0,
     "Like RLHF, but other LLMs (reviewers, critics) supply the evaluations instead of humans."),
]


def sanitize_tool_name(name: str, max_length: int = 64) -> str:
    """Kept here for completeness (CrewAI's tool-name cleaner); the Session 8 lab uses the same function."""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub(r"([a-z])([A-Z])", r"\1_\2", name).lower()
    name = re.sub(r"[\'\"]+", "", name)
    name = re.sub(r"[^a-zA-Z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if len(name) > max_length:
        suffix = "_" + hashlib.sha256(name.encode()).hexdigest()[:8]
        name = name[: max_length - len(suffix)].rstrip("_") + suffix
    return name


def to_json(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)
