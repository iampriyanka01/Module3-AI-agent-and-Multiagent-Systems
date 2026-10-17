"""
Engine for the Building Multi-Agent Systems lab (Agents, Session 6).

Plain Python, so every lab runs with NO API key and with neither LangChain nor CrewAI installed.
It contains:
  * three customer-support desks (centralized, decentralized, hybrid) that report every message and model call,
  * a tiny ReAct agent with a pretend search tool and a conversation memory,
  * a "mini CrewAI": Agent, Task, Crew, Process.sequential and Process.hierarchical, written from scratch to show
    what the real framework does for you. It is a teaching model of the ideas in the notebook, NOT the CrewAI library.
All facts, prices and search results are PRETEND data written for this exercise, never live information.
"""
from __future__ import annotations

import re


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
        """One agent turn. In Live mode Gemini writes it; otherwise (or on error) the scripted demo text is used."""
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


class Log:
    """A message log plus a model-call counter, so each design reports who said what to whom and what it cost."""

    def __init__(self):
        self.messages: list = []
        self.calls = 0

    def send(self, sender, recipient, kind, content, task_id="", direct=False):
        self.messages.append({"from": sender, "to": recipient, "kind": kind, "task_id": task_id, "content": content, "direct": direct})

    def call(self, n: int = 1):
        self.calls += n

    def rows(self) -> list:
        return [{"#": i + 1, "from": m["from"], "to": m["to"], "kind": m["kind"], "task": m["task_id"],
                 "content": m["content"][:120], "direct": "✅ peer to peer" if m["direct"] else ""} for i, m in enumerate(self.messages)]


# ----------------------------------------------------------------------------------------
# Part 1: one ReAct agent with a search tool and a memory (the notebook's "simple_customer_support")
# ----------------------------------------------------------------------------------------
STEPS_OF_AN_AGENT = [
    ("1 · Imports and setup", "Bring in the LLM wrapper, the search tool, the memory class, and load the API keys from a .env file.", "from langchain_google_genai import ChatGoogleGenerativeAI ... load_dotenv()"),
    ("2 · Agent initialisation", "Read GOOGLE_API_KEY and TAVILY_API_KEY from environment variables. Keys never go in the code.", "os.getenv(\"GOOGLE_API_KEY\")"),
    ("3 · AI model setup", "The brain: a Gemini model. temperature=0.3 keeps answers focused and fairly repeatable.", "ChatGoogleGenerativeAI(model=..., temperature=0.3)"),
    ("4 · Search tool", "The hands: real-time web search, limited to the 2 best results to keep the prompt short.", "TavilySearchResults(max_results=2)"),
    ("5 · Memory", "A buffer that stores earlier messages and puts them back into the prompt on every turn.", "ConversationBufferMemory(memory_key=\"chat_history\", return_messages=True)"),
    ("6 · Agent creation", "Wire model, tools and memory into one ReAct-style agent that decides when to search.", "initialize_agent(tools=[search], llm=llm, memory=memory, ...)"),
    ("7 · Chat loop", "Read input, call the agent, print the answer, catch API errors, stop on 'quit'.", "while True: ... agent.invoke(user_input)"),
]

CHAT_SCRIPT = ["What's the latest news about AI?",
               "What is the weather like in Bali in February?",
               "And what is the current visa fee there?"]

SEARCH_INDEX = [
    {"keys": ["ai", "news"], "query": "latest AI news",
     "results": [("PretendTech Daily", "Sample result: teams are trying groups of cooperating agents instead of one big agent."),
                 ("AI Weekly (pretend)", "Sample result: new frameworks focus on role-based orchestration of several agents."),
                 ("Pretend Blog", "Sample result: people compare manual orchestration with higher-level frameworks.")]},
    {"keys": ["bali", "weather"], "query": "Bali weather in February",
     "results": [("Pretend Travel Weather", "Sample result: February is in the wet season: warm and humid, with short heavy showers."),
                 ("Pretend Island Guide", "Sample result: mornings are usually clearer than afternoons."),
                 ("Pretend Forum", "Sample result: pack a light rain jacket.")]},
    {"keys": ["bali", "visa"], "query": "Bali visa fee for Indian passport holders",
     "results": [("Pretend Immigration Notes", "Sample result: entry fees and rules change; the official immigration site is the only reliable source."),
                 ("Pretend Travel Desk", "Sample result: carry a passport valid for at least six months and an onward ticket.")]},
    {"keys": ["goa", "weather"], "query": "Goa weather this week",
     "results": [("Pretend Weather Desk", "Sample result: warm and sunny with a sea breeze in the evening."),
                 ("Pretend Beach Guide", "Sample result: expect crowds at the weekend.")]},
]
KNOWLEDGE = {
    "what is a react agent": "A ReAct agent alternates between reasoning and acting: it thinks, calls a tool, reads the result, and thinks again.",
    "what is memory": "Memory is what your code stores and puts back into the prompt; the model itself remembers nothing between calls.",
}
PLACES = ["bali", "goa", "kerala"]
SEARCH_WORDS = ("latest", "news", "weather", "today", "current", "now", "this week", "price", "fee")
REFERS_BACK = (" there", " it ", " that place")


def needs_search(question: str) -> bool:
    low = question.lower()
    return any(w in low for w in SEARCH_WORDS)


def resolve_reference(question: str, memory: list) -> tuple:
    """If the question says 'there' and memory knows the last place discussed, fill it in. Returns (question, place or '')."""
    padded = " " + question.lower().rstrip("?.! ") + " "
    if not any(r in padded for r in REFERS_BACK):
        return question, ""
    for q, _a in reversed(memory):
        for p in PLACES:
            if p in q.lower():
                return re.sub(r"\bthere\b", "in " + p.title(), question, flags=re.I), p
    return question, ""


def react_turn(question: str, memory: list, memory_on: bool, max_results: int = 2) -> dict:
    """One turn of the ReAct agent. Returns the steps (Thought / Action / Observation / Final answer), model calls and the answer."""
    steps, calls = [], 0
    mem = memory if memory_on else []
    q, place = resolve_reference(question, mem)
    if q == question and any(r in " " + question.lower().rstrip("?.! ") + " " for r in REFERS_BACK):
        calls += 1
        steps.append(("Thought", "The question says 'there', but I have no record of an earlier place. I cannot search for 'there'."))
        steps.append(("Final answer", "Which place do you mean? (With memory off, the agent cannot see your earlier messages.)"))
        return {"steps": steps, "calls": calls, "answer": steps[-1][1], "searched": False, "resolved": question, "clarify": True}
    if place:
        steps.append(("Memory", "Chat history says the last place was %s, so the question becomes: \"%s\"" % (place.title(), q)))
    if needs_search(q):
        calls += 1
        hit = next((s for s in SEARCH_INDEX if all(k in q.lower() for k in s["keys"])), None)
        steps.append(("Thought", "This needs current information, so I will use the search tool instead of guessing."))
        steps.append(("Action", 'search("%s")  (max_results=%d)' % (hit["query"] if hit else q, max_results)))
        if hit:
            shown = hit["results"][:max_results]
            steps.append(("Observation", " | ".join("%s: %s" % r for r in shown)))
            calls += 1
            steps.append(("Final answer", "Based on the search: " + " ".join(r[1].replace("Sample result: ", "") for r in shown)))
        else:
            steps.append(("Observation", "No results found in the pretend search index."))
            calls += 1
            steps.append(("Final answer", "I searched but found nothing useful, so I will not guess."))
        return {"steps": steps, "calls": calls, "answer": steps[-1][1], "searched": True, "resolved": q, "clarify": False}
    calls += 1
    key = re.sub(r"[?.!]", "", q.lower()).strip()
    steps.append(("Thought", "I can answer this from what I already know; no search needed."))
    steps.append(("Final answer", KNOWLEDGE.get(key, "(Answered from the model's own knowledge. The demo only knows a few definitions.)")))
    return {"steps": steps, "calls": calls, "answer": steps[-1][1], "searched": False, "resolved": q, "clarify": False}


def react_chat(turns: list, memory_on: bool, max_results: int = 2) -> list:
    """Run a scripted conversation. Memory stores (question, answer) pairs only when memory is on."""
    memory, out = [], []
    for t in turns:
        r = react_turn(t, memory, memory_on, max_results)
        r["question"] = t
        out.append(r)
        if memory_on:
            memory.append((t, r["answer"]))
    return out


# ----------------------------------------------------------------------------------------
# Part 2: the support desk, built three ways
# ----------------------------------------------------------------------------------------
DOMAINS = ["product", "policy", "general"]
AGENT = {d: d + " agent" for d in DOMAINS}
FALLBACK = "I could not match this to a package or a policy, so a human colleague will reply within one working day."

FACTS = {
    ("product", "bali"): "Bali Honeymoon Escape: 6 nights (Ubud and Seminyak), vegetarian meals on request, Rs 1,45,000 for two.",
    ("product", "goa"): "Goa Weekend: 2 nights beachside, breakfast included, Rs 18,000 for two.",
    ("product", "kerala"): "Kerala Backwaters: 4 nights including a houseboat night, Rs 52,000 for two.",
    ("policy", "cancel"): "Cancellation: free up to 14 days before departure, a 20% fee from 14 to 7 days, and no refund inside 7 days.",
    ("policy", "refund"): "Refunds return to the original payment method in 7 to 10 working days.",
    ("policy", "insurance"): "Travel insurance is optional and bought separately; no package includes it.",
    ("policy", "reschedule"): "One free date change is allowed up to 10 days before departure.",
    ("general", "hours"): "Support is open 9 am to 9 pm IST, Monday to Saturday.",
}
TOPIC_WORDS = {
    ("product", "bali"): ["bali"],
    ("product", "goa"): ["goa"],
    ("product", "kerala"): ["kerala", "houseboat"],
    ("policy", "cancel"): ["cancel"],
    ("policy", "refund"): ["refund"],
    ("policy", "insurance"): ["insurance"],
    ("policy", "reschedule"): ["reschedule", "date change", "change the date"],
    ("general", "hours"): ["open on", "opening hours", "hours", "contact", "sunday", "saturday"],
}
QUERIES = [
    "What does the Bali package include, and what does it cost?",
    "Can I cancel for free, and how long does the refund take?",
    "Hi! Are you open on Sundays?",
    "I booked the Goa weekend. If I cancel 10 days before, what do I get back?",
    "Is travel insurance included in the Kerala houseboat trip?",
]
ODD_QUERY = "Do you arrange Himalayan treks for school groups?"


def name_of(topic) -> str:
    return "%s/%s" % topic


def detect_topics(query: str) -> list:
    low = query.lower()
    return [t for t, words in TOPIC_WORDS.items() if any(w in low for w in words)]


def primary_domain(topics: list, down=()) -> str:
    """The domain that owns the most topics in the query (ties: product, then policy). Falls back to general."""
    counts = {d: sum(1 for t in topics if t[0] == d) for d in DOMAINS}
    cands = [d for d in DOMAINS if counts[d] and d not in down]
    return min(cands, key=lambda d: (-counts[d], DOMAINS.index(d))) if cands else "general"


def _facts(topics) -> str:
    return " ".join(FACTS[t] for t in topics)


def _result(log: Log, topics: list, covered: list, reply: str, answered: bool, **extra) -> dict:
    complete = answered and all(t in covered for t in topics)
    return {"log": log, "calls": log.calls, "messages": len(log.messages), "needs": topics, "covered": covered,
            "reply": reply, "complete": complete, **extra}


def centralized(query: str, split: bool = False, down=()) -> dict:
    """Manager/worker: the orchestrator reads the query, hands it to worker agents and sends the reply."""
    log, topics = Log(), detect_topics(query)
    log.send("customer", "orchestrator", "query", query, "T1")
    if "orchestrator" in down:
        log.send("(network)", "customer", "timeout", "the orchestrator did not answer")
        return _result(log, topics, [], "No answer: the orchestrator is down, and every message goes through it.", False, mode="orchestrator down")
    log.call()                                   # the orchestrator classifies (and, if splitting, decomposes) the query
    targets = [primary_domain(topics)] if not split else (list(dict.fromkeys(t[0] for t in topics)) or ["general"])
    covered, parts = [], []
    for dom in targets:
        mine = [t for t in topics if t[0] == dom]
        log.send("orchestrator", AGENT[dom], "task", ("answer: " + ", ".join(name_of(t) for t in mine)) if mine else "answer the question", "T1")
        if dom in down:
            log.send("(network)", "orchestrator", "timeout", AGENT[dom] + " did not answer")
            continue
        log.call()
        text = _facts(mine) if mine else FALLBACK
        parts.append(text)
        covered += mine
        log.send(AGENT[dom], "orchestrator", "answer", text, "T1")
    if split and len(targets) > 1 and parts:
        log.call()                               # the orchestrator composes one reply from several answers
    reply = " ".join(parts) if parts else "No answer: the agent that owns this topic is down."
    log.send("orchestrator", "customer", "reply", reply, "T1")
    return _result(log, topics, covered, reply, bool(parts), mode="splits the query" if split else "one agent gets the whole query")


ROTATION = {"product": "policy", "policy": "general", "general": "product"}


def decentralized(query: str, entry: str = "general", max_hops: int = 4, general_forwards: bool = False, down=()) -> dict:
    """Peer/swarm: no orchestrator. Each agent answers what it owns and forwards the rest to a peer."""
    log, topics = Log(), detect_topics(query)
    remaining, covered, parts = list(topics), [], []
    hops, gave_up, current = 0, False, entry
    for _ in range(len(DOMAINS)):                # the customer can reach any agent: if the first is down, they retry another
        if current not in down:
            break
        log.send("customer", AGENT[current], "query", query, "T1")
        log.send("(network)", "customer", "timeout", AGENT[current] + " did not answer; the customer tries another agent")
        current = ROTATION[current]
    log.send("customer", AGENT[current], "query", query, "T1")
    answered = False
    while True:
        if current in down:
            log.send("(network)", "customer", "timeout", AGENT[current] + " did not answer")
            break
        log.call()
        mine = [t for t in remaining if t[0] == current]
        if mine:
            parts.append(_facts(mine))
            covered += mine
            remaining = [t for t in remaining if t not in mine]
        if not topics:
            done = current == "general" and not general_forwards
            if done:
                parts.append(FALLBACK)
        else:
            done = not remaining
        if done:
            answered = True
            log.send(AGENT[current], "customer", "reply", " ".join(parts), "T1")
            break
        if hops >= max_hops:
            gave_up = True
            log.send(AGENT[current], "customer", "gave up", "hop limit reached (%d forwards)" % max_hops, "T1")
            break
        nxt = remaining[0][0] if remaining else ROTATION[current]
        need = ("still needed: " + ", ".join(name_of(t) for t in remaining)) if remaining else "not sure who can answer this"
        log.send(AGENT[current], AGENT[nxt], "forward", need, "T1", direct=True)
        if nxt in down:
            log.send("(network)", AGENT[current], "timeout", AGENT[nxt] + " did not answer")
            parts.append("(I could not reach the %s for: %s.)" % (AGENT[nxt], ", ".join(name_of(t) for t in remaining)))
            answered = bool(covered)
            log.send(AGENT[current], "customer", "reply", " ".join(parts), "T1")
            break
        hops += 1
        current = nxt
    reply = " ".join(parts) if answered else ("No reply: the agents kept forwarding until the hop limit." if gave_up else "No answer: the agent that received the query is down.")
    return _result(log, topics, covered, reply, answered, hops=hops, gave_up=gave_up, wasted=log.calls if gave_up else 0,
                   mode="entry: " + AGENT[entry])


def hybrid(query: str, direct: bool = True, down=(), fallback: bool = True) -> dict:
    """Federated specialists: the orchestrator routes, but agents ask each other directly for the other half of a question."""
    topics = detect_topics(query)
    if "orchestrator" in down:
        if not fallback:
            log = Log()
            log.send("customer", "orchestrator", "query", query, "T1")
            log.send("(network)", "customer", "timeout", "the orchestrator did not answer")
            return _result(log, topics, [], "No answer: the orchestrator is down and no peer-to-peer fallback was built.", False, mode="orchestrator down, no fallback")
        r = decentralized(query, entry="general", down=down)
        r["mode"] = "orchestrator down: peers took over"
        return r
    log = Log()
    log.send("customer", "orchestrator", "query", query, "T1")
    log.call()                                   # routing
    first = primary_domain(topics, down)
    log.send("orchestrator", AGENT[first], "task", query, "T1")
    covered, parts = [], []
    if first in down:
        log.send("(network)", "orchestrator", "timeout", AGENT[first] + " did not answer")
        return _result(log, topics, [], "No answer: the first agent is down.", False, mode="routed to a dead agent")
    log.call()
    mine = [t for t in topics if t[0] == first]
    if mine:
        parts.append(_facts(mine))
        covered += mine
    elif not topics:
        parts.append(FALLBACK)
    for dom in dict.fromkeys(t[0] for t in topics if t[0] != first):
        need = [t for t in topics if t[0] == dom]
        if direct:
            log.send(AGENT[first], AGENT[dom], "ask", "please answer: " + ", ".join(name_of(t) for t in need), "T1", direct=True)
        else:
            log.send(AGENT[first], "orchestrator", "ask", "I need the %s for: %s" % (AGENT[dom], ", ".join(name_of(t) for t in need)), "T1")
            log.call()                           # the orchestrator decides where to send it
            log.send("orchestrator", AGENT[dom], "task", "please answer: " + ", ".join(name_of(t) for t in need), "T1")
        if dom in down:
            log.send("(network)", AGENT[first], "timeout", AGENT[dom] + " did not answer")
            parts.append("(I could not reach the %s for: %s.)" % (AGENT[dom], ", ".join(name_of(t) for t in need)))
            continue
        log.call()
        text = _facts(need)
        covered += need
        parts.append(text)
        log.send(AGENT[dom], AGENT[first] if direct else "orchestrator", "answer", text, "T1", direct=direct)
        if not direct:
            log.send("orchestrator", AGENT[first], "answer", text, "T1")
    reply = " ".join(parts)
    log.send(AGENT[first], "orchestrator", "answer", reply, "T1")
    log.send("orchestrator", "customer", "reply", reply, "T1")
    return _result(log, topics, covered, reply, True, mode="peer messages go direct" if direct else "peer messages go through the orchestrator")


def compare_designs(queries: list, entry: str = "general", down=(), direct: bool = True, fallback: bool = True) -> list:
    """Run the same questions through every design. One row per design."""
    makers = [("Centralized: one agent gets the query", "Central (one)", lambda q: centralized(q, False, down)),
              ("Centralized: orchestrator splits it", "Central (split)", lambda q: centralized(q, True, down)),
              ("Decentralized: peers forward", "Decentral", lambda q: decentralized(q, entry, 4, False, down)),
              ("Hybrid: route, then ask peers", "Hybrid", lambda q: hybrid(q, direct, down, fallback))]
    rows = []
    for name, short, fn in makers:
        res = [fn(q) for q in queries]
        rows.append({"Design": name, "_short": short, "Fully answered": "%d of %d" % (sum(r["complete"] for r in res), len(res)),
                     "Model calls": sum(r["calls"] for r in res), "Messages": sum(r["messages"] for r in res), "_results": res})
    return rows


# ----------------------------------------------------------------------------------------
# Part 3: a mini CrewAI (Agent, Task, Crew, Process) written from scratch
# ----------------------------------------------------------------------------------------
FACT_BANK = {
    "the future of electrical vehicles": ["Charging networks are expanding fastest in cities and along highways.",
                                          "Battery costs have been falling, which makes cars cheaper over time.",
                                          "Many makers now plan to sell mostly electric models in the coming years.",
                                          "Charging speed matters as much as driving range for everyday users.",
                                          "Second-hand batteries are being reused for home energy storage."],
    "vegetarian travel in bali": ["Many Ubud cafes list vegan and vegetarian dishes on the main menu.",
                                  "Temple-area food stalls often serve vegetable-based local dishes.",
                                  "Beach clubs may need a day's notice for vegetarian set menus.",
                                  "Cooking classes often include a vegetarian option.",
                                  "Rice, tempeh and tofu are common local staples."],
}


def sample_facts(topic: str) -> list:
    """PRETEND facts. Known topics come from a small bank; any other topic gets clearly labelled placeholders."""
    key = topic.lower().strip()
    if key in FACT_BANK:
        return FACT_BANK[key]
    return ["(sample) Fact %d about %s would be written here by a model in Live mode." % (i, topic) for i in range(1, 6)]


DEFAULT_CREW = {
    "agents": {
        "researcher": {"role": "Research Specialist", "goal": "Research interesting facts about the topic: {topic}",
                       "backstory": "You are an expert at finding relevant and factual data."},
        "writer": {"role": "Creative Writer", "goal": "Write a short blog summary using the research",
                   "backstory": "You are skilled at writing engaging summaries based on provided content."},
    },
    "tasks": [
        {"id": "task1", "description": "Find 3-5 interesting and recent facts about {topic}.", "expected_output": "A bullet list of 3-5 facts",
         "agent": "researcher", "context": []},
        {"id": "task2", "description": "Write a 100-word blog post summary about {topic} using the facts from the research.",
         "expected_output": "A blog post summary", "agent": "writer", "context": ["task1"]},
    ],
}


def placeholders(text: str) -> list:
    return re.findall(r"\{(\w+)\}", text)


def fill(text: str, inputs: dict) -> str:
    return re.sub(r"\{(\w+)\}", lambda m: str(inputs[m.group(1)]) if m.group(1) in inputs else m.group(0), text)


def missing_inputs(cfg: dict, inputs: dict) -> list:
    names = []
    for a in cfg["agents"].values():
        names += placeholders(a["goal"]) + placeholders(a["backstory"]) + placeholders(a["role"])
    for t in cfg["tasks"]:
        names += placeholders(t["description"]) + placeholders(t["expected_output"])
    return sorted({n for n in names if n not in inputs})


def skill_of_agent(agent: dict) -> str:
    """Guess what an agent does from its role first, then its goal (the role is the stronger signal)."""
    role, goal = agent["role"].lower(), agent["goal"].lower()
    for text in (role, goal):
        if any(w in text for w in ("writ", "blog", "edit")):
            return "write"
        if any(w in text for w in ("research", "investigat", "find ")):
            return "research"
    return "general"


def skill_of_task(task: dict) -> str:
    text = task["description"].lower()
    if any(w in text for w in ("find ", "research", "facts about", "investigat")) and not any(w in text for w in ("write", "blog")):
        return "research"
    if any(w in text for w in ("write", "blog", "summary", "draft")):
        return "write"
    return "general"


def bullets(text: str) -> list:
    return [ln[2:].strip() for ln in text.splitlines() if ln.startswith("- ")]


def meets_expected(task: dict, output: str) -> tuple:
    """Check the output against the task's expected_output, e.g. 'A bullet list of 3-5 facts'."""
    m = re.search(r"(\d+)\s*-\s*(\d+)", task["expected_output"])
    if m and "bullet" in task["expected_output"].lower():
        lo, hi, n = int(m.group(1)), int(m.group(2)), len(bullets(output))
        return (lo <= n <= hi), "%d bullet(s), expected %d to %d" % (n, lo, hi)
    return bool(output.strip()), "non-empty" if output.strip() else "empty"


def do_task(task: dict, agent: dict, topic: str, context_outputs: list, sloppy: bool, backend: Backend | None = None) -> str:
    """What an agent 'produces'. Demo: scripted by skill. Live: Gemini writes it from the same role, goal and context."""
    skill = skill_of_agent(agent)
    desc = task["description"]
    if skill == "research":
        n = 2 if sloppy else 4
        demo = lambda: "\n".join("- " + f for f in sample_facts(topic)[:n])
    elif skill == "write":
        facts = [b for o in context_outputs for b in bullets(o)]
        if not facts:
            demo = lambda: "[no research was passed to this task] I have nothing to base a summary on. A real model would probably invent facts here."
        else:
            demo = lambda: ("%s: " % topic.capitalize()) + " ".join(f.rstrip(".") + "." for f in facts) + " (demo text built from the research bullets.)"
    else:
        demo = lambda: "Done: " + desc
    if backend is None:
        return demo()
    system = "Role: %s. Goal: %s. Backstory: %s" % (agent["role"], agent["goal"], agent["backstory"])
    request = desc + "\nExpected output: " + task["expected_output"] + ("\n\nContext from earlier tasks:\n" + "\n\n".join(context_outputs) if context_outputs else "")
    return backend.ask(system, request, demo)


def run_crew(cfg: dict, inputs: dict, process: str = "sequential", manager: dict | None = None, sloppy: bool = False,
             use_context: bool = True, backend: Backend | None = None) -> dict:
    """kickoff(): run the crew. Returns outputs, a message log, the model-call count and any error the framework would raise."""
    log = Log()
    res = {"error": "", "outputs": {}, "final": "", "log": log, "calls": 0, "checks": [], "reworked": []}
    miss = missing_inputs(cfg, inputs)
    if miss:
        res["error"] = "kickoff(inputs=...) has no value for: " + ", ".join("{%s}" % m for m in miss) + ". The text would be sent to the model with the placeholder unfilled."
        return res
    if process == "hierarchical" and not manager:
        res["error"] = "A hierarchical process needs a manager: give the crew a manager_llm or a manager_agent."
        return res
    order = [t["id"] for t in cfg["tasks"]]
    for i, t in enumerate(cfg["tasks"]):
        for dep in t["context"] if use_context else []:
            if dep not in order:
                res["error"] = "%s uses context=[%s], but there is no task called %s." % (t["id"], dep, dep)
                return res
            if order.index(dep) >= i:
                res["error"] = "%s needs the output of %s, but %s runs later. Order the tasks so dependencies come first." % (t["id"], dep, dep)
                return res
    topic = str(inputs.get("topic", "the topic"))
    outputs = res["outputs"]
    log.send("you", "crew", "kickoff", str(inputs))
    if process == "hierarchical":
        log.call()
        log.send("manager", "manager", "plan", "split the work into %d task(s) and assign each to the best-fitting agent" % len(cfg["tasks"]))
    for t in cfg["tasks"]:
        if process == "hierarchical":
            want = skill_of_task(t)
            owner = next((k for k, a in cfg["agents"].items() if skill_of_agent(a) == want), t["agent"])
            log.send("manager", owner, "delegate", fill(t["description"], inputs))
        else:
            owner = t["agent"]
            log.send("crew", owner, "task", fill(t["description"], inputs))
        agent = cfg["agents"][owner]
        ctx = [outputs[d] for d in (t["context"] if use_context else []) if d in outputs]
        log.call()
        out = do_task({**t, "description": fill(t["description"], inputs)}, agent, topic, ctx, sloppy, backend)
        log.send(owner, "manager" if process == "hierarchical" else "crew", "result", out.replace("\n", " ")[:160])
        if process == "hierarchical" and manager and manager.get("kind") == "agent" and manager.get("has_tools"):
            log.call()
            ok, why = meets_expected(t, out)
            res["checks"].append({"task": t["id"], "ok": ok, "why": why})
            log.send("manager", owner, "review", ("accepted: " if ok else "REJECTED: ") + why)
            if not ok:
                log.call()
                out = do_task({**t, "description": fill(t["description"], inputs)}, agent, topic, ctx, False, backend)
                res["reworked"].append(t["id"])
                log.send(owner, "manager", "result (redo)", out.replace("\n", " ")[:160])
                log.call()
                ok2, why2 = meets_expected(t, out)
                res["checks"].append({"task": t["id"] + " (redo)", "ok": ok2, "why": why2})
                log.send("manager", owner, "review", ("accepted: " if ok2 else "REJECTED: ") + why2)
        elif process == "hierarchical":
            ok, why = meets_expected(t, out)
            res["checks"].append({"task": t["id"], "ok": ok, "why": "not checked by the manager (" + why + ")"})
            log.send("manager", owner, "review", "accepted without checking")
        outputs[t["id"]] = out
    if process == "hierarchical":
        log.call()
        log.send("manager", "you", "final", "assemble the finished result")
    res["final"] = outputs[cfg["tasks"][-1]["id"]]
    res["calls"] = log.calls
    return res


def crew_code(cfg: dict, inputs: dict, process: str = "sequential", manager: dict | None = None) -> str:
    """The CrewAI code (as in the notebook) that this configuration corresponds to."""
    lines = ["from crewai import Agent, Task, Crew, Process, LLM", "", 'llm = LLM(model="gemini/gemini-2.5-flash", temperature=0.1)', ""]
    for k, a in cfg["agents"].items():
        lines += ["%s_agent = Agent(" % k, "    role=%r," % a["role"], "    goal=%r," % a["goal"], "    backstory=%r," % a["backstory"], "    llm=llm,", "    verbose=True,", ")", ""]
    for t in cfg["tasks"]:
        lines += ["%s = Task(" % t["id"], "    description=%r," % t["description"], "    expected_output=%r," % t["expected_output"], "    agent=%s_agent," % t["agent"]]
        if t["context"]:
            lines.append("    context=[%s]," % ", ".join(t["context"]))
        lines += [")", ""]
    lines += ["crew = Crew(", "    agents=[%s]," % ", ".join(k + "_agent" for k in cfg["agents"]), "    tasks=[%s]," % ", ".join(t["id"] for t in cfg["tasks"])]
    if process == "hierarchical":
        lines.append("    process=Process.hierarchical,")
        lines.append("    manager_agent=manager_agent,   # a full Agent with role, goal and tools" if manager and manager.get("kind") == "agent" else '    manager_llm="gemini/gemini-2.5-flash",   # just a model, no role or tools')
    else:
        lines.append("    process=Process.sequential,")
    lines += ["    verbose=True,", ")", "", "crew.kickoff(inputs=%r)" % inputs]
    return "\n".join(lines)


def process_costs(cfg: dict, inputs: dict, sloppy: bool) -> list:
    rows = []
    for label, proc, mgr in (("Sequential (no manager)", "sequential", None),
                             ("Hierarchical, manager_llm", "hierarchical", {"kind": "llm"}),
                             ("Hierarchical, manager_agent with a check tool", "hierarchical", {"kind": "agent", "has_tools": True})):
        r = run_crew(cfg, inputs, proc, mgr, sloppy)
        if r["error"]:
            rows.append({"Process": label, "Model calls": "-", "Facts the writer received": "-", "Short research caught?": r["error"][:60]})
            continue
        first = r["outputs"].get(cfg["tasks"][0]["id"], "")
        rows.append({"Process": label, "Model calls": r["calls"], "Facts the writer received": len(bullets(first)),
                     "Short research caught?": "yes, redone" if r["reworked"] else ("no" if sloppy else "n/a (research was fine)")})
    return rows


# ----------------------------------------------------------------------------------------
# Part 4: LangChain vs CrewAI (the notebook's comparison table) and patterns vs architectures
# ----------------------------------------------------------------------------------------
FRAMEWORK_TABLE = [
    {"Need": "Orchestration of several agents", "LangChain (by hand)": "You coordinate agents and pass outputs yourself", "CrewAI": "The Crew object assigns tasks and runs the workflow", "by_hand": True},
    {"Need": "Roles for agents", "LangChain (by hand)": "Roles are written into prompts by you", "CrewAI": "role, goal and backstory are fields every Agent has", "by_hand": True},
    {"Need": "Memory shared between agents", "LangChain (by hand)": "Manual memory chaining; sharing needs extra code", "CrewAI": "Short-term, long-term and shared memory are built in", "by_hand": True},
    {"Need": "Tools per agent", "LangChain (by hand)": "You wire each tool into each agent", "CrewAI": "Tools attach to an agent like plugins", "by_hand": True},
    {"Need": "Handing results from one step to the next", "LangChain (by hand)": "Custom code passes outputs between chains", "CrewAI": "Task context passes earlier outputs on", "by_hand": True},
    {"Need": "Scaling from a few agents to many", "LangChain (by hand)": "Needs more custom logic as the team grows", "CrewAI": "Built for fast prototyping and larger crews", "by_hand": True},
    {"Need": "One agent with a tool and a chat memory", "LangChain (by hand)": "A few lines: this is what LangChain is for", "CrewAI": "Works, but more machinery than you need", "by_hand": False},
]

PATTERN_TABLE = [
    {"Pattern (how agents interact)": "Manager / worker", "Description": "A central agent delegates to task-specific workers", "Best architecture": "Centralized or hybrid", "Why it fits": "Clear hierarchy and control", "In this session": "Page 2; CrewAI hierarchical process (page 9)"},
    {"Pattern (how agents interact)": "Peer / swarm", "Description": "Agents coordinate with local rules", "Best architecture": "Decentralized", "Why it fits": "Fault-tolerant and scalable", "In this session": "Page 3"},
    {"Pattern (how agents interact)": "Pipeline", "Description": "Sequential hand-off of tasks", "Best architecture": "Centralized or hybrid", "Why it fits": "Needs orchestration to enforce the order", "In this session": "CrewAI sequential process (page 8)"},
    {"Pattern (how agents interact)": "Specialists / federated teams", "Description": "Agents with deep domain expertise", "Best architecture": "Hybrid", "Why it fits": "Efficient handling of complex tasks", "In this session": "Page 4"},
]
