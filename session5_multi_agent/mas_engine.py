"""
Engine for the Multi-Agent Systems lab (Agents, Session 5).

Everything here is plain Python so the labs run with NO API key. Each design returns what it did (messages, model calls, results),
because the point of the session is that every extra agent costs extra messages and extra model calls.
All data (guides, prices, bookings) is PRETEND data written for this exercise, never live information.
"""
from __future__ import annotations

import json
import random
import re
from itertools import combinations


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


# ----------------------------------------------------------------------------------------
# Messages: the communication layer every demo uses
# ----------------------------------------------------------------------------------------
REQUIRED_FIELDS = ("from", "to", "kind", "task_id", "content")


class Log:
    """A message log plus a model-call counter, so each design reports who said what to whom and what it cost."""

    def __init__(self):
        self.messages: list = []
        self.calls = 0

    def send(self, sender, recipient, kind, content, task_id="", direct=False):
        m = {"from": sender, "to": recipient, "kind": kind, "task_id": task_id, "content": content, "direct": direct}
        self.messages.append(m)
        return m

    def call(self, n: int = 1):
        self.calls += n

    def rows(self) -> list:
        return [{"#": i + 1, "from": m["from"], "to": m["to"], "kind": m["kind"], "task": m["task_id"],
                 "content": m["content"][:110], "direct": "✅ peer to peer" if m["direct"] else ""} for i, m in enumerate(self.messages)]


def validate_message(text: str) -> dict:
    """Can another agent (or plain code) read this message? Structured JSON can be checked; free text cannot."""
    try:
        data = json.loads(text)
    except Exception:
        return {"ok": False, "why": "Not valid JSON: a receiving agent has to guess who sent it, who it is for and which task it belongs to."}
    if not isinstance(data, dict):
        return {"ok": False, "why": "JSON, but not an object with fields."}
    missing = [f for f in REQUIRED_FIELDS if f not in data or str(data[f]).strip() == ""]
    if missing:
        return {"ok": False, "why": "Missing field(s): " + ", ".join(missing) + ". Without them it cannot be routed or matched to its task."}
    return {"ok": True, "why": "All five fields present: routable, loggable, and checkable by code.", "data": data}


# ----------------------------------------------------------------------------------------
# Chapter 1: generalist versus specialists (the honeymoon request)
# ----------------------------------------------------------------------------------------
HONEYMOON_REQUEST = ("We're an Indian couple getting married in February. Plan our honeymoon in Bali: what visas we need, "
                     "whether we need travel insurance, a budget under Rs 2 lakh for both of us, and a 6-day vegetarian-friendly itinerary.")
TOPICS = ["visa", "insurance", "budget", "itinerary"]

GENERALIST = {
    "visa": "You will need a visa for Bali, so check the requirements before you travel.",
    "insurance": "Travel insurance is worth considering.",
    "budget": "Bali can be done within a moderate budget if you plan well.",
    "itinerary": "Spend your six days across Ubud for culture and food, a beach area for rest, and one day for a temple trip. Choose vegetarian cafes in Ubud.",
}
SPECIALIST = {
    "visa": ("Indian passport holders can usually enter Bali on a visa on arrival. Your passport should be valid for at least six months from arrival, with a return "
             "or onward ticket. Keep digital and printed copies of both. Fees and rules change, so verify the current fee and entry conditions on the official "
             "Indonesian immigration website before you book."),
    "insurance": ("Yes, buy insurance: medical cover abroad is the part you cannot easily self-fund. Pick a policy covering medical care, trip cancellation (useful with "
                  "a wedding date) and scooter-related accidents if you plan to ride. Read the exclusions for adventure activities."),
    "budget": ("For two people, a realistic plan is about Rs 1.4 lakh: return flights Rs 36,000 to 56,000, six nights in mid-range stays Rs 60,000, food Rs 24,000, "
               "local transport Rs 12,000, insurance and visa about Rs 6,000. That leaves a buffer under your Rs 2 lakh limit. Treat these as rough estimates and check live prices."),
    "itinerary": ("Day 1: arrive, rest in Ubud. Day 2: Ubud rice terraces and vegetarian cafes. Day 3: temple and waterfall circuit. Day 4: transfer to a quiet beach. "
                  "Day 5: spa and sunset dinner. Day 6: slow morning, souvenirs, flight home. Ask for vegetarian meals in advance at beachside places."),
}
# Which worker owns which topic when we use three workers (insurance and budget share one)
WORKER_FOR = {"visa": "visa_worker", "insurance": "budget_worker", "budget": "budget_worker", "itinerary": "itinerary_worker"}


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


def depth_by_topic(generalist_down: bool = False, specialist_down: str = "") -> dict:
    """Words spent on each topic by one generalist and by a team of four specialists (one may be down)."""
    gen = {t: (0 if generalist_down else word_count(GENERALIST[t])) for t in TOPICS}
    team = {t: (0 if t == specialist_down else word_count(SPECIALIST[t])) for t in TOPICS}
    return {"generalist": gen, "specialists": team}


# ----------------------------------------------------------------------------------------
# Chapter 2: single agent or MAS? (complexity, distribution, dynamics)
# ----------------------------------------------------------------------------------------
SCENARIOS = [
    {"name": "A chatbot that gives bus times for one city", "complex": False, "distributed": False, "dynamic": False,
     "why": "Small, one data source, a stable timetable."},
    {"name": "A thermostat controlling one room", "complex": False, "distributed": False, "dynamic": False,
     "why": "One sensor, one actuator, one goal."},
    {"name": "An LLM answering basic factual questions", "complex": False, "distributed": False, "dynamic": False,
     "why": "Self-contained knowledge, no changing environment."},
    {"name": "A smart power grid with thousands of generators and homes", "complex": True, "distributed": True, "dynamic": True,
     "why": "Huge, spread over many sites, and demand and supply change every second."},
    {"name": "City-wide traffic management", "complex": True, "distributed": True, "dynamic": True,
     "why": "Accidents, roadworks and crowds change by the minute at thousands of junctions."},
    {"name": "Healthcare, finance and travel LLMs planning one medical trip abroad", "complex": True, "distributed": True, "dynamic": False,
     "why": "Several domains with separate data and tools; the task itself is multi-step but not changing minute to minute."},
]


def mas_verdict(complex_: bool, distributed: bool, dynamic: bool) -> str:
    n = sum([complex_, distributed, dynamic])
    return "Multi-agent system" if n >= 2 else "Single agent"


# ----------------------------------------------------------------------------------------
# Chapter 3a: centralized architecture (a coordinator routes every query)
# ----------------------------------------------------------------------------------------
SPECIALISTS = {
    "bookings_bot": "bookings",
    "refunds_bot": "refunds",
    "advice_bot": "advice",
}
QUERIES = [
    ("Q1", "Where is my booking confirmation for Snow Valley Cottage?", "bookings_bot"),
    ("Q2", "I want a refund for my cancelled Goa trip.", "refunds_bot"),
    ("Q3", "Is it safe to drive to Manali when it's snowing?", "advice_bot"),
    ("Q4", "Can you suggest vegetarian restaurants in Old Manali?", "advice_bot"),
]
BOT_REPLIES = {
    "bookings_bot": "I have found your booking and re-sent the confirmation to your email.",
    "refunds_bot": "Your refund has been raised and should arrive in 5 to 7 working days.",
    "advice_bot": "Carry chains, drive in daylight, and check the road status first; for food, try the cafes along the Old Manali road.",
}


def coordinator_route(query: str) -> str:
    """The coordinator's routing rule (in the notebook an LLM does this with structured output)."""
    q = query.lower()
    if "refund" in q or "cancel" in q:
        return "refunds_bot"
    if "booking" in q or "confirmation" in q or "reservation" in q:
        return "bookings_bot"
    return "advice_bot"


def centralized_desk(coordinator_up: bool = True, fallback: bool = False) -> dict:
    """Run the four queries through a central coordinator. With `fallback`, specialists take queries in turn when it is down."""
    log = Log()
    for qid, text, _ in QUERIES:
        log.send("customer", "coordinator", "query", text, qid)
    answered, misrouted = 0, 0
    bots = list(SPECIALISTS)
    if coordinator_up:
        log.call(1)            # one structured routing call for the whole batch
        for qid, text, truth in QUERIES:
            bot = coordinator_route(text)
            log.send("coordinator", bot, "assign", text, qid)
            log.call(1)
            log.send(bot, "customer", "answer", BOT_REPLIES[bot], qid)
            answered += 1
            misrouted += bot != truth
    elif fallback:
        for i, (qid, text, truth) in enumerate(QUERIES):          # no coordinator: each query goes to whoever is next
            bot = bots[i % len(bots)]
            log.send(bot, "customer", "answer", BOT_REPLIES[bot], qid)
            log.call(1)
            answered += 1
            misrouted += bot != truth
    return {"log": log, "answered": answered, "total": len(QUERIES), "misrouted": misrouted, "calls": log.calls}


def desk_load(n_queries: int, centralized: bool, route_per_min: int = 6, bots: int = 3, answer_per_min: int = 3) -> dict:
    """A toy queueing model. The coordinator routes `route_per_min` queries a minute; each bot answers `answer_per_min`."""
    answer_time = n_queries / (bots * answer_per_min)
    if centralized:
        route_time = n_queries / route_per_min
        return {"minutes_to_clear": round(max(route_time, answer_time) + 1 / route_per_min, 1), "bottleneck": "the coordinator" if route_time > answer_time else "the specialists"}
    return {"minutes_to_clear": round(answer_time + 0.3, 1), "bottleneck": "the specialists"}


# ----------------------------------------------------------------------------------------
# Chapter 3b: decentralized architecture (peers decide locally; a status board may be stale)
# ----------------------------------------------------------------------------------------
ACTUAL = {"bot_A": 1, "bot_B": 4, "bot_C": 3, "bot_D": 0}
STALE_BOARD = {"bot_A": 1, "bot_B": 4, "bot_C": 0, "bot_D": 0}
REFUSE_AT = 3          # a peer with this many queries (or more) refuses a handover


def handover(actual: dict, board: dict, me: str = "bot_B", query: str = "VIP customer: our flight was cancelled, please rebook us today.",
             max_attempts: int = 3, refuse_at: int = REFUSE_AT) -> dict:
    """bot `me` is overloaded. It trusts the board, hands the query to the least-loaded peer, and may be refused."""
    log = Log()
    board = dict(board)
    refused: list = []
    log.send("customer", me, "query", query, "Q9")
    handled_by, attempts = None, 0
    while handled_by is None and attempts < max_attempts:
        attempts += 1
        log.call(1)       # each local decision is one model call
        options = sorted((v, k) for k, v in board.items() if k != me and k not in refused)
        if not options or board[me] <= options[0][0] or options[0][0] >= refuse_at:
            log.send(me, "customer", "answer", "I'll handle this myself, though I'm busy.", "Q9")
            handled_by = me
            break
        peer = options[0][1]
        log.send(me, peer, "handover", query, "Q9", direct=True)
        if actual[peer] >= refuse_at:
            log.send(peer, me, "refuse", "Sorry, I actually have %d queries." % actual[peer], "Q9", direct=True)
            board[peer] = actual[peer]
            refused.append(peer)
        else:
            log.send(peer, me, "accept", "I'll take it.", "Q9", direct=True)
            handled_by = peer
    if handled_by is None:
        log.send(me, "customer", "answer", "Sorry for the wait: I'm handling this myself now.", "Q9")
        handled_by = me
    return {"log": log, "attempts": attempts, "handled_by": handled_by, "refused": refused, "calls": log.calls}


def broadcast_cost(n_agents: int) -> dict:
    """Messages caused by ONE change in one agent's queue, under three ways of keeping everyone informed."""
    return {"everyone tells everyone": n_agents * (n_agents - 1),
            "shared board (post once)": n_agents,
            "hybrid (report to coordinator)": n_agents}


def handover_study(n_bots: int, staleness: int, n_queries: int = 300, seed: int = 7) -> dict:
    """Average hand-overs per VIP query when boards are out of date by up to `staleness` queries."""
    rng = random.Random(seed)
    hops_total, msgs_total = 0, 0
    for _ in range(n_queries):
        names = ["bot_%d" % i for i in range(n_bots)]
        actual = {n: rng.randint(0, 6) for n in names}
        actual[names[0]] = 5
        board = {n: max(0, v + rng.randint(-staleness, staleness)) for n, v in actual.items()}
        board[names[0]] = actual[names[0]]
        res = handover(actual, board, me=names[0], max_attempts=5)
        hops_total += res["attempts"]
        msgs_total += len(res["log"].messages)
    return {"avg_attempts": round(hops_total / n_queries, 2), "avg_messages": round(msgs_total / n_queries, 2)}


def duplicate_work(claims: bool, n_queries: int = 5) -> dict:
    """Two chatbots watch the same queue. Without a claim message both may answer the same customer."""
    answers, claim_msgs = 0, 0
    for _ in range(n_queries):
        if claims:
            claim_msgs += 1       # the first bot posts "I take it"; the second sees it and skips
            answers += 1
        else:
            answers += 2          # both saw it at the same moment and answered
    return {"answers_sent": answers, "queries": n_queries, "duplicates": answers - n_queries, "claim_messages": claim_msgs}


# ----------------------------------------------------------------------------------------
# Chapter 3c: hybrid architecture (central plan and synthesis, peer to peer in the middle)
# ----------------------------------------------------------------------------------------
GOA_BOOKING = {"stay": "Palm Grove Homestay, Goa", "dates": "14-17 Feb", "paid": 12000, "cancellation_fee_rate": 0.20}
COORG_OPTIONS = [
    {"stay": "Coffee Estate Cottage, Coorg", "price": 8800, "notes": "quiet, veg kitchen"},
    {"stay": "Misty Hills Resort, Coorg", "price": 13500, "notes": "spa, mixed menu"},
]
HYBRID_STAGES = [
    ("1 · Task intake", "Coordinator", "Receives the request and works out the intent"),
    ("2 · Task planning", "Coordinator", "Splits it into a cancellation task and a booking task"),
    ("3 · Delegation", "Coordinator", "Sends each task to the right specialist"),
    ("4 · Execution", "Specialists", "Each works on its own, with its own tools"),
    ("5 · Collaboration", "Specialists", "Booking agent asks cancellation agent for the credit, directly"),
    ("6 · Reporting", "Specialists", "Each sends its result back to the coordinator"),
    ("7 · Synthesis", "Coordinator", "Combines the reports into one answer"),
    ("8 · Delivery", "Coordinator", "Presents the final answer to the couple"),
]


def refund_due(booking: dict) -> dict:
    fee = booking["paid"] * booking["cancellation_fee_rate"]
    return {"paid": booking["paid"], "fee": round(fee), "refund": round(booking["paid"] - fee)}


def hybrid_rebooking(fee_rate: float = 0.20, direct: bool = True, backend: Backend | None = None) -> dict:
    """Cancel one stay and rebook another using the refund. The refund is arithmetic done by CODE, never by an LLM."""
    booking = dict(GOA_BOOKING, cancellation_fee_rate=fee_rate)
    log = Log()
    request = "Please cancel our Goa homestay for 14-17 Feb and book something quiet in Coorg for the same dates instead, using the refund."
    log.send("user", "request_manager", "request", request, "H1")
    log.call(1)                                                 # planning
    log.send("request_manager", "cancellation_agent", "assign", "Cancel the Goa booking and work out the refund", "H1")
    log.send("request_manager", "booking_agent", "assign", "Book something quiet in Coorg for 14-17 Feb using the refund", "H1")
    money = refund_due(booking)
    log.call(1)                                                 # cancellation agent's report
    cancel_report = "Cancelled %s. Rs %d paid, %d%% fee (Rs %d), refund Rs %d." % (booking["stay"], money["paid"], round(fee_rate * 100), money["fee"], money["refund"])
    if direct:
        log.send("booking_agent", "cancellation_agent", "question", "How much refund credit can I use?", "H1", direct=True)
        log.send("cancellation_agent", "booking_agent", "answer", "Rs %d of credit." % money["refund"], "H1", direct=True)
    else:
        log.send("booking_agent", "request_manager", "question", "How much refund credit can I use?", "H1")
        log.call(1)
        log.send("request_manager", "cancellation_agent", "question", "How much refund credit?", "H1")
        log.send("cancellation_agent", "request_manager", "answer", "Rs %d of credit." % money["refund"], "H1")
        log.call(1)
        log.send("request_manager", "booking_agent", "answer", "Rs %d of credit." % money["refund"], "H1")
    quiet = [o for o in COORG_OPTIONS if "quiet" in o["notes"]]
    pick = min(quiet or COORG_OPTIONS, key=lambda o: abs(o["price"] - money["refund"]))
    to_pay = max(0, pick["price"] - money["refund"])
    log.call(1)                                                 # booking agent's report
    book_report = "Chose %s at Rs %d. Credit Rs %d, still to pay Rs %d." % (pick["stay"], pick["price"], money["refund"], to_pay)
    log.send("cancellation_agent", "request_manager", "report", cancel_report, "H1")
    log.send("booking_agent", "request_manager", "report", book_report, "H1")
    log.call(1)                                                 # synthesis
    if to_pay > 0:
        tail = "so you pay Rs %d more" % to_pay
    else:
        tail = "and Rs %d of your refund is left over" % (money["refund"] - pick["price"])
    final = "Your Goa stay is cancelled with a refund of Rs %d. We booked %s for Rs %d, %s." % (money["refund"], pick["stay"], pick["price"], tail)
    if backend is not None:
        final = backend.ask("You are the request manager. Combine your agents' reports into one short, friendly answer for the couple.",
                            "Request: %s\nCancellation: %s\nBooking: %s" % (request, cancel_report, book_report), final)
    log.send("request_manager", "user", "answer", final, "H1")
    return {"log": log, "money": money, "pick": pick, "to_pay": to_pay, "final": final, "calls": log.calls,
            "direct_messages": sum(m["direct"] for m in log.messages)}


ARCH_COMPARE = [
    {"Aspect": "Control", "Centralized": "One controller manages all agents", "Decentralized": "No central control; agents act autonomously", "Hybrid": "Central oversight plus agent autonomy"},
    {"Aspect": "Decision-making", "Centralized": "Controller decides and assigns", "Decentralized": "Each agent decides for itself", "Hybrid": "Coordinator plans globally; agents execute locally"},
    {"Aspect": "Scalability", "Centralized": "Limited: the controller is a bottleneck", "Decentralized": "High: agents scale independently", "Hybrid": "Moderate to high"},
    {"Aspect": "Reliability", "Centralized": "Low: single point of failure", "Decentralized": "High: no central node to fail", "Hybrid": "Higher: agents plus a central fallback"},
    {"Aspect": "Speed of response", "Centralized": "Slower: depends on the controller", "Decentralized": "Faster: local decisions", "Hybrid": "Balanced"},
    {"Aspect": "Communication", "Centralized": "Everything flows through the controller", "Decentralized": "Peer to peer", "Hybrid": "Direct between agents, aggregated centrally"},
    {"Aspect": "Best for", "Centralized": "Small teams, simple workflows, strong oversight", "Decentralized": "Large, dynamic, distributed systems", "Hybrid": "Global strategy plus local autonomy (LLM support, healthcare)"},
]

DESIGN_SCENARIOS = [
    {"name": "DilSeTrips during the Goa Carnival, when support queries jump tenfold for three days",
     "arch": "Hybrid", "pattern": "Manager / worker",
     "why": "A thin coordinator only routes and watches load; many identical workers answer independently, so you can add workers without touching the manager.",
     "risk": "The coordinator becomes the bottleneck if it does more than route."},
    {"name": "A legal-document review service where every answer must be approved by a lawyer before it is sent",
     "arch": "Centralized", "pattern": "Sequential pipeline",
     "why": "Order and approval matter more than speed: draft, check, then a human gate. One controller enforces the rules uniformly.",
     "risk": "An error early in the pipeline flows to every later stage, and the human gate can become the slow point."},
    {"name": "Monitoring 5,000 homestays' live availability across India",
     "arch": "Decentralized", "pattern": "Peer / swarm",
     "why": "Data is spread over thousands of sites and changes constantly; one small agent per homestay reports to a shared board, with no central brain to overwhelm.",
     "risk": "Stale or inconsistent global knowledge: the board may be out of date when someone reads it."},
    {"name": "A research assistant that must read 300 travel blogs about Ladakh and produce one guide",
     "arch": "Hybrid", "pattern": "Federated teams",
     "why": "Group agents by topic (routes, food, permits); teams work in parallel on their own sources, then one synthesiser merges reports in a shared format.",
     "risk": "Integration: if teams use different formats or overlap, the merged guide contradicts or double-counts itself."},
]


# ----------------------------------------------------------------------------------------
# Chapter 4a: manager / worker (and the requirement the manager forgot)
# ----------------------------------------------------------------------------------------
REQUIREMENTS = ["visa", "insurance", "budget", "itinerary"]
PLANS = {
    "Complete plan": {
        "tasks": [("visa_worker", "Work out the visa and documents for an Indian couple going to Bali."),
                  ("budget_worker", "Estimate a budget under Rs 2 lakh and say whether travel insurance is needed."),
                  ("itinerary_worker", "Outline a 6-day vegetarian-friendly itinerary.")],
        "coverage": {"visa": "visa_worker", "insurance": "budget_worker", "budget": "budget_worker", "itinerary": "itinerary_worker"}},
    "Forgets insurance": {
        "tasks": [("visa_worker", "Work out the visa and documents for an Indian couple going to Bali."),
                  ("budget_worker", "Estimate a budget under Rs 2 lakh."),
                  ("itinerary_worker", "Outline a 6-day vegetarian-friendly itinerary.")],
        "coverage": {"visa": "visa_worker", "budget": "budget_worker", "itinerary": "itinerary_worker"}},
    "Mentions insurance but assigns it to nobody": {
        "tasks": [("visa_worker", "Work out the visa and documents for an Indian couple going to Bali."),
                  ("budget_worker", "Estimate a budget under Rs 2 lakh; leave insurance aside for now."),
                  ("itinerary_worker", "Outline a 6-day vegetarian-friendly itinerary.")],
        "coverage": {"visa": "visa_worker", "budget": "budget_worker", "itinerary": "itinerary_worker"}},
}
WORKER_OUTPUT = {
    "visa_worker": SPECIALIST["visa"],
    "budget_worker": SPECIALIST["budget"] + " " + SPECIALIST["insurance"],
    "itinerary_worker": SPECIALIST["itinerary"],
}


def keyword_check(plan_name: str) -> list:
    """The notebook's check: is each requirement WORD present in some task text? Returns the requirements it flags."""
    text = " ".join(t for _, t in PLANS[plan_name]["tasks"]).lower()
    return [r for r in REQUIREMENTS if r not in text]


def coverage_check(plan_name: str) -> list:
    """HW3's check: the manager must say which worker covers each requirement; code rejects any that has none."""
    cov = PLANS[plan_name]["coverage"]
    return [r for r in REQUIREMENTS if not cov.get(r)]


def manager_worker(plan_name: str, strict_check: bool = False) -> dict:
    log = Log()
    log.send("user", "manager", "request", HONEYMOON_REQUEST, "M1")
    log.call(1)                                                # the manager's plan
    plan = PLANS[plan_name]
    flagged = coverage_check(plan_name) if strict_check else keyword_check(plan_name)
    results = {}
    for worker, task in plan["tasks"]:
        log.send("manager", worker, "assign", task, "M1")
        log.call(1)
        results[worker] = WORKER_OUTPUT[worker]
        log.send(worker, "manager", "report", WORKER_OUTPUT[worker], "M1")
    log.call(1)                                                # integration
    covered = [r for r in REQUIREMENTS if plan["coverage"].get(r)]
    log.send("manager", "user", "answer", "Integrated plan covering: " + ", ".join(covered), "M1")
    return {"log": log, "flagged": flagged, "results": results, "covered": covered,
            "missing_in_answer": [r for r in REQUIREMENTS if r not in covered], "calls": log.calls}


# ----------------------------------------------------------------------------------------
# Chapter 4b: peer / swarm (a shared board and two local rules)
# ----------------------------------------------------------------------------------------
MANALI_GUIDE = {   # pretend guide text, written for this exercise
    "See": ["Hadimba Temple sits in a quiet cedar forest.", "Solang Valley has wide views and snow in winter.", "Vashisht has hot springs and old temples.", "The Beas river walk is calm at sunrise."],
    "Do": ["Try short guided treks near Old Manali.", "Paragliding runs from Solang in season.", "Rafting is on offer on the Beas in summer.", "Yoga classes run in several Old Manali cafes."],
    "Eat": ["Old Manali has many vegetarian cafes with mountain views.", "Siddu with ghee is a local steamed bread worth trying.", "Several cafes serve full vegetarian thalis.", "Book dinner early in winter because many places close by 9 pm."],
    "Get around": ["Local buses are cheap but slow.", "Taxis have fixed rate cards at the stand.", "Scooter rental is common but icy roads are risky.", "Walk between Old Manali and the Mall Road if the weather is clear."],
}
COUPLE_PRIORITY = ["Eat", "See", "Do", "Get around"]       # what a quiet vegetarian couple cares about most


def swarm(mode: str = "turns", peers: int = 3, priority: bool = False, guide: dict | None = None) -> dict:
    """Peers share a board. Rule 1: claim a section and post two NEW insights. Rule 2: if nothing is left, write the summary.
    mode: 'turns' (one peer acts at a time), 'race' (peers act together, read the same board, may claim the same section),
          'lock' (peers act together but a claim is atomic: nobody gets a section someone else has claimed)."""
    guide = guide or MANALI_GUIDE
    unclaimed = list(guide)
    if priority:
        unclaimed.sort(key=lambda s: COUPLE_PRIORITY.index(s) if s in COUPLE_PRIORITY else 99)
    board: list = []
    log = Log()
    wasted, rounds = 0, 0
    names = ["peer_%d" % (i + 1) for i in range(peers)]
    while unclaimed and rounds < 20:
        rounds += 1
        if mode == "turns":
            batch = [(names[(rounds - 1) % peers], unclaimed.pop(0))]
        elif mode == "lock":
            n = min(peers, len(unclaimed))
            batch = [(names[i], unclaimed.pop(0)) for i in range(n)]
        else:                                                  # race: everyone reads the same first section
            first = unclaimed[0]
            n = min(peers, len(unclaimed))
            batch = [(names[i], first) for i in range(n)]
            unclaimed.pop(0)
        seen_this_round = set(board)
        for peer, section in batch:
            fresh = [s for s in guide[section] if "(%s) %s" % (section, s) not in seen_this_round][:2]
            for s in fresh:
                item = "(%s) %s" % (section, s)
                board.append(item)
            log.call(1)
            log.send(peer, "board", "post", "claimed '%s', posted %d insights" % (section, len(fresh)), "S1")
        if mode == "race":
            wasted += len(batch) - 1                           # every extra peer redid a section someone else had just taken
    summary_peer = names[0]
    log.call(1)
    top = [b for b in dict.fromkeys(board)]
    summary = "Manali briefing from %d insights: " % len(top) + " ".join(x.split(") ", 1)[1] for x in top[:4])
    log.send(summary_peer, "user", "answer", summary, "S1")
    dup_items = len(board) - len(set(board))
    return {"log": log, "board": board, "rounds": rounds, "calls": log.calls, "duplicates": dup_items, "wasted_calls": wasted, "summary": summary,
            "order": [m["content"].split("'")[1] for m in log.messages if m["kind"] == "post"]}


# ----------------------------------------------------------------------------------------
# Chapter 4c: federated teams (a shared report format makes integration plain code)
# ----------------------------------------------------------------------------------------
DOCUMENTS_REPORT = {"team": "documents_team", "findings": ["Visa on arrival is usual for Indian passports.", "Passport needs six months' validity."],
                    "costs": [{"item": "Visa on arrival, two people", "amount_inr": 2000, "category": "documents"},
                              {"item": "Passport photos and printouts", "amount_inr": 1500, "category": "documents"}],
                    "open_questions": ["Confirm the current visa fee on the official site."]}
MONEY_REPORT = {"team": "money_team", "findings": ["Mid-range stays keep the plan under Rs 2 lakh."],
                "costs": [{"item": "Return flights, two people", "amount_inr": 36000, "category": "flights"},
                          {"item": "Six nights, mid-range stays", "amount_inr": 60000, "category": "stay"},
                          {"item": "Food, two people", "amount_inr": 24000, "category": "food"},
                          {"item": "Local transport", "amount_inr": 12000, "category": "transport"},
                          {"item": "Travel insurance, two people", "amount_inr": 4000, "category": "insurance"}],
                "open_questions": ["Check live flight prices for February."]}
FOOD_REPORT = {"team": "food_team", "findings": ["Ubud has many vegetarian cafes."],
               "costs": [{"item": "Vegetarian meals and cafes, two people", "amount_inr": 22000, "category": "food"}],
               "open_questions": ["Which cafes take advance bookings?"]}
FREE_TEXT = {"documents_team": "Visa about Rs 2,000 and a bit for paperwork, maybe 1.5k.",
             "money_team": "Flights ~36k, hotels Rs 60,000/-, food around twenty four thousand, transport 12k, insurance 4,000."}
BUDGET = 200000


def integrate(reports: list) -> dict:
    """Plain code over standard reports: add costs, list open questions, flag two teams costing the same category."""
    total, rows, seen = 0, [], {}
    overlaps = []
    for rep in reports:
        for c in rep["costs"]:
            total += c["amount_inr"]
            rows.append({"team": rep["team"], "item": c["item"], "category": c.get("category", "?"), "amount (Rs)": c["amount_inr"]})
            cat = c.get("category")
            if cat in seen and seen[cat] != rep["team"]:
                overlaps.append("%s costed by both %s and %s" % (cat, seen[cat], rep["team"]))
            seen.setdefault(cat, rep["team"])
    questions = ["[%s] %s" % (r["team"], q) for r in reports for q in r["open_questions"]]
    return {"rows": rows, "total": total, "left": BUDGET - total, "overlaps": overlaps, "questions": questions}


def naive_parse(text: str) -> list:
    """What integration looks like WITHOUT a shared format: pull 'Rs 12,345'-style numbers out of free text."""
    return [int(m.replace(",", "")) for m in re.findall(r"(?i)rs\.?\s*([0-9][0-9,]*)", text)]


# ----------------------------------------------------------------------------------------
# Chapter 4d: sequential pipeline (errors propagate; a checker stops them)
# ----------------------------------------------------------------------------------------
FIXED_COSTS = {"stay": 60000, "food": 24000, "transport": 12000, "insurance": 4000, "visa": 2000}


def pipeline(flight_per_person: int, checker: str = "none", min_plausible: int = 15000) -> dict:
    """planner -> budgeter -> writer. checker: 'none', 'research' (before planning) or 'budget' (after the budgeter)."""
    log = Log()
    notes = "Return flights Mumbai-Bali: Rs %d per person." % flight_per_person
    log.send("research", "next stage", "output", notes, "P1")
    if checker == "research":
        log.call(1)
        if flight_per_person < min_plausible:
            log.send("checker", "pipeline", "halt", "Stopped before planning: Rs %d per person return to Bali is implausibly low." % flight_per_person, "P1")
            return {"log": log, "halted": True, "halted_at": "research", "calls": log.calls, "message": None, "total": None, "left": None}
    log.call(1)
    log.send("planner", "next stage", "output", "6-day Bali plan: Ubud then a quiet beach.", "P1")
    total = flight_per_person * 2 + sum(FIXED_COSTS.values())
    left = BUDGET - total
    log.call(1)
    log.send("budgeter", "next stage", "output", "Total Rs %d for two; Rs %d of the Rs 2 lakh budget left." % (total, left), "P1")
    if checker == "budget":
        log.call(1)
        if left > 0.35 * BUDGET:
            log.send("checker", "pipeline", "halt", "Stopped after budgeting: Rs %d left over looks too good to be true." % left, "P1")
            return {"log": log, "halted": True, "halted_at": "budget", "calls": log.calls, "message": None, "total": total, "left": left}
    log.call(1)
    msg = ("Great news: you will have Rs %d to spare, so upgrade to a private pool villa!" % left) if left > 60000 else "Your plan fits the budget, with Rs %d to spare." % left
    log.send("writer", "user", "answer", msg, "P1")
    return {"log": log, "halted": False, "halted_at": "", "calls": log.calls, "message": msg, "total": total, "left": left}


# ----------------------------------------------------------------------------------------
# Cost table (what a team costs): fixed reference runs
# ----------------------------------------------------------------------------------------
def reference_costs() -> list:
    rows = [
        ("1. Generalist answers the honeymoon request", 1),
        ("1. Generalist + one specialist (visa only)", 2),
        ("3.1 Centralized desk (coordinator up)", centralized_desk(True)["calls"]),
        ("3.2 Decentralized hand-over (stale board)", handover(ACTUAL, STALE_BOARD)["calls"]),
        ("3.2 Decentralized hand-over (accurate board)", handover(ACTUAL, ACTUAL)["calls"]),
        ("3.3 Hybrid rebooking (direct messages)", hybrid_rebooking()["calls"]),
        ("3.3 Hybrid rebooking (everything via manager)", hybrid_rebooking(direct=False)["calls"]),
        ("4.1 Manager / worker", manager_worker("Complete plan")["calls"]),
        ("4.2 Peer / swarm (turns)", swarm("turns")["calls"]),
        ("4.2 Peer / swarm (race, no lock)", swarm("race")["calls"]),
        ("4.4 Pipeline, no checker", pipeline(4000, "none")["calls"]),
        ("4.4 Pipeline, research checker (halts)", pipeline(4000, "research")["calls"]),
    ]
    return [{"Design": n, "Model calls": c} for n, c in rows]
