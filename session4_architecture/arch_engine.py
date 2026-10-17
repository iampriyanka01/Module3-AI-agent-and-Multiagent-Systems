"""
arch_engine.py
==============
Logic behind the Architecture Lab (Session 4). No Streamlit code here.

DEMO mode (no key): rule-based simulations, with pretend data, so every lab runs offline.
LIVE mode (Gemini key): the labs marked "live-capable" call Gemini through LangChain.
"""
from __future__ import annotations

import re

# ----------------------------------------------------------------------------------------
# Backend (Gemini live, or demo simulation)
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
                    kwargs["rate_limiter"] = InMemoryRateLimiter(requests_per_second=4 / 60, check_every_n_seconds=0.5,
                                                                 max_bucket_size=1)
                self.llm = ChatGoogleGenerativeAI(model=chat_model, google_api_key=api_key, **kwargs)
                self.live = True
            except Exception as exc:
                self.error = "Could not start Gemini: " + str(exc)

    @staticmethod
    def _resolve(demo):
        return demo() if callable(demo) else demo

    def ask(self, prompt: str, demo) -> str:
        if self.live:
            try:
                out = self.llm.invoke(prompt)
                self.last_source = "gemini"
                text = getattr(out, "text", None)
                return text if isinstance(text, str) else str(out.content)
            except Exception as exc:
                self.error, self.last_source = "Gemini call failed, showing the demo simulation: " + str(exc), "demo (fallback)"
                return self._resolve(demo)
        self.last_source = "demo"
        return self._resolve(demo)

    def structured(self, schema: dict, prompt: str, demo):
        if self.live:
            try:
                out = self.llm.with_structured_output(schema).invoke(prompt)
                self.last_source = "gemini"
                return out
            except Exception as exc:
                self.error, self.last_source = "Gemini call failed, showing the demo simulation: " + str(exc), "demo (fallback)"
                return self._resolve(demo)
        self.last_source = "demo"
        return self._resolve(demo)


# ----------------------------------------------------------------------------------------
# Pretend data used by the labs (written for this exercise, not real forecasts)
# ----------------------------------------------------------------------------------------
PLACES = ["mumbai", "shimla", "manali", "agra", "amritsar", "delhi", "gurgaon", "old manali"]
WEATHER = {
    "mumbai": "Mumbai (pretend): 31C humid. Rain on 5 of the next 7 days, heavy showers Tue to Thu.",
    "shimla": "Shimla (pretend): 9C, clear and cold. Light snow possible on Sunday.",
    "manali": "Manali (pretend): 2C, snow on 3 of the next 7 days, some road closures.",
    "old manali": "Old Manali (pretend): 2C, snow on 3 of the next 7 days, some road closures.",
    "agra": "Agra (pretend): 28C, sunny and dry all week.",
    "amritsar": "Amritsar (pretend): 14C, foggy mornings, dry afternoons.",
    "delhi": "Delhi (pretend): 33C, partly cloudy.",
    "gurgaon": "Gurgaon (pretend): 32C, sunny.",
}
GUIDE = {
    "shimla": "Shimla guide (pretend): do not miss a sunset walk along The Ridge and Mall Road.",
    "manali": "Manali guide (pretend): do not miss the cafes of Old Manali and Hadimba Temple.",
    "old manali": "Old Manali guide (pretend): do not miss the riverside cafes and Hadimba Temple.",
    "agra": "Agra guide (pretend): do not miss the Taj Mahal at sunrise.",
    "amritsar": "Amritsar guide (pretend): do not miss the Golden Temple at dawn.",
}
HOTELS = "Hotels (pretend): Pine View Hotel Rs 4,500/night; Mall Road Grand Rs 5,200/night (noisy)."
HOMESTAYS = "Homestays (pretend): Apple Orchard Homestay Rs 2,200/night, quiet, veg meals; Riverside Guesthouse Rs 1,800/night."
LANDMARK_TO_CITY = {"taj mahal": "Agra", "golden temple": "Amritsar", "hadimba": "Manali", "the ridge": "Shimla"}


def find_place(text: str):
    low = text.lower()
    for p in sorted(PLACES, key=len, reverse=True):
        if p in low:
            return p.title()
    for landmark, city in LANDMARK_TO_CITY.items():
        if landmark in low:
            return city
    return None


# ----------------------------------------------------------------------------------------
# 2.1 Deliberative: plan, then let CODE check the plan against preconditions
# ----------------------------------------------------------------------------------------
ACTION_RULES = {
    "ask_user": {"needs": [], "gives": ["dates", "user approval"]},
    "check_weather": {"needs": ["destination"], "gives": ["weather"]},
    "search_hotels": {"needs": ["destination", "dates"], "gives": ["hotel options"]},
    "book_hotel": {"needs": ["hotel options", "user approval"], "gives": ["booking"]},
    "write_itinerary": {"needs": ["weather", "hotel options"], "gives": ["itinerary"]},
    "send_summary": {"needs": ["itinerary"], "gives": ["summary sent"]},
}
DEMO_PLAN = ["ask_user", "search_hotels", "book_hotel", "check_weather", "write_itinerary", "send_summary"]
PLAN_SCHEMA = {"title": "plan", "description": "An ordered plan", "type": "object",
               "properties": {"steps": {"type": "array", "items": {"type": "object", "properties": {
                   "action": {"type": "string", "enum": list(ACTION_RULES)}, "why": {"type": "string"}},
                   "required": ["action", "why"]}}}, "required": ["steps"]}


def validate_plan(plan: list[str], known_at_start: list[str]) -> tuple[list[dict], bool]:
    known = list(known_at_start)
    rows, valid = [], True
    for i, action in enumerate(plan):
        rule = ACTION_RULES[action]
        missing = [n for n in rule["needs"] if n not in known]
        if missing:
            valid = False
        rows.append({"step": i + 1, "action": action, "needs": ", ".join(rule["needs"]) or "-",
                     "gives": ", ".join(rule["gives"]), "check": ("INVALID: missing " + ", ".join(missing)) if missing else "ok"})
        known.extend(rule["gives"])
    return rows, valid


def goal_reached(plan: list[str], known_at_start: list[str], goal_items: list[str]) -> bool:
    known = list(known_at_start)
    for a in plan:
        known.extend(ACTION_RULES[a]["gives"])
    return all(g in known for g in goal_items)


# ----------------------------------------------------------------------------------------
# 2.2 Reactive: keyword reflexes and subsumption
# ----------------------------------------------------------------------------------------
EMERGENCY_WORDS = ["stranded", "broke down", "broken down", "accident", "help us", "emergency", "we're lost", "injured"]
INSTANT_ANSWERS = {
    "booking reference": "Your booking reference is in the confirmation email and under My Trips > Bookings.",
    "cancel my booking": "You can cancel free of charge up to 48 hours before check-in under My Trips > Bookings > Cancel.",
}
EMERGENCY_REPLY = ("EMERGENCY: Call 112 now (India's national emergency number). Stay with your vehicle if it is safe, "
                   "switch on hazard lights, and share your live location with family. We are alerting the DilSeTrips support team.")
SATURDAY_MESSAGE = "Help, our car broke down on the road to Rohtang and it's getting dark. What do we do?"
TRICKY_MESSAGES = [
    "Our car died near Rohtang pass and there's no signal, what now?",
    "I'm lost for words, this trip was amazing!",
    "Can you help us find a quiet cafe in Old Manali?",
]


def reactive_layer(message: str, words=None):
    """Fixed rules, no LLM. Returns a reply or None (pass the message down)."""
    low = message.lower()
    for w in (EMERGENCY_WORDS if words is None else words):
        if w and w in low:
            return EMERGENCY_REPLY
    for phrase, reply in INSTANT_ANSWERS.items():
        if phrase in low:
            return reply
    return None


def looks_like_emergency(message: str) -> bool:
    """Stand-in for a cheap LLM safety check (HW2)."""
    low = message.lower()
    return any(w in low for w in ["died", "dead battery", "no signal", "stuck", "stranded", "injur", "bleeding",
                                  "lost in", "accident", "emergency", "help, ", "broke down"])


def subsumption(bumper: bool, battery_low: bool, moving_ok: bool) -> dict:
    """Robot vacuum. Layer 0 (avoid) > charger behaviour > Layer 1 (wander) > Layer 2 (clean)."""
    layers = [
        {"layer": "0 Avoid obstacles", "wants": "reverse_and_turn()" if bumper else None},
        {"layer": "0b Find charger", "wants": "go_to_charger()" if battery_low and not bumper else None},
        {"layer": "1 Wander", "wants": "move_forward()" if moving_ok else None},
        {"layer": "2 Clean", "wants": "vacuum_on()" if moving_ok and not battery_low else None},
    ]
    winner = next((l for l in layers if l["wants"] and l["layer"].startswith("0")), None)
    out = []
    for l in layers:
        if l["wants"] is None:
            status = "quiet"
        elif winner is not None and l is not winner:
            status = "SUPPRESSED by " + winner["layer"]
        else:
            status = "ACTIVE"
        out.append({"layer": l["layer"], "wants": l["wants"] or "-", "status": status})
    active = [o["wants"] for o in out if o["status"] == "ACTIVE"]
    return {"layers": out, "actions": active or ["idle"]}


# ----------------------------------------------------------------------------------------
# 2.3 BDI
# ----------------------------------------------------------------------------------------
TRIP_CONTEXT = ("Prasad and his partner fly Delhi to Kullu on 6E 2073 at 11:05 on 20 December, then stay 3 nights at "
                "Snow Valley Cottage, Old Manali (check-in 1 pm). She is vegetarian; he gets sick on long road trips. "
                "Budget is tight. They want a quiet, relaxing trip, and Solang Valley on day 2.")
BDI_SCHEMA = {"title": "bdi_state", "description": "An agent's beliefs, desires and committed intention", "type": "object",
              "properties": {"beliefs": {"type": "array", "items": {"type": "string"}},
                             "desires": {"type": "array", "items": {"type": "string"}},
                             "intention": {"type": "string"}, "plan": {"type": "array", "items": {"type": "string"}}},
              "required": ["beliefs", "desires", "intention", "plan"]}
RECONSIDER_SCHEMA = {"title": "reconsideration", "description": "How an event affects the current intention", "type": "object",
                     "properties": {"change_size": {"type": "string", "enum": ["small: patch a step", "big: reconsider the intention"]},
                                    "reason": {"type": "string"}, "intention": {"type": "string"},
                                    "plan": {"type": "array", "items": {"type": "string"}}},
                     "required": ["change_size", "reason", "intention", "plan"]}
BDI_EVENTS = ["Snow Valley Cottage says check-in will be at 3 pm instead of 1 pm.",
              "IndiGo has cancelled 6E 2073: a snowstorm will close Kullu airport for two days."]


def demo_bdi() -> dict:
    return {"beliefs": ["Flight 6E 2073 departs Delhi 11:05 on 20 December", "Stay is Snow Valley Cottage, Old Manali, check-in 1 pm",
                        "Partner is vegetarian", "Prasad gets sick on long road trips", "Budget is tight"],
            "desires": ["Have a quiet, relaxing trip", "Visit Solang Valley on day 2", "Keep costs low", "Avoid long road journeys"],
            "intention": "Deliver the planned 3-night Old Manali trip as booked",
            "plan": ["Fly 6E 2073 to Kullu", "Check in at Snow Valley Cottage at 1 pm", "Rest on day 1", "Solang Valley on day 2",
                     "Vegetarian dinners in Old Manali"]}


def demo_reconsider(bdi: dict, event: str) -> dict:
    low = event.lower()
    big = any(w in low for w in ["cancel", "closed", "close ", "snowstorm", "strike", "unavailable", "sold out"])
    if big:
        return {"change_size": "big: reconsider the intention",
                "reason": "The flight is gone and the airport is closed, so the committed intention cannot be met as planned.",
                "intention": "Reach Old Manali safely by another route, or move the trip",
                "plan": ["Check refund or rebooking for 6E 2073", "Compare a later flight with a train to Chandigarh",
                         "Avoid a long taxi leg if possible (motion sickness)", "Tell the cottage about the new arrival date"]}
    plan = [("Check in at Snow Valley Cottage at 3 pm (updated)" if "check-in" in s.lower() else s) for s in bdi["plan"]]
    return {"change_size": "small: patch a step", "reason": "Only one step is affected; the intention still holds.",
            "intention": bdi["intention"], "plan": plan}


# ----------------------------------------------------------------------------------------
# 2.4 Layered orchestrator (all timings are ILLUSTRATIVE, to show relative cost)
# ----------------------------------------------------------------------------------------
def packing_list(weather_text: str) -> list[str]:
    w = weather_text.lower()
    items = []
    if "rain" in w or "showers" in w:
        items += ["umbrella", "rain jacket", "waterproof bag cover"]
    if "snow" in w or "cold" in w or "2c" in w or "9c" in w:
        items += ["thermals", "woollen jacket", "gloves and cap"]
    if "sunny" in w or "sun" in w:
        items += ["sunscreen", "hat"]
    items += ["power bank", "ID copies", "refillable bottle"]
    return items[:8]


def orchestrate(message: str, words=None, safety_check=False) -> dict:
    trace, calls, secs = [], 0, 0.0

    def add(step, result, c, s):
        nonlocal calls, secs
        calls += c
        secs += s
        trace.append({"step": step, "result": result, "LLM calls": c, "seconds (illustrative)": s})

    reflex = reactive_layer(message, words)
    add("Reactive layer: keyword rules", "FIRED" if reflex else "passed down", 0, 0.001)
    if reflex:
        return {"layer": "reactive", "reply": reflex, "trace": trace, "calls": calls, "seconds": secs}
    if safety_check:
        hit = looks_like_emergency(message)
        add("Cheap LLM safety check (HW2)", "EMERGENCY" if hit else "not an emergency", 1, 1.2)
        if hit:
            return {"layer": "reactive (via safety check)", "reply": EMERGENCY_REPLY, "trace": trace, "calls": calls, "seconds": secs}
    place = find_place(message)
    routine = ("pack" in message.lower()) and place is not None
    add("Router (small LLM call)", "routine: packing list" if routine else "deliberative: needs planning", 1, 1.2)
    if routine:
        forecast = WEATHER.get(place.lower(), "no forecast")
        add("Sequencer step 1: weather tool (our code)", forecast[:60] + "...", 0, 0.4)
        items = packing_list(forecast)
        add("Sequencer step 2: LLM writes the list", "list written", 1, 1.5)
        return {"layer": "sequencer", "reply": "Packing list for " + place + ": " + ", ".join(items) + ".",
                "trace": trace, "calls": calls, "seconds": secs}
    add("Deliberative: plan", "3-step plan", 1, 1.5)
    add("Deliberative: ReAct cycles with tools", "3 thought/action/observation cycles", 3, 4.5)
    add("Deliberative: synthesise answer", "answer written", 1, 1.5)
    guide = GUIDE.get(place.lower()) if place else None
    reply = ("Here is a relaxed 2-day food trail" + (" in " + place if place else "") + " for a vegetarian couple: "
             "Day 1 a pure-veg breakfast spot, a heritage walk and a regional thali lunch; Day 2 a market food walk "
             "and an early dinner. " + (guide or "")).strip()
    return {"layer": "deliberative", "reply": reply, "trace": trace, "calls": calls, "seconds": secs}


# ----------------------------------------------------------------------------------------
# 2.5 CoALA: propose, evaluate, select
# ----------------------------------------------------------------------------------------
SITUATION = ("The couple is in Old Manali. Tomorrow was planned as Solang Valley, but the forecast shows heavy snow "
             "and road closures. They are vegetarian and on a tight budget.")
PROPOSE_SCHEMA = {"title": "proposals", "description": "Candidate next actions", "type": "object",
                  "properties": {"candidates": {"type": "array", "items": {"type": "object", "properties": {
                      "action": {"type": "string"},
                      "action_type": {"type": "string", "enum": ["external: grounding", "internal: retrieval", "internal: reasoning",
                                                                  "internal: learning"]}}, "required": ["action", "action_type"]}}},
                  "required": ["candidates"]}
EVALUATE_SCHEMA = {"title": "evaluations", "description": "Scores for each candidate action", "type": "object",
                   "properties": {"scores": {"type": "array", "items": {"type": "object", "properties": {
                       "action": {"type": "string"}, "safety": {"type": "integer"}, "usefulness": {"type": "integer"},
                       "cost": {"type": "integer", "description": "10 = cheapest"}}, "required": ["action", "safety", "usefulness", "cost"]}}},
                   "required": ["scores"]}
DEMO_CANDIDATES = [
    {"action": "Check the latest road status with the weather tool", "action_type": "external: grounding", "safety": 9, "usefulness": 7, "cost": 9},
    {"action": "Recall which indoor places this couple enjoyed last time", "action_type": "internal: retrieval", "safety": 9, "usefulness": 6, "cost": 10},
    {"action": "Re-plan tomorrow as an indoor day: cafes, a cooking class, a museum", "action_type": "internal: reasoning", "safety": 10, "usefulness": 8, "cost": 7},
    {"action": "Insist on Solang Valley and book a taxi anyway", "action_type": "external: grounding", "safety": 2, "usefulness": 5, "cost": 4},
]


def coala_select(scores: list[dict], weights: tuple[float, float, float]) -> tuple[list[dict], str]:
    ws, wu, wc = weights
    rows = []
    for s in scores:
        total = ws * s["safety"] + wu * s["usefulness"] + wc * s["cost"]
        rows.append({**s, "total": round(total, 2),
                     "working": "%.2fx%d + %.2fx%d + %.2fx%d = %.1f" % (ws, s["safety"], wu, s["usefulness"], wc, s["cost"], total)})
    best = max(rows, key=lambda r: r["total"])["action"]
    return rows, best


SHOPFAST_STEPS = [
    ("User", "I never received the sneakers I ordered last week! Order #12345.", "prompt"),
    ("Think (internal: retrieval)", "Agent retrieves the shipping policy (knowledge base) and the customer's history (memory).", "retrieval"),
    ("Think (internal: reasoning)", "Working memory: customer is upset; first find the order status.", "reasoning"),
    ("Decide", "Propose [query_database, apologise, escalate], evaluate, select query_database.", "decision"),
    ("Act (external: grounding)", "query_database(order_number=12345) -> shipped, tracking QP789XYZ, last updated 3 days ago.", "action"),
    ("Observe", "Outcome returned. Human review optional: a person may approve a 10% coupon.", "observe"),
    ("Learn (internal: learning)", "Write the interaction to episodic memory: 'customer #12345 happy with expedited shipping'.", "learning"),
]


# ----------------------------------------------------------------------------------------
# 3.x Patterns: ReAct simulation, Planning vs ReAct, tool use, reflection
# ----------------------------------------------------------------------------------------
def parse_action(text: str):
    """Find 'Action: tool_name[input]' and return (tool_name, input), or (None, None). (from the notebook)"""
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("Action:") and "[" in line:
            action_text = line[len("Action:"):].strip()
            return action_text.split("[")[0].strip(), action_text.split("[", 1)[1].rstrip("]").strip()
    return None, None


def run_tool(tool: str, arg: str, hotels_up=True, homestays_up=True) -> str:
    key = arg.lower().strip()
    if tool == "get_weather_forecast":
        return WEATHER.get(key, "ERROR: no forecast for " + arg)
    if tool == "search_travel_guide":
        return GUIDE.get(key, "ERROR: no guide for " + arg)
    if tool == "search_hotels":
        return HOTELS if hotels_up else "ERROR: hotel search service unavailable (HTTP 503). Try again later."
    if tool == "search_homestays":
        return HOMESTAYS if homestays_up else "ERROR: homestay search service unavailable (HTTP 503)."
    return "ERROR: no tool called " + tool


def needs_of(question: str) -> list[str]:
    q = question.lower()
    needs = []
    if any(w in q for w in ["weather", "week", "umbrella", "rain", "cold", "snow", "forecast"]):
        needs.append("weather")
    if any(w in q for w in ["guide", "miss", "see ", "sights", "visit"]):
        needs.append("guide")
    if any(w in q for w in ["stay", "hotel", "homestay", "sleep", "accommodation"]):
        needs.append("stay")
    return needs or ["weather"]


def react_simulate(question: str, hotels_up=True, homestays_up=True) -> list[dict]:
    place = find_place(question)
    if place is None:
        return [{"thought": "I do not recognise the place. I should ask instead of guessing.", "final": "Which place do you mean?"}]
    steps, facts = [], []
    for need in needs_of(question):
        if need == "weather":
            obs = run_tool("get_weather_forecast", place)
            steps.append({"thought": "I need the forecast for " + place + ". I will use the weather tool.",
                          "action": "get_weather_forecast[" + place + "]", "observation": obs})
            facts.append(obs)
        elif need == "guide":
            obs = run_tool("search_travel_guide", place)
            steps.append({"thought": "I need one must-see in " + place + ". I will check the travel guide.",
                          "action": "search_travel_guide[" + place + "]", "observation": obs})
            facts.append(obs)
        else:
            obs = run_tool("search_hotels", place, hotels_up, homestays_up)
            steps.append({"thought": "Somewhere to stay: the obvious tool is the hotel search.",
                          "action": "search_hotels[" + place + "]", "observation": obs})
            if obs.startswith("ERROR"):
                obs2 = run_tool("search_homestays", place, hotels_up, homestays_up)
                steps.append({"thought": "The hotel search failed. A homestay search can answer the same need, so I will try it.",
                              "action": "search_homestays[" + place + "]", "observation": obs2})
                obs = obs2
            facts.append(obs)
    failed = all(f.startswith("ERROR") for f in facts)
    if failed:
        final = "Sorry, every service I tried is unavailable right now, so I cannot answer reliably. Please try again later."
    else:
        final = " ".join(f for f in facts if not f.startswith("ERROR"))
    steps.append({"thought": "I have what I need." if not failed else "Nothing worked, so I should say so honestly.", "final": final})
    return steps


def planning_simulate(task_needs: list[str], place: str, hotels_up=True, homestays_up=True) -> dict:
    plan = []
    if "weather" in task_needs:
        plan.append(("get_weather_forecast", place))
    if "stay" in task_needs:
        plan.append(("search_hotels", place))
    log, broke = [], False
    for i, (tool, arg) in enumerate(plan):
        result = run_tool(tool, arg, hotels_up, homestays_up)
        log.append({"step": i + 1, "call": tool + "(" + arg + ")", "result": result})
        if result.startswith("ERROR"):
            broke = True
            break
    return {"plan": plan, "log": log, "broke": broke, "finished": not broke}


TOOLS = {
    "get_weather_forecast": "Information retrieval: the 7-day forecast for a town.",
    "add_reminder": "Action execution: add a reminder to the couple's app.",
    "convert_currency": "Computation: convert an amount between currencies (pretend rates).",
}
CURRENCY_WORDS = {"dollar": "USD", "usd": "USD", "euro": "EUR", "eur": "EUR", "pound": "GBP", "gbp": "GBP", "rupee": "INR", "inr": "INR"}


def demo_tool_call(request: str, ask_if_missing: bool) -> dict:
    low = request.lower()
    if any(w in low for w in ["remind", "reminder"]):
        when = re.search(r"(\d{1,2}(?::\d{2})?\s?(?:am|pm)|tomorrow|tonight)", low)
        return {"tool": "add_reminder", "params": {"text": request, "when": when.group(1) if when else "(not stated)"},
                "note": "Selected the action tool."}
    if any(w in low for w in ["rupee", "dollar", "euro", "pound", "convert", "currency", "usd", "inr"]):
        amount = re.search(r"([\d,]+(?:\.\d+)?)", request)
        amt = float(amount.group(1).replace(",", "")) if amount else None
        codes = []
        for word, code in CURRENCY_WORDS.items():
            if word in low and code not in codes:
                codes.append(code)
        src = codes[0] if codes else "INR"
        dst = codes[1] if len(codes) > 1 else None
        if dst is None:
            if ask_if_missing:
                return {"tool": None, "params": {}, "note": "No target currency was named, so the agent asks: 'Which currency would you like to convert to?'"}
            return {"tool": "convert_currency", "params": {"amount": amt, "from_currency": src, "to_currency": "USD"},
                    "note": "PARAMETER HALLUCINATION: nobody said USD. The model invented a target currency."}
        return {"tool": "convert_currency", "params": {"amount": amt, "from_currency": src, "to_currency": dst}, "note": "Selected the computation tool."}
    place = find_place(request)
    if "weather" in low or "forecast" in low or "rain" in low:
        note = "Inferred '" + place + "' from world knowledge (correct and useful)." if place else "No place found."
        return {"tool": "get_weather_forecast", "params": {"place": place or ""}, "note": note}
    return {"tool": None, "params": {}, "note": "No tool needed: answer directly."}


def convert_currency(amount: float, src: str, dst: str) -> str:
    rates = {"INR": 1.0, "USD": 84.0, "EUR": 91.0, "GBP": 107.0}
    return "%.0f %s = %.2f %s (pretend rates)" % (amount, src, amount * rates[src] / rates[dst], dst)


GRAVITY_TASK = "Explain gravity to a five-year-old."
GRAVITY_FIRST = ("Gravity is a fundamental interaction of nature that causes mutual attraction between all things with mass; "
                 "in general relativity it is a consequence of the curvature of spacetime.")
GRAVITY_CRITIQUE = {"meets_requirement": False,
                    "critique": "Far too complex: 'fundamental interaction', 'general relativity' and 'curvature of spacetime' mean nothing to a five-year-old.",
                    "reasoning": "The answer was written for physicists, not for the stated audience.",
                    "lessons": ["Match vocabulary to the audience", "Use a concrete picture instead of definitions"]}
GRAVITY_REFINED = ("Imagine the world is a big bouncy trampoline and you are a bowling ball on it. You make a dip, and a little marble "
                   "rolls into it. Gravity is like that: big things like the Earth make a giant invisible dip that keeps everything from floating away.")
CRITIC_SCHEMA = {"title": "critique", "description": "A structured self-review", "type": "object",
                 "properties": {"meets_requirement": {"type": "boolean"}, "critique": {"type": "string"},
                                "reasoning": {"type": "string"}, "lessons": {"type": "array", "items": {"type": "string"}}},
                 "required": ["meets_requirement", "critique", "reasoning", "lessons"]}
LOG_LESSONS = ["Every critical step needs a fallback tool (e.g. homestay search when hotel search is down).",
               "Check the log for ERROR results early and re-plan from the current state instead of abandoning the task."]


# ----------------------------------------------------------------------------------------
# 4. Capstone: reactive + planning + ReAct/tool use + reflection (and the reference bug)
# ----------------------------------------------------------------------------------------
def capstone(message: str, buggy: bool = False) -> dict:
    trace = []
    reflex = reactive_layer(message)
    if reflex:
        trace.append({"stage": "REACTIVE", "detail": "Keyword rule fired: answered instantly, no LLM call."})
        return {"final": reflex, "trace": trace, "llm_calls": 0}
    place = find_place(message) or "Mumbai"
    plan = ["Identify the place the user asked about", "Fetch the forecast with the weather tool", "Summarise and advise on umbrellas"]
    trace.append({"stage": "PLAN", "detail": " | ".join(plan)})
    fetched = "Gurgaon" if buggy else place
    obs = WEATHER.get(fetched.lower(), "ERROR: no forecast")
    trace.append({"stage": "REACT + TOOL USE", "detail": "Action: get_weather_forecast[%s]  (asked for: %s)" % (fetched, place)})
    trace.append({"stage": "OBSERVATION", "detail": obs})
    draft = "Weather for %s this week: %s" % (place, obs.split(": ", 1)[-1])
    if "rain" in obs.lower():
        draft += " Yes, carry umbrellas."
    elif "sunny" in obs.lower():
        draft += " Umbrellas are not needed."
    trace.append({"stage": "DRAFT ANSWER", "detail": draft})
    wrong = fetched.lower() != place.lower()
    if buggy:
        trace.append({"stage": "REFLECTION", "detail": "supported: True (the reviewer sees only the observation text, so it cannot tell the data belongs to another city)"})
    else:
        trace.append({"stage": "REFLECTION", "detail": "supported: True (place, numbers and advice all match the observation)"})
    return {"final": draft, "trace": trace, "llm_calls": 5, "wrong_city": wrong}


# ----------------------------------------------------------------------------------------
# Architecture chooser (heuristic, based on the session's comparison table)
# ----------------------------------------------------------------------------------------
ARCH_PROFILE = {
    # speed, planning/foresight, adaptability, predictability, open-endedness, simplicity (0 to 5)
    "Deliberative": {"speed": 1, "foresight": 5, "adaptability": 1, "predictability": 4, "open-endedness": 2, "simplicity": 1},
    "Reactive": {"speed": 5, "foresight": 0, "adaptability": 1, "predictability": 5, "open-endedness": 0, "simplicity": 5},
    "BDI": {"speed": 3, "foresight": 3, "adaptability": 3, "predictability": 3, "open-endedness": 3, "simplicity": 2},
    "Layered (hybrid)": {"speed": 4, "foresight": 4, "adaptability": 4, "predictability": 4, "open-endedness": 3, "simplicity": 1},
    "CoALA (LLM agent)": {"speed": 2, "foresight": 4, "adaptability": 5, "predictability": 2, "open-endedness": 5, "simplicity": 2},
}


def choose_architecture(importance: dict) -> list[dict]:
    rows = []
    for name, prof in ARCH_PROFILE.items():
        score = sum(importance[k] * prof[k] for k in importance)
        rows.append({"architecture": name, "score": score})
    top = max(r["score"] for r in rows) or 1
    for r in rows:
        r["fit %"] = round(100 * r["score"] / top)
    return sorted(rows, key=lambda r: r["score"], reverse=True)
