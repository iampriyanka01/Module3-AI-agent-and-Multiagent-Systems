"""
foundations_engine.py
=====================
All the logic behind the Foundations Lab (Session 1). No Streamlit code lives here, so every function
can be read, tested and reused on its own.

Two modes (same idea as the Session 3 and 4 apps):
  * DEMO mode (no API key): rule-based "scripted models" and pretend data. Every lab still runs offline.
    The demo model is a SIMULATION: it behaves the way a good LLM usually does, so you can study the
    surrounding machinery (tools, loop, workflows) without spending a single API call.
  * LIVE mode (Gemini key): the same labs call Gemini through LangChain, exactly as in the notebook.
    If a live call fails, the lab falls back to the demo answer and says why.

Everything marked "pretend" is data written for this exercise, not real flights, forecasts or prices.
"""
from __future__ import annotations

import heapq
import inspect
import json
import random
import re
import time
from collections import deque
from datetime import date, datetime, timedelta

MODEL_DEFAULT = "gemini-3.5-flash-lite"


# ----------------------------------------------------------------------------------------
# Backend: Gemini (live) or simulation (demo)
# ----------------------------------------------------------------------------------------
class Backend:
    """One object the whole app talks to. `demo` arguments are callables or values used when not live."""

    def __init__(self, api_key: str = "", chat_model: str = MODEL_DEFAULT, rate_limit: bool = True):
        self.live, self.error, self.last_source, self.llm = False, "", "demo", None
        self.calls, self.last_seconds, self.notify = 0, 0.0, None
        if api_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                kwargs = {}
                if rate_limit:
                    from langchain_core.rate_limiters import InMemoryRateLimiter
                    kwargs["rate_limiter"] = InMemoryRateLimiter(requests_per_second=4 / 60,
                                                                 check_every_n_seconds=0.5, max_bucket_size=1)
                self.llm = ChatGoogleGenerativeAI(model=chat_model, google_api_key=api_key, **kwargs)
                self.live = True
            except Exception as exc:  # missing package, bad arguments...
                self.error = "Could not start Gemini: " + str(exc)

    # -- helpers
    @staticmethod
    def _resolve(demo):
        return demo() if callable(demo) else demo

    def _say(self, text: str):
        if self.notify:
            try:
                self.notify(text)
            except Exception:
                pass

    def _fail(self, exc: Exception, demo):
        self.error = "Gemini call failed, showing the demo simulation: " + str(exc)[:300]
        self.last_source = "demo (fallback)"
        return self._resolve(demo)

    @staticmethod
    def _lc_messages(messages):
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
        out = []
        for m in messages:
            role = m["role"]
            if role == "system":
                out.append(SystemMessage(content=m["content"]))
            elif role == "user":
                out.append(HumanMessage(content=m["content"]))
            elif role == "assistant":
                calls = [{"name": c["name"], "args": c["args"], "id": c["id"], "type": "tool_call"}
                         for c in m.get("tool_calls", [])]
                out.append(AIMessage(content=m.get("content", ""), tool_calls=calls))
            elif role == "tool":
                out.append(ToolMessage(content=m["content"], tool_call_id=m["tool_call_id"]))
        return out

    @staticmethod
    def _text_of(out) -> str:
        text = getattr(out, "text", None)
        if callable(text):          # older langchain versions expose .text() as a method
            text = text()
        return text if isinstance(text, str) else str(out.content)

    # -- plain text (a prompt string, or a list of {"role","content"} dicts)
    def ask(self, prompt, demo) -> str:
        if self.live:
            try:
                self._say("Calling Gemini...")
                t0 = time.time()
                payload = prompt if isinstance(prompt, str) else self._lc_messages(prompt)
                out = self.llm.invoke(payload)
                self.calls += 1
                self.last_seconds = time.time() - t0
                self.last_source = "gemini"
                return self._text_of(out)
            except Exception as exc:
                return self._fail(exc, demo)
        self.last_source = "demo"
        return self._resolve(demo)

    # -- structured output (JSON schema -> dict)
    def structured(self, schema: dict, prompt: str, demo):
        if self.live:
            try:
                self._say("Calling Gemini (structured output)...")
                t0 = time.time()
                out = self.llm.with_structured_output(schema).invoke(prompt)
                self.calls += 1
                self.last_seconds = time.time() - t0
                self.last_source = "gemini"
                return out
            except Exception as exc:
                return self._fail(exc, demo)
        self.last_source = "demo"
        return self._resolve(demo)

    # -- one tool-calling decision for the agent loop
    def decide(self, messages: list[dict], tools: dict, policy) -> dict:
        """Return {"text": str, "tool_calls": [{"id","name","args"}]}. Live: Gemini decides. Demo: the scripted policy."""
        if self.live:
            try:
                self._say("Gemini is deciding the next step...")
                t0 = time.time()
                bound = self.llm.bind_tools(list(tools.values()))
                reply = bound.invoke(self._lc_messages(messages))
                self.calls += 1
                self.last_seconds = time.time() - t0
                self.last_source = "gemini"
                calls = [{"id": c.get("id") or "call_%d" % i, "name": c["name"], "args": c["args"]}
                         for i, c in enumerate(reply.tool_calls)]
                return {"text": self._text_of(reply), "tool_calls": calls}
            except Exception as exc:
                self._fail(exc, None)
                return policy(messages)
        self.last_source = "demo"
        return policy(messages)


# ----------------------------------------------------------------------------------------
# Tools: ordinary Python functions + what the model actually sees
# ----------------------------------------------------------------------------------------
def tool_schema(fn) -> dict:
    """Build the JSON description of a tool from its name, type hints and docstring (what bind_tools sends)."""
    doc = inspect.getdoc(fn) or ""
    first, _, rest = doc.partition("\n\nArgs:")
    arg_docs = {}
    for line in rest.splitlines():
        if ":" in line:
            name, _, text = line.strip().partition(":")
            arg_docs[name.strip()] = text.strip()
    type_names = {str: "string", int: "integer", float: "number", bool: "boolean"}
    props, required = {}, []
    for pname, param in inspect.signature(fn).parameters.items():
        props[pname] = {"type": type_names.get(param.annotation, "string"), "description": arg_docs.get(pname, "")}
        if param.default is inspect.Parameter.empty:
            required.append(pname)
    return {"type": "function", "function": {"name": fn.__name__, "description": " ".join(first.split()),
                                             "parameters": {"type": "object", "properties": props, "required": required}}}


def next_saturday(today: date | None = None) -> date:
    today = today or date.today()
    gap = (5 - today.weekday()) % 7
    return today + timedelta(days=gap or 7)


def fresh_calendar(today: date | None = None) -> list[dict]:
    sat = next_saturday(today)
    return [{"date": sat.isoformat(), "time": "18:00-21:00", "title": "Office cricket match at Shivaji Park"}]


def _valid_iso(text: str) -> bool:
    try:
        datetime.strptime(text, "%Y-%m-%d")
        return True
    except Exception:
        return False


def make_calendar_tools(calendar: list[dict], validate: bool = False, with_update: bool = False,
                        today: date | None = None) -> dict:
    """The concierge's tools. `calendar` is a plain list that the tools read and change."""
    now = today or date.today()

    def get_today() -> str:
        """Return today's date and weekday. Use it before working out dates like 'this Saturday'."""
        return "Today is %s (%s)." % (now.isoformat(), now.strftime("%A"))

    def check_calendar(date_text: str) -> str:
        """List the events in Prasad's calendar on one date.

        Args:
            date_text: The date in YYYY-MM-DD format, for example 2026-10-03.
        """
        found = [e["time"] + " " + e["title"] for e in calendar if e["date"] == date_text]
        return ("Events on %s: " % date_text + "; ".join(found)) if found else "No events on %s." % date_text

    def add_calendar_event(date_text: str, time_text: str, title: str) -> str:
        """Add an event to Prasad's calendar.

        Args:
            date_text: The date in YYYY-MM-DD format, for example 2026-10-03.
            time_text: The start time in 24-hour HH:MM format, for example 19:00.
            title: A short description of the event.
        """
        if validate and not _valid_iso(date_text):
            return "ERROR: date_text must be in YYYY-MM-DD format, for example 2026-10-03. You sent '%s'." % date_text
        calendar.append({"date": date_text, "time": time_text, "title": title})
        return "Added: %s %s %s" % (date_text, time_text, title)

    tools = {"get_today": get_today, "check_calendar": check_calendar, "add_calendar_event": add_calendar_event}

    if with_update:
        def update_calendar_event(date_text: str, old_time: str, new_time: str) -> str:
            """Change the start time of an existing event.

            Args:
                date_text: The date of the event in YYYY-MM-DD format.
                old_time: The current start time in HH:MM format.
                new_time: The new start time in HH:MM format.
            """
            for event in calendar:
                if event["date"] == date_text and event["time"] == old_time:
                    event["time"] = new_time
                    return "Updated: %s now starts at %s (%s)" % (event["title"], new_time, date_text)
            return "ERROR: no event on %s at %s." % (date_text, old_time)
        tools["update_calendar_event"] = update_calendar_event
    return tools


# ----------------------------------------------------------------------------------------
# The agent loop (by hand, with a step limit), plus the scripted "model" used in demo mode
# ----------------------------------------------------------------------------------------
def run_agent_loop(backend: Backend, goal: str, instructions: str, tools: dict, policy, max_steps: int = 6,
                   robust: bool = True, history: list[dict] | None = None) -> dict:
    """
    Perceive -> reason -> act -> feedback, repeated. Mirrors `run_agent` in the notebook.
    Returns {"messages", "events", "final", "steps", "stopped", "crashed"}.
    `events` is a flat, display-friendly list: model requests, tool runs, errors and the final answer.
    """
    messages = list(history) if history else [{"role": "system", "content": instructions}]
    messages.append({"role": "user", "content": goal})
    events, final, stopped, crashed = [], "", False, False
    uid = 0
    for step in range(1, max_steps + 1):
        decision = backend.decide(messages, tools, policy)
        calls = []
        for c in decision.get("tool_calls", []):
            uid += 1
            calls.append({"id": c.get("id") or "call_%d_%d" % (step, uid), "name": c["name"], "args": dict(c["args"])})
        messages.append({"role": "assistant", "content": decision.get("text", ""), "tool_calls": calls})
        if not calls:
            final = decision.get("text", "")
            events.append({"step": step, "kind": "final", "text": final})
            return {"messages": messages, "events": events, "final": final, "steps": step, "stopped": False,
                    "crashed": False}
        if decision.get("text"):
            events.append({"step": step, "kind": "thought", "text": decision["text"]})
        for c in calls:
            events.append({"step": step, "kind": "request", "name": c["name"], "args": c["args"], "id": c["id"]})
            fn = tools.get(c["name"])
            try:
                if fn is None:
                    raise KeyError(c["name"])
                result = str(fn(**c["args"]))
                kind = "result"
            except Exception as exc:
                if not robust:
                    events.append({"step": step, "kind": "crash",
                                   "text": "%s: %s  -> the whole agent stopped" % (type(exc).__name__, exc)})
                    return {"messages": messages, "events": events, "final": "", "steps": step, "stopped": False,
                            "crashed": True}
                if fn is None:
                    result = "ERROR: there is no tool called '%s'. Available tools: %s." % (c["name"], ", ".join(tools))
                else:
                    result = "ERROR: could not run %s with %s (%s). Check the argument names." % (c["name"], c["args"], exc)
                kind = "error"
            events.append({"step": step, "kind": kind, "name": c["name"], "text": result})
            messages.append({"role": "tool", "content": result, "tool_call_id": c["id"]})
    events.append({"step": max_steps, "kind": "limit",
                   "text": "Stopped: step limit (%d) reached without a final answer." % max_steps})
    return {"messages": messages, "events": events, "final": "", "steps": max_steps, "stopped": True, "crashed": False}


def _turn_slice(messages: list[dict]):
    """Everything after the latest user message: (goal_text, calls_made, results_received)."""
    idx = max(i for i, m in enumerate(messages) if m["role"] == "user")
    goal = messages[idx]["content"]
    calls, results = [], []
    pending = {}
    for m in messages[idx + 1:]:
        if m["role"] == "assistant":
            for c in m.get("tool_calls", []):
                calls.append(c)
                pending[c["id"]] = c
        elif m["role"] == "tool":
            results.append((pending.get(m["tool_call_id"], {}).get("name", ""), pending.get(m["tool_call_id"], {}).get("args", {}),
                            m["content"]))
    return goal, calls, results


def _find_today(messages: list[dict]):
    for m in messages:
        if m["role"] == "tool":
            hit = re.search(r"Today is (\d{4}-\d{2}-\d{2})", m["content"])
            if hit:
                return datetime.strptime(hit.group(1), "%Y-%m-%d").date()
    return None


def concierge_policy(honest: bool = True, sloppy_date: bool = False, typo_tool: bool = False, with_update: bool = False):
    """
    The scripted stand-in for Gemini in demo mode. It reads the conversation so far and returns the next
    decision, the way a good model would: get today's date, check the calendar, add an event, answer.
    """
    def policy(messages: list[dict]) -> dict:
        goal, calls, results = _turn_slice(messages)
        low = goal.lower()
        today = _find_today(messages)
        names_ok = {"get_today", "check_calendar", "add_calendar_event", "update_calendar_event"}
        good = [(n, a, r) for (n, a, r) in results if n in names_ok]
        n_calls = lambda name: [x for x in good if x[0] == name]

        def call(name, args, thought=""):
            return {"text": thought, "tool_calls": [{"id": "", "name": name, "args": args}]}

        if today is None:
            return call("get_today", {}, "I need today's date before I can work out what 'this Saturday' means.")
        sat = next_saturday(today)
        sun = sat + timedelta(days=1)

        # ---- follow-up: change the time of an event we just added
        if ("8 pm" in low or "8pm" in low or "make it" in low) and "book" not in low:
            if with_update:
                upd = n_calls("update_calendar_event")
                if upd:
                    return {"text": "Done. " + upd[-1][2] + ". I still cannot confirm anything with the café itself.", "tool_calls": []}
                return call("update_calendar_event", {"date_text": sun.isoformat(), "old_time": "19:00", "new_time": "20:00"},
                            "There is an update tool, so I can change the time of the event I added earlier.")
            adds = n_calls("add_calendar_event")
            if honest:
                return {"text": "I have no tool that can move or delete an event, so I can't change the 7 pm entry. "
                                "I could add a second event at 8 pm, but that would leave a duplicate. Shall I?", "tool_calls": []}
            if adds:
                return {"text": "Done! I've moved your dinner to 8 pm.", "tool_calls": []}
            return call("add_calendar_event", {"date_text": sun.isoformat(), "time_text": "20:00",
                                               "title": "Dinner for two at the pink café, Dadar"},
                        "I'll just add an 8 pm event.")

        # ---- is the evening free?
        wants_booking = "book" in low or "add it" in low
        if not wants_booking:
            cal = n_calls("check_calendar")
            if not cal:
                return call("check_calendar", {"date_text": sat.isoformat()}, "Today's date is known. Now I check Saturday (%s)." % sat)
            if cal[-1][2].startswith("No events"):
                return {"text": "Yes, you are free on Saturday %s. Your calendar is empty that day." % sat.isoformat(), "tool_calls": []}
            return {"text": "No. On Saturday %s you have: %s" % (sat.isoformat(), cal[-1][2].split(": ", 1)[-1]), "tool_calls": []}

        # ---- book (with a fallback day if Saturday is busy)
        fallback = "sunday" in low or "busy" in low
        sat_checks = [x for x in n_calls("check_calendar") if x[1].get("date_text") == sat.isoformat()]
        sun_checks = [x for x in n_calls("check_calendar") if x[1].get("date_text") == sun.isoformat()]
        if not sat_checks:
            if typo_tool and not any(c["name"] == "check_calender" for c in calls):
                return call("check_calender", {"date_text": sat.isoformat()}, "Let me look at Saturday's calendar.")
            return call("check_calendar", {"date_text": sat.isoformat()}, "First I check Saturday (%s) for clashes." % sat)
        sat_free = sat_checks[-1][2].startswith("No events")
        target, target_date = ("Saturday", sat) if sat_free else ("Sunday", sun)
        if not sat_free:
            if not fallback:
                return {"text": "You're busy on Saturday evening (%s). Would you like another day?" % sat_checks[-1][2].split(": ", 1)[-1],
                        "tool_calls": []}
            if not sun_checks:
                return call("check_calendar", {"date_text": sun.isoformat()}, "Saturday is taken, so I check the fallback day, Sunday.")
        adds = n_calls("add_calendar_event")
        sent = [c for c in calls if c["name"] == "add_calendar_event"]
        if not adds or adds[-1][2].startswith("ERROR"):
            if sloppy_date and not sent:
                return call("add_calendar_event", {"date_text": target, "time_text": "19:00",
                                                   "title": "Dinner for two at the pink café, Dadar"},
                            "The day is free. I add the event for %s at 7 pm." % target)
            return call("add_calendar_event", {"date_text": target_date.isoformat(), "time_text": "19:00",
                                               "title": "Dinner for two at the pink café, Dadar"},
                        "The day is free. I add the event for %s %s at 7 pm." % (target, target_date.isoformat()))
        when = "%s %s" % (target, target_date.isoformat())
        busy_note = "" if sat_free else "You're busy on Saturday (cricket match until 21:00), so I used Sunday. "
        if honest:
            return {"text": busy_note + "I added 'Dinner for two at the pink café' to your calendar for %s at 19:00. "
                            "I can't book the table at the café itself (I have no tool for that), so please call them." % when,
                    "tool_calls": []}
        return {"text": busy_note + "Done! I've booked a table for two at the pink café in Dadar for %s at 7 pm and added it to your calendar." % when,
                "tool_calls": []}
    return policy


# ---- FlyBot 2.0 (pretend airline systems) -------------------------------------------------
WEATHER_ALERTS = {
    "KUU": "Snowstorm warning at Kullu-Manali (Bhuntar) airport until 10:00 tomorrow. No arrivals or departures before then.",
    "DEL": "No weather alerts.",
}
FLIGHTS_DEL_KUU = [
    {"flight": "6E 2071", "departs": "07:15", "arrives": "08:40"},
    {"flight": "6E 2073", "departs": "11:05", "arrives": "12:30"},
    {"flight": "AI 9811", "departs": "13:20", "arrives": "14:45"},
]
BAGGAGE = {"snowboard": "Snowboards are oversized baggage. Check them in at the special baggage counter at least 2 hours before departure."}
HOTELS_NEAR = {"DEL": "Aerocity Stay (2 km, Rs 6,500, free airport shuttle); Terminal Inn (4 km, Rs 3,900, shuttle every 30 min)."}


def make_airline_tools() -> dict:
    def check_weather_alerts(airport_code: str) -> str:
        """Return current weather alerts for an airport.

        Args:
            airport_code: Three-letter airport code, for example KUU for Kullu or DEL for Delhi.
        """
        return WEATHER_ALERTS.get(airport_code.upper(), "No weather alerts.")

    def search_flights(origin: str, destination: str) -> str:
        """Search tomorrow's flights between two airports.

        Args:
            origin: Three-letter code of the departure airport, for example DEL.
            destination: Three-letter code of the arrival airport, for example KUU.
        """
        if origin.upper() == "DEL" and destination.upper() == "KUU":
            return "Tomorrow: " + "; ".join("%s departs %s arrives %s" % (f["flight"], f["departs"], f["arrives"]) for f in FLIGHTS_DEL_KUU)
        return "No flights found for that route in this demo."

    def get_baggage_policy(item: str) -> str:
        """Return the airline's baggage rules for a special item.

        Args:
            item: The item, for example snowboard.
        """
        for known, rule in BAGGAGE.items():
            if known in item.lower():
                return rule
        return "Standard baggage rules apply."

    def search_hotels_near_airport(airport_code: str) -> str:
        """Find hotels close to an airport for a stranded passenger.

        Args:
            airport_code: Three-letter airport code, for example DEL.
        """
        return HOTELS_NEAR.get(airport_code.upper(), "No hotels found in this demo.")

    return {f.__name__: f for f in (check_weather_alerts, search_flights, get_baggage_policy, search_hotels_near_airport)}


def flybot_policy(messages: list[dict]) -> dict:
    goal, calls, results = _turn_slice(messages)
    low = goal.lower()
    done = [r[0] for r in results]

    def call(name, args, thought):
        return {"text": thought, "tool_calls": [{"id": "", "name": name, "args": args}]}

    if "cancel" in low:
        if "check_weather_alerts" not in done:
            return call("check_weather_alerts", {"airport_code": "KUU"}, "First I confirm the snowstorm warning at Kullu.")
        if "search_flights" not in done:
            return call("search_flights", {"origin": "DEL", "destination": "KUU"}, "Now I look at tomorrow's flights.")
        if "get_baggage_policy" not in done:
            return call("get_baggage_policy", {"item": "snowboard"}, "He mentioned a snowboard, so I check the baggage rule.")
        return {"text": "I'm sorry about the cancellation. The weather alert says nothing moves at Kullu before 10:00, so 6E 2071 "
                        "(07:15) is not an option. I recommend **6E 2073**, departing 11:05 and arriving 12:30. Your snowboard is "
                        "oversized baggage: check it in at the special counter at least 2 hours before departure (by 09:05). "
                        "Shall I go ahead? Please confirm.", "tool_calls": []}
    if "hotel" in low:
        if "search_hotels_near_airport" not in done:
            return call("search_hotels_near_airport", {"airport_code": "DEL"}, "A hotel near Delhi airport is part of the same journey.")
        return {"text": "Two options near Delhi airport: **Aerocity Stay** (2 km, Rs 6,500, free shuttle) or **Terminal Inn** (4 km, Rs 3,900, "
                        "shuttle every 30 minutes). With a 11:05 flight tomorrow, either works. Which would you like?", "tool_calls": []}
    if "business lite" in low or "fare" in low:
        return {"text": "I don't have a tool that reads the Business Lite fare rules, so I can't state the allowance without guessing. "
                        "Please check the fare conditions on the airline's site.", "tool_calls": []}
    return {"text": "How can I help with your journey?", "tool_calls": []}


# ----------------------------------------------------------------------------------------
# Chapter 1: bare LLM replies (demo)
# ----------------------------------------------------------------------------------------
def bare_llm_reply(request: str, overconfident: bool = False) -> str:
    low = request.lower()
    if overconfident:
        if "flight" in low:
            return "Done! I've booked the cheapest IndiGo flight from Mumbai to Kullu for next Friday. Your confirmation number is 6E-88231."
        return "Your table for two at the pink café in Dadar is booked for Saturday at 7 pm, and I've added it to your calendar. Enjoy!"
    if "flight" in low:
        return ("I can't search live fares or book flights. For the cheapest IndiGo option, check IndiGo's website or an aggregator "
                "such as MakeMyTrip, and filter by price for next Friday.")
    return ("I can't make bookings or edit your calendar. You could call the café to reserve a table for two on Saturday at 7 pm, "
            "then add the event to your calendar yourself.")


# ----------------------------------------------------------------------------------------
# Chapter 2: passive vs proactive assistants (demo)
# ----------------------------------------------------------------------------------------
PROACTIVE_NOTES = {
    "manali": ["Late December brings heavy snow; are you ready for road closures on the way to Manali?",
               "Hotels and Volvo buses sell out for the New Year week; do you want to book them early?"],
    "goa": ["July is monsoon season in Goa and many beach shacks are closed; would indoor and spice-farm plans suit you?",
            "Is a sea-facing room worth it when swimming is restricted by rough water?"],
    "kyoto": ["Rain is expected in the shoulder seasons; shall I suggest indoor temples and museums?",
              "Is your passport valid for six months beyond the trip?"],
    "default": ["What is your budget, and have you checked that your documents are valid?",
                "Is the season right for what you want to do there, or should the dates move?"],
}


def assistant_reply(style: str, request: str) -> str:
    low = request.lower()
    place = next((p for p in PROACTIVE_NOTES if p in low), "default")
    dest = place.title() if place != "default" else "your destination"
    if style == "Passive":
        return ("Sure. For %s I can help plan transport, stays and sightseeing. Tell me which one to start with." % dest)
    n1, n2 = PROACTIVE_NOTES[place]
    return "Great goal. Two things before we plan: (1) %s (2) %s" % (n1, n2)


# ----------------------------------------------------------------------------------------
# Chapter 3.1: rule-based agents
# ----------------------------------------------------------------------------------------
CAFE_RULES = {
    "book": "Please send the date, time and number of guests.",
    "menu": "Our menu is at pinkcafe.example/menu",
    "timing": "We are open from 8 am to 11 pm.",
}
CAFE_MESSAGES = [
    "Can I book a table for 2 this Saturday at 7?",
    "My date doesn't drink coffee after 5, do you have anything without caffeine?",
    "Is it quiet enough on a Saturday evening to actually talk?",
]


def rule_bot_reply(rules: dict, message: str, fallback: str = "Sorry, I did not understand. Please call us.") -> tuple[str, str]:
    """Returns (reply, matched keyword or '')."""
    low = message.lower()
    for keyword, answer in rules.items():
        if keyword.strip() and keyword.lower() in low:
            return answer, keyword
    return fallback, ""


def demo_understand(message: str) -> dict:
    """Simulated structured understanding (what with_structured_output returns in live mode)."""
    low = message.lower()
    if re.search(r"\b(book|reserve|reservation|table for)\b", low):
        return {"message": message, "intent": "booking", "details": "table reservation request"}
    if re.search(r"caffeine|decaf|coffee|menu|drink|vegan|veg\b|food|eat|dessert|chai|tea", low):
        return {"message": message, "intent": "menu_question", "details": "looking for a caffeine-free option" if re.search(r"caffeine|decaf|without", low) else "asking about the menu"}
    if re.search(r"quiet|loud|crowd|noisy|ambien|talk|music|busy|vibe|romantic", low):
        return {"message": message, "intent": "ambience_question", "details": "wants a quiet place to talk" if re.search(r"quiet|talk", low) else "asking about the atmosphere"}
    return {"message": message, "intent": "other", "details": "not about booking, menu or ambience"}


UNDERSTANDING_SCHEMA = {
    "title": "message_understanding", "description": "What each café customer wants", "type": "object",
    "properties": {"messages": {"type": "array", "items": {"type": "object", "properties": {
        "message": {"type": "string"},
        "intent": {"type": "string", "enum": ["booking", "menu_question", "ambience_question", "other"]},
        "details": {"type": "string", "description": "The specific need, in a few words"}},
        "required": ["message", "intent", "details"]}}},
    "required": ["messages"],
}


# ----------------------------------------------------------------------------------------
# Chapter 3.2: reactive agents and memory you manage yourself
# ----------------------------------------------------------------------------------------
def memoryless_reply(history_sent: list[dict], question: str) -> str:
    """Demo: the model only 'knows' what is in the messages it was sent."""
    text = " ".join(m["content"] for m in history_sent)
    name = re.search(r"I'm (\w+)", text)
    drink = re.search(r"favourite drink is ([\w\s]+?)(?:\.|$)", text)
    if "name" in question.lower() or "drink" in question.lower():
        if name and drink:
            return "You're %s, and your favourite drink is %s." % (name.group(1), drink.group(1).strip())
        return "I'm sorry, I don't have any information about your name or your favourite drink. I can't remember earlier conversations."
    return "Could you tell me more?"


# ----------------------------------------------------------------------------------------
# Chapter 3.3: deliberative agents: planning, replanning, and search
# ----------------------------------------------------------------------------------------
PLAN_TOOLS = ["get_today", "check_calendar", "call_cafe", "add_calendar_event", "send_whatsapp"]
SURPRISES = {
    "The café is closed all day Sunday (private event)": "call_cafe returned: 'Sorry, we are closed all day this Sunday for a private event.'",
    "The café is fully booked on Saturday evening": "call_cafe returned: 'Saturday evening is fully booked, we only have a table after 9:30 pm.'",
    "Prasad's cricket match is cancelled": "check_calendar returned: 'The office cricket match on Saturday was cancelled.'",
}


def demo_plan() -> list[dict]:
    return [
        {"action": "Find today's date to resolve 'this weekend'", "tool": "get_today"},
        {"action": "Check Prasad's calendar for Saturday and Sunday", "tool": "check_calendar"},
        {"action": "Phone the pink café to book a table for two on a free evening", "tool": "call_cafe"},
        {"action": "Add the booking to Prasad's calendar", "tool": "add_calendar_event"},
        {"action": "Send the invitation to his match on WhatsApp", "tool": "send_whatsapp"},
    ]


def demo_replan(surprise_key: str) -> list[dict]:
    if surprise_key.startswith("The café is closed"):
        return [
            {"action": "Saturday 18:00-21:00 is blocked by cricket, so propose Saturday at 21:30 (the café is open until 23:00)", "tool": "none"},
            {"action": "Phone the café to book a table for two on Saturday at 21:30", "tool": "call_cafe"},
            {"action": "Add 'Dinner at the pink café, 21:30' to the calendar", "tool": "add_calendar_event"},
            {"action": "Send the invitation, mentioning the late start", "tool": "send_whatsapp"},
        ]
    if surprise_key.startswith("The café is fully booked"):
        return [
            {"action": "Accept the 21:30 table: the match ends at 21:00, so there is no clash", "tool": "call_cafe"},
            {"action": "Add 'Dinner at the pink café, 21:30' to the calendar", "tool": "add_calendar_event"},
            {"action": "Send the invitation and suggest a light snack before", "tool": "send_whatsapp"},
        ]
    return [
        {"action": "Re-check Saturday: the evening is now free", "tool": "check_calendar"},
        {"action": "Phone the café to book Saturday at 19:00, the originally preferred time", "tool": "call_cafe"},
        {"action": "Add the booking to the calendar", "tool": "add_calendar_event"},
        {"action": "Send the invitation", "tool": "send_whatsapp"},
    ]


PLAN_SCHEMA = {
    "title": "plan", "description": "An ordered plan to reach a goal", "type": "object",
    "properties": {"steps": {"type": "array", "items": {"type": "object", "properties": {
        "action": {"type": "string", "description": "What to do in this step"},
        "tool": {"type": "string", "description": "Which tool the step uses, or 'none'"}},
        "required": ["action", "tool"]}}},
    "required": ["steps"],
}

GRIDS = {
    "Open office": ["S.......", "........", "........", "........", "........", ".......G"],
    "Wall with a gap": ["S...#...", "....#...", "....#...", "........", "....#...", "....#..G"],
    "Maze": ["S.#.....", ".##.###.", "....#...", ".####.#.", "......#.", "####.#.G"],
}


def search_grid(grid: list[str], algo: str) -> dict:
    """BFS, DFS or A* on a small grid. Returns the order cells were explored and the final path."""
    rows, cols = len(grid), len(grid[0])
    start = goal = None
    for r, line in enumerate(grid):
        for c, ch in enumerate(line):
            if ch == "S":
                start = (r, c)
            elif ch == "G":
                goal = (r, c)

    def neighbours(cell):
        r, c = cell
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != "#":
                yield (nr, nc)

    def h(cell):
        return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

    parent, order, seen = {start: None}, [], {start}
    if algo == "BFS":
        frontier = deque([start])
        pop = frontier.popleft
        push = frontier.append
        nonempty = lambda: bool(frontier)
    elif algo == "DFS":
        frontier = [start]
        pop = frontier.pop
        push = frontier.append
        nonempty = lambda: bool(frontier)
    else:  # A*
        counter = 0
        cost = {start: 0}
        heap = [(h(start), h(start), counter, start)]
        found = False
        while heap:
            _, _, _, cell = heapq.heappop(heap)
            if cell in order:
                continue
            order.append(cell)
            if cell == goal:
                found = True
                break
            for nxt in neighbours(cell):
                new_cost = cost[cell] + 1
                if nxt not in cost or new_cost < cost[nxt]:
                    cost[nxt] = new_cost
                    parent[nxt] = cell
                    counter += 1
                    heapq.heappush(heap, (new_cost + h(nxt), h(nxt), counter, nxt))
        path = []
        if found:
            cell = goal
            while cell is not None:
                path.append(cell)
                cell = parent[cell]
            path.reverse()
        return {"order": order, "path": path, "explored": len(order), "found": found, "start": start, "goal": goal}

    found = False
    while nonempty():
        cell = pop()
        order.append(cell)
        if cell == goal:
            found = True
            break
        for nxt in neighbours(cell):
            if nxt not in seen:
                seen.add(nxt)
                parent[nxt] = cell
                push(nxt)
    path = []
    if found:
        cell = goal
        while cell is not None:
            path.append(cell)
            cell = parent[cell]
        path.reverse()
    return {"order": order, "path": path, "explored": len(order), "found": found, "start": start, "goal": goal}


def grid_html(grid: list[str], result: dict, upto: int | None = None) -> str:
    """Small HTML table: walls dark, explored cells blue, final path green."""
    order = result["order"][: (upto if upto is not None else len(result["order"]))]
    explored = set(order)
    path = set(result["path"]) if (upto is None or upto >= len(result["order"])) else set()
    cells = []
    for r, line in enumerate(grid):
        tds = []
        for c, ch in enumerate(line):
            bg, label = "#ffffff", ""
            if ch == "#":
                bg = "#343a40"
            elif (r, c) in path:
                bg = "#8ce99a"
            elif (r, c) in explored:
                bg = "#a5d8ff"
            if ch == "S":
                label, bg = "S", "#ffd43b"
            elif ch == "G":
                label, bg = "G", "#ff8787"
            tds.append('<td style="width:34px;height:34px;text-align:center;border:1px solid #adb5bd;background:%s;color:#212529;font-weight:700">%s</td>' % (bg, label))
        cells.append("<tr>%s</tr>" % "".join(tds))
    return '<table style="border-collapse:collapse;margin:0.4rem 0">%s</table>' % "".join(cells)


# ----------------------------------------------------------------------------------------
# Chapter 3.4: learning agents
# ----------------------------------------------------------------------------------------
LESSON_TASKS = {
    "Invite to the pink café (Saturday 9:30 pm)": "Write a message inviting my match to the pink café in Dadar this Saturday at 9:30 pm.",
    "Invite to a street-food walk in Girgaon (Sunday evening)": "Write a message asking her to join a street-food walk in Girgaon next Sunday evening.",
}
DEFAULT_FEEDBACK = ("Too long and too formal, she'll laugh at me. Also she's vegetarian and never drinks "
                    "coffee late at night, so don't suggest the espresso or anything with chicken.")


def demo_lessons(feedback: str) -> list[str]:
    """Simulated 'learning element': feedback -> up to three general rules."""
    low = feedback.lower()
    rules = []
    if re.search(r"long|short|formal|casual|laugh|stiff", low):
        rules.append("Keep messages short and casual, like a friend texting.")
    if re.search(r"vegetarian|veg\b|chicken|meat|egg|non-veg", low):
        rules.append("Never suggest non-vegetarian food: she is vegetarian.")
    if re.search(r"coffee|espresso|caffeine|tea", low):
        rules.append("Do not suggest coffee or caffeine late at night.")
    if re.search(r"emoji", low):
        rules.append("Use at most one emoji.")
    if re.search(r"cheap|budget|expensive", low):
        rules.append("Keep suggestions budget-friendly.")
    return rules[:3] or ["Follow the user's feedback in future messages."]


def demo_draft(task: str, lessons: list[str]) -> str:
    low = " ".join(lessons).lower()
    short = "short" in low or "casual" in low
    veg = "vegetarian" in low
    nocoffee = "coffee" in low or "caffeine" in low
    cafe = "caf" in task.lower()
    if cafe:
        drink = "an espresso" if not nocoffee else "a warm kesar milk"
        food = "a chicken sandwich" if not veg else "a cheese toastie"
        if short:
            if veg:
                return "Hey! Pink café, Dadar, Saturday 9:30? They do a great cheese toastie. I'm buying 🙂"
            return "Hey! Pink café, Dadar, Saturday 9:30? Espresso and chicken sandwiches on me 🙂"
        return ("Dear friend, I would be delighted if you would join me for an evening at the pink café in Dadar this Saturday at 9:30 pm. "
                "We could enjoy %s and %s while we converse at leisure. Kindly let me know whether this arrangement suits your schedule." % (drink, food))
    # street-food walk
    food = "chicken rolls and kebabs" if not veg else "vada pav and sev puri"
    if short:
        return "Street-food walk in Girgaon next Sunday evening? I know a stall for %s. In?" % food
    return ("Hello! I would like to cordially invite you to a street-food walk through Girgaon next Sunday evening. We shall sample %s "
            "and discover the area's culinary heritage. Please do let me know if you are available." % food)


def run_bandit(epsilon: float, rounds: int, seed: int = 7) -> dict:
    """Epsilon-greedy over three routes: the exploration vs exploitation dial."""
    routes = {"A: safe but slow": (6.0, 1.0), "B: risky shortcut": (8.0, 4.0), "C: long way round": (3.0, 0.5)}
    names = list(routes)
    rng = random.Random(seed)
    est = {n: 0.0 for n in names}
    cnt = {n: 0 for n in names}
    total, cumulative, picks = 0.0, [], []
    for _ in range(rounds):
        if rng.random() < epsilon:
            choice = rng.choice(names)
        else:
            best = max(est.values())
            choice = next(n for n in names if est[n] == best)
        mean, std = routes[choice]
        reward = rng.gauss(mean, std)
        cnt[choice] += 1
        est[choice] += (reward - est[choice]) / cnt[choice]
        total += reward
        cumulative.append(total)
        picks.append(choice)
    return {"cumulative": cumulative, "counts": cnt, "estimates": est, "picks": picks, "total": total,
            "true_means": {n: routes[n][0] for n in names}}


# ----------------------------------------------------------------------------------------
# Chapter 3.5: domain-specific agents (FlyBot 1.0, hand-coded rules)
# ----------------------------------------------------------------------------------------
FLYBOT_RULES = {
    "book a flight": "Sure. Please tell me your origin, destination and travel date.",
    "change my booking": "Please share your booking reference number.",
    "baggage allowance": "You can find our baggage policy at airline.example/baggage.",
}
FLYBOT_MESSAGES = {
    "Adaptability": ("Hi, my connecting flight from Delhi to Kullu was just canceled due to a snowstorm warning, and I'll miss the start of my trip. "
                     "I need to be rebooked on the next available flight, and you need to make sure I have enough time to re-check my snowboard."),
    "Scope": "Great, thanks. Can you recommend a good hotel near Delhi airport since I'm stuck here tonight?",
    "Knowledge": "What is the baggage allowance for the new Business Lite fare?",
}


def flybot_reply(message: str) -> str:
    low = message.lower()
    for keyword, answer in FLYBOT_RULES.items():
        if keyword in low:
            return answer
    if "cancel" in low:
        return "To rebook a cancelled flight, please visit airline.example/rebook."
    return "I'm sorry, I can only assist with flight-related queries."


# ----------------------------------------------------------------------------------------
# Chapter 5: why agents, and why not always
# ----------------------------------------------------------------------------------------
def chance_all_right(p: float, steps: int) -> float:
    return p ** steps


WORKFLOW_SCENARIOS = [
    # (scenario, correct answer, why)
    ("Nightly job: extract sales data, transform it, load it into the warehouse.", "Workflow",
     "The steps never change, so improvisation only adds risk."),
    ("A user types any request about their trip and expects the system to figure out what to do.", "Agent",
     "You cannot know in advance which task or tools will be needed."),
    ("Turn every uploaded photo of a metal part into a CAD file: describe, model, render, grade.", "Workflow",
     "You know exactly what to do with every image."),
    ("Payment processing: validate the card, charge, update the ledger, send a receipt.", "Workflow",
     "Reliability matters more than creativity; skipping a step causes real damage."),
    ("An assistant that handles whatever Prasad asks about his calendar and trips.", "Agent",
     "Open-ended goals with several possible tools."),
    ("Classify a support ticket, route it to the right team, draft a first reply.", "Workflow",
     "A fixed, predictable sequence with a routing step."),
]

FRAMEWORKS = [
    ("LangChain", "Building blocks for LLM apps with tools, memory and reasoning; create_agent"),
    ("LangGraph", "Graph-based, stateful control of single or multi-agent systems"),
    ("CrewAI", "Teams of role-based agents collaborating"),
    ("AutoGen", "Microsoft's open-source framework for collaborative multi-agent systems"),
    ("LlamaIndex", "Retrieval-first framework that brings your data into agent reasoning"),
    ("OpenAI Agents SDK", "Orchestrating teams of agents on OpenAI's platform"),
    ("Smolagents", "Hugging Face's lightweight library where agents act by writing code"),
    ("Atomic Agents", "A library for modular, composable agents"),
]


# ----------------------------------------------------------------------------------------
# Chapter 6.1: evaluator-optimizer (bio writer)
# ----------------------------------------------------------------------------------------
BIO_CRITERIA = {
    "concise": "Under 50 words.",
    "coffee_vector": "Mentions coffee and, playfully, vector search.",
    "no_cliche": "No cliches such as 'partner in crime', 'love to laugh' or 'work hard, play hard'.",
    "question": "Ends with a question that invites a reply.",
}
CLICHES = ["partner in crime", "love to laugh", "work hard, play hard"]


def compose_bio(flags: dict) -> str:
    parts = ["Prasad, 29, Dadar born and bred, built DilSeVector after years of watching friends swipe endlessly"]
    if not flags["concise"]:
        parts.append("and he spends every waking hour thinking about how people connect, how algorithms can help them, how startups grow, "
                     "how teams stay motivated and how a founder balances ambition with a life outside the office")
    parts.append("loves kesar cold coffee and thinks the best matches are nearest neighbours in vector space" if flags["coffee_vector"]
                 else "enjoys long walks and good conversation")
    if not flags["no_cliche"]:
        parts.append("is looking for a partner in crime who will love to laugh, because he believes in work hard, play hard")
    text = parts[0]
    for p in parts[1:]:
        text += ", " + p
    text += ". Which café should we try first?" if flags["question"] else "."
    return text


def bio_grade(bio: str, active: list[str]) -> dict:
    """Mechanical grader (plain code): which active criteria fail?"""
    low = bio.lower()
    results = {
        "concise": len(bio.split()) < 50,
        "coffee_vector": "coffee" in low and "vector" in low,
        "no_cliche": not any(c in low for c in CLICHES),
        "question": bio.strip().endswith("?"),
    }
    failed = [k for k in active if not results[k]]
    feedback = " ".join("Fix: " + BIO_CRITERIA[k] for k in failed)
    return {"accepted": not failed, "failed": failed, "feedback": feedback, "results": results, "words": len(bio.split())}


def run_eval_opt(active: list[str], max_rounds: int, backend: Backend | None = None) -> list[dict]:
    """Producer writes, grader checks, feedback goes back. Returns one dict per round."""
    flags = {"concise": False, "coffee_vector": True, "no_cliche": False, "question": False}
    rounds = []
    bio = compose_bio(flags)
    for n in range(1, max_rounds + 1):
        grade = bio_grade(bio, active)
        rounds.append({"round": n, "bio": bio, **grade})
        if grade["accepted"]:
            break
        for k in grade["failed"]:      # the producer reads the feedback and fixes exactly what was named
            flags[k] = True
        bio = compose_bio(flags)
    return rounds


# ----------------------------------------------------------------------------------------
# Chapter 6.2: parallelization (materials)
# ----------------------------------------------------------------------------------------
MATERIAL_CRITERIA = {
    "Metal": "high strength and stiffness, durability against wear and fatigue, thermal conductivity, machinability, recyclability",
    "Polymer": "lightweight, corrosion resistance, flexibility, low cost for mass production, impact resistance",
    "Ceramic": "high hardness and wear resistance, high temperature resistance, chemical inertness, dimensional stability, low ductility acceptable",
    "Composite": "high strength-to-weight ratio, tailored properties, corrosion resistance, complex shapes, cost versus performance",
    "Elastomer": "flexibility and elasticity, impact absorption, sealing and waterproofing, low temperature performance, durability under repeated deformation",
    "Wood": "aesthetic appearance, moderate strength, lightweight, machinability, thermal insulation, cost effectiveness, sustainability",
}
PARTS = {
    "Coffee-grinder burr": {
        "description": "A toothed disc about 64 mm across that crushes roasted beans all day, must stay sharp, be food-safe and not warm the coffee.",
        "scores": {"Metal": (9, "Hardened steel stays sharp and is food-safe, but conducts some heat into the beans."),
                   "Polymer": (2, "Too soft: teeth would wear down within days."),
                   "Ceramic": (9, "Extreme hardness, stays sharp for years and barely heats the coffee; brittle if a stone gets in."),
                   "Composite": (4, "Hard to make sharp, uniform teeth; the cost is not justified."),
                   "Elastomer": (1, "Cannot crush beans."), "Wood": (1, "Wears out and could taint the coffee.")},
    },
    "Phone case": {
        "description": "A slim shell that protects a phone from drops, must be light, cheap to mass-produce and slightly grippy.",
        "scores": {"Metal": (4, "Strong but heavy and blocks radio signals."),
                   "Polymer": (9, "Light, cheap, moulds into any shape and absorbs knocks."),
                   "Ceramic": (2, "Shatters on a drop."),
                   "Composite": (7, "Carbon-fibre cases are strong and thin but costly."),
                   "Elastomer": (8, "Silicone grips well and absorbs impact; can attract lint."), "Wood": (3, "Looks nice but cracks and is thick.")},
    },
    "Bicycle tyre": {
        "description": "A flexible ring that grips the road, must absorb bumps, handle repeated flexing and seal air.",
        "scores": {"Metal": (1, "Cannot flex."), "Polymer": (4, "Too stiff on its own."),
                   "Ceramic": (1, "Cannot flex."), "Composite": (5, "Used for reinforcement cords inside the tyre."),
                   "Elastomer": (10, "Rubber flexes millions of times, grips and seals air."), "Wood": (1, "Cannot flex.")},
    },
}


# ----------------------------------------------------------------------------------------
# Chapter 6.3: chaining (write, then revise)
# ----------------------------------------------------------------------------------------
DRAFT_ARTICLE = (
    "In today's fast-paced world, prompt engineering is a game-changer! 🚀 As an AI language model, I can tell you that a prompt is the "
    "text you give a model, and its wording changes the answer. Let's dive in. Good prompts state the role, the task and the format. "
    "Few-shot examples show the model the pattern to copy, and chain-of-thought asks it to reason step by step. 😊 Test every prompt on "
    "real inputs and keep the ones that work. It's a total game-changer for developers who want reliable output."
)
CRINGE = ["in today's fast-paced world", "game-changer", "let's dive in", "dive in", "unlock the power"]


def count_violations(text: str) -> dict:
    low = text.lower()
    emojis = len(re.findall(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", text))
    ai = len(re.findall(r"as an ai (language )?model|i am an ai|as a language model", low))
    cringe = sum(low.count(c) for c in CRINGE)
    return {"emojis": emojis, "ai mentions": ai, "cringe phrases": cringe}


def revise_article(text: str) -> str:
    out = re.sub(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]\uFE0F?", "", text)
    out = re.sub(r"As an AI( language)? model, I can tell you that a prompt", "A prompt", out)
    out = out.replace("In today's fast-paced world, ", "").replace("Let's dive in. ", "")
    out = out.replace("is a game-changer!", "is a core skill of applied machine learning.")
    out = out.replace("It's a total game-changer for developers who want reliable output.", "This discipline matters most to developers who need reliable output.")
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out[:1].upper() + out[1:]


# ----------------------------------------------------------------------------------------
# Chapter 6.4: routing (video genres)
# ----------------------------------------------------------------------------------------
GENRE_PROMPTS = {
    "Entertainment": "Write a high-energy, culturally-relevant script about [TOPIC] using trendy language and engaging hooks that balance entertainment value with insider insights.",
    "Educational": "Develop a clear, engaging script about [TOPIC] that transforms complex information into digestible insights using relatable examples and thought-provoking questions.",
    "Comedy": "Create a sharp, unexpected script about [TOPIC] that subverts expectations with clever observations, perfect timing, and relatable scenarios that build to satisfying punchlines.",
    "Personal vlog": "Write an authentic, intimate script about [TOPIC] that balances vulnerability with relatable insights, using conversational storytelling that invites viewers into your personal experience.",
    "Reviews": "Create a decisive, experience-based script about [TOPIC] highlighting key strengths and weaknesses with specific examples that support your clear final verdict.",
    "Storytelling": "Write an immersive script about [TOPIC] using vivid details and strategic pacing that builds emotional connection through authentic storytelling.",
}
GENRE_KEYWORDS = {
    "Educational": ["explain", "what is", "how does", "how to", "learn", "tutorial", "guide", "in 60 seconds", "embedding", "python", "function"],
    "Comedy": ["funny", "joke", "fails", "awkward", "roast", "hilarious", "parody"],
    "Personal vlog": ["i met", "my day", "how i", "my journey", "i quit", "my life", "partner"],
    "Reviews": ["review", "vs", "worth it", "best", "unboxing", "rating", "compared"],
    "Storytelling": ["story", "once upon", "legend", "history of", "tale", "the night"],
    "Entertainment": ["trend", "viral", "celebrity", "reaction", "top 10", "surfing", "challenge"],
}
ROUTER_SCHEMA = {"title": "route", "description": "The genre of a short video topic", "type": "object",
                 "properties": {"genre": {"type": "string", "enum": list(GENRE_PROMPTS)}}, "required": ["genre"]}


def demo_route(topic: str) -> dict:
    low = topic.lower()
    scores = {g: sum(1 for k in kws if k in low) for g, kws in GENRE_KEYWORDS.items()}
    best = max(scores.values())
    genre = next((g for g in GENRE_KEYWORDS if scores[g] == best), "Entertainment") if best > 0 else "Entertainment"
    return {"genre": genre, "scores": scores}


SCRIPT_OPENERS = {
    "Entertainment": "HOOK: You will not believe this. {t}. Quick cuts, trending sound, big text on screen. Finish with a challenge to the viewer.",
    "Educational": "Scene 1: A simple question on screen. Scene 2: {t}, explained with one everyday example. Scene 3: ask the viewer one question to check they got it.",
    "Comedy": "Setup: a calm voice-over about {t}. Twist: the obvious thing goes wrong. Punchline: it was the viewer's fault all along.",
    "Personal vlog": "Camera at arm's length, soft light. 'So, this is the story of {t}, and I was nervous about telling it.' Talk like to one friend.",
    "Reviews": "Verdict first: {t}. Then two strengths, two weaknesses, one example each, and a clear score out of ten.",
    "Storytelling": "Slow open on one detail. A character wants something. {t}. A turn near the end that changes how we see the start.",
}


def demo_script(genre: str, topic: str) -> str:
    return SCRIPT_OPENERS[genre].format(t=topic.rstrip("."))


# ----------------------------------------------------------------------------------------
# Chapter 8: the travel planner (pretend Manali guide, loosely modelled on a travel guide)
# ----------------------------------------------------------------------------------------
GUIDE = {
    "Hadimba Devi Temple": {"kind": "attraction", "quiet": True, "note": "Wooden cedar-forest temple, a short walk from Mall Road. Quietest early morning."},
    "Old Manali": {"kind": "attraction", "quiet": True, "note": "Across the river: riverside cafés, guesthouses and an easy stroll over the bridge."},
    "Vashisht Hot Springs": {"kind": "attraction", "quiet": True, "note": "Village with natural hot-spring baths and a stone temple, a short ride from town."},
    "Manu Temple": {"kind": "attraction", "quiet": True, "note": "Small temple in Old Manali, a gentle uphill walk."},
    "Solang Valley": {"kind": "attraction", "quiet": False, "note": "Adventure hub, crowded in season, 14 km from town."},
    "Mall Road": {"kind": "attraction", "quiet": False, "note": "Main shopping street, crowded and noisy in the evening."},
    "Beas River walk": {"kind": "attraction", "quiet": True, "note": "Easy riverside walk near Old Manali."},
    "Old Manali riverside cafés": {"kind": "food", "quiet": True, "note": "Cafés along the river with vegetarian options."},
    "Pure-veg dhaba near Mall Road": {"kind": "food", "quiet": False, "note": "Simple pure-vegetarian thali, busy at lunch."},
    "Snow Valley Cottage": {"kind": "stay", "quiet": True, "note": "Quiet stay with a pure-veg kitchen."},
}
GUIDE_NAMES = list(GUIDE)
NOT_IN_GUIDE = {
    "Skyline Rooftop Café": "Invented by the model: no such place in the guide.",
    "Rohtang Pass snow point": "Roads to Rohtang are normally closed in late December: stale or wrong for the season.",
    "Sethan village sunrise point": "Added by the optimizer as a 'hidden gem'. It sounds plausible, but nothing in the research mentions it.",
}


def _item(place, text, time=""):
    return {"place": place, "text": text, "time": time}


def itinerary_genai() -> list[dict]:
    return [
        {"day": 1, "items": [_item("Hadimba Devi Temple", "Start at the cedar-forest temple.", "Morning"),
                             _item("Mall Road", "Browse the shops and street food.", "Afternoon"),
                             _item("Skyline Rooftop Café", "Dinner with a valley view.", "Evening")]},
        {"day": 2, "items": [_item("Rohtang Pass snow point", "Full-day snow trip to the pass.", "All day")]},
        {"day": 3, "items": [_item("Solang Valley", "Adventure activities.", "Morning"),
                             _item("Old Manali", "Evening wander through the cafés.", "Evening")]},
    ]


def itinerary_single_agent() -> list[dict]:
    return [
        {"day": 1, "items": [_item("Hadimba Devi Temple", "Walk through the cedar forest to the temple.", "Morning"),
                             _item("Mall Road", "Lunch and a stroll.", "Afternoon"),
                             _item("Old Manali", "Cross the bridge for the evening.", "Evening")]},
        {"day": 2, "items": [_item("Solang Valley", "Day trip to the valley.", "Morning"),
                             _item("Pure-veg dhaba near Mall Road", "Thali on the way back.", "Afternoon")]},
        {"day": 3, "items": [_item("Vashisht Hot Springs", "Soak in the hot springs.", "Morning"),
                             _item("Manu Temple", "Short walk up to the temple.", "Afternoon")]},
    ]


def role_outputs(pass_prefs_to_planner: bool = False, qa_sees_research: bool = False, prefs: str = "vegetarian, quiet, short walks") -> dict:
    """The four-role planner. Each role is a prompt; here is what each would typically hand to the next."""
    research = ("Attractions: " + "; ".join("%s (%s)" % (n, GUIDE[n]["note"]) for n in GUIDE_NAMES if GUIDE[n]["kind"] == "attraction")
                + ". Food: " + "; ".join("%s (%s)" % (n, GUIDE[n]["note"]) for n in GUIDE_NAMES if GUIDE[n]["kind"] == "food")
                + ". Stay: Snow Valley Cottage (quiet, pure-veg kitchen). Hotel prices: not covered by the guide.")
    if pass_prefs_to_planner:
        draft = [
            {"day": 1, "items": [_item("Hadimba Devi Temple", "Walk through the cedar forest.", "Morning"),
                                 _item("Old Manali riverside cafés", "Vegetarian lunch by the river.", "Afternoon"),
                                 _item("Manu Temple", "Short walk up to the temple.", "Evening")]},
            {"day": 2, "items": [_item("Vashisht Hot Springs", "Hot springs and village.", "Morning"),
                                 _item("Beas River walk", "Easy riverside walk.", "Afternoon")]},
            {"day": 3, "items": [_item("Old Manali", "Slow morning, cafés.", "Morning"),
                                 _item("Snow Valley Cottage", "Rest and early dinner (pure-veg kitchen).", "Evening")]},
        ]
    else:
        draft = [
            {"day": 1, "items": [_item("Hadimba Devi Temple", "Walk through the cedar forest.", "Morning"),
                                 _item("Mall Road", "Shopping and food.", "Afternoon"),
                                 _item("Old Manali", "Evening in the cafés.", "Evening")]},
            {"day": 2, "items": [_item("Solang Valley", "Adventure day.", "Morning"),
                                 _item("Pure-veg dhaba near Mall Road", "Thali.", "Afternoon")]},
            {"day": 3, "items": [_item("Vashisht Hot Springs", "Hot springs.", "Morning"),
                                 _item("Manu Temple", "Temple walk.", "Afternoon")]},
        ]
    # optimizer: apply preferences (drop noisy places) and "add small enhancements" from memory
    optimized = []
    for d in draft:
        items = [dict(i) for i in d["items"] if GUIDE.get(i["place"], {}).get("quiet", True)]
        optimized.append({"day": d["day"], "items": items})
    if not any(i["place"] == "Old Manali riverside cafés" for i in optimized[0]["items"]):
        optimized[0]["items"].insert(1, _item("Old Manali riverside cafés", "Vegetarian lunch by the river.", "Afternoon"))
    if len(optimized[1]["items"]) < 2:
        optimized[1]["items"].insert(0, _item("Beas River walk", "Easy riverside walk.", "Morning"))
    optimized[1]["items"].append(_item("Sethan village sunrise point", "A hidden gem for a quiet sunset. Save some energy for Day 3 because it has the longest walk.", "Evening"))
    # QA
    if qa_sees_research:
        verified = [{"day": d["day"], "items": [dict(i) for i in d["items"] if i["place"] in GUIDE]} for d in optimized]
    else:
        verified = [{"day": d["day"], "items": [dict(i) for i in d["items"]]} for d in optimized]
    return {"research": research, "draft": draft, "optimized": optimized, "verified": verified}


def itinerary_text(days: list[dict]) -> str:
    lines = ["Final Answer:"]
    for d in days:
        body = " ".join("%s: %s %s" % (i["time"], i["place"], i["text"]) if i["time"] else "%s %s" % (i["place"], i["text"]) for i in d["items"])
        lines.append("Day %d: %s" % (d["day"], body))
    return "\n".join(lines)


def grounding_check(days: list[dict]) -> dict:
    found, missing = [], []
    for d in days:
        for i in d["items"]:
            (found if i["place"] in GUIDE else missing).append(i["place"])
    return {"found_in_research": found, "not_in_research": missing}


def generate_ics(plan_text: str, start: date | None = None, structured_days: list[dict] | None = None) -> tuple[bytes, int]:
    """
    Minimal .ics writer (no extra package). With `structured_days` it makes exactly one event per day (the fix for HW7).
    Without it, it splits the text with the notebook's regular expression, which can be fooled by 'Day 3' inside Day 2.
    """
    start = start or date.today()
    events = []
    if structured_days is not None:
        for d in structured_days:
            body = " ".join("%s %s" % (i["place"], i["text"]) for i in d["items"])
            events.append((d["day"], body))
    else:
        text = plan_text.replace("Final Answer:", "").strip()
        found = re.findall(r"Day (\d+)[:\s]+(.*?)(?=Day \d+|$)", text, flags=re.DOTALL)
        if not found:
            events.append((1, text))
        else:
            events = [(int(n), body.strip()) for n, body in found]

    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")

    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Agentic FlyTripper//EN"]
    for n, body in events:
        day = start + timedelta(days=n - 1)
        lines += ["BEGIN:VEVENT", "UID:day%d-%s@dilsetrips.example" % (n, stamp), "DTSTAMP:" + stamp,
                  "DTSTART;VALUE=DATE:" + day.strftime("%Y%m%d"), "DTEND;VALUE=DATE:" + (day + timedelta(days=1)).strftime("%Y%m%d"),
                  "SUMMARY:" + esc("Day %d Itinerary" % n), "DESCRIPTION:" + esc(body), "END:VEVENT"]
    lines.append("END:VCALENDAR")
    return ("\r\n".join(lines) + "\r\n").encode("utf-8"), len(events)


def search_travel_guide(place: str) -> str:
    """Real Wikivoyage lookup (needs internet). Used only when the person presses the 'real lookup' button or runs Live mode."""
    import requests
    response = requests.get("https://en.wikivoyage.org/w/api.php",
                            params={"action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1, "titles": place, "format": "json"},
                            headers={"User-Agent": "GenAI-course-agents-lab/1.0 (educational use)"}, timeout=20)
    pages = response.json()["query"]["pages"]
    page_text = ""
    for pid in pages:
        if "extract" in pages[pid]:
            page_text = pages[pid]["extract"]
    if not page_text:
        return "No Wikivoyage page found for '%s'." % place
    wanted = ["Understand", "Get in", "Get around", "See", "Do", "Eat", "Sleep", "Stay safe"]
    result, current = "Wikivoyage guide for %s:\n" % place, ""
    for line in page_text.split("\n"):
        if line.startswith("== ") and line.endswith(" =="):
            current = line[3:-3].strip()
            if current in wanted:
                result += "\n## %s\n" % current
        elif current in wanted and len(result) < 8000:
            result += line + "\n"
    return result
