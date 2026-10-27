"""One message, from the couple's words to DilSeTrips' reply, in seven steps.

  1. Reflex       obvious emergency words get an instant safety reply, with no AI at all      (Session 4)
  2. Understand   the LLM reads the message into structured facts                             (Session 2)
  3. Check        urgent? missing details? then reply straight away                           (Session 4)
  4. Delegate     the flight agent first, then the destination agent, which can ask it        (Session 5)
  5. Reply        the LLM writes one warm reply from the agents' reports
  6. Reflect      the LLM checks the reply against the reports, and fixes mistakes            (Session 4)
  7. Learn        anything worth remembering is saved to Pinecone                             (Session 3)
"""

from datetime import date

import agents
import memory
from config import PROMPTS, llm

SAFETY_REPLY = ("If anyone is in danger, call 112 now (India's emergency number). Stay somewhere safe and share "
                "your live location with someone you trust. DilSeTrips support will call you right away.")
REFLEX_WORDS = ["accident", "emergency", "injured", "stranded", "broke down", "broken down"]

# The shape of the LLM's answer when it reads a message
TRIP_SCHEMA = {
    "title": "trip_request",
    "description": "What the couple is asking for",
    "type": "object",
    "properties": {
        "urgent": {"type": "boolean", "description": "True only if someone may be in danger or stranded right now"},
        "wants_flights": {"type": "boolean"},
        "origin": {"type": "string", "description": "City they fly from, or empty"},
        "destination": {"type": "string", "description": "City they go to, or empty"},
        "travel_date": {"type": "string", "description": "YYYY-MM-DD, or empty"},
        "travellers": {"type": "integer"},
        "wants_booking": {"type": "boolean"},
        "wants_destination_info": {"type": "boolean", "description": "Weather, places, food or the first day"},
        "wants_hotel": {"type": "boolean"},
        "nights": {"type": "integer", "description": "Nights of stay, or 0 if not said"},
        "needs_more_info": {"type": "boolean", "description": "True if a key detail is missing"},
        "question_to_ask": {"type": "string", "description": "One short, warm question, if a detail is missing"},
    },
    "required": ["urgent", "wants_flights", "wants_booking", "wants_destination_info", "wants_hotel", "needs_more_info"],
}

# The shape of the LLM's answer when it checks the reply
CHECK_SCHEMA = {
    "title": "reply_check",
    "description": "Whether the reply only uses facts from the reports",
    "type": "object",
    "properties": {
        "ok": {"type": "boolean"},
        "problems": {"type": "array", "items": {"type": "string"}},
        "corrected_reply": {"type": "string", "description": "A fixed reply if ok is false, otherwise empty"},
    },
    "required": ["ok", "problems", "corrected_reply"],
}


def handle_message(couple, message, conversation):
    """Run one message through DilSeTrips and return the reply."""
    agents.trip["couple"] = couple
    agents.trip["flight_report"] = ""
    agents.say("couple", "coordinator", message)

    # 1. Reflex: no AI needed for an obvious emergency
    for word in REFLEX_WORDS:
        if word in message.lower():
            print(f"   [reflex] emergency word '{word}': answering instantly")
            return SAFETY_REPLY

    # 2. Understand: the LLM turns the message into structured facts
    recent = "\n".join(conversation[-6:])
    request = llm.with_structured_output(TRIP_SCHEMA).invoke(
        PROMPTS["perception"].format(today=date.today().isoformat(), conversation=recent, message=message))
    print(f"   [understood] {request}")

    # 3. Check: urgent, or a key detail missing?
    if request["urgent"]:
        print("   [triage] the LLM judged this urgent")
        return SAFETY_REPLY
    if request["needs_more_info"] and request.get("question_to_ask", "") != "":
        print("   [clarify] a key detail is missing: asking")
        memory.remember(couple, message)
        return request["question_to_ask"]

    # 4. Delegate: flights first, because the destination agent may need the landing time
    reports = ""
    if request["wants_flights"]:
        task = (f"The couple wrote: {message}\nFrom {request.get('origin', '')} to {request.get('destination', '')} on "
                f"{request.get('travel_date', '')}, {request.get('travellers', 2)} travellers. "
                f"They want to book: {request['wants_booking']}.")
        agents.say("coordinator", "flight agent", task)
        report = agents.run_agent(agents.flight_agent, task, "flight agent")
        agents.trip["flight_report"] = report
        agents.say("flight agent", "coordinator", report)
        reports = reports + "FLIGHT AGENT REPORT:\n" + report + "\n\n"
    if request["wants_destination_info"] or request["wants_hotel"]:
        task = (f"The couple wrote: {message}\nDestination: {request.get('destination', '')}, travel date: "
                f"{request.get('travel_date', '')}, {request.get('travellers', 2)} travellers. "
                f"They want a hotel: {request['wants_hotel']}. Nights: {request.get('nights', 0)} (0 means not said).")
        agents.say("coordinator", "destination agent", task)
        report = agents.run_agent(agents.destination_agent, task, "destination agent")
        agents.say("destination agent", "coordinator", report)
        reports = reports + "DESTINATION AGENT REPORT:\n" + report + "\n\n"
    if reports == "":
        reports = "(No agent was needed for this message.)"

    # 5. Reply: one warm message from the reports
    reply = llm.invoke(PROMPTS["reply"].format(today=date.today().strftime("%A %d %B %Y"), conversation=recent,
                                               message=message, reports=reports)).text

    # 6. Reflect: check the reply against the reports (only when agents produced facts to check)
    if reports != "(No agent was needed for this message.)":
        check = llm.with_structured_output(CHECK_SCHEMA).invoke(
            PROMPTS["check"].format(message=message, reports=reports, reply=reply))
        if check["ok"]:
            print("   [reflect] the reply matches the reports")
        else:
            print(f"   [reflect] fixed: {check['problems']}")
            if check["corrected_reply"] != "":
                reply = check["corrected_reply"]

    # 7. Learn: save anything worth remembering
    memory.remember(couple, message)
    return reply
