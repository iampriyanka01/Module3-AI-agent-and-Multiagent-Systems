"""Live demos with five couples.   Run:   python demo_couples.py

Each couple teaches DilSeTrips something different. Then the same agents behave differently for each of them,
because only their memories differ.
"""

from datetime import date, timedelta

from config import missing_keys

if len(missing_keys()) > 0:
    print("Missing in your .env file:", ", ".join(missing_keys()))
    raise SystemExit

# Imported only after the keys are checked
import agents
import coordinator
import memory

COUPLES = [
    {"id": "prasad", "names": "Prasad and Ananya (Mumbai)",
     "teach": ["We hate long layovers. A direct flight is worth paying a bit more for.", "Ananya is vegetarian.",
               "We always want window seats."],
     "trap": "No sugar in my chai, please.", "trap_lesson": "must NOT become 'Prasad is diabetic'",
     "request": "Hi! Please book us a flight from Mumbai to Chandigarh on {date}. What will the weather be like there? "
                "Oh, and we always want window seats."},
    {"id": "arjun", "names": "Arjun and Neha (Delhi)",
     "teach": ["We're saving for a house, so the cheapest option always wins, even with a stop.",
               "Neha is strictly vegetarian, no eggs either.", "We like flying after work, so evening flights suit us."],
     "trap": "Ugh, Mondays at the office are the worst.", "trap_lesson": "small talk: nothing should be saved",
     "request": "Find us flights from Delhi to Goa on {date}, as cheap as possible please."},
    {"id": "kabir", "names": "Kabir and Zoya (Bengaluru)",
     "teach": ["We never take flights before 10 in the morning. We are not morning people.",
               "We love nightlife and live music.", "We usually travel with just cabin bags."],
     "trap": "Zoya said she might want to try scuba diving someday, maybe.", "trap_lesson": "too vague to save",
     "request": "We want to fly from Bengaluru to Goa on {date} for a weekend of beaches and live music. Find us a hotel too."},
    {"id": "rohan", "names": "Rohan and Meera (Chennai)",
     "teach": ["Rohan uses a wheelchair, so we need airlines that handle wheelchair assistance well.",
               "We'd rather take one long direct flight than any connection. Transfers are really hard for us.",
               "Meera loves temples and old architecture."],
     "trap": "My mother thinks we travel too much, haha.", "trap_lesson": "someone else's opinion: nothing to save",
     "request": "Plan our anniversary trip from Chennai to Jaipur on {date}: flights, a hotel, and what we should see."},
    {"id": "sana", "names": "Sana and Dev (Kolkata)",
     "teach": ["We're early birds. The earlier the flight, the better.", "Dev is allergic to peanuts.",
               "We love trekking and the mountains."],
     "trap": "My passport number is K9876543, in case you need it.", "trap_lesson": "identity numbers must never be saved",
     "request": "Find us a flight from Kolkata to Bagdogra on {date}, and tell us what to see around Darjeeling."},
]


def future_date(days_ahead):
    """A date in the future, as words ('20 December') and as YYYY-MM-DD."""
    day = date.today() + timedelta(days=days_ahead)
    return f"{day.day} {day.strftime('%B')}", day.isoformat()


def teach_couples():
    """Demo 1: each couple tells DilSeTrips about themselves. Watch what is saved, and what is refused."""
    for couple in COUPLES:
        print("\n" + "=" * 70 + f"\n{couple['names']}\n" + "=" * 70)
        memory.forget(couple["id"])                       # start fresh
        for sentence in couple["teach"]:
            print(f'They say: "{sentence}"')
            memory.remember(couple["id"], sentence)
        print(f'\nTRAP: "{couple["trap"]}"   (expected: {couple["trap_lesson"]})')
        saved = memory.remember(couple["id"], couple["trap"])
        if len(saved) == 0:
            print("Result: nothing saved. Trap avoided.")
        else:
            print(f"Result: SAVED {saved}. Discuss: was that right?")


def show_what_they_taught():
    """Demo 2: everything DilSeTrips now remembers about each couple (no AI calls: Pinecone only)."""
    for couple in COUPLES:
        print(f"\n{couple['names']}:")
        saved = memory.show_memories(couple["id"])
        if len(saved) == 0:
            print("  nothing yet (run demo 1 first)")
        for item in saved:
            print(f"  - {item.get('text')}   (importance {item.get('importance')})")


def same_request_for_everyone():
    """Demo 3: the SAME flight request for every couple. Only their memories differ."""
    words, iso = future_date(45)
    request = f"Find us a flight for two from Mumbai to Goa on {iso}."
    print(f'Every couple sends: "{request}"')
    for couple in COUPLES:
        print("\n" + "=" * 70 + f"\n{couple['names']}\n" + "=" * 70)
        agents.trip["couple"] = couple["id"]
        report = agents.run_agent(agents.flight_agent, request, "flight agent")
        print("\n" + report)


def full_conversation():
    """Demo 4: one couple's own request, through the whole system."""
    for number in range(len(COUPLES)):
        print(f"  {number + 1}. {COUPLES[number]['names']}")
    choice = input("Which couple (1 to 5)? ").strip()
    if choice not in ["1", "2", "3", "4", "5"]:
        print("Please type a number from 1 to 5.")
        return
    couple = COUPLES[int(choice) - 1]
    words, iso = future_date(45)
    message = couple["request"].format(date=words)
    print(f"\n{couple['names']}: {message}\n")
    reply = coordinator.handle_message(couple["id"], message, [])
    print("\nDilSeTrips:", reply)


def learn_then_use():
    """Demo 5: Kabir and Zoya mention a new preference, and the very next request uses it."""
    words, iso = future_date(45)
    conversation = []
    first = "Quick thing: we can't stand flights that land after midnight."
    print(f"Kabir and Zoya: {first}")
    reply = coordinator.handle_message("kabir", first, conversation)
    conversation.append("couple: " + first)
    conversation.append("DilSeTrips: " + reply)
    print("\nDilSeTrips:", reply)
    second = f"Find us a flight from Bengaluru to Goa on {words}."
    print(f"\nKabir and Zoya: {second}")
    print("\nDilSeTrips:", coordinator.handle_message("kabir", second, conversation))


if __name__ == "__main__":
    while True:
        print("\nDemos:")
        print("  1. Teach the five couples (watch the traps)")
        print("  2. Show what each couple taught DilSeTrips")
        print("  3. The same flight request for every couple")
        print("  4. One couple's full trip request")
        print("  5. Learn a preference, then use it")
        print("  q. Quit")
        choice = input("Choose: ").strip().lower()
        if choice == "q":
            break
        elif choice == "1":
            teach_couples()
        elif choice == "2":
            show_what_they_taught()
        elif choice == "3":
            same_request_for_everyone()
        elif choice == "4":
            full_conversation()
        elif choice == "5":
            learn_then_use()
        else:
            print("Please choose 1 to 5, or q.")
