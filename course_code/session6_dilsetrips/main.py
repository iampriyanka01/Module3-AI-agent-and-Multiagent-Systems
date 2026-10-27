"""Chat with DilSeTrips.   Run:   python main.py

Commands while chatting:  memories  (what DilSeTrips remembers)   forget  (delete it all)   quit
"""

from config import missing_keys

missing = missing_keys()
if len(missing) > 0:
    print("Missing in your .env file:", ", ".join(missing))
    print("Copy .env.example to .env and add your keys, then run again.")
    raise SystemExit

# Imported only after the keys are checked: these files connect to Gemini and Pinecone as soon as they load
import coordinator
import memory

couple = input("Which couple are you? (for example: prasad) ").strip().lower()
if couple == "":
    couple = "prasad"
conversation = []          # short-term memory: this chat's messages (Session 3)

print(f"\nHi! You're chatting as '{couple}'. Commands: memories, forget, quit.")
while True:
    message = input("\nYou: ").strip()
    if message == "quit":
        break
    if message == "":
        continue
    if message == "memories":
        for saved in memory.show_memories(couple):
            print(f"  - {saved.get('text')}   (importance {saved.get('importance')}, from: \"{saved.get('quote')}\")")
        continue
    if message == "forget":
        memory.forget(couple)
        print("  Everything DilSeTrips remembered about you is deleted.")
        continue

    reply = coordinator.handle_message(couple, message, conversation)
    conversation.append("couple: " + message)
    conversation.append("DilSeTrips: " + reply)
    print("\nDilSeTrips:", reply)
