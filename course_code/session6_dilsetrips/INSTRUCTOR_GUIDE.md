# DilSeTrips: instructor guide

Students build DilSeTrips file by file. After each file they run one command and see something real happen. About 3 hours of building, then 30 minutes of demos.

## Before class

**Students (the day before):** Python 3.10 to 3.13, and three free keys: Gemini, Pinecone (Starter plan), and LiteAPI (copy the **private Sandbox Key**, not the public or Stripe key). Flights and hotels work in LiteAPI's sandbox with no access request.

**You (the day before):**
1. Set up `.env` and run `python tools.py`. You should see weather, a guide extract, an airport code, flights and hotels.
2. Run `python demo_couples.py`, choose **1** (teach the couples, about 20 Gemini calls) and **3** (the same request for everyone). Read the output.
3. LiteAPI's docs warn that sandbox flight data can be thin on some routes. If a demo route returns nothing, keep a route that works in reserve.

**Budget:** a full trip request is about 15 Gemini calls, so 3 to 5 minutes on the free tier.

## The build, file by file

| # | File | Check by running | What students learn | Session |
|---|---|---|---|---|
| 1 | `requirements.txt`, `.env` | `pip install -r requirements.txt` | Keys live in `.env`, never in code or on GitHub | |
| 2 | `config.py` | `python -c "import config; print(config.missing_keys())"` should print `[]` | The model is only the brain; one shared model with a rate limit | 1, 2 |
| 3 | `tools.py` | `python tools.py` | Tools are plain Python: the agent's senses and hands. No AI here | 1, 2 |
| 4 | `prompts.yaml` | Read the perception and flight agent prompts together | A good prompt has a role, a goal, steps, judgement with examples, and limits | 2 |
| 5 | `memory.py` | `python memory.py` | Vector search finds similar memories; the AI judges which matter; code checks the AI | 3 |
| 6 | `agents.py` | `python agents.py` (after teaching Prasad a preference, below) | An agent is a model + tools + instructions, running a loop | 1, 4 |
| 7 | `coordinator.py` | (used by main.py) | Seven steps: reflex, understand, check, delegate, reply, reflect, learn | 4, 5 |
| 8 | `main.py` | `python main.py`, then the example message | The whole system working together | all |

To give Prasad a preference before step 6:

```bash
python -c "import memory; memory.remember('prasad', 'We hate long layovers, and Ananya is vegetarian.')"
```

## What to point out at each step

**`tools.py`:** run it and show real flight numbers and hotel names. Point out `find_hotels` does three steps (hotel list, room prices, then join them by id): that join is plain code, and it's where many real bugs live.

**`prompts.yaml`:** read the perception prompt aloud. The JSON schema in `coordinator.py` says *what shape* the answer has; the prompt teaches *how to decide* what goes in it. Show the good and bad clarifying questions side by side.

**`memory.py`:** run it. Students should see two memories saved, then "No sugar in my chai" refused: the AI might guess "diabetic", but the quote check in code stops it. That's Session 3's hallucination problem, solved with three lines of Python. Then point at `recall()`: Pinecone finds *similar* memories, and the AI decides which are *useful* ("vegetarian" doesn't matter for choosing a flight).

**`agents.py`:** run it and read the `used ...` lines: that's the ReAct loop (Session 4). Look for `direct_only: True`: the agent recalled "hate long layovers" and chose that filter itself. Then `ask_flight_agent`: one agent asking another (Session 5).

**`coordinator.py`:** walk through the seven numbered steps. The key sentence: *every step that needs judgement is an AI call; every step that needs certainty is code.*

**`main.py`:** send the example message, then type `memories`. "Window seats" should be there, with their own words as the source.

## Where each session appears

| Session | Idea | Where |
|---|---|---|
| 1 | An agent is a loop with tools, not just a model | `agents.py` |
| 2 | Perception: turning a message into structured facts | step 2 in `coordinator.py` |
| 2 | PEAS: tools are sensors and actuators | `tools.py` |
| 3 | Short-term memory (the conversation) and long-term memory (Pinecone) | `main.py`, `memory.py` |
| 3 | Hallucination, privacy and forgetting | the quote check; the passport trap; `forget` |
| 4 | A reactive layer before any thinking | `REFLEX_WORDS` |
| 4 | ReAct, tool use and reflection | the agents' loops; step 6 |
| 5 | A coordinator with specialists that also talk to each other | `coordinator.py`, `ask_flight_agent` |

## Demos: `python demo_couples.py`

| Couple | What they teach | Their trap |
|---|---|---|
| Prasad and Ananya | Hate layovers; vegetarian; window seats | "No sugar in my chai" must not become "diabetic" |
| Arjun and Neha | Cheapest always wins; evening flights | Small talk: nothing to save |
| Kabir and Zoya | Never before 10 am; live music; cabin bags | A vague "maybe" |
| Rohan and Meera | Rohan uses a wheelchair; direct only | Someone else's opinion |
| Sana and Dev | Early birds; peanut allergy | A passport number |

Suggested order: **1** (teach, watch the traps), **2** (what they taught), **3** (the same request for everyone: the centrepiece; same code, five different searches), then **5** (learn a preference, then use it). Demo 3 makes about 30 Gemini calls (8 to 10 minutes).

These run on a real model. If a trap gets saved, or an agent doesn't use a filter, say so and discuss it: should the fix go in the prompt, or in code?

## Troubleshooting

| Problem | Fix |
|---|---|
| "Missing in your .env file" | Copy `.env.example` to `.env` and fill in all three keys |
| LiteAPI error about an invalid key | Use the **private** Sandbox Key, not the public or Stripe one |
| No flights found | Sandbox data is thin on some routes: try another route or date |
| No hotels in "Goa" | Hotels are listed by town: the destination agent is told to try Panaji |
| Pinecone error about dimensions | An old index with the same name exists: delete it in the Pinecone console |
| Gemini 429 / RESOURCE_EXHAUSTED | The free-tier limit: wait a minute |
| A reply takes several minutes | Expected on the free tier: about 4 Gemini calls a minute |
