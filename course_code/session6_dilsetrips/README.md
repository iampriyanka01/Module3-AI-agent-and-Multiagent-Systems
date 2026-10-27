# DilSeTrips: a travel concierge built from AI agents

A couple types:

> Hi! Please book us a flight from Mumbai to Chandigarh on 20 December. What will the weather be like there? Oh, and we always want window seats.

DilSeTrips reads the message, sends a **flight agent** and a **destination agent** to work (they even talk to each other), writes one reply, checks it for mistakes, and remembers "window seats" for next time.

Everything is real: flights and hotels come from LiteAPI's free sandbox, weather from Open-Meteo, places from Wikivoyage, and memories are stored in Pinecone. (Sandbox prices are for testing, so DilSeTrips calls them indicative.)

## The files

| File | What it does |
|---|---|
| `config.py` | Reads your keys and sets up the Gemini model |
| `prompts.yaml` | Every instruction the AI gets, in plain English |
| `tools.py` | Plain Python that fetches weather, places, airports, flights and hotels (no AI) |
| `memory.py` | Saves and recalls memories in Pinecone, with the AI judging what matters |
| `agents.py` | The flight agent and the destination agent |
| `coordinator.py` | One message, from start to reply, in seven steps |
| `main.py` | Chat with DilSeTrips |
| `demo_couples.py` | Five couples, five different results |

## Setup

You need three free keys:

| Key | Where to get it |
|---|---|
| `GOOGLE_API_KEY` | [aistudio.google.com](https://aistudio.google.com/apikey) |
| `PINECONE_API_KEY` | [app.pinecone.io](https://app.pinecone.io) (Starter plan) |
| `LITEAPI_API_KEY` | [dashboard.liteapi.travel](https://dashboard.liteapi.travel/register): copy the **private Sandbox Key** |

Then, in a terminal in this folder (Python 3.10 to 3.13):

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Mac or Linux
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and paste in your three keys.

## Run it

```bash
python tools.py          # check the tools work: weather, a guide, flights and hotels
python main.py           # chat (type: memories, forget, or quit)
python demo_couples.py   # the five-couple demos
```

A full trip request makes about 15 Gemini calls. The free tier allows a few a minute, so a reply takes 3 to 5 minutes. That's the rate limit, not a problem.

## How one message works

1. **Reflex:** obvious emergency words get an instant safety reply, with no AI.
2. **Understand:** the AI turns the message into structured facts: where, when, what they want.
3. **Check:** if it's urgent, reply with safety advice; if a detail is missing, ask one question.
4. **Delegate:** the flight agent searches flights; the destination agent checks weather, places and hotels, and asks the flight agent when the couple lands.
5. **Reply:** the AI writes one warm reply from the agents' reports.
6. **Reflect:** the AI checks the reply against the reports and fixes mistakes.
7. **Learn:** anything worth remembering is saved to Pinecone.

## Ideas to try

- Change a rule in `prompts.yaml` (for example, how the flight agent picks flights) and see how the replies change.
- Add a word to `REFLEX_WORDS` in `coordinator.py`.
- Teach a sixth couple in `demo_couples.py`.
