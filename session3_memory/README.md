# Memory Lab: interactive companion to "Agents, Session 3: Memory in AI Agents"

## Run it
```bash
cd memory_lab
pip install -r requirements.txt        # (demo mode only needs: streamlit pandas; Pinecone needs: pinecone)
streamlit run app.py
```
Open the URL Streamlit prints (usually http://localhost:8501). Always launch with `streamlit run app.py`,
not `python app.py`, otherwise you will see nothing.

## Files
| File | What it is |
|---|---|
| `app.py` | The Streamlit interface: 9 pages (explanation + hands-on lab + code peek) |
| `memory_engine.py` | All logic: toy embeddings, memory store, scoring, simulated "mock LLM", Gemini wrapper |
| `requirements.txt` | Dependencies |

## Two modes
* **Demo (default)**: no key needed. Embeddings are a small offline stand-in and answers are rule-based
  simulations that react to what is in the prompt (the real lesson: the model only knows what you put back in).
* **Live (Gemini)**: choose it in the sidebar and paste a key, or set it outside the code:
  * environment variable `GOOGLE_API_KEY`, or
  * `.streamlit/secrets.toml` (copy `secrets.toml.example`). **Add `.streamlit/secrets.toml` to `.gitignore`.**
  If a live call fails (rate limit, bad key) the app falls back to the demo answer and tells you why.

## Pages
0. Start here (the five tickets, glossary)
1. Why memory matters (same question, different memory; traditional vs agentic; agent loop)
2. Vector DB vs knowledge graph (similarity search lab, triples + path-following lab)
3. Short-term memory, Ticket 1 (token growth, window overflow, five strategies, homework 2)
4. Long-term lifecycle, Tickets 2 and 5 (generate, store, retrieve with tunable weights, integrate, update, soft vs hard delete)
5. Forgetting at scale (periodical, history-based, combined)
6. Semantic, episodic, procedural, Ticket 4 (episode lab, rule-rewriting lab)
7. When memory goes wrong, Ticket 3 (eager vs careful extractor, memory audit, redaction, user isolation)
8. Review (tickets closed, quiz, homework map, cheat sheet)


## Pinecone (persistent long-term memory)
1. Switch to **Live (Gemini)** in the sidebar (Pinecone stores real embeddings).
2. Page 4 > **Store** tab > paste your Pinecone API key, keep or change the index name, click **Connect**.
   The index is created if missing (serverless, cosine, dimension detected from your Gemini embedding model).
3. First time: **Push all local memories**. Later sessions: **Load memories from Pinecone**.
4. After that, add / update (supersede) / soft delete / hard delete / retrieve all go to Pinecone, one namespace per user.
Keys can also come from `PINECONE_API_KEY` and `GOOGLE_API_KEY` (environment variables or `.streamlit/secrets.toml`).

## If Live mode seems stuck
The sidebar shows a live status line while Gemini is working, and the free-tier limiter (sidebar > Model settings) spaces calls out.
Raise "Max Gemini calls per minute" if your quota allows, or press **Test Gemini connection** to check your key.
