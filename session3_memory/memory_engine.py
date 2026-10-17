"""
memory_engine.py
================
All the logic behind the Memory Lab app. No Streamlit code lives here, so every
function can be read, tested and reused on its own.

Two modes:
  * DEMO mode (no API key): a toy embedding + rule-based "mock LLM" so every lab
    still runs offline. Demo answers are *simulations* that react to what is in the
    prompt, which is exactly the lesson: the model only knows what you put back in.
  * LIVE mode (Gemini key): the same labs call Gemini through LangChain, as in the notebook.
"""
from __future__ import annotations

import hashlib
import math
import re
import time
from datetime import date, timedelta

# ----------------------------------------------------------------------------------------
# Sample data (taken from the notebook)
# ----------------------------------------------------------------------------------------
CONVERSATION = [
    {"role": "user", "content": "Hi! My wife Neha and I want a 3-day trip to Amritsar next month. She's strictly vegetarian."},
    {"role": "assistant", "content": "Lovely choice! Amritsar is wonderful for vegetarian food. How will you travel?"},
    {"role": "user", "content": "We'll take the train from Delhi. Which one is best?"},
    {"role": "assistant", "content": "The Shatabdi Express is the most comfortable, about 6 hours from New Delhi."},
    {"role": "user", "content": "Suggest two hotels near the Golden Temple."},
    {"role": "assistant", "content": "1) Heritage Haveli: Rs 3,200 a night, 5 minutes' walk. 2) City Comfort Inn: Rs 2,100 a night, 15 minutes' walk."},
    {"role": "user", "content": "What should we see on day 1?"},
    {"role": "assistant", "content": "The Golden Temple at dawn, Jallianwala Bagh, then the Partition Museum."},
    {"role": "user", "content": "And day 2?"},
    {"role": "assistant", "content": "The Wagah border ceremony in the afternoon; the old city bazaars in the morning."},
    {"role": "user", "content": "Is it cold there in December?"},
    {"role": "assistant", "content": "Yes, nights drop to around 5 C and mornings can be foggy, so pack warm layers."},
]

PROBE = ("Suggest one dinner place for us tonight in Amritsar, and remind me exactly how much "
         "the first hotel you suggested costs per night. Two sentences.")

SAMPLE_FACTS = (
    "Prasad prefers quiet hotels. Prasad's partner eats only vegetarian food.\n"
    "Snow Valley Cottage is located in Old Manali. Snow Valley Cottage is quiet. Snow Valley Cottage serves vegetarian food.\n"
    "Mall Road Grand is located in Manali. Mall Road Grand is noisy.\n"
    "River Bend Homestay is located in Old Manali. River Bend Homestay serves vegetarian food."
)

SAMPLE_DEBRIEF = ("We're back from Manali (20 to 23 December)! Old Manali was so quiet, Snow Valley Cottage was perfect and the veg "
                  "food was great. But the flight to Chandigarh plus 8 hours in a taxi wrecked me, and two days of snow meant we "
                  "couldn't go to Solang at all.")

DEFAULT_PROCEDURE = [
    "Greet the couple by name.",
    "Use the weather tool for weather questions.",
    "Suggest at most three options.",
]
SAMPLE_FEEDBACK = ("You told us the weather for the week I was asking in, not for our December trip dates. "
                   "And you keep giving me five hotel options when I asked for a short answer.")

DEFAULT_USAGE = {
    "Prasad's partner is vegetarian.": {"uses_previous_month": 14, "uses_this_month": 12, "usefulness": [0.9, 1.0, 0.8]},
    "Prasad gets motion sickness on long bus rides.": {"uses_previous_month": 6, "uses_this_month": 5, "usefulness": [0.7, 0.9, 0.8]},
    "Prasad prefers window seats on flights.": {"uses_previous_month": 8, "uses_this_month": 1, "usefulness": [0.8, 0.7, 0.9]},
    "Prasad once asked about beach shacks in Goa.": {"uses_previous_month": 9, "uses_this_month": 1, "usefulness": [0.1, 0.2, 0.0]},
}

# JSON schemas, exactly the "shapes" the notebook asks Gemini to fill in
TRIPLES_SCHEMA = {
    "title": "triples", "description": "Facts as (subject, relation, object) triples", "type": "object",
    "properties": {"triples": {"type": "array", "items": {"type": "object", "properties": {
        "subject": {"type": "string"},
        "relation": {"type": "string", "description": "A short verb phrase such as 'prefers' or 'located in'"},
        "object": {"type": "string"}}, "required": ["subject", "relation", "object"]}}},
    "required": ["triples"],
}
ENTITY_SCHEMA = {
    "title": "entities", "description": "Facts about each person, place or thing mentioned", "type": "object",
    "properties": {"entities": {"type": "array", "items": {"type": "object", "properties": {
        "name": {"type": "string"}, "facts": {"type": "array", "items": {"type": "string"}}},
        "required": ["name", "facts"]}}},
    "required": ["entities"],
}
MEMORY_SCHEMA = {
    "title": "memories", "description": "Facts worth remembering from a user's message", "type": "object",
    "properties": {"memories": {"type": "array", "items": {"type": "object", "properties": {
        "text": {"type": "string", "description": "One self-contained fact in the third person, e.g. 'Prasad lives in Pune.'"},
        "kind": {"type": "string", "enum": ["semantic", "episodic", "procedural"]},
        "importance": {"type": "integer", "description": "1 (trivia) to 5 (critical, e.g. dietary needs, allergies)"},
        "confidence": {"type": "number", "description": "0 to 1: how sure you are that this is true"},
        "source_quote": {"type": "string", "description": "The exact words in the message that support this fact"}},
        "required": ["text", "kind", "importance", "confidence", "source_quote"]}}},
    "required": ["memories"],
}
EPISODE_SCHEMA = {
    "title": "episode", "description": "A record of one experience", "type": "object",
    "properties": {"when": {"type": "string"}, "what_happened": {"type": "string"},
                   "what_worked": {"type": "array", "items": {"type": "string"}},
                   "what_did_not_work": {"type": "array", "items": {"type": "string"}}},
    "required": ["when", "what_happened", "what_worked", "what_did_not_work"],
}
RULES_SCHEMA = {"title": "procedure", "description": "The agent's updated working rules", "type": "object",
                "properties": {"rules": {"type": "array", "items": {"type": "string"}}}, "required": ["rules"]}
REPLACE_SCHEMA = {"title": "replacement", "description": "Which memory a new fact replaces", "type": "object",
                  "properties": {"replaces_memory_id": {"type": "string", "description": "The id, or 'none'"},
                                 "reason": {"type": "string"}},
                  "required": ["replaces_memory_id", "reason"]}
RATINGS_SCHEMA = {"title": "memory_ratings", "description": "How useful each memory was for the answer", "type": "object",
                  "properties": {"ratings": {"type": "array", "items": {"type": "object", "properties": {
                      "memory": {"type": "string"},
                      "usefulness": {"type": "number", "description": "0 (no help) to 1 (essential)"}},
                      "required": ["memory", "usefulness"]}}},
                  "required": ["ratings"]}

CAREFUL_RULES = ("You decide what a travel concierge should remember about a user. Extract only facts the user "
                 "EXPLICITLY stated. Never guess or infer health conditions, religion, income or anything not said. "
                 "Never store card numbers, passwords, OTPs or ID numbers. Skip small talk. "
                 "Return an empty list if nothing is worth remembering.")
EAGER_RULES = ("You build a rich profile of the user for a travel concierge. Infer as much as you can about their health, "
               "lifestyle and preferences from what they say, so future suggestions are personalised.")


# ----------------------------------------------------------------------------------------
# Tokens (a rough estimate: about 4 characters per token)
# ----------------------------------------------------------------------------------------
def approx_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def messages_tokens(messages: list[dict]) -> int:
    return sum(approx_tokens(m["content"]) + 3 for m in messages)


def transcript(messages: list[dict]) -> str:
    return "".join(("User: " if m["role"] == "user" else "Assistant: ") + m["content"] + "\n" for m in messages)


def trim_last(messages: list[dict], n: int) -> list[dict]:
    """Keep the last n messages (a 'window'), making sure the window starts on a user message."""
    window = messages[-n:] if n > 0 else []
    while window and window[0]["role"] != "user":
        window = window[1:]
    return window


# ----------------------------------------------------------------------------------------
# Toy embedding (DEMO mode). Real embeddings learn "meaning" from huge data; this one fakes it
# with hashed words plus a small hand-made concept list, so similar texts get similar vectors.
# ----------------------------------------------------------------------------------------
_STOP = set("a an the in on of to is are for us we our my i it and or at by with that this be has have had they them their "
            "he she her his you your me can would will as from about once".split())
_CONCEPTS = {
    "travel": "trip travel getaway weekend holiday vacation tour plan visit go drive destination relaxed relax short next".split(),
    "food": "food eat vegetarian veg dinner lunch meal restaurant chai sugar diet vegan cuisine dhaba drink".split(),
    "transport": "bus train flight seat window aisle taxi car drive sick motion ride road fly airline".split(),
    "home": "live home city move moved pune bengaluru based reside town".split(),
    "health": "allergic allergy diabetes diabetic sick motion medicine health peanut".split(),
    "hotel": "hotel stay homestay cottage room balcony quiet noisy cosy resort".split(),
    "beach": "beach goa shack sea".split(),
}
# Planning a trip also needs to know where you live, what you eat, how you travel, where you sleep.
_LINKS = {"travel": {"home": 0.6, "food": 0.35, "transport": 0.6, "hotel": 0.4}}
_DIM = 256


def _bucket(token: str) -> int:
    return int(hashlib.md5(token.encode()).hexdigest(), 16) % _DIM


def _stem(word: str) -> str:
    return word[:-1] if word.endswith("s") and not word.endswith("ss") and len(word) > 3 else word


def toy_embed(text: str) -> list[float]:
    vec = [0.0] * _DIM
    hits = set()
    for raw in re.findall(r"[a-z]+", text.lower()):
        if raw in _STOP:
            continue
        tok = _stem(raw)
        vec[_bucket("w:" + tok)] += 1.0
        for concept, words in _CONCEPTS.items():
            if tok in words or raw in words:
                hits.add(concept)
    for concept in hits:
        vec[_bucket("c:" + concept)] += 1.5
        for other, weight in _LINKS.get(concept, {}).items():
            vec[_bucket("c:" + other)] += 1.5 * weight
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


# ----------------------------------------------------------------------------------------
# Backend: Gemini (live) or simulation (demo)
# ----------------------------------------------------------------------------------------
class Backend:
    """Wraps the LLM + embeddings. Every call takes a `demo` value/callable used when there is no key
    (or when a live call fails, so the lessons never get stuck).
    `notify` (optional) is called with a short status text before/after each live call."""

    def __init__(self, api_key: str = "", chat_model: str = "gemini-3.5-flash-lite",
                 embed_model: str = "models/gemini-embedding-001", calls_per_minute: int = 10):
        self.live = False
        self.error = ""
        self.last_source = "demo"
        self.calls = 0
        self.last_seconds = 0.0
        self.llm = None
        self.embedder = None
        self.notify = None
        self.rpm = calls_per_minute
        self._cache: dict = {}
        if api_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
                kwargs = {}
                if calls_per_minute and calls_per_minute > 0:
                    from langchain_core.rate_limiters import InMemoryRateLimiter
                    kwargs["rate_limiter"] = InMemoryRateLimiter(requests_per_second=calls_per_minute / 60.0,
                                                                 check_every_n_seconds=0.2, max_bucket_size=1)
                self.llm = ChatGoogleGenerativeAI(model=chat_model, google_api_key=api_key, **kwargs)
                self.embedder = GoogleGenerativeAIEmbeddings(model=embed_model, google_api_key=api_key)
                self.live = True
            except Exception as exc:  # missing package, bad key format, ...
                self.error = f"Could not start Gemini: {exc}"

    # --- small helpers ---------------------------------------------------------------------
    @property
    def embedder_name(self) -> str:
        return "gemini" if self.live else "toy"

    @staticmethod
    def _resolve(demo):
        return demo() if callable(demo) else demo

    def _say(self, text: str):
        if self.notify:
            try:
                self.notify(text)
            except Exception:
                pass

    def _begin(self, what: str):
        wait = ""
        if self.rpm and self.rpm > 0:
            wait = f" (rate limit: {self.rpm} calls/min, so it may wait up to {60 // self.rpm} s first)"
        self._say(f"⏳ Calling Gemini: {what}{wait}")

    def _end(self, t0: float):
        self.last_seconds = time.perf_counter() - t0
        self._say(f"✅ Gemini answered in {self.last_seconds:.1f} s  |  calls this session: {self.calls}")

    def _fail(self, exc: Exception, demo):
        self.error = f"Gemini call failed, showing the demo simulation instead: {exc}"
        self.last_source = "demo (fallback)"
        self._say("⚠️ Gemini call failed, using the demo simulation")
        return self._resolve(demo)

    # --- chat ----------------------------------------------------------------------------
    def ask(self, messages: list[dict], demo) -> str:
        """messages = [{'role': 'system'|'user'|'assistant', 'content': ...}, ...]"""
        if self.live:
            self._begin("writing an answer")
            t0 = time.perf_counter()
            try:
                from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
                kinds = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}
                out = self.llm.invoke([kinds[m["role"]](content=m["content"]) for m in messages])
                self.calls += 1
                self.last_source = "gemini"
                self._end(t0)
                text = getattr(out, "text", None)
                return text if isinstance(text, str) else str(out.content)
            except Exception as exc:
                return self._fail(exc, demo)
        self.last_source = "demo"
        return self._resolve(demo)

    def structured(self, schema: dict, prompt: str, demo):
        if self.live:
            self._begin("filling a structured form")
            t0 = time.perf_counter()
            try:
                out = self.llm.with_structured_output(schema).invoke(prompt)
                self.calls += 1
                self.last_source = "gemini"
                self._end(t0)
                return out
            except Exception as exc:
                return self._fail(exc, demo)
        self.last_source = "demo"
        return self._resolve(demo)

    # --- embeddings (cached: the same text is never embedded twice) -------------------------
    def embed_many(self, texts: list[str]) -> list[list[float]]:
        name = self.embedder_name
        out: list = [None] * len(texts)
        todo = []
        for i, t in enumerate(texts):
            hit = self._cache.get((name, t))
            if hit is None:
                todo.append(i)
            else:
                out[i] = hit
        if todo:
            batch = [texts[i] for i in todo]
            vecs = None
            if self.live:
                self._begin(f"embedding {len(batch)} text(s)")
                t0 = time.perf_counter()
                try:
                    vecs = self.embedder.embed_documents(batch)
                    self.calls += 1
                    self._end(t0)
                except Exception as exc:
                    self.error = f"Embedding call failed, switched to the toy embedding: {exc}"
                    self._say("⚠️ Embedding call failed, switched to the toy embedding")
                    self.live = False
            if vecs is None:
                vecs = [toy_embed(t) for t in batch]
            name = self.embedder_name
            for i, v in zip(todo, vecs):
                out[i] = v
                self._cache[(name, texts[i])] = v
        return out

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def similarity(self, a: list[float], b: list[float]) -> float:
        c = max(0.0, cosine(a, b))
        return c if self.embedder_name == "gemini" else math.sqrt(c)  # sqrt: toy cosines are small; keeps 0..1 feel


# ----------------------------------------------------------------------------------------
# Pinecone: a real, persistent vector database (one NAMESPACE per user)
# ----------------------------------------------------------------------------------------
class PineconeSync:
    """Thin wrapper around the Pinecone SDK (pip install pinecone).
    Each user's memories live in their own namespace, so a query for one user can never see another's."""

    def __init__(self, api_key: str, index_name: str = "dilsetrips-memory", cloud: str = "aws",
                 region: str = "us-east-1", dimension: int | None = None, metric: str = "cosine"):
        from pinecone import Pinecone, ServerlessSpec
        if not api_key:
            raise ValueError("A Pinecone API key is required.")
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.calls = 0
        try:
            existing = list(self.pc.list_indexes().names())
        except Exception:
            existing = [i["name"] for i in self.pc.list_indexes()]
        self.created = False
        if index_name not in existing:
            if not dimension:
                raise ValueError("Cannot create the index: embedding dimension unknown.")
            self.pc.create_index(name=index_name, dimension=dimension, metric=metric,
                                 spec=ServerlessSpec(cloud=cloud, region=region))
            self.created = True
            for _ in range(60):
                if self._ready():
                    break
                time.sleep(2)
        desc = self.pc.describe_index(index_name)
        self.dimension = getattr(desc, "dimension", None) or desc["dimension"]
        if dimension and self.dimension != dimension:
            raise ValueError(f"Index '{index_name}' has dimension {self.dimension} but Gemini embeddings have {dimension}. "
                             "Use a new index name (or delete the old index).")
        self.index = self.pc.Index(index_name)

    def _ready(self) -> bool:
        status = self.pc.describe_index(self.index_name).status
        try:
            return bool(status["ready"])
        except Exception:
            return bool(getattr(status, "ready", False))

    @staticmethod
    def _meta(item: dict) -> dict:
        m = dict(item["meta"])
        m["text"] = item["text"]
        return {k: ("" if v is None else v) for k, v in m.items()}

    def upsert(self, items: list[dict]) -> int:
        by_ns: dict[str, list] = {}
        for it in items:
            by_ns.setdefault(it["meta"]["user_id"], []).append(
                {"id": it["meta"]["memory_id"], "values": [float(x) for x in it["vec"]], "metadata": self._meta(it)})
        for ns, vecs in by_ns.items():
            for i in range(0, len(vecs), 100):
                self.index.upsert(vectors=vecs[i:i + 100], namespace=ns)
                self.calls += 1
        return len(items)

    def set_status(self, item: dict) -> None:
        self.index.update(id=item["meta"]["memory_id"], set_metadata={"status": item["meta"]["status"]},
                          namespace=item["meta"]["user_id"])
        self.calls += 1

    def delete(self, ids: list[str], namespace: str) -> None:
        for i in range(0, len(ids), 100):
            self.index.delete(ids=ids[i:i + 100], namespace=namespace)
            self.calls += 1

    def query(self, user_id: str, qvec: list[float], k: int, kinds=None, active_only=True) -> list[tuple]:
        flt: dict = {}
        if active_only:
            flt["status"] = {"$eq": "active"}
        if kinds:
            flt["kind"] = {"$in": list(kinds)}
        kwargs = dict(vector=[float(x) for x in qvec], top_k=k, namespace=user_id, include_metadata=True)
        if flt:
            kwargs["filter"] = flt
        res = self.index.query(**kwargs)
        self.calls += 1
        return [(m.id, float(m.score), dict(m.metadata or {})) for m in res.matches]

    def namespace_counts(self) -> dict[str, int]:
        stats = self.index.describe_index_stats()
        self.calls += 1
        out = {}
        for ns, info in dict(stats.namespaces).items():
            count = getattr(info, "vector_count", None)
            if count is None and isinstance(info, dict):
                count = info.get("vector_count", 0)
            out[ns or "(default)"] = int(count or 0)
        return out

    def load_all(self) -> list[tuple]:
        rows = []
        for ns in self.namespace_counts():
            real_ns = "" if ns == "(default)" else ns
            ids: list[str] = []
            for page in self.index.list(namespace=real_ns):
                ids.extend(list(page))
            for i in range(0, len(ids), 100):
                res = self.index.fetch(ids=ids[i:i + 100], namespace=real_ns)
                self.calls += 1
                for vid, vec in dict(res.vectors).items():
                    rows.append((vid, list(vec.values), dict(vec.metadata or {})))
        return rows


# ----------------------------------------------------------------------------------------
# The long-term memory store (a tiny vector store with metadata; optionally mirrored to Pinecone)
# ----------------------------------------------------------------------------------------
class MemoryStore:
    def __init__(self, backend: Backend, remote: PineconeSync | None = None):
        self.b = backend
        self.items: dict[str, dict] = {}
        self.counter = 1
        self.version = 0                 # bumps on every change; used to invalidate cached searches
        self.remote = remote
        self.remote_error = ""
        self._cand: dict = {}

    # --- Pinecone plumbing ---------------------------------------------------------------
    def remote_ready(self) -> bool:
        """Pinecone is only used with real (Gemini) embeddings: the toy vectors have a different size."""
        return self.remote is not None and self.b.embedder_name == "gemini"

    def _touch(self):
        self.version += 1
        self._cand.clear()

    def _remote_do(self, label: str, fn):
        if not self.remote_ready():
            return None
        try:
            return fn()
        except Exception as exc:
            self.remote_error = f"Pinecone {label} failed: {exc}"
            return None

    def _push(self, items: list[dict]):
        def go():
            self._ensure_vectors(items)
            if self.b.embedder_name == "gemini":
                self.remote.upsert(items)
        self._remote_do("upsert", go)

    def push_all(self) -> int:
        items = self.all()
        self._ensure_vectors(items)
        return self.remote.upsert(items)

    def pull_from_remote(self) -> int:
        rows = self.remote.load_all()
        self.items = {}
        top = 0
        for mid, vec, meta in rows:
            m = {k: meta.get(k) for k in ("memory_id", "user_id", "kind", "importance", "confidence", "source_quote",
                                          "stored_on", "status", "version", "supersedes")}
            m["memory_id"] = mid
            m["importance"] = int(m.get("importance") or 3)
            m["confidence"] = float(m.get("confidence") or 1.0)
            m["version"] = int(m.get("version") or 1)
            m["status"] = m.get("status") or "active"
            m["supersedes"] = m.get("supersedes") or ""
            self.items[mid] = {"text": meta.get("text", ""), "vec": vec, "embedder": "gemini", "meta": m}
            found = re.search(r"(\d+)$", mid)
            top = max(top, int(found.group(1)) if found else 0)
        self.counter = top + 1
        self._touch()
        return len(rows)

    # --- writes ----------------------------------------------------------------------------
    def add(self, user_id, text, kind="semantic", importance=3, confidence=1.0, source_quote="",
            days_ago=0, version=1, supersedes="") -> str:
        mid = f"mem-{self.counter}"
        self.counter += 1
        self.items[mid] = {
            "text": text, "vec": None, "embedder": None,
            "meta": {"memory_id": mid, "user_id": user_id, "kind": kind, "importance": int(importance),
                     "confidence": float(confidence), "source_quote": source_quote,
                     "stored_on": (date.today() - timedelta(days=int(days_ago))).isoformat(),
                     "status": "active", "version": version, "supersedes": supersedes},
        }
        self._touch()
        self._push([self.items[mid]])
        return mid

    def supersede(self, old_id: str, new_text: str, source_quote: str) -> str:
        old = self.items[old_id]
        old["meta"]["status"] = "superseded"
        self._touch()
        self._remote_do("status update", lambda: self.remote.set_status(old))
        m = old["meta"]
        return self.add(m["user_id"], new_text, m["kind"], m["importance"], 1.0, source_quote,
                        version=m["version"] + 1, supersedes=old_id)

    def soft_delete_user(self, user_id: str) -> int:
        mine = self.of_user(user_id)
        for it in mine:
            it["meta"]["status"] = "deleted"
        self._touch()
        for it in mine:
            self._remote_do("status update", lambda it=it: self.remote.set_status(it))
        return len(mine)

    def hard_delete_user(self, user_id: str) -> list[str]:
        ids = [it["meta"]["memory_id"] for it in self.of_user(user_id)]
        self._remote_do("delete", lambda: self.remote.delete(ids, user_id) if ids else None)
        for mid in ids:
            del self.items[mid]
        self._touch()
        return ids

    # --- reads -----------------------------------------------------------------------------
    # embeddings are computed lazily (and re-computed if you switch between Demo and Live)
    def _ensure_vectors(self, items: list[dict]):
        todo = [it for it in items if it["vec"] is None or it["embedder"] != self.b.embedder_name]
        if todo:
            vecs = self.b.embed_many([it["text"] for it in todo])
            for it, v in zip(todo, vecs):
                it["vec"], it["embedder"] = v, self.b.embedder_name

    def all(self) -> list[dict]:
        return list(self.items.values())

    def of_user(self, user_id: str, active_only=False) -> list[dict]:
        return [it for it in self.items.values()
                if it["meta"]["user_id"] == user_id and (not active_only or it["meta"]["status"] == "active")]

    def users(self) -> list[str]:
        seen = []
        for it in self.items.values():
            if it["meta"]["user_id"] not in seen:
                seen.append(it["meta"]["user_id"])
        return seen

    def search(self, query, user_id=None, kinds=None, k=8, active_only=True) -> list[tuple[dict, float]]:
        if self.remote_ready() and user_id is not None:          # real Pinecone query, inside this user's namespace
            def go():
                matches = self.remote.query(user_id, self.b.embed(query), k, kinds, active_only)
                out = []
                for mid, score, meta in matches:
                    it = self.items.get(mid)
                    if it is None:
                        it = {"text": meta.get("text", ""), "vec": None, "embedder": None,
                              "meta": {"memory_id": mid, "user_id": user_id, "kind": meta.get("kind", "semantic"),
                                       "importance": int(meta.get("importance") or 3), "confidence": float(meta.get("confidence") or 1.0),
                                       "source_quote": meta.get("source_quote", ""), "stored_on": meta.get("stored_on", date.today().isoformat()),
                                       "status": meta.get("status", "active"), "version": int(meta.get("version") or 1),
                                       "supersedes": meta.get("supersedes", "")}}
                    out.append((it, score))
                return out
            result = self._remote_do("query", go)
            if result is not None:
                return result[:k]
        pool = [it for it in self.items.values()
                if (user_id is None or it["meta"]["user_id"] == user_id)
                and (not active_only or it["meta"]["status"] == "active")
                and (kinds is None or it["meta"]["kind"] in kinds)]
        if not pool:
            return []
        self._ensure_vectors(pool)
        qv = self.b.embed(query)
        scored = [(it, self.b.similarity(qv, it["vec"])) for it in pool]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]

    def _candidates(self, user_id, query, pool):
        """The similarity step, cached: moving a weight slider re-scores but never re-calls Gemini or Pinecone."""
        key = (user_id, query, pool, self.version, self.b.embedder_name, self.remote_ready())
        if key not in self._cand:
            self._cand[key] = self.search(query, user_id=user_id, k=pool)
        return self._cand[key]

    def retrieve(self, user_id, query, k=3, weights=(0.6, 0.3, 0.1), pool=8) -> list[dict]:
        """pre-filter -> similarity -> re-rank by (similarity, recency, importance) -> top k"""
        w_sim, w_rec, w_imp = weights
        rows = []
        for it, sim in self._candidates(user_id, query, pool):
            days_old = (date.today() - date.fromisoformat(it["meta"]["stored_on"])).days
            recency = 1 / (1 + days_old / 30)
            importance = it["meta"]["importance"] / 5
            score = w_sim * sim + w_rec * recency + w_imp * importance
            rows.append({"item": it, "similarity": sim, "recency": recency, "importance": importance,
                         "days_old": days_old, "score": score,
                         "working": f"{w_sim:.1f}x{sim:.2f} + {w_rec:.1f}x{recency:.2f} + {w_imp:.1f}x{importance:.2f} = {score:.3f}"})
        rows.sort(key=lambda r: r["score"], reverse=True)
        return rows[:k]

    def rows(self, user_id=None) -> list[dict]:
        out = []
        for it in self.items.values():
            m = it["meta"]
            if user_id and m["user_id"] != user_id:
                continue
            out.append({"id": m["memory_id"], "user": m["user_id"], "kind": m["kind"], "importance": m["importance"],
                        "confidence": m["confidence"], "stored_on": m["stored_on"], "status": m["status"],
                        "version": m["version"], "supersedes": m["supersedes"], "memory": it["text"],
                        "source_quote": m["source_quote"],
                        "has_embedding": it["vec"] is not None})
        return out


def seed_store(store: MemoryStore) -> None:
    """The memories the notebook writes 'from earlier sessions' plus today's extracted ones."""
    store.add("prasad", "Prasad's partner is vegetarian.", "semantic", 5, 1.0, "she's vegetarian", days_ago=40)
    store.add("prasad", "Prasad gets motion sickness on long bus rides.", "semantic", 4, 1.0, "I get sick on long bus rides", days_ago=40)
    store.add("prasad", "Prasad prefers window seats on flights.", "semantic", 2, 0.9, "window seat please", days_ago=120)
    store.add("prasad", "Prasad once asked about beach shacks in Goa.", "episodic", 1, 0.8, "any good beach shacks in Goa?", days_ago=200)
    store.add("arjun", "Arjun's wife Neha is strictly vegetarian.", "semantic", 5, 1.0, "She's strictly vegetarian", days_ago=10)
    store.add("arjun", "Arjun prefers trains to flights.", "semantic", 3, 1.0, "We'll take the train", days_ago=10)
    store.add("prasad", "Prasad lives in Pune.", "semantic", 3, 1.0, "We live in Pune, by the way")
    store.add("prasad", "Prasad would like a relaxed weekend getaway they can drive to.", "semantic", 2, 0.9,
              "We'd love a relaxed weekend getaway we can drive to")


def memory_context(ranked: list[dict]) -> str:
    """Turn retrieved memories into prompt text, with provenance and a flag for low confidence."""
    text = ""
    for r in ranked:
        meta = r["item"]["meta"]
        line = f"- {r['item']['text']} (from the user's words on {meta['stored_on']}: \"{meta['source_quote']}\")"
        if meta["confidence"] < 0.6:
            line += " [unconfirmed: ask before relying on this]"
        text += line + "\n"
    return text


# ----------------------------------------------------------------------------------------
# DEMO-mode simulations. They react to what is in the prompt, never to hidden knowledge.
# ----------------------------------------------------------------------------------------
def demo_persona_answer(system_text: str) -> str:
    s = system_text.lower()
    if any(w in s for w in ("teacher", "analog", "simple", "child", "beginner")):
        return ("Think of an embedding as a map reference for meaning. Every word or sentence gets a spot on the map, "
                "and ideas that mean similar things sit close together, like 'cat' and 'kitten' being neighbours. "
                "The computer finds related ideas just by looking at who lives nearby.")
    if any(w in s for w in ("researcher", "technical", "precise", "engineer", "scientist")):
        return ("An embedding is a dense, fixed-dimensional vector produced by a learned encoder so that semantic similarity "
                "maps to geometric proximity (typically cosine similarity). Trained with contrastive or next-token objectives, "
                "embeddings support nearest-neighbour retrieval, clustering and downstream classification.")
    return ("An embedding turns text into a list of numbers that captures its meaning. Texts with similar meaning get similar "
            "numbers, which lets software search by meaning instead of exact words.")


def _norm_quality(obj: str) -> str:
    o = obj.lower()
    if "quiet" in o:
        return "quiet"
    if "noisy" in o:
        return "noisy"
    if "vegetarian" in o:
        return "vegetarian food"
    return obj.strip()


def demo_triples(text: str, normalise: bool = True) -> list[dict]:
    out = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n", text):
        s = sentence.strip().rstrip(".!?")
        if not s:
            continue
        fix = _norm_quality if normalise else (lambda o: o.strip())
        for pattern, relation, use_fix in (
            (r"(.+?) is located in (.+)$", "located in", False),
            (r"(.+?) (?:prefers|wants|needs) (.+)$", "wants", True),
            (r"(.+?) eats only (.+)$", "wants", True),
            (r"(.+?) serves (.+)$", "serves", True),
            (r"(.+?) is (.+)$", "is", True),
        ):
            m = re.match(pattern, s)
            if m:
                out.append({"subject": m.group(1).strip(), "relation": relation,
                            "object": fix(m.group(2)) if use_fix else m.group(2).strip()})
                break
    return out


def graph_wants(graph: list[dict], person_names: list[str]) -> list[tuple[str, str]]:
    names = [p.lower() for p in person_names]
    return [(t["relation"], t["object"]) for t in graph if t["subject"].lower() in names]


def graph_paths(graph: list[dict], person_names: list[str]) -> list[dict]:
    """person --wants--> quality <-- hotel : every such path is evidence that a hotel fits."""
    names = [p.lower() for p in person_names]
    paths = []
    for relation, wanted in graph_wants(graph, person_names):
        for t in graph:
            if t["object"].lower() == wanted.lower() and t["subject"].lower() not in names:
                paths.append({"want": wanted, "relation": relation, "hotel": t["subject"],
                              "text": f"Prasad (or partner) --{relation}--> {wanted} <-- {t['subject']}"})
    return paths


def best_hotels(paths: list[dict]) -> tuple[list[str], int]:
    """Hotels reached by the most distinct wants."""
    coverage: dict[str, set] = {}
    for p in paths:
        coverage.setdefault(p["hotel"], set()).add(p["want"].lower())
    if not coverage:
        return [], 0
    top = max(len(v) for v in coverage.values())
    return [h for h, v in coverage.items() if len(v) == top], top


def graph_dot(graph: list[dict], highlight: list[str] | None = None) -> str:
    highlight = [h.lower() for h in (highlight or [])]
    lines = ["digraph G {", "rankdir=LR;", 'node [shape=box, style="rounded,filled", fillcolor="#eef2ff", fontname="Helvetica"];',
             'edge [fontname="Helvetica", fontsize=10];']
    for t in graph:
        sub, obj = t["subject"].replace('"', "'"), t["object"].replace('"', "'")
        extra = ', color="#d9480f", penwidth=2' if t["subject"].lower() in highlight else ""
        label = t["relation"]
        lines.append('"%s" -> "%s" [label="%s"%s];' % (sub, obj, label, extra))
    lines.append("}")
    return "\n".join(lines)


# --- short-term memory simulations -------------------------------------------------------
def demo_summary(older_only: bool = False, allergy: bool = False, protect: bool = False) -> str:
    if older_only:
        s = ("Arjun and wife Neha (vegetarian) plan a 3-day Amritsar trip next month by train from Delhi; "
             "the assistant suggested two hotels near the Golden Temple and day-1 sights.")
    else:
        s = ("Arjun and his wife Neha, who is vegetarian, plan a 3-day Amritsar trip next month by train from Delhi. "
             "Two hotels near the Golden Temple were suggested, plus sights for days 1 and 2 and a warning about cold December nights.")
    if allergy:
        s = s.replace("who is vegetarian", "who is vegetarian with some dietary restrictions").replace("(vegetarian)", "(vegetarian, some dietary restrictions)")
    if protect:
        s += " CRITICAL FACTS (kept verbatim): Neha is strictly vegetarian." + (" Neha has a peanut allergy." if allergy else "")
    return s


def demo_entities(allergy: bool = False) -> list[dict]:
    neha = ["is Arjun's wife", "is strictly vegetarian"] + (["has a peanut allergy"] if allergy else [])
    return [
        {"name": "Neha", "facts": neha},
        {"name": "Amritsar", "facts": ["3-day trip next month", "nights around 5 C in December, mornings foggy"]},
        {"name": "Shatabdi Express", "facts": ["most comfortable train from New Delhi", "about 6 hours"]},
        {"name": "Heritage Haveli", "facts": ["Rs 3,200 a night", "5 minutes' walk from the Golden Temple"]},
        {"name": "City Comfort Inn", "facts": ["Rs 2,100 a night", "15 minutes' walk from the Golden Temple"]},
    ]


def entities_text(entities: list[dict]) -> str:
    return "".join(f"- {e['name']}: " + "; ".join(e["facts"]) + "\n" for e in entities)


def demo_probe_answer(memory_messages: list[dict]) -> str:
    text = " ".join(m["content"] for m in memory_messages).lower()
    knows_veg = "vegetarian" in text
    knows_price = "3,200" in text
    dinner = ("For dinner tonight, try a pure-vegetarian dhaba near the Golden Temple, such as Kesar Da Dhaba. "
              if knows_veg else
              "For dinner tonight, try the famous Amritsari butter chicken at a dhaba on Lawrence Road. ")
    price = ("The first hotel, Heritage Haveli, costs Rs 3,200 a night."
             if knows_price else
             "I don't have the first hotel's price in front of me, could you remind me which one you mean?")
    return dinner + price


def build_strategies(conversation, window_n, summary_all, summary_old, entities) -> dict[str, list[dict]]:
    window = trim_last(conversation, window_n)
    return {
        "Buffer": conversation,
        f"Window (last {window_n})": window,
        "Summary": [{"role": "system", "content": "Summary of the conversation so far: " + summary_all}],
        "Summary + buffer": [{"role": "system", "content": "Summary of earlier conversation: " + summary_old}]
        + trim_last(conversation, 4),
        "Entity": [{"role": "system", "content": "Known facts:\n" + entities_text(entities)}],
    }


# --- long-term memory simulations ---------------------------------------------------------
def find_sensitive(message: str) -> list[tuple[str, str]]:
    found = []
    for label, pattern in (("card number", r"\b(?:\d[ -]?){13,16}\b"), ("OTP", r"\bOTP\s*(?:is|:)?\s*\d{4,8}\b"),
                           ("phone number", r"\b\d{5}[ -]?\d{5}\b")):
        for m in re.finditer(pattern, message, flags=re.I):
            found.append((label, m.group(0)))
    return found


def redact(message: str) -> str:
    out = message
    for label, value in find_sensitive(message):
        out = out.replace(value, f"[{label} removed]")
    return out


def _mem(text, kind, importance, confidence, quote):
    return {"text": text, "kind": kind, "importance": importance, "confidence": confidence, "source_quote": quote}


def demo_generate_memories(user_name: str, message: str, mode: str = "careful") -> list[dict]:
    """A rule-based stand-in for the LLM extractor. 'eager' reproduces Ticket 3 and the privacy leak."""
    out = []
    n = user_name
    m = re.search(r"\b(?:we|I) (?:just )?(?:live in|moved to) ([A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)?)", message)
    if m:
        out.append(_mem(f"{n} lives in {m.group(1)}.", "semantic", 3, 1.0, m.group(0)))
    m = re.search(r"(?:strictly |only )?vegetarian", message, flags=re.I)
    if m:
        out.append(_mem(f"{n}'s party is vegetarian.", "semantic", 5, 1.0, m.group(0)))
    m = re.search(r"allergic to ([a-z ]+)", message, flags=re.I)
    if m:
        out.append(_mem(f"{n} is allergic to {m.group(1).strip()}.", "semantic", 5, 1.0, m.group(0)))
    m = re.search(r"We'd love ([^.!?]+)", message)
    if m:
        out.append(_mem(f"{n} would like {m.group(1).strip()}.", "semantic", 2, 0.9, m.group(0)))
    m = re.search(r"\bprefer ([^.!?]+)", message, flags=re.I)
    if m:
        out.append(_mem(f"{n} prefers {m.group(1).strip()}.", "semantic", 3, 0.95, m.group(0)))
    m = re.search(r"no sugar in my chai[^.!?]*?(?:,\s*I've given it up)?(?=[.!?]|$)", message, flags=re.I)
    if m:
        out.append(_mem(f"{n} has given up sugar in their chai.", "semantic", 2, 0.95, m.group(0)))
        if mode == "eager":
            out.append(_mem(f"{n} may be diabetic.", "semantic", 4, 0.55, "no sugar in my chai"))
            out.append(_mem(f"{n} is health-conscious and avoids sugar for medical reasons.", "semantic", 3, 0.6,
                            "no sugar in my chai"))
    if mode == "eager":
        for label, value in find_sensitive(message):
            out.append(_mem(f"{n}'s {label} is {value}.", "semantic", 4, 1.0, value))
    return out


_AUDIT_STOP = _STOP | {"may", "also"}


def audit_memory(memory_text: str, source_quote: str, user_name: str = "") -> dict:
    """Does the quote actually support the memory? A crude word-overlap check (live mode could ask Gemini)."""
    def words(s):
        return {_stem(w) for w in re.findall(r"[a-z]+", s.lower()) if w not in _AUDIT_STOP and w != user_name.lower()}
    mem_words, quote_words = words(memory_text), words(source_quote)
    if not mem_words:
        return {"supported": False, "overlap": 0.0}
    overlap = len(mem_words & quote_words) / len(mem_words)
    return {"supported": overlap >= 0.5, "overlap": overlap}


_CITY_PLACES = {"pune": "Lonavala or Mahabaleshwar", "bengaluru": "Coorg or Chikmagalur", "delhi": "Jim Corbett or Rishikesh",
                "mumbai": "Alibaug or Igatpuri"}


def demo_getaway_answer(memory_text: str) -> str:
    t = memory_text.lower()
    m = re.search(r"lives in ([a-z]+)", t)
    if not m:
        return ("Happy to help! First, where do you live, and what kind of food do you eat? "
                "Then I can suggest a weekend getaway that suits you.")
    city = m.group(1)
    place = _CITY_PLACES.get(city, "a hill station a few hours away")
    parts = [f"Since you live in {city.title()}, try {place}: an easy drive for a relaxed weekend."]
    if "vegetarian" in t:
        parts.append("Both have plenty of good vegetarian food.")
    if "motion sickness" in t:
        parts.append("Go by car rather than a long bus ride, since bus journeys make you sick.")
    first = t.strip().splitlines()[0] if t.strip() else ""
    parts.append("The memory that shaped this most: " + first.lstrip("- ").split(" (from")[0])
    return " ".join(parts)


def demo_next_trip(memory_text: str) -> str:
    t = memory_text.lower()
    if "trip episode" in t and "manali" in t:
        return ("Try Udaipur in February: lakeside, quiet, with excellent vegetarian food. It avoids what went wrong in Manali "
                "(8 hours in a taxi, snow closing sights), because it is a short flight with mild weather.")
    return ("Manali again! Old Manali is quiet and Snow Valley Cottage serves vegetarian food, which matches everything "
            "I know about you two. (It does not know you were just there in December: Ticket 4.)")


def demo_episode(debrief: str) -> dict:
    if debrief.strip() == SAMPLE_DEBRIEF:
        return {"when": "20 to 23 December", "what_happened": "Trip to Manali, staying in Old Manali.",
                "what_worked": ["Old Manali was quiet", "Snow Valley Cottage was perfect", "Vegetarian food was great"],
                "what_did_not_work": ["Flight to Chandigarh plus 8 hours in a taxi was exhausting",
                                      "Two days of snow meant no Solang visit"]}
    sentences = [s.strip() for s in re.split(r"(?<=[.!])\s+", debrief) if s.strip()]
    good, bad = [], []
    for s in sentences[1:]:
        if re.search(r"\bbut\b", s, flags=re.I):
            before, _, after = re.split(r"(\bbut\b)", s, maxsplit=1, flags=re.I)
            if before.strip(" ,"):
                good.append(before.strip(" ,."))
            bad.extend(x.strip(" ,.") for x in re.split(r",? and ", after) if x.strip(" ,."))
        else:
            good.append(s.strip(" ."))
    when = re.search(r"\(([^)]+)\)", debrief)
    return {"when": when.group(1) if when else "recently", "what_happened": sentences[0].rstrip("!.") if sentences else debrief,
            "what_worked": good, "what_did_not_work": bad}


def episode_text(ep: dict) -> str:
    return (f"Trip episode ({ep['when']}): {ep['what_happened']} Worked: {'; '.join(ep['what_worked'])}. "
            f"Did not work: {'; '.join(ep['what_did_not_work'])}.")


def demo_rewrite_rules(rules: list[str], feedback: str) -> list[str]:
    fb = feedback.lower()
    out = list(rules)
    if "weather" in fb:
        out = [("The weather tool covers only the next 7 days: for later dates say so, and always use the trip dates the user gave."
                if "weather" in r.lower() else r) for r in out]
    if any(w in fb for w in ("option", "hotel", "short", "long", "brief")):
        out = [("Obey the length the user asks for: when they ask for a short answer, give one or two options at most."
                if "option" in r.lower() else r) for r in out]
    if out == list(rules):
        out.append("Apply this feedback in future answers: " + feedback.strip()[:120])
    return out


def demo_replacement(store: MemoryStore, user_id: str, new_fact: str) -> dict:
    ranked = store.search(new_fact, user_id=user_id, k=1)
    if not ranked or ranked[0][1] < 0.3:
        return {"replaces_memory_id": "none", "reason": "Nothing in memory is close enough in meaning (demo heuristic)."}
    item, sim = ranked[0]
    return {"replaces_memory_id": item["meta"]["memory_id"],
            "reason": f"Closest in meaning (similarity {sim:.2f}). Demo heuristic: real Gemini reasons about the facts."}
