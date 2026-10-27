"""Long-term memory in Pinecone (Session 3).

Pinecone stores each memory as a vector (numbers that capture its meaning), so we can find memories
with a SIMILAR meaning to a question. Each couple gets their own "namespace": a separate drawer,
so one couple's memories never mix with another's.

The LLM makes two judgement calls, both with structured output:
  remember(): is anything in this message worth remembering, and how important is it?
  recall():   of the memories Pinecone found, which ones actually matter for this question?
Code does the checking: the quote must really be in the message, importance must be 1 to 5,
and we skip anything we already know.

Try it:   python memory.py
"""

from datetime import date

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from config import EMBEDDING_SIZE, PINECONE_API_KEY, PINECONE_INDEX, PROMPTS, embeddings, llm

pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
if not pinecone_client.has_index(PINECONE_INDEX):
    print(f"Creating the Pinecone index '{PINECONE_INDEX}' (first run only, takes up to a minute)...")
    pinecone_client.create_index(name=PINECONE_INDEX, dimension=EMBEDDING_SIZE, metric="cosine",
                                 spec=ServerlessSpec(cloud="aws", region="us-east-1"))
index = pinecone_client.Index(PINECONE_INDEX)
store = PineconeVectorStore(index=index, embedding=embeddings)

# The shape of the LLM's answer when it decides what to remember
EXTRACTION_SCHEMA = {
    "title": "memories",
    "description": "Lasting facts worth remembering from one message",
    "type": "object",
    "properties": {
        "memories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The fact, as one short sentence"},
                    "kind": {"type": "string", "enum": ["semantic", "episodic", "procedural"]},
                    "importance": {"type": "integer", "description": "From 1 (trivia) to 5 (critical, e.g. a food or health need)"},
                    "quote": {"type": "string", "description": "Their exact words this fact comes from"},
                },
                "required": ["text", "kind", "importance", "quote"],
            },
        },
    },
    "required": ["memories"],
}

# The shape of the LLM's answer when it judges which memories matter
RECALL_SCHEMA = {
    "title": "memory_check",
    "description": "For each numbered memory, whether it matters for the question",
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "number": {"type": "integer"},
                    "keep": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["number", "keep", "reason"],
            },
        },
    },
    "required": ["verdicts"],
}


def remember(couple, message):
    """Find lasting facts in a message and save them. Returns the list of facts saved."""
    result = llm.with_structured_output(EXTRACTION_SCHEMA).invoke(PROMPTS["memory_extraction"].format(message=message))
    saved = []
    for memory in result["memories"]:
        # Check 1: the quote must really be in their message (stops invented facts)
        if memory["quote"].lower() not in message.lower():
            print(f"   [memory] refused: '{memory['text']}' (they never said that)")
            continue
        # Check 2: importance must be between 1 and 5
        importance = min(max(memory["importance"], 1), 5)
        # Check 3: skip it if we already know something almost identical
        similar = store.similarity_search_with_score(memory["text"], k=1, namespace=couple)
        if len(similar) > 0 and similar[0][1] >= 0.92:
            print(f"   [memory] already known: '{memory['text']}'")
            continue
        details = {"kind": memory["kind"], "importance": importance, "quote": memory["quote"],
                   "saved_on": date.today().isoformat()}
        store.add_texts([memory["text"]], metadatas=[details], namespace=couple)
        print(f"   [memory] saved ({memory['kind']}, importance {importance}): {memory['text']}")
        saved.append(memory["text"])
    return saved


def recall(couple, question):
    """The memories that matter for a question, chosen by the LLM from Pinecone's most similar ones."""
    found = store.similarity_search_with_score(question, k=6, namespace=couple)
    if len(found) == 0:
        return "No memories about this couple yet."

    # Number the memories, so the LLM can refer to them simply
    numbered = ""
    for number in range(len(found)):
        document = found[number][0]
        numbered = numbered + (f"{number + 1}. {document.page_content} (importance {document.metadata.get('importance')}, "
                               f"saved on {document.metadata.get('saved_on')})\n")

    judgement = llm.with_structured_output(RECALL_SCHEMA).invoke(
        PROMPTS["memory_recall"].format(question=question, memories=numbered))

    kept = []
    for verdict in judgement["verdicts"]:
        position = verdict["number"] - 1
        if position < 0 or position >= len(found):
            continue                       # the LLM gave a number we never showed it: ignore it
        text = found[position][0].page_content
        if verdict["keep"]:
            print(f"   [recall] keep: {text}  ({verdict['reason']})")
            kept.append("- " + text)
        else:
            print(f"   [recall] drop: {text}  ({verdict['reason']})")
    if len(kept) == 0:
        return "No memories that matter for this question."
    return "\n".join(kept)


def show_memories(couple):
    """Every memory saved for a couple, as a list of dictionaries."""
    memories = []
    for page_of_ids in index.list(namespace=couple):
        fetched = index.fetch(ids=page_of_ids, namespace=couple)
        for memory_id in fetched.vectors:
            memories.append(fetched.vectors[memory_id].metadata)
    return memories


def forget(couple):
    """The right to be forgotten: delete everything saved for a couple."""
    index.delete(delete_all=True, namespace=couple)


if __name__ == "__main__":
    remember("demo", "We hate long layovers, and Ananya is vegetarian. No sugar in my chai, please.")
    print(recall("demo", "Find us a flight from Mumbai to Chandigarh"))
    forget("demo")
