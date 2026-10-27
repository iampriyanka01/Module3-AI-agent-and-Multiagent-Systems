"""Settings and the shared models. Every other file imports what it needs from here.

Keys are read from the .env file (copy .env.example to .env and fill it in).
"""

import os

import yaml
from dotenv import load_dotenv
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv(".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
LITEAPI_API_KEY = os.getenv("LITEAPI_API_KEY", "")      # the PRIVATE sandbox key from the LiteAPI dashboard

PINECONE_INDEX = "dilsetrips-memory"
EMBEDDING_SIZE = 768        # the Pinecone index is created with this size; the two must match

# One model for every agent, limited to 4 calls a minute so we stay within Gemini's free tier
limiter = InMemoryRateLimiter(requests_per_second=4 / 60, check_every_n_seconds=0.5, max_bucket_size=1)
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", google_api_key=GOOGLE_API_KEY, rate_limiter=limiter)

# Turns text into numbers (vectors), so Pinecone can find memories with a similar meaning
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GOOGLE_API_KEY,
                                          output_dimensionality=EMBEDDING_SIZE)

# Every prompt lives in prompts.yaml. PROMPTS["perception"] is the perception prompt, and so on.
with open("prompts.yaml", encoding="utf-8") as prompt_file:
    PROMPTS = yaml.safe_load(prompt_file)


def missing_keys():
    """The names of any keys not set in .env yet."""
    missing = []
    if GOOGLE_API_KEY == "":
        missing.append("GOOGLE_API_KEY")
    if PINECONE_API_KEY == "":
        missing.append("PINECONE_API_KEY")
    if LITEAPI_API_KEY == "":
        missing.append("LITEAPI_API_KEY")
    return missing
