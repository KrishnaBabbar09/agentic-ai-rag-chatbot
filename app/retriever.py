import os
import re
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


def tokenize(text: str):
    """Convert text into lowercase words."""
    return set(
        re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    )


def search_documents(query: str, top_k: int = 5):
    """Search Pinecone using the original query."""

    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is missing.")

    if not INDEX_NAME:
        raise ValueError("PINECONE_INDEX_NAME is missing.")

    query_vector = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    results = index.query(
        vector=query_vector,
        top_k=20,
        include_metadata=True
    )

    query_words = tokenize(query)

    candidates = []

    for match in results["matches"]:

        text = match["metadata"]["text"]

        semantic_score = float(match["score"])

        text_words = tokenize(text)

        overlap = query_words.intersection(text_words)

        lexical_score = (
            len(overlap) / max(len(query_words), 1)
        )

        combined_score = (
            semantic_score * 0.75
            + lexical_score * 0.25
        )

        candidates.append({
            "match": match,
            "score": combined_score
        })

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    selected = candidates[:top_k]

    formatted_matches = []

    for item in selected:

        match = item["match"]

        formatted_matches.append({
            "id": match["id"],
            "score": item["score"],
            "metadata": match["metadata"]
        })

    return {
        "matches": formatted_matches
    }
if __name__ == "__main__":

    query = "What are the core pillars of Agentic AI?"

    results = search_documents(
        query,
        top_k=10
    )

    print("\nSearch results:\n")

    for i, match in enumerate(
        results["matches"],
        start=1
    ):

        print(f"--- Result {i} ---")

        print(
            f"Score: {match['score']:.4f}"
        )

        print(
            f"Page: {match['metadata']['page']}"
        )

        print(
            f"Chunk: {match['metadata']['chunk_number']}"
        )

        print(
            f"\n{match['metadata']['text']}\n"
        )