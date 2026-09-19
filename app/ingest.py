import os
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from .chunker import create_chunks


ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def ingest_documents():
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is missing.")

    if not INDEX_NAME:
        raise ValueError("PINECONE_INDEX_NAME is missing.")

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Loading and chunking PDF...")
    chunks = create_chunks()

    print(f"Total chunks: {len(chunks)}")

    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    print("Clearing old vectors from Pinecone...")
    index.delete(delete_all=True)

    vectors = []

    print("Creating embeddings...")

    for i, chunk in enumerate(chunks):
        vector = model.encode(
            chunk["text"],
            normalize_embeddings=True
        ).tolist()

        vectors.append({
            "id": f"chunk-{i}",
            "values": vector,
            "metadata": {
                "text": chunk["text"],
                "page": chunk["page"],
                "chunk_number": chunk["chunk_number"],
                "source": chunk["source"]
            }
        })

        if (i + 1) % 10 == 0:
            print(f"Embedded {i + 1}/{len(chunks)} chunks")

    print("Uploading vectors to Pinecone...")

    batch_size = 50

    for start in range(0, len(vectors), batch_size):
        batch = vectors[start:start + batch_size]

        index.upsert(vectors=batch)

        print(
            f"Uploaded {min(start + batch_size, len(vectors))}"
            f"/{len(vectors)} vectors"
        )

    print("\nIngestion completed successfully!")


if __name__ == "__main__":
    ingest_documents()