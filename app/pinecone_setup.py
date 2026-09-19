import os
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec


ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")


def create_index():
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is missing.")

    if not INDEX_NAME:
        raise ValueError("PINECONE_INDEX_NAME is missing.")

    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = [index["name"] for index in pc.list_indexes()]

    if INDEX_NAME in existing_indexes:
        print(f"Index '{INDEX_NAME}' already exists.")
        return

    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

    print(f"Created Pinecone index: {INDEX_NAME}")


if __name__ == "__main__":
    create_index()