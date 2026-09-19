from sentence_transformers import SentenceTransformer

from .chunker import create_chunks


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def create_embeddings():
    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    chunks = create_chunks()

    first_chunk = chunks[0]["text"]

    vector = model.encode(first_chunk)

    print("Embedding created successfully!")
    print(f"Embedding dimensions: {len(vector)}")
    print(f"First 5 values: {vector[:5]}")


if __name__ == "__main__":
    create_embeddings()