import os
from dotenv import load_dotenv

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")

print("OpenAI key loaded:", bool(openai_key))
print("Pinecone key loaded:", bool(pinecone_key))
print("Pinecone index:", index_name)