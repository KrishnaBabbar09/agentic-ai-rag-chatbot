Agentic AI RAG Chatbot

A Python-based Retrieval-Augmented Generation (RAG) chatbot that answers questions strictly using the provided Agentic AI eBook. The project uses LangGraph for workflow orchestration, Pinecone for vector search, Sentence Transformers for embeddings, and Ollama/Llama 3.2 for local answer generation.

Features
PDF document ingestion using PyMuPDF
Recursive text chunking with overlap
Semantic embeddings using all-MiniLM-L6-v2
Vector storage and retrieval using Pinecone
Multi-query retrieval for improved context retrieval
LangGraph-based RAG workflow
Relevance gate to reduce unsupported answers
Local LLM generation using Ollama
Answers grounded only in retrieved eBook content
Returns:
Final answer
Retrieval confidence/relevance score
Retrieved context chunks
Source page and chunk information
FastAPI REST API
Interactive Swagger API documentation
Architecture
                    Agentic AI eBook PDF
                            |
                            v
                       PyMuPDF
                            |
                            v
                    Text Chunking
                            |
                            v
              Sentence Transformer
               all-MiniLM-L6-v2
                            |
                            v
                        Pinecone
                     Vector Database
                            |
                            |
User Question -----------> RAG Workflow
                              |
                              v
                    Multi-Query Retrieval
                              |
                              v
                       Relevance Gate
                         /        \
                        /          \
                 Relevant       Not Relevant
                    |                 |
                    v                 v
              Ollama LLM        Grounded fallback
             Llama 3.2 3B
                    |
                    v
             Grounded Answer
                    |
                    v
                FastAPI
                    |
                    v
              JSON Response

              Technology Stack
Technology	Purpose
Python 3.12	Programming language
PyMuPDF	PDF text extraction
LangChain Text Splitters	Text chunking
Sentence Transformers	Text embeddings
Pinecone	Vector database
LangGraph	RAG workflow orchestration
Ollama	Local LLM runtime
Llama 3.2 3B	Answer generation
FastAPI	REST API
Pydantic	Request/response validation
Uvicorn	API server

Project Structure
agentic-ai-rag-chatbot/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── graph.py
│   ├── retriever.py
│   ├── ingest.py
│   ├── chunker.py
│   ├── pdf_loader.py
│   ├── pinecone_setup.py
│   ├── embeddings.py
│   └── config_test.py
│
├── data/
│   └── Ebook-Agentic-AI.pdf
│
├── scripts/
├── tests/
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
> The provided eBook PDF is kept locally and is excluded from Git using `.gitignore`.

## Prerequisites

Before running the project, make sure you have:

- Python 3.12
- A Pinecone account and API key
- Ollama

### Ollama Setup

Install Ollama and download the required model:

```powershell
ollama pull llama3.2:3b