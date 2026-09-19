# Agentic AI RAG Chatbot

A Python-based Retrieval-Augmented Generation (RAG) chatbot that answers questions strictly using the provided **Agentic AI eBook**.

The project uses:

- **LangGraph** for RAG workflow orchestration
- **Pinecone** for vector storage and retrieval
- **Sentence Transformers** for semantic embeddings
- **Ollama / Llama 3.2 3B** for local answer generation
- **FastAPI** for the REST API
- **PyMuPDF** for PDF text extraction

---

## Features

- PDF document ingestion using PyMuPDF
- Recursive text chunking with overlap
- Semantic embeddings using `all-MiniLM-L6-v2`
- Vector storage and retrieval using Pinecone
- Multi-query retrieval for improved context retrieval
- LangGraph-based RAG workflow
- Retrieval relevance gate
- Local LLM generation using Ollama
- Answers grounded only in retrieved eBook content
- Returns:
  - Final answer
  - Retrieval relevance score
  - Retrieved context chunks
  - Source page and chunk information
- FastAPI REST API
- Interactive Swagger API documentation

---

## Architecture

```text
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
User Question ------------> RAG Workflow
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
              Ollama LLM        Grounded Fallback
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
```

---

## RAG Workflow

The chatbot follows this pipeline:

1. The Agentic AI eBook PDF is loaded using PyMuPDF.
2. Extracted text is divided into overlapping chunks.
3. Each chunk is converted into a 384-dimensional embedding using `all-MiniLM-L6-v2`.
4. The embeddings and chunk metadata are stored in Pinecone.
5. A user question is converted into an embedding.
6. Multiple retrieval queries are used to improve context retrieval.
7. Pinecone retrieves candidate chunks.
8. Semantic similarity and lexical overlap are combined into a retrieval relevance score.
9. The most relevant context chunks are passed to the LLM.
10. The LLM is instructed to answer only from the retrieved eBook context.
11. If sufficient relevant information is not available, the system returns a grounded fallback response.
12. The API returns the answer, retrieval score, and retrieved context.

---

## Grounding Strategy

The chatbot is designed to prevent unsupported answers by providing the LLM with only retrieved content from the Agentic AI eBook.

The generation prompt instructs the LLM to:

- Use only the provided eBook context.
- Avoid outside knowledge.
- Avoid inventing information.
- Combine relevant chunks when an answer continues across multiple chunks.
- Return a fallback response when the retrieved context does not contain enough information.

This keeps the generated response grounded in the supplied document.

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.12 | Programming language |
| PyMuPDF | PDF text extraction |
| LangChain Text Splitters | Text chunking |
| Sentence Transformers | Text embeddings |
| all-MiniLM-L6-v2 | Embedding model |
| Pinecone | Vector database |
| LangGraph | RAG workflow orchestration |
| Ollama | Local LLM runtime |
| Llama 3.2 3B | Answer generation |
| FastAPI | REST API |
| Pydantic | Request/response validation |
| Uvicorn | API server |

---

## Project Structure

```text
agentic-ai-rag-chatbot/
|
├── app/
|   ├── __init__.py
|   ├── main.py
|   ├── graph.py
|   ├── retriever.py
|   ├── ingest.py
|   ├── chunker.py
|   ├── pdf_loader.py
|   ├── pinecone_setup.py
|   ├── embeddings.py
|   └── config_test.py
|
├── data/
|   └── Ebook-Agentic-AI.pdf
|
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
└── LICENSE
```

> The provided eBook PDF is kept locally and is excluded from Git using `.gitignore`.

---

## Prerequisites

Before running the project, make sure you have:

- Python 3.12
- A Pinecone account and API key
- Ollama
- Git

---

## 1. Clone the Repository

```bash
git clone https://github.com/KrishnaBabbar09/agentic-ai-rag-chatbot.git
cd agentic-ai-rag-chatbot
```

---

## 2. Create a Virtual Environment

### Windows PowerShell

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

---

## 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

---

## 4. Install and Configure Ollama

Install Ollama and download the required model:

```powershell
ollama pull llama3.2:3b
```

Make sure Ollama is running before starting the chatbot.

---

## 5. Add the PDF

Place the provided eBook at:

```text
data/Ebook-Agentic-AI.pdf
```

The PDF is intentionally excluded from Git because of the project's `.gitignore` configuration.

---

## 6. Configure Environment Variables

Create a `.env` file in the project root.

Use `.env.example` as a template:

```text
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=agentic-ai-rag
```

Do not commit the `.env` file to GitHub.

---

## 7. Create the Pinecone Index

Run:

```powershell
python -m app.pinecone_setup
```

The project creates a Pinecone index with:

- Dimension: `384`
- Metric: `cosine`
- Cloud: AWS
- Region: `us-east-1`

The dimension matches the output size of `all-MiniLM-L6-v2`.

---

## 8. Ingest the PDF

Run:

```powershell
python -m app.ingest
```

The ingestion process is:

```text
PDF
 ↓
Text Extraction
 ↓
Chunking
 ↓
Embedding Generation
 ↓
Pinecone Upload
```

The current document produces approximately 96 text chunks with the configured chunking settings.

---

## 9. Run the FastAPI Server

Start the API with:

```powershell
uvicorn app.main:app --port 8000
```

If port `8000` is already in use, another port can be used:

```powershell
uvicorn app.main:app --port 8001
```

The API will then be available at:

```text
http://127.0.0.1:8000
```

or, when using port 8001:

```text
http://127.0.0.1:8001
```

---

## 10. Swagger API Documentation

FastAPI provides interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

If using port 8001:

```text
http://127.0.0.1:8001/docs
```

The `/docs` page can be used to test the chatbot without requiring a separate frontend.

---

## API Endpoints

### GET `/`

Checks whether the API is running.

Example response:

```json
{
  "message": "Agentic AI RAG Chatbot API is running!"
}
```

---

### GET `/health`

Returns the API health status.

Example response:

```json
{
  "status": "healthy"
}
```

---

### POST `/ask`

Accepts a question and runs the complete LangGraph RAG workflow.

Request:

```json
{
  "question": "What are the core pillars of Agentic AI?"
}
```

Response structure:

```json
{
  "answer": "The six core pillars of Agentic AI are ...",
  "confidence": 0.7230,
  "retrieved_context": [
    {
      "text": "Retrieved eBook content...",
      "page": 19,
      "chunk": 1,
      "score": 0.7230
    }
  ]
}
```

The exact answer, score, and retrieved chunks depend on the query.

---

## Retrieval Score

The API returns a `confidence` field representing the strongest combined retrieval relevance score among the selected context chunks.

The retrieval score combines:

- Semantic similarity from Pinecone
- Lexical overlap between the query and retrieved text

The score is used as a **relevance signal**.

It should not be interpreted as a calibrated probability or as a percentage chance that the generated answer is correct.

---

## Sample Queries

The following queries can be used to test the chatbot:

1. **What is Agentic AI?**

2. **How does Agentic AI differ from an LLM?**

3. **What are the core pillars of Agentic AI?**

4. **What are the benefits and challenges of multi-agent systems?**

5. **What are the building blocks of an agentic AI system?**

6. **What are common execution patterns in multi-agent workflows?**

### Out-of-Scope Example

```text
What is the capital of France?
```

This question is unrelated to the provided Agentic AI eBook and can be used to test the chatbot's grounding behavior.

---

## Example Core Concepts

The chatbot can retrieve and answer questions related to topics covered in the provided eBook, including:

- Agentic AI
- Large Language Models and agents
- Core pillars of Agentic AI
- Agent characteristics and types
- Multi-Agent Systems
- Multi-Agent System architecture
- Agent orchestration
- Planning and task decomposition
- Execution patterns
- Multi-agent sales forecasting scenarios

---

## Security

Sensitive configuration values should be stored in `.env`.

The repository intentionally excludes:

```text
.env
.venv/
data/*.pdf
```

API keys should never be committed to the repository.

---

## Limitations

- The chatbot is restricted to the provided Agentic AI eBook.
- Answer quality depends on retrieval quality and the local LLM.
- Retrieval scores are relevance signals rather than calibrated confidence probabilities.
- The application currently uses a local Llama 3.2 3B model through Ollama.
- The provided PDF must be supplied locally before ingestion.
- Pinecone credentials are required for vector storage and retrieval.
- There is currently no authentication layer for the FastAPI endpoint.

---

## Future Improvements

Possible improvements include:

- Add a dedicated chat frontend using Streamlit or React.
- Add conversational memory for multi-turn conversations.
- Add automated evaluation using a fixed question-answer test set.
- Improve retrieval using a dedicated reranking model.
- Add document citation formatting to generated answers.
- Add API authentication.
- Add automated tests for ingestion, retrieval, and generation.
- Add Docker support for easier deployment.
- Add monitoring and logging for retrieval quality and API performance.

---

## Assignment Requirement Mapping

| Requirement | Implementation |
|---|---|
| PDF ingestion | PyMuPDF |
| Text chunking | RecursiveCharacterTextSplitter |
| Embeddings | Sentence Transformers |
| Vector database | Pinecone |
| RAG framework | LangGraph |
| LLM generation | Ollama / Llama 3.2 3B |
| Grounded answers | Context-restricted generation prompt |
| API | FastAPI |
| Final answer | `/ask` response |
| Retrieved context | `/ask` response |
| Confidence / score | Retrieval relevance score |
| Sample queries | Included above |
| Architecture explanation | Included above |
| Setup instructions | Included above |

---

## License

This project is licensed under the MIT License.

---

## Author

**Krishna Babbar**

GitHub: https://github.com/KrishnaBabbar09