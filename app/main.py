from fastapi import FastAPI
from pydantic import BaseModel

from .graph import rag_graph


app = FastAPI(
    title="Agentic AI RAG Chatbot",
    description="A grounded RAG chatbot built from the Agentic AI eBook.",
    version="1.0.0"
)


class QuestionRequest(BaseModel):
    question: str


class ContextItem(BaseModel):
    text: str
    page: int
    chunk: int
    score: float


class QuestionResponse(BaseModel):
    answer: str
    confidence: float
    retrieved_context: list[ContextItem]


@app.get("/")
def root():
    return {
        "message": "Agentic AI RAG Chatbot API is running!"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):

    result = rag_graph.invoke({
        "question": request.question,
        "retrieved_context": [],
        "answer": "",
        "confidence": 0.0
    })

    return {
        "answer": result["answer"],
        "confidence": result["confidence"],
        "retrieved_context": result["retrieved_context"]
    }