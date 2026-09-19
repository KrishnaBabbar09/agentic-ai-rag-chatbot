from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama

from .retriever import search_documents


# =========================================================
# RAG STATE
# =========================================================

class RAGState(TypedDict):
    question: str
    retrieved_context: list
    answer: str
    confidence: float


# =========================================================
# LOCAL LLM
# =========================================================

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# =========================================================
# RETRIEVAL
# =========================================================

def retrieve(state: RAGState):
    """
    Retrieve relevant chunks from Pinecone using multiple
    retrieval queries.

    Multiple queries improve recall when an important
    section is split across multiple chunks.
    """

    question = state["question"]

    queries = [
        question,
        f"{question} perception reasoning planning",
        f"{question} learning execution verification"
    ]

    all_matches = []

    # -----------------------------------------------------
    # Run all retrieval queries
    # -----------------------------------------------------

    for retrieval_query in queries:

        results = search_documents(
            query=retrieval_query,
            top_k=5
        )

        all_matches.extend(
            results["matches"]
        )

    # -----------------------------------------------------
    # Remove duplicate chunks
    # Keep the strongest score for each chunk
    # -----------------------------------------------------

    unique_matches = {}

    for match in all_matches:

        match_id = match["id"]

        if (
            match_id not in unique_matches
            or match["score"] > unique_matches[match_id]["score"]
        ):
            unique_matches[match_id] = match

    matches = list(unique_matches.values())

    # -----------------------------------------------------
    # Sort by retrieval score
    # -----------------------------------------------------

    matches.sort(
        key=lambda match: match["score"],
        reverse=True
    )

    # -----------------------------------------------------
    # Identify strongest page
    #
    # This helps preserve related chunks when a section
    # spans multiple chunks on the same page.
    # -----------------------------------------------------

    strongest_page = None

    if matches:
        strongest_page = int(
            matches[0]["metadata"]["page"]
        )

    # -----------------------------------------------------
    # Separate chunks from strongest page
    # -----------------------------------------------------

    same_page = []
    other_pages = []

    for match in matches:

        page = int(
            match["metadata"]["page"]
        )

        if page == strongest_page:
            same_page.append(match)
        else:
            other_pages.append(match)

    # -----------------------------------------------------
    # Keep chunks from the strongest page in document order
    # -----------------------------------------------------

    same_page.sort(
        key=lambda match: int(
            match["metadata"]["chunk_number"]
        )
    )

    # -----------------------------------------------------
    # Keep chunks from other pages ordered by score
    # -----------------------------------------------------

    other_pages.sort(
        key=lambda match: match["score"],
        reverse=True
    )

    # -----------------------------------------------------
    # Combine results
    #
    # Strongest section comes first.
    # -----------------------------------------------------

    matches = same_page + other_pages

    # Keep enough context for generation.
    matches = matches[:10]

    # -----------------------------------------------------
    # Build final context
    # -----------------------------------------------------

    context = []

    for match in matches:

        metadata = match["metadata"]

        context.append({
            "text": metadata["text"],
            "page": int(metadata["page"]),
            "chunk": int(metadata["chunk_number"]),
            "score": float(match["score"])
        })

    # -----------------------------------------------------
    # Retrieval confidence
    #
    # This is a retrieval-score proxy, NOT a calibrated
    # probability.
    # -----------------------------------------------------

    confidence = max(
        [
            item["score"]
            for item in context
        ],
        default=0.0
    )

    return {
        "retrieved_context": context,
        "confidence": confidence
    }


# =========================================================
# RELEVANCE CHECK
# =========================================================

def check_relevance(state: RAGState):
    """
    Decide whether the retrieved context is strong enough
    to send to the LLM.
    """

    context = state["retrieved_context"]

    # No retrieved context
    if not context:
        return "not_relevant"

    # -----------------------------------------------------
    # IMPORTANT:
    #
    # Our retrieval score is a similarity/relevance proxy,
    # not a calibrated probability.
    #
    # The correct Page 19 result currently scores around
    # 0.417, so 0.45 was too strict.
    # -----------------------------------------------------

    if state["confidence"] < 0.35:
        return "not_relevant"

    return "relevant"


# =========================================================
# GENERATION
# =========================================================

def generate(state: RAGState):
    """
    Generate an answer using ONLY the retrieved eBook
    context.
    """

    question = state["question"]
    context = state["retrieved_context"]

    # -----------------------------------------------------
    # Format retrieved context
    # -----------------------------------------------------

    context_text = "\n\n".join(
        [
            (
                f"[Page {item['page']} | "
                f"Chunk {item['chunk']}]\n"
                f"{item['text']}"
            )
            for item in context
        ]
    )

    # -----------------------------------------------------
    # Grounded generation prompt
    # -----------------------------------------------------

    prompt = f"""
You are a grounded question-answering assistant.

Your job is to answer the user's question ONLY from the
provided context from the Agentic AI eBook.

STRICT RULES:

1. Use ONLY information present in the provided context.

2. Do NOT use outside knowledge.

3. Do NOT invent, assume, or infer information that is
   not supported by the provided context.

4. If the answer is a list, inspect ALL relevant context
   chunks before answering.

5. If a list or explanation continues into another chunk,
   combine the relevant chunks into one complete answer.

6. Pay special attention to chunks from the same page or
   section because they may contain a continuation.

7. Do not stop after finding only part of an answer.

8. If the provided context does not contain enough
   information to answer the question, respond exactly
   with:

"I couldn't find enough information about this in the
provided Agentic AI eBook."

9. Keep the answer clear and concise.

Context from the Agentic AI eBook:
========================================

{context_text}

========================================

User question:
{question}

Before answering, carefully inspect all retrieved context
chunks and combine relevant information when the answer
continues across chunks.

Answer:
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


# =========================================================
# NO ANSWER
# =========================================================

def no_answer(state: RAGState):
    """
    Return a grounded response when retrieval is
    insufficient.
    """

    return {
        "answer": (
            "I couldn't find enough information about this "
            "in the provided Agentic AI eBook."
        )
    }


# =========================================================
# BUILD LANGGRAPH
# =========================================================

builder = StateGraph(RAGState)

# Add nodes
builder.add_node(
    "retrieve",
    retrieve
)

builder.add_node(
    "generate",
    generate
)

builder.add_node(
    "no_answer",
    no_answer
)


# =========================================================
# GRAPH FLOW
# =========================================================

# START → RETRIEVE

builder.add_edge(
    START,
    "retrieve"
)


# RETRIEVE → GENERATE / NO ANSWER

builder.add_conditional_edges(
    "retrieve",
    check_relevance,
    {
        "relevant": "generate",
        "not_relevant": "no_answer"
    }
)


# GENERATE → END

builder.add_edge(
    "generate",
    END
)


# NO ANSWER → END

builder.add_edge(
    "no_answer",
    END
)


# =========================================================
# COMPILE GRAPH
# =========================================================

rag_graph = builder.compile()


# =========================================================
# DIRECT TEST
# =========================================================

if __name__ == "__main__":

    question = "What are the core pillars of Agentic AI?"

    result = rag_graph.invoke({
        "question": question,
        "retrieved_context": [],
        "answer": "",
        "confidence": 0.0
    })

    print("\n==============================")
    print("ANSWER")
    print("==============================")

    print(result["answer"])

    print("\n==============================")
    print("RETRIEVAL SCORE")
    print("==============================")

    print(
        f"{result['confidence']:.4f}"
    )

    print("\n==============================")
    print("RETRIEVED CONTEXT")
    print("==============================")

    for item in result["retrieved_context"]:

        print(
            f"\nPage {item['page']} | "
            f"Chunk {item['chunk']} | "
            f"Score {item['score']:.4f}"
        )

        print(
            item["text"][:500]
        )