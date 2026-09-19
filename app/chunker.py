from langchain_text_splitters import RecursiveCharacterTextSplitter

from .pdf_loader import load_pdf


def create_chunks():
    pages = load_pdf()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []

    for page in pages:
        page_chunks = text_splitter.split_text(page["text"])

        for chunk_number, chunk_text in enumerate(
            page_chunks,
            start=1
        ):
            chunks.append({
                "text": chunk_text,
                "page": page["page"],
                "chunk_number": chunk_number,
                "source": "Ebook-Agentic-AI.pdf"
            })

    return chunks


if __name__ == "__main__":
    chunks = create_chunks()

    print(f"Total chunks: {len(chunks)}")

    for chunk in chunks:
        if chunk["page"] == 19:
            print(
                f"\n--- Page {chunk['page']} "
                f"Chunk {chunk['chunk_number']} ---"
            )
            print(chunk["text"])