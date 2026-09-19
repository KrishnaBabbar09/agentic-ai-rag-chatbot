from pathlib import Path
import pymupdf


PDF_PATH = Path(__file__).resolve().parent.parent / "data" / "Ebook-Agentic-AI.pdf"


def load_pdf():
    """Load the PDF and return its text page-by-page."""

    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    pages = []

    with pymupdf.open(PDF_PATH) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if text:
                pages.append({
                    "page": page_number,
                    "text": text
                })

    return pages


if __name__ == "__main__":
    pages = load_pdf()

    print(f"PDF loaded successfully!")
    print(f"Total pages with text: {len(pages)}")

    if pages:
        print("\nFirst page preview:")
        print(pages[0]["text"][:1000])