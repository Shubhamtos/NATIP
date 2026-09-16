"""Safe PDF extraction helpers for quarterly results."""

from __future__ import annotations

from io import BytesIO

from app.intelligence.quarterly_results.models import ExtractedFact

PDF_MAX_BYTES = 20 * 1024 * 1024
PDF_MAX_PAGES = 50

FACT_KEYWORDS = {
    "Financial tables": ("revenue", "sales", "income", "profit", "eps", "margin"),
    "Segment performance": ("segment", "business", "division", "geography"),
    "Other income / exceptional": ("other income", "exceptional", "one-time", "extraordinary"),
    "Finance cost / debt": ("finance cost", "interest", "borrowings", "debt"),
    "Cash flow / balance sheet": ("cash flow", "balance sheet", "working capital", "receivable"),
    "Audit observations": ("limited review", "auditor", "qualification", "emphasis of matter"),
    "Management commentary": ("outlook", "guidance", "order book", "demand", "capacity"),
}


def extract_pdf_facts(pdf_bytes: bytes, *, document_url: str) -> list[ExtractedFact]:
    """Extract page-level evidence facts from a quarterly result PDF.

    The extraction is deterministic and treats document text as untrusted evidence.
    OCR is not bundled; scanned PDFs are flagged through a low-confidence missing-text fact.
    """

    if len(pdf_bytes) > PDF_MAX_BYTES:
        return [
            ExtractedFact(
                label="PDF extraction",
                value="Skipped",
                source=document_url,
                excerpt="PDF exceeded NATIP download safety limit.",
                confidence=0.0,
            )
        ]
    try:
        from pypdf import PdfReader
    except Exception as exc:
        raise RuntimeError("Install pypdf to extract quarterly result PDFs.") from exc

    reader = PdfReader(BytesIO(pdf_bytes))
    facts: list[ExtractedFact] = []
    pages_checked = min(len(reader.pages), PDF_MAX_PAGES)
    text_seen = False
    for page_index in range(pages_checked):
        text = reader.pages[page_index].extract_text() or ""
        snippets = _snippets(text)
        if snippets:
            text_seen = True
        for label, keywords in FACT_KEYWORDS.items():
            match = _best_keyword_snippet(snippets, keywords)
            if match:
                facts.append(
                    ExtractedFact(
                        label=label,
                        value="Found",
                        source=document_url,
                        page_number=page_index + 1,
                        excerpt=match,
                        confidence=min(0.95, 0.45 + 0.1 * _keyword_count(match, keywords)),
                    )
                )
    if not text_seen:
        facts.append(
            ExtractedFact(
                label="PDF extraction",
                value="Needs OCR",
                source=document_url,
                excerpt=(
                    "No selectable text was extracted. The document may be scanned; OCR is required "
                    "before facts can be validated."
                ),
                confidence=0.0,
            )
        )
    return facts


def _snippets(text: str) -> list[str]:
    """Return readable snippets from page text."""

    normalized = " ".join(text.replace("\x00", " ").split())
    parts = normalized.replace(";", ".").split(".")
    return [
        part.strip()
        for part in parts
        if 35 <= len(part.strip()) <= 360 and any(character.isalpha() for character in part)
    ]


def _best_keyword_snippet(snippets: list[str], keywords: tuple[str, ...]) -> str | None:
    """Return the highest keyword-count snippet."""

    scored = [(_keyword_count(snippet, keywords), snippet) for snippet in snippets]
    scored = [(score, snippet) for score, snippet in scored if score > 0]
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], -len(item[1])), reverse=True)
    return scored[0][1]


def _keyword_count(text: str, keywords: tuple[str, ...]) -> int:
    """Count keyword hits in text."""

    lowered = text.casefold()
    return sum(1 for keyword in keywords if keyword in lowered)
