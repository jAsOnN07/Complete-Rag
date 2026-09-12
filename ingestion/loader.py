"""PDF -> RawDocument.

RBI circulars are digital text (no OCR needed across the current corpus), but
every page carries a bilingual letterhead whose Devanagari glyphs are mangled by
the embedded font encoding. Left in, that boilerplate dominates short chunks and
pollutes both embeddings and citations, so it is stripped here rather than later.
"""

from __future__ import annotations

import asyncio
import re
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import pymupdf

from core.models import DocumentMeta, ExtractionStats, Page, RawDocument, make_doc_id

_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_WORD_CHAR_RE = re.compile(r"[^\W\d_]", re.UNICODE)
_CIRCULAR_NO_RE = re.compile(r"RBI/\d{4}-\d{2,4}/\d+")
_LONG_DATE_RE = re.compile(r"[A-Z][a-z]+\s+\d{1,2},\s+\d{4}")

DEFAULT_HEADER_THRESHOLD = 0.6
DEFAULT_MAX_BOILERPLATE_LINE = 80
DEFAULT_DEVANAGARI_LINE_RATIO = 0.2
# Any Devanagari in a short line is letterhead: substantive circular text is
# English. Mixed lines like "टेलीफोन/Telephone No.022-... , ईमेल:..." sit below
# the ratio threshold because the Latin half is long.
MIXED_SCRIPT_FURNITURE_LEN = 120


def devanagari_ratio(text: str) -> float:
    """Share of letter characters that are Devanagari."""
    letters = _WORD_CHAR_RE.findall(text)
    if not letters:
        return 0.0
    return sum(1 for c in letters if _DEVANAGARI_RE.match(c)) / len(letters)


def normalise_whitespace(text: str) -> str:
    lines = (re.sub(r"[ \t ]+", " ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def strip_repeated_lines(
    pages: list[str],
    *,
    threshold: float = DEFAULT_HEADER_THRESHOLD,
    max_line_len: int = DEFAULT_MAX_BOILERPLATE_LINE,
) -> tuple[list[str], int]:
    """Drop short lines that recur on most pages - headers, footers, page furniture.

    Only short lines qualify: substantive prose can legitimately repeat, and
    removing it would silently delete regulatory text. Single-page documents are
    left untouched, since "on every page" is meaningless for one page.
    """
    if len(pages) < 2:
        return pages, 0

    counts: Counter[str] = Counter()
    for page in pages:
        for line in set(page.splitlines()):
            if line and len(line) <= max_line_len:
                counts[line] += 1

    cutoff = max(2, int(len(pages) * threshold))
    boilerplate = {line for line, n in counts.items() if n >= cutoff}
    if not boilerplate:
        return pages, 0

    cleaned: list[str] = []
    removed = 0
    for page in pages:
        kept: list[str] = []
        for line in page.splitlines():
            if line in boilerplate:
                removed += 1
                continue
            kept.append(line)
        cleaned.append("\n".join(kept))
    return cleaned, removed


def is_letterhead_line(
    line: str,
    *,
    ratio: float = DEFAULT_DEVANAGARI_LINE_RATIO,
    furniture_len: int = MIXED_SCRIPT_FURNITURE_LEN,
) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if devanagari_ratio(stripped) > ratio:
        return True
    # Short bilingual furniture (phone/email lines) sits under the ratio because
    # its Latin half is long.
    if _DEVANAGARI_RE.search(stripped) and len(stripped) < furniture_len:
        return True
    # Orphaned combining glyphs left behind by the embedded-font encoding.
    if len(stripped) <= 2 and not stripped.isascii() and not stripped.isdigit():
        return True
    return False


def drop_devanagari_lines(
    text: str, *, ratio: float = DEFAULT_DEVANAGARI_LINE_RATIO
) -> tuple[str, int]:
    kept: list[str] = []
    removed = 0
    for line in text.splitlines():
        if is_letterhead_line(line, ratio=ratio):
            removed += 1
            continue
        kept.append(line)
    return "\n".join(kept), removed


def _parse_long_date(text: str) -> date | None:
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(text.strip(), fmt).date()
        except ValueError:
            continue
    return None


def extract_metadata_from_text(text: str) -> dict[str, Any]:
    """Last-resort metadata recovery for documents the manifest could not attribute."""
    found: dict[str, Any] = {}
    number = _CIRCULAR_NO_RE.search(text)
    if number:
        found["circular_no"] = number.group(0)
        issued = _LONG_DATE_RE.search(text, number.end())
        if issued:
            found["issued_on"] = _parse_long_date(issued.group(0))
    return found


class PdfLoader:
    """Extracts text and metadata from a circular PDF.

    Manifest metadata wins over anything parsed out of the PDF: RBI header
    formats are not consistent enough to regex reliably, and an ever-growing
    regex is worse than a curated override file.
    """

    def __init__(
        self,
        *,
        header_threshold: float = DEFAULT_HEADER_THRESHOLD,
        max_boilerplate_line: int = DEFAULT_MAX_BOILERPLATE_LINE,
        strip_devanagari: bool = True,
    ) -> None:
        self.header_threshold = header_threshold
        self.max_boilerplate_line = max_boilerplate_line
        self.strip_devanagari = strip_devanagari
        self.last_stats: ExtractionStats | None = None

    def load_sync(
        self, path: Path, manifest_row: dict[str, Any] | None = None
    ) -> RawDocument:
        row = manifest_row or {}
        with pymupdf.open(path) as doc:
            page_texts = [page.get_text("text") for page in doc]

        stripped = 0
        if self.strip_devanagari:
            cleaned: list[str] = []
            for text in page_texts:
                text, removed = drop_devanagari_lines(text)
                stripped += removed
                cleaned.append(text)
            page_texts = cleaned

        page_texts = [normalise_whitespace(t) for t in page_texts]
        page_texts, removed = strip_repeated_lines(
            page_texts,
            threshold=self.header_threshold,
            max_line_len=self.max_boilerplate_line,
        )
        stripped += removed

        pages = [Page.build(i, text) for i, text in enumerate(page_texts, start=1)]
        meta = self._build_meta(path, row, "\n".join(page_texts))

        self.last_stats = ExtractionStats(
            doc_id=meta.doc_id,
            filename=path.name,
            pages=len(pages),
            chars=sum(p.char_count for p in pages),
            empty_pages=sum(1 for p in pages if p.char_count < 100),
            stripped_lines=stripped,
        )
        return RawDocument(meta=meta, pages=pages)

    async def load(
        self, path: Path, manifest_row: dict[str, Any] | None = None
    ) -> RawDocument:
        """PDF parsing is CPU-bound and synchronous - keep it off the event loop."""
        return await asyncio.to_thread(self.load_sync, path, manifest_row)

    async def load_many(
        self, paths: Iterable[tuple[Path, dict[str, Any]]]
    ) -> list[RawDocument]:
        return [await self.load(path, row) for path, row in paths]

    def _build_meta(
        self, path: Path, row: dict[str, Any], text: str
    ) -> DocumentMeta:
        fallback = extract_metadata_from_text(text)
        issued_on = row.get("issued_on") or fallback.get("issued_on")
        if isinstance(issued_on, str):
            issued_on = date.fromisoformat(issued_on)
        return DocumentMeta(
            doc_id=row.get("doc_id") or make_doc_id(str(path)),
            source_path=str(path),
            title=row.get("title") or path.stem.replace("-", " "),
            regulator=row.get("regulator", "RBI"),
            circular_no=row.get("circular_no") or fallback.get("circular_no"),
            issued_on=issued_on,
            source_url=row.get("source_url"),
        )
