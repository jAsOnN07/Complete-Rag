import json
from pathlib import Path

import pytest

from core.models import RawDocument
from ingestion.loader import (
    PdfLoader,
    devanagari_ratio,
    is_letterhead_line,
    normalise_whitespace,
    strip_repeated_lines,
)

CORPUS = Path("data/raw")
MANIFEST = Path("data/manifest.json")
corpus_available = MANIFEST.exists() and any(CORPUS.glob("*.pdf"))
needs_corpus = pytest.mark.skipif(
    not corpus_available, reason="corpus not fetched (run scripts.fetch_circulars)"
)


def test_normalise_whitespace_collapses_pdf_padding():
    raw = "  RESERVE   BANK  \n\n\n\n   OF INDIA   \n \n  Madam / Sir,  "
    assert normalise_whitespace(raw) == "RESERVE BANK\nOF INDIA\nMadam / Sir,"


def test_normalise_whitespace_keeps_paragraph_separation():
    assert normalise_whitespace("para one\n\n\npara two") == "para one\npara two"


def test_devanagari_ratio_detects_hindi_letterhead():
    assert devanagari_ratio("भारतीय रिज़र्व बैंक") > 0.8
    assert devanagari_ratio("RESERVE BANK OF INDIA") == 0.0
    assert devanagari_ratio("") == 0.0


def test_strip_repeated_lines_removes_boilerplate_present_on_most_pages():
    pages = [
        "RESERVE BANK OF INDIA\nbody one\npage 1 footer",
        "RESERVE BANK OF INDIA\nbody two\npage 2 footer",
        "RESERVE BANK OF INDIA\nbody three\npage 3 footer",
    ]
    cleaned, removed = strip_repeated_lines(pages, threshold=0.6)
    assert removed >= 3
    assert all("RESERVE BANK OF INDIA" not in p for p in cleaned)
    assert "body one" in cleaned[0] and "body three" in cleaned[2]


def test_strip_repeated_lines_keeps_content_on_a_single_page_document():
    """A 1-page doc would otherwise have every line stripped as 'repeated'."""
    pages = ["RESERVE BANK OF INDIA\nthe only substantive paragraph"]
    cleaned, removed = strip_repeated_lines(pages, threshold=0.6)
    assert removed == 0
    assert cleaned == pages


def test_strip_repeated_lines_ignores_long_lines():
    """Body prose can coincide across pages; only short furniture is boilerplate."""
    long_line = "This is a substantive regulatory sentence that recurs verbatim." * 2
    pages = [f"{long_line}\na", f"{long_line}\nb", f"{long_line}\nc"]
    cleaned, _ = strip_repeated_lines(pages, threshold=0.6, max_line_len=80)
    assert all(long_line in p for p in cleaned)


@pytest.mark.parametrize(
    "line,expected",
    [
        ("टेलȣफोन/Telephone No.022-22601000, ईमेल:fedcoecbd@rbi.org.in", True),
        ("ͪ", True),
        ("भारतीय रिज़र्व बैंक", True),
        ('"Caution: RBI never sends mails, SMSs or makes calls asking for personal information', True),
        ("______________________ RESERVE BANK OF INDIA________________________", True),
        ("www.rbi.org.in", True),
        ("Foreign Exchange Department, Central Office ,Central Office building, Fort, Mumbai-400 001, India", True),
        ("RESERVE BANK OF INDIA", False),
        ("RBI/2026-27/254", False),
        ("2. In furtherance of the aforementioned initiative, seven circulars", False),
        ("", False),
    ],
)
def test_is_letterhead_line(line, expected):
    assert is_letterhead_line(line) is expected


def test_letterhead_filter_keeps_long_english_body_text():
    body = (
        "In pursuance of the Reserve Bank's ongoing initiative to rationalise the "
        "regulatory framework under the Foreign Exchange Management Act, 1999."
    )
    assert is_letterhead_line(body) is False


@needs_corpus
class TestAgainstRealCorpus:
    @pytest.fixture(scope="class")
    def manifest_rows(self) -> list[dict]:
        return json.loads(MANIFEST.read_text(encoding="utf-8"))

    @pytest.fixture(scope="class")
    def loaded(self, manifest_rows) -> RawDocument:
        row = manifest_rows[0]
        return PdfLoader().load_sync(CORPUS / row["filename"], row)

    def test_load_returns_pages_with_text(self, loaded):
        assert loaded.pages
        assert loaded.char_count > 500

    def test_metadata_comes_from_manifest(self, loaded, manifest_rows):
        assert loaded.meta.title == manifest_rows[0]["title"]
        assert loaded.meta.circular_no == manifest_rows[0]["circular_no"]
        assert loaded.meta.regulator == "RBI"

    def test_hindi_letterhead_is_stripped(self, loaded):
        """The bilingual RBI letterhead is noise that would pollute embeddings."""
        assert devanagari_ratio(loaded.text) == 0.0

    def test_first_page_keeps_the_circular_identity(self, loaded):
        """Cleaning must not eat the header block the citation depends on."""
        first = loaded.pages[0].text
        assert "RESERVE BANK OF INDIA" in first
        assert loaded.meta.circular_no in first

    def test_no_runs_of_blank_lines(self, loaded):
        assert "\n\n\n" not in loaded.text

    def test_page_offsets_map_back_to_pages(self, loaded):
        assert loaded.page_for_offset(0) == 1
        assert loaded.page_for_offset(len(loaded.text) - 1) == loaded.pages[-1].page_number

    @pytest.mark.asyncio
    async def test_async_load_matches_sync(self, manifest_rows):
        row = manifest_rows[0]
        loader = PdfLoader()
        assert (await loader.load(CORPUS / row["filename"], row)).text == loader.load_sync(
            CORPUS / row["filename"], row
        ).text
