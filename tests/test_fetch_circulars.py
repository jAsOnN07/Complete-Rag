from datetime import date
from pathlib import Path

import pytest

from scripts.fetch_circulars import (
    RBI_LISTING_URL,
    CircularRef,
    is_notification_pdf,
    parse_detail,
    parse_listing,
    slugify,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def listing_html() -> str:
    return (FIXTURES / "rbi_listing.html").read_text(encoding="utf-8")


@pytest.fixture
def detail_html() -> str:
    return (FIXTURES / "rbi_detail.html").read_text(encoding="utf-8")


def test_parse_listing_finds_circulars(listing_html):
    refs = parse_listing(listing_html, base_url=RBI_LISTING_URL)
    assert refs, "expected at least one circular in the fixture"
    assert all(isinstance(r, CircularRef) for r in refs)
    assert all(r.regulator == "RBI" for r in refs)


def test_parse_listing_carries_date_header_forward(listing_html):
    """RBI emits a date row followed by N title rows; every title inherits that date."""
    refs = parse_listing(listing_html, base_url=RBI_LISTING_URL)
    dated = [r for r in refs if r.issued_on is not None]
    assert dated, "no circular picked up a date from its header row"
    assert all(isinstance(r.issued_on, date) for r in dated)


def test_parse_listing_resolves_absolute_detail_urls(listing_html):
    refs = parse_listing(listing_html, base_url=RBI_LISTING_URL)
    assert all(r.detail_url.startswith("https://") for r in refs)
    assert any("NotificationUser.aspx?Id=" in r.detail_url for r in refs)


def test_parse_listing_titles_are_clean(listing_html):
    refs = parse_listing(listing_html, base_url=RBI_LISTING_URL)
    for r in refs:
        assert r.title.strip() == r.title
        assert len(r.title) > 10
        assert "kb" not in r.title[-6:].lower()


def test_parse_detail_extracts_circular_number(detail_html):
    detail = parse_detail(detail_html)
    assert detail.circular_no == "RBI/2026-27/219"


def test_parse_detail_extracts_real_title(detail_html):
    """ID-walked circulars have no listing row, so the title must come from here."""
    detail = parse_detail(detail_html)
    assert detail.title is not None
    assert "Urban Co-operative Banks" in detail.title
    assert "kb" not in detail.title[-6:].lower()


def test_parse_detail_extracts_issue_date(detail_html):
    """The issue date is the first date after the circular number.

    A later date in the body references a prior circular, so "first" matters.
    """
    detail = parse_detail(detail_html)
    assert detail.issued_on == date(2026, 7, 30)


def test_parse_detail_picks_notification_pdf_not_site_footer_pdf(detail_html):
    """The page also links a site-wide footer PDF; picking it would poison the corpus."""
    detail = parse_detail(detail_html)
    assert detail.pdf_url is not None
    assert "/notification/PDFs/" in detail.pdf_url
    assert "Utkarsh" not in detail.pdf_url


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://rbidocs.rbi.org.in/rdocs/notification/PDFs/NT254ABC.PDF", True),
        ("https://rbidocs.rbi.org.in/rdocs/content/pdfs/Utkarsh2029.pdf", False),
        ("https://rbidocs.rbi.org.in/rdocs/Notification/PDFs/lower.pdf", True),
        ("https://example.com/some/other.pdf", False),
    ],
)
def test_is_notification_pdf(url, expected):
    assert is_notification_pdf(url) is expected


def test_slugify_produces_safe_stable_filenames():
    assert slugify("Review of Circulars issued under FEMA, 1999") == (
        "review-of-circulars-issued-under-fema-1999"
    )
    assert slugify("KYC/AML: Updates & Amendments") == "kyc-aml-updates-amendments"
    assert len(slugify("x" * 300)) <= 80


def test_slugify_is_deterministic():
    title = "Implementation of Section 51A of UAPA, 1967"
    assert slugify(title) == slugify(title)
