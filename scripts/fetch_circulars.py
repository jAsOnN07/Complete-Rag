"""Download RBI circulars into data/raw/ and write a curation manifest.

Build utility, not part of the served system - keep it out of the Docker image.
Fetch a broad batch, then curate by hand: delete what you don't want from
data/raw/ and re-run with --rebuild-manifest so the manifest matches what
survived.

    python -m scripts.fetch_circulars --limit 40 --dry-run
    python -m scripts.fetch_circulars --limit 40
    python -m scripts.fetch_circulars --rebuild-manifest
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import sys
import unicodedata
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Literal
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel

RBI_BASE = "https://www.rbi.org.in"
RBI_LISTING_URL = f"{RBI_BASE}/Scripts/NotificationUser.aspx"
RBI_DETAIL_URL = f"{RBI_BASE}/Scripts/NotificationUser.aspx?Id={{id}}&Mode=0"

# rbidocs.rbi.org.in serves an HTML interstitial instead of the PDF to
# non-browser User-Agents, so a custom agent string silently yields 45 HTML
# pages. The %PDF magic-byte check in download() is what surfaces that.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

# RBI serves notification PDFs from a dedicated path. Every other PDF on the
# page is site furniture (annual reports, accessibility statements) and would
# silently poison the corpus.
_NOTIFICATION_PDF_RE = re.compile(r"/notification/pdfs/", re.IGNORECASE)
_CIRCULAR_NO_RE = re.compile(r"RBI/\d{4}-\d{2,4}/\d+")
_DATE_ROW_RE = re.compile(r"^[A-Z][a-z]{2}\s+\d{1,2},\s+\d{4}$")
_DETAIL_ID_RE = re.compile(r"NotificationUser\.aspx\?Id=(\d+)", re.IGNORECASE)
_SIZE_SUFFIX_RE = re.compile(r"\s*\(?\d+(?:\.\d+)?\s*[kKmM][bB]\)?\s*$")
_LONG_DATE_RE = re.compile(r"[A-Z][a-z]+\s+\d{1,2},\s+\d{4}")

Regulator = Literal["RBI", "SEBI"]


class CircularRef(BaseModel):
    regulator: Regulator
    title: str
    detail_url: str
    issued_on: date | None = None
    pdf_url: str | None = None
    circular_no: str | None = None


class DetailInfo(BaseModel):
    circular_no: str | None = None
    pdf_url: str | None = None
    title: str | None = None
    issued_on: date | None = None


class ManifestEntry(BaseModel):
    doc_id: str
    filename: str
    regulator: Regulator
    title: str
    circular_no: str | None = None
    issued_on: date | None = None
    source_url: str | None = None
    detail_url: str | None = None
    sha256: str
    bytes: int


def is_notification_pdf(url: str) -> bool:
    return bool(_NOTIFICATION_PDF_RE.search(url))


def slugify(text: str, max_len: int = 80) -> str:
    normalised = unicodedata.normalize("NFKD", text)
    ascii_only = normalised.encode("ascii", "ignore").decode()
    lowered = re.sub(r"[^a-z0-9]+", "-", ascii_only.lower())
    return lowered.strip("-")[:max_len].strip("-")


def _parse_listing_date(text: str) -> date | None:
    for fmt in ("%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(text.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _clean_title(text: str) -> str:
    return _SIZE_SUFFIX_RE.sub("", text).strip()


def parse_listing(html: str, base_url: str = RBI_LISTING_URL) -> list[CircularRef]:
    """Parse an RBI notification listing.

    The table alternates a bare date row with one or more title rows, so the
    most recently seen date is carried forward onto every title beneath it.
    """
    soup = BeautifulSoup(html, "lxml")
    refs: list[CircularRef] = []
    current_date: date | None = None

    for row in soup.find_all("tr"):
        row_text = row.get_text(" ", strip=True)
        if _DATE_ROW_RE.match(row_text):
            current_date = _parse_listing_date(row_text)
            continue

        detail_anchor = row.find("a", href=_DETAIL_ID_RE)
        if detail_anchor is None:
            continue

        title = _clean_title(detail_anchor.get_text(" ", strip=True))
        if not title:
            continue

        pdf_anchor = row.find(
            "a", href=lambda h: bool(h) and is_notification_pdf(h)
        )
        refs.append(
            CircularRef(
                regulator="RBI",
                title=title,
                detail_url=urljoin(base_url, detail_anchor["href"]),
                issued_on=current_date,
                pdf_url=pdf_anchor["href"] if pdf_anchor else None,
            )
        )
    return refs


def parse_detail(html: str) -> DetailInfo:
    """Pull circular number, title and the real notification PDF from a detail page.

    The title matters because circulars reached by walking sequential IDs have
    no listing row to take a title from.
    """
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(" ")
    number_match = _CIRCULAR_NO_RE.search(text)

    # The issue date is the first long-form date after the circular number;
    # later ones in the body cite previously issued circulars.
    issued_on: date | None = None
    if number_match:
        date_match = _LONG_DATE_RE.search(text, number_match.end())
        if date_match:
            issued_on = _parse_listing_date(date_match.group(0))

    title: str | None = None
    for node in soup.find_all(attrs={"class": ["head", "tableheader"]}):
        candidate = _clean_title(node.get_text(" ", strip=True))
        # ".tableheader" also carries the "( 228 kb )" size label.
        if len(candidate) > 15 and not candidate.startswith("("):
            title = candidate
            break
    pdf_url: str | None = None
    for match in re.finditer(r'href="([^"]+\.(?:pdf|PDF))"', html):
        candidate = match.group(1)
        if is_notification_pdf(candidate):
            pdf_url = urljoin(RBI_BASE, candidate)
            break
    return DetailInfo(
        circular_no=number_match.group(0) if number_match else None,
        pdf_url=pdf_url,
        title=title,
        issued_on=issued_on,
    )


async def _get(client: httpx.AsyncClient, url: str, *, attempts: int = 3) -> httpx.Response:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            response = await client.get(url)
            if response.status_code in (429, 500, 502, 503, 504):
                raise httpx.HTTPStatusError(
                    f"retryable {response.status_code}",
                    request=response.request,
                    response=response,
                )
            response.raise_for_status()
            return response
        except Exception as exc:  # noqa: BLE001 - retried below, re-raised at the end
            last = exc
            await asyncio.sleep(2**attempt + random.random())
    assert last is not None
    raise last


async def collect_refs(
    client: httpx.AsyncClient, limit: int, delay: float
) -> list[CircularRef]:
    """Listing first; walk sequential detail IDs backwards if it isn't enough."""
    listing = await _get(client, RBI_LISTING_URL)
    refs = parse_listing(listing.text, RBI_LISTING_URL)
    print(f"listing: {len(refs)} circulars", file=sys.stderr)

    if len(refs) >= limit:
        return refs[:limit]

    known_ids = [
        int(m.group(1)) for r in refs if (m := _DETAIL_ID_RE.search(r.detail_url))
    ]
    if not known_ids:
        return refs

    next_id = min(known_ids) - 1
    seen = set(known_ids)
    while len(refs) < limit and next_id > 0:
        if next_id in seen:
            next_id -= 1
            continue
        refs.append(
            CircularRef(
                regulator="RBI",
                title=f"RBI notification {next_id}",
                detail_url=RBI_DETAIL_URL.format(id=next_id),
            )
        )
        seen.add(next_id)
        next_id -= 1
    return refs[:limit]


async def enrich(
    client: httpx.AsyncClient, ref: CircularRef, delay: float
) -> CircularRef:
    await asyncio.sleep(delay)
    try:
        page = await _get(client, ref.detail_url)
    except Exception as exc:  # noqa: BLE001
        print(f"  detail failed {ref.detail_url}: {exc}", file=sys.stderr)
        return ref
    info = parse_detail(page.text)
    placeholder = ref.title.startswith("RBI notification ")
    return ref.model_copy(
        update={
            "circular_no": info.circular_no or ref.circular_no,
            "pdf_url": ref.pdf_url or info.pdf_url,
            "title": info.title if (placeholder and info.title) else ref.title,
            "issued_on": ref.issued_on or info.issued_on,
        }
    )


async def download(
    client: httpx.AsyncClient, ref: CircularRef, out_dir: Path, delay: float
) -> ManifestEntry | None:
    if not ref.pdf_url:
        return None
    await asyncio.sleep(delay)
    try:
        response = await _get(client, ref.pdf_url)
    except Exception as exc:  # noqa: BLE001
        print(f"  download failed {ref.pdf_url}: {exc}", file=sys.stderr)
        return None

    payload = response.content
    if not payload.startswith(b"%PDF"):
        print(f"  not a PDF, skipping {ref.pdf_url}", file=sys.stderr)
        return None

    digest = hashlib.sha256(payload).hexdigest()
    stem = slugify(ref.title) or digest[:12]
    filename = f"{stem}-{digest[:8]}.pdf"
    (out_dir / filename).write_bytes(payload)

    return ManifestEntry(
        doc_id=digest[:16],
        filename=filename,
        regulator=ref.regulator,
        title=ref.title,
        circular_no=ref.circular_no,
        issued_on=ref.issued_on,
        source_url=ref.pdf_url,
        detail_url=ref.detail_url,
        sha256=digest,
        bytes=len(payload),
    )


def rebuild_manifest(raw_dir: Path, manifest_path: Path) -> list[ManifestEntry]:
    """Re-derive the manifest from whatever PDFs survived curation.

    Metadata for previously-fetched files is preserved by sha256; manually
    dropped files (e.g. SEBI PDFs saved by hand) are added with the filename
    as a placeholder title for you to edit.
    """
    previous: dict[str, dict] = {}
    if manifest_path.exists():
        for row in json.loads(manifest_path.read_text(encoding="utf-8")):
            previous[row["sha256"]] = row

    entries: list[ManifestEntry] = []
    for pdf in sorted(raw_dir.glob("*.pdf")):
        payload = pdf.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest in previous:
            row = dict(previous[digest])
            row["filename"] = pdf.name
            entries.append(ManifestEntry.model_validate(row))
            continue
        entries.append(
            ManifestEntry(
                doc_id=digest[:16],
                filename=pdf.name,
                regulator="SEBI" if "sebi" in pdf.name.lower() else "RBI",
                title=pdf.stem.replace("-", " ").strip(),
                sha256=digest,
                bytes=len(payload),
            )
        )
    return entries


def write_manifest(entries: Iterable[ManifestEntry], path: Path) -> None:
    payload = [json.loads(e.model_dump_json()) for e in entries]
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


async def refresh_metadata(manifest_path: Path, delay: float) -> int:
    """Re-read detail pages to backfill title/circular_no/date without re-downloading."""
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    headers = {"User-Agent": USER_AGENT}
    updated = 0
    async with httpx.AsyncClient(
        headers=headers, timeout=60, follow_redirects=True
    ) as client:
        for row in rows:
            if not row.get("detail_url"):
                continue
            if row.get("issued_on") and row.get("circular_no"):
                continue
            await asyncio.sleep(delay)
            try:
                page = await _get(client, row["detail_url"])
            except Exception as exc:  # noqa: BLE001
                print(f"  failed {row['detail_url']}: {exc}", file=sys.stderr)
                continue
            info = parse_detail(page.text)
            before = (row.get("circular_no"), row.get("issued_on"), row.get("title"))
            row["circular_no"] = row.get("circular_no") or info.circular_no
            row["issued_on"] = row.get("issued_on") or (
                info.issued_on.isoformat() if info.issued_on else None
            )
            row["title"] = row.get("title") or info.title
            if (row.get("circular_no"), row.get("issued_on"), row.get("title")) != before:
                updated += 1
                print(f"  updated {row['filename'][:60]}")
    entries = [ManifestEntry.model_validate(r) for r in rows]
    write_manifest(entries, manifest_path)
    print(f"refreshed metadata on {updated}/{len(rows)} entries")
    return 0


async def run(args: argparse.Namespace) -> int:
    raw_dir = Path(args.out)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest)

    if args.refresh_metadata:
        return await refresh_metadata(manifest_path, args.delay)

    if args.rebuild_manifest:
        entries = rebuild_manifest(raw_dir, manifest_path)
        write_manifest(entries, manifest_path)
        print(f"manifest rebuilt from {len(entries)} PDFs in {raw_dir}")
        return 0

    headers = {"User-Agent": USER_AGENT}
    async with httpx.AsyncClient(
        headers=headers, timeout=60, follow_redirects=True
    ) as client:
        refs = await collect_refs(client, args.limit, args.delay)
        refs = [await enrich(client, r, args.delay) for r in refs]
        usable = [r for r in refs if r.pdf_url]
        print(f"{len(usable)}/{len(refs)} refs have a notification PDF", file=sys.stderr)

        if args.dry_run:
            for r in usable:
                stamp = r.issued_on.isoformat() if r.issued_on else "????-??-??"
                print(f"  {stamp}  {r.circular_no or '-':<20}  {r.title[:70]}")
            print(f"\ndry run: {len(usable)} would be downloaded into {raw_dir}")
            return 0

        entries: list[ManifestEntry] = []
        for ref in usable:
            entry = await download(client, ref, raw_dir, args.delay)
            if entry:
                entries.append(entry)
                print(f"  saved {entry.filename}")

    merged = rebuild_manifest(raw_dir, manifest_path) if manifest_path.exists() else entries
    if entries and manifest_path.exists():
        write_manifest(entries, manifest_path)
        merged = rebuild_manifest(raw_dir, manifest_path)
    write_manifest(merged, manifest_path)
    print(f"\ndownloaded {len(entries)} PDFs; manifest lists {len(merged)} -> {manifest_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=40, help="how many circulars to fetch")
    p.add_argument("--out", default="data/raw", help="directory for downloaded PDFs")
    p.add_argument("--manifest", default="data/manifest.json")
    p.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    p.add_argument("--dry-run", action="store_true", help="list without downloading")
    p.add_argument(
        "--rebuild-manifest",
        action="store_true",
        help="re-derive the manifest from the PDFs currently in --out (run after curating)",
    )
    p.add_argument(
        "--refresh-metadata",
        action="store_true",
        help="re-read detail pages to backfill missing title/circular number/date",
    )
    return p


def main() -> int:
    return asyncio.run(run(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
