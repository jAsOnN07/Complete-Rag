"""Gold evaluation set: schema, loading and validation.

Design choice: expected answers are keyed by document and by verbatim evidence
quotes, never by chunk id. Chunk ids embed the chunking strategy, so a
chunk-keyed gold set would silently break the moment two strategies are
compared - which is the whole point of having an eval. A retrieved chunk "hits"
when it comes from an expected document and contains one of the evidence
quotes; that is strategy-invariant and every quote is machine-checkable
against the corpus.
"""

from __future__ import annotations

import json
import re
from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.models import ScoredChunk

GOLD_PATH = Path("evaluation/gold_set.json")


class QuestionType(StrEnum):
    SINGLE_HOP = "single_hop"
    MULTI_HOP = "multi_hop"
    CROSS_CIRCULAR = "cross_circular"
    NEGATIVE = "negative"


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


def normalise(text: str) -> str:
    """Whitespace/quote-insensitive form used for evidence matching."""
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip().lower()


class GoldQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    question: str = Field(min_length=10)
    reference: str = Field(min_length=1)
    question_type: QuestionType
    difficulty: Difficulty = Difficulty.MEDIUM
    expected_doc_ids: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    generated_by: str = "human"
    verified_by: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _shape_matches_type(self) -> Self:
        if self.question_type is QuestionType.NEGATIVE:
            if self.expected_doc_ids or self.evidence:
                raise ValueError(f"{self.id}: negative questions carry no expected docs")
        else:
            if not self.expected_doc_ids:
                raise ValueError(f"{self.id}: non-negative questions need expected_doc_ids")
        if self.question_type in (QuestionType.MULTI_HOP, QuestionType.CROSS_CIRCULAR):
            if len(self.expected_doc_ids) < 2:
                raise ValueError(f"{self.id}: {self.question_type} needs >= 2 expected docs")
        return self

    @property
    def is_negative(self) -> bool:
        return self.question_type is QuestionType.NEGATIVE

    def chunk_hits(self, scored: ScoredChunk) -> bool:
        """Does this retrieved chunk count as a relevant hit for the question?"""
        if scored.chunk.doc_id not in self.expected_doc_ids:
            return False
        if not self.evidence:
            return True
        body = normalise(scored.chunk.text)
        return any(normalise(q) in body for q in self.evidence)

    def doc_hits(self, scored: ScoredChunk) -> bool:
        return scored.chunk.doc_id in self.expected_doc_ids


class GoldSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    description: str = ""
    questions: list[GoldQuestion]

    @model_validator(mode="after")
    def _ids_unique(self) -> Self:
        ids = [q.id for q in self.questions]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate question ids: {sorted(dupes)}")
        return self

    @property
    def negatives(self) -> list[GoldQuestion]:
        return [q for q in self.questions if q.is_negative]

    @property
    def positives(self) -> list[GoldQuestion]:
        return [q for q in self.questions if not q.is_negative]

    def by_type(self) -> dict[str, list[GoldQuestion]]:
        out: dict[str, list[GoldQuestion]] = {}
        for q in self.questions:
            out.setdefault(q.question_type.value, []).append(q)
        return out

    def unverified(self) -> list[GoldQuestion]:
        return [q for q in self.questions if q.verified_by is None]


def load_gold(path: Path = GOLD_PATH) -> GoldSet:
    return GoldSet.model_validate(json.loads(path.read_text(encoding="utf-8")))


def save_gold(gold: GoldSet, path: Path = GOLD_PATH) -> None:
    path.write_text(
        json.dumps(gold.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


class EvidenceProblem(BaseModel):
    question_id: str
    doc_id: str | None
    quote: str
    reason: str


def validate_evidence(gold: GoldSet, doc_texts: dict[str, str]) -> list[EvidenceProblem]:
    """Check every expected doc exists and every evidence quote appears in it.

    A gold set whose quotes drift from the corpus produces recall numbers that
    are quietly wrong; this makes that a hard failure at load time instead.
    """
    problems: list[EvidenceProblem] = []
    normalised = {doc_id: normalise(t) for doc_id, t in doc_texts.items()}
    for q in gold.questions:
        for doc_id in q.expected_doc_ids:
            if doc_id not in normalised:
                problems.append(
                    EvidenceProblem(
                        question_id=q.id, doc_id=doc_id, quote="",
                        reason="expected_doc_id not in corpus",
                    )
                )
        for quote in q.evidence:
            target = normalise(quote)
            if not any(target in normalised.get(d, "") for d in q.expected_doc_ids):
                problems.append(
                    EvidenceProblem(
                        question_id=q.id, doc_id=None, quote=quote,
                        reason="evidence quote not found in any expected doc",
                    )
                )
    return problems


def corpus_texts(
    manifest: Path = Path("data/manifest.json"), raw_dir: Path = Path("data/raw")
) -> dict[str, str]:
    """Cleaned text per doc_id, through the same loader the index uses, so an
    evidence quote that validates here is a quote a chunk can actually contain."""
    from ingestion.loader import PdfLoader

    loader = PdfLoader()
    texts: dict[str, str] = {}
    for row in json.loads(manifest.read_text(encoding="utf-8")):
        path = raw_dir / row["filename"]
        if path.exists():
            doc = loader.load_sync(path, row)
            texts[row["doc_id"]] = "\n".join(p.text for p in doc.pages)
    return texts


def check_gold(path: Path = GOLD_PATH) -> list[EvidenceProblem]:
    return validate_evidence(load_gold(path), corpus_texts())


def main() -> int:
    """python -m evaluation.gold  -> exits non-zero if any quote has drifted."""
    import sys

    gold = load_gold()
    problems = validate_evidence(gold, corpus_texts())
    unverified = [q.id for q in gold.questions if q.verified_by is None]
    print(f"{len(gold.questions)} questions, {len(unverified)} unverified"
          + (f": {', '.join(unverified)}" if unverified else ""))
    for p in problems:
        print(f"  {p.question_id}: {p.reason} {p.quote!r}", file=sys.stderr)
    print("evidence: " + ("all quotes found in the corpus" if not problems else f"{len(problems)} problem(s)"))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
