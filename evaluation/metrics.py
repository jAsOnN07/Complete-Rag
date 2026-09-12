"""Pure retrieval metrics. No I/O, no LLM - runs on every change for free.

Each function takes a relevance vector: ``rel[i]`` is True if the item at
rank ``i`` (0-indexed) is relevant. ``total_relevant`` is how many relevant
items exist in the corpus for the question, which recall needs and the
retrieved list alone cannot tell you.
"""

from __future__ import annotations

import math
from typing import Sequence


def hit_at_k(rel: Sequence[bool], k: int) -> float:
    return 1.0 if any(rel[:k]) else 0.0


def recall_at_k(rel: Sequence[bool], k: int, total_relevant: int) -> float:
    if total_relevant <= 0:
        return 0.0
    return min(sum(1 for r in rel[:k] if r), total_relevant) / total_relevant


def precision_at_k(rel: Sequence[bool], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(1 for r in rel[:k] if r) / k


def reciprocal_rank(rel: Sequence[bool]) -> float:
    for i, r in enumerate(rel):
        if r:
            return 1.0 / (i + 1)
    return 0.0


def dcg_at_k(rel: Sequence[bool], k: int) -> float:
    return sum(1.0 / math.log2(i + 2) for i, r in enumerate(rel[:k]) if r)


def ndcg_at_k(rel: Sequence[bool], k: int, total_relevant: int) -> float:
    ideal_hits = min(total_relevant, k)
    if ideal_hits <= 0:
        return 0.0
    ideal = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg_at_k(rel, k) / ideal


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
