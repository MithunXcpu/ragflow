"""recall@k and mean reciprocal rank."""

from __future__ import annotations

from retrieval_scorecard.models import Hit


def relevant_ranks(expected: tuple[str, ...], hits: list[Hit]) -> list[int]:
    wanted = set(expected)
    ranks = []
    for hit in hits:
        if wanted.intersection(hit.chunk.chunk_ids):
            ranks.append(hit.rank)
    return ranks


def recall_at_k(expected: tuple[str, ...], hits: list[Hit]) -> float:
    if not expected:
        return 0.0
    found = set()
    wanted = set(expected)
    for hit in hits:
        found.update(wanted.intersection(hit.chunk.chunk_ids))
    return len(found) / len(wanted)


def reciprocal_rank(expected: tuple[str, ...], hits: list[Hit]) -> tuple[float, int | None]:
    ranks = relevant_ranks(expected, hits)
    if not ranks:
        return 0.0, None
    first = min(ranks)
    return 1.0 / first, first


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
