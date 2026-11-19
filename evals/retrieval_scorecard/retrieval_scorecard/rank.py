"""Local fixture index: chunk windows, hashed embeddings, optional density rerank."""

from __future__ import annotations

from retrieval_scorecard.models import Chunk, Config, Hit
from retrieval_scorecard.textutil import cosine, hashed_embedding, idf_table, tokenize


def merge_sections(
    sections: list[tuple[str, str, str]],
    chunk_chars: int,
    *,
    embed,
) -> list[Chunk]:
    """Pack same-document sections until adding another would pass chunk_chars.

    A section that is already longer than chunk_chars stays intact. That keeps
    every golden marker available while still letting the window size change
    which markers share a hit.
    """
    grouped: list[list[tuple[str, str, str]]] = []
    bucket: list[tuple[str, str, str]] = []
    bucket_chars = 0
    bucket_doc = ""

    def flush() -> None:
        nonlocal bucket, bucket_chars, bucket_doc
        if bucket:
            grouped.append(bucket)
        bucket = []
        bucket_chars = 0
        bucket_doc = ""

    for section in sections:
        _chunk_id, document, text = section
        if bucket and (document != bucket_doc or bucket_chars + len(text) > chunk_chars):
            flush()
        bucket.append(section)
        bucket_doc = document
        bucket_chars += len(text)
        if bucket_chars >= chunk_chars:
            flush()
    flush()

    chunks: list[Chunk] = []
    for group in grouped:
        text = "\n\n".join(part[2] for part in group)
        chunks.append(
            Chunk(
                chunk_ids=tuple(part[0] for part in group),
                document=group[0][1],
                text=text,
                tokens=tokenize(text),
                embedding=embed(text),
            )
        )
    return chunks


def _coverage(query: list[str], doc_tokens: list[str], idf: dict[str, float]) -> float:
    if not query:
        return 0.0
    present = set(doc_tokens)
    weight = sum(idf.get(token, 1.0) for token in query)
    if weight == 0:
        return 0.0
    return sum(idf.get(token, 1.0) for token in query if token in present) / weight


def _tf_bonus(query: list[str], doc_tokens: list[str], idf: dict[str, float]) -> float:
    if not query:
        return 0.0
    counts: dict[str, int] = {}
    for token in doc_tokens:
        counts[token] = counts.get(token, 0) + 1
    weight = sum(idf.get(token, 1.0) for token in query)
    if weight == 0:
        return 0.0
    scored = 0.0
    for token in query:
        scored += (min(counts.get(token, 0), 3) / 3.0) * idf.get(token, 1.0)
    return scored / weight


def _density(query: list[str], doc_tokens: list[str], idf: dict[str, float]) -> float:
    if not doc_tokens or not query:
        return 0.0
    present = set(query)
    hit = sum(idf.get(token, 1.0) for token in doc_tokens if token in present)
    total = sum(idf.get(token, 1.0) for token in doc_tokens)
    if total == 0:
        return 0.0
    return hit / total


def lexical_score(query: list[str], chunk: Chunk, idf: dict[str, float], *, rerank: bool) -> float:
    coverage = _coverage(query, chunk.tokens, idf)
    if rerank:
        return 0.40 * coverage + 0.60 * _density(query, chunk.tokens, idf)
    length_prior = 0.12 * min(len(chunk.tokens), 140) / 140
    return 0.50 * coverage + 0.38 * _tf_bonus(query, chunk.tokens, idf) + length_prior


def retrieve_local(
    question: str,
    chunks: list[Chunk],
    config: Config,
    *,
    query_embedding: list[float] | None = None,
) -> list[Hit]:
    query_tokens = tokenize(question)
    idf = idf_table([chunk.tokens for chunk in chunks] + [query_tokens])
    qvec = query_embedding if query_embedding is not None else hashed_embedding(question)
    weight = min(max(config.vector_similarity_weight, 0.0), 1.0)
    scored: list[tuple[float, int, Chunk]] = []
    for index, chunk in enumerate(chunks):
        lexical = lexical_score(query_tokens, chunk, idf, rerank=False)
        vector = cosine(qvec, chunk.embedding)
        score = weight * vector + (1.0 - weight) * lexical
        if score < config.similarity_threshold:
            continue
        scored.append((score, index, chunk))
    scored.sort(key=lambda item: (-item[0], item[1]))
    pool = scored[: config.candidate_pool]
    if config.rerank:
        reranked: list[tuple[float, int, Chunk]] = []
        for _, index, chunk in pool:
            lexical = lexical_score(query_tokens, chunk, idf, rerank=True)
            vector = cosine(qvec, chunk.embedding)
            reranked.append((weight * vector + (1.0 - weight) * lexical, index, chunk))
        reranked.sort(key=lambda item: (-item[0], item[1]))
        pool = reranked
    hits = []
    for rank, (score, _, chunk) in enumerate(pool[: config.top_k], start=1):
        hits.append(Hit(chunk=chunk, score=score, rank=rank))
    return hits
