"""Token helpers shared by ranking, judging, and chunk matching."""

from __future__ import annotations

import math
import re
import zlib

MARKER_RE = re.compile(r"chunk_id:\s*([a-z0-9][a-z0-9-]*)", re.IGNORECASE)
TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
    "does",
    "do",
    "should",
    "can",
    "may",
    "into",
    "after",
    "before",
    "about",
    "than",
    "then",
    "them",
    "they",
    "their",
    "its",
    "if",
    "we",
    "you",
    "your",
}


def normalize(text: str) -> str:
    return text.lower().replace("n·m", "nm").replace("n.m", "nm")


def stem(token: str) -> str:
    if any(ch.isdigit() for ch in token) or "-" in token:
        return token
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 5 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    tokens = []
    for raw in TOKEN_RE.findall(normalize(text)):
        token = stem(raw)
        if token in STOPWORDS or len(token) < 2:
            continue
        tokens.append(token)
    return tokens


def marker_ids(text: str) -> list[str]:
    seen: list[str] = []
    for match in MARKER_RE.findall(text or ""):
        marker = match.lower()
        if marker not in seen:
            seen.append(marker)
    return seen


def fact_tokens(text: str) -> set[str]:
    """Numbers and codes, such as 25, 4ma, CP-440, or try-start."""
    facts = set()
    for token in tokenize(text):
        if any(ch.isdigit() for ch in token) or "-" in token:
            facts.add(token)
    return facts


def hashed_embedding(text: str, dims: int = 256) -> list[float]:
    """Bag-of-tokens vector. Deterministic stand-in for an embedding API."""
    vector = [0.0] * dims
    for token in tokenize(text):
        digest = zlib.crc32(token.encode("utf-8")) & 0xFFFFFFFF
        vector[digest % dims] += 1.0
        vector[(digest >> 8) % dims] += 0.35
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def idf_table(chunks: list[list[str]]) -> dict[str, float]:
    doc_freq: dict[str, int] = {}
    for tokens in chunks:
        for token in set(tokens):
            doc_freq[token] = doc_freq.get(token, 0) + 1
    total = max(len(chunks), 1)
    return {
        token: math.log((total + 1) / (freq + 1)) + 1.0
        for token, freq in doc_freq.items()
    }
