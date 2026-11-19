"""Faithfulness: NVIDIA chat completions, with an offline mock judge."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

from retrieval_scorecard.models import Hit
from retrieval_scorecard.textutil import fact_tokens, tokenize

JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def draft_mock_answer(question: str, hits: list[Hit], reference: str) -> str:
    """Extract a sentence from the hits, or fall back to the reference answer.

    The fallback stands in for a model that answers from memory when the
    retrieved text never states the fact. Those extra tokens are then scored
    as unsupported.
    """
    query = set(tokenize(question))
    facts = fact_tokens(reference)
    best_sentence = ""
    best_overlap = 0.0
    best_has_fact = False
    for hit in hits:
        for sentence in _sentences(hit.chunk.text):
            tokens = set(tokenize(sentence))
            if not query:
                continue
            overlap = len(query & tokens) / len(query)
            has_fact = not facts or bool(facts & tokens)
            better = overlap > best_overlap + 1e-9 or (
                abs(overlap - best_overlap) <= 1e-9 and has_fact and not best_has_fact
            )
            if better:
                best_overlap = overlap
                best_sentence = sentence.strip()
                best_has_fact = has_fact
    if best_sentence and best_overlap >= 0.34 and best_has_fact:
        return best_sentence
    if best_sentence and best_overlap >= 0.34:
        return f"{best_sentence} {reference}".strip()
    return reference


def mock_faithfulness(answer: str, hits: list[Hit]) -> float:
    """Score support for specific values first, then ordinary tokens.

    Numbers and codes are the tokens a procedure note is actually for.
    A glossary hit can share the question wording and still miss every value.
    """
    context = set()
    for hit in hits:
        context.update(tokenize(hit.chunk.text))
    facts = list(fact_tokens(answer))
    if len(facts) >= 2:
        return sum(1 for token in facts if token in context) / len(facts)
    answer_tokens = [token for token in tokenize(answer) if len(token) > 2]
    if not answer_tokens:
        return 1.0
    return sum(1 for token in answer_tokens if token in context) / len(answer_tokens)


def judge_mock(question: str, hits: list[Hit], reference: str) -> tuple[str, float, str]:
    answer = draft_mock_answer(question, hits, reference)
    return answer, mock_faithfulness(answer, hits), "mock"


def judge_nvidia(
    question: str,
    hits: list[Hit],
    reference: str,
    *,
    api_key: str,
    base_url: str,
    model: str,
    opener=urllib.request.urlopen,
    timeout: float = 45.0,
) -> tuple[str, float, str]:
    context = "\n\n".join(f"[{hit.rank}] {hit.chunk.text}" for hit in hits) or "(no chunks retrieved)"
    prompt = (
        "You judge whether an answer stays inside the retrieved context.\n"
        "Write a short answer using only the retrieved context. "
        "If the context does not support an answer, set answer to NOT_IN_CONTEXT "
        "and do not add facts.\n"
        "faithfulness is the fraction of factual claims in the answer that the "
        "retrieved context supports, from 0 to 1. NOT_IN_CONTEXT scores 1.\n"
        "The reference answer is a hint for what the question is asking. "
        "It is not evidence. Do not copy it unless the context supports it.\n"
        'Reply with JSON only: {"answer": "...", "faithfulness": 0.0, "reason": "..."}\n\n'
        f"Question: {question}\n"
        f"Reference answer (not evidence): {reference}\n"
        f"Retrieved context:\n{context}"
    )
    raw = chat_completion(
        prompt,
        api_key=api_key,
        base_url=base_url,
        model=model,
        opener=opener,
        timeout=timeout,
    )
    parsed = _parse_judge_payload(raw)
    score = min(max(float(parsed["faithfulness"]), 0.0), 1.0)
    answer = str(parsed.get("answer", "")).strip() or "NOT_IN_CONTEXT"
    return answer, score, f"nvidia:{model}"


def chat_completion(
    prompt: str,
    *,
    api_key: str,
    base_url: str,
    model: str,
    opener=urllib.request.urlopen,
    timeout: float = 45.0,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "You reply with JSON only."},
            {"role": "user", "content": prompt},
        ],
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with opener(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"NVIDIA chat HTTP {exc.code}: {detail}") from exc
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError("NVIDIA chat response had no choices")
    return choices[0].get("message", {}).get("content", "")


def embed_texts(
    texts: list[str],
    *,
    api_key: str,
    base_url: str,
    model: str,
    opener=urllib.request.urlopen,
    timeout: float = 45.0,
) -> list[list[float]]:
    url = base_url.rstrip("/") + "/embeddings"
    body = {"model": model, "input": texts}
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with opener(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"NVIDIA embeddings HTTP {exc.code}: {detail}") from exc
    rows = payload.get("data") or []
    ordered = sorted(rows, key=lambda row: row.get("index", 0))
    vectors = [row.get("embedding") for row in ordered]
    if len(vectors) != len(texts) or any(not isinstance(vector, list) for vector in vectors):
        raise RuntimeError("NVIDIA embeddings response did not include one vector per input")
    return vectors


def _parse_judge_payload(raw: str) -> dict:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        match = JSON_OBJECT.search(raw or "")
        if not match:
            raise RuntimeError("NVIDIA judge did not return JSON")
        payload = json.loads(match.group(0))
    if "faithfulness" not in payload:
        raise RuntimeError("NVIDIA judge JSON missing faithfulness")
    return payload


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part for part in parts if part.strip()]
