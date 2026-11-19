"""POST /api/v1/retrieval client for a live RAGFlow dataset."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from retrieval_scorecard.models import Chunk, Config, Hit
from retrieval_scorecard.textutil import marker_ids, tokenize


class RagflowError(RuntimeError):
    pass


def retrieval_body(question: str, dataset_id: str, config: Config, rerank_id: str) -> dict:
    page_size = config.top_k
    body = {
        "dataset_ids": [dataset_id],
        "question": question,
        "page": 1,
        "page_size": page_size,
        "top_k": max(config.candidate_pool, page_size),
        "similarity_threshold": config.similarity_threshold,
        "vector_similarity_weight": config.vector_similarity_weight,
        "keyword": True,
        "include_knowledge_compilation": False,
    }
    if config.rerank and rerank_id:
        body["rerank_id"] = rerank_id
    return body


def retrieve_ragflow(
    question: str,
    dataset_id: str,
    config: Config,
    *,
    base_url: str,
    api_key: str,
    rerank_id: str,
    opener=urllib.request.urlopen,
    timeout: float = 30.0,
) -> list[Hit]:
    url = base_url.rstrip("/") + "/api/v1/retrieval"
    request = urllib.request.Request(
        url,
        data=json.dumps(retrieval_body(question, dataset_id, config, rerank_id)).encode("utf-8"),
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
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        raise RagflowError(f"RAGFlow retrieval HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RagflowError(f"RAGFlow retrieval failed: {exc.reason}") from exc
    code = payload.get("code", 0)
    if code not in (0, None):
        raise RagflowError(f"RAGFlow retrieval code {code}: {payload.get('message', '')}")
    data = payload.get("data") or {}
    chunks = data.get("chunks") or []
    hits: list[Hit] = []
    for index, raw in enumerate(chunks, start=1):
        content = str(raw.get("content") or raw.get("content_with_weight") or "")
        document = str(raw.get("document_keyword") or raw.get("document_name") or raw.get("docnm_kwd") or "")
        found = tuple(marker_ids(content))
        if not found:
            fallback = str(raw.get("id") or f"hit-{index}")
            found = (fallback,)
        chunk = Chunk(
            chunk_ids=found,
            document=document,
            text=content,
            tokens=tokenize(content),
        )
        score = float(raw.get("similarity") or raw.get("score") or 0.0)
        hits.append(Hit(chunk=chunk, score=score, rank=index))
        if len(hits) >= config.top_k:
            break
    return hits
