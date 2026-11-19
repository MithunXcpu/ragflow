"""Load the sample corpus, golden questions, and retrieval configs."""

from __future__ import annotations

import json
from pathlib import Path

from retrieval_scorecard.models import Config, Question
from retrieval_scorecard.textutil import marker_ids

ROOT = Path(__file__).resolve().parents[1]


def feature_root() -> Path:
    return ROOT


def load_questions(path: Path | None = None) -> list[Question]:
    payload = json.loads((path or ROOT / "golden" / "questions.json").read_text(encoding="utf-8"))
    questions = []
    for item in payload["questions"]:
        expected = tuple(chunk.lower() for chunk in item["expected_chunks"])
        if not expected:
            raise ValueError(f"{item['id']} has no expected chunks")
        questions.append(
            Question(
                id=item["id"],
                question=item["question"].strip(),
                expected_chunks=expected,
                reference_answer=item["reference_answer"].strip(),
            )
        )
    return questions


def load_configs(path: Path | None = None) -> list[Config]:
    payload = json.loads((path or ROOT / "configs.json").read_text(encoding="utf-8"))
    configs = []
    for item in payload:
        top_k = int(item["top_k"])
        pool = int(item.get("candidate_pool", top_k))
        if pool < top_k:
            raise ValueError(f"{item['name']}: candidate_pool must be >= top_k")
        configs.append(
            Config(
                name=item["name"],
                description=item.get("description", ""),
                chunk_chars=int(item["chunk_chars"]),
                top_k=top_k,
                candidate_pool=pool,
                rerank=bool(item["rerank"]),
                vector_similarity_weight=float(item["vector_similarity_weight"]),
                similarity_threshold=float(item.get("similarity_threshold", 0.0)),
            )
        )
    return configs


def load_documents(corpus_dir: Path | None = None) -> list[tuple[str, str]]:
    directory = corpus_dir or ROOT / "corpus"
    documents = []
    for path in sorted(directory.glob("*.md")):
        documents.append((path.name, path.read_text(encoding="utf-8")))
    if not documents:
        raise FileNotFoundError(f"no markdown corpus files in {directory}")
    return documents


def atomic_sections(documents: list[tuple[str, str]]) -> list[tuple[str, str, str]]:
    """Return (chunk_id, document_name, section_text) in file order."""
    sections: list[tuple[str, str, str]] = []
    for name, text in documents:
        current_id: str | None = None
        buffer: list[str] = []

        def flush() -> None:
            nonlocal current_id, buffer
            if current_id is None:
                buffer = []
                return
            body = "\n".join(buffer).strip()
            sections.append((current_id, name, f"chunk_id: {current_id}\n{body}"))
            current_id = None
            buffer = []

        for line in text.splitlines():
            found = marker_ids(line)
            if found and line.strip().lower().startswith("chunk_id:"):
                flush()
                current_id = found[0]
                continue
            if current_id is not None:
                buffer.append(line)
        flush()
    if not sections:
        raise ValueError("corpus has no chunk_id markers")
    return sections
