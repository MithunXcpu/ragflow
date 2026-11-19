"""Run the three configs and collect metrics."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from retrieval_scorecard.corpus import (
    atomic_sections,
    feature_root,
    load_configs,
    load_documents,
    load_questions,
)
from retrieval_scorecard.judge import embed_texts, judge_mock, judge_nvidia
from retrieval_scorecard.metrics import mean, recall_at_k, reciprocal_rank
from retrieval_scorecard.models import Config, ConfigScore, Hit, Question, QuestionScore
from retrieval_scorecard.ragflow import retrieve_ragflow
from retrieval_scorecard.rank import merge_sections, retrieve_local
from retrieval_scorecard.report import render_html
from retrieval_scorecard.textutil import hashed_embedding


@dataclass
class Settings:
    mode: str = "local"
    provider: str = "auto"
    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_model: str = "meta/llama-3.3-70b-instruct"
    nvidia_embed_model: str = "nvidia/nv-embedqa-e5-v5"
    ragflow_base_url: str = ""
    ragflow_api_key: str = ""
    ragflow_dataset_id: str = ""
    ragflow_rerank_id: str = ""

    @classmethod
    def from_env(cls, mode: str | None = None, provider: str | None = None) -> "Settings":
        return cls(
            mode=mode or os.environ.get("SCORECARD_MODE", "local"),
            provider=provider or os.environ.get("LLM_PROVIDER", "auto"),
            nvidia_api_key=os.environ.get("NVIDIA_API_KEY", ""),
            nvidia_base_url=os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            nvidia_model=os.environ.get("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct"),
            nvidia_embed_model=os.environ.get("NVIDIA_EMBED_MODEL", "nvidia/nv-embedqa-e5-v5"),
            ragflow_base_url=os.environ.get("RAGFLOW_BASE_URL", ""),
            ragflow_api_key=os.environ.get("RAGFLOW_API_KEY", ""),
            ragflow_dataset_id=os.environ.get("RAGFLOW_DATASET_ID", ""),
            ragflow_rerank_id=os.environ.get("RAGFLOW_RERANK_ID", ""),
        )


def load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass
class RunResult:
    config_scores: list[ConfigScore]
    mode: str
    embedding_backend: str
    notes: list[str]
    html: str

    def summary(self) -> dict:
        return {
            "mode": self.mode,
            "embedding_backend": self.embedding_backend,
            "notes": self.notes,
            "configs": [
                {
                    "name": item.config.name,
                    "chunk_chars": item.config.chunk_chars,
                    "top_k": item.config.top_k,
                    "rerank": item.config.rerank,
                    "recall_at_k": round(item.recall_at_k, 4),
                    "mrr": round(item.mrr, 4),
                    "faithfulness": round(item.faithfulness, 4),
                    "judge": item.questions[0].judge if item.questions else "",
                    "questions": [
                        {
                            "id": question.question_id,
                            "recall_at_k": round(question.recall_at_k, 4),
                            "mrr": round(question.reciprocal_rank, 4),
                            "first_relevant_rank": question.first_relevant_rank,
                            "faithfulness": round(question.faithfulness, 4),
                            "judge": question.judge,
                        }
                        for question in item.questions
                    ],
                }
                for item in self.config_scores
            ],
        }


def run_scorecard(
    settings: Settings,
    *,
    opener=None,
    generated_at: datetime | None = None,
    root: Path | None = None,
) -> RunResult:
    root = root or feature_root()
    questions = load_questions(root / "golden" / "questions.json")
    configs = load_configs(root / "configs.json")
    documents = load_documents(root / "corpus")
    sections = atomic_sections(documents)
    mode = _resolve_mode(settings)
    provider = _resolve_provider(settings)
    notes = [
        "Sample corpus: fictional Harborline field-service notes. Glossary sections repeat a question without the procedure value so a short result list can rank them above the procedure note. That is demo material.",
        "Expected sources are chunk_id markers in those notes, not live database ids.",
        "The mock judge scores whether reference values (numbers, codes) appear in the retrieved text. A glossary note can share the question wording and still fail that check.",
    ]
    embedding_backend = "mock-hash"
    embed = hashed_embedding
    if provider == "nvidia" and mode == "local":
        try:
            embedding_backend = f"nvidia:{settings.nvidia_embed_model}"
            embed = _nvidia_embedder(settings, sections, opener)
            notes.append(f"Embeddings from {settings.nvidia_base_url} model {settings.nvidia_embed_model}.")
        except Exception as exc:
            embedding_backend = "mock-hash"
            embed = hashed_embedding
            notes.append(f"NVIDIA embeddings failed ({exc}). Used hashed token vectors instead.")
    elif provider == "nvidia":
        notes.append(
            "Live retrieval uses the dataset's own embeddings. The faithfulness judge calls NVIDIA when the key is set."
        )
    else:
        notes.append(
            "Embeddings are hashed token vectors (mock). Set LLM_PROVIDER=nvidia and NVIDIA_API_KEY to call NVIDIA embeddings and the judge."
        )

    if mode == "live":
        notes.append(
            "Live mode calls POST /api/v1/retrieval. Chunk windows are a property of how the dataset was parsed; "
            "chunk_chars is recorded on the scorecard and is not sent as a retrieval parameter. "
            "top-k is page_size. rerank_id is sent only when RAGFLOW_RERANK_ID is set and the config turns rerank on."
        )
        if not settings.ragflow_rerank_id:
            notes.append("RAGFLOW_RERANK_ID is empty, so the rerank config still runs without a rerank_id on the live request.")
    else:
        notes.append(
            "Local fixture mode does not start RAGFlow. It chunks the sample pages, ranks them with lexical overlap plus the embedding vector, and optionally applies the density rerank."
        )

    config_scores: list[ConfigScore] = []
    judge_fallbacks = 0
    for config in configs:
        per_question: list[QuestionScore] = []
        chunks = merge_sections(sections, config.chunk_chars, embed=embed) if mode == "local" else []
        backend = _backend_label(mode, config, len(chunks), settings)
        for question in questions:
            hits = _retrieve(mode, question, config, chunks, settings, opener)
            answer, faith, judge_name, fell_back = _judge(provider, question, hits, settings, opener)
            if fell_back:
                judge_fallbacks += 1
            recall = recall_at_k(question.expected_chunks, hits)
            rr, first = reciprocal_rank(question.expected_chunks, hits)
            per_question.append(
                QuestionScore(
                    question_id=question.id,
                    question=question.question,
                    expected_chunks=question.expected_chunks,
                    retrieved_labels=tuple(hit.chunk.label for hit in hits),
                    first_relevant_rank=first,
                    recall_at_k=recall,
                    reciprocal_rank=rr,
                    faithfulness=faith,
                    answer=answer,
                    judge=judge_name,
                )
            )
        config_scores.append(
            ConfigScore(
                config=config,
                questions=per_question,
                recall_at_k=mean([item.recall_at_k for item in per_question]),
                mrr=mean([item.reciprocal_rank for item in per_question]),
                faithfulness=mean([item.faithfulness for item in per_question]),
                backend=backend,
            )
        )
    if judge_fallbacks:
        notes.append(f"NVIDIA judge fell back to the mock judge on {judge_fallbacks} question-config rows.")
    html = render_html(
        config_scores,
        mode=mode,
        embedding_backend=embedding_backend,
        generated_at=generated_at or datetime.now(timezone.utc),
        notes=notes,
    )
    return RunResult(
        config_scores=config_scores,
        mode=mode,
        embedding_backend=embedding_backend,
        notes=notes,
        html=html,
    )


def write_result(result: RunResult, html_path: Path) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(result.html, encoding="utf-8")
    json_path = html_path.with_suffix(".json")
    json_path.write_text(json.dumps(result.summary(), indent=2) + "\n", encoding="utf-8")


def _resolve_mode(settings: Settings) -> str:
    mode = settings.mode.strip().lower()
    if mode == "auto":
        if settings.ragflow_base_url and settings.ragflow_api_key and settings.ragflow_dataset_id:
            return "live"
        return "local"
    if mode not in {"local", "live"}:
        raise SystemExit(f"unknown mode {settings.mode!r}; use local, live, or auto")
    if mode == "live":
        missing = [
            name
            for name, value in (
                ("RAGFLOW_BASE_URL", settings.ragflow_base_url),
                ("RAGFLOW_API_KEY", settings.ragflow_api_key),
                ("RAGFLOW_DATASET_ID", settings.ragflow_dataset_id),
            )
            if not value
        ]
        if missing:
            raise SystemExit("live mode needs " + ", ".join(missing))
    return mode


def _resolve_provider(settings: Settings) -> str:
    provider = settings.provider.strip().lower()
    if provider == "auto":
        return "nvidia" if settings.nvidia_api_key else "mock"
    if provider not in {"mock", "nvidia"}:
        raise SystemExit(f"unknown LLM_PROVIDER {settings.provider!r}; use mock, nvidia, or auto")
    if provider == "nvidia" and not settings.nvidia_api_key:
        return "mock"
    return provider


def _nvidia_embedder(settings: Settings, sections, opener):
    texts = [section[2] for section in sections]
    vectors = embed_texts(
        texts,
        api_key=settings.nvidia_api_key,
        base_url=settings.nvidia_base_url,
        model=settings.nvidia_embed_model,
        opener=opener,
    )
    lookup = {text: vector for text, vector in zip(texts, vectors)}

    def embed(text: str) -> list[float]:
        if text in lookup:
            return lookup[text]
        pieces = [lookup[key] for key in lookup if key and key in text]
        if not pieces:
            return hashed_embedding(text)
        width = len(pieces[0])
        averaged = [0.0] * width
        for vector in pieces:
            for index, value in enumerate(vector):
                averaged[index] += value
        total = float(len(pieces))
        return [value / total for value in averaged]

    return embed


def _retrieve(mode, question: Question, config: Config, chunks, settings: Settings, opener) -> list[Hit]:
    if mode == "live":
        return retrieve_ragflow(
            question.question,
            settings.ragflow_dataset_id,
            config,
            base_url=settings.ragflow_base_url,
            api_key=settings.ragflow_api_key,
            rerank_id=settings.ragflow_rerank_id,
            opener=opener,
        )
    return retrieve_local(question.question, chunks, config)


def _judge(provider, question: Question, hits, settings: Settings, opener):
    if provider == "nvidia":
        try:
            answer, score, name = judge_nvidia(
                question.question,
                hits,
                question.reference_answer,
                api_key=settings.nvidia_api_key,
                base_url=settings.nvidia_base_url,
                model=settings.nvidia_model,
                opener=opener,
            )
            return answer, score, name, False
        except Exception:
            answer, score, name = judge_mock(question.question, hits, question.reference_answer)
            return answer, score, name, True
    answer, score, name = judge_mock(question.question, hits, question.reference_answer)
    return answer, score, name, False


def _backend_label(mode: str, config: Config, chunk_count: int, settings: Settings) -> str:
    if mode == "live":
        rerank = settings.ragflow_rerank_id if config.rerank and settings.ragflow_rerank_id else "not sent"
        return f"POST /api/v1/retrieval page_size={config.top_k} rerank_id={rerank}"
    rerank = "density rerank" if config.rerank else "no rerank"
    return f"local fixture index, {chunk_count} windows, {rerank}"
