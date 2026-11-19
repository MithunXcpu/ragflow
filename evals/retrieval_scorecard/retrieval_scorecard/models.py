"""Data records for one scorecard run."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    expected_chunks: tuple[str, ...]
    reference_answer: str


@dataclass(frozen=True)
class Config:
    name: str
    description: str
    chunk_chars: int
    top_k: int
    candidate_pool: int
    rerank: bool
    vector_similarity_weight: float
    similarity_threshold: float


@dataclass
class Chunk:
    chunk_ids: tuple[str, ...]
    document: str
    text: str
    tokens: list[str]
    embedding: list[float] = field(default_factory=list)

    @property
    def label(self) -> str:
        return "+".join(self.chunk_ids)


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int


@dataclass
class QuestionScore:
    question_id: str
    question: str
    expected_chunks: tuple[str, ...]
    retrieved_labels: tuple[str, ...]
    first_relevant_rank: int | None
    recall_at_k: float
    reciprocal_rank: float
    faithfulness: float
    answer: str
    judge: str


@dataclass
class ConfigScore:
    config: Config
    questions: list[QuestionScore]
    recall_at_k: float
    mrr: float
    faithfulness: float
    backend: str
