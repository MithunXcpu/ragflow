"""Tests for the retrieval scorecard. No network and no RAGFlow process."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval_scorecard.corpus import atomic_sections, load_configs, load_documents, load_questions
from retrieval_scorecard.judge import judge_nvidia
from retrieval_scorecard.metrics import recall_at_k, reciprocal_rank
from retrieval_scorecard.models import Chunk, Hit
from retrieval_scorecard.ragflow import retrieval_body, retrieve_ragflow
from retrieval_scorecard.rank import merge_sections
from retrieval_scorecard.runner import Settings, run_scorecard, write_result
from retrieval_scorecard.textutil import hashed_embedding, marker_ids


def _chunk(ids: tuple[str, ...]) -> Chunk:
    return Chunk(chunk_ids=ids, document="doc.md", text="chunk_id: " + ids[0], tokens=["x"])


class MetricTests(unittest.TestCase):
    def test_recall_and_mrr(self):
        hits = [
            Hit(_chunk(("other",)), 0.9, 1),
            Hit(_chunk(("gold", "extra")), 0.4, 2),
        ]
        self.assertEqual(recall_at_k(("gold", "missing"), hits), 0.5)
        self.assertEqual(reciprocal_rank(("gold",), hits), (0.5, 2))
        self.assertEqual(reciprocal_rank(("absent",), hits), (0.0, None))

    def test_marker_parse(self):
        self.assertEqual(marker_ids("chunk_id: Pump-Torque\nmore"), ["pump-torque"])


class CorpusTests(unittest.TestCase):
    def test_golden_set_matches_corpus_markers(self):
        questions = load_questions()
        self.assertGreaterEqual(len(questions), 20)
        present = {section[0] for section in atomic_sections(load_documents())}
        for question in questions:
            self.assertTrue(question.expected_chunks)
            for chunk_id in question.expected_chunks:
                self.assertIn(chunk_id, present)

    def test_merge_keeps_documents_apart(self):
        sections = atomic_sections(load_documents())
        tiny = merge_sections(sections, 1, embed=hashed_embedding)
        wide = merge_sections(sections, 100000, embed=hashed_embedding)
        self.assertEqual(len(tiny), len(sections))
        self.assertEqual(len(wide), len(load_documents()))
        configs = {config.name: config for config in load_configs()}
        self.assertEqual(configs["narrow-top1"].top_k, 1)
        self.assertTrue(configs["rerank-top1"].rerank)
        self.assertGreater(configs["wide-top4"].chunk_chars, configs["narrow-top1"].chunk_chars)


class _Body:
    def __init__(self, payload: dict):
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class ApiTests(unittest.TestCase):
    def test_live_request_shape(self):
        configs = {config.name: config for config in load_configs()}
        off = retrieval_body("How tight?", "ds-1", configs["narrow-top1"], "")
        self.assertEqual(off["dataset_ids"], ["ds-1"])
        self.assertEqual(off["page_size"], 1)
        self.assertNotIn("rerank_id", off)
        on = retrieval_body("How tight?", "ds-1", configs["rerank-top1"], "rerank-model")
        self.assertEqual(on["rerank_id"], "rerank-model")
        self.assertGreaterEqual(on["top_k"], on["page_size"])

    def test_live_response_uses_chunk_markers(self):
        captured = {}

        def opener(request, timeout=0):
            captured["url"] = request.full_url
            captured["auth"] = request.get_header("Authorization")
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return _Body(
                {
                    "code": 0,
                    "data": {
                        "chunks": [
                            {
                                "id": "live-chunk",
                                "content": "chunk_id: pump-torque\nThe setting is 25 nm.",
                                "document_keyword": "01-pump.md",
                                "similarity": 0.8,
                            }
                        ]
                    },
                }
            )

        config = load_configs()[0]
        hits = retrieve_ragflow(
            "How tight?",
            "ds-1",
            config,
            base_url="http://ragflow.local",
            api_key="secret-token",
            rerank_id="",
            opener=opener,
        )
        self.assertEqual(captured["url"], "http://ragflow.local/api/v1/retrieval")
        self.assertEqual(captured["auth"], "Bearer secret-token")
        self.assertNotIn("secret-token", json.dumps(captured["body"]))
        self.assertEqual(hits[0].chunk.chunk_ids, ("pump-torque",))
        self.assertEqual(recall_at_k(("pump-torque",), hits), 1.0)

    def test_nvidia_judge_parses_json(self):
        def opener(request, timeout=0):
            self.assertTrue(request.full_url.endswith("/chat/completions"))
            body = json.loads(request.data.decode("utf-8"))
            self.assertEqual(body["model"], "meta/llama-3.3-70b-instruct")
            self.assertNotIn("api_key", body)
            content = json.dumps({"answer": "25 nm", "faithfulness": 0.5, "reason": "partial"})
            return _Body({"choices": [{"message": {"content": content}}]})

        _answer, score, name = judge_nvidia(
            "How tight?",
            [],
            "25 nm",
            api_key="test-key",
            base_url="https://integrate.api.nvidia.com/v1",
            model="meta/llama-3.3-70b-instruct",
            opener=opener,
        )
        self.assertEqual(score, 0.5)
        self.assertEqual(name, "nvidia:meta/llama-3.3-70b-instruct")


class RunTests(unittest.TestCase):
    def test_local_mock_does_not_touch_network_and_configs_differ(self):
        def opener(*_args, **_kwargs):
            raise AssertionError("offline run called the network")

        result = run_scorecard(Settings(mode="local", provider="mock"), opener=opener)
        by_name = {item.config.name: item for item in result.config_scores}
        self.assertEqual(set(by_name), {"narrow-top1", "wide-top4", "rerank-top1"})
        self.assertLess(by_name["narrow-top1"].recall_at_k, by_name["wide-top4"].recall_at_k)
        self.assertLess(by_name["narrow-top1"].recall_at_k, by_name["rerank-top1"].recall_at_k)
        self.assertLess(by_name["narrow-top1"].mrr, by_name["wide-top4"].mrr)
        self.assertLess(by_name["narrow-top1"].faithfulness, by_name["wide-top4"].faithfulness)
        self.assertEqual(result.embedding_backend, "mock-hash")
        self.assertIn("SAMPLE DATA", result.html)
        self.assertIn("narrow-top1", result.html)
        for item in result.config_scores:
            self.assertEqual(item.questions[0].judge, "mock")
            for question in item.questions:
                self.assertGreaterEqual(question.faithfulness, 0.0)
                self.assertLessEqual(question.faithfulness, 1.0)

    def test_nvidia_failure_falls_back_to_mock(self):
        def opener(*_args, **_kwargs):
            raise urllib.error.URLError("offline")

        result = run_scorecard(
            Settings(mode="local", provider="nvidia", nvidia_api_key="test-key"),
            opener=opener,
        )
        self.assertEqual(result.embedding_backend, "mock-hash")
        self.assertTrue(any("fell back" in note.lower() or "failed" in note.lower() for note in result.notes))
        self.assertTrue(all(question.judge == "mock" for item in result.config_scores for question in item.questions))

    def test_live_mode_requires_settings(self):
        with self.assertRaises(SystemExit):
            run_scorecard(Settings(mode="live", provider="mock"))

    def test_write_report_files(self):
        result = run_scorecard(Settings(mode="local", provider="mock"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scorecard.html"
            write_result(result, path)
            payload = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(payload["mode"], "local")
            self.assertEqual(len(payload["configs"]), 3)
            self.assertIn("narrow-top1", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
