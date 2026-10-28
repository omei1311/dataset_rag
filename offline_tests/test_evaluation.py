"""Meaningful protocol and CLI checks; runnable with Python's standard library."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from eval.metrics import (
    deduplicate_candidates, evaluate_dataset, evaluate_question,
    is_relevant, validate_ks, validate_questions,
)
from eval.offline_eval import read_json, render_markdown


ROOT = Path(__file__).resolve().parents[1]


def question(qid="q", refs=None, answerable=True):
    return {
        "question_id": qid, "question": "Which evidence supports the answer?",
        "answerable": answerable,
        "relevant_chunks": refs if refs is not None else [{"document_id": "A", "chunk_id": 0}],
    }


class EvidenceMetricsTests(unittest.TestCase):
    def test_zero_is_a_valid_chunk_id(self):
        self.assertEqual(evaluate_question(question(), [{"document_id": "A", "chunk_id": 0}])["first_relevant_rank"], 1)

    def test_chunk_id_alone_does_not_cross_document_scope(self):
        self.assertFalse(is_relevant(question(), {"document_id": "B", "chunk_id": 0}))

    def test_missing_document_is_not_relaxed(self):
        self.assertFalse(is_relevant(question(), {"chunk_id": 0, "text": "the exact answer", "item_name": "A"}))

    def test_missing_chunk_id_does_not_become_zero(self):
        self.assertFalse(is_relevant(question(), {"document_id": "A"}))

    def test_file_title_can_supply_missing_runtime_document_id(self):
        self.assertTrue(is_relevant(question(), {"file_title": "A", "chunk_id": "0"}))

    def test_present_document_id_is_authoritative(self):
        self.assertFalse(is_relevant(question(), {"document_id": "B", "file_title": "A", "chunk_id": 0}))

    def test_explicit_file_title_label_requires_that_metadata(self):
        q = question(refs=[{"file_title": "A", "chunk_id": 0}])
        self.assertFalse(is_relevant(q, {"document_id": "A", "chunk_id": 0}))
        self.assertTrue(is_relevant(q, {"file_title": "A", "chunk_id": 0}))

    def test_web_text_cannot_hit_local_ground_truth(self):
        self.assertFalse(is_relevant(question(), {"source": "web", "document_id": "A", "chunk_id": 0}))

    def test_deduplicate_before_ranking_and_never_inflate_recall(self):
        q = question(refs=[{"document_id": "A", "chunk_id": 0}, {"document_id": "B", "chunk_id": 0}])
        raw = [{"document_id": "A", "chunk_id": 0}] * 4 + [{"document_id": "B", "chunk_id": 0}]
        row = evaluate_question(q, raw, (1, 2))
        self.assertEqual(row["duplicate_candidate_count"], 3)
        self.assertEqual(row["recall_at"], {"1": 0.5, "2": 1.0})

    def test_same_chunk_in_two_documents_is_two_evidence_records(self):
        docs = [{"document_id": "A", "chunk_id": 0}, {"document_id": "B", "chunk_id": 0}]
        self.assertEqual(len(deduplicate_candidates(docs)), 2)

    def test_metrics_keep_misses_in_denominator(self):
        qs = [question("a"), question("b")]
        m = evaluate_dataset(qs, {"test": {"a": [{"document_id": "A", "chunk_id": 0}], "b": []}})["configs"]["test"]["metrics"]
        self.assertEqual(m["hit_at_1"], 0.5)
        self.assertEqual(m["recall_at_1"], 0.5)
        self.assertEqual(m["mrr"], 0.5)

    def test_first_rank_and_full_list_mrr(self):
        q = question()
        docs = [{"document_id": "B", "chunk_id": 0}, {"document_id": "A", "chunk_id": 0}]
        row = evaluate_question(q, docs, (1,))
        self.assertEqual(row["hit_at"]["1"], 0)
        self.assertEqual(row["reciprocal_rank"], 0.5)

    def test_no_answer_is_separate_and_not_a_free_hit(self):
        qs = [question("answer"), question("none", [], False)]
        m = evaluate_dataset(qs, {"x": {"answer": [], "none": []}})["configs"]["x"]["metrics"]
        self.assertEqual(m["answerable_questions"], 1)
        self.assertEqual(m["hit_at_1"], 0)
        self.assertEqual(m["no_answer_empty_candidate_rate"], 1)

    def test_only_no_answer_has_null_retrieval_metrics(self):
        m = evaluate_dataset([question("none", [], False)], {"x": {"none": []}})["configs"]["x"]["metrics"]
        self.assertIsNone(m["mrr"])
        self.assertIsNone(m["recall_at_5"])

    def test_unknown_candidates_are_counted_in_diagnostics(self):
        row = evaluate_question(question(), [{"chunk_id": 0}, {"document_id": "A"}])
        self.assertEqual(row["missing_local_identity_count"], 2)
        self.assertEqual(row["hit_at"]["1"], 0)

    def test_inputs_are_not_mutated(self):
        q = question()
        docs = [{"document_id": "A", "chunk_id": 0, "extra": {"x": 1}}]
        original = deepcopy((q, docs))
        evaluate_question(q, docs)
        self.assertEqual((q, docs), original)


class DatasetValidationTests(unittest.TestCase):
    def test_empty_or_invalid_question_collections_fail(self):
        for data in ([], {}, [None], ["q"]):
            with self.subTest(data=data), self.assertRaises(ValueError):
                validate_questions(data)

    def test_duplicate_question_ids_fail(self):
        with self.assertRaises(ValueError):
            validate_questions([question(), question()])

    def test_blank_id_question_and_nonboolean_answerability_fail(self):
        for patch in ({"question_id": " "}, {"question": ""}, {"answerable": "false"}, {"synthetic": "false"}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                validate_questions([{**question(), **patch}])

    def test_answerable_without_evidence_and_no_answer_with_evidence_fail(self):
        for q in (question(refs=[]), question(answerable=False)):
            with self.subTest(q=q), self.assertRaises(ValueError):
                validate_questions([q])

    def test_invalid_or_duplicate_ground_truth_fails(self):
        refs = [{"document_id": "A", "chunk_id": 0}] * 2
        for q in (question(refs=refs), question(refs=[{"chunk_id": 0}]), question(refs=[{"document_id": "A", "chunk_id": False}])):
            with self.subTest(q=q), self.assertRaises(ValueError):
                validate_questions([q])

    def test_legacy_ids_need_document_scope(self):
        q = {"question_id": "q", "question": "evidence?", "expected_chunk_ids": [0], "expected_file_title": "A"}
        self.assertTrue(is_relevant(q, {"file_title": "A", "chunk_id": 0}))
        del q["expected_file_title"]
        with self.assertRaises(ValueError):
            validate_questions([q])

    def test_keyword_only_labels_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_questions([{"question_id": "q", "question": "what?", "expected_keywords": ["answer"]}])

    def test_missing_or_extra_prediction_ids_fail(self):
        for mapping in ({}, {"q": [], "extra": []}):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                evaluate_dataset([question()], {"x": mapping})

    def test_invalid_k_values_fail(self):
        for ks in ([], [0], [-1], [True], [1.5]):
            with self.subTest(ks=ks), self.assertRaises(ValueError):
                validate_ks(ks)

    def test_malformed_candidates_fail(self):
        for docs in ({}, [None], [{"chunk_id": False}], [{"chunk_id": 0.5}], [{"source": "unknown"}]):
            with self.subTest(docs=docs), self.assertRaises(ValueError):
                evaluate_question(question(), docs)


class OfflineCommandTests(unittest.TestCase):
    def run_command(self, *arguments):
        return subprocess.run(
            [sys.executable, "-S", *arguments], cwd=ROOT, capture_output=True,
            text=True, encoding="utf-8", timeout=30,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )

    def test_import_and_live_help_need_no_site_packages(self):
        result = self.run_command("-c", "import sys; import eval.metrics; import eval.offline_eval; import eval.run_eval; assert not any(x in sys.modules for x in ('torch','pymilvus','langchain_core','dotenv')); assert not any(x.startswith('app.') for x in sys.modules)")
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self.run_command("eval/run_eval.py", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("oracle", result.stdout)

    def test_example_cli_writes_explicitly_synthetic_reports(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.run_command("-m", "eval.offline_eval", "--output-dir", temporary)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = read_json(Path(temporary) / "report.json")
            self.assertTrue(report["synthetic"])
            self.assertIn("SYNTHETIC", (Path(temporary) / "report.md").read_text(encoding="utf-8"))
            m = report["configs"]["synthetic-ranking-A"]["metrics"]
            self.assertEqual(m["mrr"], 0.75)
            self.assertEqual(m["recall_at_1"], 0.5)
            self.assertEqual(m["recall_at_3"], 1)
            self.assertEqual(m["duplicate_candidates_removed"], 2)

    def test_live_default_refuses_synthetic_before_runtime_import(self):
        result = self.run_command("eval/run_eval.py")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SYNTHETIC", result.stderr)
        self.assertNotIn("ModuleNotFoundError", result.stderr)

    def test_invalid_cli_exits_nonzero_without_writing_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.run_command("-m", "eval.offline_eval", "--ks", "0", "--output-dir", temporary)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((Path(temporary) / "report.json").exists())

    def test_json_nonfinite_values_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.json"
            path.write_text('{"score":NaN}', encoding="utf-8")
            with self.assertRaises(ValueError):
                read_json(path)

    def test_duplicate_json_keys_are_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.json"
            path.write_text('{"synthetic":true,"synthetic":false}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON"):
                read_json(path)


if __name__ == "__main__":
    unittest.main()
