from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest

from agentic_rag import evaluation as evaluation_module
from agentic_rag.evaluation import (
    DEFAULT_DENSE_BASELINE_PATH,
    DEFAULT_EVAL_CORPUS_PATH,
    DEFAULT_HYBRID_BASELINE_PATH,
    DeterministicEmbeddings,
    EvalCase,
    EvalCorpus,
    EvalDocument,
    _average,
    _baseline_path_for_mode,
    _best_matching_sentence,
    _build_baseline_response,
    baseline_matches_expected,
    compare_eval_summaries,
    load_eval_corpus,
    main,
    run_dense_baseline,
    run_eval_baseline,
    run_hybrid_baseline,
)


def test_eval_corpus_covers_all_phase_two_categories():
    corpus = load_eval_corpus(DEFAULT_EVAL_CORPUS_PATH)

    assert {case.category for case in corpus.cases} == {
        "answerable",
        "unanswerable",
        "adversarial",
        "citation",
    }


def test_dense_baseline_matches_checked_in_snapshot():
    summary = run_dense_baseline(load_eval_corpus(DEFAULT_EVAL_CORPUS_PATH))

    assert baseline_matches_expected(summary, DEFAULT_DENSE_BASELINE_PATH)
    assert summary["metrics"] == {
        "cases_total": 6,
        "retrieval_recall_at_k": 1.0,
        "grounded_rate": 1.0,
        "citation_validity": 1.0,
        "insufficient_context_accuracy": 1.0,
    }


def test_dense_baseline_matches_service_reranking_behavior_on_adversarial_case():
    summary = run_dense_baseline(load_eval_corpus(DEFAULT_EVAL_CORPUS_PATH))
    adversarial_case = next(
        case
        for case in summary["cases"]
        if case["case_id"] == "adversarial_reward_hacking_paraphrase"
    )

    assert adversarial_case["retrieval_recall"] == 1.0
    assert adversarial_case["grounded"] is True
    assert adversarial_case["citation_valid"] is True
    assert adversarial_case["retrieved_document_ids"] == [
        "reward_hacking_overview",
        "metrics_decoy",
    ]


def test_hybrid_baseline_matches_checked_in_snapshot():
    summary = run_hybrid_baseline(load_eval_corpus(DEFAULT_EVAL_CORPUS_PATH))

    assert baseline_matches_expected(summary, DEFAULT_HYBRID_BASELINE_PATH)
    assert summary["metrics"] == {
        "cases_total": 6,
        "retrieval_recall_at_k": 1.0,
        "grounded_rate": 1.0,
        "citation_validity": 1.0,
        "insufficient_context_accuracy": 1.0,
    }


def test_hybrid_comparison_matches_dense_baseline_when_reranking_is_aligned():
    dense_summary = run_dense_baseline(load_eval_corpus(DEFAULT_EVAL_CORPUS_PATH))
    hybrid_summary = run_hybrid_baseline(load_eval_corpus(DEFAULT_EVAL_CORPUS_PATH))

    comparison = compare_eval_summaries(dense_summary, hybrid_summary)

    assert comparison["candidate_retrieval_mode"] == "hybrid"
    assert comparison["metric_deltas"] == {
        "retrieval_recall_at_k_delta": 0.0,
        "grounded_rate_delta": 0.0,
        "citation_validity_delta": 0.0,
        "insufficient_context_accuracy_delta": 0.0,
    }
    assert comparison["improved_cases"] == []
    assert comparison["regressed_cases"] == []


def test_deterministic_embeddings_returns_zero_vector_when_text_has_no_tokens():
    embeddings = DeterministicEmbeddings(dimensions=8)

    assert embeddings.embed_query("!!!") == [0.0] * 8


def test_load_eval_corpus_rejects_duplicate_document_ids(tmp_path):
    payload = {
        "name": "dup-docs",
        "version": "1.0",
        "documents": [
            {"id": "doc-1", "title": "A", "source": "s1", "content": "alpha"},
            {"id": "doc-1", "title": "B", "source": "s2", "content": "beta"},
        ],
        "cases": [],
    }
    path = Path(tmp_path) / "dup.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate document ids"):
        load_eval_corpus(path)


def test_load_eval_corpus_rejects_unsupported_category(tmp_path):
    payload = {
        "name": "bad-category",
        "version": "1.0",
        "documents": [{"id": "doc-1", "title": "A", "source": "s1", "content": "alpha"}],
        "cases": [
            {
                "id": "case-1",
                "category": "nonsense",
                "question": "q",
                "gold_document_ids": ["doc-1"],
            }
        ],
    }
    path = Path(tmp_path) / "bad-category.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported eval case category"):
        load_eval_corpus(path)


def test_load_eval_corpus_rejects_missing_gold_documents(tmp_path):
    payload = {
        "name": "missing-doc",
        "version": "1.0",
        "documents": [{"id": "doc-1", "title": "A", "source": "s1", "content": "alpha"}],
        "cases": [
            {
                "id": "case-1",
                "category": "answerable",
                "question": "q",
                "gold_document_ids": ["doc-2"],
            }
        ],
    }
    path = Path(tmp_path) / "missing-doc.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="references missing documents"):
        load_eval_corpus(path)


def test_baseline_path_for_mode_returns_dense_and_hybrid_paths():
    assert _baseline_path_for_mode("dense") == DEFAULT_DENSE_BASELINE_PATH
    assert _baseline_path_for_mode("hybrid") == DEFAULT_HYBRID_BASELINE_PATH


def test_best_matching_sentence_falls_back_to_stripped_text_when_empty():
    assert _best_matching_sentence("reward hacking", "   ") == ""


def test_build_baseline_response_handles_empty_retrieval_and_low_overlap():
    empty_response = _build_baseline_response("reward hacking", [])
    assert empty_response.answer_text is None
    assert empty_response.cited_document_ids == ()

    low_overlap_response = _build_baseline_response(
        "reward hacking",
        [
            EvalDocument(
                "doc-1", "Title", "https://example.com", "diffusion video"
            ).to_langchain_document()
        ],
    )
    assert low_overlap_response.answer_text is not None
    assert "enough relevant context" in low_overlap_response.answer_text
    assert low_overlap_response.cited_document_ids == ()


def test_average_returns_zero_for_empty_input():
    assert _average([]) == 0.0


def test_run_eval_baseline_rejects_non_positive_retrieval_k():
    corpus = EvalCorpus(
        name="tiny",
        version="1.0",
        documents=(EvalDocument("doc-1", "A", "s1", "alpha"),),
        cases=(
            EvalCase(
                id="case-1",
                category="answerable",
                question="alpha?",
                gold_document_ids=("doc-1",),
                expected_answer_substrings=("alpha",),
                expect_insufficient_context=False,
            ),
        ),
    )

    with pytest.raises(ValueError, match="retrieval_k must be greater than zero"):
        run_eval_baseline(corpus, retrieval_mode="dense", retrieval_k=0)


def test_run_eval_baseline_dense_uses_fetch_k_and_reranking(monkeypatch):
    corpus = EvalCorpus(
        name="tiny",
        version="1.0",
        documents=(
            EvalDocument("doc-1", "Reward Hacking", "s1", "Reward hacking is exploitable."),
        ),
        cases=(
            EvalCase(
                id="case-1",
                category="answerable",
                question="What is reward hacking?",
                gold_document_ids=("doc-1",),
                expected_answer_substrings=("Reward hacking",),
                expect_insufficient_context=False,
            ),
        ),
    )
    seen: dict[str, Any] = {}

    class FakeVectorStore:
        client = object()

        def as_retriever(self, *, search_kwargs):
            seen["search_kwargs"] = search_kwargs
            return object()

    class FakeRerankingRetriever:
        def __init__(self, base_retriever, *, final_k):
            seen["base_retriever"] = base_retriever
            seen["final_k"] = final_k

        def invoke(self, query: str):
            seen["query"] = query
            return [corpus.documents[0].to_langchain_document()]

    monkeypatch.setattr(
        evaluation_module.QdrantVectorStore,
        "from_documents",
        lambda **kwargs: (
            seen.update(
                {
                    "retrieval_mode": kwargs["retrieval_mode"],
                    "sparse_embedding": kwargs["sparse_embedding"],
                }
            )
            or FakeVectorStore()
        ),
    )
    monkeypatch.setattr(evaluation_module, "RerankingRetriever", FakeRerankingRetriever)
    monkeypatch.setattr(evaluation_module, "_close_client", lambda _client: None)

    summary = run_eval_baseline(corpus, retrieval_mode="dense", retrieval_k=3)

    assert summary["metrics"]["retrieval_recall_at_k"] == 1.0
    assert seen["retrieval_mode"] is evaluation_module.RetrievalMode.DENSE
    assert seen["sparse_embedding"] is None
    assert seen["search_kwargs"] == {"k": evaluation_module.retrieval_fetch_k(3)}
    assert seen["final_k"] == 3
    assert seen["query"] == "What is reward hacking?"


def test_compare_eval_summaries_detects_regressions_and_case_mismatches():
    baseline = {
        "retrieval_mode": "dense",
        "metrics": {
            "retrieval_recall_at_k": 1.0,
            "grounded_rate": 1.0,
            "citation_validity": 1.0,
            "insufficient_context_accuracy": 1.0,
        },
        "cases": [
            {
                "case_id": "case-1",
                "grounded": True,
                "citation_valid": True,
                "insufficient_context_correct": False,
            }
        ],
    }
    candidate = {
        "retrieval_mode": "hybrid",
        "metrics": {
            "retrieval_recall_at_k": 0.5,
            "grounded_rate": 0.0,
            "citation_validity": 0.0,
            "insufficient_context_accuracy": 1.0,
        },
        "cases": [
            {
                "case_id": "case-1",
                "grounded": False,
                "citation_valid": False,
                "insufficient_context_correct": False,
            }
        ],
    }

    comparison = compare_eval_summaries(baseline, candidate)

    assert comparison["regressed_cases"] == ["case-1"]
    with pytest.raises(ValueError, match="same case ids"):
        compare_eval_summaries(baseline, {**candidate, "cases": [{"case_id": "other"}]})


def test_compare_eval_summaries_detects_improvements():
    baseline = {
        "retrieval_mode": "dense",
        "metrics": {
            "retrieval_recall_at_k": 0.5,
            "grounded_rate": 0.0,
            "citation_validity": 0.0,
            "insufficient_context_accuracy": 1.0,
        },
        "cases": [
            {
                "case_id": "case-1",
                "grounded": False,
                "citation_valid": False,
                "insufficient_context_correct": False,
            }
        ],
    }
    candidate = {
        "retrieval_mode": "hybrid",
        "metrics": {
            "retrieval_recall_at_k": 1.0,
            "grounded_rate": 1.0,
            "citation_validity": 1.0,
            "insufficient_context_accuracy": 1.0,
        },
        "cases": [
            {
                "case_id": "case-1",
                "grounded": True,
                "citation_valid": True,
                "insufficient_context_correct": False,
            }
        ],
    }

    comparison = compare_eval_summaries(baseline, candidate)

    assert comparison["improved_cases"] == ["case-1"]
    assert comparison["regressed_cases"] == []


def test_evaluation_main_writes_output_and_comparison(monkeypatch, tmp_path, capsys):
    summary = {
        "corpus_name": "tiny",
        "corpus_version": "1.0",
        "retrieval_mode": "dense",
        "retrieval_k": 3,
        "metrics": {
            "retrieval_recall_at_k": 1.0,
            "grounded_rate": 1.0,
            "citation_validity": 1.0,
            "insufficient_context_accuracy": 1.0,
            "cases_total": 1,
        },
        "cases": [
            {
                "case_id": "case-1",
                "grounded": True,
                "citation_valid": True,
                "insufficient_context_correct": False,
            }
        ],
    }
    compare_path = Path(tmp_path) / "baseline.json"
    compare_path.write_text(json.dumps(summary), encoding="utf-8")
    output_path = Path(tmp_path) / "summary.json"

    monkeypatch.setattr(evaluation_module, "load_eval_corpus", lambda path: object())
    monkeypatch.setattr(evaluation_module, "run_eval_baseline", lambda *args, **kwargs: summary)

    exit_code = main(
        ["--compare-to", str(compare_path), "--output", str(output_path), "--retrieval-k", "3"]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["comparison"]["regressed_cases"] == []
    assert capsys.readouterr().out == ""


def test_evaluation_main_rejects_invalid_retrieval_k():
    with pytest.raises(SystemExit) as exc_info:
        main(["--retrieval-k", "0"])

    assert exc_info.value.code == 2


def test_evaluation_main_returns_one_when_baseline_check_fails(monkeypatch):
    monkeypatch.setattr(evaluation_module, "load_eval_corpus", lambda path: object())
    monkeypatch.setattr(
        evaluation_module,
        "run_eval_baseline",
        lambda *args, **kwargs: {
            "corpus_name": "tiny",
            "corpus_version": "1.0",
            "retrieval_mode": "dense",
            "retrieval_k": 3,
            "metrics": {
                "retrieval_recall_at_k": 1.0,
                "grounded_rate": 1.0,
                "citation_validity": 1.0,
                "insufficient_context_accuracy": 1.0,
                "cases_total": 1,
            },
            "cases": [],
        },
    )
    monkeypatch.setattr(
        evaluation_module, "baseline_matches_expected", lambda *args, **kwargs: False
    )

    assert main(["--check-baseline"]) == 1


def test_evaluation_module_main_guard(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["agentic_rag.evaluation"])
    monkeypatch.delitem(sys.modules, "agentic_rag.evaluation", raising=False)

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("agentic_rag.evaluation", run_name="__main__")

    assert exc_info.value.code == 0
