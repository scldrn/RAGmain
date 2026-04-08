from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, cast

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode

from agentic_rag.retrieval import RerankingRetriever, TokenSparseEmbeddings, retrieval_fetch_k

INSUFFICIENT_CONTEXT_ANSWER = "I do not have enough relevant context to answer reliably."
DEFAULT_EVAL_CORPUS_PATH = Path("tests/eval/corpus.json")
DEFAULT_DENSE_BASELINE_PATH = Path("tests/eval/baseline_dense.json")
DEFAULT_HYBRID_BASELINE_PATH = Path("tests/eval/baseline_hybrid.json")
EVAL_COLLECTION_NAME = "eval_documents"
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
ANSWER_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "each",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "rather",
    "the",
    "their",
    "than",
    "to",
    "was",
    "what",
    "when",
    "which",
    "why",
}

EvalCaseCategory = Literal["answerable", "unanswerable", "adversarial", "citation"]
EvalRetrievalMode = Literal["dense", "hybrid"]
VALID_EVAL_CASE_CATEGORIES: tuple[EvalCaseCategory, ...] = (
    "answerable",
    "unanswerable",
    "adversarial",
    "citation",
)
EVAL_METRIC_NAMES = (
    "retrieval_recall_at_k",
    "grounded_rate",
    "citation_validity",
    "insufficient_context_accuracy",
)
EVAL_CASE_SCORE_FIELDS = (
    "grounded",
    "citation_valid",
    "insufficient_context_correct",
)


@dataclass(frozen=True)
class EvalDocument:
    id: str
    title: str
    source: str
    content: str

    def to_langchain_document(self) -> Document:
        return Document(
            page_content=self.content,
            metadata={
                "doc_id": self.id,
                "source": self.source,
                "title": self.title,
            },
        )


@dataclass(frozen=True)
class EvalCase:
    id: str
    category: EvalCaseCategory
    question: str
    gold_document_ids: tuple[str, ...]
    expected_answer_substrings: tuple[str, ...]
    expect_insufficient_context: bool


@dataclass(frozen=True)
class EvalCorpus:
    name: str
    version: str
    documents: tuple[EvalDocument, ...]
    cases: tuple[EvalCase, ...]


@dataclass(frozen=True)
class EvalResponse:
    retrieved_document_ids: tuple[str, ...]
    answer_text: str | None
    cited_document_ids: tuple[str, ...]

    @property
    def answered(self) -> bool:
        return bool(self.answer_text) and self.answer_text != INSUFFICIENT_CONTEXT_ANSWER


@dataclass(frozen=True)
class CaseEvaluation:
    case_id: str
    category: EvalCaseCategory
    retrieved_document_ids: tuple[str, ...]
    cited_document_ids: tuple[str, ...]
    answered: bool
    retrieval_recall: float
    grounded: bool
    citation_valid: bool
    insufficient_context_correct: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "retrieved_document_ids": list(self.retrieved_document_ids),
            "cited_document_ids": list(self.cited_document_ids),
            "answered": self.answered,
            "retrieval_recall": round(self.retrieval_recall, 4),
            "grounded": self.grounded,
            "citation_valid": self.citation_valid,
            "insufficient_context_correct": self.insufficient_context_correct,
        }


class DeterministicEmbeddings(Embeddings):
    """Stable bag-of-words embeddings for offline retrieval evaluation."""

    def __init__(self, *, dimensions: int = 96) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectorize(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vectorize(text)

    def _vectorize(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in _normalized_tokens(text):
            index = _stable_hash(token) % self.dimensions
            vector[index] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


def _stable_hash(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest(), 16)


def _normalize_token(token: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            return token[: -len(suffix)]
    return token


def _normalized_tokens(text: str) -> tuple[str, ...]:
    raw_tokens = [_normalize_token(token) for token in TOKEN_PATTERN.findall(text.lower())]
    base_tokens = [token for token in raw_tokens if token and token not in STOPWORDS]
    phrase_tokens = [
        f"{left}_{right}"
        for left, right in zip(base_tokens, base_tokens[1:], strict=False)
        if left != right
    ]
    return tuple(base_tokens + phrase_tokens)


def load_eval_corpus(path: str | Path = DEFAULT_EVAL_CORPUS_PATH) -> EvalCorpus:
    raw_payload = json.loads(Path(path).read_text(encoding="utf-8"))
    documents = tuple(EvalDocument(**item) for item in raw_payload["documents"])
    cases = tuple(
        EvalCase(
            id=item["id"],
            category=item["category"],
            question=item["question"],
            gold_document_ids=tuple(item.get("gold_document_ids", ())),
            expected_answer_substrings=tuple(item.get("expected_answer_substrings", ())),
            expect_insufficient_context=bool(item.get("expect_insufficient_context", False)),
        )
        for item in raw_payload["cases"]
    )
    corpus = EvalCorpus(
        name=raw_payload["name"],
        version=raw_payload["version"],
        documents=documents,
        cases=cases,
    )
    _validate_corpus(corpus)
    return corpus


def _validate_corpus(corpus: EvalCorpus) -> None:
    document_ids = [document.id for document in corpus.documents]
    if len(set(document_ids)) != len(document_ids):
        raise ValueError("Eval corpus contains duplicate document ids.")

    available_ids = set(document_ids)
    for case in corpus.cases:
        if case.category not in VALID_EVAL_CASE_CATEGORIES:
            raise ValueError(f"Unsupported eval case category '{case.category}'.")
        missing_ids = set(case.gold_document_ids) - available_ids
        if missing_ids:
            raise ValueError(
                f"Eval case '{case.id}' references missing documents: {sorted(missing_ids)}"
            )


def _baseline_path_for_mode(retrieval_mode: EvalRetrievalMode) -> Path:
    if retrieval_mode == "dense":
        return DEFAULT_DENSE_BASELINE_PATH
    return DEFAULT_HYBRID_BASELINE_PATH


def _resolve_baseline_path(
    retrieval_mode: EvalRetrievalMode,
    baseline_path: str | Path | None = None,
) -> Path:
    if baseline_path is not None:
        return Path(baseline_path)
    return _baseline_path_for_mode(retrieval_mode)


def _qdrant_retrieval_mode(retrieval_mode: EvalRetrievalMode) -> RetrievalMode:
    return RetrievalMode.DENSE if retrieval_mode == "dense" else RetrievalMode.HYBRID


def _eval_sparse_embedding(retrieval_mode: EvalRetrievalMode) -> TokenSparseEmbeddings | None:
    if retrieval_mode == "dense":
        return None
    return TokenSparseEmbeddings()


def _build_eval_retriever(
    vectorstore: QdrantVectorStore, *, retrieval_k: int
) -> RerankingRetriever:
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": retrieval_fetch_k(retrieval_k)})
    return RerankingRetriever(base_retriever, final_k=retrieval_k)


def run_eval_baseline(
    corpus: EvalCorpus,
    *,
    retrieval_mode: EvalRetrievalMode,
    retrieval_k: int = 3,
) -> dict[str, Any]:
    if retrieval_k <= 0:
        raise ValueError("retrieval_k must be greater than zero.")

    with TemporaryDirectory() as temp_dir:
        vectorstore = QdrantVectorStore.from_documents(
            documents=[document.to_langchain_document() for document in corpus.documents],
            embedding=DeterministicEmbeddings(),
            path=temp_dir,
            collection_name=EVAL_COLLECTION_NAME,
            retrieval_mode=_qdrant_retrieval_mode(retrieval_mode),
            sparse_embedding=_eval_sparse_embedding(retrieval_mode),
        )
        try:
            retriever = _build_eval_retriever(vectorstore, retrieval_k=retrieval_k)
            evaluations = []
            for case in corpus.cases:
                retrieved_documents = list(retriever.invoke(case.question))
                response = _build_baseline_response(case.question, retrieved_documents)
                evaluations.append(_evaluate_case(case, response))
        finally:
            _close_client(getattr(vectorstore, "client", None))

    metrics = _summarize_metrics(corpus, evaluations)
    return {
        "corpus_name": corpus.name,
        "corpus_version": corpus.version,
        "retrieval_mode": retrieval_mode,
        "retrieval_k": retrieval_k,
        "metrics": metrics,
        "cases": [evaluation.to_dict() for evaluation in evaluations],
    }


def run_dense_baseline(
    corpus: EvalCorpus,
    *,
    retrieval_k: int = 3,
) -> dict[str, Any]:
    return run_eval_baseline(corpus, retrieval_mode="dense", retrieval_k=retrieval_k)


def run_hybrid_baseline(
    corpus: EvalCorpus,
    *,
    retrieval_k: int = 3,
) -> dict[str, Any]:
    return run_eval_baseline(corpus, retrieval_mode="hybrid", retrieval_k=retrieval_k)


def _close_client(client: Any) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        close()


def _build_baseline_response(
    question: str,
    retrieved_documents: list[Document],
) -> EvalResponse:
    retrieved_document_ids = _retrieved_document_ids(retrieved_documents)
    if not retrieved_documents:
        return EvalResponse(
            retrieved_document_ids=retrieved_document_ids,
            answer_text=None,
            cited_document_ids=(),
        )

    top_document = retrieved_documents[0]
    overlap = len(
        set(_normalized_tokens(question)) & set(_normalized_tokens(top_document.page_content))
    )
    if overlap < 1:
        return EvalResponse(
            retrieved_document_ids=retrieved_document_ids,
            answer_text=INSUFFICIENT_CONTEXT_ANSWER,
            cited_document_ids=(),
        )

    sentence = _best_matching_sentence(question, top_document.page_content)
    cited_document_id = str(top_document.metadata.get("doc_id", "unknown"))
    return EvalResponse(
        retrieved_document_ids=retrieved_document_ids,
        answer_text=f"{sentence} [Source 1]",
        cited_document_ids=(cited_document_id,),
    )


def _retrieved_document_ids(documents: list[Document]) -> tuple[str, ...]:
    return tuple(str(document.metadata.get("doc_id", "unknown")) for document in documents)


def _best_matching_sentence(question: str, text: str) -> str:
    parts = [part.strip() for part in ANSWER_SENTENCE_PATTERN.split(text.strip()) if part.strip()]
    if not parts:
        return text.strip()

    question_tokens = set(_normalized_tokens(question))
    return max(
        parts,
        key=lambda part: (
            len(question_tokens & set(_normalized_tokens(part))),
            -parts.index(part),
        ),
    )


def _evaluate_case(case: EvalCase, response: EvalResponse) -> CaseEvaluation:
    gold_document_ids = set(case.gold_document_ids)
    retrieved_document_ids = set(response.retrieved_document_ids)
    cited_document_ids = set(response.cited_document_ids)
    answer_text = (response.answer_text or "").lower()

    if gold_document_ids:
        retrieval_recall = len(gold_document_ids & retrieved_document_ids) / len(gold_document_ids)
    else:
        retrieval_recall = 1.0

    if case.expect_insufficient_context:
        grounded = INSUFFICIENT_CONTEXT_ANSWER.lower() in answer_text or not response.answered
        citation_valid = not response.cited_document_ids
        insufficient_context_correct = grounded
    else:
        grounded = bool(response.answer_text) and all(
            substring.lower() in answer_text for substring in case.expected_answer_substrings
        )
        citation_valid = bool(response.cited_document_ids) and (
            cited_document_ids <= retrieved_document_ids
            and bool(cited_document_ids & gold_document_ids)
        )
        insufficient_context_correct = False

    return CaseEvaluation(
        case_id=case.id,
        category=case.category,
        retrieved_document_ids=response.retrieved_document_ids,
        cited_document_ids=response.cited_document_ids,
        answered=response.answered,
        retrieval_recall=retrieval_recall,
        grounded=grounded,
        citation_valid=citation_valid,
        insufficient_context_correct=insufficient_context_correct,
    )


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _summarize_metrics(
    corpus: EvalCorpus,
    evaluations: list[CaseEvaluation],
) -> dict[str, Any]:
    answerable_evaluations = [
        evaluation for evaluation in evaluations if evaluation.category != "unanswerable"
    ]
    unanswerable_evaluations = [
        evaluation for evaluation in evaluations if evaluation.category == "unanswerable"
    ]
    return {
        "cases_total": len(corpus.cases),
        "retrieval_recall_at_k": round(
            _average([evaluation.retrieval_recall for evaluation in answerable_evaluations]), 4
        ),
        "grounded_rate": round(
            _average([1.0 if evaluation.grounded else 0.0 for evaluation in evaluations]),
            4,
        ),
        "citation_validity": round(
            _average([1.0 if evaluation.citation_valid else 0.0 for evaluation in evaluations]),
            4,
        ),
        "insufficient_context_accuracy": round(
            _average(
                [
                    1.0 if evaluation.insufficient_context_correct else 0.0
                    for evaluation in unanswerable_evaluations
                ]
            ),
            4,
        ),
    }


def baseline_matches_expected(
    actual_summary: dict[str, Any],
    baseline_path: str | Path | None = None,
) -> bool:
    path = _resolve_baseline_path(
        cast(EvalRetrievalMode, actual_summary["retrieval_mode"]),
        baseline_path,
    )
    expected_summary = json.loads(path.read_text(encoding="utf-8"))
    return actual_summary == expected_summary


def compare_eval_summaries(
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
) -> dict[str, Any]:
    baseline_cases = _evaluation_cases_by_id(baseline_summary)
    candidate_cases = _evaluation_cases_by_id(candidate_summary)
    if set(baseline_cases) != set(candidate_cases):
        raise ValueError("Baseline and candidate eval summaries must cover the same case ids.")
    improved_cases = []
    regressed_cases = []

    for case_id, candidate_case in candidate_cases.items():
        baseline_case = baseline_cases[case_id]
        candidate_score = _case_score(candidate_case)
        baseline_score = _case_score(baseline_case)
        if candidate_score > baseline_score:
            improved_cases.append(case_id)
        elif candidate_score < baseline_score:
            regressed_cases.append(case_id)

    return {
        "baseline_retrieval_mode": baseline_summary["retrieval_mode"],
        "candidate_retrieval_mode": candidate_summary["retrieval_mode"],
        "metric_deltas": {
            f"{name}_delta": round(
                candidate_summary["metrics"][name] - baseline_summary["metrics"][name],
                4,
            )
            for name in EVAL_METRIC_NAMES
        },
        "improved_cases": sorted(improved_cases),
        "regressed_cases": sorted(regressed_cases),
    }


def _evaluation_cases_by_id(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {case["case_id"]: case for case in summary["cases"]}


def _case_score(case: dict[str, Any]) -> int:
    return sum(1 for key in EVAL_CASE_SCORE_FIELDS if case[key])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the offline retrieval evaluation for dense or hybrid mode."
    )
    parser.add_argument(
        "--corpus",
        default=str(DEFAULT_EVAL_CORPUS_PATH),
        help="Path to the checked-in eval corpus JSON file.",
    )
    parser.add_argument(
        "--baseline",
        help="Optional path to the checked-in baseline JSON file for the selected mode.",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=("dense", "hybrid"),
        default="dense",
        help="Retrieval mode to evaluate.",
    )
    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=3,
        help="Number of documents to retrieve for each eval case.",
    )
    parser.add_argument(
        "--check-baseline",
        action="store_true",
        help="Exit non-zero if the current eval does not match the checked-in baseline.",
    )
    parser.add_argument(
        "--compare-to",
        help="Optional path to another eval summary JSON file to compare against.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write the eval summary JSON.",
    )
    args = parser.parse_args(argv)
    if args.retrieval_k <= 0:
        parser.error("--retrieval-k must be greater than zero.")

    summary = run_eval_baseline(
        load_eval_corpus(args.corpus),
        retrieval_mode=cast(EvalRetrievalMode, args.retrieval_mode),
        retrieval_k=args.retrieval_k,
    )
    baseline_path = _resolve_baseline_path(
        cast(EvalRetrievalMode, args.retrieval_mode),
        args.baseline,
    )
    output_summary = dict(summary)
    if args.compare_to:
        baseline_summary = json.loads(Path(args.compare_to).read_text(encoding="utf-8"))
        output_summary["comparison"] = compare_eval_summaries(baseline_summary, summary)
    payload = json.dumps(output_summary, indent=2, sort_keys=True)

    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)

    if args.check_baseline and not baseline_matches_expected(summary, baseline_path):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
