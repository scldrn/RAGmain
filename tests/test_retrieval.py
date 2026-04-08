from __future__ import annotations

from langchain_core.documents import Document

from agentic_rag.retrieval import (
    RerankingRetriever,
    TokenSparseEmbeddings,
    _best_excerpt_for_query,
    _looks_like_boilerplate,
    _looks_like_heading_tail,
    curate_retrieved_documents,
    retrieval_fetch_k,
    score_document_for_query,
)


def test_retrieval_fetch_k_expands_candidate_pool():
    assert retrieval_fetch_k(3) == 6
    assert retrieval_fetch_k(6) == 12


def test_token_sparse_embeddings_returns_stable_sparse_vector():
    embeddings = TokenSparseEmbeddings()

    vector = embeddings.embed_query("Reward hacking reward hacking")

    assert vector.indices
    assert len(vector.indices) == len(vector.values)
    assert vector.indices == sorted(vector.indices)


def test_token_sparse_embeddings_returns_empty_sparse_vector_for_empty_text():
    embeddings = TokenSparseEmbeddings()

    vector = embeddings.embed_query("")

    assert vector.indices == []
    assert vector.values == []


def test_curate_retrieved_documents_deduplicates_and_balances_sources():
    query = "What is reward hacking?"
    documents = [
        Document(
            page_content="Reward hacking exploits flaws in the reward function.",
            metadata={"source": "https://a.example/doc", "title": "A1"},
        ),
        Document(
            page_content="Reward hacking exploits flaws in the reward function.",
            metadata={"source": "https://a.example/doc", "title": "A1 duplicate"},
        ),
        Document(
            page_content="Hallucination is unsupported generation in language models.",
            metadata={"source": "https://b.example/1", "title": "B1"},
        ),
        Document(
            page_content=(
                "Specification gaming is a reward hacking failure mode in reinforcement learning."
            ),
            metadata={"source": "https://c.example/1", "title": "C1"},
        ),
        Document(
            page_content="A second chunk from source A about reward hacking and proxies.",
            metadata={"source": "https://a.example/doc", "title": "A2"},
        ),
    ]

    curated = curate_retrieved_documents(query, documents, final_k=3)

    assert len(curated) == 3
    assert len({doc.page_content for doc in curated}) == 3
    assert {doc.metadata["source"] for doc in curated} == {
        "https://a.example/doc",
        "https://c.example/1",
    }


def test_curate_retrieved_documents_trims_context_to_budget():
    query = "What is temporal consistency?"
    long_text = " ".join(
        [
            "Temporal consistency matters for coherent motion."
            " Diffusion video models denoise latent sequences."
        ]
        * 80
    )
    documents = [
        Document(
            page_content=long_text,
            metadata={"source": "https://video.example/1", "title": "Video"},
        ),
        Document(
            page_content=long_text.replace("Temporal consistency", "Frame coherence"),
            metadata={"source": "https://video.example/2", "title": "Video 2"},
        ),
    ]

    curated = curate_retrieved_documents(
        query,
        documents,
        final_k=2,
        max_total_chars=700,
        max_document_chars=400,
    )

    assert curated
    assert sum(len(doc.page_content) for doc in curated) <= 700
    assert all(len(doc.page_content) <= 400 for doc in curated)


def test_curate_retrieved_documents_skips_boilerplate_and_budget_overflow():
    query = "What is reward hacking?"
    documents = [
        Document(
            page_content="Reward hacking exploits flaws in the reward function.",
            metadata={"source": "https://a.example/1", "title": "A"},
        ),
        Document(
            page_content="References# [1] [2] [3]",
            metadata={"source": "https://b.example/1", "title": "B"},
        ),
        Document(
            page_content="Reward hacking also appears in long horizon settings with proxy rewards.",
            metadata={"source": "https://c.example/1", "title": "C"},
        ),
    ]

    curated = curate_retrieved_documents(
        query,
        documents,
        final_k=3,
        max_total_chars=len(documents[0].page_content) + 5,
    )

    assert [doc.metadata["source"] for doc in curated] == ["https://a.example/1"]


def test_curate_retrieved_documents_hits_boilerplate_skip_branch(monkeypatch):
    query = "reward hacking"
    documents = [
        Document(
            page_content="reward hacking reference block",
            metadata={"source": "https://a.example/1", "title": "A"},
        ),
        Document(
            page_content="reward hacking explains proxy rewards clearly.",
            metadata={"source": "https://b.example/1", "title": "B"},
        ),
    ]

    monkeypatch.setattr(
        "agentic_rag.retrieval._looks_like_boilerplate",
        lambda text: text == "reward hacking reference block",
    )

    curated = curate_retrieved_documents(query, documents, final_k=2)

    assert [doc.metadata["source"] for doc in curated] == ["https://b.example/1"]


def test_curate_retrieved_documents_drops_trailing_heading_outline():
    query = "What does Lilian Weng say about reward hacking?"
    text = (
        "With the rise of language models generalizing to a broad spectrum of tasks and "
        "RLHF becomes a de facto method for alignment training, reward hacking in RL "
        "training of language models has become a critical practical challenge. "
        "Instances where the model learns to modify unit tests to pass coding tasks are "
        "pretty concerning. Hope I will be able to cover the mitigation part in a "
        "dedicated post soon. Background# Reward Function in RL# Reward function defines "
        "the task, and reward shaping significantly impacts learning efficiency and "
        "accuracy in reinforcement learning."
    )
    documents = [
        Document(
            page_content=text,
            metadata={"source": "https://a.example/1", "title": "Reward Hacking"},
        )
    ]

    curated = curate_retrieved_documents(query, documents, final_k=1)

    assert curated
    assert "Background#" not in curated[0].page_content
    assert "critical practical challenge" in curated[0].page_content


def test_curate_retrieved_documents_filters_weak_tail_matches():
    query = "What is reward hacking?"
    documents = [
        Document(
            page_content="Reward hacking exploits flaws in the reward function repeatedly.",
            metadata={"source": "https://a.example/1", "title": "Reward Hacking"},
        ),
        Document(
            page_content=(
                "Reward hacking is a specification gaming failure in reinforcement learning."
            ),
            metadata={"source": "https://b.example/1", "title": "Specification Gaming"},
        ),
        Document(
            page_content="Reward Hacking in Reinforcement Learning",
            metadata={"source": "https://c.example/1", "title": "Nav Link"},
        ),
    ]

    curated = curate_retrieved_documents(query, documents, final_k=3)

    assert len(curated) == 2
    assert {doc.metadata["source"] for doc in curated} == {
        "https://a.example/1",
        "https://b.example/1",
    }


def test_best_excerpt_for_query_returns_raw_text_when_short_text_has_no_sentence_boundaries():
    assert (
        _best_excerpt_for_query("reward hacking", "reward hacking", max_chars=100)
        == "reward hacking"
    )


def test_best_excerpt_for_query_returns_empty_stripped_text_when_short_text_has_no_segments(
    monkeypatch,
):
    class EmptySentencePattern:
        def split(self, _text: str):
            return []

    monkeypatch.setattr("agentic_rag.retrieval.SENTENCE_PATTERN", EmptySentencePattern())

    assert _best_excerpt_for_query("reward hacking", "   ", max_chars=100) == ""


def test_best_excerpt_for_query_truncates_long_text_without_sentence_boundaries(monkeypatch):
    class EmptySentencePattern:
        def split(self, _text: str):
            return []

    monkeypatch.setattr("agentic_rag.retrieval.SENTENCE_PATTERN", EmptySentencePattern())
    text = "reward" * 40

    assert _best_excerpt_for_query("reward hacking", text, max_chars=20) == text[:20]


def test_looks_like_boilerplate_detects_markers_and_dense_citations():
    assert _looks_like_boilerplate("Powered by static site generator") is True
    assert _looks_like_boilerplate("[1] [2] [3]") is True


def test_looks_like_heading_tail_handles_empty_text():
    assert _looks_like_heading_tail("") is False


def test_score_document_for_query_rewards_overlap():
    query = "What is reward hacking?"
    matching_document = Document(
        page_content="Reward hacking exploits flaws in a reward function.",
        metadata={"title": "Reward Hacking"},
    )
    non_matching_document = Document(
        page_content="Diffusion video models denoise latent sequences.",
        metadata={"title": "Video"},
    )

    assert score_document_for_query(query, matching_document, rank=0) > score_document_for_query(
        query,
        non_matching_document,
        rank=0,
    )


def test_reranking_retriever_returns_curated_documents():
    class FakeRetriever:
        def invoke(self, _query: str):
            return [
                Document(
                    page_content="Reward hacking exploits flaws in the reward signal.",
                    metadata={"source": "https://a.example/1", "title": "A1"},
                ),
                Document(
                    page_content="Reward hacking exploits flaws in the reward signal.",
                    metadata={"source": "https://a.example/1", "title": "A1 duplicate"},
                ),
                Document(
                    page_content="Reward tampering interferes with the reward mechanism.",
                    metadata={"source": "https://b.example/1", "title": "B1"},
                ),
            ]

    retriever = RerankingRetriever(FakeRetriever(), final_k=2)

    curated = retriever.invoke("Explain reward hacking")

    assert len(curated) == 1
    assert curated[0].metadata["source"] == "https://a.example/1"
