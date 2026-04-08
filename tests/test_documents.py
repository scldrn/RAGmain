from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from agentic_rag.documents import (
    DEFAULT_FETCH_MAX_ATTEMPTS,
    DEFAULT_FETCH_RETRY_BACKOFF_SECONDS,
    _is_noise_line,
    _is_title_case_heading_word,
    _load_documents_with_retries,
    _looks_like_outline_heading,
    _looks_like_sentence_line,
    _retry_delay_seconds,
    _strip_leading_outline,
    clean_document_content,
    clean_documents,
    load_documents,
    preprocess_documents,
    split_documents,
)
from agentic_rag.errors import DocumentLoadError


def _make_doc(content: str, url: str = "https://example.com") -> Document:
    return Document(page_content=content, metadata={"source": url})


# ---------------------------------------------------------------------------
# load_documents
# ---------------------------------------------------------------------------


def test_load_documents_returns_flattened_list():
    doc1 = _make_doc("alpha", "https://a.com")
    doc2 = _make_doc("beta", "https://b.com")

    with patch("agentic_rag.documents.WebBaseLoader") as MockLoader:
        MockLoader.side_effect = [
            MagicMock(load=MagicMock(return_value=[doc1])),
            MagicMock(load=MagicMock(return_value=[doc2])),
        ]
        result = load_documents(["https://a.com", "https://b.com"])

    assert result == [doc1, doc2]


def test_load_documents_single_url():
    doc = _make_doc("content")

    with patch("agentic_rag.documents.WebBaseLoader") as MockLoader:
        MockLoader.return_value = MagicMock(load=MagicMock(return_value=[doc]))
        result = load_documents(["https://example.com"])

    assert result == [doc]


def test_load_documents_uses_timeout_and_raise_for_status():
    doc = _make_doc("content")

    with patch("agentic_rag.documents.WebBaseLoader") as MockLoader:
        MockLoader.return_value = MagicMock(load=MagicMock(return_value=[doc]))

        load_documents(["https://example.com"], timeout_seconds=9.5)

    MockLoader.assert_called_once_with(
        "https://example.com",
        requests_kwargs={"timeout": 9.5},
        raise_for_status=True,
        show_progress=False,
    )


def test_load_documents_retries_transient_failures():
    doc = _make_doc("content")
    failing_loader = MagicMock(load=MagicMock(side_effect=OSError("temporary error")))
    successful_loader = MagicMock(load=MagicMock(return_value=[doc]))

    with (
        patch("agentic_rag.documents.WebBaseLoader") as MockLoader,
        patch("agentic_rag.documents.time.sleep") as mock_sleep,
    ):
        MockLoader.side_effect = [failing_loader, successful_loader]
        result = load_documents(["https://example.com"], max_attempts=2)

    assert result == [doc]
    assert MockLoader.call_count == 2
    mock_sleep.assert_called_once_with(DEFAULT_FETCH_RETRY_BACKOFF_SECONDS)


def test_load_documents_with_retries_raises_after_retry_exhaustion():
    failing_loader = MagicMock(load=MagicMock(side_effect=OSError("temporary error")))

    with (
        patch("agentic_rag.documents.WebBaseLoader", return_value=failing_loader) as MockLoader,
        patch("agentic_rag.documents.time.sleep") as mock_sleep,
    ):
        with pytest.raises(DocumentLoadError, match="after 2 attempts"):
            _load_documents_with_retries(
                "https://example.com",
                timeout_seconds=20.0,
                max_attempts=2,
            )

    assert MockLoader.call_count == 2
    mock_sleep.assert_called_once_with(DEFAULT_FETCH_RETRY_BACKOFF_SECONDS)


def test_load_documents_skips_failed_urls_when_others_succeed(monkeypatch, caplog):
    doc = _make_doc("content", "https://good.example")

    def fake_load(url: str, *, timeout_seconds: float, max_attempts: int) -> list[Document]:
        assert timeout_seconds == 20.0
        assert max_attempts == DEFAULT_FETCH_MAX_ATTEMPTS
        if "bad" in url:
            raise DocumentLoadError("nope")
        return [doc]

    monkeypatch.setattr("agentic_rag.documents._load_documents_with_retries", fake_load)

    with caplog.at_level("WARNING"):
        result = load_documents(["https://bad.example", "https://good.example"])

    assert result == [doc]
    assert "Skipping source URL 'https://bad.example'" in caplog.text
    assert "Loaded 1 of 2 source URLs" in caplog.text


def test_load_documents_raises_typed_error_when_all_urls_fail(monkeypatch):
    monkeypatch.setattr(
        "agentic_rag.documents._load_documents_with_retries",
        lambda *args, **kwargs: (_ for _ in ()).throw(DocumentLoadError("boom")),
    )

    with pytest.raises(DocumentLoadError, match="Failed to load any source documents"):
        load_documents(["https://bad-url.example"])


def test_clean_document_content_strips_navigation_and_reference_tail():
    raw_text = """
    Lil'Log
    Posts
    Archive
    Search
    Tags
    FAQ

    Reward hacking occurs when an agent exploits flaws in the reward function.
    It can optimize a proxy reward without achieving the intended task.

    Citation
    Or @article{weng2024rewardhack,
      title = "Reward Hacking in Reinforcement Learning."
    }
    References
    [1] Example citation
    """.strip()

    cleaned = clean_document_content(raw_text)

    assert "Posts" not in cleaned
    assert "Archive" not in cleaned
    assert "References" not in cleaned
    assert "Reward hacking occurs when an agent exploits flaws" in cleaned


def test_is_noise_line_detects_powered_by_and_date_metadata():
    assert _is_noise_line("Powered by Hugo") is True
    assert _is_noise_line("Date: November 28, 2024") is True
    assert _is_noise_line("Estimated reading time: 10 min") is True


def test_is_noise_line_detects_symbol_only_lines():
    assert _is_noise_line("###") is True


def test_clean_documents_preserves_metadata():
    documents = [
        Document(
            page_content="Lil'Log\nReward hacking occurs when reward functions are exploitable.",
            metadata={"source": "https://example.com/doc", "title": "Doc"},
        )
    ]

    cleaned = clean_documents(documents)

    assert cleaned[0].metadata == {"source": "https://example.com/doc", "title": "Doc"}
    assert "Lil'Log" not in cleaned[0].page_content
    assert "Reward hacking occurs" in cleaned[0].page_content


def test_clean_document_content_strips_leading_heading_outline():
    raw_text = """
    Reward Hacking in Reinforcement Learning | Lil'Log
    Reward Hacking in Reinforcement Learning
    Background
    Reward Function in RL
    Spurious Correlation
    Let’s Define Reward Hacking
    Reward hacking occurs when an agent exploits flaws in the reward function.
    It can optimize a proxy reward without achieving the intended task.
    """.strip()

    cleaned = clean_document_content(raw_text)

    assert cleaned.startswith(
        "Reward hacking occurs when an agent exploits flaws in the reward function."
    )
    assert "Background" not in cleaned


def test_clean_document_content_strips_question_style_outline_headings():
    raw_text = """
    Why does Reward Hacking Exist?
    Hacking RL Environment
    Hacking RLHF of LLMs
    Reward hacking occurs when an agent exploits flaws in the reward function.
    It can optimize a proxy reward without achieving the intended task.
    """.strip()

    cleaned = clean_document_content(raw_text)

    assert cleaned.startswith(
        "Reward hacking occurs when an agent exploits flaws in the reward function."
    )
    assert "Why does Reward Hacking Exist?" not in cleaned
    assert "Hacking RL Environment" not in cleaned


def test_clean_document_content_strips_long_title_case_question_outline_headings():
    raw_text = """
    Why Does Reward Hacking Exist in Reinforcement Learning Systems?
    Reward Tampering
    Goal Misgeneralization
    Reward hacking occurs when an agent exploits flaws in the reward function.
    It can optimize a proxy reward without achieving the intended task.
    """.strip()

    cleaned = clean_document_content(raw_text)

    assert cleaned.startswith(
        "Reward hacking occurs when an agent exploits flaws in the reward function."
    )
    assert "Why Does Reward Hacking Exist in Reinforcement Learning Systems?" not in cleaned
    assert "Reward Tampering" not in cleaned


def test_clean_document_content_deduplicates_adjacent_lines():
    cleaned = clean_document_content(
        "Reward hacking occurs when an agent exploits flaws\n"
        "Reward hacking occurs when an agent exploits flaws\n"
        "It can optimize a proxy reward."
    )

    assert cleaned.count("Reward hacking occurs when an agent exploits flaws") == 1


def test_clean_document_content_falls_back_to_normalized_text_when_everything_is_filtered():
    cleaned = clean_document_content("Posts\nArchive\nSearch")

    assert cleaned == "Posts Archive Search"


def test_strip_leading_outline_handles_empty_input():
    assert _strip_leading_outline([]) == []


def test_looks_like_sentence_line_detects_question_and_long_text_and_reward_hacking_phrase():
    assert (
        _looks_like_sentence_line("What training budget was used in reward hacking studies?")
        is True
    )
    assert (
        _looks_like_sentence_line(
            "Why Does Reward Hacking Exist in Reinforcement Learning Systems?"
        )
        is False
    )
    assert (
        _looks_like_sentence_line("one two three four five six seven eight nine ten eleven twelve")
        is True
    )
    assert _looks_like_sentence_line("reward hacking occurs when agents exploit proxies") is True


def test_looks_like_outline_heading_rejects_short_question_fragments():
    assert _looks_like_outline_heading("Why now?") is False


def test_is_title_case_heading_word_accepts_acronyms():
    assert _is_title_case_heading_word("RLHF") is True


def test_retry_delay_seconds_uses_exponential_backoff():
    assert _retry_delay_seconds(1) == DEFAULT_FETCH_RETRY_BACKOFF_SECONDS
    assert _retry_delay_seconds(2) == DEFAULT_FETCH_RETRY_BACKOFF_SECONDS * 2


# ---------------------------------------------------------------------------
# split_documents
# ---------------------------------------------------------------------------


def test_split_documents_produces_chunks():
    long_content = " ".join(["word"] * 500)
    docs = [_make_doc(long_content)]
    chunks = split_documents(docs, chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1


def test_split_documents_preserves_metadata():
    docs = [_make_doc("short content", url="https://src.com")]
    chunks = split_documents(docs, chunk_size=500, chunk_overlap=0)
    assert all(c.metadata.get("source") == "https://src.com" for c in chunks)


def test_split_documents_returns_list_for_empty_input():
    result = split_documents([], chunk_size=500, chunk_overlap=0)
    assert result == []


# ---------------------------------------------------------------------------
# preprocess_documents
# ---------------------------------------------------------------------------


def test_preprocess_documents_loads_and_splits(monkeypatch):
    doc = _make_doc(" ".join(["word"] * 200))

    monkeypatch.setattr(
        "agentic_rag.documents.load_documents",
        lambda _urls, **_kwargs: [doc],
    )
    result = preprocess_documents(["https://example.com"], chunk_size=50, chunk_overlap=5)
    assert len(result) >= 1
    assert all(isinstance(c, Document) for c in result)


def test_preprocess_documents_forwards_fetch_timeout(monkeypatch):
    seen_kwargs: dict[str, object] = {}

    def fake_load_documents(_urls, **kwargs):
        seen_kwargs.update(kwargs)
        return [_make_doc("content")]

    monkeypatch.setattr("agentic_rag.documents.load_documents", fake_load_documents)

    preprocess_documents(
        ["https://example.com"],
        chunk_size=50,
        chunk_overlap=5,
        fetch_timeout_seconds=7.5,
    )

    assert seen_kwargs == {"timeout_seconds": 7.5}
