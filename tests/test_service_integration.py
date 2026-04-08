from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage

from agentic_rag.graph import INSUFFICIENT_CONTEXT_MESSAGE
from agentic_rag.service import AgenticRagService
from agentic_rag.settings import AgenticRagSettings


@dataclass(frozen=True)
class FakeConfig:
    provider: str
    model: str


class KeywordEmbeddings(Embeddings):
    """Deterministic keyword embeddings that make retrieval behavior easy to assert."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectorize(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vectorize(text)

    @staticmethod
    def _vectorize(text: str) -> list[float]:
        lowered = text.lower()
        reward_axis = float("reward hacking" in lowered or "reward signal" in lowered)
        proxy_axis = float("proxy" in lowered)
        hallucination_axis = float("hallucination" in lowered)
        generic_axis = float(not any((reward_axis, proxy_axis, hallucination_axis)))
        return [reward_axis, proxy_axis, hallucination_axis, generic_axis]


class InspectableResponseModel:
    def __init__(
        self,
        *,
        rewrite_map: dict[str, str] | None = None,
        answer_by_marker: dict[str, str] | None = None,
    ) -> None:
        self.rewrite_map = rewrite_map or {}
        self.answer_by_marker = answer_by_marker or {}
        self.rewrite_prompts: list[str] = []
        self.answer_prompts: list[str] = []

    def invoke(self, messages):
        prompt = messages[0]["content"]
        if prompt.startswith("Rewrite the user's question"):
            self.rewrite_prompts.append(prompt)
            question = prompt.split("Original question:\n", 1)[1].strip()
            return AIMessage(content=self.rewrite_map.get(question, question))

        if prompt.startswith("You answer questions using only the retrieved context."):
            self.answer_prompts.append(prompt)
            context = prompt.split("Context:\n", 1)[1]
            for marker, answer in self.answer_by_marker.items():
                if marker in context:
                    return AIMessage(content=answer)
            return AIMessage(content="I do not know.")

        raise AssertionError(f"Unexpected prompt: {prompt}")


class InspectableGraderModel:
    def __init__(self, *, relevant_markers: tuple[str, ...]) -> None:
        self.relevant_markers = relevant_markers
        self.prompts: list[str] = []

    def invoke(self, messages):
        prompt = messages[0]["content"]
        self.prompts.append(prompt)
        context = prompt.split("Retrieved context:\n", 1)[1]
        decision = "yes" if any(marker in context for marker in self.relevant_markers) else "no"
        return AIMessage(content=f'{{"binary_score":"{decision}"}}')


def make_service(
    tmp_path,
    *,
    max_rewrites: int = 2,
    retrieval_k: int = 3,
) -> AgenticRagService:
    return AgenticRagService(
        AgenticRagSettings(
            source_urls=("https://example.com/source",),
            chat_provider="google",
            chat_model="gemini-2.5-flash",
            chat_api_key="chat-key",
            embedding_provider="google",
            embedding_model="gemini-embedding-2-preview",
            index_cache_dir=str(tmp_path),
            retrieval_mode="dense",
            max_rewrites=max_rewrites,
            retrieval_k=retrieval_k,
        )
    )


def configure_service(
    monkeypatch,
    service: AgenticRagService,
    *,
    documents: list[Document],
    response_model: InspectableResponseModel,
    grader_model: InspectableGraderModel,
) -> None:
    models = [response_model, grader_model]

    monkeypatch.setattr(
        AgenticRagService,
        "_resolve_provider_configs",
        lambda self: (
            FakeConfig(provider="google", model="gemini-2.5-flash"),
            FakeConfig(provider="google", model="gemini-embedding-2-preview"),
        ),
    )
    monkeypatch.setattr(
        AgenticRagService,
        "_create_embeddings",
        lambda self, _config: KeywordEmbeddings(),
    )
    monkeypatch.setattr(
        AgenticRagService,
        "_create_chat_model",
        lambda self, _config: models.pop(0),
    )
    monkeypatch.setattr(
        "agentic_rag.service.preprocess_documents",
        lambda *_args, **_kwargs: list(documents),
    )


def step_names(result) -> list[str]:
    return [step.node_name for step in result.steps]


def test_query_end_to_end_returns_grounded_answer_from_curated_context(monkeypatch, tmp_path):
    relevant_sentence = "Reward hacking exploits flaws in the reward signal."
    documents = [
        Document(
            page_content=relevant_sentence,
            metadata={"source": "https://example.com/reward", "title": "Reward Hacking"},
        ),
        Document(
            page_content=relevant_sentence,
            metadata={
                "source": "https://example.com/reward",
                "title": "Reward Hacking Duplicate",
            },
        ),
        Document(
            page_content="Reward hacking references [1] [2] [3]",
            metadata={"source": "https://example.com/reward", "title": "References"},
        ),
        Document(
            page_content="Hallucination is unsupported generation in language models.",
            metadata={"source": "https://example.com/hallucination", "title": "Hallucination"},
        ),
    ]
    response_model = InspectableResponseModel(
        answer_by_marker={
            relevant_sentence: "Reward hacking exploits flaws in the reward signal. [Source 1]"
        }
    )
    grader_model = InspectableGraderModel(relevant_markers=(relevant_sentence,))
    service = make_service(tmp_path, retrieval_k=3)
    configure_service(
        monkeypatch,
        service,
        documents=documents,
        response_model=response_model,
        grader_model=grader_model,
    )

    result = service.query("What is reward hacking?")

    assert result.answer_text == "Reward hacking exploits flaws in the reward signal. [Source 1]"
    assert result.termination_reason is None
    assert result.rewrites_used == 0
    assert step_names(result) == ["prepare_retrieval", "retrieve", "generate_answer"]
    answer_prompt = response_model.answer_prompts[0]
    context = answer_prompt.split("Context:\n", 1)[1]
    assert context.count("[Source") == 1
    assert context.count(relevant_sentence) == 1
    assert "Reward hacking references [1] [2] [3]" not in context
    assert "Hallucination is unsupported generation" not in context


def test_query_end_to_end_rewrites_then_recovers_with_relevant_context(monkeypatch, tmp_path):
    relevant_sentence = "Reward hacking exploits flaws in the reward signal."
    rewrite_target = "What is reward hacking?"
    documents = [
        Document(
            page_content="Proxy servers route traffic for enterprise networks.",
            metadata={"source": "https://example.com/proxy", "title": "Proxy Servers"},
        ),
        Document(
            page_content=relevant_sentence,
            metadata={"source": "https://example.com/reward", "title": "Reward Hacking"},
        ),
    ]
    response_model = InspectableResponseModel(
        rewrite_map={"What is proxy gaming?": rewrite_target},
        answer_by_marker={
            relevant_sentence: "Reward hacking exploits flaws in the reward signal. [Source 1]"
        },
    )
    grader_model = InspectableGraderModel(relevant_markers=(relevant_sentence,))
    service = make_service(tmp_path, max_rewrites=2, retrieval_k=2)
    configure_service(
        monkeypatch,
        service,
        documents=documents,
        response_model=response_model,
        grader_model=grader_model,
    )

    result = service.query("What is proxy gaming?")

    assert result.answer_text == "Reward hacking exploits flaws in the reward signal. [Source 1]"
    assert result.termination_reason is None
    assert result.rewrites_used == 1
    assert step_names(result) == [
        "prepare_retrieval",
        "retrieve",
        "rewrite_question",
        "prepare_retrieval",
        "retrieve",
        "generate_answer",
    ]
    assert len(response_model.rewrite_prompts) == 1
    assert "What is proxy gaming?" in response_model.rewrite_prompts[0]
    assert len(grader_model.prompts) == 2
    assert "Proxy servers route traffic for enterprise networks." in grader_model.prompts[0]
    assert relevant_sentence in grader_model.prompts[1]


def test_query_end_to_end_stops_when_it_cannot_ground_an_answer(monkeypatch, tmp_path):
    documents = [
        Document(
            page_content="General alignment overviews cover broad categories of agent behavior.",
            metadata={"source": "https://example.com/alignment", "title": "Alignment Overview"},
        )
    ]
    response_model = InspectableResponseModel()
    grader_model = InspectableGraderModel(
        relevant_markers=("Reward hacking exploits flaws in the reward signal.",)
    )
    service = make_service(tmp_path, max_rewrites=2, retrieval_k=2)
    configure_service(
        monkeypatch,
        service,
        documents=documents,
        response_model=response_model,
        grader_model=grader_model,
    )

    result = service.query("What is causal scrubbing?")

    assert result.answer_text == INSUFFICIENT_CONTEXT_MESSAGE
    assert result.termination_reason == "insufficient_context"
    assert result.rewrites_used == 1
    assert step_names(result) == [
        "prepare_retrieval",
        "retrieve",
        "rewrite_question",
        "insufficient_context",
    ]
    assert response_model.answer_prompts == []
    assert grader_model.prompts == []
