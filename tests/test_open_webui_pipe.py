from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path


def _load_open_webui_pipe_module():
    module_name = "test_open_webui_agentic_rag_pipe"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    module_path = (
        Path(__file__).resolve().parents[1] / "integrations" / "open_webui" / "agentic_rag_pipe.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load Open WebUI pipe module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


agentic_rag_pipe = _load_open_webui_pipe_module()
Pipe = agentic_rag_pipe.Pipe
PipeBackendError = agentic_rag_pipe.PipeBackendError


class EventCollector:
    def __init__(self) -> None:
        self.events: list[dict] = []

    async def __call__(self, event: dict) -> None:
        self.events.append(event)


def test_pipe_exposes_default_collection_as_single_model():
    pipe = Pipe()
    pipe.valves = pipe.Valves(
        DEFAULT_COLLECTION_NAME="documents",
        MODEL_NAME_PREFIX="Agentic RAG",
    )

    assert pipe.pipes() == [
        {
            "id": "agentic-rag.documents",
            "name": "Agentic RAG: Documents",
        }
    ]


def test_pipe_exposes_multiple_models_from_collection_map():
    pipe = Pipe()
    pipe.valves = pipe.Valves(
        DEFAULT_COLLECTION_NAME="documents",
        COLLECTIONS="documents=Base KB\nresearch-notes=Research Notes",
        MODEL_NAME_PREFIX="RAG",
    )

    assert pipe.pipes() == [
        {
            "id": "agentic-rag.documents",
            "name": "RAG: Base KB",
        },
        {
            "id": "agentic-rag.research-notes",
            "name": "RAG: Research Notes",
        },
    ]


def test_pipe_sends_latest_user_message_and_selected_collection(monkeypatch):
    captured: dict = {}

    def fake_post_json(url, payload, *, headers, timeout):
        captured["url"] = url
        captured["payload"] = payload
        captured["headers"] = headers
        captured["timeout"] = timeout
        return {
            "status": "ok",
            "message": "Respuesta final",
            "metrics": {"request_id": "req-123"},
        }

    monkeypatch.setattr(agentic_rag_pipe, "_post_json", fake_post_json)

    pipe = Pipe()
    pipe.valves = pipe.Valves(
        API_BASE_URL="http://rag.local:8000/",
        DEFAULT_COLLECTION_NAME="documents",
        COLLECTIONS="documents=Base KB\nteam-a=Team A",
        AUTH_HEADER_NAME="X-API-Key",
        AUTH_HEADER_VALUE="secret",
        REQUEST_TIMEOUT_SECONDS=42.0,
    )
    events = EventCollector()

    response = asyncio.run(
        pipe.pipe(
            {
                "model": "agentic-rag.team-a",
                "messages": [
                    {"role": "user", "content": "Primer mensaje"},
                    {
                        "role": "assistant",
                        "content": "Respuesta previa",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Consulta final"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,abc"},
                            },
                        ],
                    },
                ],
            },
            __event_emitter__=events,
        )
    )

    assert response == "Respuesta final"
    assert captured == {
        "url": "http://rag.local:8000/query",
        "payload": {
            "question": "Consulta final",
            "collection_name": "team-a",
        },
        "headers": {
            "Content-Type": "application/json",
            "X-API-Key": "secret",
        },
        "timeout": 42.0,
    }
    assert events.events == [
        {
            "type": "status",
            "data": {
                "description": "Consultando Agentic RAG...",
                "done": False,
            },
        },
        {
            "type": "status",
            "data": {
                "description": "Respuesta recibida desde Agentic RAG.",
                "done": True,
            },
        },
    ]


def test_pipe_can_append_response_metadata(monkeypatch):
    monkeypatch.setattr(
        agentic_rag_pipe,
        "_post_json",
        lambda *args, **kwargs: {
            "status": "insufficient_context",
            "message": "No tengo suficiente contexto.",
            "metrics": {"request_id": "req-456"},
        },
    )

    pipe = Pipe()
    pipe.valves = pipe.Valves(
        DEFAULT_COLLECTION_NAME="research-notes",
        SHOW_STATUS=False,
        SHOW_METADATA_FOOTER=True,
    )

    response = asyncio.run(
        pipe.pipe(
            {
                "messages": [{"role": "user", "content": "Pregunta"}],
            }
        )
    )

    assert response == (
        "No tengo suficiente contexto.\n"
        "\n"
        "---\n"
        "collection: research-notes\n"
        "status: insufficient_context\n"
        "request_id: req-456"
    )


def test_pipe_returns_readable_backend_errors(monkeypatch):
    monkeypatch.setattr(
        agentic_rag_pipe,
        "_post_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            PipeBackendError("upstream unavailable", status_code=503)
        ),
    )

    pipe = Pipe()
    pipe.valves = pipe.Valves(SHOW_STATUS=False)

    response = asyncio.run(pipe.pipe({"messages": [{"role": "user", "content": "Hola"}]}))

    assert response == "Agentic RAG API error: 503 upstream unavailable"


def test_pipe_requires_textual_user_message():
    pipe = Pipe()
    pipe.valves = pipe.Valves(SHOW_STATUS=False)

    response = asyncio.run(
        pipe.pipe(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": "https://example.com"}}
                        ],
                    }
                ]
            }
        )
    )

    assert response == "No encontré texto del usuario para consultar en Agentic RAG."
