"""
title: Agentic RAG Pipe
author: RAGmain
author_url: https://github.com/scldrn/RAGmain
version: 0.1.0
license: MIT
"""

from __future__ import annotations

import asyncio
import inspect
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class CollectionSpec:
    collection_name: str
    display_name: str


class PipeBackendError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _default_display_name(collection_name: str) -> str:
    return collection_name.replace("-", " ").replace("_", " ").strip().title()


def _parse_collection_specs(raw_value: str, default_collection_name: str) -> list[CollectionSpec]:
    entries = [
        entry.strip() for entry in raw_value.replace(",", "\n").splitlines() if entry.strip()
    ]
    if not entries:
        return [
            CollectionSpec(
                collection_name=default_collection_name,
                display_name=_default_display_name(default_collection_name),
            )
        ]

    specs: list[CollectionSpec] = []
    seen: set[str] = set()
    for entry in entries:
        collection_name, separator, display_name = entry.partition("=")
        normalized_collection = collection_name.strip()
        normalized_display = display_name.strip() if separator else ""
        if not normalized_collection:
            continue
        if normalized_collection in seen:
            continue
        seen.add(normalized_collection)
        specs.append(
            CollectionSpec(
                collection_name=normalized_collection,
                display_name=normalized_display or _default_display_name(normalized_collection),
            )
        )

    return specs or [
        CollectionSpec(
            collection_name=default_collection_name,
            display_name=_default_display_name(default_collection_name),
        )
    ]


def _message_part_text(part: Any) -> str:
    if isinstance(part, str):
        return part.strip()
    if not isinstance(part, dict):
        return ""
    if part.get("type") == "text" and isinstance(part.get("text"), str):
        return part["text"].strip()
    text = part.get("text")
    if isinstance(text, str):
        return text.strip()
    return ""


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""

    parts = [_message_part_text(part) for part in content]
    return "\n".join(part for part in parts if part)


def _latest_user_message(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""

    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") != "user":
            continue
        text = _message_text(message.get("content"))
        if text:
            return text

    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        text = _message_text(message.get("content"))
        if text:
            return text

    return ""


def _headers(auth_header_name: str, auth_header_value: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    normalized_name = auth_header_name.strip()
    normalized_value = auth_header_value.strip()
    if normalized_name and normalized_value:
        headers[normalized_name] = normalized_value
    return headers


def _error_detail(raw_body: bytes) -> str:
    if not raw_body:
        return "Empty error response from Agentic RAG API."

    decoded = raw_body.decode("utf-8", errors="replace").strip()
    if not decoded:
        return "Empty error response from Agentic RAG API."

    try:
        payload = json.loads(decoded)
    except json.JSONDecodeError:
        return decoded

    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail.strip()

    return decoded


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            charset = response.headers.get_content_charset("utf-8")
            response_body = response.read().decode(charset)
    except urllib.error.HTTPError as exc:
        raise PipeBackendError(
            _error_detail(exc.read()),
            status_code=exc.code,
        ) from exc
    except urllib.error.URLError as exc:
        raise PipeBackendError(str(exc.reason or exc)) from exc

    try:
        parsed = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise PipeBackendError("Agentic RAG API returned invalid JSON.") from exc

    if not isinstance(parsed, dict):
        raise PipeBackendError("Agentic RAG API returned an unexpected payload.")
    return parsed


async def _emit_status(
    event_emitter: Any,
    *,
    description: str,
    done: bool,
) -> None:
    if event_emitter is None:
        return

    result = event_emitter(
        {
            "type": "status",
            "data": {
                "description": description,
                "done": done,
            },
        }
    )
    if inspect.isawaitable(result):
        await result


class Pipe:
    class Valves(BaseModel):
        API_BASE_URL: str = Field(
            default="http://host.docker.internal:8000",
            description="Base URL for the standalone Agentic RAG API service.",
        )
        API_QUERY_PATH: str = Field(
            default="/query",
            description="Path of the Agentic RAG query endpoint.",
        )
        DEFAULT_COLLECTION_NAME: str = Field(
            default="documents",
            description="Collection used when no manifold collection override is selected.",
        )
        COLLECTIONS: str = Field(
            default="",
            description=(
                "Optional newline-separated collection mapping entries in the form "
                "'collection_name=Display Name'."
            ),
        )
        MODEL_NAME_PREFIX: str = Field(
            default="Agentic RAG",
            description="Prefix shown in the Open WebUI model selector.",
        )
        REQUEST_TIMEOUT_SECONDS: float = Field(
            default=60.0,
            description="Timeout for requests to the Agentic RAG API.",
            gt=0,
        )
        AUTH_HEADER_NAME: str = Field(
            default="",
            description="Optional auth header name for a reverse proxy in front of the API.",
        )
        AUTH_HEADER_VALUE: str = Field(
            default="",
            description="Optional auth header value for a reverse proxy in front of the API.",
        )
        SHOW_STATUS: bool = Field(
            default=True,
            description="Emit progress status events in Open WebUI while the request is running.",
        )
        SHOW_METADATA_FOOTER: bool = Field(
            default=False,
            description="Append collection and request metadata to the final answer.",
        )

    def __init__(self) -> None:
        self.type = "manifold"
        self.id = "agentic-rag"
        self.valves = self.Valves()

    def _collection_specs(self) -> list[CollectionSpec]:
        return _parse_collection_specs(
            self.valves.COLLECTIONS,
            default_collection_name=self.valves.DEFAULT_COLLECTION_NAME.strip() or "documents",
        )

    def _model_id(self, collection_name: str) -> str:
        return f"{self.id}.{collection_name}"

    def _collection_name_for_model(self, model_name: str) -> str:
        model_to_collection = {
            self._model_id(spec.collection_name): spec.collection_name
            for spec in self._collection_specs()
        }
        if model_name in model_to_collection:
            return model_to_collection[model_name]

        prefix = f"{self.id}."
        if model_name.startswith(prefix):
            suffix = model_name[len(prefix) :].strip()
            if suffix:
                return suffix

        return self.valves.DEFAULT_COLLECTION_NAME.strip() or "documents"

    def pipes(self) -> list[dict[str, str]]:
        prefix = self.valves.MODEL_NAME_PREFIX.strip() or "Agentic RAG"
        return [
            {
                "id": self._model_id(spec.collection_name),
                "name": f"{prefix}: {spec.display_name}",
            }
            for spec in self._collection_specs()
        ]

    async def pipe(
        self,
        body: dict[str, Any],
        __event_emitter__: Any = None,
        __metadata__: dict[str, Any] | None = None,
    ) -> str:
        question = _latest_user_message(body.get("messages"))
        if not question:
            return "No encontré texto del usuario para consultar en Agentic RAG."

        if self.valves.SHOW_STATUS:
            await _emit_status(
                __event_emitter__,
                description="Consultando Agentic RAG...",
                done=False,
            )

        collection_name = self._collection_name_for_model(str(body.get("model", "")))
        api_base_url = self.valves.API_BASE_URL.rstrip("/")
        query_path = self.valves.API_QUERY_PATH
        if not query_path.startswith("/"):
            query_path = f"/{query_path}"

        try:
            response_payload = await asyncio.to_thread(
                _post_json,
                f"{api_base_url}{query_path}",
                {
                    "question": question,
                    "collection_name": collection_name,
                },
                headers=_headers(
                    self.valves.AUTH_HEADER_NAME,
                    self.valves.AUTH_HEADER_VALUE,
                ),
                timeout=float(self.valves.REQUEST_TIMEOUT_SECONDS),
            )
        except PipeBackendError as exc:
            if self.valves.SHOW_STATUS:
                await _emit_status(
                    __event_emitter__,
                    description="Agentic RAG devolvió un error.",
                    done=True,
                )
            status_prefix = f"{exc.status_code} " if exc.status_code is not None else ""
            return f"Agentic RAG API error: {status_prefix}{exc}".strip()

        if self.valves.SHOW_STATUS:
            await _emit_status(
                __event_emitter__,
                description="Respuesta recibida desde Agentic RAG.",
                done=True,
            )

        message = response_payload.get("message")
        if not isinstance(message, str) or not message.strip():
            return "Agentic RAG API no devolvió un mensaje utilizable."

        if not self.valves.SHOW_METADATA_FOOTER:
            return message

        metrics = response_payload.get("metrics")
        request_id = metrics.get("request_id") if isinstance(metrics, dict) else None
        footer_lines = [
            "",
            "---",
            f"collection: {collection_name}",
            f"status: {response_payload.get('status', 'unknown')}",
        ]
        if isinstance(request_id, str) and request_id.strip():
            footer_lines.append(f"request_id: {request_id}")
        return f"{message.rstrip()}\n" + "\n".join(footer_lines)
