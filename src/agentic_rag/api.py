from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any, Literal

from fastapi import FastAPI, Query, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import Response

from agentic_rag.errors import AgenticRagError, ConfigurationError
from agentic_rag.graph import message_text
from agentic_rag.service import AgenticRagService, HealthStatus, IndexStatus, QueryResult, QueryStep
from agentic_rag.settings import AgenticRagSettings

DEFAULT_API_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8000
logger = logging.getLogger(__name__)


class OperationMetrics(BaseModel):
    request_id: str
    duration_ms: float = Field(ge=0)
    collection_name: str


class ErrorResponse(BaseModel):
    detail: str
    error_type: str
    request_id: str


class ToolCallPayload(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class QueryTraceStep(BaseModel):
    node_name: str
    message: str
    tool_calls: list[ToolCallPayload] = Field(default_factory=list)


class QueryTrace(BaseModel):
    steps: list[QueryTraceStep]
    step_count: int


class IndexStatusPayload(BaseModel):
    cache_path: str
    fingerprint: str
    exists: bool
    document_count: int | None = None
    collection_name: str


class QueryRequest(BaseModel):
    question: str
    collection_name: str | None = None


class IngestRequest(BaseModel):
    collection_name: str | None = None


class QueryResponse(BaseModel):
    status: Literal["ok", "insufficient_context"]
    answer: str | None
    message: str
    termination_reason: str | None = None
    rewrites_used: int
    document_count: int
    index_status: IndexStatusPayload
    trace: QueryTrace
    metrics: OperationMetrics


class IngestResponse(BaseModel):
    status: Literal["ready"]
    index_status: IndexStatusPayload
    metrics: OperationMetrics


class HealthResponse(BaseModel):
    ok: bool
    chat_provider: str
    chat_model: str
    embedding_provider: str
    embedding_model: str
    ingestion_mode: str
    retrieval_mode: str
    index_cache_dir: str
    detail: str | None = None
    collection_name: str
    metrics: OperationMetrics


class IndexStatusResponse(BaseModel):
    index_status: IndexStatusPayload
    metrics: OperationMetrics


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def _operation_metrics(
    request: Request,
    *,
    start_time: float,
    collection_name: str,
) -> OperationMetrics:
    return OperationMetrics(
        request_id=_request_id(request),
        duration_ms=round((time.perf_counter() - start_time) * 1000.0, 3),
        collection_name=collection_name,
    )


def _error_response(request: Request, *, status_code: int, exc: Exception) -> JSONResponse:
    payload = ErrorResponse(
        detail=str(exc),
        error_type=type(exc).__name__,
        request_id=_request_id(request),
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump())


def _normalized_collection_override(collection_name: str | None) -> str | None:
    if collection_name is None:
        return None

    normalized = collection_name.strip()
    if not normalized:
        raise ConfigurationError("collection_name must be a non-empty string when provided.")
    return normalized


def _service_for_collection(request: Request, collection_name: str | None) -> AgenticRagService:
    base_settings = request.app.state.base_settings
    normalized_collection_name = _normalized_collection_override(collection_name)
    if normalized_collection_name is not None:
        return AgenticRagService(
            settings=replace(base_settings, collection_name=normalized_collection_name)
        )
    return AgenticRagService(settings=base_settings)


def _index_status_payload(status: IndexStatus) -> IndexStatusPayload:
    return IndexStatusPayload(
        cache_path=str(status.cache_path),
        fingerprint=status.fingerprint,
        exists=status.exists,
        document_count=status.document_count,
        collection_name=status.collection_name,
    )


def _query_trace_step(step: QueryStep) -> QueryTraceStep:
    tool_calls = []
    for tool_call in getattr(step.message, "tool_calls", None) or []:
        if not isinstance(tool_call, dict):
            continue
        tool_calls.append(
            ToolCallPayload(
                name=str(tool_call.get("name", "")),
                args=dict(tool_call.get("args", {})),
            )
        )
    return QueryTraceStep(
        node_name=step.node_name,
        message=message_text(step.message).strip(),
        tool_calls=tool_calls,
    )


def _query_response(
    request: Request,
    *,
    result: QueryResult,
    start_time: float,
) -> QueryResponse:
    answer = None if result.termination_reason == "insufficient_context" else result.answer_text
    return QueryResponse(
        status="insufficient_context" if answer is None else "ok",
        answer=answer,
        message=result.answer_text,
        termination_reason=result.termination_reason,
        rewrites_used=result.rewrites_used,
        document_count=result.document_count,
        index_status=_index_status_payload(result.index_status),
        trace=QueryTrace(
            steps=[_query_trace_step(step) for step in result.steps],
            step_count=len(result.steps),
        ),
        metrics=_operation_metrics(
            request,
            start_time=start_time,
            collection_name=result.index_status.collection_name,
        ),
    )


def _health_response(
    request: Request,
    *,
    status_payload: HealthStatus,
    start_time: float,
) -> HealthResponse:
    return HealthResponse(
        ok=status_payload.ok,
        chat_provider=status_payload.chat_provider,
        chat_model=status_payload.chat_model,
        embedding_provider=status_payload.embedding_provider,
        embedding_model=status_payload.embedding_model,
        ingestion_mode=status_payload.ingestion_mode,
        retrieval_mode=status_payload.retrieval_mode,
        index_cache_dir=str(status_payload.index_cache_dir),
        detail=status_payload.detail,
        collection_name=status_payload.collection_name,
        metrics=_operation_metrics(
            request,
            start_time=start_time,
            collection_name=status_payload.collection_name,
        ),
    )


def create_api_app(settings: AgenticRagSettings | None = None) -> FastAPI:
    app = FastAPI(
        title="Agentic RAG API",
        version="0.1.0",
        description="FastAPI runtime for the shared Agentic RAG service layer.",
    )
    app.state.base_settings = settings or AgenticRagSettings()
    if app.state.base_settings.cors_allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(app.state.base_settings.cors_allow_origins),
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def attach_request_context(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request.state.request_id = uuid.uuid4().hex
        response = await call_next(request)
        response.headers["X-Request-ID"] = _request_id(request)
        return response

    @app.exception_handler(ConfigurationError)
    async def configuration_error_handler(
        request: Request,
        exc: ConfigurationError,
    ) -> JSONResponse:
        return _error_response(request, status_code=status.HTTP_400_BAD_REQUEST, exc=exc)

    @app.exception_handler(AgenticRagError)
    async def agentic_rag_error_handler(
        request: Request,
        exc: AgenticRagError,
    ) -> JSONResponse:
        return _error_response(request, status_code=status.HTTP_503_SERVICE_UNAVAILABLE, exc=exc)

    @app.post(
        "/query",
        response_model=QueryResponse,
        responses={
            400: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    def query(payload: QueryRequest, request: Request) -> QueryResponse:
        start_time = time.perf_counter()
        service = _service_for_collection(request, payload.collection_name)
        result = service.query(payload.question)
        response = _query_response(request, result=result, start_time=start_time)
        logger.info(
            "api_query request_id=%s collection=%s duration_ms=%.3f "
            "termination_reason=%s rewrites=%s",
            response.metrics.request_id,
            response.metrics.collection_name,
            response.metrics.duration_ms,
            response.termination_reason,
            response.rewrites_used,
        )
        return response

    @app.post(
        "/ingest",
        response_model=IngestResponse,
        responses={
            400: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    def ingest(request: Request, payload: IngestRequest | None = None) -> IngestResponse:
        start_time = time.perf_counter()
        service = _service_for_collection(request, payload.collection_name if payload else None)
        index_status = service.ingest()
        response = IngestResponse(
            status="ready",
            index_status=_index_status_payload(index_status),
            metrics=_operation_metrics(
                request,
                start_time=start_time,
                collection_name=index_status.collection_name,
            ),
        )
        logger.info(
            "api_ingest request_id=%s collection=%s duration_ms=%.3f documents=%s",
            response.metrics.request_id,
            response.metrics.collection_name,
            response.metrics.duration_ms,
            response.index_status.document_count,
        )
        return response

    @app.get(
        "/health",
        response_model=HealthResponse,
        responses={
            400: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    def health(
        request: Request,
        collection_name: str | None = Query(default=None),
    ) -> HealthResponse:
        start_time = time.perf_counter()
        service = _service_for_collection(request, collection_name)
        response = _health_response(request, status_payload=service.health(), start_time=start_time)
        logger.info(
            "api_health request_id=%s collection=%s duration_ms=%.3f ok=%s",
            response.metrics.request_id,
            response.metrics.collection_name,
            response.metrics.duration_ms,
            response.ok,
        )
        return response

    @app.get(
        "/index/status",
        response_model=IndexStatusResponse,
        responses={
            400: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    def index_status(
        request: Request,
        collection_name: str | None = Query(default=None),
    ) -> IndexStatusResponse:
        start_time = time.perf_counter()
        service = _service_for_collection(request, collection_name)
        status_payload = service.index_status()
        response = IndexStatusResponse(
            index_status=_index_status_payload(status_payload),
            metrics=_operation_metrics(
                request,
                start_time=start_time,
                collection_name=status_payload.collection_name,
            ),
        )
        logger.info(
            "api_index_status request_id=%s collection=%s duration_ms=%.3f exists=%s documents=%s",
            response.metrics.request_id,
            response.metrics.collection_name,
            response.metrics.duration_ms,
            response.index_status.exists,
            response.index_status.document_count,
        )
        return response

    return app


def run() -> None:
    import uvicorn

    uvicorn.run(
        "agentic_rag.api:create_api_app",
        factory=True,
        host=os.getenv("AGENTIC_RAG_API_HOST", DEFAULT_API_HOST),
        port=int(os.getenv("AGENTIC_RAG_API_PORT", str(DEFAULT_API_PORT))),
    )


if __name__ == "__main__":
    run()
