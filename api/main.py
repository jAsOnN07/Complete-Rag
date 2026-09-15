"""FastAPI application.

The RagService is constructed once in the lifespan rather than per request, and
the Langfuse exporter is flushed on shutdown - Fargate's SIGTERM kills the
process before the background exporter drains, which silently loses traces.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.routes import router
from api.schemas import ErrorResponse
from core.config import get_settings
from core.errors import UpstreamServiceError

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.settings = settings

    if getattr(app.state, "service", None) is None:
        from core.service import build_service

        app.state.service = build_service(settings)

    logger.info(
        "rag api ready: collection=%s fingerprint=%s",
        settings.collection_name(),
        settings.fingerprint(),
    )
    try:
        yield
    finally:
        from observability.tracing import get_tracer

        try:
            await get_tracer().flush()
        except Exception:  # noqa: BLE001 - shutdown must not raise
            logger.warning("langfuse flush failed on shutdown", exc_info=True)


def create_app(service: object | None = None) -> FastAPI:
    app = FastAPI(
        title="RBI Circular Q&A",
        version="0.1.0",
        summary="Cited question answering over RBI circulars.",
        lifespan=lifespan,
    )
    app.state.service = service
    app.include_router(router)
    from api.ui_routes import router as ui_router

    app.include_router(ui_router)

    # The showcase UI: static files served by the API itself, mounted last so
    # every route above wins. No build step; ships in the same image.
    from pathlib import Path

    from fastapi.staticfiles import StaticFiles

    static_dir = Path(__file__).resolve().parent.parent / "ui" / "static"
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="ui")

    # Tracer first (it installs the shared provider), then HTTP instrumentation
    # on that provider, so request spans are the root of every trace.
    from observability.otel import instrument_app
    from observability.tracing import get_tracer

    get_tracer()
    instrument_app(app)

    @app.exception_handler(UpstreamServiceError)
    async def _upstream_error_handler(
        request: Request, exc: UpstreamServiceError
    ) -> JSONResponse:
        logger.error("upstream failure: %s", exc.detail)
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(detail=exc.detail).model_dump(),
        )

    @app.exception_handler(ValueError)
    async def _value_error_handler(
        request: Request, exc: ValueError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400, content=ErrorResponse(detail=str(exc)).model_dump()
        )

    return app


app = create_app()
