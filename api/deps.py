"""Dependency providers.

The service is built once at startup and handed to routes through FastAPI's
dependency system, so tests swap in fakes via `dependency_overrides` rather than
patching module internals.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from core.config import Settings, get_settings
from core.service import RagService


def get_service(request: Request) -> RagService:
    service = getattr(request.app.state, "service", None)
    if service is None:  # pragma: no cover - guarded by lifespan
        raise RuntimeError("RagService is not initialised")
    return service


def get_app_settings(request: Request) -> Settings:
    return getattr(request.app.state, "settings", None) or get_settings()


ServiceDep = Annotated[RagService, Depends(get_service)]
SettingsDep = Annotated[Settings, Depends(get_app_settings)]
