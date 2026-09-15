"""Dependency providers.

The service is built once at startup and handed to routes through FastAPI's
dependency system, so tests swap in fakes via `dependency_overrides` rather than
patching module internals.
"""

from __future__ import annotations

import hmac
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from core.config import Settings, get_settings
from core.service import RagService


def get_service(request: Request) -> RagService:
    service = getattr(request.app.state, "service", None)
    if service is None:  # pragma: no cover - guarded by lifespan
        raise RuntimeError("RagService is not initialised")
    return service


def get_app_settings(request: Request) -> Settings:
    return getattr(request.app.state, "settings", None) or get_settings()


def require_token(request: Request) -> None:
    """Shared-token gate for anything that spends provider quota.

    Header X-Access-Token, or ?token= for EventSource (which cannot set
    headers). A no-op when UI_ACCESS_TOKEN is unset. Health probes, docs and
    the static UI never pass through here.
    """
    settings = get_app_settings(request)
    expected = settings.ui_access_token
    if expected is None:
        return
    presented = request.headers.get("x-access-token") or request.query_params.get("token") or ""
    if not hmac.compare_digest(presented, expected.get_secret_value()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="missing or invalid access token")


ServiceDep = Annotated[RagService, Depends(get_service)]
SettingsDep = Annotated[Settings, Depends(get_app_settings)]
TokenDep = Depends(require_token)
