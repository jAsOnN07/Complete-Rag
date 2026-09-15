"""Direct Bedrock chat client.

This is the M2 placeholder. At M7 it is replaced by the Portkey gateway, which
owns routing, retries and the Groq fallback - so there is deliberately no retry
or fallback logic here beyond botocore's transport-level throttle handling.

Uses the Converse API rather than invoke_model: it is model-agnostic (so
switching Claude versions is a config change, not a body-shape change) and it
returns token usage directly, which Langfuse needs for cost accounting.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from core.errors import UpstreamServiceError
from core.ports import LlmDelta, LlmResult
from observability.tracing import Tracer, get_tracer

PROVIDER = "bedrock"


class BedrockLlmClient:
    def __init__(
        self,
        *,
        client: Any,
        model_id: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        tracer: Tracer | None = None,
    ) -> None:
        self._client = client
        self._model_id = model_id
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._tracer = tracer or get_tracer()

    @property
    def model_id(self) -> str:
        return self._model_id

    def _converse_sync(self, system: str, user: str) -> dict[str, Any]:
        return self._client.converse(
            modelId=self._model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={
                "maxTokens": self._max_tokens,
                "temperature": self._temperature,
            },
        )

    async def complete(self, system: str, user: str) -> LlmResult:
        async with self._tracer.observe(
            "llm.answer",
            as_type="generation",
            input={"system_chars": len(system), "user_chars": len(user)},
            model=self._model_id,
        ) as span:
            try:
                response = await asyncio.to_thread(self._converse_sync, system, user)
            except Exception as exc:  # noqa: BLE001 - re-raised as a typed error
                raise UpstreamServiceError(
                    "bedrock", "converse", f"{type(exc).__name__}: {exc}", cause=exc
                ) from exc
            text = "".join(
                block.get("text", "")
                for block in response["output"]["message"]["content"]
            )
            usage = response.get("usage", {})
            result = LlmResult(
                text=text,
                model_id=self._model_id,
                provider=PROVIDER,
                input_tokens=usage.get("inputTokens", 0),
                output_tokens=usage.get("outputTokens", 0),
                stop_reason=response.get("stopReason"),
            )
            span.update(
                output={"chars": len(text)},
                usage_details={
                    "input": result.input_tokens,
                    "output": result.output_tokens,
                },
                metadata={"provider": PROVIDER, "stop_reason": result.stop_reason},
            )
            return result

    def _converse_stream_sync(self, system: str, user: str) -> Any:
        return self._client.converse_stream(
            modelId=self._model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"maxTokens": self._max_tokens, "temperature": self._temperature},
        )

    async def stream(self, system: str, user: str) -> AsyncIterator[LlmDelta]:
        async with self._tracer.observe(
            "llm.answer", as_type="generation",
            input={"system_chars": len(system), "user_chars": len(user), "stream": True},
            model=self._model_id,
        ) as span:
            try:
                response = await asyncio.to_thread(self._converse_stream_sync, system, user)
            except Exception as exc:  # noqa: BLE001
                raise UpstreamServiceError(
                    "bedrock", "converse_stream", f"{type(exc).__name__}: {exc}", cause=exc
                ) from exc
            # The event stream is a blocking iterator; drain it off the loop in
            # pages so tokens still flow rather than arriving all at once.
            events = response["stream"]
            it = iter(events)
            usage: dict[str, int] = {}
            stop = None
            chars = 0
            while True:
                event = await asyncio.to_thread(next, it, None)
                if event is None:
                    break
                if "contentBlockDelta" in event:
                    text = event["contentBlockDelta"]["delta"].get("text", "")
                    chars += len(text)
                    yield LlmDelta(text)
                elif "messageStop" in event:
                    stop = event["messageStop"].get("stopReason")
                elif "metadata" in event:
                    usage = event["metadata"].get("usage", {})
            span.update(
                output={"chars": chars},
                usage_details={"input": usage.get("inputTokens", 0), "output": usage.get("outputTokens", 0)},
                metadata={"provider": PROVIDER, "stop_reason": stop},
            )
            yield LlmDelta(
                done=True, model_id=self._model_id, provider=PROVIDER,
                input_tokens=usage.get("inputTokens", 0), output_tokens=usage.get("outputTokens", 0),
                stop_reason=stop,
            )


def build_bedrock_llm(settings: Any = None) -> BedrockLlmClient:
    import boto3
    from botocore.config import Config

    if settings is None:
        from core.config import get_settings

        settings = get_settings()

    client = boto3.client(
        "bedrock-runtime",
        region_name=settings.aws_region,
        config=Config(retries={"max_attempts": 3, "mode": "adaptive"}),
        aws_access_key_id=(
            settings.aws_access_key_id.get_secret_value()
            if settings.aws_access_key_id
            else None
        ),
        aws_secret_access_key=(
            settings.aws_secret_access_key.get_secret_value()
            if settings.aws_secret_access_key
            else None
        ),
    )
    return BedrockLlmClient(client=client, model_id=settings.bedrock_llm_model_id)


class PortkeyLlmClient:
    """Chat completions through the Portkey gateway.

    No model is passed on the request: the Portkey config owns routing and
    passing one would override the fallback chain. The provider that actually
    served is derived from the model id in the response.
    """

    def __init__(
        self,
        *,
        client: Any,
        settings: Any,
        tracer: Tracer | None = None,
    ) -> None:
        self._client = client
        self._settings = settings
        self._tracer = tracer or get_tracer()

    @property
    def model_id(self) -> str:
        return self._settings.primary_model_id

    async def complete(self, system: str, user: str) -> LlmResult:
        from generation.gateway import provider_from_model

        async with self._tracer.observe(
            "llm.answer",
            as_type="generation",
            input={"system_chars": len(system), "user_chars": len(user)},
            model=self._settings.primary_model_id,
        ) as span:
            try:
                response = await self._client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=self._settings.llm_max_tokens,
                    temperature=self._settings.llm_temperature,
                )
            except Exception as exc:  # noqa: BLE001 - re-raised as a typed error
                raise UpstreamServiceError(
                    "portkey", "chat.completions", f"{type(exc).__name__}: {exc}", cause=exc
                ) from exc

            choice = response.choices[0]
            text = choice.message.content or ""
            usage = getattr(response, "usage", None)
            served = getattr(response, "model", None)
            provider = provider_from_model(served, self._settings)
            result = LlmResult(
                text=text,
                model_id=served or self._settings.primary_model_id,
                provider=provider,
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                stop_reason=getattr(choice, "finish_reason", None),
            )
            span.update(
                output={"chars": len(text), "served_model": served},
                usage_details={"input": result.input_tokens, "output": result.output_tokens},
                metadata={"provider": provider, "stop_reason": result.stop_reason},
            )
            return result

    async def stream(self, system: str, user: str) -> AsyncIterator[LlmDelta]:
        from generation.gateway import provider_from_model

        async with self._tracer.observe(
            "llm.answer", as_type="generation",
            input={"system_chars": len(system), "user_chars": len(user), "stream": True},
            model=self._settings.primary_model_id,
        ) as span:
            try:
                stream = await self._client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=self._settings.llm_max_tokens,
                    temperature=self._settings.llm_temperature,
                    stream=True,
                    stream_options={"include_usage": True},
                )
            except Exception as exc:  # noqa: BLE001
                raise UpstreamServiceError(
                    "portkey", "chat.completions.stream", f"{type(exc).__name__}: {exc}", cause=exc
                ) from exc

            served = None
            stop = None
            usage_in = usage_out = 0
            chars = 0
            try:
                async for chunk in stream:
                    served = getattr(chunk, "model", None) or served
                    choices = getattr(chunk, "choices", None) or []
                    if choices:
                        delta = getattr(choices[0], "delta", None)
                        text = getattr(delta, "content", None) or ""
                        if text:
                            chars += len(text)
                            yield LlmDelta(text)
                        stop = getattr(choices[0], "finish_reason", None) or stop
                    usage = getattr(chunk, "usage", None)
                    if usage:
                        usage_in = getattr(usage, "prompt_tokens", 0) or usage_in
                        usage_out = getattr(usage, "completion_tokens", 0) or usage_out
            except Exception as exc:  # noqa: BLE001
                raise UpstreamServiceError(
                    "portkey", "chat.completions.stream", f"{type(exc).__name__}: {exc}", cause=exc
                ) from exc

            provider = provider_from_model(served, self._settings)
            span.update(
                output={"chars": chars, "served_model": served},
                usage_details={"input": usage_in, "output": usage_out},
                metadata={"provider": provider, "stop_reason": stop},
            )
            yield LlmDelta(
                done=True, model_id=served or self._settings.primary_model_id,
                provider=provider, input_tokens=usage_in, output_tokens=usage_out, stop_reason=stop,
            )


def build_llm(settings: Any = None) -> PortkeyLlmClient | BedrockLlmClient:
    """Portkey unless explicitly told to call Bedrock directly."""
    if settings is None:
        from core.config import get_settings

        settings = get_settings()
    if settings.llm_backend == "bedrock":
        return build_bedrock_llm(settings)

    from generation.gateway import build_portkey_client

    return PortkeyLlmClient(client=build_portkey_client(settings), settings=settings)
