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
from core.ports import LlmResult
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

    async def stream(self, system: str, user: str) -> AsyncIterator[str]:
        """Streaming lands properly at M8 alongside the windowed output guard."""
        result = await self.complete(system, user)
        yield result.text


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
