# app/api/core/aoai/service.py

import asyncio
from typing import AsyncGenerator, Optional
from openai import AsyncAzureOpenAI
from app.api.core.config import settings


class AzureOpenAIChatClient:
    def __init__(self) -> None:
        # Cast validated types (e.g., HttpUrl) to str so the SDK can .rstrip("/")
        self.endpoint: str = str(settings.azure_endpoint_url)
        self.api_key: str = str(settings.azure_openai_api_key)
        self.api_version: str = str(settings.azure_openai_version)
        self.deployment: Optional[str] = (
            str(settings.azure_deployment_name) if settings.azure_deployment_name else None
        )

        if not self.endpoint or not self.api_key or not self.api_version:
            raise RuntimeError("Missing Azure OpenAI settings (endpoint/api_key/api_version).")

        self.aclient = AsyncAzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
            azure_deployment=self.deployment,  # keep if you want a client-level default
        )

        # Default model (Azure uses deployment name as the "model" value)
        self.default_model = self.deployment

    async def chat_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Streams Server-Sent Events lines:
          - text tokens:      data: <chunk>\n\n
          - heartbeat:        : heartbeat\n\n
          - finish reason:    event: finish\ndata: <reason>\n\n
          - terminal marker:  data: [DONE]\n\n
        """
        model = self.default_model or str(settings.azure_deployment_name)
        if not model:
            yield 'data: [ERROR] Missing Azure deployment/model name\n\n'
            yield 'data: [DONE]\n\n'
            return

        # Kick off with a heartbeat so the client renders quickly
        yield ': heartbeat\n\n'

        try:
            stream = await self.aclient.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                # Optional: cap tokens so Azure doesn't end immediately with finish_reason="stop"
                #max_tokens=512,
            )

            heartbeat_every = 15.0
            next_hb = asyncio.get_event_loop().time() + heartbeat_every

            async for chunk in stream:
                # Some chunks may have no choices. Guard accordingly.
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    # No payload in this chunk (keepalive, metadata, etc.)
                    pass

                for choice in choices:
                    delta = getattr(choice, "delta", None)

                    # 1) Content delta (most common during streaming)
                    piece = getattr(delta, "content", None) if delta else None
                    if piece:
                        yield f"data: {piece}\n\n"

                    # 2) Tool calls delta (if you use tools/function calling)
                    tool_calls = getattr(delta, "tool_calls", None) if delta else None
                    if tool_calls:
                        # Emit tool call JSON so your client can handle it if needed
                        # You can json.dumps(tool_calls) if you prefer one-line payloads
                        yield f"event: tool_calls\ndata: {tool_calls}\n\n"

                    # 3) Finish reason (stop, length, tool_calls, content_filter, etc.)
                    finish = getattr(choice, "finish_reason", None)
                    if finish:
                        yield f"event: finish\ndata: {finish}\n\n"

                # Heartbeat to keep proxies from closing the connection
                now = asyncio.get_event_loop().time()
                if now >= next_hb:
                    yield ': heartbeat\n\n'
                    next_hb = now + heartbeat_every

                await asyncio.sleep(0.01)

            # End-of-stream marker
            yield 'data: [DONE]\n\n'

        except Exception as e:
            # Send an SSE-friendly error event and close cleanly
            yield f'data: [ERROR] {type(e).__name__}: {e}\n\n'
            yield 'data: [DONE]\n\n'

    async def aclose(self) -> None:
        try:
            await self.aclient.close()
        except Exception:
            pass


azure_chat_client = AzureOpenAIChatClient()