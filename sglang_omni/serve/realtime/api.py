# SPDX-License-Identifier: Apache-2.0
"""WebSocket endpoint for the Real-Time API (``/v1/realtime``)."""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from sglang_omni.client import Client
from sglang_omni.serve.realtime.events import parse_client_event
from sglang_omni.serve.realtime.protocol import SessionConfig
from sglang_omni.serve.realtime.session import RealtimeSession

logger = logging.getLogger(__name__)


def register_realtime(app: FastAPI) -> None:
    """Register the ``/v1/realtime`` WebSocket endpoint on *app*.

    Expects ``app.state.client`` to be a :class:`Client` instance (set
    by :func:`sglang_omni.serve.openai_api.create_app`).
    """

    @app.websocket("/v1/realtime")
    async def realtime_ws(ws: WebSocket) -> None:
        await ws.accept()

        client: Client = app.state.client
        model_name: str = getattr(app.state, "model_name", None) or "sglang-omni"
        config = SessionConfig(model=model_name)

        async def send_event(event: dict[str, Any]) -> None:
            """Send a JSON event to the client."""
            try:
                await ws.send_text(json.dumps(event))
            except Exception:
                logger.debug("Failed to send event", exc_info=True)

        session = RealtimeSession(
            client=client,
            config=config,
            send_event=send_event,
        )

        try:
            await session.initialize()

            while True:
                raw = await ws.receive_text()
                try:
                    data = parse_client_event(raw)
                except ValueError as exc:
                    await send_event(
                        {
                            "type": "error",
                            "error": {
                                "type": "invalid_request_error",
                                "message": str(exc),
                            },
                        }
                    )
                    continue

                await session.handle_event(data)

        except WebSocketDisconnect:
            logger.info("Realtime session %s disconnected", session.session_id)
        except Exception:
            logger.exception(
                "Realtime session %s error", session.session_id
            )
        finally:
            await session.close()
