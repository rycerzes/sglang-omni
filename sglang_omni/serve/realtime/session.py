# SPDX-License-Identifier: Apache-2.0
"""RealtimeSession — server-side state machine for one WebSocket connection.

Handles dispatching client events, managing the audio buffer, conversation,
and response lifecycle, including barge-in (user interruption).
"""

from __future__ import annotations

import base64
import logging
import uuid
from typing import Any, Callable, Coroutine

from sglang_omni.client import Client
from sglang_omni.serve.realtime import events
from sglang_omni.serve.realtime.audio_buffer import InputAudioBuffer
from sglang_omni.serve.realtime.conversation import (
    ContentPart,
    Conversation,
    ConversationItem,
)
from sglang_omni.serve.realtime.protocol import SessionConfig
from sglang_omni.serve.realtime.response import ResponseController
from sglang_omni.serve.realtime.vad import VADEvent

logger = logging.getLogger(__name__)


class RealtimeSession:
    """Server-side state for a single realtime WebSocket session.

    Args:
        client: Pipeline client for generating responses.
        session_id: Unique session identifier.
        config: Initial session configuration.
        send_event: Async callback to push a JSON-serialisable event dict
            onto the WebSocket.
    """

    def __init__(
        self,
        *,
        client: Client,
        session_id: str | None = None,
        config: SessionConfig | None = None,
        send_event: Callable[[dict[str, Any]], Coroutine[Any, Any, None]],
    ) -> None:
        self.session_id = session_id or f"sess_{uuid.uuid4().hex[:24]}"
        self._client = client
        self._config = config or SessionConfig()
        self._send = send_event

        self._conversation = Conversation()
        self._audio_buffer: InputAudioBuffer | None = None
        self._active_response: ResponseController | None = None
        self._closed = False

        self._rebuild_audio_buffer()

    async def initialize(self) -> None:
        """Send the ``session.created`` event to the client."""
        await self._send(
            events.session_created(self.session_id, self._config.to_dict())
        )

    async def handle_event(self, data: dict[str, Any]) -> None:
        """Dispatch a parsed client event."""
        event_type = data.get("type", "")
        handler = self._HANDLERS.get(event_type)
        if handler is None:
            await self._send(
                events.error_event(
                    f"Unknown event type: {event_type}",
                    code="unknown_event",
                    event_id=data.get("event_id"),
                )
            )
            return
        try:
            await handler(self, data)
        except Exception:
            logger.exception("Error handling event %s", event_type)
            await self._send(
                events.error_event(
                    f"Internal error processing {event_type}",
                    code="internal_error",
                    event_id=data.get("event_id"),
                )
            )

    async def close(self) -> None:
        """Clean up when the WebSocket is closed."""
        self._closed = True
        if self._active_response and self._active_response.is_active:
            await self._active_response.cancel()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _handle_session_update(self, data: dict[str, Any]) -> None:
        session_data = data.get("session", {})
        self._config.update_from_dict(session_data)
        self._rebuild_audio_buffer()
        await self._send(
            events.session_updated(self.session_id, self._config.to_dict())
        )

    async def _handle_input_audio_buffer_append(self, data: dict[str, Any]) -> None:
        audio_b64 = data.get("audio", "")
        if not audio_b64:
            return
        pcm_bytes = base64.b64decode(audio_b64)
        if self._audio_buffer is None:
            return

        vad_events = self._audio_buffer.append(pcm_bytes)
        for ev in vad_events:
            if ev == VADEvent.SPEECH_STARTED:
                # Barge-in: cancel active response if any.
                if self._active_response and self._active_response.is_active:
                    await self._interrupt_response()
                await self._send(
                    events.input_audio_buffer_speech_started(
                        self._audio_buffer.total_ms,
                        self._audio_buffer.pending_item_id or "",
                    )
                )
            elif ev == VADEvent.SPEECH_STOPPED:
                item_id = self._audio_buffer.pending_item_id or ""
                await self._send(
                    events.input_audio_buffer_speech_stopped(
                        self._audio_buffer.total_ms, item_id
                    )
                )
                # Auto-commit and trigger response when using server VAD.
                await self._commit_and_respond()

    async def _handle_input_audio_buffer_commit(self, data: dict[str, Any]) -> None:
        await self._commit_audio()

    async def _handle_input_audio_buffer_clear(self, data: dict[str, Any]) -> None:
        if self._audio_buffer:
            self._audio_buffer.clear()
        await self._send(events.input_audio_buffer_cleared())

    async def _handle_conversation_item_create(self, data: dict[str, Any]) -> None:
        item_data = data.get("item", {})
        item = ConversationItem.from_dict(item_data)
        prev_id = self._conversation.last_item_id()
        self._conversation.append(item)
        await self._send(
            events.conversation_item_created(item.to_dict(), prev_id)
        )

    async def _handle_conversation_item_delete(self, data: dict[str, Any]) -> None:
        item_id = data.get("item_id", "")
        removed = self._conversation.delete(item_id)
        if removed:
            await self._send(events.conversation_item_deleted(item_id))
        else:
            await self._send(
                events.error_event(
                    f"Item not found: {item_id}",
                    code="item_not_found",
                    event_id=data.get("event_id"),
                )
            )

    async def _handle_conversation_item_truncate(self, data: dict[str, Any]) -> None:
        item_id = data.get("item_id", "")
        content_index = data.get("content_index", 0)
        audio_end_ms = data.get("audio_end_ms", 0)
        ok = self._conversation.truncate_audio(item_id, content_index, audio_end_ms)
        if ok:
            await self._send(
                events.conversation_item_truncated(item_id, content_index, audio_end_ms)
            )
        else:
            await self._send(
                events.error_event(
                    f"Cannot truncate item {item_id}",
                    code="truncation_failed",
                    event_id=data.get("event_id"),
                )
            )

    async def _handle_response_create(self, data: dict[str, Any]) -> None:
        # Cancel any existing active response.
        if self._active_response and self._active_response.is_active:
            await self._active_response.cancel()

        resp = ResponseController(
            client=self._client,
            send_event=self._send,
            model=self._config.model,
            instructions=self._config.instructions,
            temperature=self._config.temperature,
            modalities=self._config.modalities,
            max_tokens=self._config.max_response_output_tokens,
            voice=self._config.voice,
        )
        self._active_response = resp
        messages = self._conversation.to_messages()
        await resp.start(messages)

    async def _handle_response_cancel(self, data: dict[str, Any]) -> None:
        if self._active_response and self._active_response.is_active:
            await self._active_response.cancel()

    # Handler dispatch table.
    _HANDLERS: dict[
        str,
        Callable[[RealtimeSession, dict[str, Any]], Coroutine[Any, Any, None]],
    ] = {
        "session.update": _handle_session_update,
        "input_audio_buffer.append": _handle_input_audio_buffer_append,
        "input_audio_buffer.commit": _handle_input_audio_buffer_commit,
        "input_audio_buffer.clear": _handle_input_audio_buffer_clear,
        "conversation.item.create": _handle_conversation_item_create,
        "conversation.item.delete": _handle_conversation_item_delete,
        "conversation.item.truncate": _handle_conversation_item_truncate,
        "response.create": _handle_response_create,
        "response.cancel": _handle_response_cancel,
    }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_audio_buffer(self) -> None:
        """(Re-)create the audio buffer from the current config."""
        td = self._config.turn_detection
        if td and td.type == "server_vad":
            self._audio_buffer = InputAudioBuffer(
                session_sample_rate=self._config.input_audio_sample_rate,
                threshold=td.threshold,
                silence_duration_ms=td.silence_duration_ms,
                prefix_padding_ms=td.prefix_padding_ms,
            )
        else:
            # Manual / push-to-talk mode: buffer without VAD.
            self._audio_buffer = InputAudioBuffer(
                session_sample_rate=self._config.input_audio_sample_rate,
                threshold=1.0,  # Effectively disable VAD.
                silence_duration_ms=999999,
            )

    async def _commit_audio(self) -> None:
        """Commit the audio buffer as a user conversation item."""
        if self._audio_buffer is None:
            return
        audio, item_id = self._audio_buffer.commit()
        prev_id = self._conversation.last_item_id()

        item = ConversationItem(
            id=item_id,
            type="message",
            role="user",
            status="completed",
            content=[ContentPart(type="input_audio")],
        )
        self._conversation.append(item)

        await self._send(
            events.input_audio_buffer_committed(item_id, prev_id)
        )
        await self._send(
            events.conversation_item_created(item.to_dict(), prev_id)
        )

    async def _commit_and_respond(self) -> None:
        """Commit audio and immediately trigger a response (server VAD)."""
        await self._commit_audio()
        await self._handle_response_create({})

    async def _interrupt_response(self) -> None:
        """Cancel the active response (barge-in)."""
        if not self._active_response or not self._active_response.is_active:
            return
        await self._active_response.cancel()

        # Add the partial output to the conversation.
        output_item = self._active_response.build_output_item()
        self._conversation.append(output_item)
