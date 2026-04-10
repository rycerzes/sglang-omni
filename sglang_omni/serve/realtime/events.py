# SPDX-License-Identifier: Apache-2.0
"""Event type definitions for the Real-Time API.

Follows the OpenAI Realtime API event naming conventions so that compatible
clients can connect without modification.
"""

from __future__ import annotations

import uuid
from typing import Any


def _new_event_id() -> str:
    return f"evt_{uuid.uuid4().hex[:24]}"


def server_event(event_type: str, **kwargs: Any) -> dict[str, Any]:
    """Build a server-sent event dict."""
    return {"type": event_type, "event_id": _new_event_id(), **kwargs}


def parse_client_event(raw: str | bytes) -> dict[str, Any]:
    """Parse and minimally validate a client-sent JSON event.

    Raises ``ValueError`` on malformed data.
    """
    import json

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Event must be a JSON object")
    if "type" not in data:
        raise ValueError("Event must contain a 'type' field")
    return data


def session_created(session_id: str, session_config: dict[str, Any]) -> dict[str, Any]:
    return server_event(
        "session.created",
        session={"id": session_id, **session_config},
    )


def session_updated(session_id: str, session_config: dict[str, Any]) -> dict[str, Any]:
    return server_event(
        "session.updated",
        session={"id": session_id, **session_config},
    )


def input_audio_buffer_speech_started(
    audio_start_ms: int, item_id: str
) -> dict[str, Any]:
    return server_event(
        "input_audio_buffer.speech_started",
        audio_start_ms=audio_start_ms,
        item_id=item_id,
    )


def input_audio_buffer_speech_stopped(
    audio_end_ms: int, item_id: str
) -> dict[str, Any]:
    return server_event(
        "input_audio_buffer.speech_stopped",
        audio_end_ms=audio_end_ms,
        item_id=item_id,
    )


def input_audio_buffer_committed(
    item_id: str, previous_item_id: str | None
) -> dict[str, Any]:
    return server_event(
        "input_audio_buffer.committed",
        item_id=item_id,
        previous_item_id=previous_item_id,
    )


def input_audio_buffer_cleared() -> dict[str, Any]:
    return server_event("input_audio_buffer.cleared")


def conversation_item_created(
    item: dict[str, Any], previous_item_id: str | None = None
) -> dict[str, Any]:
    return server_event(
        "conversation.item.created",
        item=item,
        previous_item_id=previous_item_id,
    )


def conversation_item_done(item: dict[str, Any]) -> dict[str, Any]:
    return server_event("conversation.item.done", item=item)


def conversation_item_deleted(item_id: str) -> dict[str, Any]:
    return server_event("conversation.item.deleted", item_id=item_id)


def conversation_item_truncated(
    item_id: str, content_index: int, audio_end_ms: int
) -> dict[str, Any]:
    return server_event(
        "conversation.item.truncated",
        item_id=item_id,
        content_index=content_index,
        audio_end_ms=audio_end_ms,
    )


def response_created(response: dict[str, Any]) -> dict[str, Any]:
    return server_event("response.created", response=response)


def response_output_item_added(
    response_id: str, output_index: int, item: dict[str, Any]
) -> dict[str, Any]:
    return server_event(
        "response.output_item.added",
        response_id=response_id,
        output_index=output_index,
        item=item,
    )


def response_output_item_done(
    response_id: str, output_index: int, item: dict[str, Any]
) -> dict[str, Any]:
    return server_event(
        "response.output_item.done",
        response_id=response_id,
        output_index=output_index,
        item=item,
    )


def response_content_part_added(
    response_id: str, item_id: str, output_index: int, content_index: int, part: dict
) -> dict[str, Any]:
    return server_event(
        "response.content_part.added",
        response_id=response_id,
        item_id=item_id,
        output_index=output_index,
        content_index=content_index,
        part=part,
    )


def response_audio_delta(
    response_id: str,
    item_id: str,
    output_index: int,
    content_index: int,
    delta: str,
) -> dict[str, Any]:
    """Audio output chunk (base64-encoded)."""
    return server_event(
        "response.audio.delta",
        response_id=response_id,
        item_id=item_id,
        output_index=output_index,
        content_index=content_index,
        delta=delta,
    )


def response_audio_done(
    response_id: str, item_id: str, output_index: int, content_index: int
) -> dict[str, Any]:
    return server_event(
        "response.audio.done",
        response_id=response_id,
        item_id=item_id,
        output_index=output_index,
        content_index=content_index,
    )


def response_audio_transcript_delta(
    response_id: str,
    item_id: str,
    output_index: int,
    content_index: int,
    delta: str,
) -> dict[str, Any]:
    return server_event(
        "response.audio_transcript.delta",
        response_id=response_id,
        item_id=item_id,
        output_index=output_index,
        content_index=content_index,
        delta=delta,
    )


def response_audio_transcript_done(
    response_id: str,
    item_id: str,
    output_index: int,
    content_index: int,
    transcript: str,
) -> dict[str, Any]:
    return server_event(
        "response.audio_transcript.done",
        response_id=response_id,
        item_id=item_id,
        output_index=output_index,
        content_index=content_index,
        transcript=transcript,
    )


def response_text_delta(
    response_id: str, item_id: str, output_index: int, content_index: int, delta: str
) -> dict[str, Any]:
    return server_event(
        "response.text.delta",
        response_id=response_id,
        item_id=item_id,
        output_index=output_index,
        content_index=content_index,
        delta=delta,
    )


def response_text_done(
    response_id: str,
    item_id: str,
    output_index: int,
    content_index: int,
    text: str,
) -> dict[str, Any]:
    return server_event(
        "response.text.done",
        response_id=response_id,
        item_id=item_id,
        output_index=output_index,
        content_index=content_index,
        text=text,
    )


def response_done(response: dict[str, Any]) -> dict[str, Any]:
    return server_event("response.done", response=response)


def error_event(
    message: str,
    code: str | None = None,
    param: str | None = None,
    event_id: str | None = None,
) -> dict[str, Any]:
    err: dict[str, Any] = {
        "type": "invalid_request_error",
        "message": message,
    }
    if code:
        err["code"] = code
    if param:
        err["param"] = param
    if event_id:
        err["event_id"] = event_id
    return server_event("error", error=err)


CLIENT_EVENT_TYPES = frozenset(
    {
        "session.update",
        "input_audio_buffer.append",
        "input_audio_buffer.commit",
        "input_audio_buffer.clear",
        "conversation.item.create",
        "conversation.item.delete",
        "conversation.item.truncate",
        "response.create",
        "response.cancel",
    }
)
