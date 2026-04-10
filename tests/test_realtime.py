# SPDX-License-Identifier: Apache-2.0
"""Tests for the Real-Time API"""

from __future__ import annotations

import base64
import json
from typing import Any

import numpy as np
import pytest

from sglang_omni.serve.realtime.audio_buffer import _resample
from sglang_omni.serve.realtime.conversation import (
    ContentPart,
    Conversation,
    ConversationItem,
)
from sglang_omni.serve.realtime.events import (
    CLIENT_EVENT_TYPES,
    parse_client_event,
    server_event,
)
from sglang_omni.serve.realtime.protocol import SessionConfig, TurnDetectionConfig
from sglang_omni.serve.realtime.session import RealtimeSession


# ------------------------------------------------------------------
# protocol tests
# ------------------------------------------------------------------


class TestTurnDetectionConfig:
    def test_round_trip(self) -> None:
        cfg = TurnDetectionConfig(threshold=0.3, silence_duration_ms=500)
        d = cfg.to_dict()
        cfg2 = TurnDetectionConfig.from_dict(d)
        assert cfg2 is not None
        assert cfg2.threshold == 0.3
        assert cfg2.silence_duration_ms == 500

    def test_from_dict_none(self) -> None:
        assert TurnDetectionConfig.from_dict(None) is None


class TestSessionConfig:
    def test_defaults(self) -> None:
        cfg = SessionConfig()
        assert cfg.model == "sglang-omni"
        assert cfg.temperature == 0.7
        assert cfg.turn_detection is not None

    def test_update_from_dict(self) -> None:
        cfg = SessionConfig()
        cfg.update_from_dict({"temperature": 0.5, "voice": "alloy"})
        assert cfg.temperature == 0.5
        assert cfg.voice == "alloy"
        # Unchanged fields are preserved.
        assert cfg.model == "sglang-omni"

    def test_to_dict(self) -> None:
        cfg = SessionConfig(model="test")
        d = cfg.to_dict()
        assert d["model"] == "test"
        assert "turn_detection" in d


# ------------------------------------------------------------------
# events tests
# ------------------------------------------------------------------


class TestEvents:
    def test_server_event_has_type_and_id(self) -> None:
        ev = server_event("test.event", foo="bar")
        assert ev["type"] == "test.event"
        assert ev["event_id"].startswith("evt_")
        assert ev["foo"] == "bar"

    def test_parse_client_event_valid(self) -> None:
        raw = json.dumps({"type": "session.update", "session": {}})
        data = parse_client_event(raw)
        assert data["type"] == "session.update"

    def test_parse_client_event_bytes(self) -> None:
        raw = json.dumps({"type": "response.create"}).encode()
        data = parse_client_event(raw)
        assert data["type"] == "response.create"

    def test_parse_client_event_invalid_json(self) -> None:
        with pytest.raises(ValueError, match="Invalid JSON"):
            parse_client_event("not json")

    def test_parse_client_event_not_object(self) -> None:
        with pytest.raises(ValueError, match="JSON object"):
            parse_client_event("[]")

    def test_parse_client_event_missing_type(self) -> None:
        with pytest.raises(ValueError, match="type"):
            parse_client_event('{"foo": 1}')

    def test_client_event_types(self) -> None:
        assert "session.update" in CLIENT_EVENT_TYPES
        assert "response.create" in CLIENT_EVENT_TYPES
        assert "input_audio_buffer.append" in CLIENT_EVENT_TYPES


# ------------------------------------------------------------------
# conversation tests
# ------------------------------------------------------------------


class TestConversation:
    def test_append_and_get(self) -> None:
        conv = Conversation()
        item = ConversationItem(id="i1", role="user")
        conv.append(item)
        assert len(conv) == 1
        assert conv.get("i1") is item
        assert conv.last_item_id() == "i1"

    def test_delete(self) -> None:
        conv = Conversation()
        conv.append(ConversationItem(id="i1"))
        conv.append(ConversationItem(id="i2"))
        removed = conv.delete("i1")
        assert removed is not None
        assert removed.id == "i1"
        assert len(conv) == 1
        assert conv.get("i1") is None

    def test_delete_nonexistent(self) -> None:
        conv = Conversation()
        assert conv.delete("nope") is None

    def test_to_messages(self) -> None:
        conv = Conversation()
        conv.append(
            ConversationItem(
                id="i1",
                role="user",
                content=[ContentPart(type="input_text", text="hello")],
            )
        )
        conv.append(
            ConversationItem(
                id="i2",
                role="assistant",
                content=[ContentPart(type="text", text="hi there")],
            )
        )
        msgs = conv.to_messages()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "hello"
        assert msgs[1]["content"] == "hi there"

    def test_item_round_trip(self) -> None:
        item = ConversationItem(
            id="i1",
            role="user",
            content=[ContentPart(type="input_text", text="hi")],
        )
        d = item.to_dict()
        item2 = ConversationItem.from_dict(d)
        assert item2.id == "i1"
        assert item2.content[0].text == "hi"


# ------------------------------------------------------------------
# audio_buffer tests (no VAD model — just buffer mechanics)
# ------------------------------------------------------------------


class TestResample:
    def test_same_rate(self) -> None:
        audio = np.ones(100, dtype=np.float32)
        out = _resample(audio, 16000, 16000)
        np.testing.assert_array_equal(out, audio)

    def test_downsample(self) -> None:
        audio = np.ones(240, dtype=np.float32)
        out = _resample(audio, 24000, 16000)
        # 240 samples at 24k -> 160 samples at 16k
        assert len(out) == 160

    def test_upsample(self) -> None:
        audio = np.ones(100, dtype=np.float32)
        out = _resample(audio, 16000, 24000)
        assert len(out) == 150


# ------------------------------------------------------------------
# session tests
# ------------------------------------------------------------------


class DummyClient:
    """Stub client that yields pre-configured chunks."""

    def __init__(self, chunks: list[Any] | None = None) -> None:
        self._chunks = chunks or []

    async def generate(self, request: Any, request_id: str | None = None):
        for chunk in self._chunks:
            yield chunk

    async def abort(self, request_id: str, **kwargs: Any) -> Any:
        return None

    def health(self) -> dict[str, Any]:
        return {"running": True}


class TestRealtimeSession:
    @pytest.fixture()
    def sent_events(self) -> list[dict[str, Any]]:
        return []

    @pytest.fixture()
    def session(self, sent_events: list[dict[str, Any]]) -> RealtimeSession:
        client = DummyClient()

        async def send_event(ev: dict[str, Any]) -> None:
            sent_events.append(ev)

        return RealtimeSession(
            client=client,  # type: ignore[arg-type]
            config=SessionConfig(
                turn_detection=TurnDetectionConfig(type=None),  # Disable VAD
            ),
            send_event=send_event,
        )

    @pytest.mark.asyncio
    async def test_initialize_sends_session_created(
        self, session: RealtimeSession, sent_events: list[dict[str, Any]]
    ) -> None:
        await session.initialize()
        assert len(sent_events) == 1
        assert sent_events[0]["type"] == "session.created"
        assert sent_events[0]["session"]["id"] == session.session_id

    @pytest.mark.asyncio
    async def test_session_update(
        self, session: RealtimeSession, sent_events: list[dict[str, Any]]
    ) -> None:
        await session.handle_event(
            {"type": "session.update", "session": {"temperature": 0.1}}
        )
        assert any(e["type"] == "session.updated" for e in sent_events)

    @pytest.mark.asyncio
    async def test_conversation_item_create(
        self, session: RealtimeSession, sent_events: list[dict[str, Any]]
    ) -> None:
        await session.handle_event(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                },
            }
        )
        created = [e for e in sent_events if e["type"] == "conversation.item.created"]
        assert len(created) == 1

    @pytest.mark.asyncio
    async def test_conversation_item_delete(
        self, session: RealtimeSession, sent_events: list[dict[str, Any]]
    ) -> None:
        # Create then delete.
        await session.handle_event(
            {
                "type": "conversation.item.create",
                "item": {"id": "test-item", "type": "message", "role": "user"},
            }
        )
        await session.handle_event(
            {"type": "conversation.item.delete", "item_id": "test-item"}
        )
        deleted = [e for e in sent_events if e["type"] == "conversation.item.deleted"]
        assert len(deleted) == 1

    @pytest.mark.asyncio
    async def test_unknown_event_returns_error(
        self, session: RealtimeSession, sent_events: list[dict[str, Any]]
    ) -> None:
        await session.handle_event({"type": "totally.unknown"})
        errors = [e for e in sent_events if e["type"] == "error"]
        assert len(errors) == 1
        assert "unknown" in errors[0]["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_audio_buffer_clear(
        self, session: RealtimeSession, sent_events: list[dict[str, Any]]
    ) -> None:
        await session.handle_event({"type": "input_audio_buffer.clear"})
        cleared = [
            e for e in sent_events if e["type"] == "input_audio_buffer.cleared"
        ]
        assert len(cleared) == 1

    @pytest.mark.asyncio
    async def test_audio_buffer_commit(
        self, session: RealtimeSession, sent_events: list[dict[str, Any]]
    ) -> None:
        # Append some audio then commit.
        pcm = np.zeros(480, dtype=np.int16).tobytes()
        audio_b64 = base64.b64encode(pcm).decode()
        await session.handle_event(
            {"type": "input_audio_buffer.append", "audio": audio_b64}
        )
        await session.handle_event({"type": "input_audio_buffer.commit"})
        committed = [
            e for e in sent_events if e["type"] == "input_audio_buffer.committed"
        ]
        assert len(committed) == 1

    @pytest.mark.asyncio
    async def test_close_is_idempotent(
        self, session: RealtimeSession
    ) -> None:
        await session.close()
        await session.close()  # Should not raise.
