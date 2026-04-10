# SPDX-License-Identifier: Apache-2.0
"""Conversation state management for the Real-Time API."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ContentPart:
    """A single content part within a conversation item.

    Attributes:
        type: ``"input_audio"``, ``"input_text"``, ``"text"``, or ``"audio"``.
        text: Text content (for text parts).
        audio: Base64-encoded audio data (for audio parts).
        transcript: Transcript of audio content.
    """

    type: str = "input_audio"
    text: str | None = None
    audio: str | None = None
    transcript: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": self.type}
        if self.text is not None:
            d["text"] = self.text
        if self.audio is not None:
            d["audio"] = self.audio
        if self.transcript is not None:
            d["transcript"] = self.transcript
        return d


@dataclass
class ConversationItem:
    """A single item (message turn) in the conversation.

    Attributes:
        id: Unique item identifier.
        type: ``"message"`` or ``"function_call"`` / ``"function_call_output"``.
        role: ``"user"``, ``"assistant"``, or ``"system"``.
        status: ``"completed"``, ``"in_progress"``, or ``"incomplete"``.
        content: List of content parts.
    """

    id: str = ""
    type: str = "message"
    role: str = "user"
    status: str = "completed"
    content: list[ContentPart] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = f"item_{uuid.uuid4().hex[:24]}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "role": self.role,
            "status": self.status,
            "content": [p.to_dict() for p in self.content],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationItem:
        content = [
            ContentPart(
                type=p.get("type", "input_text"),
                text=p.get("text"),
                audio=p.get("audio"),
                transcript=p.get("transcript"),
            )
            for p in data.get("content", [])
        ]
        return cls(
            id=data.get("id", ""),
            type=data.get("type", "message"),
            role=data.get("role", "user"),
            status=data.get("status", "completed"),
            content=content,
        )


class Conversation:
    """Ordered collection of conversation items.

    The conversation tracks items by insertion order and provides
    lookup and manipulation methods used by the session.
    """

    def __init__(self) -> None:
        self._items: list[ConversationItem] = []
        self._index: dict[str, int] = {}

    @property
    def items(self) -> list[ConversationItem]:
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def get(self, item_id: str) -> ConversationItem | None:
        idx = self._index.get(item_id)
        if idx is None:
            return None
        return self._items[idx]

    def last_item_id(self) -> str | None:
        if not self._items:
            return None
        return self._items[-1].id

    def append(self, item: ConversationItem) -> None:
        """Add an item to the end of the conversation."""
        self._index[item.id] = len(self._items)
        self._items.append(item)

    def delete(self, item_id: str) -> ConversationItem | None:
        """Remove an item by id.  Returns the removed item, or ``None``."""
        idx = self._index.pop(item_id, None)
        if idx is None:
            return None
        item = self._items.pop(idx)
        # Rebuild index after removal.
        self._index = {it.id: i for i, it in enumerate(self._items)}
        return item

    def truncate_audio(
        self, item_id: str, content_index: int, audio_end_ms: int
    ) -> bool:
        """Truncate an assistant audio part.

        Removes audio data beyond *audio_end_ms* from the specified content
        part.  Returns ``True`` if the item was found and truncated.
        """
        item = self.get(item_id)
        if item is None or content_index >= len(item.content):
            return False
        part = item.content[content_index]
        if part.type not in ("audio",):
            return False
        # For now, just clear the audio data — the client already has the
        # audio up to audio_end_ms.
        part.audio = None
        return True

    def to_messages(self) -> list[dict[str, Any]]:
        """Convert the conversation to a list of message dicts suitable
        for ``GenerateRequest.messages``.
        """
        messages: list[dict[str, Any]] = []
        for item in self._items:
            if item.type != "message":
                continue
            content_parts: list[Any] = []
            for part in item.content:
                if part.type in ("input_text", "text") and part.text:
                    content_parts.append(part.text)
                # Audio is passed via metadata, not inline in messages.
            if content_parts:
                messages.append(
                    {"role": item.role, "content": content_parts[0]}
                )
            elif item.role == "user":
                # Audio-only user turn — use empty content, audio via metadata.
                messages.append({"role": item.role, "content": ""})
        return messages
