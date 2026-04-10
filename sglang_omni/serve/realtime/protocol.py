# SPDX-License-Identifier: Apache-2.0
"""Configuration schemas for Real-Time API sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TurnDetectionConfig:
    """Voice activity detection / turn detection configuration.

    Attributes:
        type: ``"server_vad"`` for automatic detection, or ``None`` for
            manual commit (push-to-talk).
        threshold: Speech probability threshold for Silero VAD (0–1).
        prefix_padding_ms: Audio to keep before detected speech start.
        silence_duration_ms: How long silence must last to trigger
            end-of-turn.
    """

    type: str | None = "server_vad"
    threshold: float = 0.5
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 700

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "threshold": self.threshold,
            "prefix_padding_ms": self.prefix_padding_ms,
            "silence_duration_ms": self.silence_duration_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TurnDetectionConfig | None:
        if data is None:
            return None
        return cls(
            type=data.get("type", "server_vad"),
            threshold=data.get("threshold", 0.5),
            prefix_padding_ms=data.get("prefix_padding_ms", 300),
            silence_duration_ms=data.get("silence_duration_ms", 700),
        )


@dataclass
class SessionConfig:
    """Per-session configuration.

    Attributes:
        model: Model name (e.g. ``"qwen3-omni"``).
        instructions: System-level prompt / instructions.
        modalities: Output modalities (``["text"]`` or ``["text", "audio"]``).
        voice: Voice identifier for audio output.
        input_audio_format: Expected input format (``"pcm16"``).
        output_audio_format: Output format (``"pcm16"``).
        input_audio_sample_rate: Input sample rate in Hz.
        output_audio_sample_rate: Output sample rate in Hz.
        turn_detection: VAD config, or ``None`` to disable.
        temperature: Sampling temperature.
        max_response_output_tokens: Cap on output tokens per response.
    """

    model: str = "sglang-omni"
    instructions: str = ""
    modalities: list[str] = field(default_factory=lambda: ["text", "audio"])
    voice: str = "default"
    input_audio_format: str = "pcm16"
    output_audio_format: str = "pcm16"
    input_audio_sample_rate: int = 24000
    output_audio_sample_rate: int = 24000
    turn_detection: TurnDetectionConfig | None = field(
        default_factory=TurnDetectionConfig
    )
    temperature: float = 0.7
    max_response_output_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "instructions": self.instructions,
            "modalities": list(self.modalities),
            "voice": self.voice,
            "input_audio_format": self.input_audio_format,
            "output_audio_format": self.output_audio_format,
            "turn_detection": self.turn_detection.to_dict()
            if self.turn_detection
            else None,
            "temperature": self.temperature,
            "max_response_output_tokens": self.max_response_output_tokens,
        }

    def update_from_dict(self, data: dict[str, Any]) -> None:
        """Merge partial updates into this config.

        Only keys that are present in *data* are applied.
        """
        if "instructions" in data:
            self.instructions = data["instructions"]
        if "modalities" in data:
            self.modalities = list(data["modalities"])
        if "voice" in data:
            self.voice = data["voice"]
        if "input_audio_format" in data:
            self.input_audio_format = data["input_audio_format"]
        if "output_audio_format" in data:
            self.output_audio_format = data["output_audio_format"]
        if "turn_detection" in data:
            raw = data["turn_detection"]
            self.turn_detection = TurnDetectionConfig.from_dict(raw)
        if "temperature" in data:
            self.temperature = data["temperature"]
        if "max_response_output_tokens" in data:
            self.max_response_output_tokens = data["max_response_output_tokens"]
