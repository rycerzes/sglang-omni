# SPDX-License-Identifier: Apache-2.0
"""Energy-based Voice Activity Detector using webrtcvad."""

from __future__ import annotations

import struct
from collections import deque
from dataclasses import dataclass, field


@dataclass
class VadConfig:
    """Configuration for :class:`EnergyVad`."""

    sample_rate: int = 16000
    """Input sample rate in Hz. Must be 8000, 16000, 32000, or 48000."""

    aggressiveness: int = 3
    """webrtcvad aggressiveness 0-3. Higher = more aggressive filtering."""

    frame_duration_ms: int = 20
    """Frame size in milliseconds. Must be 10, 20, or 30."""

    min_speech_s: float = 0.25
    """Minimum consecutive speech duration to declare speech started (seconds)."""

    min_silence_s: float = 0.60
    """Minimum consecutive silence duration to declare speech stopped (seconds)."""

    preroll_s: float = 0.18
    """Pre-roll duration to prepend to speech segment (seconds)."""


@dataclass
class VadEvent:
    """Events emitted by :class:`EnergyVad`."""

    speech_started: bool = False
    """True when the VAD transitions from silence to speech."""

    speech_stopped: bool = False
    """True when the VAD transitions from speech to silence."""


class EnergyVad:
    """Stateful Voice Activity Detector based on webrtcvad.

    Processes exactly one frame (``frame_duration_ms`` ms) at a time and emits
    a :class:`VadEvent` for each frame.

    Usage example::

        vad = EnergyVad()
        for frame in audio_frames:
            event = vad.process(frame)
            if event.speech_started:
                print("Speech started!")

    Attributes:
        config: :class:`VadConfig` controlling VAD behavior.
        is_speech: Whether the VAD currently considers input as speech.
        preroll_buffer: Circular buffer of recent audio frames used for
            pre-roll when speech starts.
    """

    def __init__(self, config: VadConfig | None = None) -> None:
        import webrtcvad

        self.config = config or VadConfig()
        self._vad = webrtcvad.Vad(self.config.aggressiveness)

        sr = self.config.sample_rate
        fd = self.config.frame_duration_ms
        self._frame_samples: int = sr * fd // 1000
        # Expected bytes per frame: 2 bytes per sample (PCM16)
        self._frame_bytes: int = self._frame_samples * 2

        # Sliding window sizes
        start_frames = max(1, int(self.config.min_speech_s * 1000 / fd))
        stop_frames = max(1, int(self.config.min_silence_s * 1000 / fd))
        preroll_frames = max(0, int(self.config.preroll_s * 1000 / fd))

        self._start_required_frames: int = max(1, int(start_frames * 0.6))
        self._stop_required_unvoiced: int = max(1, int(stop_frames * 0.5))

        self._start_window: deque[bool] = deque(maxlen=start_frames)
        self._stop_window: deque[bool] = deque(maxlen=stop_frames)

        self.is_speech: bool = False

        self.preroll_buffer: deque[bytes] = deque(maxlen=max(1, preroll_frames))

        self._pending: bytes = b""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def frame_bytes(self) -> int:
        """Number of PCM16 bytes expected per frame."""
        return self._frame_bytes

    @property
    def frame_samples(self) -> int:
        """Number of samples per frame."""
        return self._frame_samples

    def process(self, pcm16_frame: bytes) -> VadEvent:
        """Process a single PCM-16 frame and return a :class:`VadEvent`.

        Args:
            pcm16_frame: Raw PCM-16 little-endian bytes, must be exactly
                :attr:`frame_bytes` bytes.

        Returns:
            :class:`VadEvent` with *at most one* flag set.
        """
        if len(pcm16_frame) != self._frame_bytes:
            raise ValueError(
                f"Expected {self._frame_bytes} bytes per frame, "
                f"got {len(pcm16_frame)}"
            )

        voiced = self._vad.is_speech(pcm16_frame, self.config.sample_rate)

        if not self.is_speech:
            # Update pre-roll buffer before we decide to start
            self.preroll_buffer.append(pcm16_frame)

        self._start_window.append(voiced)
        self._stop_window.append(voiced)

        event = VadEvent()

        if not self.is_speech:
            start_voiced = sum(self._start_window)
            if start_voiced >= self._start_required_frames:
                self.is_speech = True
                event.speech_started = True
                # Clear stop window so we don't immediately stop
                self._stop_window.clear()
        else:
            unvoiced = sum(1 for v in self._stop_window if not v)
            if unvoiced >= self._stop_required_unvoiced:
                self.is_speech = False
                event.speech_stopped = True
                # Clear start window so we don't immediately re-trigger
                self._start_window.clear()
                self.preroll_buffer.clear()

        return event

    def process_chunk(self, pcm16_bytes: bytes) -> list[VadEvent]:
        """Process an arbitrary-length PCM16 byte string.

        Buffers incomplete frames between calls.

        Args:
            pcm16_bytes: Raw PCM-16 little-endian bytes.

        Returns:
            List of :class:`VadEvent` objects, one per complete frame
            processed (may be empty if no complete frame was accumulated).
        """
        self._pending += pcm16_bytes
        events: list[VadEvent] = []
        while len(self._pending) >= self._frame_bytes:
            frame = self._pending[: self._frame_bytes]
            self._pending = self._pending[self._frame_bytes :]
            events.append(self.process(frame))
        return events

    def reset(self) -> None:
        """Reset internal state (call on session start)."""
        self._start_window.clear()
        self._stop_window.clear()
        self.preroll_buffer.clear()
        self._pending = b""
        self.is_speech = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def float32_to_pcm16(samples: "list | bytes") -> bytes:
        """Convert a float32 numpy array or list of floats to PCM16LE bytes."""
        import numpy as np  # type: ignore[import]

        arr = np.asarray(samples, dtype=np.float32)
        arr = np.clip(arr, -1.0, 1.0)
        pcm = (arr * 32767).astype(np.int16)
        return pcm.tobytes()

    @staticmethod
    def pcm16_to_float32(pcm16_bytes: bytes) -> "list[float]":
        """Convert PCM16LE bytes to list of float32 in [-1, 1]."""
        n = len(pcm16_bytes) // 2
        shorts = struct.unpack(f"<{n}h", pcm16_bytes[:n * 2])
        return [s / 32767.0 for s in shorts]
