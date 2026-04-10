# SPDX-License-Identifier: Apache-2.0
"""Input audio buffer that accumulates PCM chunks and runs VAD."""

from __future__ import annotations

import uuid

import numpy as np

from sglang_omni.serve.realtime.vad import SileroVAD, VADEvent, VADState

# Silero VAD operates at 16 kHz regardless of the session sample rate.
_VAD_SAMPLE_RATE = 16000
_VAD_CHUNK_SAMPLES = 512  # 32 ms at 16 kHz


class InputAudioBuffer:
    """Accumulates raw PCM16 audio and detects speech boundaries via VAD.

    The buffer stores audio at the *session* sample rate (e.g. 24 kHz) and
    internally resamples to 16 kHz for VAD processing.

    Args:
        session_sample_rate: Sample rate of incoming audio (Hz).
        threshold: VAD speech probability threshold.
        silence_duration_ms: Silence required to trigger end-of-turn.
        prefix_padding_ms: Audio to keep before speech start.
    """

    def __init__(
        self,
        *,
        session_sample_rate: int = 24000,
        threshold: float = 0.5,
        silence_duration_ms: int = 700,
        prefix_padding_ms: int = 300,
    ) -> None:
        self._session_rate = session_sample_rate
        self._prefix_padding_ms = prefix_padding_ms

        self._vad = SileroVAD(
            threshold=threshold,
            silence_duration_ms=silence_duration_ms,
            prefix_padding_ms=prefix_padding_ms,
            sample_rate=_VAD_SAMPLE_RATE,
        )

        # Raw audio at session sample rate (float32, mono).
        self._chunks: list[np.ndarray] = []
        self._total_samples: int = 0

        # Resampled audio accumulator for VAD (16 kHz float32).
        self._vad_buf = np.empty(0, dtype=np.float32)

        # Track the sample offset where speech started (session rate).
        self._speech_start_sample: int | None = None

        # Pending item id created when speech is detected.
        self._pending_item_id: str | None = None

    @property
    def total_ms(self) -> int:
        """Total audio duration in the buffer (ms)."""
        return self._total_samples * 1000 // self._session_rate

    @property
    def pending_item_id(self) -> str | None:
        return self._pending_item_id

    @property
    def is_speech(self) -> bool:
        return self._vad.state == VADState.LISTENING

    def append(self, pcm16_bytes: bytes) -> list[VADEvent]:
        """Append raw PCM16 bytes and run VAD.

        Args:
            pcm16_bytes: Little-endian signed 16-bit PCM audio at the
                session sample rate.

        Returns:
            List of VAD events triggered by this chunk.
        """
        # Decode PCM16 → float32 at session rate.
        samples = np.frombuffer(pcm16_bytes, dtype="<i2").astype(np.float32) / 32768.0
        self._chunks.append(samples)
        self._total_samples += len(samples)

        # Resample to 16 kHz for VAD if needed.
        if self._session_rate == _VAD_SAMPLE_RATE:
            vad_samples = samples
        else:
            vad_samples = _resample(samples, self._session_rate, _VAD_SAMPLE_RATE)

        self._vad_buf = np.concatenate([self._vad_buf, vad_samples])

        all_events: list[VADEvent] = []
        while len(self._vad_buf) >= _VAD_CHUNK_SAMPLES:
            chunk = self._vad_buf[:_VAD_CHUNK_SAMPLES]
            self._vad_buf = self._vad_buf[_VAD_CHUNK_SAMPLES:]
            events = self._vad.process_chunk(chunk)
            for ev in events:
                if ev == VADEvent.SPEECH_STARTED:
                    self._speech_start_sample = max(
                        0,
                        self._total_samples
                        - len(samples)
                        - self._prefix_padding_ms * self._session_rate // 1000,
                    )
                    self._pending_item_id = f"item_{uuid.uuid4().hex[:24]}"
                all_events.append(ev)

        return all_events

    def commit(self) -> tuple[np.ndarray, str]:
        """Commit the buffered audio and return it.

        Returns:
            A tuple of ``(audio_float32, item_id)`` where audio is at the
            session sample rate.
        """
        if not self._chunks:
            item_id = self._pending_item_id or f"item_{uuid.uuid4().hex[:24]}"
            self._reset_buffer()
            return np.empty(0, dtype=np.float32), item_id

        audio = np.concatenate(self._chunks)
        item_id = self._pending_item_id or f"item_{uuid.uuid4().hex[:24]}"
        self._reset_buffer()
        return audio, item_id

    def clear(self) -> None:
        """Discard all buffered audio without committing."""
        self._reset_buffer()
        self._vad.reset()

    def _reset_buffer(self) -> None:
        self._chunks.clear()
        self._total_samples = 0
        self._vad_buf = np.empty(0, dtype=np.float32)
        self._speech_start_sample = None
        self._pending_item_id = None
        self._vad.reset()


def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Simple linear interpolation resample."""
    if src_rate == dst_rate:
        return audio
    ratio = dst_rate / src_rate
    n_out = int(len(audio) * ratio)
    if n_out == 0:
        return np.empty(0, dtype=np.float32)
    indices = np.arange(n_out) / ratio
    indices = np.clip(indices, 0, len(audio) - 1)
    idx_floor = indices.astype(np.intp)
    idx_ceil = np.minimum(idx_floor + 1, len(audio) - 1)
    frac = (indices - idx_floor).astype(np.float32)
    return audio[idx_floor] * (1 - frac) + audio[idx_ceil] * frac
