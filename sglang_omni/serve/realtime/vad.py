# SPDX-License-Identifier: Apache-2.0
"""Silero VAD wrapper with a state machine for turn detection.

Uses `snakers4/silero-vad` (MIT licence, ~2 MB JIT model).  The model
processes 32 ms chunks at 16 kHz and returns a speech probability in [0, 1].
"""

from __future__ import annotations

import enum
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_vad_model: Any = None
_vad_utils: Any = None


def _load_silero_vad() -> Any:
    """Lazily load the Silero VAD JIT model.

    Returns the model callable.  Raises ``ImportError`` if ``torch`` is not
    available.
    """
    global _vad_model, _vad_utils
    if _vad_model is not None:
        return _vad_model

    import torch  # noqa: F811

    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )
    _vad_model = model
    _vad_utils = utils
    logger.info("Silero VAD model loaded")
    return _vad_model


class VADState(enum.Enum):
    """Voice Activity Detector state."""

    IDLE = "idle"
    LISTENING = "listening"


class VADEvent(enum.Enum):
    SPEECH_STARTED = "speech_started"
    SPEECH_STOPPED = "speech_stopped"


class SileroVAD:
    """Lightweight Silero VAD wrapper with silence timeout.

    Parameters
    ----------
    threshold:
        Speech probability above which a chunk is considered speech.
    silence_duration_ms:
        Consecutive silence required to emit ``SPEECH_STOPPED``.
    prefix_padding_ms:
        (Informational) how much audio before detected speech start the
        caller should keep.  This class does not buffer audio itself.
    sample_rate:
        Expected input sample rate (must be 8000 or 16000 for Silero).
    """

    SUPPORTED_RATES = (8000, 16000)
    # Silero expects 512 samples at 16 kHz → 32 ms windows.
    CHUNK_SAMPLES_16K = 512

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        silence_duration_ms: int = 700,
        prefix_padding_ms: int = 300,
        sample_rate: int = 16000,
    ) -> None:
        if sample_rate not in self.SUPPORTED_RATES:
            raise ValueError(
                f"Silero VAD supports sample rates {self.SUPPORTED_RATES}, "
                f"got {sample_rate}"
            )
        self._threshold = threshold
        self._silence_duration_ms = silence_duration_ms
        self.prefix_padding_ms = prefix_padding_ms
        self._sample_rate = sample_rate
        self._state = VADState.IDLE

        # Silence tracking
        self._silence_start_sample: int | None = None
        self._total_samples: int = 0

        self._model: Any = None

    @property
    def state(self) -> VADState:
        return self._state

    def reset(self) -> None:
        """Reset the state machine."""
        self._state = VADState.IDLE
        self._silence_start_sample = None
        self._total_samples = 0
        if self._model is not None:
            self._model.reset_states()

    def process_chunk(self, audio: np.ndarray) -> list[VADEvent]:
        """Feed a chunk of float32 audio and return any triggered events.

        *audio* must be mono float32 at ``self._sample_rate``.  The chunk
        length should ideally be 512 samples (16 kHz) for best accuracy.
        """
        if self._model is None:
            self._model = _load_silero_vad()

        import torch

        tensor = torch.from_numpy(audio).float()
        prob: float = self._model(tensor, self._sample_rate).item()

        events: list[VADEvent] = []
        prev_total = self._total_samples
        self._total_samples += len(audio)

        is_speech = prob >= self._threshold

        if self._state == VADState.IDLE:
            if is_speech:
                self._state = VADState.LISTENING
                self._silence_start_sample = None
                events.append(VADEvent.SPEECH_STARTED)
        elif self._state == VADState.LISTENING:
            if is_speech:
                # Still speaking — reset silence timer
                self._silence_start_sample = None
            else:
                # Silence chunk
                if self._silence_start_sample is None:
                    self._silence_start_sample = prev_total
                elapsed_silence_ms = (
                    (self._total_samples - self._silence_start_sample)
                    * 1000
                    // self._sample_rate
                )
                if elapsed_silence_ms >= self._silence_duration_ms:
                    self._state = VADState.IDLE
                    self._silence_start_sample = None
                    events.append(VADEvent.SPEECH_STOPPED)

        return events

    @property
    def total_ms(self) -> int:
        """Total audio processed in milliseconds."""
        return self._total_samples * 1000 // self._sample_rate
