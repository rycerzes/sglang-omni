# SPDX-License-Identifier: Apache-2.0
"""Response controller — bridges realtime sessions to the pipeline."""

from __future__ import annotations

import asyncio
import base64
import logging
import uuid
from typing import Any, Callable, Coroutine

import numpy as np

from sglang_omni.client import Client, GenerateRequest, Message, SamplingParams
from sglang_omni.client.audio import audio_to_base64, to_numpy
from sglang_omni.serve.realtime import events
from sglang_omni.serve.realtime.conversation import ContentPart, ConversationItem

logger = logging.getLogger(__name__)


class ResponseController:
    """Drives a single response generation and emits realtime events.

    Each ``response.create`` client event creates one ``ResponseController``.
    The controller:
    1. Builds a ``GenerateRequest`` from the conversation history.
    2. Calls ``Client.generate()`` in a background task.
    3. Forwards audio / text deltas to the WebSocket via *send_event*.
    4. Supports cancellation (barge-in) via :meth:`cancel`.

    Args:
        client: The pipeline client.
        response_id: Unique response identifier.
        send_event: Async callback to push a server event dict to the WS.
        model: Model name for the request.
        instructions: System instructions.
        temperature: Sampling temperature.
        modalities: Output modalities list.
        max_tokens: Optional max output tokens.
        voice: Voice identifier.
    """

    def __init__(
        self,
        *,
        client: Client,
        response_id: str | None = None,
        send_event: Callable[[dict[str, Any]], Coroutine[Any, Any, None]],
        model: str = "sglang-omni",
        instructions: str = "",
        temperature: float = 0.7,
        modalities: list[str] | None = None,
        max_tokens: int | None = None,
        voice: str = "default",
        input_audio_sample_rate: int = 24000,
    ) -> None:
        self.response_id = response_id or f"resp_{uuid.uuid4().hex[:24]}"
        self._client = client
        self._send = send_event
        self._model = model
        self._instructions = instructions
        self._temperature = temperature
        self._modalities = modalities or ["text", "audio"]
        self._max_tokens = max_tokens
        self._voice = voice

        self._input_audio_sample_rate = input_audio_sample_rate
        self._request_id: str = ""
        self._task: asyncio.Task[None] | None = None
        self._cancelled = False
        self._done = False

        # Output item built during generation.
        self._output_item_id = f"item_{uuid.uuid4().hex[:24]}"
        self._text_accum = ""
        self._transcript_accum = ""
        self._audio_accum: list[np.ndarray] = []

    @property
    def is_active(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def done(self) -> bool:
        return self._done

    async def start(
        self,
        messages: list[dict[str, Any]],
        input_audios: list[Any] | None = None,
    ) -> None:
        """Begin generating a response for the given conversation history.

        Args:
            messages: Conversation messages.
            input_audios: Optional list of raw float32 numpy audio arrays
                from committed input audio buffers.  These are passed to
                the pipeline via ``GenerateRequest.metadata``.
        """
        self._request_id = uuid.uuid4().hex
        self._input_audios = input_audios
        self._task = asyncio.get_running_loop().create_task(
            self._run(messages), name=f"response-{self.response_id}"
        )

    async def cancel(self) -> None:
        """Cancel the in-flight response (barge-in)."""
        if self._cancelled or self._done:
            return
        self._cancelled = True
        if self._request_id:
            try:
                await self._client.abort(self._request_id)
            except Exception:
                logger.debug("Abort failed for %s", self._request_id, exc_info=True)
        if self._task and not self._task.done():
            self._task.cancel()

    def build_output_item(self) -> ConversationItem:
        """Build the output ConversationItem from accumulated data."""
        parts: list[ContentPart] = []
        if self._text_accum:
            parts.append(ContentPart(type="text", text=self._text_accum))
        if self._audio_accum:
            combined = np.concatenate(self._audio_accum)
            b64 = audio_to_base64(combined, sample_rate=24000, output_format="wav")
            parts.append(
                ContentPart(
                    type="audio", audio=b64, transcript=self._transcript_accum or None
                )
            )
        return ConversationItem(
            id=self._output_item_id,
            type="message",
            role="assistant",
            status="completed" if self._done and not self._cancelled else "incomplete",
            content=parts,
        )

    async def _run(self, messages: list[dict[str, Any]]) -> None:
        """Internal task that drives generation."""
        try:
            # Build request.
            msg_list = []
            if self._instructions:
                msg_list.append(Message(role="system", content=self._instructions))
            for m in messages:
                msg_list.append(Message(role=m["role"], content=m.get("content", "")))

            metadata: dict[str, Any] = {}
            if self._input_audios:
                # Tag with the session sample rate so the pipeline's
                # audio preprocessing can resample to the model's expected
                # rate (e.g. 16 kHz for Qwen3-Omni's Whisper encoder).
                metadata["audios"] = self._input_audios
                metadata["audio_sample_rate"] = self._input_audio_sample_rate

            request = GenerateRequest(
                model=self._model,
                messages=msg_list,
                sampling=SamplingParams(temperature=self._temperature),
                stream=True,
                max_tokens=self._max_tokens,
                output_modalities=self._modalities,
                metadata=metadata,
            )

            # Emit response.created.
            resp_dict = self._response_dict("in_progress")
            await self._send(events.response_created(resp_dict))

            # Emit output_item.added.
            out_item = {
                "id": self._output_item_id,
                "type": "message",
                "role": "assistant",
                "status": "in_progress",
                "content": [],
            }
            await self._send(
                events.response_output_item_added(self.response_id, 0, out_item)
            )

            # Emit content_part.added for each modality.
            content_index = 0
            if "text" in self._modalities:
                await self._send(
                    events.response_content_part_added(
                        self.response_id,
                        self._output_item_id,
                        0,
                        content_index,
                        {"type": "text", "text": ""},
                    )
                )
                text_content_idx = content_index
                content_index += 1
            else:
                text_content_idx = None

            if "audio" in self._modalities:
                await self._send(
                    events.response_content_part_added(
                        self.response_id,
                        self._output_item_id,
                        0,
                        content_index,
                        {"type": "audio", "audio": "", "transcript": ""},
                    )
                )
                audio_content_idx = content_index
                content_index += 1
            else:
                audio_content_idx = None

            # Stream from pipeline.
            async for chunk in self._client.generate(
                request, request_id=self._request_id
            ):
                if self._cancelled:
                    break

                if chunk.text and text_content_idx is not None:
                    self._text_accum += chunk.text
                    await self._send(
                        events.response_text_delta(
                            self.response_id,
                            self._output_item_id,
                            0,
                            text_content_idx,
                            chunk.text,
                        )
                    )

                if chunk.audio_data is not None and audio_content_idx is not None:
                    audio_np = to_numpy(chunk.audio_data)
                    self._audio_accum.append(audio_np)
                    b64 = base64.b64encode(
                        (audio_np * 32767).astype("<i2").tobytes()
                    ).decode("ascii")
                    await self._send(
                        events.response_audio_delta(
                            self.response_id,
                            self._output_item_id,
                            0,
                            audio_content_idx,
                            b64,
                        )
                    )

                # Emit transcript only when there is no dedicated text
                # modality — i.e. the output is audio-only and the model
                # also produces a textual representation of what it speaks.
                # When both text AND audio are requested (Qwen3-Omni default),
                # the text stream is already emitted as response.text.delta
                # above; sending it again here would double-count it.
                if (
                    chunk.modality == "text"
                    and chunk.text
                    and text_content_idx is None
                    and audio_content_idx is not None
                ):
                    self._transcript_accum += chunk.text
                    await self._send(
                        events.response_audio_transcript_delta(
                            self.response_id,
                            self._output_item_id,
                            0,
                            audio_content_idx,
                            chunk.text,
                        )
                    )

            # Emit done events.
            if not self._cancelled:
                if text_content_idx is not None:
                    await self._send(
                        events.response_text_done(
                            self.response_id,
                            self._output_item_id,
                            0,
                            text_content_idx,
                            self._text_accum,
                        )
                    )
                if audio_content_idx is not None:
                    await self._send(
                        events.response_audio_done(
                            self.response_id,
                            self._output_item_id,
                            0,
                            audio_content_idx,
                        )
                    )
                    if self._transcript_accum:
                        await self._send(
                            events.response_audio_transcript_done(
                                self.response_id,
                                self._output_item_id,
                                0,
                                audio_content_idx,
                                self._transcript_accum,
                            )
                        )

            out_item["status"] = "completed" if not self._cancelled else "incomplete"
            await self._send(
                events.response_output_item_done(self.response_id, 0, out_item)
            )

            status = "completed" if not self._cancelled else "cancelled"
            await self._send(
                events.response_done(self._response_dict(status))
            )

        except asyncio.CancelledError:
            await self._send(
                events.response_done(self._response_dict("cancelled"))
            )
        except Exception:
            logger.exception("Response %s failed", self.response_id)
            await self._send(
                events.response_done(self._response_dict("failed"))
            )
        finally:
            self._done = True

    def _response_dict(self, status: str) -> dict[str, Any]:
        return {
            "id": self.response_id,
            "status": status,
            "output": [{"id": self._output_item_id, "type": "message"}],
        }
