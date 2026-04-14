# Implementation Plan: Interactive Real-Time API

Addresses [Issue #59](https://github.com/sgl-project/sglang-omni/issues/59).

---

## 1. Problem Statement

SGLang-Omni currently exposes HTTP-based endpoints (`/v1/chat/completions`, `/v1/audio/speech`) that follow a **request-response** or **SSE streaming** pattern. Each turn is a discrete HTTP request — the user sends an audio file, the server processes it fully, and returns a response. There is no mechanism for:

- **Continuous bidirectional audio streaming** between client and server
- **Voice Activity Detection (VAD)** — the server cannot detect when the user starts/stops speaking
- **User interruption (barge-in)** — the user cannot interrupt the model mid-response
- **Session state** — no persistent conversation context across turns within a single connection
- **Low-latency turn-taking** — each turn pays the full HTTP request overhead

The goal is to implement a WebSocket-based (and eventually WebRTC-based) **Real-Time API** that enables continuous interactive voice conversations, modeled after [OpenAI's Realtime API](https://platform.openai.com/docs/guides/realtime).

---

## 2. Background & Literature Survey

### 2.1 Why a Real-Time API Is Needed

Qiu et al. [1] identify the central insight: **"realtime" comes not from any single fast model but from streaming and pipelining across components**. Their empirical evaluation demonstrates that Qwen2.5-Omni (a Level 2 speech-to-speech model similar to what SGLang-Omni serves) achieves ~13,200ms time-to-first-audio (TTFA) even with sentence-level streaming — far too slow for interactive conversation [1, Table 2]. The cascaded streaming approach (STT → LLM → TTS) achieves ~755ms measured TTFA [1, Table 3].

However, SGLang-Omni has a significant **architectural advantage** over the cascaded pipeline described in [1]: it runs native speech-to-speech models (Qwen3-Omni, Ming-Omni) with **inter-stage streaming** that overlaps the Thinker and Talker execution. This means SGLang-Omni can potentially achieve lower TTFA than either the pure cascaded approach or the naive native approach, because:

1. The Thinker streams hidden states to the Talker via `StreamQueue` (work-ahead)
2. The Talker begins audio generation before the Thinker finishes
3. Audio chunks are streamed to the client as they are generated

What is still missing is the **transport layer** — a persistent bidirectional connection that keeps the audio flowing in both directions simultaneously.

### 2.2 Speech-to-Speech Model Categories

Following the taxonomy in [1, Section 2.1]:

| Level | Description | Examples | SGLang-Omni Support |
|-------|-------------|----------|---------------------|
| **Level 1** | Truly native speech-to-speech | Moshi [2], dGSLM | Not supported |
| **Level 2** | Text LLM with native speech I/O | Qwen3-Omni, Ming-Omni, GLM-4-Voice [3] | **Primary focus** |
| **Level 3** | Cascaded pipeline (ASR → LLM → TTS) | Deepgram + vLLM + ElevenLabs | Supported via S2-Pro (TTS) |

SGLang-Omni targets Level 2 models primarily. The Real-Time API should be model-agnostic — it exposes the same WebSocket protocol regardless of whether the backend is a Level 2 native model or a Level 3 cascaded pipeline.

### 2.3 Voice Activity Detection (VAD)

VAD is essential for determining when the user has finished speaking, enabling automatic turn-taking. Qiu et al. [1, Section 5] use **Silero VAD** [4], a 2MB model that processes 32ms audio chunks in <1ms on CPU, implementing a state machine:

```
IDLE → [speech] → LISTENING → [silence 700ms] → PROCESSING → [LLM+TTS] → SPEAKING → [done] → IDLE
```

With an interruption path: `SPEAKING → [user speech] → INTERRUPTED → LISTENING`.

Silero VAD is also used by LiveKit Agents [5] and is the default VAD in Pipecat [6]. OpenAI's Realtime API offers two VAD modes [7]:

- **Server VAD**: Volume-based detection — detects speech start/end based on audio energy
- **Semantic VAD**: Model-based detection — uses a transformer to estimate whether the user has finished their turn, dynamically adjusting timeout

For Phase 1, we implement Server VAD with Silero. Semantic VAD can be explored later.

### 2.4 The OpenAI Realtime API Protocol

OpenAI's Realtime API [7][8] defines a comprehensive event-based protocol over WebSocket and WebRTC. Key design choices:

- **Session-scoped**: Each connection = one session with its own conversation state
- **Event-driven**: Both client and server communicate via typed JSON events
- **Input audio buffer**: Client streams audio chunks; the server manages a buffer with VAD
- **Conversation items**: Messages, function calls, and function outputs form a conversation
- **Response lifecycle**: Server generates responses that contain one or more items (text, audio, function calls)
- **Interruption via truncation**: When the user interrupts, the server cancels the current response and truncates unplayed audio via `conversation.item.truncate`

This protocol is becoming a de facto standard — LiveKit [5], Pipecat [6], and Twilio all integrate with it.

### 2.5 Open Source Reference Implementations

| Project | Stars | Transport | Architecture | Key Takeaway |
|---------|-------|-----------|-------------|--------------|
| [LiveKit Agents](https://github.com/livekit/agents) [5] | 10k | WebRTC (via LiveKit server) | Session-based, pluggable STT/LLM/TTS | Full-featured framework; uses Silero VAD + semantic turn detector; `@function_tool` decorator pattern |
| [Pipecat](https://github.com/pipecat-ai/pipecat) [6] | 11.1k | WebSocket, WebRTC (via Daily) | Frame-based pipeline | `SentenceAggregator` for streaming; 68+ service integrations; supports OpenAI Realtime natively |
| [OpenAI Realtime Console](https://github.com/openai/openai-realtime-console) [9] | 3.6k | WebRTC | Minimal reference client | Shows SDP negotiation, data channel events, audio track setup |
| [Salesforce Voice Agent](https://github.com/SalesforceAIResearch/enterprise-realtime-voice-agent) [1] | 4 | WebSocket | Educational from-scratch | 9-chapter progressive tutorial; Silero VAD; binary PCM protocol; AudioWorklet client |
| [Ultravox](https://github.com/fixie-ai/ultravox) [10] | 4.4k | Cloud API | Native multimodal LLM | Direct speech understanding without ASR; Llama 3.3 70B backbone |
| [Moshi](https://github.com/kyutai-labs/moshi) [2] | — | WebSocket | Full-duplex native S2S | Rust+CUDA server; only Level 1 model with a complete serving stack |

### 2.6 Latency Benchmarks from Literature

| System | TTFA (P50) | TTFA (Best) | Source |
|--------|-----------|-------------|--------|
| Qwen2.5-Omni (batch) | 26,500ms | — | [1, Table 2] |
| Qwen2.5-Omni (sentence streaming) | 13,200ms | — | [1, Table 2] |
| Cascaded: Deepgram + vLLM + ElevenLabs | 947ms | 729ms | [1, Table 3] |
| Cascaded: Deepgram + OpenAI + ElevenLabs | 958ms | 715ms | [1, Table 3] |
| Moshi (native, full-duplex) | ~200ms | — | [2] |
| LLaMA-Omni | ~226ms | — | [11] |

SGLang-Omni's inter-stage streaming should place it between the native models (~200ms) and the naive non-streaming approach (~13s). Empirical benchmarking with the Real-Time API will determine exact numbers.

---

## 3. Existing Architecture Analysis

### 3.1 What Already Exists (and Can Be Reused)

| Component | Location | Reusability |
|-----------|----------|-------------|
| **Multi-stage pipeline** | `sglang_omni/pipeline/` | ✅ Core infrastructure — unchanged |
| **Coordinator** submit/stream/abort | `sglang_omni/pipeline/coordinator.py` | ✅ `stream()` for streaming, `abort()` for interruption |
| **Inter-stage streaming** | `sglang_omni/pipeline/stage/stream_queue.py` | ✅ Already enables work-ahead between stages |
| **Request abort** | `Coordinator.abort()` → `AbortMessage` via ZMQ PUB/SUB | ✅ Propagates to all stages — perfect for barge-in |
| **Audio preprocessing** | `sglang_omni/preprocessing/audio.py` | ✅ WAV/MP3/FLAC/Opus decode to float32 |
| **Audio encoding** | `sglang_omni/client/audio.py` | ✅ Encode to WAV/MP3/Opus/PCM for output |
| **SSE streaming** | `sglang_omni/serve/openai_api.py` | 🔄 Pattern to follow, but WebSocket differs |
| **FastAPI app** | `sglang_omni/serve/openai_api.py` | ✅ Add WebSocket route to existing app |
| **Model pipelines** | `sglang_omni/models/qwen3_omni/`, `ming_omni/`, `fishaudio_s2_pro/` | ✅ Unchanged — Real-Time API is transport-layer only |
| **Request/Response types** | `sglang_omni/serve/protocol.py`, `sglang_omni/client/types.py` | 🔄 Extend with session/conversation types |

### 3.2 What Is Missing

| Component | Purpose | New Code Required |
|-----------|---------|-------------------|
| **WebSocket endpoint** | Persistent bidirectional connection | `sglang_omni/serve/realtime/api.py` |
| **Session manager** | Per-connection state (config, conversation, active response) | `sglang_omni/serve/realtime/session.py` |
| **Event protocol** | Typed client/server JSON events | `sglang_omni/serve/realtime/events.py` |
| **Input audio buffer** | Accumulate audio chunks, run VAD | `sglang_omni/serve/realtime/audio_buffer.py` |
| **Voice Activity Detection** | Detect speech start/end | `sglang_omni/serve/realtime/vad.py` |
| **Conversation state** | Track items (messages, function calls) | `sglang_omni/serve/realtime/conversation.py` |
| **Response controller** | Create, cancel, stream responses through pipeline | `sglang_omni/serve/realtime/response.py` |
| **Web client** | Browser-based real-time voice interface | `playground/realtime/` |

### 3.3 Data Flow: Current vs. Proposed

**Current (HTTP SSE):**
```
Client                        Server
  |                              |
  |-- POST /v1/chat/completions -->|
  |   (audio file + text)         |-- Coordinator.stream() -->|
  |                              |                    Pipeline stages
  |<---- SSE: text delta --------|<--- StreamMessage --|
  |<---- SSE: audio delta -------|<--- StreamMessage --|
  |<---- SSE: [DONE] ------------|<--- CompleteMessage |
  |                              |
  (connection closed)
```

**Proposed (WebSocket Real-Time):**
```
Client                                Server
  |                                      |
  |== WebSocket connect =================|
  |<-- session.created ------------------|
  |                                      |
  |-- input_audio_buffer.append -------->|  (continuous audio streaming)
  |-- input_audio_buffer.append -------->|
  |   ...                               |  VAD detects speech
  |<-- input_audio_buffer.speech_started |
  |-- input_audio_buffer.append -------->|
  |   ...                               |  VAD detects silence
  |<-- input_audio_buffer.speech_stopped |
  |<-- input_audio_buffer.committed -----|
  |<-- conversation.item.created --------|  (user message created)
  |                                      |
  |<-- response.created ----------------|  Coordinator.stream() starts
  |<-- response.audio.delta ------------|  Audio chunks from pipeline
  |<-- response.audio.delta ------------|
  |                                      |
  |-- input_audio_buffer.append -------->|  USER INTERRUPTS!
  |<-- input_audio_buffer.speech_started |
  |                                      |  Coordinator.abort() → all stages
  |<-- response.done (cancelled) --------|
  |                                      |
  |   ... (new turn begins) ...          |
```

---

## 4. Detailed Design

### 4.1 WebSocket Endpoint

```
GET /v1/realtime?model=<model_name>
Upgrade: websocket
```

Following OpenAI's convention [8], the model is specified as a query parameter. Authentication via `Authorization: Bearer <token>` header (when enabled).

The endpoint is registered alongside existing routes in the FastAPI app:

```python
# sglang_omni/serve/realtime/api.py
@app.websocket("/v1/realtime")
async def realtime_endpoint(websocket: WebSocket, model: str = Query(default=None)):
    await websocket.accept()
    session = RealtimeSession(websocket, pipeline_client, model)
    try:
        await session.run()
    finally:
        await session.cleanup()
```

### 4.2 Event Protocol

We adopt the OpenAI Realtime API event schema [7][8] for maximum compatibility. All events are JSON objects with a `type` field.

#### Client → Server Events

| Event Type | Description | Priority |
|-----------|-------------|----------|
| `session.update` | Update session config (instructions, voice, turn_detection, modalities) | P0 |
| `input_audio_buffer.append` | Append base64-encoded PCM audio chunk | P0 |
| `input_audio_buffer.commit` | Manually commit audio buffer (when VAD is disabled) | P0 |
| `input_audio_buffer.clear` | Clear audio buffer | P0 |
| `conversation.item.create` | Add a text or audio message to conversation | P1 |
| `conversation.item.delete` | Remove item from conversation | P2 |
| `conversation.item.truncate` | Truncate assistant audio at given ms offset | P1 |
| `response.create` | Trigger model response (manual mode or override) | P0 |
| `response.cancel` | Cancel in-progress response | P0 |

#### Server → Client Events

| Event Type | Description | Priority |
|-----------|-------------|----------|
| `session.created` | Emitted on connection; contains default session config | P0 |
| `session.updated` | Emitted after `session.update`; reflects new config | P0 |
| `error` | Error event with code, message, and optional event_id | P0 |
| `input_audio_buffer.speech_started` | VAD detected speech start | P0 |
| `input_audio_buffer.speech_stopped` | VAD detected speech end | P0 |
| `input_audio_buffer.committed` | Audio buffer was committed (creates user message) | P0 |
| `conversation.item.created` | New conversation item added | P1 |
| `conversation.item.done` | Conversation item finalized | P1 |
| `response.created` | Response generation started | P0 |
| `response.output_item.added` | New output item in response | P1 |
| `response.audio.delta` | Base64-encoded audio output chunk | P0 |
| `response.audio.done` | Audio output stream completed | P0 |
| `response.audio_transcript.delta` | Incremental transcript of audio output | P1 |
| `response.audio_transcript.done` | Transcript completed | P1 |
| `response.text.delta` | Incremental text output | P1 |
| `response.text.done` | Text output completed | P1 |
| `response.done` | Response completed (status: completed/cancelled/failed) | P0 |
| `rate_limits.updated` | Rate limit information | P2 |

### 4.3 Session State Machine

Each WebSocket connection manages a `RealtimeSession`:

```python
@dataclass
class RealtimeSession:
    session_id: str
    config: SessionConfig           # model, voice, instructions, modalities, turn_detection
    conversation: Conversation      # ordered list of ConversationItems
    input_audio_buffer: InputAudioBuffer  # accumulates PCM chunks + runs VAD
    active_response: Response | None      # currently generating response
    websocket: WebSocket
    pipeline_client: Client         # interface to the sglang-omni pipeline
```

**Session configuration:**
```python
@dataclass
class SessionConfig:
    model: str                           # e.g., "qwen3-omni"
    instructions: str = ""               # system prompt
    modalities: list[str] = ["text", "audio"]  # output modalities
    voice: str = "default"               # voice for audio output
    input_audio_format: str = "pcm16"    # pcm16 at 24kHz
    output_audio_format: str = "pcm16"   # pcm16 at 24kHz
    turn_detection: TurnDetectionConfig | None = TurnDetectionConfig()
    temperature: float = 0.7
    max_response_output_tokens: int | None = None
```

**Turn detection configuration:**
```python
@dataclass
class TurnDetectionConfig:
    type: str = "server_vad"             # "server_vad" or None for manual
    threshold: float = 0.5              # VAD speech probability threshold
    prefix_padding_ms: int = 300        # audio to include before speech start
    silence_duration_ms: int = 700      # silence duration to trigger end-of-turn
```

### 4.4 Voice Activity Detection

We use **Silero VAD v5** [4], consistent with the approach in [1, Section 5] and used by LiveKit [5]. Key properties:
- Model size: ~2MB (JIT-compiled PyTorch)
- Latency: <1ms per 32ms chunk on CPU
- Input: 16kHz mono PCM (512 samples per chunk for 32ms windows)
- Output: speech probability [0, 1] per chunk
- License: MIT

**VAD State Machine** (adapted from [1, Section 5]):
```
                    ┌──────────────────────────────┐
                    │                              │
                    ▼                              │
         ┌──────────────┐  speech detected   ┌────┴─────────┐
         │     IDLE     │ ────────────────→  │  LISTENING   │
         └──────────────┘                    └──────┬───────┘
                 ▲                                  │
                 │ done                             │ silence > threshold
                 │                                  ▼
         ┌──────┴───────┐                   ┌──────────────┐
         │   SPEAKING   │ ←──── response ── │  PROCESSING  │
         └──────┬───────┘    generated      └──────────────┘
                │
                │ user speech (BARGE-IN)
                ▼
         ┌──────────────┐
         │ INTERRUPTED  │ ──→ cancel response, then → LISTENING
         └──────────────┘
```

**Implementation:**
```python
class SileroVAD:
    """Silero VAD wrapper with state machine for turn detection."""
    
    def __init__(self, config: TurnDetectionConfig):
        self.model, _ = torch.hub.load('snakers4/silero-vad', 'silero_vad')
        self.threshold = config.threshold
        self.silence_duration_ms = config.silence_duration_ms
        self.prefix_padding_ms = config.prefix_padding_ms
        self.state = VADState.IDLE
        self._speech_start_sample = 0
        self._silence_start_ms = 0
    
    def process_chunk(self, audio: np.ndarray, sample_rate: int = 16000) -> list[VADEvent]:
        """Process a chunk of audio, return any state transition events."""
        prob = self.model(torch.from_numpy(audio), sample_rate).item()
        events = []
        
        if self.state == VADState.IDLE and prob >= self.threshold:
            self.state = VADState.LISTENING
            events.append(VADEvent.SPEECH_STARTED)
        elif self.state == VADState.LISTENING and prob < self.threshold:
            # Start silence timer
            if silence_exceeded(self.silence_duration_ms):
                self.state = VADState.PROCESSING
                events.append(VADEvent.SPEECH_STOPPED)
        elif self.state == VADState.SPEAKING and prob >= self.threshold:
            self.state = VADState.INTERRUPTED
            events.append(VADEvent.SPEECH_STARTED)  # barge-in!
        
        return events
```

### 4.5 Input Audio Buffer

The input audio buffer accumulates audio chunks from the client and feeds them to the VAD:

```python
class InputAudioBuffer:
    """Accumulates audio chunks and manages VAD."""
    
    def __init__(self, vad: SileroVAD, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.vad = vad
        self._chunks: list[np.ndarray] = []
        self._prefix_buffer: collections.deque  # circular buffer for prefix padding
    
    def append(self, base64_audio: str) -> list[VADEvent]:
        """Append base64-encoded PCM16 audio, run VAD, return events."""
        audio_bytes = base64.b64decode(base64_audio)
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Resample to 16kHz for VAD if needed
        vad_audio = resample(audio, self.sample_rate, 16000) if self.sample_rate != 16000 else audio
        
        self._chunks.append(audio)
        self._prefix_buffer.append(audio)
        
        return self.vad.process_chunk(vad_audio)
    
    def commit(self) -> np.ndarray:
        """Commit and return all buffered audio as a single array, then clear."""
        if not self._chunks:
            return np.array([], dtype=np.float32)
        full_audio = np.concatenate(self._chunks)
        self._chunks.clear()
        return full_audio
    
    def clear(self):
        """Discard all buffered audio."""
        self._chunks.clear()
```

### 4.6 Interruption (Barge-In) Flow

This is the most critical feature requested in [Issue #59](https://github.com/sgl-project/sglang-omni/issues/59). The flow leverages the **existing abort infrastructure** in the pipeline:

1. **Detection**: VAD detects `SPEECH_STARTED` while `active_response` is not None
2. **Cancel response**: Call `coordinator.abort(response_request_id)` — this broadcasts `AbortMessage` via ZMQ PUB/SUB to all pipeline stages
3. **Pipeline stops**: All stages receive the abort and stop processing (already implemented)
4. **Notify client**: Send `response.done` with `status=cancelled`
5. **Truncate conversation**: Remove unplayed audio from the last assistant message
6. **Continue listening**: VAD transitions to LISTENING, audio buffer accumulates new user speech
7. **New response**: When VAD detects silence, commit buffer as new user message and trigger a new response

The key insight is that **SGLang-Omni's abort mechanism already handles the hard part** — propagating cancellation through the multi-stage pipeline via ZMQ. The Real-Time API layer only needs to:
1. Detect when to abort (VAD during response generation)
2. Track how much audio was played (for truncation)
3. Manage conversation state updates

### 4.7 Conversation Management

The conversation tracks all items exchanged during the session:

```python
@dataclass
class ConversationItem:
    id: str
    type: str           # "message" | "function_call" | "function_call_output"
    role: str | None     # "system" | "user" | "assistant" (for messages)
    content: list[ContentPart]
    status: str = "completed"  # "in_progress" | "completed" | "truncated"

@dataclass
class Conversation:
    items: list[ConversationItem] = field(default_factory=list)
    
    def to_generate_request(self, config: SessionConfig) -> GenerateRequest:
        """Convert conversation history to a pipeline GenerateRequest."""
        messages = []
        media_paths = []
        for item in self.items:
            # Convert items to the message format expected by the pipeline
            ...
        return GenerateRequest(
            messages=messages,
            audios=media_paths,
            modalities=config.modalities,
            ...
        )
```

When a response is triggered, the entire conversation is converted to a `GenerateRequest` and submitted to the existing pipeline via `Coordinator.stream()`.

### 4.8 Response Lifecycle

```python
class ResponseController:
    """Manages the lifecycle of a model response."""
    
    async def create_response(self, session: RealtimeSession) -> None:
        """Create a new response from the current conversation state."""
        request_id = generate_request_id()
        gen_request = session.conversation.to_generate_request(session.config)
        
        session.active_response = Response(id=request_id, status="in_progress")
        await session.send_event("response.created", {...})
        
        try:
            async for msg in session.pipeline_client.stream(gen_request, request_id):
                if isinstance(msg, StreamMessage):
                    if msg.data.get("audio"):
                        await session.send_event("response.audio.delta", {
                            "delta": base64.b64encode(msg.data["audio"]).decode(),
                            "response_id": request_id,
                        })
                    if msg.data.get("text"):
                        await session.send_event("response.text.delta", {
                            "delta": msg.data["text"],
                            "response_id": request_id,
                        })
                elif isinstance(msg, CompleteMessage):
                    break
            
            session.active_response.status = "completed"
            await session.send_event("response.done", {...})
        
        except asyncio.CancelledError:
            session.active_response.status = "cancelled"
            await session.send_event("response.done", {"status": "cancelled"})
        
        finally:
            session.active_response = None
    
    async def cancel_response(self, session: RealtimeSession) -> None:
        """Cancel the active response (e.g., due to barge-in)."""
        if session.active_response:
            await session.pipeline_client.abort(session.active_response.id)
```

---

## 5. Implementation Plan

### Phase 1: WebSocket Real-Time API (Core) — MVP

**Goal**: A working WebSocket endpoint that supports continuous voice conversation with VAD and interruption.

#### New Files

| File | Purpose |
|------|---------|
| `sglang_omni/serve/realtime/__init__.py` | Package init |
| `sglang_omni/serve/realtime/api.py` | WebSocket endpoint and route registration |
| `sglang_omni/serve/realtime/session.py` | `RealtimeSession` — per-connection state machine |
| `sglang_omni/serve/realtime/events.py` | Event type definitions, serialization, validation |
| `sglang_omni/serve/realtime/audio_buffer.py` | `InputAudioBuffer` — audio accumulation + VAD integration |
| `sglang_omni/serve/realtime/vad.py` | `SileroVAD` — Silero VAD wrapper with state machine |
| `sglang_omni/serve/realtime/conversation.py` | `Conversation`, `ConversationItem` — conversation state |
| `sglang_omni/serve/realtime/response.py` | `ResponseController` — response lifecycle management |
| `sglang_omni/serve/realtime/protocol.py` | `SessionConfig`, `TurnDetectionConfig` — configuration schemas |

#### Modified Files

| File | Change |
|------|--------|
| `sglang_omni/serve/openai_api.py` | Register `/v1/realtime` WebSocket route via `register_realtime_routes(app, client, model)` |
| `pyproject.toml` | Add optional dependency: `silero-vad` or `onnxruntime` (for Silero ONNX variant) |

#### Task Breakdown

| # | Task | Est. Complexity | Dependencies |
|---|------|-----------------|--------------|
| 1 | Define event types and protocol schemas | Medium | None |
| 2 | Implement VAD wrapper (`vad.py`) | Medium | Silero VAD model |
| 3 | Implement input audio buffer (`audio_buffer.py`) | Medium | VAD (#2) |
| 4 | Implement conversation state management | Medium | Events (#1) |
| 5 | Implement response controller (bridge to pipeline) | High | Conversation (#4) |
| 6 | Implement session state machine | High | All above |
| 7 | Implement WebSocket endpoint | Medium | Session (#6) |
| 8 | Integrate with existing API server | Low | Endpoint (#7) |
| 9 | Write tests (unit + integration) | Medium | All above |
| 10 | Write documentation | Low | All above |

### Phase 2: Client & Frontend

**Goal**: A browser-based real-time voice chat interface.

#### New Files

| File | Purpose |
|------|---------|
| `playground/realtime/index.html` | HTML entry point |
| `playground/realtime/app.js` | WebSocket client, AudioWorklet, UI logic |
| `playground/realtime/audio-capture-processor.js` | AudioWorklet for microphone capture (PCM16 at 24kHz) |
| `playground/realtime/audio-playback-processor.js` | AudioWorklet for audio playback with jitter buffer |
| `playground/realtime/styles.css` | Styling |

Following the pattern in [1, Section 6.2], the browser client uses **AudioWorklet** processors for low-latency audio I/O:
- **Capture**: AudioWorkletProcessor captures mic audio at native sample rate, resamples to target rate, buffers into chunks, and posts PCM16 data for WebSocket transmission
- **Playback**: Queue-based processor receives audio chunks and plays them smoothly with proper sample rate handling
- **Echo cancellation**: Browser's built-in `echoCancellation: true` in `getUserMedia`

### Phase 3: WebRTC Support

**Goal**: WebRTC peer connection for browser clients, with native media tracks instead of base64-over-WebSocket.

| File | Purpose |
|------|---------|
| `sglang_omni/serve/realtime/webrtc.py` | WebRTC signaling and media handling via `aiortc` |
| `sglang_omni/serve/realtime/webrtc_api.py` | REST endpoint for SDP exchange (`/v1/realtime/calls`) |

WebRTC advantages over WebSocket for audio:
- **Built-in codecs**: Opus encoding/decoding handled by the browser and server
- **Jitter buffer**: WebRTC handles network jitter natively
- **Echo cancellation**: More robust AEC via WebRTC stack
- **NAT traversal**: ICE/STUN/TURN for connectivity through firewalls

However, WebRTC adds complexity (SDP negotiation, ICE candidates, media track management) and requires the `aiortc` library for Python. WebSocket is sufficient for server-to-server use cases.

### Phase 4: Advanced Features

| Feature | Description |
|---------|-------------|
| **Semantic VAD** | Model-based turn detection (vs. volume-based) for more natural conversations |
| **Function calling** | Tool use during real-time sessions (model calls functions, results feed back) |
| **Multi-modal input** | Image/video input during real-time sessions |
| **Conversation branching** | Out-of-band responses (OpenAI's `conversation: "none"` pattern) |
| **Session persistence** | Save/restore session state across connections |

### Phase 5: End-to-End Streaming Pipeline (Composable TTS)

*Aligns with [vllm-omni#2115](https://github.com/vllm-project/vllm-omni/issues/2115) Theme 1 (Streaming as a Core Feature) and Theme 2 (TTS as a Composable Layer).*

**Goal**: Enable true end-to-end streaming where LLM token generation feeds directly into TTS synthesis with minimal buffering, following the vision: `[Any text model] → [sentence chunker] → [TTS decoder] → audio`.

**Scope**: Initially targets **Qwen3-Omni** (native S2S, already has inter-stage streaming) and the **cascaded pipeline** (text LLM + edge-tts / S2-Pro). Other models are not guaranteed to work the same way — each TTS backend has different latency characteristics, chunking requirements, and streaming capabilities. New models are added one at a time after verifying compatibility.

| # | Task | Description |
|---|------|-------------|
| 1 | **Sentence chunking bridge stage** | Implement a `SentenceChunkerStage` between LLM decode and TTS. Accumulates tokens, detects sentence boundaries, emits complete sentences to TTS as they form. Follow the pattern from Pipecat's `SentenceAggregator` [6]. Initially wired into the cascaded pipeline config only. |
| 2 | **Streaming text input for S2-Pro TTS** | Extend the `/v1/audio/speech/stream` WebSocket endpoint (currently Qwen3-TTS only) to S2-Pro. Other TTS backends added later as their streaming interfaces are validated. |
| 3 | **Cascaded pipeline streaming** | Wire the sentence chunker into the cascaded pipeline so each sentence triggers an independent TTS synthesis, streaming audio back per-sentence instead of waiting for full LLM output. |
| 4 | **Voice/speaker parameter routing** | Design how voice and style parameters flow from the Real-Time API session config through the composable pipeline to the TTS backend. Scoped to edge-tts and S2-Pro initially. |

### Phase 6: Usability & Developer Experience

*Aligns with [vllm-omni#2115](https://github.com/vllm-project/vllm-omni/issues/2115) Theme 3 (Usability).*

**Goal**: Make deployment as simple as the models are small (consumer GPU friendly).

| # | Task | Description |
|---|------|-------------|
| 1 | **Voice management API** | Unified flow for voice upload, caching, and voice clone prompt creation. Upload once, use everywhere, persist across restarts. Scoped to S2-Pro and edge-tts initially. |
| 2 | **Auto-detection & zero-config** | Auto-detect model type from HuggingFace config for supported models (Qwen3-Omni, S2-Pro, cascaded). Auto-dispatch stage configs. Reduce required CLI arguments. |
| 3 | **TTS benchmarking** | Add benchmark support for `/v1/audio/speech` measuring TTFA, RTF, and throughput. Initially for Qwen3-Omni and S2-Pro. |
| 4 | **TTS model developer guide** | Document the process for adding a new TTS model: stage config, `async_chunk` integration, preprocessing hooks. Keep it concrete with existing models as examples rather than abstract. |

### Phase 7: Reliability & Quality Gates

*Aligns with [vllm-omni#2115](https://github.com/vllm-project/vllm-omni/issues/2115) Theme 5 (Reliability & Quality).*

**Goal**: Production-grade stability for supported models with automated quality regression detection.

| # | Task | Description |
|---|------|-------------|
| 1 | **Perceptual quality metrics in CI** | Add UTMOS (naturalness) and speaker embedding similarity (voice cloning) to the test suite for Qwen3-Omni and S2-Pro. Whisper WER alone misses audio quality regressions. |
| 2 | **Long-running stability tests** | Validate Qwen3-Omni and S2-Pro servers with 1000+ sequential requests. Profile memory in async chunk mode. Ensure graceful OOM handling on small GPUs. |
| 3 | **Voice cloning regression tests** | Dedicated quality tests for S2-Pro voice cloning — measure speaker similarity against reference embeddings, not just WER. |
| 4 | **Real-Time API load testing** | Concurrent WebSocket sessions with simulated audio. Measure TTFA under load, memory per session, max concurrent sessions. |

---

## 6. Audio Format & Encoding

### Wire Format

Following OpenAI's convention [8]:
- **Default**: PCM16 (signed 16-bit little-endian) at 24kHz, mono
- **Transport**: Base64-encoded in JSON events (WebSocket text frames)
- **Chunk size**: Client-determined, up to 15MB per event (recommended: 20-100ms chunks, i.e., 960-4800 bytes of raw PCM at 24kHz)

### Format Conversion

The pipeline models may expect different sample rates:
- **Qwen3-Omni**: 24kHz input/output (matches wire format)
- **Ming-Omni**: Model-specific rate
- **S2-Pro**: 44.1kHz output (needs downsampling for wire format)

Resampling is handled transparently by the audio buffer and response controller using `scipy.signal.resample` or `torchaudio.functional.resample`.

---

## 7. Concurrency & Resource Management

### Per-Session Resources

Each WebSocket connection holds:
- VAD model instance (~2MB GPU/CPU memory) — can be shared across sessions with a pool
- Audio buffer (variable, capped at configurable max duration, e.g., 60s = ~2.8MB at 24kHz PCM16)
- Conversation history (text + metadata, typically <1MB)
- One pipeline request slot (when response is active)

### Session Limits

- **Max concurrent sessions**: Configurable (default: 32)
- **Max session duration**: 60 minutes (matching OpenAI's limit [7])
- **Max audio buffer size**: 60 seconds of audio
- **Idle timeout**: Configurable (default: 5 minutes of no activity)

### Thread Safety

- Each session runs in its own asyncio task
- The pipeline coordinator is already thread-safe (ZMQ-based messaging)
- VAD model inference is CPU-bound — should use `asyncio.to_thread()` or a thread pool
- Audio buffer operations are session-local, no cross-session locking needed

---

## 8. Testing Strategy

### Unit Tests

| Test | File | What It Covers |
|------|------|----------------|
| VAD state transitions | `tests/test_realtime_vad.py` | Speech start/stop detection, silence timeout, interruption |
| Audio buffer operations | `tests/test_realtime_audio_buffer.py` | Append, commit, clear, PCM encoding/decoding |
| Event serialization | `tests/test_realtime_events.py` | JSON parsing, validation, round-trip |
| Conversation state | `tests/test_realtime_conversation.py` | Item CRUD, to_generate_request conversion |
| Session state machine | `tests/test_realtime_session.py` | State transitions, config updates |

### Integration Tests

| Test | File | What It Covers |
|------|------|----------------|
| WebSocket connection lifecycle | `tests/test_realtime_api.py` | Connect, session.created, session.update, disconnect |
| Text conversation | `tests/test_realtime_api.py` | Send text item → response.create → get text response |
| Audio round-trip | `tests/test_realtime_api.py` | Send audio chunks → VAD → response → audio output |
| Interruption | `tests/test_realtime_api.py` | Start response → send speech → verify cancellation |

### Load Tests

- Concurrent WebSocket sessions with simulated audio streams
- Measure TTFA under load
- Memory usage per session

---

## 9. Dependencies

### Required (Phase 1)

| Package | Version | Purpose | License |
|---------|---------|---------|---------|
| `websockets` or FastAPI built-in | — | WebSocket support (FastAPI has it via Starlette) | BSD |
| `silero-vad` | ≥5.0 | Voice Activity Detection | MIT |
| `numpy` | (existing) | Audio array operations | BSD |
| `torch` | (existing) | Silero VAD model runtime | BSD |

### Optional (Phase 3)

| Package | Version | Purpose | License |
|---------|---------|---------|---------|
| `aiortc` | ≥1.9 | WebRTC peer connection for Python | BSD |

**Note**: FastAPI already supports WebSocket natively via Starlette — no additional WebSocket library is needed. Silero VAD can be loaded via `torch.hub` or as a standalone JIT model file, avoiding a separate pip dependency.

---

## 10. Open Questions

1. **Should we maintain full OpenAI Realtime API compatibility, or define a simpler subset?**
   - Recommendation: Start with a compatible subset (P0 events only), expand as needed. Full compatibility means implementing function calling, MCP, SIP, etc. which are Phase 4 concerns.

2. **VAD: Server-side only, or allow client-side VAD?**
   - Recommendation: Server-side VAD by default (matching OpenAI's `server_vad`). Allow `turn_detection: null` for client-managed turn-taking (push-to-talk).

3. **Audio format: PCM16 only, or support Opus/G.711?**
   - Recommendation: PCM16 at 24kHz for Phase 1 (simplest, matches model expectations). Opus for Phase 3 (WebRTC uses Opus natively).

4. **How to handle multi-turn context for native S2S models?**
   - Native models (Qwen3-Omni) may not support appending arbitrary audio history the same way text LLMs do. Need to investigate whether the model can accept a sequence of audio turns or if each turn is independent.

5. **Should the Real-Time API bypass the HTTP server entirely?**
   - No — it should be a WebSocket route on the same FastAPI server. This simplifies deployment (single port) and allows shared configuration.

---

## 11. Success Criteria

- [ ] WebSocket connection to `/v1/realtime` with session lifecycle events
- [ ] Continuous audio streaming from client to server via `input_audio_buffer.append`
- [ ] Server-side VAD with `speech_started`/`speech_stopped` events
- [ ] Automatic response generation when VAD detects end-of-turn
- [ ] Streaming audio output via `response.audio.delta` events
- [ ] User interruption (barge-in) cancels active response and starts new turn
- [ ] Conversation context maintained across turns within a session
- [ ] Browser-based demo (playground/realtime/) showing interactive voice chat
- [ ] Sub-2-second TTFA for the full round-trip (audio in → audio out)
- [ ] Tests covering VAD, events, session state, and interruption flow

---

## References

[1] Qiu, J., Chen, Z., Yang, L., Zhu, M., Liu, Z., Tan, J., Zhao, W., Murthy, R., Ram, R., Prabhakar, A., et al. "Building Enterprise Realtime Voice Agents from Scratch: A Technical Tutorial." *arXiv preprint arXiv:2603.05413*, 2026. https://arxiv.org/html/2603.05413v1 — Code: https://github.com/SalesforceAIResearch/enterprise-realtime-voice-agent

[2] Défossez, A., Éliabel, L., et al. "Moshi: a speech-text foundation model for real-time dialogue." *Kyutai*, 2024. https://github.com/kyutai-labs/moshi

[3] GLM-4-Voice. *Zhipu AI*, 2024. Apache 2.0.

[4] Silero Team. "Silero VAD: pre-trained enterprise-grade Voice Activity Detector." 2021. https://github.com/snakers4/silero-vad — MIT License.

[5] LiveKit. "LiveKit Agents: A framework for building realtime voice AI agents." 2024. https://github.com/livekit/agents — Apache 2.0. 10k+ stars.

[6] Pipecat. "Open Source framework for voice and multimodal conversational AI." 2024. https://github.com/pipecat-ai/pipecat — BSD-2-Clause. 11.1k+ stars.

[7] OpenAI. "Realtime API." https://platform.openai.com/docs/guides/realtime — Accessed April 2026.

[8] OpenAI. "Realtime API Reference." https://platform.openai.com/docs/api-reference/realtime — Client events, server events, models.

[9] OpenAI. "Realtime Console." https://github.com/openai/openai-realtime-console — MIT License. Reference WebRTC client.

[10] Fixie AI. "Ultravox: A fast multimodal LLM for real-time voice." https://github.com/fixie-ai/ultravox — MIT License. 4.4k+ stars.

[11] Fang, Q., et al. "LLaMA-Omni: Seamless Speech Interaction with Large Language Models." 2024. arXiv:2409.06666.
