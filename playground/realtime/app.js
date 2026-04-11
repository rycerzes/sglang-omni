/**
 * SGLang-Omni Realtime Voice Chat — WebSocket client + UI logic.
 *
 * Connects to the /v1/realtime WebSocket endpoint and drives a
 * continuous voice conversation with AudioWorklet-based capture/playback.
 */

(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);

  // DOM refs
  const connectBtn = $("connect-btn");
  const disconnectBtn = $("disconnect-btn");
  const statusDot = $("status-dot");
  const statusText = $("status-text");
  const transcript = $("transcript");
  const instructionsEl = $("instructions");
  const temperatureEl = $("temperature");
  const temperatureValueEl = $("temperature-value");
  const themeToggle = $("theme-toggle");
  const vadToggle = $("vad-toggle");
  const pushToTalkBtn = $("push-to-talk-btn");
  const inputLevel = $("input-level");
  const outputLevel = $("output-level");
  const queueInfo = $("queue-info");
  const clearBtn = $("clear-btn");

  // API base — allow override via injected global or same-origin.
  const API_BASE = window.SGLANG_OMNI_API_BASE || "";
  const WS_BASE = API_BASE.replace(/^http/, "ws") || `ws://${location.host}`;

  // State
  let ws = null;
  let audioCtx = null;
  let captureNode = null;
  let playbackNode = null;
  let micStream = null;
  let connected = false;
  let useServerVad = true;
  let pttActive = false;

  // ------------------------------------------------------------------
  // Theme
  // ------------------------------------------------------------------

  function getTheme() {
    return document.documentElement.getAttribute("data-theme") || "dark";
  }
  function setTheme(theme) {
    if (theme === "dark") {
      document.documentElement.removeAttribute("data-theme");
    } else {
      document.documentElement.setAttribute("data-theme", theme);
    }
    try { localStorage.setItem("sglang-rt-theme", theme); } catch (_) {}
  }
  try {
    const saved = localStorage.getItem("sglang-rt-theme");
    if (saved === "light") setTheme("light");
  } catch (_) {}
  if (themeToggle) {
    themeToggle.addEventListener("click", () => {
      setTheme(getTheme() === "dark" ? "light" : "dark");
    });
  }

  // ------------------------------------------------------------------
  // Transcript rendering
  // ------------------------------------------------------------------

  function appendTranscript(role, text, cls) {
    const div = document.createElement("div");
    div.className = `msg msg-${role}` + (cls ? ` ${cls}` : "");
    const label = document.createElement("span");
    label.className = "msg-role";
    label.textContent = role === "user" ? "You" : "Assistant";
    const content = document.createElement("span");
    content.className = "msg-text";
    content.textContent = text;
    div.appendChild(label);
    div.appendChild(content);
    transcript.appendChild(div);
    transcript.scrollTop = transcript.scrollHeight;
    return content;
  }

  // Currently streaming assistant message element (for appending deltas).
  let activeAssistantEl = null;
  let activeAssistantText = "";

  function startAssistantMessage() {
    activeAssistantText = "";
    activeAssistantEl = appendTranscript("assistant", "", "streaming");
  }

  function appendAssistantDelta(text) {
    if (!activeAssistantEl) startAssistantMessage();
    activeAssistantText += text;
    activeAssistantEl.textContent = activeAssistantText;
    transcript.scrollTop = transcript.scrollHeight;
  }

  function finishAssistantMessage() {
    if (activeAssistantEl) {
      activeAssistantEl.parentElement.classList.remove("streaming");
    }
    activeAssistantEl = null;
    activeAssistantText = "";
  }

  // ------------------------------------------------------------------
  // Audio setup
  // ------------------------------------------------------------------

  async function setupAudio() {
    audioCtx = new AudioContext({ sampleRate: 48000 });

    // Microphone
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });

    await audioCtx.audioWorklet.addModule("audio-capture-processor.js");
    await audioCtx.audioWorklet.addModule("audio-playback-processor.js");

    // Capture node: mic → worklet → PCM16 messages
    const source = audioCtx.createMediaStreamSource(micStream);
    captureNode = new AudioWorkletNode(audioCtx, "audio-capture-processor", {
      processorOptions: { targetRate: 24000, bufferSize: 2400 },
    });
    captureNode.port.onmessage = (e) => {
      if (e.data.type === "audio" && ws && ws.readyState === WebSocket.OPEN) {
        // Base64-encode the PCM16 buffer.
        const bytes = new Uint8Array(e.data.pcm16);
        const b64 = arrayBufferToBase64(bytes);
        ws.send(JSON.stringify({
          type: "input_audio_buffer.append",
          audio: b64,
        }));
        // Update input level meter.
        updateLevel(inputLevel, e.data.pcm16);
      }
    };
    source.connect(captureNode);
    captureNode.connect(audioCtx.destination); // required for worklet to run

    // Playback node
    playbackNode = new AudioWorkletNode(audioCtx, "audio-playback-processor", {
      processorOptions: { sourceRate: 24000 },
      outputChannelCount: [1],
    });
    playbackNode.port.onmessage = (e) => {
      if (e.data.type === "stats" && queueInfo) {
        queueInfo.textContent = `Buffer: ${Math.round(e.data.queuedMs)} ms`;
      }
    };
    playbackNode.connect(audioCtx.destination);
  }

  function teardownAudio() {
    if (captureNode) {
      captureNode.port.postMessage({ type: "stop" });
      captureNode.disconnect();
      captureNode = null;
    }
    if (playbackNode) {
      playbackNode.port.postMessage({ type: "stop" });
      playbackNode.disconnect();
      playbackNode = null;
    }
    if (micStream) {
      micStream.getTracks().forEach((t) => t.stop());
      micStream = null;
    }
    if (audioCtx) {
      audioCtx.close();
      audioCtx = null;
    }
  }

  // ------------------------------------------------------------------
  // WebSocket
  // ------------------------------------------------------------------

  async function connect() {
    if (connected) return;

    try {
      await setupAudio();
    } catch (err) {
      setStatus("error", "Microphone access denied");
      return;
    }

    setStatus("connecting", "Connecting...");
    ws = new WebSocket(`${WS_BASE}/v1/realtime`);

    ws.onopen = () => {
      connected = true;
      setStatus("connected", "Connected");
      // Send session config.
      ws.send(JSON.stringify({
        type: "session.update",
        session: {
          instructions: instructionsEl ? instructionsEl.value : "",
          temperature: parseFloat(temperatureEl ? temperatureEl.value : "0.7"),
          turn_detection: useServerVad
            ? { type: "server_vad", threshold: 0.5, silence_duration_ms: 700 }
            : null,
          modalities: ["text", "audio"],
        },
      }));
    };

    ws.onmessage = (e) => {
      let event;
      try { event = JSON.parse(e.data); } catch { return; }
      handleServerEvent(event);
    };

    ws.onerror = () => {
      setStatus("error", "Connection error");
    };

    ws.onclose = () => {
      connected = false;
      setStatus("disconnected", "Disconnected");
      teardownAudio();
    };
  }

  function disconnect() {
    if (ws) {
      ws.close();
      ws = null;
    }
    connected = false;
    setStatus("disconnected", "Disconnected");
    teardownAudio();
    finishAssistantMessage();
  }

  // ------------------------------------------------------------------
  // Server event handling
  // ------------------------------------------------------------------

  function handleServerEvent(event) {
    const t = event.type;

    if (t === "session.created" || t === "session.updated") {
      // Session confirmed.
      return;
    }

    if (t === "input_audio_buffer.speech_started") {
      // User started talking — interrupt playback.
      if (playbackNode) playbackNode.port.postMessage({ type: "clear" });
      return;
    }

    if (t === "input_audio_buffer.speech_stopped") {
      // Will be followed by committed + response.
      return;
    }

    if (t === "input_audio_buffer.committed") {
      appendTranscript("user", "(audio)", "audio-msg");
      return;
    }

    if (t === "response.audio.delta") {
      // Base64-encoded PCM16 audio chunk.
      if (event.delta && playbackNode) {
        const bytes = base64ToArrayBuffer(event.delta);
        // Update level BEFORE postMessage, which transfers (detaches) the buffer.
        updateLevel(outputLevel, bytes);
        playbackNode.port.postMessage({ type: "audio", pcm16: bytes }, [bytes]);
      }
      return;
    }

    if (t === "response.audio_transcript.delta") {
      appendAssistantDelta(event.delta || "");
      return;
    }

    if (t === "response.text.delta") {
      appendAssistantDelta(event.delta || "");
      return;
    }

    if (t === "response.done") {
      finishAssistantMessage();
      return;
    }

    if (t === "error") {
      const msg = event.error ? event.error.message : "Unknown error";
      appendTranscript("system", `Error: ${msg}`, "error-msg");
      return;
    }
  }

  // ------------------------------------------------------------------
  // Push-to-talk
  // ------------------------------------------------------------------

  function pttDown() {
    if (!connected || useServerVad) return;
    pttActive = true;
    pushToTalkBtn.classList.add("active");
  }

  function pttUp() {
    if (!connected || !pttActive) return;
    pttActive = false;
    pushToTalkBtn.classList.remove("active");
    // Commit the audio buffer and request a response.
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "input_audio_buffer.commit" }));
      ws.send(JSON.stringify({ type: "response.create" }));
    }
  }

  // ------------------------------------------------------------------
  // Helpers
  // ------------------------------------------------------------------

  function setStatus(state, text) {
    statusDot.className = "dot dot-" + state;
    statusText.textContent = text;
    connectBtn.disabled = state === "connected" || state === "connecting";
    disconnectBtn.disabled = state !== "connected";
    if (pushToTalkBtn) {
      pushToTalkBtn.disabled = state !== "connected" || useServerVad;
    }
  }

  function arrayBufferToBase64(buffer) {
    let binary = "";
    const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
  }

  function base64ToArrayBuffer(b64) {
    const binary = atob(b64);
    const bytes = new ArrayBuffer(binary.length);
    const view = new Uint8Array(bytes);
    for (let i = 0; i < binary.length; i++) {
      view[i] = binary.charCodeAt(i);
    }
    return bytes;
  }

  function updateLevel(el, pcm16Buffer) {
    if (!el) return;
    const int16 = new Int16Array(pcm16Buffer);
    let sum = 0;
    for (let i = 0; i < int16.length; i++) sum += Math.abs(int16[i]);
    const avg = sum / int16.length / 32768;
    const pct = Math.min(100, Math.round(avg * 400)); // scale for visibility
    el.style.width = pct + "%";
  }

  // ------------------------------------------------------------------
  // Event listeners
  // ------------------------------------------------------------------

  connectBtn.addEventListener("click", connect);
  disconnectBtn.addEventListener("click", disconnect);

  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      transcript.innerHTML = "";
      activeAssistantEl = null;
      activeAssistantText = "";
    });
  }

  if (temperatureEl && temperatureValueEl) {
    temperatureEl.addEventListener("input", () => {
      temperatureValueEl.textContent = parseFloat(temperatureEl.value).toFixed(2);
    });
  }

  if (vadToggle) {
    vadToggle.addEventListener("change", () => {
      useServerVad = vadToggle.checked;
      if (pushToTalkBtn) pushToTalkBtn.disabled = useServerVad || !connected;
      // Update session if connected.
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
          type: "session.update",
          session: {
            turn_detection: useServerVad
              ? { type: "server_vad", threshold: 0.5, silence_duration_ms: 700 }
              : null,
          },
        }));
      }
    });
  }

  if (pushToTalkBtn) {
    pushToTalkBtn.addEventListener("mousedown", pttDown);
    pushToTalkBtn.addEventListener("mouseup", pttUp);
    pushToTalkBtn.addEventListener("mouseleave", pttUp);
    pushToTalkBtn.addEventListener("touchstart", (e) => { e.preventDefault(); pttDown(); });
    pushToTalkBtn.addEventListener("touchend", (e) => { e.preventDefault(); pttUp(); });
  }

  // Initial state
  setStatus("disconnected", "Disconnected");
})();
