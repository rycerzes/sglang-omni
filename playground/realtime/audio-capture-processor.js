/**
 * AudioWorklet processor for microphone capture.
 *
 * Captures audio from the microphone at the browser's native sample rate,
 * resamples to the target rate (default 24 kHz), and posts PCM16 chunks
 * to the main thread for WebSocket transmission.
 */

class AudioCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.targetRate = (options.processorOptions && options.processorOptions.targetRate) || 24000;
    this.bufferSize = (options.processorOptions && options.processorOptions.bufferSize) || 2400; // 100ms at 24kHz
    this.buffer = new Float32Array(0);
    this.running = true;

    this.port.onmessage = (e) => {
      if (e.data.type === "stop") {
        this.running = false;
      }
    };
  }

  process(inputs) {
    if (!this.running) return false;

    const input = inputs[0];
    if (!input || !input[0] || input[0].length === 0) return true;

    const raw = input[0]; // mono channel
    const resampled = this._resample(raw, sampleRate, this.targetRate);

    // Append to buffer.
    const next = new Float32Array(this.buffer.length + resampled.length);
    next.set(this.buffer);
    next.set(resampled, this.buffer.length);
    this.buffer = next;

    // Emit complete chunks.
    while (this.buffer.length >= this.bufferSize) {
      const chunk = this.buffer.slice(0, this.bufferSize);
      this.buffer = this.buffer.slice(this.bufferSize);

      // Convert float32 [-1, 1] to PCM16 little-endian.
      const pcm16 = new Int16Array(chunk.length);
      for (let i = 0; i < chunk.length; i++) {
        const s = Math.max(-1, Math.min(1, chunk[i]));
        pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      this.port.postMessage({ type: "audio", pcm16: pcm16.buffer }, [pcm16.buffer]);
    }

    return true;
  }

  _resample(input, fromRate, toRate) {
    if (fromRate === toRate) return input;
    const ratio = toRate / fromRate;
    const outLen = Math.round(input.length * ratio);
    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const srcIdx = i / ratio;
      const lo = Math.floor(srcIdx);
      const hi = Math.min(lo + 1, input.length - 1);
      const frac = srcIdx - lo;
      out[i] = input[lo] * (1 - frac) + input[hi] * frac;
    }
    return out;
  }
}

registerProcessor("audio-capture-processor", AudioCaptureProcessor);
