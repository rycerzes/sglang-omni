/**
 * AudioWorklet processor for audio playback with jitter buffer.
 *
 * Receives PCM16 audio chunks from the main thread and plays them back
 * smoothly. Handles sample rate conversion and provides a small queue
 * to absorb network jitter.
 */

class AudioPlaybackProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.sourceRate = (options.processorOptions && options.processorOptions.sourceRate) || 24000;
    this.queue = []; // Array of Float32Array chunks (at output sampleRate)
    this.offset = 0; // Read position within queue[0]
    this.running = true;

    this.port.onmessage = (e) => {
      if (e.data.type === "audio") {
        // e.data.pcm16 is an ArrayBuffer of Int16 samples at sourceRate.
        const int16 = new Int16Array(e.data.pcm16);
        const float32 = new Float32Array(int16.length);
        for (let i = 0; i < int16.length; i++) {
          float32[i] = int16[i] / 32768;
        }
        // Resample from sourceRate to output sampleRate if needed.
        const resampled = this._resample(float32, this.sourceRate, sampleRate);
        this.queue.push(resampled);
      } else if (e.data.type === "clear") {
        this.queue = [];
        this.offset = 0;
      } else if (e.data.type === "stop") {
        this.running = false;
      }
    };
  }

  process(inputs, outputs) {
    if (!this.running) return false;

    const output = outputs[0];
    if (!output || !output[0]) return true;

    const channel = output[0];
    let written = 0;

    while (written < channel.length && this.queue.length > 0) {
      const chunk = this.queue[0];
      const available = chunk.length - this.offset;
      const needed = channel.length - written;
      const toCopy = Math.min(available, needed);

      channel.set(chunk.subarray(this.offset, this.offset + toCopy), written);
      written += toCopy;
      this.offset += toCopy;

      if (this.offset >= chunk.length) {
        this.queue.shift();
        this.offset = 0;
      }
    }

    // Fill remaining with silence.
    for (let i = written; i < channel.length; i++) {
      channel[i] = 0;
    }

    // Report queue depth to main thread for UI.
    if (currentFrame % 12000 === 0) { // ~every 250ms at 48kHz
      let totalSamples = -this.offset;
      for (const chunk of this.queue) totalSamples += chunk.length;
      this.port.postMessage({ type: "stats", queuedMs: (totalSamples / sampleRate) * 1000 });
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

registerProcessor("audio-playback-processor", AudioPlaybackProcessor);
