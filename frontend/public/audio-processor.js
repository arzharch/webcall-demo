// audio-processor.js
class AudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.port.onmessage = (event) => {
      // Handle messages from the main thread if needed
    };
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (input.length > 0) {
      const pcmData = new Int16Array(input[0].length);
      for (let i = 0; i < input[0].length; i++) {
        pcmData[i] = Math.max(-1, Math.min(1, input[0][i])) * 0x7FFF;
      }
      this.port.postMessage(pcmData.buffer, [pcmData.buffer]);
    }
    return true;
  }
}

registerProcessor('audio-processor', AudioProcessor);
