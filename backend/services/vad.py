from collections import deque
from typing import Optional, Tuple

import webrtcvad


class SpeechGate:
    """Energy/VAD gate that emits voiced chunks to feed downstream STT."""

    def __init__(
        self,
        sample_rate: int,
        aggressiveness: int,
        frame_ms: int,
        padding_ms: int,
    ) -> None:
        if aggressiveness not in (0, 1, 2, 3):
            raise ValueError("VAD aggressiveness must be between 0 and 3")

        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_bytes = int(sample_rate * frame_ms / 1000) * 2  # 16-bit mono
        self.vad = webrtcvad.Vad(aggressiveness)

        padding_frames = max(1, padding_ms // frame_ms)
        self.ring_buffer: deque[Tuple[bytes, bool]] = deque(maxlen=padding_frames)
        self.triggered = False
        self.voiced_frames: list[bytes] = []

    def process(self, frame: bytes) -> Optional[bytes]:
        if len(frame) != self.frame_bytes:
            return None

        is_speech = self.vad.is_speech(frame, self.sample_rate)
        self.ring_buffer.append((frame, is_speech))

        if not self.triggered:
            num_voiced = len([chunk for chunk in self.ring_buffer if chunk[1]])
            if num_voiced > 0.9 * self.ring_buffer.maxlen:
                self.triggered = True
                self.voiced_frames.extend(chunk for chunk, _ in self.ring_buffer)
                self.ring_buffer.clear()
        else:
            self.voiced_frames.append(frame)
            num_unvoiced = len([chunk for chunk in self.ring_buffer if not chunk[1]])
            if num_unvoiced > 0.9 * self.ring_buffer.maxlen:
                self.triggered = False
                voiced_audio = b"".join(self.voiced_frames)
                self.voiced_frames.clear()
                self.ring_buffer.clear()
                return voiced_audio

        return None

    def flush(self) -> Optional[bytes]:
        if self.voiced_frames:
            voiced_audio = b"".join(self.voiced_frames)
            self.voiced_frames.clear()
            self.ring_buffer.clear()
            self.triggered = False
            return voiced_audio
        return None
*** End of File