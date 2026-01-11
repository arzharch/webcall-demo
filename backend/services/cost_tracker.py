from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from backend.config import Settings, get_settings


@dataclass
class CostTracker:
    """Tracks token/audio usage and estimates vendor costs."""

    settings: Settings = field(default_factory=get_settings)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    stt_seconds: float = 0.0
    tts_characters: int = 0

    def add_llm_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens += max(0, prompt_tokens)
        self.completion_tokens += max(0, completion_tokens)

    def add_stt_seconds(self, seconds: float) -> None:
        self.stt_seconds += max(0.0, seconds)

    def add_tts_characters(self, characters: int) -> None:
        self.tts_characters += max(0, characters)

    def snapshot(self) -> Dict[str, float]:
        llm_in_cost = (
            self.prompt_tokens / 1000.0 * self.settings.LLM_INPUT_COST_PER_1K_TOKENS
        )
        llm_out_cost = (
            self.completion_tokens / 1000.0 * self.settings.LLM_OUTPUT_COST_PER_1K_TOKENS
        )
        stt_cost = (self.stt_seconds / 60.0) * self.settings.STT_COST_PER_MINUTE
        tts_cost = (
            self.tts_characters / 1_000_000.0 * self.settings.TTS_COST_PER_MILLION_CHARS
        )

        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "stt_seconds": self.stt_seconds,
            "tts_characters": self.tts_characters,
            "estimated_usd": round(llm_in_cost + llm_out_cost + stt_cost + tts_cost, 4),
            "llm_cost_usd": round(llm_in_cost + llm_out_cost, 4),
            "stt_cost_usd": round(stt_cost, 4),
            "tts_cost_usd": round(tts_cost, 4),
        }

    def reset(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.stt_seconds = 0.0
        self.tts_characters = 0