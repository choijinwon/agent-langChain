from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _path_from_env(name: str, default: str) -> Path:
    return Path(os.getenv(name, default)).expanduser().resolve()


@dataclass(frozen=True, slots=True)
class Settings:
    host: str = os.getenv("AGENT_HOST", "127.0.0.1")
    port: int = int(os.getenv("AGENT_PORT", "8080"))
    provider: str = os.getenv("AGENT_PROVIDER", "demo").lower()
    model: str = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
    api_key: str | None = os.getenv("OPENAI_API_KEY")
    base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    data_dir: Path = _path_from_env("AGENT_DATA_DIR", "./data")
    knowledge_dir: Path = _path_from_env("AGENT_KNOWLEDGE_DIR", "./knowledge")
    max_steps: int = int(os.getenv("AGENT_MAX_STEPS", "6"))

    def validate(self) -> None:
        if self.provider not in {"demo", "openai"}:
            raise ValueError("AGENT_PROVIDER must be 'demo' or 'openai'")
        if self.provider == "openai" and not self.api_key:
            raise ValueError("OPENAI_API_KEY is required when AGENT_PROVIDER=openai")
        if not 1 <= self.max_steps <= 20:
            raise ValueError("AGENT_MAX_STEPS must be between 1 and 20")

