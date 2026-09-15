"""Loads .env and the YAML configs into simple, typed-ish Python objects."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT_DIR / "config"

load_dotenv(ROOT_DIR / ".env")


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class VoiceConfig:
    provider: str = "edge"
    edge_voice: str = "en-US-GuyNeural"
    elevenlabs_voice_id: str = ""
    elevenlabs_model: str = "eleven_turbo_v2_5"


@dataclass
class LLMConfig:
    provider: str = "anthropic"
    model: str = "claude-sonnet-5"
    temperature: float = 0.7
    max_reply_tokens: int = 200


@dataclass
class STTConfig:
    model_size: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "en"


@dataclass
class BehaviorConfig:
    max_turns: int = 30
    silence_timeout_seconds: float = 1.2
    barge_in: bool = True
    end_call_phrases: list[str] = field(default_factory=lambda: ["bye", "goodbye"])


@dataclass
class AgentConfig:
    name: str = "Alex"
    role: str = "AI Assistant"
    company: str = "Your Company"
    active_script: str = "sales_outbound"
    llm: LLMConfig = field(default_factory=LLMConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)


def load_agent_config(path: Path | None = None) -> AgentConfig:
    """Load config/agent_config.yaml into an AgentConfig."""
    path = path or (CONFIG_DIR / "agent_config.yaml")
    raw = _load_yaml(path)

    agent_raw = raw.get("agent", {})
    return AgentConfig(
        name=agent_raw.get("name", "Alex"),
        role=agent_raw.get("role", "AI Assistant"),
        company=agent_raw.get("company", "Your Company"),
        active_script=agent_raw.get("active_script", "sales_outbound"),
        llm=LLMConfig(**raw.get("llm", {})),
        voice=VoiceConfig(**raw.get("voice", {})),
        stt=STTConfig(**raw.get("stt", {})),
        behavior=BehaviorConfig(**raw.get("behavior", {})),
    )


def load_call_scripts(path: Path | None = None) -> dict[str, Any]:
    """Load config/call_scripts.yaml (scripts + objection handling + guidelines)."""
    path = path or (CONFIG_DIR / "call_scripts.yaml")
    return _load_yaml(path)


def get_env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)
