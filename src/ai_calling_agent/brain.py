"""The LLM 'brain': builds a system prompt from the agent persona + the active
call script + objection-handling examples + global guidelines, then drives a
multi-turn conversation with either Claude or GPT.

This is the piece that replaces "training a model" — a good system prompt
with concrete few-shot examples gets you 90% of the way there for a hackathon
demo, in minutes instead of weeks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import AgentConfig, get_env, load_call_scripts


def _article(word: str) -> str:
    """'a' or 'an', picked from the word's first letter (good enough for role names)."""
    return "an" if word[:1].upper() in "AEIOU" else "a"


def build_system_prompt(cfg: AgentConfig, scripts: dict[str, Any]) -> str:
    script = scripts.get(cfg.active_script)
    if script is None:
        available = ", ".join(scripts.keys() - {"objection_handling", "global_guidelines"})
        raise KeyError(
            f"active_script '{cfg.active_script}' not found in call_scripts.yaml. "
            f"Available: {available}"
        )

    opening = script["opening_line"].format(agent_name=cfg.name, company=cfg.company)
    questions = "\n".join(f"  - {q}" for q in script.get("qualifying_questions", []))
    objections = "\n".join(
        f"  Caller: \"{o['objection']}\"\n  You: \"{o['response']}\""
        for o in scripts.get("objection_handling", [])
    )
    guidelines = "\n".join(f"  - {g}" for g in scripts.get("global_guidelines", []))
    role_article = _article(cfg.role)

    return f"""You are {cfg.name}, {role_article} {cfg.role} at {cfg.company}, on a live phone call.

GOAL FOR THIS CALL:
{script['goal'].strip()}

YOUR OPENING LINE (say this first, verbatim, then adapt naturally to what the caller says):
"{opening}"

QUESTIONS TO WORK IN NATURALLY (don't recite them as a checklist):
{questions}

WHEN THE CALLER PUSHES BACK, HANDLE IT LIKE THESE EXAMPLES (match the tone and brevity, don't copy verbatim):
{objections}

WRAP-UP LINE ONCE THE GOAL IS MET:
"{script['closing_line'].strip()}"

RULES YOU MUST FOLLOW EVERY TURN:
{guidelines}

Remember: you are speaking, not writing. Sound like a real person on a call:
short sentences, natural fillers are fine ("sure", "got it", "makes sense"),
no bullet points, no markdown, no emoji."""


@dataclass
class Turn:
    role: str  # "user" or "assistant"
    content: str


@dataclass
class ConversationBrain:
    cfg: AgentConfig
    system_prompt: str
    history: list[Turn] = field(default_factory=list)

    @classmethod
    def create(cls, cfg: AgentConfig) -> "ConversationBrain":
        scripts = load_call_scripts()
        system_prompt = build_system_prompt(cfg, scripts)
        return cls(cfg=cfg, system_prompt=system_prompt)

    def opening_line(self) -> str:
        """Return the scripted opening line and seed it into history."""
        scripts = load_call_scripts()
        script = scripts[self.cfg.active_script]
        opening = script["opening_line"].format(
            agent_name=self.cfg.name, company=self.cfg.company
        )
        self.history.append(Turn("assistant", opening))
        return opening

    def respond(self, user_text: str) -> str:
        """Feed the caller's transcribed speech in, get the agent's reply out."""
        self.history.append(Turn("user", user_text))
        reply = self._call_llm()
        self.history.append(Turn("assistant", reply))
        return reply

    def should_end_call(self, user_text: str) -> bool:
        lowered = user_text.lower()
        return any(phrase in lowered for phrase in self.cfg.behavior.end_call_phrases)

    def _call_llm(self) -> str:
        if self.cfg.llm.provider == "openai":
            return self._call_openai_compatible(api_key=get_env("OPENAI_API_KEY"))
        if self.cfg.llm.provider == "groq":
            # Groq serves an OpenAI-compatible API, just at a different base URL —
            # same client, same request shape, and it's very low-latency, which
            # matters a lot for a live voice call.
            return self._call_openai_compatible(
                api_key=get_env("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1"
            )
        return self._call_anthropic()

    def _call_anthropic(self) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=get_env("ANTHROPIC_API_KEY"))
        messages = [{"role": t.role, "content": t.content} for t in self.history]
        resp = client.messages.create(
            model=self.cfg.llm.model,
            system=self.system_prompt,
            messages=messages,
            max_tokens=self.cfg.llm.max_reply_tokens,
            temperature=self.cfg.llm.temperature,
        )
        return "".join(block.text for block in resp.content if block.type == "text").strip()

    def _call_openai_compatible(self, api_key: str, base_url: str | None = None) -> str:
        """Shared path for OpenAI and Groq — both speak the same chat.completions API."""
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url)
        messages = [{"role": "system", "content": self.system_prompt}]
        messages += [{"role": t.role, "content": t.content} for t in self.history]
        resp = client.chat.completions.create(
            model=self.cfg.llm.model,
            messages=messages,
            max_tokens=self.cfg.llm.max_reply_tokens,
            temperature=self.cfg.llm.temperature,
        )
        return resp.choices[0].message.content.strip()
