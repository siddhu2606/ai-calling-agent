"""Type instead of speak — talk to the agent's brain over plain text.

Useful for quickly checking your LLM API key / call script / objection
handling logic works before wiring up mic + speakers (or a real phone call).
No STT/TTS involved, so no audio devices are needed.

Usage:
    python scripts/text_chat_demo.py
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_calling_agent.brain import ConversationBrain  # noqa: E402
from ai_calling_agent.config import load_agent_config  # noqa: E402


def main() -> None:
    cfg = load_agent_config()
    print(f"LLM provider: {cfg.llm.provider}  model: {cfg.llm.model}  script: {cfg.active_script}")
    print("Type as the 'caller'. Ctrl+C to quit.\n")

    brain = ConversationBrain.create(cfg)
    opening = brain.opening_line()
    print(f"{cfg.name}: {opening}")

    while True:
        try:
            user_text = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nCall ended.")
            break
        if not user_text:
            continue
        if brain.should_end_call(user_text):
            print(f"{cfg.name}: Alright, thanks for your time, take care!")
            break
        reply = brain.respond(user_text)
        print(f"{cfg.name}: {reply}")


if __name__ == "__main__":
    main()
