"""Run the agent locally through your mic + speakers. No phone/Twilio needed.

Usage:
    python scripts/local_demo.py
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # safe on legacy cmd.exe codepages
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_calling_agent.config import load_agent_config  # noqa: E402
from ai_calling_agent.conversation import ConversationLoop  # noqa: E402


def main() -> None:
    cfg = load_agent_config()
    print(f"Starting local demo as '{cfg.name}' running script '{cfg.active_script}'")
    print("Press Ctrl+C at any time to stop.\n")
    loop = ConversationLoop(cfg)
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nCall ended.")


if __name__ == "__main__":
    main()
