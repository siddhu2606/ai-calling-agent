"""Run the FastAPI server that bridges real phone calls (via Twilio) to the
agent. See README.md "Real phone calls (Twilio mode)" for full setup
(Twilio number + ngrok + webhook config).

Usage:
    python scripts/run_twilio_server.py
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # safe on legacy cmd.exe codepages
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import uvicorn  # noqa: E402

if __name__ == "__main__":
    uvicorn.run(
        "ai_calling_agent.telephony.twilio_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
