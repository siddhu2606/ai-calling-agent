"""Have the agent call a real phone number (outbound).

The Twilio server (scripts/run_twilio_server.py) must already be running and
publicly reachable (PUBLIC_BASE_URL in .env) before you run this.

Usage:
    python scripts/place_call.py +15551234567
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_calling_agent.telephony.twilio_server import place_outbound_call  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/place_call.py +15551234567")
        sys.exit(1)

    sid = place_outbound_call(sys.argv[1])
    print(f"Call placed. SID: {sid}")
