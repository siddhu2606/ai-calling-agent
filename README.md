# 🤖 AI Calling Agent

A voice AI agent that **talks like a real person on the phone** — pauses naturally, responds in real time, and handles objections — built for hackathons using an orchestration of pre-trained models (not a from-scratch trained model; see "How this actually works" below).

Two ways to run it:
1. **Local demo mode** — talk to it through your laptop mic/speakers. No phone account needed. Great for a quick hackathon demo table.
2. **Real phone call mode** — plugs into Twilio so it can actually make/receive phone calls.

## How this actually works (read this before your demo/pitch)

Nobody trains a voice model or an LLM from scratch for a hackathon — that needs huge datasets and weeks of GPU time. What actually gets built (and what the viral "AI cold-calling" demos are doing under the hood) is a **pipeline of pre-trained models glued together with your own logic and prompts**:

```
   Your voice                                                    Agent's voice
       │                                                               ▲
       ▼                                                               │
 ┌───────────┐      ┌────────────────┐      ┌───────────────┐   ┌───────────┐
 │    STT    │ ───▶ │   LLM "brain"  │ ───▶ │   text reply   │──▶│    TTS    │
 │ (Whisper) │      │ (Claude/GPT +  │      │                │   │ (voice)   │
 │  local    │      │  your scripts) │      │                │   │           │
 └───────────┘      └────────────────┘      └───────────────┘   └───────────┘
```

- **STT (ears)**: `faster-whisper`, running fully local/offline — free, no API key.
- **Brain**: Claude or GPT, given a system prompt built from *your* persona + call scripts + objection-handling examples (`config/call_scripts.yaml`). This is the "trained with 5 call scripts" step from the demo you heard — it's prompt engineering / few-shot examples, not fine-tuning, and it's genuinely effective.
- **TTS (voice)**: `edge-tts` by default — completely free, no API key, natural-sounding neural voices. Swap in **ElevenLabs** for a *cloned* voice (upload ~1 min of a real voice sample → get a custom `voice_id`) — this is the one place you can legitimately say "we trained a custom voice."
- **Telephony**: Twilio Programmable Voice + Media Streams for real inbound/outbound phone calls.

This is exactly the architecture behind tools like Vapi/Retell/Bland — and it's the correct scope for a hackathon: a real, working, demo-able system in hours, not a research project.

## Quickstart (local demo — do this first)

```bash
cd ai-calling-agent
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt

copy .env.example .env
# edit .env and add GROQ_API_KEY (or ANTHROPIC_API_KEY / OPENAI_API_KEY,
# matching whichever provider is set in config/agent_config.yaml -> llm.provider)

python scripts\local_demo.py
```

Speak into your mic after "Listening...". The agent transcribes, thinks, replies out loud, and loops — a full spoken conversation, entirely on your laptop.

**No mic yet, or just testing your API key / call script?**
```bash
python scripts\text_chat_demo.py
```
Same brain, plain text in/out, no audio devices touched.

## Real phone calls (Twilio mode)

1. Create a free Twilio account → get a phone number with Voice capability.
2. `pip install -r requirements.txt` (already includes `fastapi`, `uvicorn`, `twilio`).
3. Expose your local server publicly (needed for Twilio to reach your websocket):
   ```bash
   ngrok http 8000
   ```
4. In the Twilio number's config, set the **Voice webhook** to `https://<your-ngrok-domain>/voice` (HTTP POST).
5. Run the server:
   ```bash
   python scripts\run_twilio_server.py
   ```
6. Call your Twilio number — you're now talking to the agent over a real phone call.

Outbound calling (agent calls a number) is in `src/ai_calling_agent/telephony/twilio_server.py::place_outbound_call` — needs `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER` in `.env`.

## Customizing for your hackathon

- **Persona / voice / LLM settings** → `config/agent_config.yaml`
- **Call scripts & objection handling** → `config/call_scripts.yaml` (add your own scenario — sales, support, appointment booking, event registration, whatever your hackathon track needs)
- **Clone a real voice** → get an ElevenLabs API key, upload a sample clip, put the `voice_id` in `agent_config.yaml`, set `voice.provider: elevenlabs`

## Project layout

```
config/
  agent_config.yaml     persona, LLM provider/model, voice settings
  call_scripts.yaml     scripts + objection-handling examples (the "training data")
src/ai_calling_agent/
  stt.py                speech-to-text (faster-whisper, local)
  tts.py                text-to-speech (edge-tts free / ElevenLabs cloned voice)
  brain.py              LLM conversation logic, builds prompt from config
  conversation.py       turn-taking loop: listen -> think -> speak
  telephony/
    twilio_server.py    FastAPI app for real phone calls via Twilio Media Streams
scripts/
  local_demo.py         run the agent via your mic/speakers
  run_twilio_server.py  run the phone-call server
tests/
  test_brain.py         sanity checks for prompt building (no API key needed)
```

## Known limits (be upfront about these when judges ask)

- Latency: local Whisper + non-streaming TTS adds ~1-3s per turn. Fine for a demo; a production system would stream STT/TTS instead of waiting for full utterances.
- Interruption/barge-in ("stop talking, let me speak") is stubbed in `conversation.py` but not fully wired — a good "future work" line for your pitch.
- Twilio mode needs a public URL (ngrok) during local dev — deploy the FastAPI server somewhere reachable (Render/Railway/Fly.io) to avoid that for the actual hackathon demo.

## License

MIT — do whatever you want with it for your hackathons.
