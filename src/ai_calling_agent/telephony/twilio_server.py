"""FastAPI server that lets the agent make/receive real phone calls via
Twilio Programmable Voice + Media Streams.

Flow for an inbound call:
  1. Twilio hits POST /voice  -> we return TwiML that opens a bidirectional
     Media Stream websocket back to us.
  2. Twilio connects to WS /media and starts sending 8kHz mu-law audio
     frames as they come in on the call.
  3. We buffer frames, detect when the caller has paused (simple RMS VAD),
     run STT -> brain -> TTS on that utterance, and stream the reply's audio
     back to Twilio as outbound mu-law frames.
  4. Repeat until the call ends or an end-call phrase is heard.

Run with: python scripts/run_twilio_server.py
Needs a public URL (ngrok, or a real deployment) for Twilio to reach this
server — see README.md "Real phone calls (Twilio mode)".
"""

from __future__ import annotations

import asyncio
import audioop
import base64
import json
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from twilio.rest import Client as TwilioClient
from twilio.twiml.voice_response import Connect, VoiceResponse

from ..brain import ConversationBrain
from ..config import get_env, load_agent_config
from ..stt import SpeechToText
from ..tts import TextToSpeech

app = FastAPI(title="AI Calling Agent - Twilio bridge")

AGENT_CFG = load_agent_config()
# Load these once at startup, not per-call — the Whisper model load is the
# slow part (a few seconds), and every call would otherwise pay for it.
STT = SpeechToText(AGENT_CFG.stt)

TWILIO_SAMPLE_RATE = 8000
WHISPER_SAMPLE_RATE = 16000
FRAME_MS = 20  # Twilio sends/expects 20ms mu-law frames (160 bytes @ 8kHz)
FRAME_BYTES = int(TWILIO_SAMPLE_RATE * FRAME_MS / 1000)  # 160
SILENCE_RMS_THRESHOLD = 400  # on 16-bit PCM scale, tune against real calls
SILENCE_FRAMES_NEEDED = int(1.2 * 1000 / FRAME_MS)  # ~1.2s of silence ends a turn
MIN_SPEECH_FRAMES = int(0.4 * 1000 / FRAME_MS)


@app.post("/voice")
async def voice_webhook(request: Request) -> Response:
    """Twilio hits this when a call starts (inbound) or connects (outbound)."""
    public_base = get_env("PUBLIC_BASE_URL").rstrip("/")
    ws_url = public_base.replace("https://", "wss://").replace("http://", "ws://")
    vr = VoiceResponse()
    connect = Connect()
    connect.stream(url=f"{ws_url}/media")
    vr.append(connect)
    return Response(content=str(vr), media_type="application/xml")


def place_outbound_call(to_number: str) -> str:
    """Have the agent call `to_number`. Returns the Twilio Call SID."""
    account_sid = get_env("TWILIO_ACCOUNT_SID")
    auth_token = get_env("TWILIO_AUTH_TOKEN")
    from_number = get_env("TWILIO_FROM_NUMBER")
    public_base = get_env("PUBLIC_BASE_URL").rstrip("/")
    if not all([account_sid, auth_token, from_number, public_base]):
        raise RuntimeError(
            "Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER and "
            "PUBLIC_BASE_URL in .env before placing outbound calls."
        )
    client = TwilioClient(account_sid, auth_token)
    call = client.calls.create(
        to=to_number,
        from_=from_number,
        url=f"{public_base}/voice",
    )
    return call.sid


@dataclass
class CallState:
    stream_sid: str = ""
    brain: ConversationBrain = field(default_factory=lambda: ConversationBrain.create(AGENT_CFG))
    tts: TextToSpeech = field(default_factory=lambda: TextToSpeech(AGENT_CFG.voice))
    frame_buffer: list[bytes] = field(default_factory=list)
    consecutive_silence: int = 0
    speech_frame_count: int = 0
    started_speaking: bool = False


def mulaw_frame_to_pcm16(frame: bytes) -> bytes:
    return audioop.ulaw2lin(frame, 2)


def frame_rms(pcm16: bytes) -> float:
    if not pcm16:
        return 0.0
    return audioop.rms(pcm16, 2)


def pcm8k_to_whisper_input(pcm16_8k: bytes) -> np.ndarray:
    """8kHz 16-bit mono PCM bytes -> float32 mono array at 16kHz for Whisper."""
    pcm16_16k, _ = audioop.ratecv(pcm16_8k, 2, 1, TWILIO_SAMPLE_RATE, WHISPER_SAMPLE_RATE, None)
    audio = np.frombuffer(pcm16_16k, dtype=np.int16).astype(np.float32) / 32768.0
    return audio


def synthesize_to_mulaw_frames(tts: TextToSpeech, text: str) -> list[bytes]:
    """Text -> list of 160-byte mu-law frames at 8kHz, ready to stream to Twilio."""
    audio_path = tts.synthesize_to_file(text)
    data, samplerate = sf.read(audio_path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)  # downmix to mono

    pcm16 = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    pcm16_8k, _ = audioop.ratecv(pcm16, 2, 1, samplerate, TWILIO_SAMPLE_RATE, None)
    mulaw = audioop.lin2ulaw(pcm16_8k, 2)

    frames = [mulaw[i : i + FRAME_BYTES] for i in range(0, len(mulaw), FRAME_BYTES)]
    return frames


async def send_audio_to_twilio(ws: WebSocket, stream_sid: str, frames: list[bytes]) -> None:
    """Stream mu-law frames back to Twilio, paced at real-time (20ms/frame)."""
    for frame in frames:
        payload = base64.b64encode(frame).decode("ascii")
        await ws.send_text(
            json.dumps({"event": "media", "streamSid": stream_sid, "media": {"payload": payload}})
        )
        await asyncio.sleep(FRAME_MS / 1000)


async def handle_utterance(ws: WebSocket, state: CallState) -> bool:
    """Run STT -> brain -> TTS on the buffered utterance. Returns False to end the call."""
    pcm8k = b"".join(state.frame_buffer)
    state.frame_buffer.clear()
    state.consecutive_silence = 0
    state.speech_frame_count = 0
    state.started_speaking = False

    whisper_input = pcm8k_to_whisper_input(pcm8k)
    user_text = STT.transcribe_pcm(whisper_input)
    if not user_text:
        return True
    print(f"Caller: {user_text}")

    if state.brain.should_end_call(user_text):
        frames = synthesize_to_mulaw_frames(state.tts, "Alright, thanks for calling — take care!")
        await send_audio_to_twilio(ws, state.stream_sid, frames)
        return False

    reply = state.brain.respond(user_text)
    print(f"Agent: {reply}")
    frames = synthesize_to_mulaw_frames(state.tts, reply)
    await send_audio_to_twilio(ws, state.stream_sid, frames)
    return True


@app.websocket("/media")
async def media_stream(ws: WebSocket) -> None:
    await ws.accept()
    state = CallState()

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            event = msg.get("event")

            if event == "start":
                state.stream_sid = msg["start"]["streamSid"]
                opening = state.brain.opening_line()
                print(f"Agent: {opening}")
                frames = synthesize_to_mulaw_frames(state.tts, opening)
                await send_audio_to_twilio(ws, state.stream_sid, frames)

            elif event == "media":
                mulaw_frame = base64.b64decode(msg["media"]["payload"])
                pcm16 = mulaw_frame_to_pcm16(mulaw_frame)
                rms = frame_rms(pcm16)

                if rms > SILENCE_RMS_THRESHOLD:
                    state.started_speaking = True
                    state.consecutive_silence = 0
                    state.speech_frame_count += 1
                    state.frame_buffer.append(mulaw_frame)
                elif state.started_speaking:
                    state.consecutive_silence += 1
                    state.frame_buffer.append(mulaw_frame)
                    if state.consecutive_silence >= SILENCE_FRAMES_NEEDED:
                        if state.speech_frame_count >= MIN_SPEECH_FRAMES:
                            keep_going = await handle_utterance(ws, state)
                            if not keep_going:
                                await ws.close()
                                return
                        else:
                            state.frame_buffer.clear()
                            state.started_speaking = False
                            state.consecutive_silence = 0

            elif event == "stop":
                break

    except WebSocketDisconnect:
        pass
