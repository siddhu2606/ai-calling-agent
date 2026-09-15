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
import os
import traceback
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
# Override via VAD_RMS_THRESHOLD in .env without a code change/redeploy while calibrating.
SILENCE_RMS_THRESHOLD = float(get_env("VAD_RMS_THRESHOLD", "400"))
SILENCE_FRAMES_NEEDED = int(1.2 * 1000 / FRAME_MS)  # ~1.2s of silence ends a turn
RMS_SMOOTHING_FRAMES = 3  # ~60ms rolling average -- absorbs brief line noise/crackle
# so a single noisy frame doesn't reset the silence counter and stall turn-end detection
MIN_SPEECH_FRAMES = int(0.6 * 1000 / FRAME_MS)  # was 0.4s -- too eager, tripped on breath/line noise


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
    recent_rms: list[float] = field(default_factory=list)  # small rolling window, see frame_rms smoothing
    total_frames_seen: int = 0
    max_rms_seen: float = 0.0


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


async def synthesize_to_mulaw_frames(tts: TextToSpeech, text: str) -> list[bytes]:
    """Text -> list of 160-byte mu-law frames at 8kHz, ready to stream to Twilio."""
    audio_path = await tts.asynthesize_to_file(text)
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        # Seen once against a live Twilio call: edge-tts returned without error
        # but produced no audio (a transient hiccup talking to its backend).
        # Fail loudly and specifically here rather than letting soundfile raise
        # a cryptic "File does not exist (possibly a pipe?)" a layer down.
        raise RuntimeError(f"TTS produced no audio file for text: {text!r}")

    data, samplerate = sf.read(audio_path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)  # downmix to mono

    pcm16 = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    pcm16_8k, _ = audioop.ratecv(pcm16, 2, 1, samplerate, TWILIO_SAMPLE_RATE, None)
    mulaw = audioop.lin2ulaw(pcm16_8k, 2)

    frames = [mulaw[i : i + FRAME_BYTES] for i in range(0, len(mulaw), FRAME_BYTES)]
    return frames


async def speak_safe(ws: WebSocket, state: CallState, text: str, retries: int = 1) -> bool:
    """Synthesize + send `text`, retrying once on failure instead of killing the
    call. Returns False (call continues) if every attempt failed -- the caller
    just stays silent for that turn rather than the whole handler crashing."""
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            frames = await synthesize_to_mulaw_frames(state.tts, text)
            await send_audio_to_twilio(ws, state.stream_sid, frames)
            return True
        except Exception as e:  # noqa: BLE001 - deliberately broad: never crash a live call
            last_error = e
            print(f"[speak_safe] attempt {attempt + 1} failed: {e!r}")
    print(f"[speak_safe] giving up after {retries + 1} attempts: {last_error!r}")
    return False


async def send_audio_to_twilio(ws: WebSocket, stream_sid: str, frames: list[bytes]) -> None:
    """Stream mu-law frames back to Twilio, paced at real-time (20ms/frame).

    Confirmed live: sleeping a fixed 20ms *after* each frame lets the per-frame
    base64/JSON/send overhead accumulate as drift, so actual delivery slowly
    falls behind real-time -- Twilio's playout buffer then periodically runs
    dry, which is heard as choppy/cutting audio. Pace against an absolute
    clock instead: compute each frame's exact scheduled time up front and only
    sleep however long is left to reach it (0 if we're already behind), so
    per-frame overhead never compounds across the whole utterance.
    """
    frame_interval = FRAME_MS / 1000
    loop = asyncio.get_event_loop()
    start = loop.time()
    for i, frame in enumerate(frames):
        payload = base64.b64encode(frame).decode("ascii")
        await ws.send_text(
            json.dumps({"event": "media", "streamSid": stream_sid, "media": {"payload": payload}})
        )
        target_time = start + (i + 1) * frame_interval
        remaining = target_time - loop.time()
        if remaining > 0:
            await asyncio.sleep(remaining)


async def handle_utterance(ws: WebSocket, state: CallState) -> bool:
    """Run STT -> brain -> TTS on the buffered utterance. Returns False to end the call."""
    pcm8k = b"".join(state.frame_buffer)
    state.frame_buffer.clear()
    state.consecutive_silence = 0
    state.speech_frame_count = 0
    state.started_speaking = False

    whisper_input = pcm8k_to_whisper_input(pcm8k)
    # STT and the LLM call are both blocking/CPU-or-network-bound — run them in
    # worker threads so they don't stall uvicorn's event loop (which is also
    # responsible for reading the next incoming audio frames on this call).
    try:
        user_text = await asyncio.to_thread(STT.transcribe_pcm, whisper_input)
    except Exception as e:
        print(f"[handle_utterance] STT failed: {e!r}")
        await speak_safe(ws, state, "Sorry, I didn't catch that, could you say it again?")
        return True

    if not user_text:
        return True
    print(f"Caller: {user_text}")

    if state.brain.should_end_call(user_text):
        await speak_safe(ws, state, "Alright, thanks for calling, take care!")
        return False

    try:
        reply = await asyncio.to_thread(state.brain.respond, user_text)
    except Exception as e:
        print(f"[handle_utterance] LLM call failed: {e!r}")
        reply = "Sorry, I'm having a little trouble on my end, could you say that again?"
    print(f"Agent: {reply}")

    await speak_safe(ws, state, reply)
    return True


def _log_end_of_call_diagnostics(state: CallState) -> None:
    """Best-effort diagnostics printed when a call ends, so a silent/unresponsive
    call is debuggable from the log alone instead of guesswork."""
    print(
        f"[call end] frames_seen={state.total_frames_seen} "
        f"max_rms_seen={state.max_rms_seen:.0f} (threshold={SILENCE_RMS_THRESHOLD:.0f}) "
        f"speech_was_detected={state.started_speaking or state.speech_frame_count > 0} "
        f"buffered_frames_at_end={len(state.frame_buffer)}"
    )
    if state.max_rms_seen > 0 and state.max_rms_seen < SILENCE_RMS_THRESHOLD:
        print(
            "[call end] HINT: max_rms_seen never crossed the threshold -- "
            "SILENCE_RMS_THRESHOLD is likely too high for this call's audio level. "
            "Lower it via VAD_RMS_THRESHOLD in .env."
        )
    if state.frame_buffer and state.speech_frame_count >= MIN_SPEECH_FRAMES:
        # There's a real, never-finalized utterance sitting in the buffer --
        # the call ended before enough trailing silence was seen. Can't speak
        # a reply back (the stream is closing), but worth knowing what STT
        # would have made of it.
        try:
            pcm8k = b"".join(state.frame_buffer)
            whisper_input = pcm8k_to_whisper_input(pcm8k)
            text = STT.transcribe_pcm(whisper_input)
            print(f"[call end] HINT: never-finalized speech in buffer transcribed to: {text!r}")
        except Exception as e:
            print(f"[call end] (diagnostic transcription of leftover buffer failed: {e!r})")


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
                await speak_safe(ws, state, opening)

            elif event == "media":
                mulaw_frame = base64.b64decode(msg["media"]["payload"])
                pcm16 = mulaw_frame_to_pcm16(mulaw_frame)
                raw_rms = frame_rms(pcm16)

                # Smooth over a few frames so a single noisy/crackly frame on a
                # real phone line can't reset the silence counter and stall
                # turn-end detection indefinitely -- this is the leading
                # suspect for a call where the agent never responds at all.
                state.recent_rms.append(raw_rms)
                if len(state.recent_rms) > RMS_SMOOTHING_FRAMES:
                    state.recent_rms.pop(0)
                rms = sum(state.recent_rms) / len(state.recent_rms)

                state.total_frames_seen += 1
                state.max_rms_seen = max(state.max_rms_seen, raw_rms)

                if rms > SILENCE_RMS_THRESHOLD:
                    if not state.started_speaking:
                        print(f"[VAD] speech detected (smoothed rms={rms:.0f}, threshold={SILENCE_RMS_THRESHOLD:.0f})")
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
                            print(
                                f"[VAD] discarding utterance, too short "
                                f"({state.speech_frame_count} voiced frames < {MIN_SPEECH_FRAMES} needed)"
                            )
                            state.frame_buffer.clear()
                            state.started_speaking = False
                            state.consecutive_silence = 0
                            state.speech_frame_count = 0

            elif event == "stop":
                _log_end_of_call_diagnostics(state)
                break

    except WebSocketDisconnect:
        _log_end_of_call_diagnostics(state)
    except Exception:
        # Last-resort net: log clearly and end this call gracefully instead of
        # an opaque ASGI traceback and a dead-silent line for the caller.
        _log_end_of_call_diagnostics(state)
        print("[media_stream] unexpected error, ending call:")
        traceback.print_exc()
