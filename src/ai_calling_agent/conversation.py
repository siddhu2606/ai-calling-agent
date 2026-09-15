"""Turn-taking loop for the local mic/speaker demo: record -> transcribe ->
think -> speak -> repeat.

Recording auto-stops on silence (simple RMS-based voice activity detection) so
you don't need a push-to-talk button — you just talk, pause, and the agent
responds, like on a real call.
"""

from __future__ import annotations

import time

import numpy as np
import sounddevice as sd

from .brain import ConversationBrain
from .config import AgentConfig
from .stt import SpeechToText
from .tts import TextToSpeech, play_file

SAMPLE_RATE = 16000
BLOCK_SIZE = 1600  # 100ms blocks at 16kHz
SILENCE_RMS_THRESHOLD = 0.01
MAX_RECORD_SECONDS = 20
MIN_SPEECH_SECONDS = 0.4


def record_until_silence(
    silence_timeout: float = 1.2,
    max_seconds: float = MAX_RECORD_SECONDS,
) -> np.ndarray:
    """Record from the mic and stop once the caller has gone quiet.

    Returns mono float32 PCM at 16kHz, ready for faster-whisper.
    """
    print("Listening... (speak now)")
    blocks: list[np.ndarray] = []
    silence_blocks_needed = int(silence_timeout / (BLOCK_SIZE / SAMPLE_RATE))
    consecutive_silence = 0
    started_speaking = False
    start_time = time.time()

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=BLOCK_SIZE
    ) as stream:
        while time.time() - start_time < max_seconds:
            block, _overflow = stream.read(BLOCK_SIZE)
            block = block[:, 0]
            rms = float(np.sqrt(np.mean(block**2)))

            if rms > SILENCE_RMS_THRESHOLD:
                started_speaking = True
                consecutive_silence = 0
                blocks.append(block)
            elif started_speaking:
                consecutive_silence += 1
                blocks.append(block)
                if consecutive_silence >= silence_blocks_needed:
                    break

    if not blocks:
        return np.zeros(0, dtype="float32")
    return np.concatenate(blocks)


class ConversationLoop:
    """Wires STT + brain + TTS together for the local demo."""

    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        print("Loading speech-to-text model (first run downloads weights)...")
        self.stt = SpeechToText(cfg.stt)
        self.tts = TextToSpeech(cfg.voice)
        self.brain = ConversationBrain.create(cfg)

    def speak(self, text: str) -> None:
        print(f"{self.cfg.name}: {text}")
        audio_path = self.tts.synthesize_to_file(text)
        play_file(audio_path)

    def run(self) -> None:
        opening = self.brain.opening_line()
        self.speak(opening)

        for _ in range(self.cfg.behavior.max_turns):
            pcm = record_until_silence(self.cfg.behavior.silence_timeout_seconds)
            duration = len(pcm) / SAMPLE_RATE
            if duration < MIN_SPEECH_SECONDS:
                print("(didn't catch that — try again)")
                continue

            user_text = self.stt.transcribe_pcm(pcm)
            if not user_text:
                print("(didn't catch that — try again)")
                continue
            print(f"You: {user_text}")

            if self.brain.should_end_call(user_text):
                self.speak("Alright, thanks for your time — take care!")
                break

            reply = self.brain.respond(user_text)
            self.speak(reply)
        else:
            self.speak("We're at time for this call — I'll follow up. Thanks!")
