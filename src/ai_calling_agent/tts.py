"""Text-to-speech: two interchangeable backends behind one interface.

- EdgeTTS: free, no API key, natural neural voices (Microsoft Edge's TTS).
  Good default for hackathons — "just works" out of the box.
- ElevenLabsTTS: premium / voice-cloned. Point `elevenlabs_voice_id` at a
  cloned voice (upload a sample clip in the ElevenLabs dashboard) if you want
  to legitimately say "we trained a custom voice" in your pitch.

Both a sync API (`synthesize_to_file`, for the local mic/speaker demo, which
runs outside any event loop) and an async API (`asynthesize_to_file`, for the
Twilio server, which runs inside uvicorn's event loop) are provided. Calling
the sync `asyncio.run()`-based path from inside an already-running event loop
raises "asyncio.run() cannot be called from a running event loop" — that's
exactly the bug this split avoids.
"""

from __future__ import annotations

import asyncio
import tempfile

from .config import VoiceConfig, get_env


class TextToSpeech:
    """Facade that picks the configured backend."""

    def __init__(self, cfg: VoiceConfig):
        self.cfg = cfg
        if cfg.provider == "elevenlabs":
            self._backend: _Backend = _ElevenLabsBackend(cfg)
        else:
            self._backend = _EdgeTTSBackend(cfg)

    def synthesize_to_file(self, text: str, out_path: str | None = None) -> str:
        """Synchronous entrypoint — use from plain scripts (local demo), never
        from inside an async event loop."""
        if out_path is None:
            out_path = tempfile.mktemp(suffix=self._backend.file_suffix)
        self._backend.synthesize_to_file(text, out_path)
        return out_path

    async def asynthesize_to_file(self, text: str, out_path: str | None = None) -> str:
        """Async entrypoint — use from inside an async event loop (the Twilio
        websocket handler)."""
        if out_path is None:
            out_path = tempfile.mktemp(suffix=self._backend.file_suffix)
        await self._backend.asynthesize_to_file(text, out_path)
        return out_path


class _Backend:
    file_suffix = ".mp3"

    def synthesize_to_file(self, text: str, out_path: str) -> None:
        raise NotImplementedError

    async def asynthesize_to_file(self, text: str, out_path: str) -> None:
        raise NotImplementedError


class _EdgeTTSBackend(_Backend):
    file_suffix = ".mp3"

    def __init__(self, cfg: VoiceConfig):
        self.voice = cfg.edge_voice

    async def _save(self, text: str, out_path: str) -> None:
        import edge_tts

        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(out_path)

    def synthesize_to_file(self, text: str, out_path: str) -> None:
        asyncio.run(self._save(text, out_path))

    async def asynthesize_to_file(self, text: str, out_path: str) -> None:
        await self._save(text, out_path)


class _ElevenLabsBackend(_Backend):
    file_suffix = ".mp3"

    def __init__(self, cfg: VoiceConfig):
        from elevenlabs.client import ElevenLabs

        api_key = get_env("ELEVENLABS_API_KEY")
        if not api_key:
            raise RuntimeError(
                "voice.provider is 'elevenlabs' but ELEVENLABS_API_KEY is not set in .env"
            )
        if not cfg.elevenlabs_voice_id:
            raise RuntimeError(
                "voice.provider is 'elevenlabs' but elevenlabs_voice_id is empty in "
                "config/agent_config.yaml. Clone a voice in the ElevenLabs dashboard "
                "and paste its voice_id there."
            )
        self.client = ElevenLabs(api_key=api_key)
        self.voice_id = cfg.elevenlabs_voice_id
        self.model_id = cfg.elevenlabs_model

    def synthesize_to_file(self, text: str, out_path: str) -> None:
        audio = self.client.text_to_speech.convert(
            voice_id=self.voice_id,
            model_id=self.model_id,
            text=text,
        )
        with open(out_path, "wb") as f:
            for chunk in audio:
                if chunk:
                    f.write(chunk)

    async def asynthesize_to_file(self, text: str, out_path: str) -> None:
        # The ElevenLabs SDK is a blocking/sync HTTP client — run it in a
        # worker thread so it doesn't block uvicorn's event loop.
        await asyncio.to_thread(self.synthesize_to_file, text, out_path)


def play_file(path: str) -> None:
    """Play an audio file through the default output device (local demo mode)."""
    import soundfile as sf
    import sounddevice as sd

    data, samplerate = sf.read(path, dtype="float32", always_2d=False)
    sd.play(data, samplerate)
    sd.wait()
