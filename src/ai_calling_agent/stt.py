"""Speech-to-text: wraps faster-whisper, running fully local/offline.

No API key needed. This is the same engine we used to transcribe your
WhatsApp voice note earlier in this session — proven to work on this machine.
"""

from __future__ import annotations

import numpy as np
from faster_whisper import WhisperModel

from .config import STTConfig


class SpeechToText:
    def __init__(self, cfg: STTConfig):
        self.cfg = cfg
        self._model = WhisperModel(
            cfg.model_size, device=cfg.device, compute_type=cfg.compute_type
        )

    def transcribe_file(self, path: str) -> str:
        """Transcribe a .wav/.mp3/etc. file on disk and return the full text."""
        segments, _info = self._model.transcribe(
            path, beam_size=5, language=self.cfg.language
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    def transcribe_pcm(self, pcm: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe raw float32 PCM audio (mono, -1..1 range) held in memory.

        Used by the local mic demo so we never have to round-trip through a
        temp file just to run STT.
        """
        segments, _info = self._model.transcribe(
            pcm, beam_size=5, language=self.cfg.language, vad_filter=True
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
