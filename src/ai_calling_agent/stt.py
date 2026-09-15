"""Speech-to-text: wraps faster-whisper, running fully local/offline.

No API key needed. This is the same engine we used to transcribe your
WhatsApp voice note earlier in this session — proven to work on this machine.
"""

from __future__ import annotations

import numpy as np
from faster_whisper import WhisperModel

from .config import STTConfig

# Whisper is well known to "hallucinate" boilerplate phrases (its training data
# was full of YouTube captions) when fed silence, background noise, or very
# short/low-information audio -- "Thank you for watching", "Please subscribe",
# etc. Confirmed live: a caller's brief pause/breath on a phone call produced
# "Thank you so much for watching." out of near-silence. Filter segments the
# model itself flags as likely non-speech, rather than trying to blocklist
# every possible hallucinated phrase.
NO_SPEECH_PROB_THRESHOLD = 0.6


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
        return self._join_confident_segments(segments)

    def transcribe_pcm(self, pcm: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe raw float32 PCM audio (mono, -1..1 range) held in memory.

        Used by the local mic demo and the Twilio bridge so we never have to
        round-trip through a temp file just to run STT.
        """
        segments, _info = self._model.transcribe(
            pcm, beam_size=5, language=self.cfg.language, vad_filter=True
        )
        return self._join_confident_segments(segments)

    @staticmethod
    def _join_confident_segments(segments) -> str:
        kept = [
            seg.text.strip()
            for seg in segments
            if seg.no_speech_prob < NO_SPEECH_PROB_THRESHOLD and seg.text.strip()
        ]
        return " ".join(kept).strip()
