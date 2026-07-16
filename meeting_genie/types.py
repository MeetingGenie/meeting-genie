"""The one data shape everything agrees on.

Audio side (audio.py + transcribe.py) PRODUCES Utterances into a queue.
Brain side (correction, trigger, brain, summarize) CONSUMES them.
As long as both sides respect this contract, neither blocks the other.

Changes to this file get flagged to the other person BEFORE merging.
"""
from dataclasses import dataclass
from typing import Literal


@dataclass
class Utterance:
    speaker: Literal["me", "them"]  # "me" = mic track, "them" = loopback track
    text: str                       # transcript (post-correction downstream)
    ts: float                       # time.monotonic() at utterance START, same clock both tracks
    silence_after_ms: int           # wall-clock silence after this utterance ended
    confidence: float               # whisper avg_logprob for the segment (0.0 if unknown)
