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
    confidence: float               # 0..1 from STT token logprobs (0.0 = unknown)

    # ADDED - see HANDOFF.md "Bugs found" #7.
    #
    # time.monotonic() at the moment the SPEECH stopped: the end of the last
    # loud audio chunk in this utterance. NOT the moment this object was built.
    #
    # Why it exists: trigger.py measures "how long has the user been silent".
    # It used to measure that from when the Utterance ARRIVED, which is only
    # after STT has finished transcribing it. That silently stacked the whole
    # STT time on top of the 2000ms silence window. This field lets trigger.py
    # measure against the audio clock instead.
    #
    # Defaults to 0.0 so anything that builds an Utterance without it
    # (tools/fake_queue.py, test/test_pipeline.py, Meesha's tests) keeps
    # working unchanged. Consumers treat 0.0 as "unknown" and fall back to
    # arrival time.
    audio_end_ts: float = 0.0