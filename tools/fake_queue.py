"""Fake-queue harness: replays a scripted meeting into an Utterance queue.

This is how Track A/B develop WITHOUT a mic, Whisper, or Windows.
Devansh's Mac never needs to hear anything.

Script format (tools/fixtures/*.txt), one utterance per line:
    speaker | silence_after_ms | text

Usage:
    from tools.fake_queue import replay
    q = replay("tools/fixtures/sample_meeting.txt", speed=5.0)  # 5x real time
    while (u := q.get()) is not None:
        print(u)
"""
import threading
import time
from pathlib import Path
from queue import Queue

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meeting_genie.types import Utterance


def load_script(path: str) -> list[Utterance]:
    utterances = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        speaker, silence_ms, text = (part.strip() for part in line.split("|", 2))
        utterances.append(Utterance(
            speaker=speaker,            # type: ignore[arg-type]
            text=text,
            ts=0.0,                     # filled at replay time
            silence_after_ms=int(silence_ms),
            confidence=0.95,
        ))
    return utterances


def replay(path: str, speed: float = 1.0) -> "Queue[Utterance | None]":
    """Feed the script into a queue with realistic timing. None marks the end."""
    q: "Queue[Utterance | None]" = Queue()
    script = load_script(path)

    def run():
        for u in script:
            u.ts = time.monotonic()
            q.put(u)
            time.sleep((u.silence_after_ms / 1000.0) / speed)
        q.put(None)

    threading.Thread(target=run, daemon=True).start()
    return q


if __name__ == "__main__":
    q = replay("tools/fixtures/sample_meeting.txt", speed=10.0)
    while (u := q.get()) is not None:
        print(f"[{u.speaker:4}] ({u.silence_after_ms:>5}ms after) {u.text}")
