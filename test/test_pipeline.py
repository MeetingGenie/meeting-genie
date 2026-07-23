"""
MeetingGenie - test_pipeline.py

Replay an existing transcript.jsonl through the pipeline.

Tests:
    ✓ Trigger
    ✓ Brain
    ✓ Overlay

Skips:
    ✗ AudioRecorder
    ✗ Whisper
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path
import yaml

from meeting_genie.overlay import Overlay
from meeting_genie.brain import Brain
from meeting_genie.trigger import Trigger
from meeting_genie.summarize import Summarizer
from meeting_genie.types import Utterance

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "meeting_genie" / "config.yaml"
with open(CONFIG, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

TRANSCRIPT = Path(
    r"meetings\2026-07-23_1541\transcript.jsonl"
)

RECENT_LINES = 12


def load_config():

    with open(CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


cfg = load_config()

overlay = Overlay()

brain = Brain(cfg, overlay)

summarizer = Summarizer(cfg)

trigger = Trigger(
    cfg,
    lambda questions: brain.submit(
        questions=questions,
        recent_transcript="\n".join(recent_transcript),
        summary="\n".join(
            "- " + x
            for x in summarizer.get_summary()
        ),
    ),
)

recent_transcript = deque(maxlen=RECENT_LINES)


brain.start()
summarizer.start()


# ----------------------------------------------------------


def replay():

    print("\n========== REPLAY START ==========\n")

    with open(TRANSCRIPT, "r", encoding="utf-8") as f:

        for line in f:

            data = json.loads(line)

            utterance = Utterance(
                speaker=data["speaker"],
                text=data["text"],
                ts=data["ts"],
                silence_after_ms=data["silence_after_ms"],
                confidence=data["confidence"],
            )

            print(
                f"{utterance.speaker.upper():4} : "
                f"{utterance.text}"
            )

            recent_transcript.append(
                f"{utterance.speaker}: {utterance.text}"
            )

            summarizer.feed(utterance)

            trigger.feed(utterance)

            #
            # Small delay so it behaves like a meeting.
            #
            time.sleep(0.35)

    print("\n========== END OF TRANSCRIPT ==========\n")


threading.Thread(target=replay, daemon=True,).start()


overlay.start()