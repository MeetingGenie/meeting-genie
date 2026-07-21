"""
MeetingGenie - test_brain.py

Manual integration test for brain.py.

This test bypasses:
- audio.py
- transcribe.py
- correction.py
- trigger.py
- summarizer.py

It verifies:

Overlay
    ↑
 Brain
    ↑
Fake meeting context
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import yaml

# ------------------------------------------------------------------
# Make sure the project root is importable.
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ------------------------------------------------------------------

from meeting_genie.overlay import Overlay
from meeting_genie.brain import Brain


# ------------------------------------------------------------------

CONFIG_PATH = (
    PROJECT_ROOT
    / "meeting_genie"
    / "config.yaml"
)


def load_config():

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------

def submit_fake_question(brain: Brain):

    #
    # Give Tkinter time to initialize.
    #

    time.sleep(2)

    summary = """
The team is discussing Greek mythology for an educational podcast.

Earlier in the meeting they covered:
- The twelve Olympian gods.
- The Titan war (Titanomachy).
- Zeus becoming king of Olympus.
- The relationships between major gods.
- Why myths changed between different Greek city-states.

The group agreed that the podcast should explain myths in a beginner-friendly way without assuming prior knowledge.
"""

    transcript = """
Them:
Earlier we explained how Zeus defeated the Titans and became ruler of Olympus.

Me:
Yeah... and uh... Croutons... he was like... eating... uh... soup? Because of... prophecy thing...

Them:
Right. Cronus swallowed his children because he feared one of them would overthrow him, just as he had overthrown his own father.

Me:
Yeah that's... um... then Zeus got hidden somewhere... island maybe? Then... goat milk... I don't remember.

Them:
Exactly. Zeus was hidden on Crete and later returned to challenge Cronus.

Me:
Thank you for watching our video

Them:
Athena was born from Zeus's head after Hephaestus split it open.

Me:
Yeah, that's the one.

Them:
Now here's something I want to clarify for the audience.

How is Greek mythology different from Roman mythology? Are the gods actually different, or are they mostly the same with different names?
"""

    questions = [
        "How is Greek mythology different from Roman mythology? Are the gods actually different, or are they mostly the same with different names?"
    ]

    brain.submit(
        questions=questions,
        recent_transcript=transcript,
        summary=summary,
    )


# ------------------------------------------------------------------

def main():

    cfg = load_config()

    overlay = Overlay()

    brain = Brain(
        cfg=cfg,
        overlay=overlay,
    )

    brain.start()

    worker = threading.Thread(
        target=submit_fake_question,
        args=(brain,),
        daemon=True,
    )

    worker.start()

    overlay.start()


# ------------------------------------------------------------------

if __name__ == "__main__":
    main()