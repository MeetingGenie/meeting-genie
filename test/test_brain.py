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
The speaker recounts the Greek myth of the Olympian gods and Prometheus. After Zeus, Poseidon, and Hades divide the world among themselves, Zeus and the Olympians rule the cosmos. Prometheus, pitying humanity's primitive existence, creates humans from clay and gives them fire so they can build civilization. Zeus, angered by Prometheus' deception during a sacrifice and by humanity's growing power, takes fire away. Prometheus steals it back from Olympus, leading Zeus to punish him by chaining him to a mountain where an eagle devours his regenerating liver every day.
Still seeking revenge on humanity, Zeus orders Hephaestus to create Pandora, the first woman, and the gods bestow various gifts upon her, including curiosity. She is given a sealed box and warned never to open it. After marrying Epimetheus, Prometheus' brother, Pandora eventually opens the box, releasing suffering, disease, old age, greed, envy, and other evils into the world. She closes it too late, leaving only Hope inside. The story concludes by explaining that although humanity must now endure hardship, hope remains, enabling people to persevere through adversity.
"""

    transcript = """
Them: Prometheus had advised his brother not to take anything from a god, but he was so struck by Pandora's beauty that he accepted her without thought.
Them: For a time, the two would live happily together, exploring Niger and having daughter named Hera, who bought them great joy.
Them: But in all her curiosity, Pandora's mind would always wander back to the bot. As days turned to weeks and weeks and months of curiosity turned into a burning desire.
Them: Finally, she could resist no more peeking into the box to see what was inside, but the moment she did, a great cloud filled the air.
Them: Out sprung, all the evils Prometheus have kept away from man, with greed and envy, as well as old age and disease, all escaping into the world.
Them: By the time Pandora managed to close the lid, there was only one thing remaining inside. HOO
Them: Mr. Pizzou's greatest punishment for with hope, men would clong the knives through all adversity, ensuring they enjoyed the hardships that now burdened them for the rest of time.
Them: Hyderabad, A perfect time where humanity lived without care or worry.
Me: This is a work meeting transcript. Hyderabad, Apexon, MeetingGenie, MeetingGenie, MeetingGenie, MeetingGenie, MeetingGenie, MeetingGenie, MeetingGenie.
Them: Never growing old, they were to live off the wild fruits of the land, enjoying all that Nietzsche had to offer, but with old age escaping from Pandora's Box, a time of-
Them: Who is Pandora?
Me: This is a work meeting transcript. Hyderabad, Apexon, MeetingGenie, MeetingGenie, MeetingGenie, MeetingGenie, MeetingGenie.
    """

    questions = [
        "Who is Pandora?"]
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