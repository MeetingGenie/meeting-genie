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
The speakers have been discussing the history of space exploration, beginning with how public perception of the Space Race often focuses on the Moon landing while overlooking the decades of scientific and engineering progress that made it possible. They agree that Apollo 11 was an extraordinary achievement, but one speaker argues that the earlier Mercury and Gemini programs deserve far more recognition because they solved fundamental problems like orbital rendezvous, long-duration spaceflight, and astronaut safety. The other speaker agrees, adding that breakthroughs are usually remembered while the incremental work that enables them fades into the background.
The conversation shifts toward the motivations behind the Space Race. Rather than viewing it purely as scientific curiosity, they discuss how competition between the United States and the Soviet Union accelerated technological development. One speaker points out that geopolitical rivalry produced enormous funding that would have been difficult to justify otherwise. They compare this to modern AI development, suggesting that competition often speeds innovation, although it can also encourage unnecessary secrecy and duplicated effort.
From there, they discuss whether human spaceflight provides enough value compared to robotic missions. One speaker initially argues that robotic probes are far more cost-effective because they can operate for years without risking human lives. The other agrees from a scientific perspective but believes crewed missions inspire the public in a way robots rarely do. They compare images from the Voyager missions, the Mars rovers, and the Apollo program, concluding that people emotionally connect with astronauts because they can imagine themselves standing in those environments.
The speakers spend some time discussing the Voyager missions. They are fascinated that both spacecraft continue sending useful scientific data decades after launch despite being built with computing power far below modern smartphones. One speaker jokes that engineering teams in the 1970s somehow built hardware that outlived several generations of computers. This leads to a conversation about designing systems for reliability instead of maximum performance, with both agreeing that modern consumer technology often prioritizes rapid iteration over longevity.
The discussion then moves to Mars exploration. One speaker believes humans will eventually establish a permanent settlement, though probably much later than optimistic predictions suggest. The other is more skeptical, arguing that the engineering challenges—radiation exposure, reduced gravity, psychological isolation, and the enormous cost of transporting supplies—remain underestimated by the public. They agree that sending people to Mars is significantly harder than landing on the Moon because rescue missions would be practically impossible during much of the journey.
They briefly compare NASA with newer private companies entering the space industry. One speaker appreciates how reusable rockets have dramatically reduced launch costs, making ambitious missions more economically realistic. However, they also note that government agencies still perform a unique role by funding long-term scientific missions that may never become commercially profitable. They conclude that public and private organizations will probably continue depending on one another rather than replacing each other.
Later, the conversation turns philosophical. They debate whether humanity should prioritize solving problems on Earth before investing heavily in space exploration. One speaker rejects the idea that these goals are mutually exclusive, arguing that scientific research often creates technologies that eventually benefit everyday life. Examples like satellite navigation, weather forecasting, advanced materials, and medical imaging are mentioned. The other agrees but adds that advocates sometimes exaggerate these benefits, making it important to justify exploration honestly rather than promising unrealistic economic returns.
Before the recent transcript begins, the speakers begin talking about future deep-space missions. They compare sending astronauts to Mars with robotic exploration of Europa, Titan, and Enceladus. One speaker becomes particularly interested in the search for extraterrestrial life, arguing that discovering even simple microbial organisms elsewhere in the Solar System would fundamentally change humanity's understanding of biology and its place in the universe. The discussion naturally leads into how scientists decide which worlds are currently considered the most promising places to search for life.
"""

    transcript = """
    THEM: Actually... wait, I always hear people talk about Mars first, but Europa sounds just as exciting to me.
    ME: yeah... ice... ocean... under maybe
    THEM: Right. If there's a liquid ocean underneath all that ice, that's incredibly interesting.
    ME: mm... life maybe... tiny things
    THEM: Exactly. Even microbes would completely change biology.
    ME: yeah... not alone... maybe
    THEM: Although Enceladus is fascinating too because we already know material is being blasted into space through those geysers.
    ME: easier... sample... maybe don't dig
    THEM: That's the clever part. You could potentially fly through the plume instead of drilling kilometers through ice.
    ME: yeah... less... impossible
    THEM: Titan is another weird one though. Lakes, rivers... just made of methane instead of water.
    ME: crazy... different country
    THEM: Which makes me wonder whether we're being too Earth-centric when we define what life even is.
    ME: maybe... meeting geenie... i dont know
    THEM: Exactly. We always assume liquid water because that's what works here.
    ME: yeah... baywatch
    THEM: So if scientists had to rank the most promising places in our Solar System to search for extraterrestrial life today, what would that list actually look like?
    """

    questions = [
        "What locations in our Solar System do scientists currently consider the most promising places to search for extraterrestrial life, and why?"]
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