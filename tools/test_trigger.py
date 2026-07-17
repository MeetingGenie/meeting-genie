"""Loads the real config.yaml and runs Trigger against the fixture.

    python3 tools/test_trigger.py
"""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meeting_genie.trigger import Trigger
from tools.fake_queue import load_script

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "meeting_genie" / "config.yaml").read_text())
FIXTURE = ROOT / "tools" / "fixtures" / "sample_meeting.txt"


def main():
    script = load_script(str(FIXTURE))
    trigger = Trigger(CFG)

    fired_count = 0
    for u in script:
        result = trigger.feed(u)
        if result:
            fired_count += 1
            print("FIRED:", result)

    print(f"\ntotal fires: {fired_count}")


if __name__ == "__main__":
    main()