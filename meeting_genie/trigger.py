"""
MeetingGenie - trigger.py

Determines when the assistant should answer.

Two mechanisms, deliberately separate (see HANDOFF.md):
  1. score_sentence() - is this text a question?
  2. a background timer thread - has the user actually been silent since?

FIXED IN THIS VERSION (HANDOFF.md bug #7, "additive silence delay"):
The silence clock used to start when an Utterance ARRIVED, which is only
after STT finished transcribing it. So a 2000ms silence window really meant
600ms (segment cut) + STT time + 2000ms of real-world silence before firing.
It now measures against utterance.audio_end_ts - the moment the speech
actually stopped - so 2000ms means 2000ms.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

INTERROGATIVES = {
    "what", "why", "when", "where", "who", "how",
}

AUXILIARIES = {
    "is", "are", "do", "does", "did",
    "can", "could", "would", "should",
    "will", "have", "has",
}

GREETINGS = {
    "hi", "hello", "hey",
    "okay", "ok", "so",
    "anyway",
    "um", "uh",
}


def strip_greeting(sentence: str) -> str:
    words = sentence.strip().split()

    while words and words[0].strip(",.!?").lower() in GREETINGS:
        words = words[1:]

    return " ".join(words)


def score_sentence(sentence: str, cfg: dict) -> float:
    words = sentence.strip().split()

    cleaned = strip_greeting(sentence)
    clean_words = cleaned.split()

    score = 0.0

    # Use configured min/max word counts when applying length-based signals.
    min_words = int(cfg.get("trigger", {}).get("min_words", 3))
    max_words = int(cfg.get("trigger", {}).get("max_words", 40))

    # Only consider a trailing question mark as a strong signal when the
    # utterance has at least `min_words` words — short interjections like
    # "Jane?" shouldn't force the assistant to answer.
    if sentence.strip().endswith("?") and len(words) >= min_words:
        score += cfg["trigger"]["signals"]["ends_with_question_mark"]

    first_word = clean_words[0].strip(",.!?").lower() if clean_words else ""
    if first_word in INTERROGATIVES:
        score += cfg["trigger"]["signals"]["starts_with_interrogative"]

    elif first_word in AUXILIARIES:
        score += cfg["trigger"]["signals"]["starts_with_auxiliary"]

    if min_words <= len(words) <= max_words:
        score += cfg["trigger"]["signals"]["length_in_range"]

    return score


def speech_end_time(utterance, now: float) -> float:
    """When did this utterance's speech actually stop, on the monotonic clock?

    Prefers utterance.audio_end_ts (set by transcribe.py's Segmenter from the
    last loud audio chunk). Falls back to arrival time when it's missing or
    zero, which is the case for tools/fake_queue.py and any older producer.

    Clamped to `now`: an audio timestamp in the future would mean a clock
    mismatch between tracks, and we'd rather under-fire than fire instantly.
    """
    ts = getattr(utterance, "audio_end_ts", 0.0) or 0.0
    if ts <= 0.0:
        return now
    return min(ts, now)


class Trigger:

    def __init__(
        self,
        cfg: dict,
        on_trigger: Callable[[list[str]], None],
    ):
        self.cfg = cfg
        self.on_trigger = on_trigger

        self.pending_questions: list[str] = []

        self.pending_since: Optional[float] = None

        # Monotonic time that speech last STOPPED on either track.
        # Was last_utterance_time (arrival time) before the bug #7 fix.
        #
        # Starts as None, NOT time.monotonic(). Seeding it with construction
        # time breaks the max() below: the first real utterance's audio ended
        # BEFORE we were constructed-plus-STT-time, so max() would throw the
        # true audio timestamp away and we'd silently be back to timing from
        # arrival. Caught by the "STT took 1.5s" test.
        self.last_speech_end: Optional[float] = None

        self.lock = threading.Lock()

        self.running = False

        self.thread = threading.Thread(
            target=self._run_timer,
            daemon=True,
            name="TriggerTimer",
        )

    def start(self):
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)

    def feed(self, utterance):

        now = time.monotonic()
        ended = speech_end_time(utterance, now)

        with self.lock:

            # max(), not assignment: utterances from the two tracks can be
            # delivered out of order relative to their audio clocks (one
            # source's STT call may finish after a later one from the other).
            # Silence is measured from the MOST RECENT speech on either track,
            # so an out-of-order older utterance must not rewind the clock.
            if self.last_speech_end is None:
                self.last_speech_end = ended
            else:
                self.last_speech_end = max(self.last_speech_end, ended)

            score = score_sentence(utterance.text, self.cfg)

            # Guard: ignore low-confidence microphone transcriptions to
            # avoid false positives from noisy local mic input. The default
            # min_confidence is conservative; callers can raise it in config.
            min_conf = float(self.cfg.get("trigger", {}).get("min_confidence", 0.45))
            if utterance.speaker == "me":
                conf_val = float(getattr(utterance, "confidence", 0.0) or 0.0)
                if conf_val < min_conf:
                    # Treat as non-question due to low confidence
                    score = 0.0

            # Use strict greater-than for the threshold to avoid exact-equal
            # edge cases where a single signal equals the threshold.
            if score > float(self.cfg["trigger"]["threshold"]):
                # Questions can originate from the meeting audio OR from the
                # user's microphone. The old source-specific branch made mic
                # questions impossible to answer by clearing them here.
                if (
                    not self.pending_questions
                    or self.pending_questions[-1] != utterance.text
                ):
                    self.pending_questions.append(utterance.text)
                self.pending_since = ended

            elif utterance.speaker == "me":
                # The user started a normal statement, so they are likely
                # answering a previous question themselves.
                self._clear_pending()

    def mark_audio_activity(self, timestamp: float) -> None:
        """Keep the silence timer honest while either track is still talking."""
        with self.lock:
            if self.last_speech_end is None:
                self.last_speech_end = timestamp
            else:
                self.last_speech_end = max(self.last_speech_end, timestamp)

    def _run_timer(self):

        silence_seconds = (
            self.cfg["trigger"]["silence_ms"] / 1000
        )

        while self.running:

            time.sleep(0.1)

            fire = None
            silence_at_fire = 0.0

            with self.lock:

                if not self.pending_questions or self.last_speech_end is None:
                    continue

                silence = (
                    time.monotonic()
                    - self.last_speech_end
                )

                if silence >= silence_seconds:

                    # Only answer the LAST detected question.
                    fire = [self.pending_questions[-1]]
                    silence_at_fire = silence

                    self._clear_pending()

            if fire:

                try:
                    # Instrumentation: real silence at fire time should now sit
                    # just above trigger.silence_ms. If it's consistently much
                    # larger, STT is still the bottleneck and the model swap
                    # needs revisiting - not the trigger.
                    print(
                        f"[Trigger] Fired after {silence_at_fire * 1000:.0f}ms "
                        f"real silence (target {silence_seconds * 1000:.0f}ms): {fire}"
                    )
                    self.on_trigger(fire)

                except Exception:
                    import traceback
                    traceback.print_exc()

    def _clear_pending(self):

        self.pending_questions.clear()
        self.pending_since = None
