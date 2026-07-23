"""
MeetingGenie - trigger.py

Determines when the assistant should answer.
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

    if sentence.strip().endswith("?"):
        score += cfg["trigger"]["signals"]["ends_with_question_mark"]

    if any(w.lower() in INTERROGATIVES for w in clean_words[:5]):
        score += cfg["trigger"]["signals"]["starts_with_interrogative"]

    elif any(w.lower() in AUXILIARIES for w in clean_words[:5]):
        score += cfg["trigger"]["signals"]["starts_with_auxiliary"]

    if 3 <= len(words) <= 40:
        score += cfg["trigger"]["signals"]["length_in_range"]

    return score


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
        self.last_utterance_time = time.monotonic()

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

        with self.lock:

            self.last_utterance_time = now

            if utterance.speaker == "them":

                score = score_sentence(
                    utterance.text,
                    self.cfg,
                )

                if score >= self.cfg["trigger"]["threshold"]:

                    # Don't queue duplicates
                    if (
                        not self.pending_questions
                        or self.pending_questions[-1] != utterance.text
                    ):
                        self.pending_questions.append(
                            utterance.text
                        )

                    # restart silence timer
                    self.pending_since = now

            elif utterance.speaker == "me":

                # someone answered -> cancel
                self._clear_pending()

    def _run_timer(self):

        silence_seconds = (
            self.cfg["trigger"]["silence_ms"] / 1000
        )

        while self.running:

            time.sleep(0.1)

            fire = None

            with self.lock:

                if not self.pending_questions:
                    continue

                silence = (
                    time.monotonic()
                    - self.last_utterance_time
                )

                if silence >= silence_seconds:

                    # Only answer the LAST detected question.
                    fire = [self.pending_questions[-1]]

                    self._clear_pending()

            if fire:

                try:
                    print("[Trigger] Fired:", fire)
                    self.on_trigger(fire)

                except Exception:
                    import traceback
                    traceback.print_exc()

    def _clear_pending(self):

        self.pending_questions.clear()
        self.pending_since = None