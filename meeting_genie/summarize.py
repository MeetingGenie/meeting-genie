"""Keeps a running bullet-point summary of the meeting so
far, so brain.py always has bounded-size context instead of the whole
growing transcript.

Design:
- Own background thread, own timer. Never blocks whoever is calling feed().
  Same reasoning as _whisper_worker in transcribe.py: a slow LLM call on the
  same thread as incoming utterances would delay/drop them.
- Each summarization run = previous summary + raw utterances since then.
  Never the whole meeting from scratch - that's what keeps it bounded and
  is why it can compound indefinitely without blowing up in size.
- feed() only appends to a buffer, thread-safe via a lock. The actual LLM
  call happens on the timer thread, not here.
- Empty buffer when the timer fires -> skip that round. No LLM call, no
  point summarizing silence.
- First run, no previous summary yet -> that's fine, prompt just says so.
- Priority 3 in the whole system (behind Whisper, behind brain.py's answer
  generation) - this is what "own thread, own timer" buys us: it can run
  late and nobody notices.
"""
import json
import threading
import time

import requests


class Summarizer:
    def __init__(self, cfg, on_summary_updated=None):
        self.cfg = cfg
        self.on_summary_updated = on_summary_updated  # called with new bullets list

        self._buffer = []
        self._lock = threading.Lock()
        self._current_bullets = []
        self._thread = None
        self._running = False

    def feed(self, utterance):
        """Fast, thread-safe. Just appends - the LLM call happens elsewhere,
        on the timer thread, not here."""
        with self._lock:
            self._buffer.append(utterance)

    def get_summary(self):
        """Latest bullets. What brain.py reads for context. Thread-safe."""
        with self._lock:
            return list(self._current_bullets)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _loop(self):
        interval = self.cfg["summarize"]["interval_seconds"]
        while self._running:
            time.sleep(interval)
            self._run_one_cycle()

    def _run_one_cycle(self):
        with self._lock:
            pending = self._buffer
            self._buffer = []
            previous_bullets = list(self._current_bullets)

        if not pending:
            return  # nothing new since last summary, skip - no LLM call

        transcript = "\n".join(f"{u.speaker}: {u.text}" for u in pending)
        new_bullets = self._call_llm(previous_bullets, transcript)
        if new_bullets is None:
            return  # LLM call failed - keep the old summary, don't lose it

        with self._lock:
            self._current_bullets = new_bullets

        if self.on_summary_updated:
            self.on_summary_updated(new_bullets)

    def _call_llm(self, previous_bullets, transcript):
        s = self.cfg["summarize"]
        previous_text = "\n".join(f"- {b}" for b in previous_bullets) or "(no summary yet)"
        system_prompt = s["system_prompt"].format(max_bullets=s["max_bullets"])

        user_message = (
            f"Previous summary:\n{previous_text}\n\n"
            f"New conversation since then:\n{transcript}"
        )

        try:
            response = requests.post(
                s["ollama_url"],
                json={
                    "model": s["model"],
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    "stream": False,
                    "format": "json",
                    "options": {"num_predict": s["num_predict"]},
                },
                timeout=30,
            )
            response.raise_for_status()
            content = response.json()["message"]["content"]
            parsed = json.loads(content)
            bullets = parsed.get("bullets", [])
            return bullets[: s["max_bullets"]]
        except Exception:
            # Network hiccup, bad JSON, Ollama not running, whatever -
            # summarizer failing should never crash the whole app.
            return None