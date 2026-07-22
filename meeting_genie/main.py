"""Wiring only. No business logic lives here.

Construction order matters:
  1. Overlay   - builds the tk window (does not block yet)
  2. Brain     - needs the overlay instance passed in
  3. Summarizer, Trigger, AudioRecorder
  4. Background threads for transcription
  5. overlay.start() LAST - it calls tk mainloop() and blocks forever

Threading:
  - tkinter must own the main thread, so everything else runs on daemon threads
    underneath it.
  - overlay.expand()/collapse() are thread-safe (they push onto its internal
    queue), so the hotkey callback can safely call them from its own thread.

Hotkey behaviour (hybrid, agreed with Meesha):
  - pending answer exists  -> just reveal it
  - nothing pending        -> force a fresh generation from recent context

Note on trigger.py: the merged version returns a plain list[str] of questions
from feed() and has no force() or conversation context. brain.submit() needs
recent_transcript, so main.py keeps its own small rolling transcript and
implements the force path itself.
"""
from __future__ import annotations

import threading
from collections import deque
from datetime import datetime
from pathlib import Path

import yaml

from meeting_genie.audio import AudioRecorder
from meeting_genie.brain import Brain
from meeting_genie.overlay import Overlay
from meeting_genie.summarize import Summarizer
from meeting_genie.transcribe import run_transcription_loop
from meeting_genie.trigger import Trigger

CONFIG_PATH = Path(__file__).with_name("config.yaml")
RECENT_TRANSCRIPT_LINES = 12


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def make_meeting_dir(cfg: dict) -> Path:
    root = Path(cfg.get("output", {}).get("meetings_dir", "meetings"))
    path = root / datetime.now().strftime("%Y-%m-%d_%H%M")
    path.mkdir(parents=True, exist_ok=True)
    return path


class MeetingGenie:
    def __init__(self):
        self.cfg = load_config()
        self.meeting_dir = make_meeting_dir(self.cfg)
        self.summary_path = self.meeting_dir / "summary.md"
        self.transcript_path = self.meeting_dir / "transcript.jsonl"

        # Rolling window of recent lines, used as brain's recent_transcript.
        # trigger.py doesn't track this itself, so main.py owns it.
        self.recent = deque(maxlen=RECENT_TRANSCRIPT_LINES)
        self.recent_lock = threading.Lock()

        # Order matters: overlay first, brain needs it.
        self.overlay = Overlay()
        self.brain = Brain(self.cfg, self.overlay)
        self.summarizer = Summarizer(self.cfg, on_summary_updated=self._write_summary)
        self.trigger = Trigger(self.cfg)
        self.recorder = AudioRecorder()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _write_summary(self, bullets: list[str]) -> None:
        """Called by Summarizer on its own thread whenever a new summary lands."""
        try:
            text = "\n".join(f"- {b}" for b in bullets)
            self.summary_path.write_text(text + "\n", encoding="utf-8")
        except Exception:
            pass  # writing the summary file must never break the pipeline

    def _current_summary_text(self) -> str:
        bullets = self.summarizer.get_summary()
        if not bullets:
            return "(no summary yet)"
        return "\n".join(f"- {b}" for b in bullets)

    def _recent_transcript_text(self) -> str:
        with self.recent_lock:
            return "\n".join(self.recent)

    def _on_utterance(self, utterance) -> None:
        """Called from the transcription thread for every finished utterance."""
        line = f"{utterance.speaker}: {utterance.text}"
        with self.recent_lock:
            self.recent.append(line)
        self._append_transcript(utterance)

        # Summarizer just buffers; its own timer thread does the LLM call.
        self.summarizer.feed(utterance)

        # Trigger returns a list of question strings when it decides to fire.
        questions = self.trigger.feed(utterance)
        if questions:
            self._ask_brain(questions)

    def _append_transcript(self, utterance) -> None:
        try:
            import json
            record = {
                "speaker": utterance.speaker,
                "text": utterance.text,
                "ts": utterance.ts,
                "silence_after_ms": utterance.silence_after_ms,
                "confidence": utterance.confidence,
            }
            with open(self.transcript_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass  # transcript logging must never break the pipeline

    def _ask_brain(self, questions: list[str]) -> None:
        self.brain.submit(
            questions=questions,
            recent_transcript=self._recent_transcript_text(),
            summary=self._current_summary_text(),
        )

    def _on_hotkey(self) -> None:
        """Hybrid: reveal a waiting answer, or force a fresh one."""
        if self.overlay.has_pending_content():
            self.overlay.expand()
            return

        # Nothing pending - force a generation from whatever was just said.
        # trigger.py has no force(), so main.py does it: take the most recent
        # "them" line as the question.
        with self.recent_lock:
            them_lines = [l for l in self.recent if l.startswith("them: ")]
        if not them_lines:
            self.overlay.show_message("Nothing to answer yet.")
            self.overlay.expand()
            return

        question = them_lines[-1][len("them: "):]
        self._ask_brain([question])
        self.overlay.expand()  # user explicitly asked, so show it streaming in

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        print(f"MeetingGenie starting. Meeting folder: {self.meeting_dir}")

        self.brain.start()        # its own worker thread + prewarm
        self.summarizer.start()   # its own timer thread
        self.recorder.start()     # audio callbacks on their own threads

        transcribe_thread = threading.Thread(
            target=run_transcription_loop,
            args=(self.recorder, self.cfg, self._on_utterance),
            daemon=True,
            name="TranscribeLoop",
        )
        transcribe_thread.start()

        self._register_hotkey()

        print("Running. Overlay is live. Ctrl+Shift+Q quits.")
        try:
            self.overlay.start()  # BLOCKS here until the window closes
        finally:
            self.shutdown()

    def _register_hotkey(self) -> None:
        combo = self.cfg.get("overlay", {}).get("hotkey", "ctrl+shift+space")
        try:
            import keyboard
            keyboard.add_hotkey(combo, self._on_hotkey)
            print(f"Hotkey registered: {combo}")
        except Exception as exc:
            # keyboard needs root on macOS and can fail on locked-down setups.
            # Not fatal - automatic triggering still works without it.
            print(f"Hotkey NOT registered ({exc}). Auto-trigger still active.")

    def shutdown(self) -> None:
        print("Shutting down...")
        try:
            self.recorder.stop()
        except Exception:
            pass
        try:
            self.summarizer.stop()
        except Exception:
            pass
        try:
            self.brain.stop()
        except Exception:
            pass
        print(f"Done. Output in: {self.meeting_dir}")


def main() -> None:
    MeetingGenie().run()


if __name__ == "__main__":
    main()