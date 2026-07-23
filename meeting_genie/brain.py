"""Track A. Talks to Ollama. Answer generation + scheduling.

- Max ONE generation in flight; newer trigger supersedes and cancels older.
- stream=true, render tokens as they arrive. num_predict cap from config.
- Pre-warm on startup so the first real answer isn't paying model-load time.
- Priority 2: below whisper, above summarizer.
- Log every generation to meetings/<stamp>/suggestions.json with its trigger reason.

MeetingGenie - brain.py
Responsibilities
----------------
- Accept trigger requests.
- Build prompts for Ollama.
- Stream generated tokens.
- Never block transcription.
- Only ONE generation may exist at a time.
- Newer requests supersede older ones.
- Prewarm Ollama on startup.
- Log generations.

This module intentionally knows nothing about:

- audio capture
- whisper
- VAD
- correction
- trigger heuristics
- tkinter internals

It only talks to Ollama and the overlay.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import Optional

import requests


class Brain:
    """
    Background Ollama scheduler.

    Public API
    ----------
    start()
    stop()

    submit(
        questions,
        recent_transcript,
        summary
    )

    One worker thread.

    One active generation.

    Never blocks callers.
    """

    def __init__(self, cfg: dict, overlay):

        self.cfg = cfg
        self.overlay = overlay

        brain_cfg = cfg["brain"]

        self.model = brain_cfg["model"]
        self.temperature = brain_cfg["temperature"]
        self.top_p = brain_cfg["top_p"]
        self.num_predict = brain_cfg["num_predict"]
        self.timeout = brain_cfg["timeout"]
        self.system_prompt = brain_cfg["system_prompt"]
        self.prewarm_prompt = brain_cfg["prewarm_prompt"]

        self.base_url = brain_cfg.get("host","http://127.0.0.1:11434",)
        self.log_path = Path(cfg["logging"]["suggestions_file"])
        self.log_path.parent.mkdir(parents=True,exist_ok=True)

        # Worker queue.
        # Queue size = 1
        # New requests replace older ones.
        self.request_queue = queue.Queue(maxsize=1)
        # Thread state
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None
        # Generation cancellation.
        self.generation_lock = threading.Lock()
        self.generation_id = 0

    ##################################################################
    # Lifecycle
    ##################################################################

    def start(self):

        if self.running:
            return

        self.running = True

        #
        # Warm model.
        #

        self._prewarm()

        self.worker_thread = threading.Thread(
            target=self._worker,
            daemon=True,
            name="BrainWorker"
        )

        self.worker_thread.start()

    def stop(self):

        self.running = False

        try:
            self.request_queue.put_nowait(None)
        except queue.Full:
            pass

        if self.worker_thread is not None:
            self.worker_thread.join(timeout=2)

    ##################################################################
    # Public API
    ##################################################################

    def submit(
        self,
        questions: list[str],
        recent_transcript: str,
        summary: str,
    ):
        print("[Brain] Submit:", questions)
        """
        Schedule a new generation.

        Any previous queued request is discarded.

        Running generations will cancel
        themselves automatically.
        """

        with self.generation_lock:
            self.generation_id += 1
            generation = self.generation_id

        payload = {
            "generation": generation,
            "questions": questions,
            "recent": recent_transcript,
            "summary": summary,
            "submitted": time.monotonic(),
        }

        #
        # Replace queued request.
        #

        while True:

            try:
                self.request_queue.get_nowait()

            except queue.Empty:
                break

        self.request_queue.put(payload)

    ##################################################################
    # Worker
    ##################################################################

    def _worker(self):

        while self.running:

            request = self.request_queue.get()

            if request is None:
                continue

            generation = request["generation"]

            prompt = self._build_prompt(
                summary=request["summary"],
                transcript=request["recent"],
                questions=request["questions"],
            )

            self.overlay.show_loading()

            start = time.perf_counter()

            answer = self._generate_stream(
                generation,
                prompt
            )

            latency = (
                time.perf_counter() - start
            ) * 1000

            if answer is not None:

                self._log_generation(
                    prompt=prompt,
                    answer=answer,
                    trigger=request["questions"],
                    latency=latency,
                )
        ##################################################################
    # Ollama
    ##################################################################

    def _prewarm(self) -> None:
        """
        Load the model into memory so the first
        real request doesn't pay model load time.
        """

        try:
            requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": self.prewarm_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,
                        "num_predict": 1,
                    },
                },
                timeout=self.timeout,
            )

        except Exception:
            #
            # Failure here is not fatal.
            # Ollama may not have started yet.
            #
            pass

    ##################################################################

    def _build_prompt(self,summary: str,transcript: str,questions: list[str],) -> str:
        question_block = "\n".join(f"- {q}" for q in questions)
        prompt = f"""{self.system_prompt}
        ## Meeting Summary
        {summary.strip()}
        ## Recent Conversation
        {transcript.strip()}
        ## Question To Answer
        {question_block}
        ## Instructions
        You are answering someone's question during a meeting.
        Output ONLY the answer.
        Do not explain the meeting.
        Do not mention the transcript.
        Do not mention the summary.
        Do not say the summary is empty.
        Begin immediately with the answer.
        Keep the response under {self.num_predict} tokens.
        """
        return prompt

    def _generate_stream(
    self,
    generation: int,
    prompt: str,
) -> Optional[str]:
        """
        Stream Ollama output to the overlay.

        Returns:
            None if cancelled or failed.
            Full response string otherwise.
        """

        body = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_predict": self.num_predict,
            },
        }

        print("\nSending prompt to Ollama...")
        print(f"Prompt length: {len(prompt)} characters")

        full_response = []

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=body,
                stream=True,
                timeout=(10, None),      # <-- IMPORTANT CHANGE
            )

            response.raise_for_status()

            print("HTTP Connected.\nStreaming...\n")

            # Clear the "Thinking..." message
            self.overlay.show_message("")
            self.overlay.expand()

            for line in response.iter_lines(decode_unicode=True):

                if not line:
                    continue

                # Cancel if a newer generation arrived
                with self.generation_lock:
                    if generation != self.generation_id:
                        response.close()
                        return None

                packet = json.loads(line)

                token = packet.get("response", "")

                if token:
                    full_response.append(token)

                    # Stream token to overlay
                    self.overlay.append_text(token)

                    # Also print live to terminal
                    print(token, end="", flush=True)

                if packet.get("done", False):
                    break

            print("\n\n========== FINISHED ==========\n")

            return "".join(full_response)

        except requests.RequestException as e:
            import traceback

            traceback.print_exc()

            self.overlay.show_message(
                f"Unable to contact Ollama:\n{e}"
            )

            return None

        except Exception as e:
            import traceback

            traceback.print_exc()

            self.overlay.show_message(
                f"Generation error:\n{e}"
            )

            return None
    ##################################################################
    # Logging
    ##################################################################

    def _log_generation(
        self,
        prompt: str,
        answer: str,
        trigger: list[str],
        latency: float,
    ) -> None:

        record = {

            "timestamp": time.time(),

            "trigger": trigger,

            "latency_ms": round(
                latency,
                2,
            ),

            "prompt": prompt,

            "answer": answer,
        }

        try:

            if self.log_path.exists():

                with open(
                    self.log_path,
                    "r",
                    encoding="utf-8",
                ) as f:

                    data = json.load(f)

            else:

                data = []

            data.append(record)

            with open(
                self.log_path,
                "w",
                encoding="utf-8",
            ) as f:

                json.dump(
                    data,
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

        except Exception:

            #
            # Logging must never stop
            # answer generation.
            #

            pass
    