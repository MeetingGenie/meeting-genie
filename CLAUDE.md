# Claude Code context for MeetingGenie

- The `Utterance` dataclass in `meeting_genie/types.py` is the contract between the
  audio side and the brain side. Do not change it without flagging the other collaborator.
- Every tunable (trigger weights, thresholds, prompts, models) lives in
  `meeting_genie/config.yaml`. Never hardcode a threshold in a .py file.
- STT is faster-whisper (NOT openai-whisper). LLM is local Ollama (NOT cloud APIs).
  Structured Ollama calls use format=json.
- Loopback capture is Windows-only (PyAudioWPatch). MicSource uses sounddevice and
  must stay cross-platform (one dev is on macOS).
- Priority rule: whisper transcription is never blocked; one LLM answer generation
  in flight max; summarizer yields to both.
- The overlay must never steal keyboard focus.
- Ownership: audio.py is shared/paired. trigger/brain/overlay = Track A.
  transcribe/correction/summarize = Track B. Don't rewrite files on the other track
  without being asked.
- When porting code from the old prototype (Meesha16/Voice_Assistant), say so in the
  commit message.
