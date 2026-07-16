# MeetingGenie

Live meeting copilot. Runs locally during a Teams/Meet/Zoom call:
tracks the conversation, keeps a rolling summary, and quietly suggests
an answer when a question is directed at you and you go silent.

Local everything: faster-whisper (STT) + Ollama (LLM). No cloud, no tokens.

## How it fits together

```
MicSource ("me") ──┐
                   ├─► Utterance queue ─► correction ─┬─► trigger ─► brain ─► overlay
LoopbackSource ────┘         ▲                        └─► summarize ─► summary.md
  ("them")             transcribe.py
```

The `Utterance` dataclass in `meeting_genie/types.py` is the contract between
the audio side and the brain side. See `meeting_genie/config.yaml` for every
tunable knob — behavior is data.

## Dev without audio

`python tools/fake_queue.py` replays a scripted meeting into the queue.
The whole brain side (trigger/brain/overlay/correction/summarize) develops
against this — no mic, no Whisper, no Windows needed.

## Platform notes

- Loopback capture (the "them" track) is Windows-only for now (WASAPI via
  PyAudioWPatch). The demo machine is Windows.
- Mic capture and everything downstream is cross-platform.
- Headphones required in v1 (keeps the mic from hearing the speakers).
  No-headphones mode via echo dedup is planned — see the build plan doc.

## Setup

```
pip install -r requirements.txt
ollama pull llama3.1:8b
python -m meeting_genie.main
```
