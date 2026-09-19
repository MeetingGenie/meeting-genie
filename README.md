# MeetingGenie

A live meeting copilot that runs entirely on your own machine.

During a real Teams/Meet/Zoom call it transcribes both sides of the
conversation, keeps a rolling summary, and quietly suggests an answer in a
floating overlay when someone asks you a question and you go silent.

No cloud APIs, no API keys, no internet required once the models are
downloaded. Speech recognition runs locally via Parakeet TDT, and the language
model runs locally via Ollama.

## The idea

Detecting a question is easy. Firing a popup on every question is useless,
because most get answered instantly.

**Silence is the signal, not the question mark.** MeetingGenie only offers help
when a question is followed by roughly two seconds of you saying nothing. That
is a direct measurement of "stuck" rather than a guess, and nearly every design
decision in the system follows from it.

## How it works

```
Microphone ──► MicSource ──────┐
                               ├──► transcribe.py ──► Utterance
System audio ──► LoopbackSource┘      (Parakeet)         │
                                                         │
                       ┌─────────────────────────────────┤
                       ▼                                 ▼
                 summarize.py                       trigger.py
               (rolling summary)              (question? silence passed?)
                       │                               │
                       └──────────► brain.py ◄─────────┘
                                 (asks Ollama)
                                       │
                                       ▼
                                  overlay.py
                                (shows the answer)
```

Two separate capture streams solve the hardest problem in the system. The mic
only hears you; the other person comes out of your speakers. Capturing them
separately means speaker labels come for free, with no diarization model
needed, because the two voices arrive on physically separate wires.

The `Utterance` dataclass in `meeting_genie/types.py` is the contract between
the audio side and the brain side. Every tunable value lives in
`meeting_genie/config.yaml`. Nothing is hardcoded, by design.

## Design decisions worth knowing

**Threading priority is enforced in code.** The CPU is the scarce resource when
everything runs locally, so the order is: speech recognition never blocks,
one answer generation at a time, and the summarizer is allowed to run late.

**Silence needs a clock, not an event.** The trigger runs its own timer thread
that wakes ten times a second and measures wall-clock silence. An early version
tried to detect silence by waiting for utterances to arrive, which can't work:
the absence of an event is not itself an event.

**The silence timer measures the audio clock, not arrival time.** Each
`Utterance` records when speech actually stopped. Measuring from when the
transcript *arrived* would silently stack the recognition time on top of the
two-second window, so a 2000 ms setting really meant 3500 ms or worse.

**Different models for different jobs.** The summarizer uses `llama3.1:8b`,
which is better at condensing. The answer generator uses `llama3.2`, which is
smaller and faster, because a late suggestion is useless even if it's better
written.

**Bounded-size memory.** The summary compounds forward indefinitely (previous
summary plus new utterances, never the whole meeting) while staying capped at
eight bullets.

## Honest limitations

- **Windows only** for the full experience. Loopback capture uses WASAPI, a
  Windows API. Mic capture and everything downstream is cross-platform.
- **Headphones required.** Without them the mic hears the speakers and both
  tracks contain the same voice about 200 ms apart, which breaks transcription
  and the "is the user speaking" signal. Echo cancellation is designed but not
  built.
- **Latency is a few seconds.** Acceptable for a demo, not yet conversational.
- **Mic-side transcription is worse than loopback-side.** The loopback track is
  a clean digital signal; the mic picks up a real room.
- **Accented English still gets misheard** on ordinary words.
- `correction.py` is built and tested but deliberately not wired into the
  pipeline. It did phonetic glossary matching and worked, but it can only ever
  fix words listed in the glossary, so it wasn't worth the architectural
  weight.

## Setup

Requires **Python 3.10 or newer** and [Ollama](https://ollama.com).

```bash
pip install -r requirements.txt

ollama pull llama3.2      # answer generation
ollama pull llama3.1:8b   # rolling summary

python -m meeting_genie.main
```

The first run downloads the speech model (~645 MB) from Hugging Face and will
appear to hang for a few minutes. Cold start after that is about ten seconds.

**Controls:** `Ctrl+Shift+Space` reveals a pending answer, or forces a fresh
one if nothing is pending. `Ctrl+Shift+Q` quits. Both are configurable in
`config.yaml`.

## Developing without a mic

```bash
python tools/fake_queue.py
```

This replays a scripted meeting into the queue, so the whole text side
(trigger, brain, overlay, summarize) can be developed with no microphone, no
speech model, and no Windows.

## Project layout

| File | Job |
|---|---|
| `audio.py` | Captures mic and system audio as two separate streams |
| `transcribe.py` | Audio to text, segmentation, latency instrumentation |
| `trigger.py` | Scores whether text is a question, times real silence |
| `summarize.py` | Rolling bullet summary on its own thread |
| `brain.py` | Builds the prompt, streams the answer, cancels superseded ones |
| `overlay.py` | Floating window, four-state machine, never steals focus |
| `platformwindow.py` | Win32 window flags (topmost, no-activate) |
| `main.py` | Wires everything together, owns the hotkey |
| `types.py` | The `Utterance` contract |
| `config.yaml` | Every tunable value |

## Status

Working end to end and demoed successfully. Speech recognition was migrated
from faster-whisper to Parakeet TDT for speed; latency instrumentation is in
place and writes to `meetings/latency.jsonl`.

Known open items: the mic silence threshold is not calibrated against varied
hardware, there's no macOS loopback implementation, and there's no packaged
installer yet.

## Credits

Built by [Devansh Dhanuka](https://github.com/devanshdhanuka14) and
[Meesha Maheshwari](https://github.com/Meesha16).