# MeetingGenie

A live meeting copilot that runs entirely on your own machine.

During a real Teams/Meet/Zoom call it listens to both sides of the
conversation, transcribes them in real time, keeps a running summary, and
quietly puts a suggested answer on screen when someone asks you something and
you go silent.

Nothing leaves the laptop. No cloud APIs, no API keys, no internet needed once
the models are downloaded.

## The problem it solves

You're on a call. Someone asks you something you half-know the answer to. By
the time you've opened a tab and found it, the moment has passed and you've
said "let me get back to you on that."

The obvious version of this tool detects questions and answers them. That
version is useless. Most questions in a meeting get answered in under a
second, so an assistant that reacts to every question mark spends its time
interrupting people who were already fine.

**What actually signals you're stuck is the silence afterwards.** Someone asks
you something, and then you say nothing for two seconds. That's not a guess
about your state, it's a measurement of it. Almost every design decision below
follows from that one idea.

## How it works

### 1. Capturing both sides of the conversation

The hardest problem is that a microphone only hears you. The other person's
voice comes out of your speakers, and the mic doesn't reliably pick it up.

The fix is two completely separate capture streams:

- **Your voice** comes from the microphone, via `sounddevice`
- **Their voice** is captured from the system audio output itself, using
  WASAPI loopback — a Windows feature that lets an app record whatever is
  being played through the speakers

This means knowing who said what is free. There's no speaker-identification
model anywhere in the system, because the two voices arrive on physically
different wires. Everything from the mic is tagged "me", everything from
loopback is tagged "them", and that's the whole mechanism.

A few things this design has to handle: loopback fires no callback at all when
nothing is playing, so every silence timer uses the wall clock rather than
counting audio samples. System audio also runs at whatever rate the device
prefers (usually 48kHz stereo), so it gets resampled down to 16kHz mono before
anything else touches it.

### 2. Turning audio into sentences

Audio arrives in 30-millisecond frames. Each frame gets a loudness measurement,
and frames are buffered until the speaker pauses for **600 milliseconds** —
that's the cut point for one utterance.

Mic and loopback get **separate loudness thresholds**, because a microphone in
a real room and a clean digital audio stream are nothing alike. A single shared
threshold meant room noise on the mic side was being classified as speech and
sent to the recognizer, which then hallucinated words out of nothing. At
startup the app also measures the actual noise floor for two seconds and sets
the threshold above it, rather than trusting a hardcoded number.

Completed utterances go to **Parakeet TDT 0.6B**, an on-device speech
recognition model running through ONNX Runtime at int8 precision. On CPU it
transcribes several times faster than the audio plays.

Recognition runs on **its own background thread fed by a queue**. The capture
loop only does cheap work — read a frame, measure it, buffer it — and never
waits for transcription to finish. Without that split, a slow recognition call
stalls audio capture and frames get dropped outright.

The result is an `Utterance`: who spoke, what they said, when the speech
started, when it *stopped*, how long the silence after it was, and a
confidence score derived from the model's own token probabilities.

### 3. Deciding whether to speak up

Two independent mechanisms, deliberately kept separate.

**First, is this even a question?** Each utterance from the other person gets
scored, and only clears the bar at 3.0 points:

| Signal | Points |
|---|---|
| Ends with a question mark | +3.0 |
| Starts with what / why / how / when / where / who | +3.0 |
| Starts with is / are / can / could / should / will | +2.5 |
| Sensible sentence length (4–40 words) | +0.5 |
| Ends with a tag like "...right?" or "...make sense?" | **−3.0** |
| Sounds like their own intent: "should I share my screen" | **−2.0** |
| You've already started answering | **−5.0** |

The negative weights matter as much as the positive ones. "That works, right?"
is a question mark that doesn't want an answer. Greetings get stripped off the
front first, and the interrogative check only looks at the first few words —
an early version matched "is" anywhere in a sentence and fired on "this is not
working."

**Second, have you actually gone quiet?** A separate timer thread wakes ten
times a second and checks the wall clock. It only fires when a question is
pending *and* 2000 milliseconds of real silence have passed.

These are separate on purpose. Silence was originally just another scored
signal, which did nothing at all — a real question already scores past the
threshold without it, so silence never changed the outcome. It has to be a
gate, not a point value.

One subtle detail worth knowing: the silence clock measures from **when the
speech ended**, not when the transcript arrived. Those are different moments,
separated by however long recognition took. Measuring from arrival silently
stacked recognition time on top of the window, so a 2000ms setting was really
waiting 3500ms or more.

### 4. Writing the answer

When the trigger fires, the language model gets three things: the pending
question, the recent transcript, and the running summary of the meeting so far.

It runs on **Ollama with llama3.2** at a low temperature, capped at 180 tokens.
The system prompt tells it to write sentences the user could say out loud —
no preamble, no markdown, no summarizing the meeting back at them. It's also
told the transcript comes from imperfect speech recognition and that proper
nouns may be garbled, so it should infer intent from context.

Answers stream in token by token. If a newer question arrives mid-generation,
the old one is cancelled — each generation checks its own ID against the
current one on every token. The model is pre-warmed at startup so the first
real answer doesn't pay the load cost.

### 5. Keeping a running summary

Every 180 seconds, a background thread sends the previous summary plus only
the new lines since then to **llama3.1:8b** — a bigger model than the answer
generator, because condensing rewards quality over speed.

It never sends the whole meeting. The summary compounds forward indefinitely
while staying capped at 8 bullets, so a two-hour call costs the same as a
ten-minute one. This thread is the lowest priority in the system; if it runs a
few seconds late, nobody notices.

### 6. Showing it without getting in the way

The overlay is a floating window with four states: idle, ready (a small pill),
loading, and expanded with the streaming answer.

It **never steals keyboard focus** — on Windows that means specific window
flags that let it appear on top without stealing your typing or, worse,
pulling focus out of your call.

Nothing auto-expands. Answers queue silently into a buffer and wait. Only the
hotkey reveals them, so a suggestion appearing on screen is always something
you asked for.

## Design decisions

**Behavior is data.** Every threshold, weight, model name, interval, and prompt
lives in `config.yaml`. You can retune the entire system the night before a
demo without opening a `.py` file. This came directly from a prototype where a
hardcoded silence threshold made headset mics look permanently silent — an
invisible bug that cost hours.

**The CPU is the scarce resource,** since everything runs locally. Thread
priority is enforced in code: speech recognition is never blocked, only one
answer generates at a time, and the summarizer is allowed to run late.

**Different models for different jobs.** Summarizing rewards a bigger model;
answering rewards a faster one, because a late suggestion is useless even if
it's better written.

**Silence needs a clock.** The absence of speech cannot be detected by waiting
for speech to arrive. It requires something that wakes up on its own and looks
at the time.

## Setup

Requires **Python 3.10+** and [Ollama](https://ollama.com).

```bash
pip install -r requirements.txt

ollama pull llama3.2      # answer generation
ollama pull llama3.1:8b   # rolling summary

python -m meeting_genie.main
```

First run downloads the speech model (~645 MB) and will look frozen for a few
minutes. Cold start after that is around ten seconds.

**Headphones are required.** Without them the mic picks up your speakers, both
tracks end up containing the same voice about 200ms apart, and the "is the user
speaking" signal breaks entirely.

### Controls

- `Ctrl+Shift+Space` — reveal a waiting answer, or force a fresh one if
  nothing is pending
- `Ctrl+Shift+Q` — quit

Both configurable in `config.yaml`.

### Developing without a microphone

```bash
python tools/fake_queue.py
```

Replays a scripted meeting into the pipeline, so the trigger, answer
generation, summary and overlay can all be developed with no mic, no speech
model, and no Windows.

## Project layout

| File | Job |
|---|---|
| `audio.py` | Mic and system audio capture as two separate streams |
| `transcribe.py` | Audio to text, segmentation, latency instrumentation |
| `trigger.py` | Question scoring plus the wall-clock silence gate |
| `summarize.py` | Rolling bullet summary on its own thread |
| `brain.py` | Prompt construction, streaming generation, cancellation |
| `overlay.py` | Floating window, four states, never steals focus |
| `platformwindow.py` | Windows-specific window flags |
| `main.py` | Wires everything together, owns the hotkey |
| `types.py` | The `Utterance` contract between the two halves |
| `config.yaml` | Every tunable value in the system |

## Status

Working end to end. Every utterance's timings are logged to
`meetings/latency.jsonl`, and each meeting's transcript, summary and generated
suggestions are written to `meetings/`.

Currently open: the microphone loudness threshold isn't calibrated across
varied hardware, there's no macOS implementation of loopback capture, echo
cancellation isn't built (hence the headphones requirement), and there's no
packaged installer yet.

## Credits

Built by [Devansh Dhanuka](https://github.com/devanshdhanuka14) and
[Meesha Maheshwari](https://github.com/Meesha16).
