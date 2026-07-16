"""PRODUCES: Utterance-shaped raw audio segments into the pipeline.

Phase 0. Built as a pair on Meesha's Windows machine (loopback is Windows-only).

Owns:
- MicSource   -> speaker="me".   sounddevice, cross-platform (works on the Mac too).
- LoopbackSource -> speaker="them". PyAudioWPatch WASAPI loopback. WINDOWS ONLY:
  raise NotImplementedError("LoopbackSource is Windows-only for now") on other platforms.
- Startup noise-floor calibration (never hardcode a silence threshold).
- VAD at 20-30ms frames. Timestamps from time.monotonic(), SAME clock for both sources.
- Resample everything to 16kHz mono before it leaves this module.

Gotchas (learned the hard way / from research):
- WASAPI loopback fires NO callback when nothing is playing. Silence timers must be
  wall-clock based, never sample-count based.
- Loopback captures ALL system audio. Mute Spotify/Slack before demos.
- Bluetooth headset MICS force Windows into HFP (8-16kHz). Wired headset, or BT
  headphones for output + built-in mic.
"""
