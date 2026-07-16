"""Track B. Audio segments -> text via faster-whisper.

- Model from config (small.en, int8). Feed numpy directly: no temp WAV, no ffmpeg.
- Build initial_prompt from glossary.terms + prompt_extras (vocabulary biasing:
  the cheapest big win for accents and proper nouns).
- Emit Utterance with confidence = segment avg_logprob.
- NOTHING may block this path. Priority 1 in the whole system.
"""
