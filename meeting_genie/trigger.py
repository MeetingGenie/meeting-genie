"""Track A. Decides WHEN the popup fires. Consumes Utterances, emits trigger events.

- Sentence-split every utterance FIRST (back-to-back questions arrive merged).
- Score each sentence per config.trigger.signals. Fire above threshold.
- Anchor auxiliary/interrogative checks to sentence START. 'contains' is how the
  prototype ended up needing a 30s cooldown.
- silence >= silence_ms with the me-track quiet is THE struggle signal.
- me-track goes hot -> cancel pending/in-flight generation (user_already_answering).
- Hotkey bypasses scoring entirely, always.
- (post-MVP, headphones=false) correlation gate: mic energy tracking the loopback
  envelope is the speakers, not the user.
"""
