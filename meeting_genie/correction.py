"""Track B. Fix ASR errors deterministically before anyone downstream sees them.

- Fuzzy-match each unknown token against glossary terms (RapidFuzz / Double Metaphone).
  Above config threshold -> replace. Sub-millisecond, no LLM.
- Confidence gate: segments above correction.confidence_gate skip correction entirely.
- (post-MVP, headphones=false mode) Echo dedup: mic-track text that fuzzy-matches
  loopback-track text within echo_dedup_window_ms is the speakers, not the user. Drop it.
"""
