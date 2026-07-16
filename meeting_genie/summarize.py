"""Track B. Rolling summary. OWN THREAD - must never block transcription.

- Every interval_seconds: last chunk of transcript + previous summary -> new bullet summary.
- Priority 3: skip the tick if brain.py is mid-generation; running late is invisible.
- Ollama with format=json so output always parses.
- Append to meetings/<stamp>/summary.md.
"""
