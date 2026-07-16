"""Track A. Talks to Ollama. Answer generation + scheduling.

- Max ONE generation in flight; newer trigger supersedes and cancels older.
- stream=true, render tokens as they arrive. num_predict cap from config.
- Pre-warm on startup so the first real answer isn't paying model-load time.
- Priority 2: below whisper, above summarizer.
- Log every generation to meetings/<stamp>/suggestions.json with its trigger reason.
"""
