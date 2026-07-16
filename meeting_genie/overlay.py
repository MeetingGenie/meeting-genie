"""Track A. The pill / popup. IRON RULE: never steal keyboard focus. Ever.

States: idle -> ready (corner pill) -> expanded (hotkey, streams answer, Esc closes)
        -> stale (unopened for stale_seconds -> fade and drop).
- Global hotkey from config. Quiet-mode toggle pauses popups, summaries continue.
"""
