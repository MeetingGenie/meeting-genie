"""Wiring only. Sources -> queue -> transcribe -> correction -> {trigger->brain->overlay, summarize}.

Threads: audio callbacks | transcribe (never blocked) | brain worker | summarize | UI main.
Creates meetings/<yyyy-mm-dd_hhmm>/ and hands the path to summarize + brain.
No business logic in this file.
"""
