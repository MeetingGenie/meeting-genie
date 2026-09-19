
import sys, time, yaml, numpy as np
sys.path.insert(0, '.')
from meeting_genie.types import Utterance
from meeting_genie.trigger import Trigger
from meeting_genie.transcribe import Segmenter

cfg = yaml.safe_load(open("meeting_genie/config.yaml"))
assert "compute_type" not in cfg["transcribe"], "old whisper keys still present"
print("config OK:", cfg["transcribe"]["model"], "| max_utterance_s", cfg["transcribe"]["max_utterance_s"])

Utterance(speaker="them", text="x", ts=1.0, silence_after_ms=600, confidence=0.9)
print("types.py backward-compatible OK")

seg = Segmenter(cfg, "mic"); t = 100.0
for _ in range(5): seg.feed(np.full(960, 5000.0), t); t += 0.06
last = t - 0.06
for _ in range(12):
    r = seg.feed(np.zeros(960), t); t += 0.06
    if r: break
assert abs(r[2] - (last + 0.06)) < 1e-9, "audio_end_ts wrong"
print(f"segmenter OK: audio_end_ts={r[2]:.2f} silence={r[3]}ms")

fired = []
tr = Trigger(cfg, on_trigger=lambda q: fired.append(time.monotonic())); tr.start()
now = time.monotonic(); t0 = now
tr.feed(Utterance("them", "What is the deployment timeline?", now-3.0, 600, 0.9, audio_end_ts=now-1.5))
while not fired and time.monotonic()-t0 < 4: time.sleep(0.02)
tr.stop()
w = fired[0]-t0
print(f"trigger OK: waited {w*1000:.0f}ms more, total real silence {(w+1.5)*1000:.0f}ms")
assert 0.35 < w < 0.75, f"FIX NOT ACTIVE - waited {w:.2f}s"
print("\nALL PASS")