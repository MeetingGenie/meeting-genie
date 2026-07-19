import numpy as np
from scipy.signal import resample_poly
import time

def bytes_to_samples(audio_bytes):
    return np.frombuffer(audio_bytes, dtype=np.int16)

def resample_to_16k(samples, original_rate):
    samples = samples.astype(np.float64)
    return resample_poly(samples, 16000, original_rate)

from dataclasses import dataclass

@dataclass
class FakeChunk:
    """Stand-in for audio.py's AudioChunk, used only for testing on Mac.
    Same fields, so prepare_chunk() can't tell the difference. Delete once
    we can import the real one."""
    source: str
    timestamp: float
    sample_rate: int
    channels: int
    audio: bytes

def prepare_chunk(chunk):
    samples = bytes_to_samples(chunk.audio)
    return resample_to_16k(samples, chunk.sample_rate)

# ---------------------------------------------------------------------------
# Segmenter: buffers prepared (16k) samples per speaker, cuts an "utterance"
# once silence crosses segment_silence_ms. Energy-based VAD (RMS threshold),
# not full webrtcvad - simple, fast, good enough to start with.
# ---------------------------------------------------------------------------

def rms(samples):
    """Root-mean-square loudness of a chunk. Silence -> near 0."""
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples))))


class Segmenter:
    """One instance per source ("me" or "them"). Feed it prepared 16k chunks,
    it hands back a finished utterance (samples + how long the silence after
    it was) once the speaker pauses."""

    def __init__(self, cfg, source):
        self.cfg = cfg
        self.source = source
        t = cfg["transcribe"]
        # mic and loopback have very different noise floors - a shared
        # threshold made the mic classify room noise as speech, which Whisper
        # then hallucinated into repeats of the initial_prompt.
        key = "mic_silence_rms" if source == "mic" else "loopback_silence_rms"
        self.silence_rms = t[key]
        self.segment_silence_ms = t["segment_silence_ms"]
        self.chunk_ms = cfg["audio"]["vad_frame_ms"]
        self._buffer = []          # list of numpy arrays, current utterance so far
        self._silence_ms = 0       # how long we've been quiet since last speech
        self._start_ts = None

    def feed(self, samples, ts):
        """samples: 1-D numpy array, already resampled to 16k. ts: chunk time.
        Returns (audio, start_ts, silence_after_ms) when an utterance is ready,
        else None."""
        loud = rms(samples) >= self.silence_rms

        if loud:
            if self._start_ts is None:
                self._start_ts = ts
            self._buffer.append(samples)
            self._silence_ms = 0
            return None

        if not self._buffer:
            return None  # silence before any speech started

        self._silence_ms += self.chunk_ms
        if self._silence_ms < self.segment_silence_ms:
            return None  # not quiet long enough yet

        audio = np.concatenate(self._buffer)
        start_ts = self._start_ts
        silence_after_ms = self._silence_ms
        self._buffer = []
        self._start_ts = None
        self._silence_ms = 0
        return audio, start_ts, silence_after_ms


# ---------------------------------------------------------------------------
# Whisper: turns a finished utterance's audio into text + confidence.
# ---------------------------------------------------------------------------

_whisper_model = None


def _get_model(cfg):
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        t = cfg["transcribe"]
        _whisper_model = WhisperModel(t["model"], compute_type=t["compute_type"])
    return _whisper_model


def build_initial_prompt(cfg):
    terms = cfg.get("glossary", {}).get("terms", [])
    extras = cfg["transcribe"].get("prompt_extras", "")
    return (extras + " " + ", ".join(terms)).strip()


def transcribe_audio(audio_16k, cfg):
    """audio_16k: 1-D float64 numpy array at 16kHz. Returns (text, confidence)."""
    model = _get_model(cfg)
    # faster-whisper wants float32 in [-1, 1], not raw int16-range floats.
    audio_norm = (audio_16k / 32768.0).astype(np.float32)
    prompt = build_initial_prompt(cfg)

    segments, info = model.transcribe(
        audio_norm,
        initial_prompt=prompt,
        language="en",
    )
    segments = list(segments)
    if not segments:
        return "", 0.0

    text = " ".join(s.text.strip() for s in segments).strip()
    # avg_logprob is negative (closer to 0 = more confident). Convert to a
    # rough 0..1 confidence so the rest of the pipeline has an easy number.
    avg_logprob = sum(s.avg_logprob for s in segments) / len(segments)
    confidence = max(0.0, min(1.0, 1.0 + avg_logprob / 5.0))
    return text, confidence


# ---------------------------------------------------------------------------
# Final assembly: audio.py's source -> our speaker, Whisper output -> Utterance.
# ---------------------------------------------------------------------------

SOURCE_TO_SPEAKER = {"mic": "me", "loopback": "them"}


def make_utterance(source, text, confidence, start_ts, silence_after_ms):
    from meeting_genie.types import Utterance
    return Utterance(
        speaker=SOURCE_TO_SPEAKER[source],
        text=text,
        ts=start_ts,
        silence_after_ms=silence_after_ms,
        confidence=confidence,
    )


def process_segment(source, audio_16k, start_ts, silence_after_ms, cfg):
    """One finished utterance's audio -> a real Utterance, or None if Whisper
    heard nothing (silence misdetected as speech, etc)."""
    text, confidence = transcribe_audio(audio_16k, cfg)
    if not text:
        return None
    return make_utterance(source, text, confidence, start_ts, silence_after_ms)

def _whisper_worker(work_queue, cfg, on_utterance):
    """Runs on its own thread. Pulls finished segments off work_queue and
    does the (slow) Whisper call here - so the capture loop below never
    waits on transcription and never misses incoming audio while Whisper
    is busy. This is what fixed segments getting dropped/delayed."""
    while True:
        item = work_queue.get()
        if item is None:  # sentinel to stop the thread
            break
        source, audio, start_ts, silence_ms = item
        u = process_segment(source, audio, start_ts, silence_ms, cfg)
        if u is not None:
            on_utterance(u)


def run_transcription_loop(recorder, cfg, on_utterance):
    """recorder: a running audio.AudioRecorder. on_utterance: callback that
    takes one Utterance, called whenever mic or loopback produces a finished
    sentence. Runs forever until recorder.stop() is called elsewhere.

    Whisper transcription happens on a SEPARATE thread from chunk reading.
    Segmentation (buffering + silence detection) is cheap and stays in this
    loop; the slow part (process_segment, which calls faster-whisper) gets
    handed off to _whisper_worker via a queue. Without this split, a single
    slow transcription call would stall get_mic_chunk/get_loopback_chunk and
    audio arriving during that stall would be delayed or lost."""
    import queue
    import threading

    mic_seg = Segmenter(cfg, "mic")
    loop_seg = Segmenter(cfg, "loopback")
    work_queue = queue.Queue()

    worker = threading.Thread(
        target=_whisper_worker, args=(work_queue, cfg, on_utterance), daemon=True
    )
    worker.start()

    while recorder.is_running():
        got_something = False

        mic_chunk = recorder.get_mic_chunk()
        if mic_chunk is not None:
            got_something = True
            samples = prepare_chunk(mic_chunk)
            result = mic_seg.feed(samples, mic_chunk.timestamp)
            if result is not None:
                audio, start_ts, silence_ms = result
                work_queue.put(("mic", audio, start_ts, silence_ms))

        loop_chunk = recorder.get_loopback_chunk()
        if loop_chunk is not None:
            got_something = True
            samples = prepare_chunk(loop_chunk)
            result = loop_seg.feed(samples, loop_chunk.timestamp)
            if result is not None:
                audio, start_ts, silence_ms = result
                work_queue.put(("loopback", audio, start_ts, silence_ms))

        if not got_something:
            time.sleep(0.01)

    work_queue.put(None)  # stop the worker thread cleanly