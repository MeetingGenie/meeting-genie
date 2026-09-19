"""Audio -> text. Parakeet TDT via onnx-asr (swapped from faster-whisper).

WHY THE SWAP
  - Parakeet TDT is several times faster than small.en on CPU, and CPU is the
    scarce resource in a local-only pipeline (HANDOFF.md, "Everything local").
  - It has a different failure mode (transducer, not encoder-decoder), so it
    should not reproduce bug #4 - Whisper echoing its own initial_prompt back
    as "speech" when fed near-silent audio.

WHAT THE SWAP COSTS - read this before wondering where things went
  - NO initial_prompt. Parakeet has no vocabulary-biasing equivalent, so
    glossary.terms no longer reaches the STT stage at all. HANDOFF.md parked
    correction.py partly BECAUSE initial_prompt covered the glossary for free.
    That justification is now gone. Flagged, not silently reversed.
  - confidence is now derived from Parakeet token logprobs instead of
    Whisper's avg_logprob. The 0..1 mapping is NOT calibrated - logprob_scale
    in config.yaml is a starting guess, exactly like the old hardcoded /5.0.
  - Parakeet has a hard ~20-30s audio limit per call (Whisper chunked
    internally). The Segmenter now force-cuts at transcribe.max_utterance_s.

NEW: latency instrumentation. Every stage is timed and logged, because every
latency number in STATUS.md so far has been a guess.
"""
import json
import time

import numpy as np
from scipy.signal import resample_poly


def bytes_to_samples(audio_bytes, channels=1):
    samples = np.frombuffer(audio_bytes, dtype=np.int16)
    # Convert stereo -> mono
    if channels > 1:
        samples = samples.reshape(-1, channels)
        samples = samples.mean(axis=1)
    return samples


def resample_to_16k(samples, original_rate):
    # float64 cast is load-bearing: scipy 1.13.1 + numpy <2.3 silently returns
    # all zeros for int16 input. See HANDOFF.md bug #2. Do not remove.
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
    samples = bytes_to_samples(
        chunk.audio,
        chunk.channels,
    )
    samples = resample_to_16k(samples, chunk.sample_rate)
    # Remove DC offset which can confuse the model and VAD.
    if samples.size:
        samples = samples - float(np.mean(samples))
        # Gentle normalization: boost very low-energy signals slightly so the
        # ASR model has a better chance, but avoid strong amplification of
        # background noise. Use conservative thresholds.
        energy = rms(samples)
        if energy > 0 and energy < 400.0:
            gain = min(2.0, 400.0 / (energy + 1e-6))
            samples = samples * gain
    return samples


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
    it hands back a finished utterance once the speaker pauses.

    Returns (audio, start_ts, audio_end_ts, silence_after_ms).

    audio_end_ts is the monotonic time the LAST LOUD CHUNK ended. That is the
    real "when did they stop talking" moment, and it is what trigger.py now
    measures silence against. It is deliberately NOT derived from
    len(audio)/16000, because the buffer only holds loud chunks - short pauses
    inside an utterance would make that under-count.
    """

    def __init__(self, cfg, source):
        self.cfg = cfg
        self.source = source
        t = cfg["transcribe"]
        # mic and loopback have very different noise floors - a shared
        # threshold made the mic classify room noise as speech, which the STT
        # model then hallucinated into text. See HANDOFF.md bug #4.
        key = "mic_silence_rms" if source == "mic" else "loopback_silence_rms"
        self.silence_rms = t[key]
        self.segment_silence_ms = t["segment_silence_ms"]
        # Parakeet has a hard per-call audio limit. Whisper did not, so this
        # guard is new. Without it, one long uninterrupted monologue silently
        # produces garbage or an error instead of text.
        self.max_utterance_s = float(t.get("max_utterance_s", 20))
        self.chunk_ms = cfg["audio"]["vad_frame_ms"]
        self.last_feed_was_speech = False
        self._buffer = []           # list of numpy arrays, current utterance
        self._buffered_samples = 0  # running count, so we don't re-sum every chunk
        self._silence_ms = 0        # how long we've been quiet since last speech
        self._start_ts = None
        self._last_voice_ts = None

    def feed(self, samples, ts):
        """samples: 1-D numpy array, already resampled to 16k. ts: chunk time.
        Returns (audio, start_ts, audio_end_ts, silence_after_ms) when an
        utterance is ready, else None."""
        loud = rms(samples) >= self.silence_rms
        self.last_feed_was_speech = loud

        if loud:
            if self._start_ts is None:
                self._start_ts = ts
            self._buffer.append(samples)
            self._buffered_samples += len(samples)
            self._last_voice_ts = ts
            self._silence_ms = 0

            # Force-cut before we exceed what the model can accept in one call.
            if self._buffered_samples / 16000.0 >= self.max_utterance_s:
                return self._cut(silence_after_ms=0)
            return None

        if not self._buffer:
            return None  # silence before any speech started

        # Keep quiet frames within an active sentence. Discarding each
        # below-threshold frame chops low-volume words and natural pauses out
        # of the waveform before Parakeet receives it.
        self._buffer.append(samples)
        self._buffered_samples += len(samples)
        self._silence_ms += self.chunk_ms
        if self._silence_ms < self.segment_silence_ms:
            return None  # not quiet long enough yet

        return self._cut(silence_after_ms=self._silence_ms)

    def _cut(self, silence_after_ms):
        audio = np.concatenate(self._buffer)
        start_ts = self._start_ts
        # The last loud chunk covers chunk_ms of audio, so speech ended at its
        # timestamp plus its own duration.
        audio_end_ts = (self._last_voice_ts or start_ts) + (self.chunk_ms / 1000.0)

        self._buffer = []
        self._buffered_samples = 0
        self._start_ts = None
        self._last_voice_ts = None
        self._silence_ms = 0
        return audio, start_ts, audio_end_ts, silence_after_ms


# ---------------------------------------------------------------------------
# STT: turns a finished utterance's audio into text + confidence.
# ---------------------------------------------------------------------------

_asr_model = None

# Debug dumping controls
_debug_dumped = 0

def _maybe_dump_mic_debug(pre_samples, post_samples, text, confidence, cfg):
    """Write example pre/post mic WAVs and a small JSON metadata file when
    debug.dump_mic_examples is enabled in config. Limits to debug.dump_count
    files to avoid filling disk."""
    global _debug_dumped
    dbg_cfg = cfg.get("debug", {}) if isinstance(cfg, dict) else {}
    if not dbg_cfg.get("dump_mic_examples", False):
        return
    max_count = int(dbg_cfg.get("dump_count", 8))
    if _debug_dumped >= max_count:
        return
    try:
        from pathlib import Path
        import wave
        import json

        out_dir = Path(cfg.get("output", {}).get("meetings_dir", "meetings")) / "debug_mic_examples"
        out_dir.mkdir(parents=True, exist_ok=True)

        idx = _debug_dumped + 1
        pre_path = out_dir / f"mic_{idx:02d}_pre.wav"
        post_path = out_dir / f"mic_{idx:02d}_post.wav"
        meta_path = out_dir / f"mic_{idx:02d}.json"

        # samples expected as numpy arrays at 16k, float-like in int16 range
        def write_wav(path, samples):
            arr = np.asarray(samples)
            # clip to int16 range
            arr_i16 = np.clip(arr, -32768, 32767).astype(np.int16)
            with wave.open(str(path), "wb") as h:
                h.setnchannels(1)
                h.setsampwidth(2)
                h.setframerate(16000)
                h.writeframes(arr_i16.tobytes())

        write_wav(pre_path, pre_samples)
        write_wav(post_path, post_samples)

        meta = {
            "text": text,
            "confidence": confidence,
            "pre_path": str(pre_path),
            "post_path": str(post_path),
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

        _debug_dumped += 1
        print(f"[debug] dumped mic examples: {pre_path} {post_path}")
    except Exception:
        pass


def _get_model(cfg):
    """Loads once, lazily. First call downloads the model from Hugging Face if
    it isn't cached yet - that can take a few minutes, so main.py's startup
    will look frozen the very first run. Pre-warm it before a demo."""
    global _asr_model
    if _asr_model is None:
        import onnx_asr
        import onnxruntime as rt

        t = cfg["transcribe"]

        # Same reason cpu_threads existed for Whisper: unconstrained, the STT
        # model grabs every core and starves the real-time audio callback of
        # scheduling time -> "input overflow". See HANDOFF.md bug #3.
        # onnxruntime spells it intra_op_num_threads.
        so = rt.SessionOptions()
        so.intra_op_num_threads = int(t.get("cpu_threads", 4))
        so.inter_op_num_threads = 1

        model = onnx_asr.load_model(
            t["model"],
            t.get("model_dir") or None,
            quantization=t.get("quantization") or None,
            sess_options=so,
        )

        # with_timestamps() is what exposes per-token logprobs, which is the
        # only way to get a confidence number out of Parakeet. If we ever stop
        # needing confidence, turning this off is slightly cheaper.
        if t.get("with_timestamps", True):
            model = model.with_timestamps()

        _asr_model = model
    return _asr_model


def prewarm(cfg):
    """Load the model and run one tiny inference, so the first REAL utterance
    doesn't eat the cold-start cost. Call this at startup. brain.py already
    does the equivalent for Ollama."""
    t0 = time.perf_counter()
    model = _get_model(cfg)
    silence = np.zeros(16000, dtype=np.float32)
    try:
        model.recognize(silence, sample_rate=16000)
    except Exception as exc:
        print(f"[stt] prewarm inference failed (not fatal): {exc}")
    print(f"[stt] model ready in {(time.perf_counter() - t0) * 1000:.0f}ms")


def _confidence_from_logprobs(logprobs, cfg):
    """Parakeet gives per-token logprobs (negative, closer to 0 = more sure).
    Same shape of mapping we used for Whisper's avg_logprob.

    HONEST NOTE: the divisor was tuned for Whisper and is very likely wrong
    for a transducer. It lives in config.yaml so it can be re-tuned once we
    have real transcripts to compare against. Until then, treat confidence as
    a relative signal, not an absolute one.
    """
    if logprobs is None:
        return 0.0  # unknown, per the types.py contract
    arr = np.asarray(logprobs, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0.0
    scale = float(cfg["transcribe"].get("logprob_scale", 5.0))
    return float(max(0.0, min(1.0, 1.0 + float(arr.mean()) / scale)))


def transcribe_audio(audio_16k, cfg):
    """audio_16k: 1-D numpy array at 16kHz, raw int16-range floats.
    Returns (text, confidence, stt_ms)."""
    model = _get_model(cfg)
    # onnx-asr wants float32 in [-1, 1], not raw int16-range floats.
    audio = (np.asarray(audio_16k) / 32768.0).astype(np.float32)

    t0 = time.perf_counter()
    result = model.recognize(audio, sample_rate=16000)
    stt_ms = (time.perf_counter() - t0) * 1000.0

    # with_timestamps() -> TimestampedResult(.text/.tokens/.timestamps/.logprobs)
    # without it        -> a plain str
    text = getattr(result, "text", result)
    text = (text or "").strip()
    confidence = _confidence_from_logprobs(getattr(result, "logprobs", None), cfg)
    return text, confidence, stt_ms


# ---------------------------------------------------------------------------
# Latency instrumentation. Every number in STATUS.md was a guess until this.
# ---------------------------------------------------------------------------

def _log_latency(cfg, record):
    log_cfg = cfg.get("logging", {})
    if not log_cfg.get("latency", True):
        return
    print(
        "[latency] {source:8} audio={audio_ms:6.0f}ms  queue_wait={queue_wait_ms:6.1f}ms  "
        "stt={stt_ms:7.1f}ms  rtf={rtf:.2f}  stale_at_delivery={stale_ms:7.1f}ms".format(**record)
    )
    path = log_cfg.get("latency_file")
    if not path:
        return
    try:
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass  # instrumentation must never break the pipeline


# ---------------------------------------------------------------------------
# Final assembly: audio.py's source -> our speaker, STT output -> Utterance.
# ---------------------------------------------------------------------------

SOURCE_TO_SPEAKER = {"mic": "me", "loopback": "them"}


def make_utterance(source, text, confidence, start_ts, audio_end_ts, silence_after_ms):
    from meeting_genie.types import Utterance
    return Utterance(
        speaker=SOURCE_TO_SPEAKER[source],
        text=text,
        ts=start_ts,
        silence_after_ms=silence_after_ms,
        confidence=confidence,
        audio_end_ts=audio_end_ts,
    )


def process_segment(source, audio_16k, start_ts, audio_end_ts, silence_after_ms, cfg):
    """One finished utterance's audio -> (Utterance, stt_ms).
    Utterance is None if the model heard nothing (silence misdetected as
    speech, etc) - stt_ms is still returned so we can measure those too."""
    # audio_16k: int16-range floats at 16kHz. Apply per-utterance cleaning
    # before calling the ASR model. This helps with mic recordings that have
    # rumble, DC offset, or low RMS compared to loopback.
    try:
        # Keep a copy of pre-cleaned samples for debug dumping
        pre_samples = np.asarray(audio_16k, dtype=np.float64).copy()
        samples = pre_samples.copy()
        # Remove any remaining DC offset
        if samples.size:
            samples = samples - float(np.mean(samples))

        # For mic source, apply a light high-pass filter to remove rumble
        # and a per-utterance RMS normalization to improve ASR clarity.
        if source == "mic":
            try:
                from scipy.signal import butter, filtfilt

                sr = 16000
                cutoff = float(cfg.get("transcribe", {}).get("highpass_hz", 80.0))
                b, a = butter(1, float(cutoff) / (sr / 2.0), btype="high", analog=False)
                # filtfilt wants float64
                samples = filtfilt(b, a, samples)
            except Exception:
                # best-effort: if filtering fails, continue without it
                pass

            # RMS-target normalization (int16-scale). Use conservative target
            # and gain cap to avoid turning background noise into speech-like
            # input that the ASR model will hallucinate.
            target_rms = float(cfg.get("transcribe", {}).get("target_rms", 3000.0))
            cur_rms = float(rms(samples)) if samples.size else 0.0
            if cur_rms > 0 and cur_rms < target_rms:
                gain = min(3.0, target_rms / (cur_rms + 1e-6))
                samples = samples * gain

        # Convert back to int16-range floats for the recognizer path.
        audio_16k = samples
        # Debug dump pre/post mic audio when requested and when source is mic
        if source == "mic":
            try:
                _maybe_dump_mic_debug(pre_samples, samples, None, None, cfg)
            except Exception:
                pass
    except Exception:
        # Cleaning must not break the pipeline - fall back to raw audio
        pass

    text, confidence, stt_ms = transcribe_audio(audio_16k, cfg)
    # If debug dumping is enabled, write the pre/post audio along with the
    # resulting transcript and confidence for offline inspection.
    if source == "mic":
        try:
            _maybe_dump_mic_debug(pre_samples, audio_16k, text, confidence, cfg)
        except Exception:
            pass
    if not text:
        return None, stt_ms
    u = make_utterance(source, text, confidence, start_ts, audio_end_ts, silence_after_ms)
    return u, stt_ms


def _stt_worker(work_queue, cfg, on_utterance):
    """Runs on its own thread. Pulls finished segments off work_queue and does
    the slow STT call HERE - so the capture loop never waits on transcription
    and never misses incoming audio while the model is busy. See HANDOFF.md
    bug #3; this split is what fixed dropped segments."""
    while True:
        item = work_queue.get()
        if item is None:  # sentinel to stop the thread
            break
        source, audio, start_ts, audio_end_ts, silence_ms, queued_at = item

        queue_wait_ms = (time.monotonic() - queued_at) * 1000.0
        u, stt_ms = process_segment(
            source, audio, start_ts, audio_end_ts, silence_ms, cfg
        )

        audio_ms = (len(audio) / 16000.0) * 1000.0
        _log_latency(cfg, {
            "source": source,
            "audio_ms": audio_ms,
            "queue_wait_ms": queue_wait_ms,
            "stt_ms": stt_ms,
            # real-time factor: <1.0 means we transcribe faster than the audio
            # plays. This is the single number that decides whether the model
            # swap was worth it.
            "rtf": (stt_ms / audio_ms) if audio_ms else 0.0,
            # how old the speech already is by the time the pipeline sees text.
            # trigger.py's 2000ms window starts counting from BEFORE this.
            "stale_ms": (time.monotonic() - audio_end_ts) * 1000.0,
            "empty": u is None,
        })

        if u is not None:
            on_utterance(u)


def run_transcription_loop(recorder, cfg, on_utterance, on_audio_activity=None):
    """recorder: a running audio.AudioRecorder. on_utterance: callback that
    takes one Utterance, called whenever mic or loopback produces a finished
    sentence. Runs forever until recorder.stop() is called elsewhere.

    STT happens on a SEPARATE thread from chunk reading. Segmentation
    (buffering + silence detection) is cheap and stays in this loop; the slow
    part gets handed off to _stt_worker via a queue. Without this split, one
    slow transcription call stalls get_mic_chunk/get_loopback_chunk and audio
    arriving during that stall is delayed or lost."""
    import queue
    import threading

    # If the recorder performed a mic calibration, use it to set a safer
    # mic silence threshold so VAD doesn't feed noisy room hum into STT.
    import copy

    cfg_local = copy.deepcopy(cfg)
    try:
        calibrated = None
        if hasattr(recorder, "get_calibrated_mic_rms"):
            calibrated = recorder.get_calibrated_mic_rms()
        if calibrated is not None:
            multiplier = float(cfg_local.get("audio", {}).get("noise_floor_multiplier", 3.0))
            existing = int(cfg_local.get("transcribe", {}).get("mic_silence_rms", 300))
            # Clamp the suggested threshold to avoid huge jumps that make VAD
            # either too insensitive or too sensitive. Limit the new value to
            # at most `existing * 5` so calibration can't explode behavior.
            suggested_raw = calibrated * multiplier
            suggested = int(max(existing, min(suggested_raw, existing * 5)))
            cfg_local.setdefault("transcribe", {})["mic_silence_rms"] = suggested
            print(f"[transcribe] using calibrated mic_silence_rms={suggested} (raw {suggested_raw})")
    except Exception:
        pass

    mic_seg = Segmenter(cfg_local, "mic")
    loop_seg = Segmenter(cfg_local, "loopback")
    work_queue = queue.Queue()

    worker = threading.Thread(
        target=_stt_worker, args=(work_queue, cfg, on_utterance), daemon=True,
        name="STTWorker",
    )
    worker.start()

    while recorder.is_running():
        got_something = False

        mic_chunk = recorder.get_mic_chunk()
        if mic_chunk is not None:
            got_something = True
            samples = prepare_chunk(mic_chunk)
            result = mic_seg.feed(samples, mic_chunk.timestamp)
            if on_audio_activity is not None and mic_seg.last_feed_was_speech:
                on_audio_activity("mic", mic_chunk.timestamp)
            if result is not None:
                audio, start_ts, audio_end_ts, silence_ms = result
                work_queue.put(
                    ("mic", audio, start_ts, audio_end_ts, silence_ms, time.monotonic())
                )

        loop_chunk = recorder.get_loopback_chunk()
        if loop_chunk is not None:
            got_something = True
            samples = prepare_chunk(loop_chunk)
            result = loop_seg.feed(samples, loop_chunk.timestamp)
            if on_audio_activity is not None and loop_seg.last_feed_was_speech:
                on_audio_activity("loopback", loop_chunk.timestamp)
            if result is not None:
                audio, start_ts, audio_end_ts, silence_ms = result
                work_queue.put(
                    ("loopback", audio, start_ts, audio_end_ts, silence_ms, time.monotonic())
                )

        if not got_something:
            time.sleep(0.01)

    work_queue.put(None)  # stop the worker thread cleanly
