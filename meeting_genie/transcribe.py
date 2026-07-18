import numpy as np
from scipy.signal import resample_poly

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