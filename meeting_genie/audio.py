
"""Production-grade audio capture service for MeetingGenie.

This module records microphone and WASAPI loopback audio for live meetings.
It writes PCM bytes straight to disk without resampling or extra processing so
that loopback stays faithful to the device stream and can run indefinitely.
"""

from __future__ import annotations

import logging
import platform
import queue
from queue import Empty
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd
import yaml

if platform.system() != "Windows":
    raise NotImplementedError("MeetingGenie audio capture is only implemented on Windows.")

try:
    import pyaudiowpatch as pyaudio
except ImportError as exc:  # pragma: no cover - environment dependent
    pyaudio = None
    _PYAUDIO_IMPORT_ERROR = exc
else:
    _PYAUDIO_IMPORT_ERROR = None

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AudioConfig:
    """Configuration for live audio capture."""

    sample_rate_output: int = 16000
    chunk_duration_ms: int = 20
    channels: int = 1
    queue_size: int = 256
    mic_device_index: int | None = None
    loopback_device_index: int | None = None

    @classmethod
    def from_config(cls) -> "AudioConfig":
        """Load audio settings from the repository config file."""
        config_path = Path(__file__).resolve().with_name("config.yaml")
        data: dict[str, Any] = {}
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
                if isinstance(loaded, dict):
                    data = loaded

        audio_cfg = data.get("audio", {}) if isinstance(data.get("audio"), dict) else {}
        return cls(
            sample_rate_output=int(audio_cfg.get("sample_rate", 16000)),
            chunk_duration_ms=int(audio_cfg.get("vad_frame_ms", 20)),
            channels=int(audio_cfg.get("channels", 1)),
            queue_size=int(audio_cfg.get("queue_size", 256)),
            mic_device_index=audio_cfg.get("mic_device_index"),
            loopback_device_index=audio_cfg.get("loopback_device_index"),
        )


@dataclass(slots=True)
class AudioChunk:
    """A chunk of raw PCM audio captured from one source."""

    source: str
    timestamp: float
    sample_rate: int
    channels: int
    audio: bytes


class DeviceManager:
    """Resolve the active microphone and WASAPI loopback devices."""

    def __init__(self, config: AudioConfig | None = None) -> None:
        self.config = config or AudioConfig.from_config()
        if pyaudio is None:
            raise RuntimeError(
                "PyAudioWPatch is required for Windows loopback capture. "
                f"Original error: {_PYAUDIO_IMPORT_ERROR}"
            )
        self._pa = pyaudio.PyAudio()

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        """Release the PyAudio instance if it exists."""
        if getattr(self, "_pa", None) is not None:
            try:
                self._pa.terminate()
            except Exception:  # pragma: no cover - best effort cleanup
                logger.debug("PyAudio termination failed", exc_info=True)
            self._pa = None

    def list_devices(self) -> list[dict[str, Any]]:
        """Return a normalized view of all sounddevice audio devices."""
        devices: list[dict[str, Any]] = []
        for index, device in enumerate(sd.query_devices()):
            devices.append(
                {
                    "index": index,
                    "name": str(device.get("name", "")),
                    "host_api": device.get("hostapi"),
                    "max_input_channels": int(device.get("max_input_channels", 0)),
                    "max_output_channels": int(device.get("max_output_channels", 0)),
                    "default_samplerate": float(device.get("default_samplerate", 0.0)),
                }
            )
        return devices

    def print_diagnostics(self) -> None:
        """Print useful diagnostics for the current audio system."""
        devices = self.list_devices()
        print("\n=== SoundDevice devices ===")
        for device in devices:
            print(
                f"[{device['index']}] {device['name']} | "
                f"in={device['max_input_channels']} out={device['max_output_channels']}"
            )

        default_input, default_output = sd.default.device
        try:
            input_name = sd.query_devices(default_input)["name"]
        except Exception:
            input_name = "<unavailable>"
        try:
            output_name = sd.query_devices(default_output)["name"]
        except Exception:
            output_name = "<unavailable>"

        print(f"\nDefault input : {input_name}")
        print(f"Default output: {output_name}")

    def get_default_microphone(self) -> dict[str, Any]:
        """Resolve the default microphone device."""
        if self._pa is None:
            self._pa = pyaudio.PyAudio()

        default_input, _ = sd.default.device
        device_info = sd.query_devices(default_input)
        if not isinstance(device_info, dict):
            raise RuntimeError("The default microphone device could not be resolved.")

        name = str(device_info.get("name", ""))
        logger.info("Selected microphone device: %s", name)
        return {
            "index": int(default_input),
            "name": name,
            "default_samplerate": float(device_info.get("default_samplerate", self.config.sample_rate_output)),
            "max_input_channels": int(device_info.get("max_input_channels", self.config.channels)),
        }

    def get_default_loopback(self) -> dict[str, Any]:
        """Resolve a matching WASAPI loopback device for the current output."""
        if self._pa is None:
            self._pa = pyaudio.PyAudio()

        default_output = sd.default.device[1]
        try:
            output_device = sd.query_devices(default_output)
            output_name = str(output_device.get("name", ""))
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("The default output device could not be resolved.") from exc

        wasapi_index = None
        for index in range(self._pa.get_host_api_count()):
            api_info = self._pa.get_host_api_info_by_index(index)
            if api_info.get("type") == pyaudio.paWASAPI:
                wasapi_index = index
                break
        if wasapi_index is None:
            raise RuntimeError("WASAPI host API was not found. Loopback capture is unavailable.")

        preferred_name = self._normalize_name(output_name)
        for index in range(self._pa.get_host_api_info_by_index(wasapi_index)["deviceCount"]):
            device_info = self._pa.get_device_info_by_host_api_device_index(wasapi_index, index)
            if int(device_info.get("maxInputChannels", 0)) <= 0:
                continue
            name = str(device_info.get("name", ""))
            normalized_name = self._normalize_name(name)
            if self._is_loopback_candidate(normalized_name, preferred_name):
                logger.info("Selected loopback device: %s", name)
                return {
                    "index": int(device_info.get("index", index)),
                    "name": name,
                    "defaultSampleRate": float(device_info.get("defaultSampleRate", self.config.sample_rate_output)),
                    "maxInputChannels": int(device_info.get("maxInputChannels", self.config.channels)),
                }

        raise RuntimeError(
            "A matching WASAPI loopback device could not be found for output device "
            f"{output_name}."
        )

    @staticmethod
    def _normalize_name(name: str) -> str:
        return " ".join(name.lower().split()).replace("(loopback)", "")

    def _is_loopback_candidate(self, name: str, preferred_name: str) -> bool:
        if "loopback" in name:
            return True
        if not preferred_name:
            return False
        if preferred_name in name:
            return True
        return name.startswith(preferred_name) or preferred_name.startswith(name)


class AudioRecorder:
    """A long-running audio capture service for microphone and loopback streams."""

    def __init__(self, config: AudioConfig | None = None, device_manager: DeviceManager | None = None) -> None:
        self.config = config or AudioConfig.from_config()
        self.device_manager = device_manager or DeviceManager(self.config)

        self._mic_stream: Any | None = None
        self._loopback_stream: Any | None = None

        self._mic_queue: queue.Queue[AudioChunk] = queue.Queue(maxsize=self.config.queue_size)
        self._loopback_queue: queue.Queue[AudioChunk] = queue.Queue(maxsize=self.config.queue_size)
        self._stop_event = threading.Event()
        self._running = False

        self._mic_device: dict[str, Any] | None = None
        self._loopback_device: dict[str, Any] | None = None
        self._mic_sample_rate: int = self.config.sample_rate_output
        self._loopback_sample_rate: int = self.config.sample_rate_output
        self._mic_channels: int = 1
        self._loopback_channels: int = 1

        self._output_paths: dict[str, Path | None] = {"mic": None, "loopback": None}
        self._wav_handles: dict[str, Any] = {"mic": None, "loopback": None}
        self._writer_thread: threading.Thread | None = None

    def start(self, output_paths: dict[str, str | Path | None] | None = None) -> None:
        """Start microphone and loopback capture and begin writing audio to disk."""
        if self._running:
            return

        self._stop_event.clear()
        self._mic_device = self.device_manager.get_default_microphone()
        self._loopback_device = self.device_manager.get_default_loopback()
        self._mic_sample_rate = int(self._mic_device.get("default_samplerate", self.config.sample_rate_output))
        self._loopback_sample_rate = int(self._loopback_device.get("defaultSampleRate", self.config.sample_rate_output))
        self._mic_channels = max(1, min(int(self._mic_device.get("max_input_channels", self.config.channels)), 2))
        self._loopback_channels = max(1, min(int(self._loopback_device.get("maxInputChannels", self.config.channels)), 2))

        self._output_paths = {
            "mic": self._resolve_output_path(output_paths, "mic"),
            "loopback": self._resolve_output_path(output_paths, "loopback"),
        }
        self._open_output_files()

        self.start_mic()
        self.start_loopback()

        self._writer_thread = threading.Thread(target=self._write_loop, daemon=True)
        self._writer_thread.start()
        self._running = True
        logger.info("Audio capture service started")

    def stop(self) -> None:
        """Stop capture streams, flush pending audio, and close files."""
        if not self._running:
            return

        logger.info("Stopping audio capture service")
        self._stop_event.set()
        self.stop_mic()
        self.stop_loopback()

        if self._writer_thread is not None and self._writer_thread.is_alive():
            self._writer_thread.join(timeout=3.0)

        self._close_output_files()
        self._running = False

    def close(self) -> None:
        """Stop recording and release device resources."""
        self.stop()
        self.device_manager.close()

    def is_running(self) -> bool:
        """Return whether the recorder is currently running."""
        return self._running
    
    def get_mic_chunk(self, timeout: float = 0.1) -> AudioChunk | None:
        """
        Return the next microphone AudioChunk.

        Args:
            timeout: Seconds to wait for a chunk before returning None.

        Returns:
            The next AudioChunk if available, otherwise None.
        """
        try:
            return self._mic_queue.get(timeout=timeout)
        except queue.Empty:
            return None


    def get_loopback_chunk(self, timeout: float = 0.1) -> AudioChunk | None:
        """
        Return the next loopback AudioChunk.

        Args:
            timeout: Seconds to wait for a chunk before returning None.

        Returns:
            The next AudioChunk if available, otherwise None.
        """
        try:
            return self._loopback_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def mic_queue_size(self) -> int:
        return self._mic_queue.qsize()

    def loopback_queue_size(self) -> int:
        return self._loopback_queue.qsize()

    def start_mic(self) -> None:
        """Start the microphone input stream with SoundDevice."""
        if self._mic_stream is not None:
            return
        if self._mic_device is None:
            self._mic_device = self.device_manager.get_default_microphone()

        samplerate = int(self._mic_device.get("default_samplerate", self.config.sample_rate_output))
        blocksize = self._chunk_size_for_rate(samplerate)
        logger.info("Starting microphone stream at %s Hz with block size %s", samplerate, blocksize)

        self._mic_stream = sd.InputStream(
            device=int(self._mic_device.get("index", -1)),
            channels=self._mic_channels,
            samplerate=samplerate,
            blocksize=blocksize,
            dtype="int16",
            callback=self._mic_callback_impl,
        )
        self._mic_stream.start()
        logger.info("Microphone stream started")

    def start_loopback(self) -> None:
        """Start the WASAPI loopback stream with PyAudioWPatch."""
        if self._loopback_stream is not None:
            return
        if self._loopback_device is None:
            self._loopback_device = self.device_manager.get_default_loopback()

        rate = int(self._loopback_device.get("defaultSampleRate", self.config.sample_rate_output))
        frames_per_buffer = self._chunk_size_for_rate(rate)
        logger.info("Starting loopback stream at %s Hz with buffer %s", rate, frames_per_buffer)

        self._loopback_stream = self.device_manager._pa.open(
            format=pyaudio.paInt16,
            channels=self._loopback_channels,
            rate=rate,
            input=True,
            input_device_index=int(self._loopback_device.get("index", -1)),
            frames_per_buffer=frames_per_buffer,
            stream_callback=self._loopback_callback_impl,
        )
        self._loopback_stream.start_stream()
        logger.info("Loopback stream started")

    def stop_mic(self) -> None:
        """Stop the microphone stream if it exists."""
        if self._mic_stream is None:
            return
        try:
            self._mic_stream.stop()
            self._mic_stream.close()
        except Exception as exc:  # pragma: no cover - may vary by platform state
            logger.warning("Microphone stream shutdown produced an error: %s", exc)
        finally:
            self._mic_stream = None
        logger.info("Microphone stream stopped")

    def stop_loopback(self) -> None:
        """Stop the loopback stream if it exists."""
        if self._loopback_stream is None:
            return
        try:
            self._loopback_stream.stop_stream()
            self._loopback_stream.close()
        except Exception as exc:  # pragma: no cover - may vary by platform state
            logger.warning("Loopback stream shutdown produced an error: %s", exc)
        finally:
            self._loopback_stream = None
        logger.info("Loopback stream stopped")

    def _resolve_output_path(self, output_paths: dict[str, str | Path | None] | None, source: str) -> Path | None:
        if output_paths is not None and source in output_paths and output_paths[source] is not None:
            return Path(output_paths[source])
        return None

    def _open_output_files(self) -> None:
        for source, path in self._output_paths.items():
            if path is None:
                self._wav_handles[source] = None
                continue

            path.parent.mkdir(parents=True, exist_ok=True)
            handle = wave.open(str(path), "wb")
            handle.setnchannels(1)
            handle.setsampwidth(2)
            sample_rate = self._mic_sample_rate if source == "mic" else self._loopback_sample_rate
            handle.setframerate(sample_rate)
            self._wav_handles[source] = handle

    def _close_output_files(self) -> None:
        for handle in self._wav_handles.values():
            if handle is not None:
                handle.close()
        self._wav_handles = {"mic": None, "loopback": None}

    def _write_loop(self) -> None:
        while not self._stop_event.is_set() or not self._mic_queue.empty() or not self._loopback_queue.empty():
            wrote_any = False
            for source in ("mic", "loopback"):
                queue_obj = self._mic_queue if source == "mic" else self._loopback_queue
                try:
                    chunk = queue_obj.get(timeout=0.05)
                except queue.Empty:
                    continue
                handle = self._wav_handles.get(source)
                if handle is not None and chunk.audio:
                    handle.writeframes(chunk.audio)
                wrote_any = True
            if not wrote_any:
                time.sleep(0.001)

    def _enqueue_chunk(self, source: str, audio_bytes: bytes) -> None:
        queue_obj = self._mic_queue if source == "mic" else self._loopback_queue
        chunk = AudioChunk(
            source=source,
            timestamp=time.monotonic(),
            sample_rate=self._mic_sample_rate if source == "mic" else self._loopback_sample_rate,
            channels=1,
            audio=audio_bytes,
        )
        try:
            queue_obj.put_nowait(chunk)
        except queue.Full:
            try:
                queue_obj.get_nowait()
            except queue.Empty:
                pass
            queue_obj.put_nowait(chunk)

    def _mic_callback_impl(self, indata: np.ndarray, frames: int, time_info: dict[str, Any], status: Any) -> None:
        """Queue microphone PCM bytes without doing any extra processing."""
        if status:
            logger.warning("Microphone stream status: %s", status)
        if self._stop_event.is_set():
            return None

        audio = np.asarray(indata, dtype=np.int16)
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1).astype(np.int16)
        else:
            audio = audio.reshape(-1).astype(np.int16)

        self._enqueue_chunk("mic", audio.tobytes())
        return None

    def _loopback_callback_impl(self, in_data: bytes, frame_count: int, time_info: dict[str, Any], status: Any) -> tuple[None, int]:
        """Queue loopback PCM bytes without doing any extra processing."""
        if status:
            logger.warning("Loopback stream status: %s", status)
        if self._stop_event.is_set() or not in_data:
            return None, pyaudio.paContinue

        audio = np.frombuffer(in_data, dtype=np.int16)
        if audio.size == 0:
            return None, pyaudio.paContinue
        if self._loopback_channels > 1:
            audio = audio.reshape(-1, self._loopback_channels)
            audio = np.mean(audio, axis=1).astype(np.int16)
        else:
            audio = audio.reshape(-1).astype(np.int16)

        self._enqueue_chunk("loopback", audio.tobytes())
        return None, pyaudio.paContinue

    @staticmethod
    def _chunk_size_for_rate(sample_rate: int) -> int:
        return max(256, int(sample_rate * (AudioConfig.from_config().chunk_duration_ms / 1000.0)))


def main() -> None:
    """Record live microphone and loopback audio until interrupted."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    manager: DeviceManager | None = None
    recorder: AudioRecorder | None = None

    try:
        manager = DeviceManager()
        manager.print_diagnostics()
        recorder = AudioRecorder(device_manager=manager)

        project_root = Path(__file__).resolve().parent.parent
        output_paths = {
            "mic": project_root / "meeting_mic.wav",
            "loopback": project_root / "meeting_loopback.wav",
        }
        print("\nRecording live meeting audio. Press Ctrl+C to stop.", flush=True)
        recorder.start(output_paths)
        while recorder.is_running():
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping live capture.", flush=True)
    except Exception as exc:
        print(f"\nAudio capture failed: {exc}", flush=True)
    finally:
        if recorder is not None:
            try:
                recorder.stop()
            except Exception:  # pragma: no cover - best effort cleanup
                pass
        if manager is not None:
            try:
                manager.close()
            except Exception:  # pragma: no cover - best effort cleanup
                pass


if __name__ == "__main__":
    main()
