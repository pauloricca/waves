from __future__ import annotations

"""Audio interface helpers for managing input/output device aliases and routes."""

import threading
from collections import deque
import numpy as np
import sounddevice as sd

from config import (
    SAMPLE_RATE,
    BUFFER_SIZE,
    AUDIO_INPUT_DEVICES,
    AUDIO_OUTPUT_DEVICES,
    AUDIO_DEFAULT_INPUT_DEVICE_KEY,
    AUDIO_DEFAULT_OUTPUT_DEVICE_KEY,
)


def normalise_channel_mapping(
    channels: int | list[int] | tuple[int, ...] | None, default: tuple[int, ...]
) -> tuple[int, ...]:
    """Normalise channel definitions to a tuple of integers."""

    if channels is None:
        return default

    if isinstance(channels, int):
        return (int(channels),)

    if isinstance(channels, (list, tuple)):
        cleaned = tuple(int(ch) for ch in channels)
        if not cleaned:
            raise ValueError("Channel mapping cannot be empty")
        return cleaned

    raise ValueError(f"Unsupported channel definition: {channels!r}")


class AudioInterfaceManager:
    """Singleton managing discovery and alias resolution for audio devices."""

    _instance: AudioInterfaceManager | None = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialised = False
        return cls._instance

    def __init__(self):
        if self._initialised:
            return

        self._initialised = True
        self._devices = sd.query_devices()
        self._print_available_devices()
        self._print_configured_aliases()

    def _print_available_devices(self):
        if not self._devices:
            print("Warning: No audio devices detected")
            return

        print("Available audio devices:")
        for index, device in enumerate(self._devices):
            input_channels = device.get('max_input_channels', 0)
            output_channels = device.get('max_output_channels', 0)
            print(f"  [{index}] {device['name']} (inputs: {input_channels}, outputs: {output_channels})")

    def _print_configured_aliases(self):
        if AUDIO_INPUT_DEVICES:
            print("Configured audio input aliases:")
            for alias, device_name in AUDIO_INPUT_DEVICES.items():
                status = "✓" if self._find_device_by_name(device_name, 'input') is not None else "✗"
                print(f"  {status} {alias}: {device_name}")

        if AUDIO_OUTPUT_DEVICES:
            print("Configured audio output aliases:")
            for alias, device_name in AUDIO_OUTPUT_DEVICES.items():
                status = "✓" if self._find_device_by_name(device_name, 'output') is not None else "✗"
                print(f"  {status} {alias}: {device_name}")

    def _resolve_alias(self, alias: str | int | None, mapping: dict[str, str], default_key: str | None):
        if alias is None:
            alias = default_key

        if alias is None:
            return None

        if isinstance(alias, int):
            return alias

        if alias in mapping:
            return mapping[alias]

        # Fall back to provided string as device name
        return alias

    def _find_device_by_name(self, name: str, direction: str) -> int | None:
        for index, device in enumerate(self._devices):
            if device['name'] == name:
                if direction == 'input' and device.get('max_input_channels', 0) > 0:
                    return index
                if direction == 'output' and device.get('max_output_channels', 0) > 0:
                    return index
        return None

    def _find_device(self, identifier: str | int | None, direction: str) -> int | None:
        if identifier is None:
            return None

        if isinstance(identifier, int):
            if 0 <= identifier < len(self._devices):
                return identifier
            print(f"Warning: Audio {direction} device index {identifier} is out of range")
            return None

        device_index = self._find_device_by_name(identifier, direction)
        if device_index is None:
            print(f"Warning: Audio {direction} device '{identifier}' not found")
        return device_index

    def resolve_input_device(self, alias: str | int | None) -> int | None:
        identifier = self._resolve_alias(alias, AUDIO_INPUT_DEVICES, AUDIO_DEFAULT_INPUT_DEVICE_KEY)
        return self._find_device(identifier, 'input')

    def resolve_output_device(self, alias: str | int | None) -> int | None:
        identifier = self._resolve_alias(alias, AUDIO_OUTPUT_DEVICES, AUDIO_DEFAULT_OUTPUT_DEVICE_KEY)
        return self._find_device(identifier, 'output')


class _InputStreamEntry:
    def __init__(self, device_index: int | None, channel_mapping: tuple[int, ...]):
        self.device_index = device_index
        self.channel_mapping = channel_mapping
        self.channel_count = len(channel_mapping)
        self.buffer = deque(maxlen=SAMPLE_RATE * 10)
        self.lock = threading.Lock()

        mapping = [ch - 1 for ch in channel_mapping] if channel_mapping else None

        self.stream = sd.InputStream(
            device=self.device_index,
            channels=self.channel_count,
            samplerate=SAMPLE_RATE,
            blocksize=BUFFER_SIZE,
            dtype='float32',
            callback=self._callback,
            mapping=mapping,
        )
        self.stream.start()
        device_name = sd.query_devices(self.device_index)['name'] if self.device_index is not None else 'default'
        print(f"Audio input stream started for {device_name} channels {self.channel_mapping}")

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio input stream status: {status}")
        with self.lock:
            self.buffer.extend(indata.copy())

    def read(self, num_samples: int) -> np.ndarray:
        if self.channel_count == 1:
            audio_data = np.zeros(num_samples, dtype=np.float32)
        else:
            audio_data = np.zeros((num_samples, self.channel_count), dtype=np.float32)

        with self.lock:
            available = min(num_samples, len(self.buffer))
            if available > 0:
                frames = [self.buffer.popleft() for _ in range(available)]
                frames_array = np.vstack(frames).astype(np.float32, copy=False)
                if self.channel_count == 1:
                    audio_data[:available] = frames_array[:, 0]
                else:
                    audio_data[:available, :] = frames_array

        return audio_data


class AudioInputStreamRegistry:
    """Manage shared input streams per device/channel selection."""

    _instance: AudioInputStreamRegistry | None = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._streams = {}
                    cls._instance._streams_lock = threading.Lock()
        return cls._instance

    def get_stream(self, device_alias: str | int | None, channel_mapping: tuple[int, ...]) -> _InputStreamEntry:
        manager = AudioInterfaceManager()
        device_index = manager.resolve_input_device(device_alias)
        key = (device_index, channel_mapping)

        with self._streams_lock:
            if key not in self._streams:
                self._streams[key] = _InputStreamEntry(device_index, channel_mapping)
            return self._streams[key]


class _OutputRoute:
    def __init__(self, device_index: int | None, channel_mapping: tuple[int, ...]):
        self.device_index = device_index
        self.channel_mapping = channel_mapping
        self.channel_count = len(channel_mapping)
        self._chunks: deque[np.ndarray] = deque()
        self._chunk_lock = threading.Lock()
        self._current_chunk: np.ndarray | None = None
        self._current_pos = 0

        mapping = [ch - 1 for ch in channel_mapping] if channel_mapping else None

        self.stream = sd.OutputStream(
            device=self.device_index,
            channels=self.channel_count,
            samplerate=SAMPLE_RATE,
            blocksize=BUFFER_SIZE,
            dtype='float32',
            callback=self._callback,
            mapping=mapping,
        )
        self.stream.start()
        device_name = sd.query_devices(self.device_index)['name'] if self.device_index is not None else 'default'
        print(f"Audio output route started for {device_name} channels {self.channel_mapping}")

    def enqueue(self, audio_data: np.ndarray):
        if audio_data.ndim == 1 and self.channel_count == 1:
            data = audio_data[:, np.newaxis]
        elif audio_data.ndim == 1 and self.channel_count > 1:
            data = np.tile(audio_data[:, np.newaxis], (1, self.channel_count))
        elif audio_data.ndim == 2:
            if self.channel_count == 1:
                mono = np.mean(audio_data, axis=1, keepdims=True)
                data = mono
            elif audio_data.shape[1] == self.channel_count:
                data = audio_data
            elif audio_data.shape[1] == 1 and self.channel_count > 1:
                data = np.tile(audio_data, (1, self.channel_count))
            else:
                raise ValueError(
                    f"Cannot route audio with {audio_data.shape[1]} channels to mapping {self.channel_mapping}"
                )
        else:
            raise ValueError("Unsupported audio buffer shape for routing")

        with self._chunk_lock:
            self._chunks.append(np.array(data, dtype=np.float32, copy=True))

    def _callback(self, outdata, frames, time_info, status):
        if status:
            print(f"Audio output route status: {status}")

        outdata.fill(0)
        frames_filled = 0

        while frames_filled < frames:
            if self._current_chunk is None or self._current_pos >= len(self._current_chunk):
                with self._chunk_lock:
                    if self._chunks:
                        self._current_chunk = self._chunks.popleft()
                        self._current_pos = 0
                    else:
                        break

            if self._current_chunk is None:
                break

            available = len(self._current_chunk) - self._current_pos
            frames_to_copy = min(frames - frames_filled, available)

            chunk_slice = self._current_chunk[self._current_pos:self._current_pos + frames_to_copy]
            outdata[frames_filled:frames_filled + frames_to_copy, :] = chunk_slice
            self._current_pos += frames_to_copy
            frames_filled += frames_to_copy


class AudioOutputRouter:
    """Routes track audio to dedicated interface outputs."""

    _instance: AudioOutputRouter | None = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._routes = {}
                    cls._instance._routes_lock = threading.Lock()
        return cls._instance

    def _get_or_create_route(self, device_alias: str | int | None, channel_mapping: tuple[int, ...]) -> _OutputRoute:
        manager = AudioInterfaceManager()
        device_index = manager.resolve_output_device(device_alias)
        key = (device_index, channel_mapping)

        with self._routes_lock:
            if key not in self._routes:
                self._routes[key] = _OutputRoute(device_index, channel_mapping)
            return self._routes[key]

    def send(self, device_alias: str | int | None, channel_mapping: tuple[int, ...], audio_data: np.ndarray):
        route = self._get_or_create_route(device_alias, channel_mapping)
        route.enqueue(audio_data)


def get_audio_interface_manager() -> AudioInterfaceManager:
    return AudioInterfaceManager()


def get_audio_input_registry() -> AudioInputStreamRegistry:
    return AudioInputStreamRegistry()


def get_audio_output_router() -> AudioOutputRouter:
    return AudioOutputRouter()


__all__ = [
    "get_audio_interface_manager",
    "get_audio_input_registry",
    "get_audio_output_router",
    "normalise_channel_mapping",
]
