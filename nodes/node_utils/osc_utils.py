"""
Shared OSC utilities for OSC input handling across OSC nodes.
"""
from __future__ import annotations

import atexit
import socket
import struct
import threading
from typing import Any


OSC_DEBUG = False


def _pad4(length: int) -> int:
    return (length + 3) & ~0x03


def _read_osc_string(data: bytes, offset: int) -> tuple[str, int]:
    end = data.find(b"\x00", offset)
    if end == -1:
        raise ValueError("Invalid OSC packet: unterminated string")
    value = data[offset:end].decode("utf-8")
    next_offset = _pad4(end + 1)
    return value, next_offset


def _read_osc_arg(tag: str, data: bytes, offset: int) -> tuple[Any, int]:
    if tag == "i":
        return struct.unpack(">i", data[offset:offset + 4])[0], offset + 4
    if tag == "f":
        return struct.unpack(">f", data[offset:offset + 4])[0], offset + 4
    if tag == "d":
        return struct.unpack(">d", data[offset:offset + 8])[0], offset + 8
    if tag == "h":
        return struct.unpack(">q", data[offset:offset + 8])[0], offset + 8
    if tag == "s":
        return _read_osc_string(data, offset)
    if tag == "T":
        return True, offset
    if tag == "F":
        return False, offset
    if tag == "N":
        return None, offset
    raise ValueError(f"Unsupported OSC type tag: {tag}")


def parse_osc_packet(data: bytes) -> tuple[str, list[Any]]:
    address, offset = _read_osc_string(data, 0)
    if address == "#bundle":
        raise ValueError("OSC bundles are not supported")

    type_tags, offset = _read_osc_string(data, offset)
    if not type_tags.startswith(","):
        raise ValueError("Invalid OSC packet: missing type tag prefix")

    args: list[Any] = []
    for tag in type_tags[1:]:
        value, offset = _read_osc_arg(tag, data, offset)
        args.append(value)

    return address, args


class _OscListener:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.values_by_address: dict[str, list[Any]] = {}
        self.last_message: tuple[str, list[Any]] | None = None
        self.error: OSError | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._socket: socket.socket | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.host, self.port))
            sock.settimeout(0.2)
            self._socket = sock
        except OSError as exc:
            self.error = exc
            if OSC_DEBUG:
                print(f"OSC listener unavailable on {self.host}:{self.port}: {exc}")
            return

        if OSC_DEBUG:
            print(f"Listening for OSC on {self.host}:{self.port}")

        try:
            while not self._stop_event.is_set():
                try:
                    packet, _ = sock.recvfrom(65535)
                except socket.timeout:
                    continue
                except OSError:
                    break

                try:
                    address, args = parse_osc_packet(packet)
                except Exception as exc:
                    if OSC_DEBUG:
                        print(f"OSC parse error: {exc}")
                    continue

                with self._lock:
                    self.values_by_address[address] = args
                    self.last_message = (address, args)

                if OSC_DEBUG:
                    print(f"OSC {address}: {args}")
        finally:
            sock.close()

    def get_args(self, address: str) -> list[Any] | None:
        if self.error is not None:
            return None
        with self._lock:
            args = self.values_by_address.get(address)
            return list(args) if args is not None else None

    def shutdown(self):
        self._stop_event.set()
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


class OscInputManager:
    """Singleton manager for shared OSC listeners."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._listeners: dict[tuple[str, int], _OscListener] = {}
        self._listeners_lock = threading.Lock()
        atexit.register(self.shutdown)

    def ensure_listener(self, host: str, port: int):
        key = (host, port)
        with self._listeners_lock:
            if key not in self._listeners:
                self._listeners[key] = _OscListener(host, port)

    def get_value(self, address: str, host: str, port: int, arg_index: int = 0) -> float | None:
        self.ensure_listener(host, port)

        listener = self._listeners[(host, port)]
        args = listener.get_args(address)
        if not args or arg_index >= len(args):
            return None

        value = args[arg_index]
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def shutdown(self):
        with self._listeners_lock:
            listeners = list(self._listeners.values())
            self._listeners.clear()

        for listener in listeners:
            listener.shutdown()
