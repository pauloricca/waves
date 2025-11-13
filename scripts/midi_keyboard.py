#!/usr/bin/env python3
"""Simple computer keyboard to MIDI note bridge.

Run with root privileges so the script can capture keyboard events:

    sudo python3 scripts/midi_keyboard.py major
    sudo python3 scripts/midi_keyboard.py minor -1

Pass an optional octave shift after the scale name to transpose the layout.

Press Shift while holding notes to latch them. Press Shift again to release 
latched notes and resume normal playback.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import mido
import keyboard

MIDI_OUTPUT_DEVICE = None # "IAC Driver Bus 1" # Set to None to use first available output


SCALES: Dict[str, List[int]] = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
}

ROW_DEFINITIONS = [
    ("z", ["z", "x", "c", "v", "b", "n", "m"], -12),
    ("a", ["a", "s", "d", "f", "g", "h", "j", "k", "l"], 0),
    ("q", ["q", "w", "e", "r", "t", "y", "u", "i", "o", "p"], 12),
]

DEFAULT_ROOT_NOTE = 60  # Middle C (C4)
DEFAULT_VELOCITY = 100
DEFAULT_CHANNEL = 0

SHIFT_KEYS = {"shift", "shift left", "left shift", "shift right", "right shift"}


@dataclass
class MidiNoteMapping:
    note_number: int
    label: str


class MidiKeyboardController:
    """Map computer keyboard keys to MIDI notes and send note events."""

    def __init__(
        self,
        output_port: mido.ports.BaseOutput,
        key_map: Dict[str, MidiNoteMapping],
        velocity: int = DEFAULT_VELOCITY,
        channel: int = DEFAULT_CHANNEL,
    ) -> None:
        self.output_port = output_port
        self.key_map = key_map
        self.velocity = max(0, min(127, velocity))
        self.channel = max(0, min(15, channel))
        self._active_notes: Dict[str, int] = {}
        self._latched_keys: set[str] = set()
        self._latch_active = False

    def handle_event(self, event: keyboard.KeyboardEvent) -> None:
        key_name = event.name
        if key_name is None:
            return

        key = key_name.lower()
        if key in SHIFT_KEYS:
            if event.event_type == "down":
                if self._latch_active:
                    self._deactivate_latch()
                else:
                    self._activate_latch()
            return

        if key not in self.key_map:
            return

        if event.event_type == "down":
            self._handle_key_down(key)
        elif event.event_type == "up":
            self._handle_key_up(key)

    def _handle_key_down(self, key: str) -> None:
        if key in self._active_notes:
            return  # Ignore repeats while key is held

        note = self.key_map[key].note_number
        self.output_port.send(
            mido.Message(
                "note_on", channel=self.channel, note=note, velocity=self.velocity
            )
        )
        self._active_notes[key] = note

    def _handle_key_up(self, key: str) -> None:
        note = self._active_notes.pop(key, None)
        if note is None:
            return

        if self._latch_active:
            # Keep the note active until the latch is released.
            self._active_notes[key] = note
            self._latched_keys.add(key)
            return

        self.output_port.send(
            mido.Message("note_off", channel=self.channel, note=note, velocity=0)
        )

    def _activate_latch(self) -> None:
        if self._latch_active:
            return
        self._latch_active = True

    def _deactivate_latch(self) -> None:
        if not self._latch_active:
            return
        self._release_latched_notes()
        self._latch_active = False

    def _release_latched_notes(self) -> None:
        for key in list(self._latched_keys):
            note = self._active_notes.pop(key, None)
            if note is not None:
                self.output_port.send(
                    mido.Message(
                        "note_off", channel=self.channel, note=note, velocity=0
                    )
                )
        self._latched_keys.clear()

    def all_notes_off(self) -> None:
        self._release_latched_notes()
        self._latch_active = False
        for note in list(self._active_notes.values()):
            self.output_port.send(
                mido.Message("note_off", channel=self.channel, note=note, velocity=0)
            )
        self._active_notes.clear()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use the computer keyboard as a MIDI controller"
    )
    parser.add_argument(
        "scale",
        choices=sorted(SCALES.keys()),
        help="Scale to use for note layout",
    )
    parser.add_argument(
        "octave_shift",
        nargs="?",
        default=0,
        type=int,
        help="Number of octaves to shift the keyboard (positive or negative)",
    )
    parser.add_argument(
        "--velocity",
        type=int,
        default=DEFAULT_VELOCITY,
        help="MIDI note velocity (0-127)",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=DEFAULT_CHANNEL,
        help="MIDI channel to send notes on (0-15)",
    )
    return parser.parse_args(argv)


def build_key_map(scale_name: str, octave_shift: int) -> Dict[str, MidiNoteMapping]:
    scale = SCALES[scale_name]
    scale_length = len(scale)
    base_note = DEFAULT_ROOT_NOTE + (octave_shift * 12)

    key_map: Dict[str, MidiNoteMapping] = {}

    for _, keys, row_offset in ROW_DEFINITIONS:
        for index, key in enumerate(keys):
            scale_degree = scale[index % scale_length]
            octave = (index // scale_length) * 12
            midi_note = base_note + row_offset + scale_degree + octave
            midi_note = max(0, min(127, midi_note))
            label = f"{key.upper()} ({scale_name} {midi_note})"
            key_map[key] = MidiNoteMapping(note_number=midi_note, label=label)

    return key_map


def select_output_port() -> Optional[mido.ports.BaseOutput]:
    output_names = mido.get_output_names()
    if not output_names:
        print("No MIDI output ports available.")
        return None

    desired_port = MIDI_OUTPUT_DEVICE if MIDI_OUTPUT_DEVICE else None
    port_name = None

    if desired_port and desired_port in output_names:
        port_name = desired_port
    elif desired_port:
        print(
            f"Configured MIDI output '{desired_port}' not found. Using first available output."
        )

    if port_name is None:
        port_name = output_names[0]

    try:
        port = mido.open_output(port_name)
        print(f"Connected to MIDI output: {port_name}")
        return port
    except Exception as exc:  # pragma: no cover - hardware dependent
        print(f"Failed to open MIDI output '{port_name}': {exc}")
        return None


def warn_if_not_root() -> None:
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        print(
            "Warning: root privileges recommended. Run with 'sudo' if key events are not detected."
        )


def print_mapping_info(key_map: Dict[str, MidiNoteMapping]) -> None:
    print("\nKeyboard mapping (key -> MIDI note):")
    for row_label, keys, _ in ROW_DEFINITIONS:
        row_display = []
        for key in keys:
            mapping = key_map[key]
            row_display.append(f"{key.upper()}:{mapping.note_number:>3d}")
        print(f"  Row {row_label.upper()}: {'  '.join(row_display)}")
    print(
        "\nPress keys to send MIDI notes. Press Shift while holding notes to latch them."
        "\nPress Shift again to unlatch. Press Ctrl+C to exit.\n"
    )


def main(argv: Optional[List[str]] = None) -> int:
    warn_if_not_root()
    args = parse_args(argv)

    key_map = build_key_map(args.scale, args.octave_shift)
    output_port = select_output_port()
    if output_port is None:
        return 1

    controller = MidiKeyboardController(
        output_port=output_port,
        key_map=key_map,
        velocity=args.velocity,
        channel=args.channel,
    )

    print_mapping_info(key_map)

    keyboard.hook(controller.handle_event)

    try:
        while True:
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        controller.all_notes_off()
        output_port.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
