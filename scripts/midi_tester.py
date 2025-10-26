#!/usr/bin/env python3
"""
List all available MIDI input devices and monitor incoming MIDI messages.
Use this to find the exact name of your MIDI controller and test CC numbers.
"""

import mido
import sys


def format_message(port_name, message):
    """Format a MIDI message for display"""
    if message.type == 'note_on':
        if message.velocity > 0:
            return f"[{port_name}] NOTE ON  - Channel: {message.channel:2d} | Note: {message.note:3d} | Velocity: {message.velocity:3d}"
        else:
            # Note on with velocity 0 is treated as note off
            return f"[{port_name}] NOTE OFF - Channel: {message.channel:2d} | Note: {message.note:3d} | Velocity: {message.velocity:3d}"
    elif message.type == 'note_off':
        return f"[{port_name}] NOTE OFF - Channel: {message.channel:2d} | Note: {message.note:3d} | Velocity: {message.velocity:3d}"
    elif message.type == 'control_change':
        return f"[{port_name}] CC       - Channel: {message.channel:2d} | CC#: {message.control:3d} | Value: {message.value:3d}"
    elif message.type == 'pitchwheel':
        return f"[{port_name}] PITCH    - Channel: {message.channel:2d} | Value: {message.pitch:6d}"
    elif message.type == 'aftertouch':
        return f"[{port_name}] AFTRTOUCH- Channel: {message.channel:2d} | Value: {message.value:3d}"
    elif message.type == 'polytouch':
        return f"[{port_name}] POLY AT  - Channel: {message.channel:2d} | Note: {message.note:3d} | Value: {message.value:3d}"
    elif message.type == 'program_change':
        return f"[{port_name}] PROGRAM  - Channel: {message.channel:2d} | Program: {message.program:3d}"
    else:
        return f"[{port_name}] {message.type.upper()} - {message}"


def monitor_midi_messages():
    """Monitor and display all MIDI messages from all input devices"""
    try:
        input_ports = mido.get_input_names()
        
        if not input_ports:
            print("\nNo MIDI input devices found.")
            print("\nMake sure your MIDI controller is:")
            print("  1. Connected to your computer")
            print("  2. Powered on")
            print("  3. Recognized by your operating system")
            return
        
        print("\n=== Available MIDI Input Devices ===")
        for i, port in enumerate(input_ports, 1):
            print(f"{i}. {port}")
        

        # Open all ports with callbacks
        open_ports = []
        for port_name in input_ports:
            try:
                # Create a callback for this specific port
                def make_callback(name):
                    def callback(message):
                        print(format_message(name, message))
                    return callback
                
                port = mido.open_input(port_name, callback=make_callback(port_name))
                open_ports.append(port)
            except Exception as e:
                print(f"✗ Could not open {port_name}: {e}")
        
        if not open_ports:
            print("\nNo MIDI ports could be opened.")
            return
        
        print("\nPlay notes, move knobs, or press keys on your MIDI controller...\n")
        
        # Keep the script running
        try:
            while True:
                pass
        except KeyboardInterrupt:
            print("\n\nStopping MIDI monitoring...")
        finally:
            # Close all ports
            for port in open_ports:
                port.close()
            print("All MIDI ports closed.")
        
    except Exception as e:
        print(f"\nError accessing MIDI devices: {e}")
        print("\nMake sure 'mido' and 'python-rtmidi' are installed:")
        print("  pip install mido python-rtmidi")


if __name__ == "__main__":
    monitor_midi_messages()
