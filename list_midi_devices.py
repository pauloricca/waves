#!/usr/bin/env python3
"""
List all available MIDI input devices.
Use this to find the exact name of your MIDI controller to configure in config.py
"""

import mido

def list_midi_devices():
    """List all available MIDI input ports"""
    print("=" * 60)
    print("AVAILABLE MIDI INPUT DEVICES")
    print("=" * 60)
    
    try:
        input_ports = mido.get_input_names()
        
        if not input_ports:
            print("\nNo MIDI input devices found.")
            print("\nMake sure your MIDI controller is:")
            print("  1. Connected to your computer")
            print("  2. Powered on")
            print("  3. Recognized by your operating system")
            return
        
        print(f"\nFound {len(input_ports)} MIDI input device(s):\n")
        
        for i, port in enumerate(input_ports, 1):
            # Highlight external controllers vs IAC Driver
            device_type = ""
            if 'IAC' in port:
                device_type = " (Virtual/IAC Driver)"
            else:
                device_type = " (External Controller)"
            
            print(f"{i}. {port}{device_type}")
        
        print("\n" + "=" * 60)
        print("CONFIGURATION INSTRUCTIONS")
        print("=" * 60)
        print("\nTo use one of these devices in your waves app:")
        print("\n1. Copy the exact device name from above")
        print("2. Open config.py")
        print("3. Set MIDI_INPUT_DEVICE_NAME to the device name")
        print("\nExample:")
        print('   MIDI_INPUT_DEVICE_NAME = "Your MIDI Controller Name"')
        print("\nOr leave it as None to auto-detect external controllers:")
        print('   MIDI_INPUT_DEVICE_NAME = None')
        print()
        
    except Exception as e:
        print(f"\nError accessing MIDI devices: {e}")
        print("\nMake sure 'mido' and 'python-rtmidi' are installed:")
        print("  pip install mido python-rtmidi")


if __name__ == "__main__":
    list_midi_devices()
