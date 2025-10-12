#!/usr/bin/env python3
import sounddevice as sd
import numpy as np
import os
import sys
import time
import traceback
from collections import deque
import threading
import gc


from config import *
from sound_library import get_sound_model, load_sound_library
from nodes.node_utils.base_node import BaseNode
from nodes.node_utils.instantiate_node import instantiate_node
from nodes.node_utils.render_context import RenderContext
from utils import look_for_duration, play, save, visualise_wave

rendered_sounds: dict[np.ndarray] = {}

def play_in_real_time(sound_node: BaseNode, duration_in_seconds: float):
    visualised_wave_buffer = deque(maxlen=10_000)
    should_stop = False
    start_time = time.time()
    last_render_time = 0
    
    # Create render context that persists across chunks
    render_context = RenderContext()
    render_context.is_realtime = True

    def audio_callback(outdata, frames, sdtime, status):
        nonlocal should_stop, last_render_time
        rendering_start_time = time.time()

        audio_data = sound_node.render(frames, context=render_context) 

        if len(audio_data) == 0:
            should_stop = True

        if should_stop:
            raise sd.CallbackStop()

        # Apply master gain
        audio_data *= RENDERED_MASTER_GAIN

        visualised_wave_buffer.extend(audio_data)

        clipped_audio_data = np.clip(audio_data, -1.0, 1.0)

        # If we got fewer samples than requested, pad with zeros or stop
        if len(clipped_audio_data) < frames:
            # This is the last chunk - pad with zeros and then stop
            padding = np.zeros(frames - len(clipped_audio_data))
            clipped_audio_data = np.concatenate([clipped_audio_data, padding])
            should_stop = True

        outdata[:] = clipped_audio_data.reshape(-1, 1)

        rendering_end_time = time.time()
        last_render_time = rendering_end_time - rendering_start_time
        
        # Clear chunk data for next render (important for realtime mode)
        render_context.clear_chunk()

    def run_visualizer_and_stats():
        # Lower priority for visualization thread to avoid interfering with audio
        try:
            os.nice(10)  # Increase niceness (lower priority) on Unix systems
        except (AttributeError, OSError):
            pass  # Windows or permission issues
        
        while not should_stop:
            if len(visualised_wave_buffer) > 0:
                try:
                    render_time_text = f"Render time: {100 * last_render_time / (BUFFER_SIZE / SAMPLE_RATE):.2f}%"
                    
                    if DO_VISUALISE_OUTPUT:
                        visualise_wave(np.array(visualised_wave_buffer), do_normalise=False, replace_previous=True, extra_lines=1)
                        if DISPLAY_RENDER_TIME_PERCENTAGE:
                            print(render_time_text, flush=True)
                    elif DISPLAY_RENDER_TIME_PERCENTAGE:
                        # Clear line and print stats only (no visualization)
                        print(f"\r{render_time_text}", end='', flush=True)
                except Exception:
                    # Silently ignore visualization errors to avoid breaking audio
                    pass
            time.sleep(1 / VISUALISATION_FPS)  # Configurable frame rate

    if (DISPLAY_RENDER_TIME_PERCENTAGE or DO_VISUALISE_OUTPUT):
        vis_thread = threading.Thread(target=run_visualizer_and_stats, daemon=True)
        vis_thread.start()

    with sd.OutputStream(callback=audio_callback, blocksize=BUFFER_SIZE, samplerate=SAMPLE_RATE, channels=1): #, latency='low'
        while not should_stop:
            # Only stop based on duration if one is specified
            if duration_in_seconds and time.time() - start_time > duration_in_seconds:
                should_stop = True
            time.sleep(0.1)

def main():
    global rendered_sounds

    if not load_sound_library(YAML_FILE):
        return

    if len(sys.argv) < 2:
        print("Usage: python waves.py <sound-name>")
        sys.exit(1)

    sound_name_to_play = sys.argv[1]    

    sound_model_to_play = get_sound_model(sound_name_to_play)
    sound_node_to_play = instantiate_node(sound_model_to_play)

    sound_duration = look_for_duration(sound_model_to_play)

    if DO_PLAY_IN_REAL_TIME:
        play_in_real_time(sound_node_to_play, sound_duration)
    else:
        # Non-realtime mode requires a duration
        if not sound_duration:
            sound_duration = DEFAULT_PLAYBACK_TIME  # Fallback for nodes without explicit duration
        
        # Create render context for non-realtime mode
        render_context = RenderContext()
        render_context.is_realtime = False
            
        rendering_start_time = time.time()
        
        if DO_PRE_RENDER_WHOLE_SOUND:
            rendered_sound = sound_node_to_play.render(int(SAMPLE_RATE * sound_duration), context=render_context)
        else:
            # Render in chunks
            rendered_sound: np.ndarray = []
            rendered_buffer: np.ndarray = None
            while (rendered_buffer is None or len(rendered_buffer) != 0) and len(rendered_sound) < sound_duration * SAMPLE_RATE:
                rendered_buffer = sound_node_to_play.render(BUFFER_SIZE, context=render_context)
                rendered_sound = np.concatenate((rendered_sound, rendered_buffer))
                # Clear chunk for next iteration
                render_context.clear_chunk()

        rendering_end_time = time.time()

        # Normalize the combined wave
        peak = np.max(np.abs(rendered_sound))
        rendered_sound *= RENDERED_MASTER_GAIN

        save(rendered_sound, f"{sound_name_to_play}.wav")

        if DO_VISUALISE_OUTPUT:
            visualise_wave(rendered_sound)

        print(f"Sound length: {len(rendered_sound) / SAMPLE_RATE:.2f} seconds")
        print(f"Render time: {rendering_end_time - rendering_start_time:.2f} seconds")
        print ("Peak:", peak)

        # rendered_sound = normalise_wave(rendered_sound)
        rendered_sound = np.clip(rendered_sound, -1, 1)
        play(rendered_sound)

if __name__ == "__main__":
    if DISABLE_GARBAGE_COLLECTION:
        gc.disable()

    if WAIT_FOR_CHANGES_IN_WAVES_YAML:
        last_modified_time = ""
        try:
            while True:
                current_modified_time = os.path.getmtime(YAML_FILE)
                if current_modified_time != last_modified_time:
                    last_modified_time = current_modified_time
                    try:
                        main()
                    except Exception as e:
                        print(f"Error: {e}")
                        traceback.print_exc()
                time.sleep(0.2)
        except KeyboardInterrupt:
            sys.exit(0)
    else:
        main()
