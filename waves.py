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
import signal
import atexit


from config import *
from sound_library import get_sound_model, load_all_sound_libraries, reload_sound_library, get_sound_filename
from nodes.node_utils.base_node import BaseNode
from nodes.node_utils.instantiate_node import instantiate_node
from nodes.node_utils.render_context import RenderContext
from utils import look_for_duration, play, save, visualise_wave
from display_stats import run_visualizer_and_stats

rendered_sounds: dict[np.ndarray] = {}

# Hot reload state management
hot_reload_lock = threading.Lock()  # Protects access to current_sound_node and model during reload
current_sound_node = None  # Currently playing node
current_sound_model = None  # Currently loaded model
yaml_changed = False  # Flag to signal that YAML has changed
changed_yaml_file = None  # Which YAML file changed (for selective reload)
hot_reload_pending_node = None  # Tuple of (new_node, old_node) waiting to be swapped in
hot_reload_in_progress = False  # True while hot reload thread is running

# Global recording buffer (thread-safe deque)
recording_buffer = None
recording_active = False
current_sound_name = None  # Track current sound name for recording


def get_innermost_node(node: BaseNode) -> BaseNode:
    """
    Unwrap pass-through nodes to find the innermost actual signal node.
    
    Pass-through nodes like ContextNode and TempoNode just add params but pass
    through their child signal unchanged. This function recursively unwraps them
    to find the actual signal-generating node (e.g., TracksNode).
    
    Args:
        node: A BaseNode instance to unwrap
        
    Returns:
        The innermost non-pass-through node
    """
    # Check if this node has a signal_node attribute (pass-through pattern)
    if hasattr(node, 'model') and hasattr(node.model, 'is_pass_through') and node.model.is_pass_through:
        if hasattr(node, 'signal_node'):
            return get_innermost_node(node.signal_node)
    
    return node


def has_tracks_node_inside(model) -> bool:
    """
    Check if a model is or contains a TracksNode, unwrapping pass-through nodes.
    
    Pass-through nodes like 'context' and 'tempo' just add render params but don't 
    change the audio. We recursively unwrap them to see if there's a TracksNode inside.
    
    Args:
        model: A BaseNodeModel to check
        
    Returns:
        True if the model is a TracksNodeModel or contains one inside pass-through nodes
    """
    from nodes.tracks import TracksNodeModel
    
    # Direct check
    if isinstance(model, TracksNodeModel):
        return True
    
    # Check if this is a pass-through node with a signal child
    if hasattr(model, 'is_pass_through') and model.is_pass_through:
        if hasattr(model, 'signal'):
            return has_tracks_node_inside(model.signal)
    
    return False


def perform_hot_reload_background(sound_name_to_play: str, changed_filename: str = None, params: dict = None):
    """
    Perform a hot reload of the sound model on a background thread.
    
    This function:
    1. Reloads only the changed YAML file
    2. Gets the new model
    3. Applies any parameters
    4. Instantiates the new tree with hot_reload=True
       (States are automatically preserved via global state registry)
    5. Stores the new tree in hot_reload_pending_node for atomic swap
    
    This is designed to run on a separate thread to avoid blocking audio rendering.
    
    Args:
        sound_name_to_play: The sound identifier
        changed_filename: The YAML filename that changed (if known)
        params: Optional parameter overrides to apply to the model
    """
    global hot_reload_pending_node, hot_reload_in_progress
    
    try:
        # Keep reference to old node for cleanup
        old_node = None
        with hot_reload_lock:
            old_node = current_sound_node
        
        # Reload only the changed file, or the file containing the sound if unknown
        if changed_filename:
            reload_sound_library(changed_filename)
        else:
            # Find which file contains this sound and reload it
            filename = get_sound_filename(sound_name_to_play)
            reload_sound_library(filename)
        
        new_model = get_sound_model(sound_name_to_play)
        
        # Apply parameters if provided
        if params:
            from nodes.node_utils.node_string_parser import apply_params_to_model
            new_model = apply_params_to_model(new_model, params)
        
        # Auto-wrap in tracks node if not already a tracks node (or contains one inside pass-through nodes)
        if not has_tracks_node_inside(new_model):
            from nodes.tracks import TracksNodeModel
            # Wrap in a tracks node with the sound name as the track name
            wrapped_model = TracksNodeModel()
            wrapped_model.__pydantic_extra__ = {sound_name_to_play: new_model}
            new_model = wrapped_model
        
        new_node = instantiate_node(new_model, sound_name_to_play, "root")
        
        # Store the new node for atomic swap on next audio chunk
        with hot_reload_lock:
            hot_reload_pending_node = (new_node, old_node)  # Store both new and old for cleanup
    
    except Exception as e:
        print(f"Hot reload error: {e}")
        traceback.print_exc()
    
    finally:
        hot_reload_in_progress = False

# Global recording buffer (thread-safe deque)
recording_buffer = None
recording_active = False
current_sound_name = None  # Track current sound name for recording
recording_track_buffers = None  # Dict of track_name -> deque for multi-track recording
recording_sound_node = None  # Reference to sound node for track export
recording_is_explicit_tracks = False  # Whether user explicitly used tracks: node

def get_unique_filename(base_filename):
    """
    Generate a unique filename by appending _2, _3, etc. if the file already exists.
    
    Args:
        base_filename: The base filename (e.g., "recording.wav")
    
    Returns:
        A unique filename that doesn't exist in OUTPUT_DIR
    """
    # Use current working directory if __file__ is not available
    try:
        base_dir = os.path.dirname(__file__)
    except NameError:
        base_dir = os.getcwd()
    
    output_dir = os.path.join(base_dir, OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    
    # Split filename into name and extension
    name, ext = os.path.splitext(base_filename)
    
    # Check if base filename exists
    filepath = os.path.join(output_dir, base_filename)
    if not os.path.exists(filepath):
        return base_filename
    
    # Try _2, _3, etc. until we find a unique name
    counter = 2
    while True:
        new_filename = f"{name}_{counter}{ext}"
        filepath = os.path.join(output_dir, new_filename)
        if not os.path.exists(filepath):
            return new_filename
        counter += 1

def save_recording(sound_name=None):
    """Save the recorded audio buffer to file(s)."""
    global recording_buffer, recording_active, recording_track_buffers, recording_sound_node, recording_is_explicit_tracks
    
    if not recording_active or recording_buffer is None or len(recording_buffer) == 0:
        return
    
    try:
        # Convert deque to numpy array
        recorded_audio = np.array(recording_buffer, dtype=np.float32)
        
        # Use sound name if provided, otherwise fall back to default
        base_filename = f"{sound_name}.wav" if sound_name else "realtime_recording.wav"
        
        # Save mixdown
        unique_filename = get_unique_filename(base_filename)
        save(recorded_audio, unique_filename)
        print(f"Recording saved: {unique_filename} ({len(recorded_audio) / SAMPLE_RATE:.2f} seconds)")
        
        # Save individual track stems if this is an explicit tracks node
        from nodes.tracks import TracksNode
        # Unwrap pass-through nodes to find the actual TracksNode
        innermost_recording_node = get_innermost_node(recording_sound_node) if recording_sound_node else None
        if DO_SAVE_MULTITRACK and recording_is_explicit_tracks and isinstance(innermost_recording_node, TracksNode) and recording_track_buffers:
            for track_name, track_buffer in recording_track_buffers.items():
                if len(track_buffer) > 0:
                    track_audio = np.array(track_buffer, dtype=np.float32)
                    track_filename = f"{sound_name}__{track_name}.wav"
                    unique_track_filename = get_unique_filename(track_filename)
                    save(track_audio, unique_track_filename)
                    print(f"Track saved: {unique_track_filename}")
    except Exception as e:
        print(f"Error saving recording: {e}")
        import traceback
        traceback.print_exc()
    finally:
        recording_active = False

def signal_handler(sig, frame):
    """Handle Ctrl-C gracefully and save recording."""
    print("\nInterrupted by user, saving recording...")
    save_recording(current_sound_name)
    sys.exit(0)

def play_in_real_time(sound_node: BaseNode, duration_in_seconds: float, sound_name: str = None, is_explicit_tracks: bool = False):
    global recording_buffer, recording_active, current_sound_name, current_sound_node, yaml_changed
    global recording_track_buffers, recording_sound_node, recording_is_explicit_tracks
    
    # Initialize recording if enabled
    if DO_RECORD_REAL_TIME:
        recording_buffer = deque()
        recording_active = True
        current_sound_name = sound_name
        recording_sound_node = sound_node
        recording_is_explicit_tracks = is_explicit_tracks
        
        # Initialize track buffers if this is an explicit tracks node
        # Unwrap pass-through nodes to find the actual TracksNode
        from nodes.tracks import TracksNode
        innermost_node = get_innermost_node(sound_node)
        if is_explicit_tracks and isinstance(innermost_node, TracksNode):
            recording_track_buffers = {name: deque() for name in innermost_node.get_track_names()}
        else:
            recording_track_buffers = None
        
        # Register cleanup function to save on normal exit
        atexit.register(lambda: save_recording(sound_name))
    
    # Set the global current node for hot reload manager
    with hot_reload_lock:
        current_sound_node = sound_node
    
    visualised_wave_buffer = deque(maxlen=10_000)
    should_stop = False
    start_time = time.time()
    last_render_time = 0
    active_sound_node = sound_node  # Local reference to current node
    stored_sound_name = sound_name  # Store the sound name for hot reload
    
    # References for display thread (using lists so they can be modified in nested scope)
    should_stop_ref = [False]
    last_render_time_ref = [0]
    recording_active_ref = [recording_active]
    
    # Create render context that persists across chunks
    render_context = RenderContext()
    render_context.is_realtime = True

    def audio_callback(outdata, frames, sdtime, status):
        global yaml_changed, hot_reload_pending_node, hot_reload_in_progress, current_sound_node, current_sound_model
        nonlocal should_stop, last_render_time, active_sound_node, render_context
        rendering_start_time = time.time()
        
        # Check if a hot reload has finished and swap in the new node
        if hot_reload_pending_node is not None:
            with hot_reload_lock:
                if hot_reload_pending_node is not None:
                    # Atomically swap in the new node
                    new_node, _old_node = hot_reload_pending_node
                    active_sound_node = new_node
                    current_sound_node = new_node
                    hot_reload_pending_node = None
                    
                    # Clear stale node instance references in the render context
                    # This prevents segfaults from keeping pointers to deleted nodes
                    render_context.clear_node_instances()
        
        # Check if YAML has changed and start background hot reload if not already in progress
        if yaml_changed and not hot_reload_in_progress:
            yaml_changed = False
            hot_reload_in_progress = True
            
            # Capture the changed file
            global changed_yaml_file
            captured_changed_file = changed_yaml_file
            changed_yaml_file = None
            
            # Add a small delay before reloading to give audio callback time to stabilize
            def delayed_reload():
                time.sleep(HOT_RELOAD_DELAY)
                perform_hot_reload_background(stored_sound_name, captured_changed_file)
            
            # Start hot reload on a separate thread
            reload_thread = threading.Thread(
                target=delayed_reload,
                daemon=True
            )
            reload_thread.start()

        audio_data = active_sound_node.render(frames, context=render_context) 

        if len(audio_data) == 0:
            should_stop = True

        if should_stop:
            raise sd.CallbackStop()

        # Apply master gain
        audio_data *= RENDERED_MASTER_GAIN

        # Detect if stereo (2D array) or mono (1D array)
        is_stereo = audio_data.ndim == 2
        
        # For visualization and recording, use mono (left channel if stereo)
        if is_stereo:
            mono_for_vis = audio_data[:, 0]  # Left channel
        else:
            mono_for_vis = audio_data

        # Add to recording buffer if recording is enabled (before clipping for visualization)
        if recording_active and recording_buffer is not None:
            recording_buffer.extend(audio_data)  # Store full stereo for recording
            
            # Also record individual tracks if this is a multi-track recording
            # Need to unwrap pass-through nodes to get to the TracksNode
            innermost_active_node = get_innermost_node(active_sound_node)
            if recording_track_buffers and hasattr(innermost_active_node, 'last_track_outputs') and innermost_active_node.last_track_outputs:
                for track_name, track_data in innermost_active_node.last_track_outputs.items():
                    if track_name in recording_track_buffers and len(track_data) > 0:
                        recording_track_buffers[track_name].extend(track_data)

        visualised_wave_buffer.extend(mono_for_vis)  # Visualize left channel only

        clipped_audio_data = np.clip(audio_data, -1.0, 1.0)

        # If we got fewer samples than requested, pad with zeros or stop
        if len(clipped_audio_data) < frames:
            # This is the last chunk - pad with zeros and then stop
            if is_stereo:
                padding = np.zeros((frames - len(clipped_audio_data), 2))
            else:
                padding = np.zeros(frames - len(clipped_audio_data))
            clipped_audio_data = np.concatenate([clipped_audio_data, padding])
            should_stop = True

        # Reshape for output: stereo stays (n, 2), mono becomes (n, 1)
        if is_stereo:
            outdata[:] = clipped_audio_data
        else:
            outdata[:] = clipped_audio_data.reshape(-1, 1)


        rendering_end_time = time.time()
        last_render_time = rendering_end_time - rendering_start_time
        last_render_time_ref[0] = last_render_time  # Update reference for display thread
        
        # Clear chunk data for next render (important for realtime mode)
        render_context.clear_chunk()

    if (DISPLAY_RENDER_STATS or DO_VISUALISE_OUTPUT):
        vis_thread = threading.Thread(
            target=run_visualizer_and_stats,
            args=(visualised_wave_buffer, should_stop_ref, start_time, last_render_time_ref, recording_active_ref),
            daemon=True
        )
        vis_thread.start()

    # Detect if the sound node outputs stereo
    # We need to check this before starting the stream
    # Unwrap pass-through nodes (like context, tempo) to find the actual signal node
    from nodes.tracks import TracksNode
    innermost_node = get_innermost_node(sound_node)
    # TracksNode always outputs stereo, other stereo-capable nodes might too
    num_channels = 2 if isinstance(innermost_node, TracksNode) else 1

    with sd.OutputStream(callback=audio_callback, blocksize=BUFFER_SIZE, samplerate=SAMPLE_RATE, channels=num_channels): #, latency='low'
        while not should_stop:
            # Update references for display thread
            should_stop_ref[0] = should_stop
            recording_active_ref[0] = recording_active
            
            # Only stop based on duration if one is specified
            if duration_in_seconds and time.time() - start_time > duration_in_seconds:
                should_stop = True
            time.sleep(0.1)

def main():
    global rendered_sounds, current_sound_node, current_sound_model

    if not load_all_sound_libraries(SOUNDS_DIR):
        return
    
    # Initialize MIDI system early to avoid hiccups during playback
    # This ensures MIDI devices are detected and opened before audio starts
    from nodes.node_utils.midi_utils import MidiInputManager
    MidiInputManager()

    if len(sys.argv) < 2:
        print("Usage: python waves.py <sound-name> [param1VALUE] [param2VALUE] ...")
        print("Examples:")
        print("  python waves.py kick")
        print("  python waves.py kick f440 a0.5")
        print("  python waves.py my_sound t2 f880")
        sys.exit(1)

    # Join all arguments to handle node string with parameters
    sound_string = ' '.join(sys.argv[1:])
    
    # Parse the node string to get name and parameters
    from nodes.node_utils.node_string_parser import parse_node_string, apply_params_to_model
    sound_name_to_play, params = parse_node_string(sound_string)

    sound_model_to_play = get_sound_model(sound_name_to_play)
    
    # Apply parameters to the model if any were provided
    if params:
        sound_model_to_play = apply_params_to_model(sound_model_to_play, params)
    
    # Auto-wrap in tracks node if not already a tracks node (or contains one inside pass-through nodes)
    is_explicit_tracks = has_tracks_node_inside(sound_model_to_play)
    
    if not is_explicit_tracks:
        from nodes.tracks import TracksNodeModel
        # Wrap in a tracks node with the sound name as the track name
        wrapped_model = TracksNodeModel()
        # Store the original sound as a track using __pydantic_extra__
        wrapped_model.__pydantic_extra__ = {sound_name_to_play: sound_model_to_play}
        sound_model_to_play = wrapped_model
    
    # Initialize the node (not as hot reload on first load)
    sound_node_to_play = instantiate_node(sound_model_to_play, sound_name_to_play, "root")
    
    # Store globals for hot reload
    with hot_reload_lock:
        current_sound_node = sound_node_to_play
        current_sound_model = sound_model_to_play

    sound_duration = look_for_duration(sound_model_to_play)

    if DO_PLAY_IN_REAL_TIME:
        play_in_real_time(sound_node_to_play, sound_duration, sound_name_to_play, is_explicit_tracks)
    else:
        # Non-realtime mode requires a duration
        if not sound_duration:
            sound_duration = DEFAULT_PLAYBACK_TIME  # Fallback for nodes without explicit duration
        
        # Create render context for non-realtime mode
        render_context = RenderContext()
        render_context.is_realtime = False
            
        rendering_start_time = time.time()
        
        # Check if this is a tracks node for multi-file export
        # Export individual tracks if:
        # 1. User explicitly used 'tracks:' node AND has at least one track
        # 2. User explicitly used 'tracks:' with multiple tracks
        # Unwrap pass-through nodes to find the actual TracksNode
        from nodes.tracks import TracksNode
        innermost_node = get_innermost_node(sound_node_to_play)
        is_tracks = isinstance(innermost_node, TracksNode)
        should_export_tracks = is_explicit_tracks and is_tracks and len(innermost_node.get_track_names()) >= 1
        
        # Render mixdown (always needed)
        if DO_PRE_RENDER_WHOLE_SOUND:
            num_samples = int(SAMPLE_RATE * sound_duration)
            rendered_sound = sound_node_to_play.render(num_samples, context=render_context)
            # Track outputs are cached in the innermost node after render
            track_outputs = innermost_node.last_track_outputs if should_export_tracks else None
        else:
            # Render mixdown in chunks and accumulate track outputs
            rendered_sound = []
            track_buffers = {name: [] for name in innermost_node.get_track_names()} if should_export_tracks else None
            
            while True:
                rendered_buffer = sound_node_to_play.render(BUFFER_SIZE, context=render_context)
                if len(rendered_buffer) == 0:
                    break
                rendered_sound.append(rendered_buffer)
                
                # Collect track outputs from this chunk
                if should_export_tracks and innermost_node.last_track_outputs:
                    for track_name, track_data in innermost_node.last_track_outputs.items():
                        if len(track_data) > 0:
                            track_buffers[track_name].append(track_data)
                
                render_context.clear_chunk()
                if sum(len(c) for c in rendered_sound) >= sound_duration * SAMPLE_RATE:
                    break
            
            rendered_sound = np.concatenate(rendered_sound) if rendered_sound else np.array([]).reshape(0, 2)
            
            # Concatenate track chunks
            if should_export_tracks:
                track_outputs = {
                    name: np.concatenate(chunks) if chunks else np.array([]).reshape(0, 2)
                    for name, chunks in track_buffers.items()
                }
            else:
                track_outputs = None

        rendering_end_time = time.time()

        # Apply master gain
        rendered_sound *= RENDERED_MASTER_GAIN
        
        # Save files
        if DO_SAVE_MULTITRACK and should_export_tracks and track_outputs:
            # Save individual track stems
            for track_name, track_data in track_outputs.items():
                track_data *= RENDERED_MASTER_GAIN
                track_filename = f"{sound_name_to_play}__{track_name}.wav"
                save(track_data, track_filename)
            
            # Save mixdown
            save(rendered_sound, f"{sound_name_to_play}.wav")
        else:
            # Single file export
            save(rendered_sound, f"{sound_name_to_play}.wav")

        if DO_VISUALISE_OUTPUT:
            # Visualize mixdown (left channel if stereo)
            if rendered_sound.ndim == 2:
                visualise_wave(rendered_sound[:, 0])
            else:
                visualise_wave(rendered_sound)

        # Calculate peak from mixdown
        peak = np.max(np.abs(rendered_sound))
        print(f"Sound length: {len(rendered_sound) / SAMPLE_RATE:.2f} seconds")
        print(f"Render time: {rendering_end_time - rendering_start_time:.2f} seconds")
        print("Peak:", peak)

        # Play back the mixdown
        rendered_sound = np.clip(rendered_sound, -1, 1)
        play(rendered_sound)


if __name__ == "__main__":
    # Register signal handler for Ctrl-C to save recording
    if DO_RECORD_REAL_TIME and DO_PLAY_IN_REAL_TIME:
        signal.signal(signal.SIGINT, signal_handler)
    
    if DISABLE_GARBAGE_COLLECTION:
        gc.disable()

    if DO_HOT_RELOAD:
        def yaml_watcher_thread():
            """
            Watch for changes to all YAML files and signal hot reloads.
            
            This runs in a separate thread and sets the yaml_changed flag
            when any YAML file is modified, tracking which file changed.
            """
            global yaml_changed, changed_yaml_file
            import glob
            
            # Get all YAML files in sounds directory
            yaml_files = glob.glob(f"{SOUNDS_DIR}/*.yaml")
            last_modified_times = {f: os.path.getmtime(f) for f in yaml_files}
            
            try:
                while True:
                    try:
                        # Re-scan for new YAML files
                        current_yaml_files = glob.glob(f"{SOUNDS_DIR}/*.yaml")
                        
                        for yaml_file in current_yaml_files:
                            try:
                                current_modified_time = os.path.getmtime(yaml_file)
                                
                                # Check if this is a new file or if it's been modified
                                if yaml_file not in last_modified_times or current_modified_time != last_modified_times[yaml_file]:
                                    # YAML file changed, signal hot reload
                                    yaml_changed = True
                                    changed_yaml_file = yaml_file
                                    last_modified_times[yaml_file] = current_modified_time
                                    break  # Only handle one change at a time
                            except Exception as e:
                                print(f"Error checking YAML file {yaml_file}: {e}")
                    except Exception as e:
                        print(f"Error scanning YAML files: {e}")
                    
                    time.sleep(0.2)
            except KeyboardInterrupt:
                pass
        
        # Start the watcher thread as a daemon so it doesn't prevent exit
        watcher = threading.Thread(target=yaml_watcher_thread, daemon=True)
        watcher.start()
        
        try:
            main()
        except KeyboardInterrupt:
            sys.exit(0)
        except SyntaxError as e:
            # Clean display of expression syntax errors
            print(f"\n{'='*60}")
            print(f"ERROR: {e}")
            print('='*60)
            sys.exit(1)
        except Exception as e:
            # For other errors, show a cleaner message but still include traceback
            print(f"\n{'='*60}")
            print(f"ERROR: {e}")
            print('='*60)
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        try:
            main()
        except SyntaxError as e:
            # Clean display of expression syntax errors
            print(f"\n{'='*60}")
            print(f"ERROR: {e}")
            print('='*60)
            sys.exit(1)
        except Exception as e:
            # For other errors, show the full traceback in non-hot-reload mode
            raise
