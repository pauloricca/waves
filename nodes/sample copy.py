from __future__ import annotations
import os
import numpy as np
from pydantic import ConfigDict
from config import SAMPLE_RATE
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.wavable_value import WavableValue
from utils import load_wav_file


# Sample node with offset parameter:
# offset allows you to shift the playhead position by a certain amount (in samples).
# This can be used with an LFO to create warble, tape slow-down/speed-up effects, etc.
# Positive values shift forward, negative values shift backward.

class SampleModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    file: str = None
    start: WavableValue = 0.0  # 0.0-1.0 (normalized position in sample)
    end: WavableValue = 1.0  # 0.0-1.0 (normalized position in sample)
    loop: bool = False
    overlap: float = 0.0  # 0.0-1.0 (normalized fraction of sample length)
    speed: WavableValue = 1.0  # playback speed multiplier (1.0 = normal speed)
    base_freq: float = None  # Hz (base frequency for freq parameter)
    freq: WavableValue = None  # Hz (target frequency - modulates speed based on base_freq)
    offset: WavableValue = 0.0  # seconds (shift playhead position)
    # When provided, 'file' is treated as a directory path. The 'chop' selects
    # one of the audio files in that directory (sorted alphabetically).
    chop: WavableValue | int | None = None


class SampleNode(BaseNode):
    def __init__(self, model: SampleModel, node_id: str, state, do_initialise_state=True):
        super().__init__(model, node_id, state, do_initialise_state)
        self.model = model
        # Only persistent playback state is kept in self.state
        if do_initialise_state:
            self.state.last_playhead_position = 0
            self.state.total_samples_rendered = 0
            self.state.current_chop_index = None
            # State for overlap crossfading when looping
            self.state.overlap_buffer = None  # Buffer to store the remaining fade-out tail
        # All other fields are regular attributes (not in state)
        self.is_in_chop_mode = model.chop is not None
        self._reset_playhead_this_chunk = False
        self._cached_tail = None  # Cache for the loop tail to avoid recomputing
        self._cached_tail_for_indices = None  # Track what indices the tail is for
        self.chop_node = self.instantiate_child_node(model.chop, "chop") if self.is_in_chop_mode else None
        self.audio_files = None
        if self.is_in_chop_mode:
            self.audio = np.zeros(1, dtype=np.float32)
        else:
            self.audio = load_wav_file(model.file)
        self.speed_node = self.instantiate_child_node(model.speed, "speed")
        self.freq_node = self.instantiate_child_node(model.freq, "freq") if model.freq is not None else None
        self.start_node = self.instantiate_child_node(model.start, "start")
        self.end_node = self.instantiate_child_node(model.end, "end")
        self.offset_node = self.instantiate_child_node(model.offset, "offset")
        # Only persistent playback state is kept in self.state (see above)

    def _resolve_project_path(self, path_str: str) -> str:
        """Resolve path relative to project root (waves/), handling ~ and env vars."""
        path_str = os.path.expandvars(os.path.expanduser(path_str))
        if os.path.isabs(path_str):
            return path_str
        project_root = os.path.dirname(os.path.dirname(__file__))  # nodes/ -> project root
        return os.path.normpath(os.path.join(project_root, path_str))

    def _ensure_chop_listing(self):
        """Build and cache the list of .wav files for chop mode."""
        if "_chop_files" in vars(self):
            return
        base = self.model.file
        if base is None:
            raise ValueError("Sample node: 'file' is required")
        base_abs = self._resolve_project_path(base)
        if not os.path.isdir(base_abs):
            raise ValueError(f"Sample node: 'file' should be a directory when 'chop' is provided. "
                           f"Not a directory: {base} (resolved: {base_abs})")
        try:
            entries = [f for f in os.listdir(base_abs) if f.lower().endswith(".wav")]
        except FileNotFoundError:
            raise ValueError(f"Sample node: directory not found: {base} (resolved: {base_abs})")
        entries.sort(key=lambda s: s.lower())
        if not entries:
            raise ValueError(f"Sample node: no .wav files found in directory: {base} (resolved: {base_abs})")
        # Cache relative paths for load_wav_file
        self._chop_files = [os.path.join(base, e) for e in entries]

    def _select_chop_path_for_index(self, idx: int) -> tuple[str, int]:
        """Return relative path of selected file by clipped index."""
        if "_chop_files" not in vars(self):
            self._ensure_chop_listing()
        files = self._chop_files
        if not files:
            raise ValueError("Sample node: internal error, chop file list is empty")
        idx = int(np.clip(int(idx), 0, len(files) - 1))
        return files[idx], idx

    def _pick_and_load_chop_audio(self, context, params):
        """Evaluate chop node once per render and swap audio if selection changed."""
        if not self.is_in_chop_mode:
            return  # not in chop mode
        # Ensure directory listing is ready
        self._ensure_chop_listing()

        # Evaluate chop as a scalar (one sample)
        try:
            raw = self.chop_node.render(1, context, **self.get_params_for_children(params))
        except Exception as e:
            raise ValueError(f"Sample node: error evaluating 'chop': {e}")
        idx_val = raw[0] if isinstance(raw, np.ndarray) and raw.size > 0 else raw
        try:
            idx_int = int(idx_val)
        except Exception:
            idx_int = 0

        path, idx_int = self._select_chop_path_for_index(idx_int)

        # Load only if changed
        if self.state.current_chop_index != idx_int or len(self.audio) <= 1:
            self.audio = load_wav_file(path)  # utils caches by path
            self.state.current_chop_index = idx_int
            # Reset playhead for this chunk to avoid mid-file discontinuity
            self._reset_playhead_this_chunk = True

    def _do_render(self, num_samples=None, context=None, num_channels=1, **params):
        # Evaluate 'chop' at every render and swap audio if needed
        self._pick_and_load_chop_audio(context, params)

        # If num_samples is None, resolve it from duration
        if num_samples is None:
            num_samples = self.resolve_num_samples(num_samples)
            if num_samples is None:
                # Render start/end for a single sample to get their initial values
                start_vals = self.start_node.render(1, context, **self.get_params_for_children(params))
                end_vals = self.end_node.render(1, context, **self.get_params_for_children(params))
                start = int(start_vals[0] * len(self.audio))
                end = int(end_vals[0] * len(self.audio))
                end = min(end, len(self.audio))
                start = max(start, 0)
                sample_length = end - start
                if sample_length <= 0:
                    return np.array([])
                # For non-looping samples, length is just the sample length
                if not self.model.loop:
                    num_samples = sample_length
                else:
                    raise ValueError("Cannot render full signal: looping sample requires duration to be specified")

        # Render start and end values for this chunk
        start_vals = self.start_node.render(num_samples, context, **self.get_params_for_children(params))
        end_vals = self.end_node.render(num_samples, context, **self.get_params_for_children(params))

        # Ensure they are arrays
        if np.isscalar(start_vals):
            start_vals = np.full(num_samples, start_vals)
        if np.isscalar(end_vals):
            end_vals = np.full(num_samples, end_vals)

        # Convert to sample indices
        start_indices = np.clip(start_vals * len(self.audio), 0, len(self.audio) - 1).astype(int)
        end_indices = np.clip(end_vals * len(self.audio), 0, len(self.audio)).astype(int)

        # Initialize playhead to start position on first render or when chop changes
        if (self.number_of_chunks_rendered == 0 and self.state.last_playhead_position == 0) or self._reset_playhead_this_chunk:
            self.state.last_playhead_position = float(start_indices[0])
            self._reset_playhead_this_chunk = False

        # Check if all start/end pairs are valid
        if np.any(end_indices <= start_indices):
            # For simplicity, when ranges are invalid, return zeros for those samples
            # In a more sophisticated implementation, we could handle this per-sample
            pass

        # Check if we have a duration limit and if we've exceeded it
        if self.model.duration is not None:
            max_total_samples = int(self.model.duration * SAMPLE_RATE)
            if self.state.total_samples_rendered >= max_total_samples:
                # Already rendered full duration
                return np.array([], dtype=np.float32)
            # Check if this chunk would exceed the duration
            samples_remaining = max_total_samples - self.state.total_samples_rendered
            if num_samples > samples_remaining:
                num_samples = samples_remaining
                # Truncate start/end arrays as well
                start_indices = start_indices[:num_samples]
                end_indices = end_indices[:num_samples]

        speed = self.speed_node.render(num_samples, context, **self.get_params_for_children(params))

        # Ensure speed is an array
        if np.isscalar(speed):
            speed = np.full(num_samples, speed)

        # If freq is set and base_freq is set, modulate speed by frequency ratio
        if self.freq_node is not None and self.model.base_freq is not None:
            freq = self.freq_node.render(num_samples, context, **self.get_params_for_children(params))
            # Ensure freq is an array
            if np.isscalar(freq):
                freq = np.full(num_samples, freq)
            # Multiply speed by the frequency ratio (current_freq / base_freq)
            speed = speed * (freq / self.model.base_freq)

        # Render offset
        offset = self.offset_node.render(num_samples, context, **self.get_params_for_children(params))

        # Ensure offset is an array
        if np.isscalar(offset):
            offset = np.full(num_samples, offset)

        # Convert offset from seconds to samples
        offset_samples = offset * SAMPLE_RATE

        # Use absolute speed for rate calculation but keep sign for direction
        abs_speed = np.abs(speed)
        abs_speed = np.maximum(abs_speed, 1e-6)  # avoid zero speed
        sign = np.sign(speed)
        dt = 1.0 / SAMPLE_RATE

        # Calculate the window lengths for each sample
        window_lengths = end_indices - start_indices
        window_lengths = np.maximum(window_lengths, 1)  # Ensure at least 1 sample

        # Integrate speed to get absolute playhead position in the audio buffer
        # The playhead moves through the actual audio buffer, not relative to the window
        playhead_delta = abs_speed * sign * dt * SAMPLE_RATE
        playhead_absolute = self.state.last_playhead_position + np.cumsum(playhead_delta)

        # Apply offset (in samples) - this is applied as a displacement, not accumulated
        # Save the unmodified playhead for updating last_playhead_position
        playhead_with_offset = playhead_absolute + offset_samples

        # Track which samples wrapped for crossfade purposes
        did_wrap = None
        
        if not self.model.loop:
            # Non-looping: find where we exceed the window bounds
            outside_bounds = (playhead_with_offset >= end_indices) | (playhead_with_offset < start_indices)
            if np.any(outside_bounds):
                # Find first sample that's outside bounds
                first_outside = np.where(outside_bounds)[0][0]
                if first_outside == 0:
                    # Already outside on first sample, we're done
                    return np.array([], dtype=np.float32)
                # Truncate to just before going outside
                num_samples = first_outside
                playhead_with_offset = playhead_with_offset[:num_samples]
                playhead_absolute = playhead_absolute[:num_samples]
                start_indices = start_indices[:num_samples]
                end_indices = end_indices[:num_samples]
            audio_indices = playhead_with_offset
        else:
            # Looping: wrap around when outside the window
            # Check which samples are outside their window
            outside_high = playhead_with_offset >= end_indices
            outside_low = playhead_with_offset < start_indices
            outside = outside_high | outside_low
            did_wrap = outside.copy()  # Track where wrapping occurred
            if np.any(outside):
                # For samples outside the window, wrap them back
                # Calculate offset from start of window
                offset_from_start = playhead_with_offset - start_indices
                # Wrap within window length
                wrapped_offset = np.mod(offset_from_start, window_lengths)
                # Update playhead for wrapped samples
                playhead_with_offset = np.where(outside, start_indices + wrapped_offset, playhead_with_offset)
            audio_indices = playhead_with_offset

        # Handle the case where we truncated early (non-looping)
        if num_samples == 0:
            return np.array([], dtype=np.float32)

        # Clip to valid audio buffer range
        audio_indices = np.clip(audio_indices, 0, len(self.audio) - 1)
        # Interpolate from the audio buffer
        wave = np.interp(audio_indices, np.arange(len(self.audio)), self.audio)

        # Handle overlap crossfading for looping samples
        if self.model.loop and self.model.overlap > 0 and did_wrap is not None:
            # Calculate overlap size in samples (based on the average window length)
            avg_window_length = int(np.mean(window_lengths))
            overlap_samples = int(avg_window_length * self.model.overlap)
            
            if overlap_samples > 0:
                # Create output buffer
                output = wave.copy()
                
                # First, apply any remaining overlap from previous chunk
                if self.state.overlap_buffer is not None:
                    samples_to_apply = min(len(self.state.overlap_buffer), num_samples)
                    
                    # Simple crossfade: tail fades out, new signal fades in
                    fade = np.linspace(0, samples_to_apply / overlap_samples, samples_to_apply)
                    
                    output[:samples_to_apply] = (
                        wave[:samples_to_apply] * fade +
                        self.state.overlap_buffer[:samples_to_apply] * (1 - fade)
                    )
                    
                    # Remove the used portion from the buffer
                    if samples_to_apply >= len(self.state.overlap_buffer):
                        # We've consumed all the overlap buffer
                        self.state.overlap_buffer = None
                    else:
                        # Keep the remaining portion for next chunk
                        self.state.overlap_buffer = self.state.overlap_buffer[samples_to_apply:]
                
                # Check if we wrapped in this chunk (and we're not already in the middle of an overlap)
                if self.state.overlap_buffer is None and np.any(did_wrap):
                    # Find the first wrap point
                    wrap_idx = np.where(did_wrap)[0][0]
                    
                    # Get the end position for this wrap
                    end_pos = end_indices[wrap_idx]
                    
                    # Check if we can use cached tail
                    cache_key = (end_pos, overlap_samples)
                    if self._cached_tail_for_indices != cache_key:
                        # Compute and cache the tail
                        tail_start_pos = max(0, end_pos - overlap_samples)
                        tail_indices = np.arange(tail_start_pos, end_pos, dtype=np.float32)
                        tail_indices = np.clip(tail_indices, 0, len(self.audio) - 1)
                        tail = np.interp(tail_indices, np.arange(len(self.audio)), self.audio)
                        
                        # Ensure tail is the right length
                        if len(tail) < overlap_samples:
                            tail = np.pad(tail, (overlap_samples - len(tail), 0), 'edge')
                        
                        self._cached_tail = tail
                        self._cached_tail_for_indices = cache_key
                    else:
                        tail = self._cached_tail
                    
                    # How many samples do we have after the wrap in this chunk?
                    samples_after_wrap = num_samples - wrap_idx
                    samples_to_fade = min(samples_after_wrap, overlap_samples)
                    
                    # Apply crossfade for the portion we have in this chunk
                    fade = np.linspace(0, samples_to_fade / overlap_samples, samples_to_fade)
                    output[wrap_idx:wrap_idx + samples_to_fade] = (
                        wave[wrap_idx:wrap_idx + samples_to_fade] * fade +
                        tail[:samples_to_fade] * (1 - fade)
                    )
                    
                    # If we haven't completed the full crossfade, store the remainder
                    if samples_to_fade < overlap_samples:
                        self.state.overlap_buffer = tail[samples_to_fade:]
                    else:
                        self.state.overlap_buffer = None
                
                wave = output
        
        self.state.last_playhead_position = playhead_absolute[-1] if len(playhead_absolute) > 0 else self.state.last_playhead_position
        self.state.total_samples_rendered += len(wave)

        return wave


SAMPLE_DEFINITION = NodeDefinition("sample", SampleNode, SampleModel)
