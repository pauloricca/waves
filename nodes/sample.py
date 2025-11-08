from __future__ import annotations
import os
import numpy as np
from pydantic import ConfigDict
from config import SAMPLE_RATE
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.wavable_value import WavableValue
from utils import load_wav_file, empty_mono, time_to_samples


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
    base_freq: WavableValue = None  # Hz (base frequency for freq parameter)
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
        # All other fields are regular attributes (not in state)
        self.is_in_chop_mode = model.chop is not None
        self._reset_playhead_this_chunk = False
        self.chop_node = self.instantiate_child_node(model.chop, "chop") if self.is_in_chop_mode else None
        self.audio_files = None
        if self.is_in_chop_mode:
            self.audio = np.zeros(1, dtype=np.float32)
        else:
            self.audio = load_wav_file(model.file)
        self.speed_node = self.instantiate_child_node(model.speed, "speed")
        self.freq_node = self.instantiate_child_node(model.freq, "freq") if model.freq is not None else None
        self.base_freq_node = self.instantiate_child_node(model.base_freq, "base_freq") if model.base_freq is not None else None
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

        # Evaluate chop as a scalar (one sample, mono)
        try:
            raw = self.chop_node.render(1, context, 1, **self.get_params_for_children(params))
        except Exception as e:
            raise ValueError(f"Sample node: error evaluating 'chop': {e}")
        
        # Extract scalar value - flatten in case of stereo/multi-dim and take first
        if isinstance(raw, np.ndarray):
            idx_val = raw.flat[0] if raw.size > 0 else 0
        else:
            idx_val = raw
        
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
            max_total_samples = time_to_samples(self.model.duration )
            if self.state.total_samples_rendered >= max_total_samples:
                # Already rendered full duration
                return empty_mono()
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
        if self.freq_node is not None and self.base_freq_node is not None:
            freq = self.freq_node.render(num_samples, context, **self.get_params_for_children(params))
            base_freq = self.base_freq_node.render(num_samples, context, **self.get_params_for_children(params))
            # Ensure both are arrays
            if np.isscalar(freq):
                freq = np.full(num_samples, freq)
            if np.isscalar(base_freq):
                base_freq = np.full(num_samples, base_freq)
            # Avoid division by zero
            base_freq = np.maximum(base_freq, 1e-6)
            # Multiply speed by the frequency ratio (current_freq / base_freq)
            speed = speed * (freq / base_freq)

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

        if not self.model.loop:
            # Non-looping: find where we exceed the window bounds
            outside_bounds = (playhead_with_offset >= end_indices) | (playhead_with_offset < start_indices)
            if np.any(outside_bounds):
                # Find first sample that's outside bounds
                first_outside = np.where(outside_bounds)[0][0]
                if first_outside == 0:
                    # Already outside on first sample, we're done
                    return empty_mono()
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
            return empty_mono()

        # Clip to valid audio buffer range
        audio_indices = np.clip(audio_indices, 0, len(self.audio) - 1)
        # Interpolate from the audio buffer
        wave = np.interp(audio_indices, np.arange(len(self.audio)), self.audio)

        self.state.last_playhead_position = playhead_absolute[-1] if len(playhead_absolute) > 0 else self.state.last_playhead_position
        self.state.total_samples_rendered += len(wave)

        return wave

        # Uncomment the following lines if you want to implement looping with overlap, but this doesn't work with variable speed
        # if self.model.loop:
        #     result = np.zeros(num_samples)
        #     overlap_samples = int(final_len * self.model.overlap)
        #     effective_length = final_len - overlap_samples if overlap_samples > 0 else final_len
        #     position = 0

        #     while position < num_samples:
        #         remaining = num_samples - position
        #         to_copy = min(final_len, remaining)

        #         segment = wave[:to_copy].copy()
        #         if overlap_samples > 0 and to_copy > overlap_samples:
        #             fade_out = np.linspace(1, 0, overlap_samples)
        #             fade_in = np.linspace(0, 1, overlap_samples)
        #             segment[-overlap_samples:] *= fade_out
        #             if position > 0:
        #                 result[position:position+overlap_samples] += wave[:overlap_samples] * fade_in
        #         result[position:position+to_copy] += segment
        #         position += effective_length
        #     return result
        # else:
        #     return np.pad(wave, (0, num_samples - final_len), 'constant')


SAMPLE_DEFINITION = NodeDefinition("sample", SampleNode, SampleModel)
