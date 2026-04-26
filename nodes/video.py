from __future__ import annotations

import os

import numpy as np
from pydantic import ConfigDict

from config import SAMPLE_RATE
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.wavable_value import WavableValue
from utils import empty_mono


class VideoModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')

    # A single source path for quick single-file use.
    file: str | None = None
    # Optional source bank. Use `clip` to switch between files in real time.
    files: list[str] | None = None
    clip: WavableValue | int | None = None

    # Transport controls.
    start: WavableValue = 0.0  # Start offset in seconds; playhead is relative to this
    clip_duration: WavableValue | None = None  # Optional loop/playhead span in seconds; defaults to full media duration
    speed: WavableValue = 1.0  # negative values run backwards
    playhead: WavableValue | None = None  # Optional explicit playhead in seconds
    paused: bool = False
    loop: bool = True

    # Presentation-oriented controls (metadata for downstream visual systems).
    fullscreen: bool = False


class VideoNode(BaseNode):
    def __init__(self, model: VideoModel, node_id: str, state=None, do_initialise_state=True):
        super().__init__(model, node_id, state, do_initialise_state)
        self.model = model

        self.start_node = self.instantiate_child_node(model.start, "start")
        self.speed_node = self.instantiate_child_node(model.speed, "speed")
        self.clip_duration_node = self.instantiate_child_node(model.clip_duration, "clip_duration") if model.clip_duration is not None else None
        self.playhead_node = self.instantiate_child_node(model.playhead, "playhead") if model.playhead is not None else None
        self.clip_node = self.instantiate_child_node(model.clip, "clip") if model.clip is not None else None

        if do_initialise_state:
            self.state.playhead_seconds = 0.0
            self.state.active_clip_index = 0

        self._sources = self._build_sources(model)
        self.last_frame_info: dict[str, object] = {}

    def _build_sources(self, model: VideoModel) -> list[str]:
        sources: list[str] = []
        if model.file:
            sources.append(model.file)
        if model.files:
            sources.extend(model.files)
        return sources

    def _update_active_clip(self, num_samples: int, context, params) -> None:
        if self.clip_node is None or not self._sources:
            return

        clip_value = self.clip_node.render(num_samples, context, **self.get_params_for_children(params))
        clip_scalar = float(clip_value.flat[-1]) if isinstance(clip_value, np.ndarray) and clip_value.size > 0 else float(clip_value)
        clip_index = int(np.clip(int(clip_scalar), 0, len(self._sources) - 1))

        if clip_index != self.state.active_clip_index:
            self.state.active_clip_index = clip_index
            self.state.playhead_seconds = 0.0

    def _resolve_playhead(self, num_samples: int, context, params, speed: np.ndarray, clip_duration: float | None) -> np.ndarray:
        if self.playhead_node is not None:
            playhead = self.playhead_node.render(num_samples, context, **self.get_params_for_children(params))
            if np.isscalar(playhead):
                playhead = np.full(num_samples, playhead, dtype=np.float32)
            playhead = np.asarray(playhead, dtype=np.float32)
            if clip_duration is not None and self.model.loop:
                playhead = np.mod(playhead, clip_duration)
            elif clip_duration is None:
                playhead = np.maximum(playhead, 0.0)
            else:
                playhead = np.maximum(playhead, 0.0)
            self.state.playhead_seconds = float(playhead[-1]) if len(playhead) > 0 else self.state.playhead_seconds
            return playhead

        delta_seconds = speed / SAMPLE_RATE
        playhead = self.state.playhead_seconds + np.cumsum(delta_seconds)

        if clip_duration is not None and self.model.loop:
            playhead = np.mod(playhead, clip_duration)
        elif clip_duration is not None:
            playhead = np.clip(playhead, 0.0, clip_duration)

        self.state.playhead_seconds = float(playhead[-1]) if len(playhead) > 0 else self.state.playhead_seconds
        return playhead

    def _do_render(self, num_samples=None, context=None, **params):
        if num_samples is None:
            num_samples = self.resolve_num_samples(num_samples)
            if num_samples is None:
                return empty_mono()

        if num_samples == 0:
            return empty_mono()

        self._update_active_clip(num_samples, context, params)

        speed = self.speed_node.render(num_samples, context, **self.get_params_for_children(params))
        start_values = self.start_node.render(num_samples, context, **self.get_params_for_children(params))

        if np.isscalar(speed):
            speed = np.full(num_samples, speed, dtype=np.float32)
        if np.isscalar(start_values):
            start_values = np.full(num_samples, start_values, dtype=np.float32)

        speed = np.asarray(speed, dtype=np.float32)
        start_values = np.asarray(start_values, dtype=np.float32)

        if self.model.paused:
            speed = np.zeros_like(speed)

        clip_duration = None
        if self.clip_duration_node is not None:
            clip_duration_values = self.clip_duration_node.render(num_samples, context, **self.get_params_for_children(params))
            if np.isscalar(clip_duration_values):
                clip_duration_values = np.full(num_samples, clip_duration_values, dtype=np.float32)
            clip_duration_values = np.asarray(clip_duration_values, dtype=np.float32)
            clip_duration = max(float(clip_duration_values[-1]) if len(clip_duration_values) else 0.0, 1e-6)
        playhead = self._resolve_playhead(num_samples, context, params, speed, clip_duration)
        playhead_normalized = playhead / clip_duration if clip_duration is not None else playhead

        signal = playhead_normalized

        active_source = None
        if self._sources:
            active_source = self._sources[self.state.active_clip_index]

        start_seconds = float(start_values[-1]) if len(start_values) else 0.0

        self.last_frame_info = {
            "source": active_source,
            "source_path": os.path.abspath(active_source) if active_source else None,
            "clip_index": self.state.active_clip_index,
            "start_seconds": start_seconds,
            "clip_duration": clip_duration,
            "fullscreen": self.model.fullscreen,
            "loop": self.model.loop,
            "playhead": float(playhead_normalized[-1]) if len(playhead_normalized) else 0.0,
            "playhead_seconds": float(playhead[-1]) if len(playhead) else self.state.playhead_seconds,
            "effective_playhead_seconds": start_seconds + (float(playhead[-1]) if len(playhead) else self.state.playhead_seconds),
            "playhead_owner": "node" if self.playhead_node is not None else "renderer",
            "speed": float(speed[-1]) if len(speed) else 0.0,
            "paused": self.model.paused,
        }

        return signal.astype(np.float32)


VIDEO_DEFINITION = NodeDefinition(
    name="video",
    model=VideoModel,
    node=VideoNode,
)
