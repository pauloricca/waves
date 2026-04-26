from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import ConfigDict

from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.wavable_value import WavableValue


class GLSLModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')

    # Input control stream (typically from visual_source in this POC).
    signal: WavableValue = 0.0

    # POC shader selector mirroring a tiny subset of GLSL-style transforms.
    shader: Literal[
        "passthrough",
        "invert",
        "pulse",
        "threshold",
        "rgb_split",
        "scanlines",
        "kaleido",
    ] = "passthrough"

    # Generic shader-like uniforms.
    mix: WavableValue = 1.0
    uniform_a: WavableValue = 0.5
    uniform_b: WavableValue = 1.0
    opacity: WavableValue = 1.0


class GLSLNode(BaseNode):
    """
    POC GLSL-style processing node.

    This is intentionally lightweight: it processes a mono control stream and
    applies a selectable effect with uniform-like parameters. It provides a
    stable node contract that can later be replaced with real GPU shaders.
    """

    def __init__(self, model: GLSLModel, node_id: str, state=None, do_initialise_state=True):
        super().__init__(model, node_id, state, do_initialise_state)
        self.model = model

        self.signal_node = self.instantiate_child_node(model.signal, "signal")
        self.mix_node = self.instantiate_child_node(model.mix, "mix")
        self.uniform_a_node = self.instantiate_child_node(model.uniform_a, "uniform_a")
        self.uniform_b_node = self.instantiate_child_node(model.uniform_b, "uniform_b")
        self.opacity_node = self.instantiate_child_node(model.opacity, "opacity")
        self.last_render_info: dict[str, float | str] = {}

    def _do_render(self, num_samples=None, context=None, **params):
        if num_samples is None:
            num_samples = self.resolve_num_samples(num_samples)
            if num_samples is None:
                return np.array([], dtype=np.float32)

        signal = self.signal_node.render(num_samples, context, **self.get_params_for_children(params))
        mix = self.mix_node.render(num_samples, context, **self.get_params_for_children(params))
        uniform_a = self.uniform_a_node.render(num_samples, context, **self.get_params_for_children(params))
        uniform_b = self.uniform_b_node.render(num_samples, context, **self.get_params_for_children(params))
        opacity = self.opacity_node.render(num_samples, context, **self.get_params_for_children(params))

        signal = np.asarray(signal, dtype=np.float32)
        mix = np.clip(np.asarray(mix, dtype=np.float32), 0.0, 1.0)
        uniform_a = np.asarray(uniform_a, dtype=np.float32)
        uniform_b = np.asarray(uniform_b, dtype=np.float32)
        opacity = np.clip(np.asarray(opacity, dtype=np.float32), 0.0, 1.0)

        if self.model.shader == "invert":
            effected = 1.0 - signal
        elif self.model.shader == "pulse":
            effected = np.sin(signal * np.pi * uniform_b) * uniform_a
        elif self.model.shader == "threshold":
            effected = (signal > uniform_a).astype(np.float32) * uniform_b
        else:
            effected = signal

        mixed = signal + (effected - signal) * mix
        self.last_render_info = {
            "shader": self.model.shader,
            "mix": float(mix[-1]) if len(mix) else 0.0,
            "uniform_a": float(uniform_a[-1]) if len(uniform_a) else 0.0,
            "uniform_b": float(uniform_b[-1]) if len(uniform_b) else 0.0,
            "opacity": float(opacity[-1]) if len(opacity) else 0.0,
            "signal_mean": float(np.mean(signal)) if signal.size else 0.0,
        }
        return (mixed * opacity).astype(np.float32)


GLSL_DEFINITION = NodeDefinition(
    name="glsl",
    model=GLSLModel,
    node=GLSLNode,
)
