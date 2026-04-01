from __future__ import annotations

import numpy as np
from pydantic import ConfigDict

from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.wavable_value import WavableValue


class FoldModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    threshold: WavableValue = 1  # Symmetric fold threshold applied at +/- threshold
    bias: WavableValue = 0  # Offset applied before folding
    multiplier: WavableValue = 1  # Gain applied after bias, before folding
    signal: WavableValue  # The incoming signal to fold


class FoldNode(BaseNode):
    def __init__(self, model: FoldModel, node_id: str, state=None, do_initialise_state=True):
        super().__init__(model, node_id, state, do_initialise_state)
        self.model = model
        self.signal_node = self.instantiate_child_node(model.signal, "signal")
        self.threshold_node = self.instantiate_child_node(model.threshold, "threshold")
        self.bias_node = self.instantiate_child_node(model.bias, "bias")
        self.multiplier_node = self.instantiate_child_node(model.multiplier, "multiplier")

    def _do_render(self, num_samples=None, context=None, **params):
        num_samples = self.resolve_num_samples(num_samples)
        child_params = self.get_params_for_children(params)

        if num_samples is None:
            signal_wave = self.render_full_child_signal(self.signal_node, context, **child_params)
            if len(signal_wave) == 0:
                return np.array([], dtype=np.float32)
            num_samples = len(signal_wave)
        else:
            signal_wave = self.signal_node.render(num_samples, context, **child_params)
            if len(signal_wave) == 0:
                return np.array([], dtype=np.float32)

        threshold_wave = self.threshold_node.render(num_samples, context, **child_params)
        bias_wave = self.bias_node.render(num_samples, context, **child_params)
        multiplier_wave = self.multiplier_node.render(num_samples, context, **child_params)

        driven_wave = (signal_wave + bias_wave) * multiplier_wave
        threshold_wave = np.maximum(np.abs(threshold_wave), 1e-12)
        period = 4.0 * threshold_wave

        folded_wave = np.abs(np.mod(driven_wave + threshold_wave, period) - 2.0 * threshold_wave) - threshold_wave
        return folded_wave.astype(np.float32, copy=False)


FOLD_DEFINITION = NodeDefinition("fold", FoldNode, FoldModel)
