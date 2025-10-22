from __future__ import annotations
from pydantic import ConfigDict
import numpy as np
from config import BUFFER_SIZE
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.wavable_value import WavableValue, wavable_value_node_factory


class HoldModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    signal: WavableValue  # WavableValue to sample
    trigger: WavableValue | None = None  # When trigger changes, resample signal
    duration: float | None = None


class HoldNode(BaseNode):
    """
    Samples 'signal' once when 'trigger' changes value, and holds it.
    
    - signal: WavableValue (node, scalar, or expression string)
    - trigger: WavableValue or None. When None, samples once on first render and holds forever.
    
    Note: Change detection is chunk-level: if trigger changes anywhere within the chunk,
          we resample once for the whole chunk.
    """
    def __init__(self, model: HoldModel):
        super().__init__(model)
        self.model = model
        self.signal_node = wavable_value_node_factory(model.signal)
        self.trigger_node = wavable_value_node_factory(model.trigger) if model.trigger is not None else None
        self._held_value: float | None = None
        self._last_trigger_value: float | None = None

    def _sample_signal_once(self, context, **params) -> float:
        """Sample the signal once and return a scalar value."""
        v = self.signal_node.render(1, context, **self.get_params_for_children(params))
        return float(v[0]) if v.size > 0 else 0.0

    def _do_render(self, num_samples=None, context=None, **params):
        if num_samples is None:
            num_samples = self.resolve_num_samples(num_samples)
            if num_samples is None:
                num_samples = BUFFER_SIZE

        # Initialize held value on first render
        if self._held_value is None:
            self._held_value = self._sample_signal_once(context, **params)

        # Prepare output with current held value
        out = np.full(num_samples, float(self._held_value), dtype=np.float32)

        # Evaluate trigger if provided
        if self.trigger_node is not None:
            trig = self.trigger_node.render(num_samples, context, **self.get_params_for_children(params))
            
            # Detect if trigger changed anywhere in this chunk
            trigger_changed = False
            first_change_idx = 0
            
            # Check change from previous chunk to first sample
            if self._last_trigger_value is None or float(trig[0]) != self._last_trigger_value:
                trigger_changed = True
                first_change_idx = 0
            # Check changes within the chunk
            elif len(trig) > 1:
                diff_result = np.diff(trig)
                if np.any(diff_result != 0):
                    trigger_changed = True
                    # Find first position where trigger changes
                    first_change_idx = np.where(diff_result != 0)[0][0] + 1
            
            # If trigger changed, resample signal once and fill from change point onwards
            if trigger_changed:
                new_value = self._sample_signal_once(context, **params)
                # Fill from the change point to the end with the new value
                out[first_change_idx:] = new_value
                # Update held value for next chunk
                self._held_value = new_value
            
            self._last_trigger_value = float(trig[-1])
        
        return out


HOLD_DEFINITION = NodeDefinition("hold", HoldNode, HoldModel)
