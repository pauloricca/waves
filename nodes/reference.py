from __future__ import annotations
import numpy as np
from pydantic import ConfigDict
from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from nodes.node_utils.node_definition_type import NodeDefinition
from nodes.node_utils.render_context import RenderContext


class ReferenceNodeModel(BaseNodeModel):
    model_config = ConfigDict(extra='forbid')
    ref: str  # The id of the node to reference


class ReferenceNode(BaseNode):
    def __init__(self, model: ReferenceNodeModel):
        super().__init__(model)
        self.ref_id = model.ref
    
    def _do_render(self, num_samples: int = None, context: RenderContext = None, **params) -> np.ndarray:
        if context is None:
            raise ValueError(f"ReferenceNode '{self.ref_id}' requires a render context")
        
        # Get the cached output from the referenced node
        wave = context.get_output(self.ref_id)
        
        if wave is None:
            # Referenced node hasn't been rendered yet
            # Could be a topological issue or the node is defined later
            raise ValueError(f"Referenced node '{self.ref_id}' has not been rendered yet. "
                           f"Ensure nodes with id='{self.ref_id}' are defined before they are referenced.")
        
        # Adjust output length to match requested samples
        if num_samples is not None and len(wave) > 0:
            if len(wave) < num_samples:
                # Pad with zeros
                padding = np.zeros(num_samples - len(wave), dtype=np.float32)
                wave = np.concatenate([wave, padding])
            elif len(wave) > num_samples:
                wave = wave[:num_samples]
        
        return wave.copy()  # Return a copy to avoid mutations


REFERENCE_DEFINITION = NodeDefinition(
    name="reference",
    model=ReferenceNodeModel,
    node=ReferenceNode
)
