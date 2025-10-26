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
    def __init__(self, model: ReferenceNodeModel, node_id: str, state=None, hot_reload=False):
        super().__init__(model, node_id, state, hot_reload)
        self.ref_id = model.ref
    
    def _do_render(self, num_samples: int = None, context: RenderContext = None, **params) -> np.ndarray:
        if context is None:
            raise ValueError(f"ReferenceNode '{self.ref_id}' requires a render context")
        
        # Get the referenced node instance
        referenced_node = context.get_node(self.ref_id)
        
        if referenced_node is None:
            # Node hasn't been registered yet - topological error
            raise ValueError(f"Referenced node '{self.ref_id}' has not been defined yet. "
                           f"Ensure nodes with id='{self.ref_id}' are defined before they are referenced.")
        
        # Call render on the referenced node - this will handle recursion tracking
        # and return zeros if max recursion depth is reached
        wave = referenced_node.render(num_samples, context, **params)
        
        return wave.copy()  # Return a copy to avoid mutations


REFERENCE_DEFINITION = NodeDefinition(
    name="reference",
    model=ReferenceNodeModel,
    node=ReferenceNode
)
