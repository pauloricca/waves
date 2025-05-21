from nodes.node_utils.base_node import BaseNode, BaseNodeModel
from dataclasses import dataclass

@dataclass
class NodeDefinition:
    name: str
    node: BaseNode
    model: BaseNodeModel
