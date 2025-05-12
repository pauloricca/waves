from models.models import BaseNodeModel
from nodes.base import BaseNode
from dataclasses import dataclass

@dataclass
class NodeDefinition:
    name: str
    node: BaseNode
    model: BaseNodeModel
