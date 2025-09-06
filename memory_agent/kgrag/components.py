from pydantic import BaseModel
from typing import List


class single(BaseModel):
    """
    Represents a single node in a graph with its relationships.
    Attributes:
        node (str): The name of the node.
        target_node (str | List[str]): The target node(s) this
            node is connected to.
        relationship (str): The type of relationship to the target node(s).
    """
    node: str
    target_node: str | list[str] | List[str]
    relationship: str


class GraphComponents(BaseModel):
    """
    Represents the components of a graph.
    Attributes:
        graph (list[single]): A list of nodes in the graph.
    """
    graph: list[single]
