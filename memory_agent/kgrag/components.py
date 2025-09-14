from pydantic import BaseModel, model_validator
from typing import List, Union, Any


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
    target_node: Union[str, List[str]]
    relationship: str

    @model_validator(mode='before')
    def normalize_target_node(cls, data: dict[str, Any]) -> dict[str, Any]:
        tn = data.get('target_node')
        if isinstance(tn, str):
            data['target_node'] = [tn]
        elif tn is None:
            # gestisci caso None se vuoi, ad esempio lista vuota o levare
            data['target_node'] = []
        # se è già una lista, assumiamo che contenga stringhe
        return data


class GraphComponents(BaseModel):
    """
    Represents the components of a graph.
    Attributes:
        graph (list[single]): A list of nodes in the graph.
    """
    graph: list[single]
