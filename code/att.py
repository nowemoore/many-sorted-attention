"""
Implementation of TypedTransformer
A framework for learning on arbitrary discrete datatypes
"""
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from datatypes import Instance, Tag, Type

__all__ = [
    "TypedTransformer",
]

## hi harry

class TypedTransformer(nn.Module):
    """
    Base model for a typed multihead transformer.
    Constructs a typed transformer for a given grammar.

    The grammar must be of the form (represented in a Python dictionary):
    T :=   A of [I, J, ...]
         | B of {I, J, ...}
         | C of n
         | D of None
    
    Where:
    - (n, [I, J]) represents an ordered list of fixed length
    - (n, {I, J}) represents an unorderd list of fixed size
    - (-1, [I, J]) represents an ordered list of variable length
    - (-1, {I, J}) represents an unorderd list of variable size
    - 10 represents a 10-dimensional floating point feature
    - None has no value i.e. Nil of None

    This is intended to provide a general framework for typed data which can attend to any data structure.
    Essentially: we construct a pointer graph, then perform message passing.
    """

    def __init__(
        self,
        types: list[Type],
        dim: int = 64,
        num_layers=4,
        num_classes=10,
    ):
        super().__init__()
        self.dim = dim
        unique_tags: list[Tag] = [tag for type_ in types for tag in type_.tags]
        self.tags: dict[Tag, int] = {
            tag: i
            for i, tag in enumerate(unique_tags)
        }
        self.embedding = nn.Embedding(len(self.tags), dim)

        # case A of [I, J]
        # initialise to embedding

        # case A of {I, J}
        # initialise to embedding

        # case A of float^n
        # initialise to sum of embedding and projection
        self.projs = {
            tag: nn.Linear(tag.sig, dim)
            for tag in self.tags.keys() if isinstance(tag.sig, int)
        }

        # case A of None
        # initialise to embedding
        self.model = gnn.GAT(in_channels=dim,
                             hidden_channels=dim,
                             num_layers=num_layers,
                             out_channels=num_classes,
                             act='SiLU',
                             norm='LayerNorm',
                             v2=True)

    def _process(self, x: Instance, tag_keys: list[int],
                 indices: dict[Instance, int], unvisited: list[int],
                 proj_values: list[torch.Tensor]):
        unvisited.append(x)
        indices[x] = len(indices)
        tag_keys.append(self.tags[x.tag])
        if isinstance(x.values, torch.Tensor):
            proj_values.append(self.projs(self.tags[x.tag]) @ x.values)
        else:
            proj_values.append(torch.zeros(self.dim))

    def initialise(self, xs: list[Instance]):
        """
        Construct the pointer graph
        Take a single object as argument
        return the embeddings corresponding to this object and edge_indices corresponding to the pointer graph
        """
        tag_keys: list[int] = []
        edge_index = []
        indices = {}
        proj_values = []
        batch = []
        for i, x in enumerate(xs):
            unvisited = []
            self._process(
                x,
                tag_keys,
                indices,
                unvisited,
                proj_values,
            )
            batch.append(i)

            while unvisited:
                y = unvisited.pop()
                if not isinstance(y.values, list):
                    continue
                for z in y.values:
                    if z not in indices:
                        self._process(
                            z,
                            tag_keys,
                            indices,
                            unvisited,
                            proj_values,
                        )
                        batch.append(i)
                    edge_index.append([indices[z], indices[y]])
        return self.embedding(
            torch.tensor(tag_keys)) + torch.stack(proj_values), torch.tensor(edge_index), torch.tensor(batch), indices

    def forward(
        self,
        xs: list[Instance],
    ):
        """
        Forward pass of the TypedTransformer
        Take an object as argument and return classification
        """
        xs, edge_index, batch, indices = self.initialise(xs)
        return self.model(xs, edge_index.T, torch.tensor), indices
