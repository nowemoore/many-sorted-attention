from dataclasses import dataclass
from typing import Optional
from torch import Tensor

__all__ = [
    "Instance"
    "Tag",
    "Type",
]


@dataclass
class Type:
    name: str
    tags: list["Tag"]

    def __repr__(self):
        return f'{self.name} := {' | '.join(map(str, self.tags))}'

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == it(other)


@dataclass
class Tag:
    name: str
    sig: Optional[Type | tuple[Type] | int]
    ordered: bool = True

    def __repr__(self):
        if self.sig is None:
            return f'{self.name} of None'
        if isinstance(self.sig, int):
            return f'{self.name} of R^{self.sig}'
        if isinstance(self.sig, Type):
            if self.ordered:
                return f'{self.name} of [{Type.name}, ...]'
            return f'{self.name} of {"{"}{Type.name}, ...{"}"}'
        if isinstance(self.sig, list):
            if self.ordered:
                return f'{self.name} of [{', '.join(map(lambda x: x.name, self.sig))}]'
            return f'{self.name} of {"{"}{', '.join(map(lambda x: x.name, self.sig))}{"}"}'
        raise ValueError(
            f'Expected contents of type Optional[set[Type] | list[Type] | int], but got {type(self.sig)}'
        )

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)

    def construct(self, contents):
        return Instance(self, contents)


@dataclass
class Instance:
    tag: Tag
    values: Optional[list["Instance"] | Tensor]

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)
    
    def __repr__(self):
        if self.values is None:
            return f'{self.tag.name}'
        elif isinstance(self.values, Tensor):
            return f'{self.tag.name}(Tensor)'
        else:
            return f'{self.tag.name}({", ".join(map(str, self.values))})'
