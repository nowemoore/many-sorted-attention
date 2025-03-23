from att import TypedTransformer
from datatypes import Instance, Tag, Type

if __name__ == '__main__':
    tree = Type('Tree', [Tag('Branch', []), Tag('Leaf', None)])
    tree.tags[0].sig = [tree, tree]

    model = TypedTransformer(
        types=[tree],
        dim=8,
        num_layers=4,
        num_classes=2,
    )

    leaf1 = tree.tags[1].construct(None)
    leaf2 = tree.tags[1].construct(None)
    node = tree.tags[0].construct([leaf1, leaf2])

    model(node)
