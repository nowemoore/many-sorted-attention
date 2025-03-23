"""
This experiment tests whether the typed transformer can learn the task:
- what is the group element formed by this compute graph?
- 
"""
import scipy.stats as stats
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import scienceplots
from tqdm import trange
import numpy as np
from att import TypedTransformer
from datatypes import Instance, Tag, Type
plt.style.use('science')


add = Tag('+', [])
g0 = Tag('0', None)
g1 = Tag('1', None)
g2 = Tag('2', None)
g3 = Tag('3', None)
g4 = Tag('4', None)
tree = Type(
    'Tree', 
    [
        add, g0, g1, g2, g3, g4
    ]
)
add.sig = [tree, tree]

ops = {
    ('0', '0'): '0',
    ('0', '1'): '1',
    ('0', '2'): '2',
    ('0', '3'): '3',
    ('0', '4'): '4',
    ('1', '0'): '1',
    ('1', '1'): '2',
    ('1', '2'): '3',
    ('1', '3'): '4',
    ('1', '4'): '0',
    ('2', '0'): '2',
    ('2', '1'): '3',
    ('2', '2'): '4',
    ('2', '3'): '0',
    ('2', '4'): '1',
    ('3', '0'): '3',
    ('3', '1'): '4',
    ('3', '2'): '0',
    ('3', '3'): '1',
    ('3', '4'): '2',
    ('4', '0'): '4',
    ('4', '1'): '0',
    ('4', '2'): '1',
    ('4', '3'): '2',
    ('4', '4'): '3',
}


def compute_group(rand_tree: Instance):
    left = rand_tree.values[0]
    right = rand_tree.values[1]
    return ops[left.colour, right.colour]


def random_tree(n):
    if n == 0:
        rand_tree = random.choice([g0, g1, g2, g3, g4]).construct(None)
        rand_tree.colour = rand_tree.tag.name
    else:
        k = random.randint(0, n - 1)
        rand_tree = add.construct(
            [
                random_tree(k),
                random_tree(n - 1 - k)
            ]
        )
        rand_tree.colour = compute_group(rand_tree)

    return rand_tree


def test(batches=1000, batch_size=16):
    model = TypedTransformer(
        types=[tree],
        dim=32,
        num_layers=10,
        num_classes=5,
    )
    size_fn = stats.poisson(20)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    accs = []
    losses = []
    for _ in trange(batches):
        optimizer.zero_grad()

        xs = [random_tree(size_fn.rvs(1).item()) for _ in range(batch_size)]
        logits, indices = model(xs)
        preds = logits.argmax(-1)
        targets = torch.zeros(len(indices), dtype=torch.long)
        for k, v in indices.items():
            targets[v] = int(k.colour)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        accs.append((preds == targets).count_nonzero().item() / targets.size(0))
    
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 4.8), sharex=True)
    ax1.set_ylabel('crossentropy loss')
    ax2.set_ylabel('accuracy')
    ax2.set_xlabel('epoch')
    xs = np.arange(batches)
    ax1.plot(xs, losses)
    ax2.plot(xs, accs)
    plt.savefig('./performance.pdf', format='pdf')


if __name__ == '__main__':
    test(batches=10000, batch_size=32)
