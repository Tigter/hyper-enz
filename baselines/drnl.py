import torch
from typing import Dict, List, Tuple


def shortest_distances(adj, src: int, nodes: List[int]) -> Dict[int, int]:
    """Compute shortest path distances from src within induced subgraph nodes using BFS."""
    from collections import deque
    node_set = set(nodes)
    dist = {n: -1 for n in nodes}
    dist[src] = 0
    q = deque([src])
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v in node_set and dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def drnl_node_labeling(adj, nodes: List[int], head_idx: int, tail_idx: int) -> torch.Tensor:
    """
    Double-Radius Node Labeling (DRNL) within induced subgraph nodes.
    head_idx and tail_idx are given in local compact id space.
    Returns labels as 1D LongTensor of length len(nodes).
    """
    node_pos = {n: i for i, n in enumerate(nodes)}

    # distances from head and tail
    dh = shortest_distances(adj, head_idx, nodes)
    dt = shortest_distances(adj, tail_idx, nodes)

    labels = torch.zeros(len(nodes), dtype=torch.long)

    for i, n in enumerate(nodes):
        d_h = dh[n]
        d_t = dt[n]
        if d_h == -1 or d_t == -1:
            # unreachable -> assign a large label bucket
            labels[i] = 0
            continue
        if n == head_idx or n == tail_idx:
            labels[i] = 1
            continue
        # DRNL formulation (Zhang & Chen; SEAL):
        # Convert to positive integers; base offset 1 to avoid 0
        d_min = min(d_h, d_t)
        d_max = max(d_h, d_t)
        label = 1 + d_min + (d_max * (d_max + 1)) // 2
        labels[i] = int(label)

    return labels

