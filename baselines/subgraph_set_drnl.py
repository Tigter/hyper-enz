from typing import List, Dict, Tuple
import torch


def bfs_k_hop_from_set(adj: Dict[int, List[int]], sources: List[int], k: int) -> List[int]:
    from collections import deque
    visited = set(sources)
    q = deque([(s, 0) for s in sources])
    while q:
        u, d = q.popleft()
        if d == k:
            continue
        for v in adj.get(u, []):
            if v not in visited:
                visited.add(v)
                q.append((v, d + 1))
    return sorted(visited)


def shortest_distances_set(adj: Dict[int, List[int]], nodes: List[int], source_set: List[int]) -> Dict[int, int]:
    from collections import deque
    node_set = set(nodes)
    dist = {n: -1 for n in nodes}
    q = deque()
    for s in source_set:
        if s in node_set:
            dist[s] = 0
            q.append(s)
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v in node_set and dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def drnl_set_labels(adj: Dict[int, List[int]], nodes: List[int], H: List[int], T: List[int]) -> torch.Tensor:
    labels = torch.zeros(len(nodes), dtype=torch.long)
    node_pos = {n: i for i, n in enumerate(nodes)}
    Hset = set(H)
    Tset = set(T)
    dh = shortest_distances_set(adj, nodes, H)
    dt = shortest_distances_set(adj, nodes, T)
    for i, n in enumerate(nodes):
        if n in Hset or n in Tset:
            labels[i] = 1
            continue
        d_h = dh[n]
        d_t = dt[n]
        if d_h == -1 or d_t == -1:
            labels[i] = 0
            continue
        d_min = min(d_h, d_t)
        d_max = max(d_h, d_t)
        labels[i] = 1 + d_min + (d_max * (d_max + 1)) // 2
    return labels

