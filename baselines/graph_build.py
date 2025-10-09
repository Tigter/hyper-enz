import torch
from typing import Dict, List, Tuple


def build_pair_graph(triples: List[Tuple[int, int, int]]):
    """
    Build an undirected pair graph G_pair from triples (h, r, t), ignoring relation.
    Returns:
      node_map: dict original_id -> compact_id
      inv_nodes: list of original_ids by compact_id
      edge_index: torch.LongTensor(2, E) in compact_id space (undirected, deduplicated)
      adj: dict[compact_id] -> List[compact_id] adjacency for BFS
    """
    nodes = set()
    edges = set()
    for h, _, t in triples:
        nodes.add(h)
        nodes.add(t)
        # undirected
        edges.add((h, t))
        edges.add((t, h))

    inv_nodes = sorted(nodes)
    node_map = {n: i for i, n in enumerate(inv_nodes)}

    if edges:
        e = torch.tensor([[node_map[u], node_map[v]] for (u, v) in edges], dtype=torch.long)
        edge_index = e.t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # adjacency list
    adj = {i: [] for i in range(len(inv_nodes))}
    for i in range(edge_index.size(1)):
        u = int(edge_index[0, i].item())
        v = int(edge_index[1, i].item())
        adj[u].append(v)

    return node_map, inv_nodes, edge_index, adj


def bfs_k_hop(adj, srcs: List[int], k: int) -> List[int]:
    """BFS from multiple sources up to k hops; returns sorted list of visited nodes."""
    from collections import deque
    visited = set(srcs)
    q = deque([(s, 0) for s in srcs])
    while q:
        u, d = q.popleft()
        if d == k:
            continue
        for v in adj.get(u, []):
            if v not in visited:
                visited.add(v)
                q.append((v, d + 1))
    return sorted(visited)


def build_pair_graph_with_nodes(edge_triples: List[Tuple[int, int, int]], node_triples: List[Tuple[int, int, int]]):
    """
    Build pair graph where edges come from edge_triples (train), but node universe is from node_triples (train+valid+test).
    This prevents missing-node issues when mapping heads/tails present in other splits.
    """
    nodes = set()
    for h, _, t in node_triples:
        nodes.add(h)
        nodes.add(t)

    inv_nodes = sorted(nodes)
    node_map = {n: i for i, n in enumerate(inv_nodes)}

    edge_set = set()
    for h, _, t in edge_triples:
        if h in node_map and t in node_map:
            edge_set.add((node_map[h], node_map[t]))
            edge_set.add((node_map[t], node_map[h]))

    if edge_set:
        e = torch.tensor(list(edge_set), dtype=torch.long)
        edge_index = e.t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    adj = {i: [] for i in range(len(inv_nodes))}
    for i in range(edge_index.size(1)):
        u = int(edge_index[0, i].item())
        v = int(edge_index[1, i].item())
        adj[u].append(v)

    return node_map, inv_nodes, edge_index, adj
