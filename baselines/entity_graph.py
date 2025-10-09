from typing import List, Tuple, Dict
import torch


def build_entity_graph_from_train(train: List[Tuple[List[int], List[int], int]], num_entities: int):
    """
    Build undirected bipartite edges between substrates and products using only train triples.
    Nodes are entity ids in [0, num_entities).
    Returns edge_index (2, E) LongTensor and adjacency dict for BFS.
    """
    edges = set()
    for H, T, _ in train:
        for s in H:
            for p in T:
                if 0 <= s < num_entities and 0 <= p < num_entities:
                    edges.add((s, p))
                    edges.add((p, s))
    if edges:
        e = torch.tensor(list(edges), dtype=torch.long)
        edge_index = e.t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    adj = {i: [] for i in range(num_entities)}
    for i in range(edge_index.size(1)):
        u = int(edge_index[0, i].item())
        v = int(edge_index[1, i].item())
        adj[u].append(v)

    return edge_index, adj

