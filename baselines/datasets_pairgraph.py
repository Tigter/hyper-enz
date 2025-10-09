import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.data import Data as GeoData
from typing import Dict, List, Tuple

from util.brenda_datasets import load_data
from baselines.graph_build import build_pair_graph, bfs_k_hop
from baselines.drnl import drnl_node_labeling


class TriplePairGraph:
    def __init__(self):
        graph_info, train_info = load_data()
        self.graph_info = graph_info
        self.train_info = train_info
        self.c_num = graph_info['c_num']
        self.e_num = graph_info['e_num']

        # pair graph from training triples only
        train_triples = train_info['train_triple']
        self.node_map, self.inv_nodes, self.edge_index, self.adj = build_pair_graph(train_triples)

        self.all_true = set(train_info['train_triple'] + train_info['valid_triple'] + train_info['test_triple'])

    def map_node(self, nid: int) -> int:
        return self.node_map[nid]


class SEALPairDataset(Dataset):
    def __init__(self, triples: List[Tuple[int, int, int]], pairgraph: TriplePairGraph, k_hop: int = 2, max_nodes: int = 3000):
        super().__init__()
        self.triples = triples
        self.pg = pairgraph
        self.k = k_hop
        self.max_nodes = max_nodes

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx: int):
        h, r, t = self.triples[idx]
        y = r - self.pg.c_num

        # compact ids in pair graph
        if h not in self.pg.node_map or t not in self.pg.node_map:
            # unseen nodes -> empty subgraph
            x = torch.zeros((1, 1), dtype=torch.long)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            batch = torch.zeros(1, dtype=torch.long)
            y = torch.tensor(y, dtype=torch.long)
            return GeoData(x=x, edge_index=edge_index, y=y, batch=batch)

        ch = self.pg.map_node(h)
        ct = self.pg.map_node(t)

        # k-hop enclosing subgraph over pair graph
        nodes = bfs_k_hop(self.pg.adj, [ch, ct], self.k)
        if len(nodes) > self.max_nodes:
            nodes = nodes[:self.max_nodes]
        local_index = {n: i for i, n in enumerate(nodes)}
        # induced edges
        edges = []
        for u in nodes:
            for v in self.pg.adj.get(u, []):
                if v in local_index:
                    edges.append((local_index[u], local_index[v]))
        if edges:
            e = torch.tensor(edges, dtype=torch.long)
            edge_index = e.t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        labels = drnl_node_labeling(self.pg.adj, nodes, ch, ct)
        x = labels.view(-1, 1)  # use label as categorical feature id; model will embed

        batch = torch.zeros(len(nodes), dtype=torch.long)
        return GeoData(x=x, edge_index=edge_index, y=torch.tensor(y, dtype=torch.long), batch=batch)


def build_seal_loaders(config: Dict, split: str):
    pg = TriplePairGraph()
    if split == 'train':
        triples = pg.train_info['train_triple']
    elif split == 'valid':
        triples = pg.train_info['valid_triple']
    else:
        triples = pg.train_info['test_triple']

    ds = SEALPairDataset(triples, pg, k_hop=config.get('k_hop', 2), max_nodes=config.get('max_nodes', 3000))
    bs = config.get('batch_size', 256 if split == 'train' else 128)
    return GeoDataLoader(ds, batch_size=bs, shuffle=(split == 'train')), pg
