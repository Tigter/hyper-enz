import argparse
import torch
import os, sys

from baselines.data.json_brenda import load_brenda_json
from baselines.entity_graph import build_entity_graph_from_train
from baselines.subgraph_set_drnl import bfs_k_hop_from_set, drnl_set_labels
from baselines.buddy.model_mlp import BuddyMLP
from torch_geometric.data import Data as GeoData
from torch_geometric.loader import DataLoader as GeoDataLoader

# Ensure project root on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--init', type=str, required=True, help='Path to checkpoint dir containing checkpoint file')
    p.add_argument('--eval_batch_size', type=int, default=128)
    p.add_argument('--hidden_dim', type=int, default=256)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--k_hop', type=int, default=2)
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--split', type=str, default='test', choices=['valid', 'test'])
    return p.parse_args()


def evaluate_lp(model, loader, device):
    model.eval()
    logs = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data.x.long(), data.batch, getattr(data, 'g_feat', None))
            score = logits[0]
            gt = int(data.y[0].item())
            if hasattr(data, 'filter_rel') and data.filter_rel is not None:
                score[data.filter_rel] = float('-inf')
                score[gt] = logits[0, gt]
            argsort = torch.argsort(score, dim=0, descending=True)
            where = (argsort == gt).nonzero(as_tuple=False)
            if where.size(0) != 1:
                continue
            rank = 1 + where.item()
            logs.append({'MRR': 1.0/rank, 'MR': float(rank), 'HITS@1': 1.0 if rank<=1 else 0.0, 'HITS@3': 1.0 if rank<=3 else 0.0, 'HITS@10': 1.0 if rank<=10 else 0.0})
    if not logs:
        return {'MRR': 0.0, 'MR': 0.0, 'HITS@1': 0.0, 'HITS@3': 0.0, 'HITS@10': 0.0}
    metrics = {k: sum(d[k] for d in logs) / len(logs) for k in logs[0].keys()}
    return metrics


def main():
    args = parse_args()
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    data_root = os.path.join(os.getcwd(), 'brenda_07')
    bundle = load_brenda_json(data_root)
    train = bundle['train']
    triples = bundle['valid'] if args.split == 'valid' else bundle['test']
    num_entities = len(bundle['entity2id'])
    num_rel = len(bundle['rel2id'])
    edge_index, adj = build_entity_graph_from_train(train, num_entities)
    # Build true map for filtering
    def normalize_pair(H, T):
        return (tuple(sorted(H)), tuple(sorted(T)))
    true_map = {}
    for H, T, y in (bundle['train'] + bundle['valid'] + bundle['test']):
        k = normalize_pair(H, T)
        true_map.setdefault(k, set()).add(y)

    # Build dataset from JSON
    def build_dataset(triples, k_hop, max_nodes, max_label=100, max_deg_bucket=10):
        graphs = []
        for H, T, y in triples:
            sources = list(set(H) | set(T))
            nodes = bfs_k_hop_from_set(adj, sources, k_hop)
            if len(nodes) > max_nodes:
                nodes = nodes[:max_nodes]
            local = {n: i for i, n in enumerate(nodes)}
            rows, cols = [], []
            for u in nodes:
                for v in adj.get(u, []):
                    if v in local:
                        rows.append(local[u]); cols.append(local[v])
            if rows:
                e = torch.tensor([rows, cols], dtype=torch.long)
            else:
                e = torch.zeros((2, 0), dtype=torch.long)
            labels = drnl_set_labels(adj, nodes, H, T)
            deg = torch.zeros(len(nodes), dtype=torch.long)
            for i in range(len(rows)):
                deg[rows[i]] += 1
            deg = torch.clamp(deg, max=max_deg_bucket)
            x = torch.stack([labels, deg], dim=1)
            hist = torch.bincount(torch.clamp(labels, max=max_label), minlength=max_label+2).float()
            hist = hist / max(1.0, hist.sum())
            hist = hist.view(1, -1)
            batch = torch.zeros(len(nodes), dtype=torch.long)
            fr = sorted([r for r in true_map.get(normalize_pair(H, T), set()) if r != y])
            filter_rel = torch.tensor(fr, dtype=torch.long) if fr else None
            graphs.append(GeoData(x=x, edge_index=e, y=torch.tensor(y, dtype=torch.long), batch=batch, g_feat=hist, filter_rel=filter_rel))
        return graphs
    graphs = build_dataset(triples, args.k_hop, 3000)
    loader = GeoDataLoader(graphs, batch_size=1, shuffle=False)
    model = BuddyMLP(args.hidden_dim, num_rel, max_label=100, dropout=args.dropout).to(device)

    ckpt_file = os.path.join(args.init, 'checkpoint') if os.path.isdir(args.init) else args.init
    state = torch.load(ckpt_file, map_location=device)
    model.load_state_dict(state['model_state_dict'])

    metrics = evaluate_lp(model, loader, device)
    print(metrics)


if __name__ == '__main__':
    main()
