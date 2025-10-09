import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import os, sys

# Ensure project root on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from baselines.data.json_brenda import load_brenda_json
from baselines.entity_graph import build_entity_graph_from_train
from baselines.subgraph_set_drnl import bfs_k_hop_from_set, drnl_set_labels
from baselines.buddy.model_mlp import BuddyMLP
from torch_geometric.data import Data as GeoData
from torch_geometric.loader import DataLoader as GeoDataLoader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--save_path', type=str, required=True)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--eval_batch_size', type=int, default=128)
    p.add_argument('--hidden_dim', type=int, default=256)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--k_hop', type=int, default=2)
    p.add_argument('--cuda', action='store_true')
    return p.parse_args()


def evaluate_filtered(model, loader, device):
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
    return {k: sum(d[k] for d in logs)/len(logs) for k in logs[0].keys()}


def main():
    args = parse_args()
    os.makedirs(os.path.join('models', args.save_path, 'log'), exist_ok=True)
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    data_root = os.path.join(os.getcwd(), 'brenda_07')
    bundle = load_brenda_json(data_root)
    train, valid = bundle['train'], bundle['valid']
    num_entities = len(bundle['entity2id'])
    num_rel = len(bundle['rel2id'])
    edge_index, adj = build_entity_graph_from_train(train, num_entities)

    # Build PyG datasets on-the-fly from JSON
    def build_dataset(triples, k_hop, max_nodes, max_label=100, max_deg_bucket=10):
        graphs = []
        for H, T, y in triples:
            sources = list(set(H) | set(T))
            nodes = bfs_k_hop_from_set(adj, sources, k_hop)
            if len(nodes) > max_nodes:
                nodes = nodes[:max_nodes]
            local = {n: i for i, n in enumerate(nodes)}
            # induced edges
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
            # degree buckets in subgraph
            deg = torch.zeros(len(nodes), dtype=torch.long)
            for i in range(len(rows)):
                deg[rows[i]] += 1
            deg = torch.clamp(deg, max=max_deg_bucket)
            x = torch.stack([labels, deg], dim=1)
            # DRNL histogram as graph-level features
            hist = torch.bincount(torch.clamp(labels, max=max_label), minlength=max_label+2).float()
            hist = hist / max(1.0, hist.sum())
            hist = hist.view(1, -1)
            batch = torch.zeros(len(nodes), dtype=torch.long)
            graphs.append(GeoData(x=x, edge_index=e, y=torch.tensor(y, dtype=torch.long), batch=batch, g_feat=hist))
        return graphs

    def normalize_pair(H, T):
        return (tuple(sorted(H)), tuple(sorted(T)))
    true_map = {}
    for H, T, y in (bundle['train'] + bundle['valid'] + bundle['test']):
        k = normalize_pair(H, T)
        true_map.setdefault(k, set()).add(y)

    def build_dataset_with_filter(triples):
        graphs = build_dataset(triples, args.k_hop, 3000)
        out = []
        for (H, T, y), g in zip(triples, graphs):
            fr = sorted([r for r in true_map.get(normalize_pair(H,T), set()) if r != y])
            g.filter_rel = torch.tensor(fr, dtype=torch.long) if fr else None
            out.append(g)
        return out

    train_graphs = build_dataset_with_filter(train)
    valid_graphs = build_dataset_with_filter(valid)
    train_loader = GeoDataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    valid_loader = GeoDataLoader(valid_graphs, batch_size=1, shuffle=False)
    model = BuddyMLP(args.hidden_dim, num_rel, max_label=100, dropout=args.dropout).to(device)
    opt = Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_mrr = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for data in pbar:
            data = data.to(device)
            logits = model(data.x.long(), data.batch, getattr(data, 'g_feat', None))
            loss = loss_fn(logits, data.y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        metrics = evaluate_filtered(model, valid_loader, device)
        print(f"Epoch {epoch} | Valid metrics: {metrics}")
        if metrics['MRR'] > best_mrr:
            best_mrr = metrics['MRR']
            torch.save({'model_state_dict': model.state_dict()}, os.path.join('models', args.save_path, 'checkpoint'))

    print(f'Best valid MRR: {best_mrr:.4f}')


if __name__ == '__main__':
    main()
