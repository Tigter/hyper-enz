import argparse
import os
import sys
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from torch_sparse import SparseTensor

# Data loaders reused from baseline
from baselines.data.json_brenda import load_brenda_json
from baselines.entity_graph import build_entity_graph_from_train


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--save_path', type=str, required=True)
    p.add_argument('--embed_dim', type=int, default=256)
    p.add_argument('--layers', type=int, default=15, help='HL-GNN K (num propagation steps)')
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--alpha', type=float, default=0.5, help='HL-GNN alpha')
    p.add_argument('--init', type=str, default='KI', choices=['SGC', 'RWR', 'KI', 'Random'])
    p.add_argument('--mlp_hidden', type=int, default=256)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--agg', type=str, default='mean', choices=['mean', 'attn'])
    p.add_argument('--cuda', action='store_true')
    return p.parse_args()


def make_batches(lst, bs):
    for i in range(0, len(lst), bs):
        yield lst[i:i + bs]


def pair_key(H: List[int], T: List[int]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    return (tuple(sorted(H)), tuple(sorted(T)))


def build_truth_map(train, valid, test) -> Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], set]:
    mp: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], set] = {}
    for split in (train, valid, test):
        for H, T, y in split:
            k = pair_key(H, T)
            if k not in mp:
                mp[k] = set()
            mp[k].add(y)
    return mp


class SetPool(nn.Module):
    def __init__(self, dim: int, agg: str = 'mean'):
        super().__init__()
        self.agg = agg
        if agg == 'attn':
            self.attn = nn.Linear(dim, 1)

    def forward(self, z: torch.Tensor, idx_list: List[List[int]]):
        # z: (N, D)
        outs = []
        for ids in idx_list:
            if len(ids) == 0:
                outs.append(torch.zeros(z.size(1), device=z.device))
                continue
            sub = z[ids]
            if self.agg == 'attn':
                w = self.attn(sub).squeeze(-1)
                w = torch.softmax(w, dim=-1)
                outs.append((sub * w.unsqueeze(-1)).sum(dim=0))
            else:
                outs.append(sub.mean(dim=0))
        return torch.stack(outs, dim=0)


class HLRelModel(nn.Module):
    def __init__(self, num_nodes: int, num_rel: int, embed_dim: int, K: int, dropout: float, alpha: float, init: str,
                 mlp_hidden: int, agg: str):
        super().__init__()
        self.embed = nn.Embedding(num_nodes, embed_dim)
        nn.init.xavier_uniform_(self.embed.weight)

        # Prepare to import HLGNN from HL-GNN/OGB
        sys.path.append(os.path.join(os.getcwd(), 'HL-GNN', 'OGB'))
        from layer import HLGNN  # noqa: E402

        self.encoder = HLGNN(in_channels=embed_dim, hidden_channels=embed_dim, out_channels=embed_dim,
                             K=K, dropout=dropout, alpha=alpha, init=init)
        self.pool = SetPool(embed_dim, agg=agg)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 3, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_rel)
        )

    def node_reps(self, adj_t: SparseTensor) -> torch.Tensor:
        x = self.embed.weight
        h = self.encoder(x, adj_t, edge_weight=None)
        return h

    def forward_logits(self, adj_t: SparseTensor, H_list: List[List[int]], T_list: List[List[int]]):
        z = self.node_reps(adj_t)
        h = self.pool(z, H_list)
        t = self.pool(z, T_list)
        ht = h * t
        feats = torch.cat([h, t, ht], dim=-1)
        logits = self.classifier(feats)
        return logits


def evaluate_filtered(model: HLRelModel, adj_t: SparseTensor, batch_data, truth_map: Dict, device: torch.device):
    model.eval()
    logs = []
    with torch.no_grad():
        for i in range(0, len(batch_data), 512):
            batch = batch_data[i:i + 512]
            Hb = [h for (h, t, y) in batch]
            Tb = [t for (h, t, y) in batch]
            yb = [y for (h, t, y) in batch]
            logits = model.forward_logits(adj_t, Hb, Tb)
            for j in range(len(batch)):
                score = logits[j].clone()
                gt = int(yb[j])
                k = pair_key(Hb[j], Tb[j])
                if k in truth_map:
                    for y_true in truth_map[k]:
                        if y_true != gt:
                            score[y_true] = float('-inf')
                argsort = torch.argsort(score, dim=0, descending=True)
                where = (argsort == gt).nonzero(as_tuple=False)
                if where.size(0) != 1:
                    continue
                rank = 1 + int(where.item())
                logs.append({
                    'MRR': 1.0 / rank,
                    'MR': float(rank),
                    'HITS@1': 1.0 if rank <= 1 else 0.0,
                    'HITS@3': 1.0 if rank <= 3 else 0.0,
                    'HITS@10': 1.0 if rank <= 10 else 0.0,
                })
    if not logs:
        return {'MRR': 0.0, 'MR': 0.0, 'HITS@1': 0.0, 'HITS@3': 0.0, 'HITS@10': 0.0}
    return {k: sum(d[k] for d in logs) / len(logs) for k in logs[0].keys()}


def main():
    args = parse_args()
    os.makedirs(os.path.join('models', args.save_path, 'log'), exist_ok=True)
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    # Load dataset (same as baseline)
    data_root = os.path.join(os.getcwd(), 'brenda_07')
    bundle = load_brenda_json(data_root)
    train, valid, test = bundle['train'], bundle['valid'], bundle['test']
    num_entities = len(bundle['entity2id'])
    num_rel = len(bundle['rel2id'])

    # Build entity graph from train only
    edge_index, _ = build_entity_graph_from_train(train, num_entities)
    edge_index = edge_index.to(device)
    row, col = edge_index[0], edge_index[1]
    val = torch.ones(row.size(0), dtype=torch.float32, device=device)
    adj_t = SparseTensor(row=row, col=col, value=val, sparse_sizes=(num_entities, num_entities)).t()

    model = HLRelModel(num_entities, num_rel, args.embed_dim, args.layers, args.dropout,
                       args.alpha, args.init, args.mlp_hidden, args.agg).to(device)

    opt = Adam(model.parameters(), lr=args.lr)
    ce_loss = nn.CrossEntropyLoss()

    truth_map = build_truth_map(train, valid, test)

    best_mrr = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        ce_total = 0.0
        pbar = tqdm(list(make_batches(train, args.batch_size)), desc=f'Epoch {epoch}')
        for batch in pbar:
            Hb = [h for (h, t, y) in batch]
            Tb = [t for (h, t, y) in batch]
            yb = torch.tensor([y for (h, t, y) in batch], dtype=torch.long, device=device)
            logits = model.forward_logits(adj_t, Hb, Tb)
            loss = ce_loss(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ce_total += loss.item() * len(batch)
            pbar.set_postfix({'ce': f'{loss.item():.4f}'})

        # Evaluate on valid (filtered)
        valid_metrics = evaluate_filtered(model, adj_t, valid, truth_map, device)
        print(f"Epoch {epoch} | Valid metrics: {valid_metrics}")

        # Save checkpoint if best
        if valid_metrics['MRR'] > best_mrr:
            best_mrr = valid_metrics['MRR']
            torch.save({'model_state_dict': model.state_dict()}, os.path.join('models', args.save_path, 'checkpoint'))

    # Final test with best parameters loaded (if any)
    ckpt_path = os.path.join('models', args.save_path, 'checkpoint')
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state['model_state_dict'])
    test_metrics = evaluate_filtered(model, adj_t, test, truth_map, device)
    print(f"Test metrics: {test_metrics}")


if __name__ == '__main__':
    main()
