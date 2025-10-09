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

from util.brenda_datasets import load_data
from baselines.graph_build import build_pair_graph, build_pair_graph_with_nodes
from baselines.models.lightgcn import LightGCLClassifier


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--save_path', type=str, required=True)
    p.add_argument('--embed_dim', type=int, default=256)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=1024)
    p.add_argument('--cuda', action='store_true')
    return p.parse_args()


def build_edge_index_from_triples(triples):
    node_map, inv_nodes, edge_index, adj = build_pair_graph(triples)
    return node_map, inv_nodes, edge_index


def make_batches(triples, batch_size):
    for i in range(0, len(triples), batch_size):
        yield triples[i:i+batch_size]


def main():
    args = parse_args()
    os.makedirs(os.path.join('models', args.save_path, 'log'), exist_ok=True)
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    graph_info, train_info = load_data()
    num_rel = graph_info['e_num']
    c_num = graph_info['c_num']
    train_triples = [(int(h), int(r), int(t)) for (h, r, t) in train_info['train_triple']]
    valid_triples = [(int(h), int(r), int(t)) for (h, r, t) in train_info['valid_triple']]

    # Build mapping from all nodes across splits to avoid missing ids; edges from train only
    all_triples = train_triples + valid_triples + [(int(h), int(r), int(t)) for (h, r, t) in train_info['test_triple']]
    node_map, inv_nodes, edge_index, _ = build_pair_graph_with_nodes(train_triples, all_triples)
    edge_index = edge_index.to(device)

    num_nodes = len(inv_nodes)
    model = LightGCLClassifier(num_nodes, args.embed_dim, args.layers, num_rel, dropout=args.dropout).to(device)
    opt = Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(list(make_batches(train_triples, args.batch_size)), desc=f'Epoch {epoch}'):
            hb = torch.tensor([node_map[x[0]] for x in batch], dtype=torch.long, device=device)
            tb = torch.tensor([node_map[x[2]] for x in batch], dtype=torch.long, device=device)
            yb = torch.tensor([x[1] - c_num for x in batch], dtype=torch.long, device=device)
            logits = model(edge_index, hb, tb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(batch)
        # simple val acc
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in make_batches(valid_triples, 1024):
                hb = torch.tensor([node_map[x[0]] for x in batch], dtype=torch.long, device=device)
                tb = torch.tensor([node_map[x[2]] for x in batch], dtype=torch.long, device=device)
                yb = torch.tensor([x[1] - c_num for x in batch], dtype=torch.long, device=device)
                logits = model(edge_index, hb, tb)
                pred = logits.argmax(dim=-1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
        val_acc = correct / max(1, total)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({'model_state_dict': model.state_dict(), 'edge_index': edge_index.detach().cpu(), 'node_map': node_map, 'inv_nodes': inv_nodes}, os.path.join('models', args.save_path, 'checkpoint'))

    print(f'Best valid acc: {best_val_acc:.4f}')


if __name__ == '__main__':
    main()
