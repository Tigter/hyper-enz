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

from baselines.datasets_pairgraph import build_seal_loaders
from baselines.models.subgraph_gin import SubgraphGIN


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--save_path', type=str, required=True)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--eval_batch_size', type=int, default=128)
    p.add_argument('--hidden_dim', type=int, default=256)
    p.add_argument('--num_layers', type=int, default=3)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--k_hop', type=int, default=2)
    p.add_argument('--cuda', action='store_true')
    return p.parse_args()


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data.x, data.edge_index, data.batch)
            pred = logits.argmax(dim=-1)
            correct += (pred == data.y).sum().item()
            total += data.y.numel()
    return correct / max(1, total)


def main():
    args = parse_args()
    os.makedirs(os.path.join('models', args.save_path, 'log'), exist_ok=True)
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    config = {
        'batch_size': args.batch_size,
        'k_hop': args.k_hop,
        'max_nodes': 3000,
    }
    train_loader, pg = build_seal_loaders(config, split='train')
    valid_loader, _ = build_seal_loaders({'batch_size': args.eval_batch_size, 'k_hop': args.k_hop, 'max_nodes': 3000}, split='valid')

    in_dim = 1  # DRNL label as input
    num_rel = pg.e_num
    model = SubgraphGIN(in_dim, args.hidden_dim, args.num_layers, num_rel, dropout=args.dropout, max_label=100).to(device)
    opt = Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for data in pbar:
            data = data.to(device)
            logits = model(data.x, data.edge_index, data.batch)
            loss = loss_fn(logits, data.y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        val_acc = evaluate(model, valid_loader, device)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'model_state_dict': model.state_dict()}, os.path.join('models', args.save_path, 'checkpoint'))

    print(f'Best valid acc: {best_acc:.4f}')


if __name__ == '__main__':
    main()
