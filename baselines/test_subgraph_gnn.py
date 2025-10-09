import argparse
import torch
import os, sys
from tqdm import tqdm

# Ensure project root on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from baselines.datasets_pairgraph import build_seal_loaders
from baselines.models.subgraph_gin import SubgraphGIN


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--init', type=str, required=True, help='Path to checkpoint dir containing checkpoint file')
    p.add_argument('--eval_batch_size', type=int, default=128)
    p.add_argument('--hidden_dim', type=int, default=256)
    p.add_argument('--num_layers', type=int, default=3)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--k_hop', type=int, default=2)
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--split', type=str, default='test', choices=['valid', 'test'])
    return p.parse_args()


def evaluate_lp(model, loader, device, e_num, all_true_triples, c_num):
    model.eval()
    logs = []
    with torch.no_grad():
        for data in tqdm(loader, desc='Evaluating'):
            data = data.to(device)
            logits = model(data.x, data.edge_index, data.batch)
            bsz = data.y.size(0)
            for i in range(bsz):
                score = logits[i]
                argsort = torch.argsort(score, dim=0, descending=True)
                gt = data.y[i].item()
                ranking = (argsort == gt).nonzero(as_tuple=False)
                if ranking.size(0) != 1:
                    continue
                rank = 1 + ranking.item()
                logs.append({
                    'MRR': 1.0 / rank,
                    'MR': float(rank),
                    'HITS@1': 1.0 if rank <= 1 else 0.0,
                    'HITS@3': 1.0 if rank <= 3 else 0.0,
                    'HITS@10': 1.0 if rank <= 10 else 0.0,
                })
    if not logs:
        return {'MRR': 0.0, 'MR': 0.0, 'HITS@1': 0.0, 'HITS@3': 0.0, 'HITS@10': 0.0}
    metrics = {k: sum(d[k] for d in logs) / len(logs) for k in logs[0].keys()}
    return metrics


def main():
    args = parse_args()
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    config = {
        'eval_batch_size': args.eval_batch_size,
        'k_hop': args.k_hop,
    }
    loader, pg = build_seal_loaders({'batch_size': args.eval_batch_size, 'k_hop': args.k_hop, 'max_nodes': 3000}, split=args.split)

    in_dim = 1
    num_rel = pg.e_num
    model = SubgraphGIN(in_dim, args.hidden_dim, args.num_layers, num_rel, dropout=args.dropout, max_label=100).to(device)

    ckpt_file = os.path.join(args.init, 'checkpoint') if os.path.isdir(args.init) else args.init
    state = torch.load(ckpt_file, map_location=device)
    model.load_state_dict(state['model_state_dict'])

    all_true_triples = set(pg.train_info['train_triple'] + pg.train_info['valid_triple'] + pg.train_info['test_triple'])
    metrics = evaluate_lp(model, loader, device, num_rel, all_true_triples, pg.c_num)
    print(metrics)


if __name__ == '__main__':
    main()
