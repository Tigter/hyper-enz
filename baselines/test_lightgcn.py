import argparse
import torch
import os, sys
from tqdm import tqdm

# Ensure project root on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from util.brenda_datasets import load_data
from baselines.models.lightgcn import LightGCLClassifier


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--init', type=str, required=True)
    p.add_argument('--embed_dim', type=int, default=256)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--split', type=str, default='test', choices=['valid', 'test'])
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    graph_info, train_info = load_data()
    num_rel = graph_info['e_num']
    c_num = graph_info['c_num']

    triples = train_info['valid_triple'] if args.split == 'valid' else train_info['test_triple']
    # Build dataset-like lists (TestRelationDataset used in main pipeline uses sampler; here we only need ranking over relations)

    # Load mappings and graph size from checkpoint
    ckpt_file = os.path.join(args.init, 'checkpoint') if os.path.isdir(args.init) else args.init
    state = torch.load(ckpt_file, map_location=device)
    inv_nodes = state.get('inv_nodes', None)
    if inv_nodes is None:
        raise RuntimeError('Checkpoint missing node mapping; please retrain LightGCN with the updated script.')
    num_nodes = len(inv_nodes)
    model = LightGCLClassifier(num_nodes, args.embed_dim, args.layers, num_rel, dropout=args.dropout).to(device)
    model.load_state_dict(state['model_state_dict'])
    edge_index = state['edge_index'].to(device)
    node_map = state['node_map']

    # Evaluate by ranking the ground-truth relation among all relations using logits
    model.eval()
    logs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(triples), 1024), desc='Evaluating'):
            batch = triples[i:i+1024]
            hb = torch.tensor([node_map[x[0]] for x in batch], dtype=torch.long, device=device)
            tb = torch.tensor([node_map[x[2]] for x in batch], dtype=torch.long, device=device)
            yb = torch.tensor([x[1] - c_num for x in batch], dtype=torch.long, device=device)
            logits = model(edge_index, hb, tb)
            for j in range(yb.size(0)):
                score = logits[j]
                argsort = torch.argsort(score, dim=0, descending=True)
                gt = int(yb[j].item())
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
    if logs:
        metrics = {k: sum(d[k] for d in logs) / len(logs) for k in logs[0].keys()}
    else:
        metrics = {'MRR': 0.0, 'MR': 0.0, 'HITS@1': 0.0, 'HITS@3': 0.0, 'HITS@10': 0.0}
    print(metrics)


if __name__ == '__main__':
    main()
