import argparse
import torch
import os
from tqdm import tqdm

from baselines.data.json_brenda import load_brenda_json
from baselines.entity_graph import build_entity_graph_from_train
from baselines.models.lightgcl import LightGCLModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--init', type=str, required=True)
    p.add_argument('--embed_dim', type=int, default=256)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--proj_dim', type=int, default=128)
    p.add_argument('--drop_prob', type=float, default=0.2)
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--agg', type=str, default='mean', choices=['mean', 'attn'])
    p.add_argument('--split', type=str, default='test', choices=['valid', 'test'])
    return p.parse_args()


def normalize_pair(H, T):
    return (tuple(sorted(H)), tuple(sorted(T)))


def main():
    args = parse_args()
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    data_root = os.path.join(os.getcwd(), 'brenda_07')
    bundle = load_brenda_json(data_root)
    train = bundle['train']
    triples = bundle['valid'] if args.split == 'valid' else bundle['test']
    num_entities = len(bundle['entity2id'])
    num_rel = len(bundle['rel2id'])

    edge_index, _ = build_entity_graph_from_train(train, num_entities)
    edge_index = edge_index.to(device)

    model = LightGCLModel(num_entities, num_rel, args.embed_dim, args.layers, args.proj_dim, args.drop_prob, agg=args.agg).to(device)
    ckpt_file = os.path.join(args.init, 'checkpoint') if os.path.isdir(args.init) else args.init
    state = torch.load(ckpt_file, map_location=device)
    model.load_state_dict(state['model_state_dict'])

    # true map for filtered ranking
    true_map = {}
    for H, T, y in (bundle['train'] + bundle['valid'] + bundle['test']):
        key = normalize_pair(H, T)
        true_map.setdefault(key, set()).add(y)

    model.eval()
    logs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(triples), 512), desc='Evaluating'):
            batch = triples[i:i+512]
            Hb = [h for (h, t, y) in batch]
            Tb = [t for (h, t, y) in batch]
            yb = [y for (h, t, y) in batch]
            logits = model.forward_logits(edge_index, Hb, Tb)
            for j in range(len(batch)):
                score = logits[j].clone()
                gt = yb[j]
                key = normalize_pair(Hb[j], Tb[j])
                # filter other true rels for same (H,T)
                for r in true_map.get(key, set()):
                    if r != gt:
                        score[r] = float('-inf')
                argsort = torch.argsort(score, dim=0, descending=True)
                where = (argsort == gt).nonzero(as_tuple=False)
                if where.size(0) != 1:
                    continue
                rank = 1 + where.item()
                logs.append({'MRR': 1.0/rank, 'MR': float(rank), 'HITS@1': 1.0 if rank<=1 else 0.0, 'HITS@3': 1.0 if rank<=3 else 0.0, 'HITS@10': 1.0 if rank<=10 else 0.0})
    if logs:
        metrics = {k: sum(d[k] for d in logs)/len(logs) for k in logs[0].keys()}
    else:
        metrics = {'MRR': 0.0, 'MR': 0.0, 'HITS@1': 0.0, 'HITS@3': 0.0, 'HITS@10': 0.0}
    print(metrics)


if __name__ == '__main__':
    main()
