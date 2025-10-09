import json
from typing import List, Tuple, Dict
import os


Triple = Tuple[List[str], List[str], str]


def load_json_split(path: str) -> List[Triple]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def build_mappings(train: List[Triple], valid: List[Triple], test: List[Triple]):
    entities = set()
    rels = set()
    for split in (train, valid, test):
        for H, T, r in split:
            for s in H:
                entities.add(s)
            for p in T:
                entities.add(p)
            rels.add(r)
    ent_list = sorted(list(entities))
    rel_list = sorted(list(rels))
    entity2id = {e: i for i, e in enumerate(ent_list)}
    rel2id = {r: i for i, r in enumerate(rel_list)}
    return entity2id, rel2id


def map_splits(train: List[Triple], valid: List[Triple], test: List[Triple], entity2id: Dict[str, int], rel2id: Dict[str, int]):
    def map_one(split: List[Triple]):
        out = []
        for H, T, r in split:
            h_ids = [entity2id[x] for x in H]
            t_ids = [entity2id[x] for x in T]
            y = rel2id[r]
            out.append((h_ids, t_ids, y))
        return out
    return map_one(train), map_one(valid), map_one(test)


def load_brenda_json(root: str):
    train = load_json_split(os.path.join(root, 'train.json'))
    valid = load_json_split(os.path.join(root, 'valid.json'))
    test = load_json_split(os.path.join(root, 'test.json'))
    entity2id, rel2id = build_mappings(train, valid, test)
    mtrain, mvalid, mtest = map_splits(train, valid, test, entity2id, rel2id)
    return {
        'train': mtrain,
        'valid': mvalid,
        'test': mtest,
        'entity2id': entity2id,
        'rel2id': rel2id,
    }

