import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool


class BuddyMLP(nn.Module):
    def __init__(self, hidden_dim: int, num_relations: int, max_label: int = 100, max_deg_bucket: int = 10, dropout: float = 0.3):
        super().__init__()
        self.label_emb = nn.Embedding(max_label + 2, hidden_dim)
        self.deg_emb = nn.Embedding(max_deg_bucket + 1, hidden_dim)
        nn.init.xavier_uniform_(self.label_emb.weight)
        nn.init.xavier_uniform_(self.deg_emb.weight)

        self.gproj = nn.Linear(max_label + 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_relations),
        )

    def forward(self, x, batch, g_feat=None):
        # x: LongTensor (N,1) or (N,2): [label,(opt)deg_bucket]
        if x.size(1) == 1:
            labels = x.view(-1)
            degs = torch.zeros_like(labels)
        else:
            labels = x[:, 0].view(-1)
            degs = x[:, 1].view(-1)
        labels = labels.clamp(min=0, max=self.label_emb.num_embeddings - 1)
        degs = degs.clamp(min=0, max=self.deg_emb.num_embeddings - 1)

        h = self.label_emb(labels) + self.deg_emb(degs)
        hg = global_add_pool(h, batch)
        if g_feat is None:
            gp = torch.zeros_like(hg)
        else:
            gp = self.gproj(g_feat)
        cat = torch.cat([hg, gp], dim=-1)
        cat = self.dropout(cat)
        logits = self.mlp(cat)
        return logits
