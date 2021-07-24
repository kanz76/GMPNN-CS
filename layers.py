
import torch
from torch import nn
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import softmax, degree
from torch_scatter import scatter


class CoAttentionLayerDrugBank(nn.Module):

    def __init__(self, n_feats):
        super().__init__()
        self.out_feats = n_feats
        self.w_j = nn.Parameter(torch.zeros(n_feats, self.out_feats))
        self.w_i = nn.Parameter(torch.zeros(n_feats, self.out_feats))
        self.bias = nn.Parameter(torch.zeros(self.out_feats))
        self.a = nn.Parameter(torch.zeros(1, n_feats))

        self.mlp = nn.Sequential(
            nn.PReLU(),
            nn.Linear(self.out_feats, 1),
        )

        glorot(self.w_j)
        glorot(self.w_i)
        glorot(self.a)
    
    
    def forward(self, x_j, x_i, pair_data):
        x_i = x_i @ self.w_i
        x_j = x_j @ self.w_j
        x_j_i  = x_j[pair_data.edge_index[0]] + x_i[pair_data.edge_index[1]] + self.bias
        x_i = x_j = None

        alpha = self.mlp(x_j_i).view(-1)
        attentions = softmax(alpha, pair_data.edge_index_batch)
        
        return attentions


class CoAttentionLayerTwosides(nn.Module):

    def __init__(self, n_feats):
        super().__init__()
        self.n_feats = n_feats
        self.bias = nn.Parameter(torch.zeros(n_feats))

        self.proj = nn.Sequential(
            nn.PReLU(),
            nn.Linear(self.n_feats, self.n_feats),
            nn.PReLU(),
            nn.Linear(self.n_feats, self.n_feats)
        )

        self.mlp = nn.Sequential(
            nn.PReLU(),
            nn.Linear(self.n_feats, self.n_feats),
            nn.PReLU(),
            nn.Linear(self.n_feats, self.n_feats),
        )

        self.w_i = nn.Parameter(torch.zeros(n_feats, n_feats))
        self.w_j = nn.Parameter(torch.zeros(n_feats, n_feats))
        self.a = nn.Parameter(torch.zeros(n_feats))
        glorot(self.w_i)
        glorot(self.w_j)
        glorot(self.a.view(1, -1))
    
    def forward(self, x_j, x_i, pair_data, rels):
        x_i = x_i @ self.w_i
        x_j = x_j @ self.w_j
        rels = self.proj(rels)
        rels = torch.repeat_interleave(rels, degree(pair_data.edge_index_batch, dtype=pair_data.edge_index_batch.dtype), dim=0)

        # alpha = (rels * self.mlp(x_j[pair_data.edge_index[0]] + x_i[pair_data.edge_index[1]])).sum(-1)
        alpha = (self.a * self.mlp(x_j[pair_data.edge_index[0]] + x_i[pair_data.edge_index[1]] + rels)).sum(-1)
        attentions = softmax(alpha, pair_data.edge_index_batch)

        return attentions
