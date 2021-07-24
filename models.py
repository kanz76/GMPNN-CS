import torch
from torch import nn
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import degree
from torch_scatter import scatter
from layers import (CoAttentionLayerDrugBank,
                    CoAttentionLayerTwosides,
                    )


class GmpnnCSNetDrugBank(nn.Module):
    def __init__(self, in_feats, edge_feats, hid_feats, rel_total, n_iter, dropout=0):


        super().__init__()
        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.rel_total = rel_total
        self.n_iter = n_iter
        self.dropout = dropout
        self.snd_hid_feats = hid_feats * 2

        self.mlp = nn.Sequential(
            nn.Linear(in_feats, hid_feats),
            CustomDropout(self.dropout),
            nn.PReLU(),
            nn.Linear(hid_feats, hid_feats),
            nn.BatchNorm1d(hid_feats),
            CustomDropout(self.dropout),
            nn.PReLU(),
            nn.Linear(hid_feats, hid_feats),
            nn.BatchNorm1d(hid_feats), 
            CustomDropout(self.dropout),
        )

        self.propagation_layer = GmpnnBlock(edge_feats, self.hid_feats, self.n_iter, dropout) 

        self.i_pro = nn.Parameter(torch.zeros(self.snd_hid_feats , self.hid_feats))
        self.j_pro = nn.Parameter(torch.zeros(self.snd_hid_feats, self.hid_feats))
        self.bias = nn.Parameter(torch.zeros(self.hid_feats ))
        self.co_attention_layer = CoAttentionLayerDrugBank(self.snd_hid_feats)
        self.rel_embs = nn.Embedding(self.rel_total, self.hid_feats)


        glorot(self.i_pro)
        glorot(self.j_pro)


        self.lin1 = nn.Sequential(
            nn.BatchNorm1d(self.snd_hid_feats),
        )
        self.lin2 = nn.Sequential(
            nn.BatchNorm1d(self.snd_hid_feats),
            CustomDropout(self.dropout),
            nn.PReLU(),
            nn.Linear(self.snd_hid_feats, self.snd_hid_feats)
        )

    def forward(self, batch):
        drug_data, unique_drug_pair, rels, drug_pair_indices, node_j_for_pairs, node_i_for_pairs = batch
        drug_data.x = self.mlp(drug_data.x)

        new_feats = self.propagation_layer(drug_data)
        drug_data.x = new_feats
        x_j = drug_data.x[node_j_for_pairs]
        x_i = drug_data.x[node_i_for_pairs]
        attentions = self.co_attention_layer(x_j, x_i, unique_drug_pair)
        pair_repr = attentions.unsqueeze(-1) * ((x_i[unique_drug_pair.edge_index[1]] @ self.i_pro) * (x_j[unique_drug_pair.edge_index[0]] @ self.j_pro))
        
        x_i = x_j = None ## Just to free up some memory space
        drug_data = new_feats = None
        node_i_for_pairs = node_j_for_pairs = None 
        attentions = None

        pair_repr = scatter(pair_repr, unique_drug_pair.edge_index_batch, reduce='add', dim=0)[drug_pair_indices]
        p_scores, n_scores = self.compute_score(pair_repr, rels)
        return p_scores, n_scores
    
    def compute_score(self, pair_repr, rels):
        batch_size = len(rels)
        neg_n = (len(pair_repr) - batch_size) // batch_size  # I case of multiple negative samples per positive sample.
        rels = torch.cat([rels, torch.repeat_interleave(rels, neg_n, dim=0)], dim=0)
        rels = self.rel_embs(rels)
        scores = (pair_repr * rels).sum(-1)
        p_scores, n_scores = scores[:batch_size].unsqueeze(-1), scores[batch_size:].view(batch_size, -1, 1)

        return p_scores, n_scores


class GmpnnCSNetTwosides(GmpnnCSNetDrugBank):
    def __init__(self, in_feats, edge_feats, hid_feats, rel_total, n_iter, dropout=0):
        super().__init__(in_feats, edge_feats, hid_feats, rel_total, n_iter, dropout)

        self.co_attention_layer = CoAttentionLayerTwosides(self.hid_feats * 2)
        self.rel_embs = nn.Embedding(self.rel_total, self.hid_feats * 2)
        self.rel_proj = nn.Sequential(
            nn.PReLU(),
            nn.Linear(self.hid_feats * 2, self.hid_feats * 2),
            nn.PReLU(),
            nn.Linear(self.hid_feats * 2, self.hid_feats),
        )
        self.s_pro = self.i_pro
        self.j_pro = self.i_pro =  None 

    def forward(self, batch):
        drug_data, drug_pairs, rels, batch_size, node_j_for_pairs, node_i_for_pairs = batch

        drug_data.x = self.mlp(drug_data.x)

        new_feats = self.propagation_layer(drug_data)
        drug_data.x = new_feats

        x_j = drug_data.x[node_j_for_pairs]
        x_i = drug_data.x[node_i_for_pairs]
        rels = self.rel_embs(rels)
        attentions = self.co_attention_layer(x_j, x_i, drug_pairs, rels)
        
        pair_repr = attentions.unsqueeze(-1) * ((x_i[drug_pairs.edge_index[1]] @ self.s_pro) * (x_j[drug_pairs.edge_index[0]] @ self.s_pro))
        drug_data = new_feats = None
        node_i_for_pairs = node_j_for_pairs = None 
        attentions = None
        x_i = x_j = None
        pair_repr = scatter(pair_repr, drug_pairs.edge_index_batch, reduce='add', dim=0)
        
        p_scores, n_scores = self.compute_score(pair_repr,  batch_size, rels)
        return p_scores, n_scores

    def compute_score(self, pair_repr, batch_size, rels):
        rels = self.rel_proj(rels)
        scores = (pair_repr * rels).sum(-1)
        p_scores, n_scores = scores[:batch_size].unsqueeze(-1), scores[batch_size:].view(batch_size, -1, 1)
        return p_scores, n_scores


class GmpnnBlock(nn.Module):
    def __init__(self, edge_feats, n_feats, n_iter, dropout):
        super().__init__()
        self.n_feats = n_feats
        self.n_iter = n_iter
        self.dropout = dropout
        self.snd_n_feats = n_feats * 2

        self.w_i = nn.Parameter(torch.Tensor(self.n_feats, self.n_feats))
        self.w_j = nn.Parameter(torch.Tensor(self.n_feats, self.n_feats))
        self.a = nn.Parameter(torch.Tensor(1, self.n_feats))
        self.bias = nn.Parameter(torch.zeros(self.n_feats))

        self.edge_emb = nn.Sequential(
            nn.Linear(edge_feats, self.n_feats)
        )

        self.lin1 = nn.Sequential(
            nn.BatchNorm1d(n_feats),
            nn.Linear(n_feats, self.snd_n_feats),
        )

        self.lin2 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            CustomDropout(self.dropout),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats),
        )

        self.lin3 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            CustomDropout(self.dropout),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats),
        )

        self.lin4 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            CustomDropout(self.dropout),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats)
        )

        glorot(self.w_i)
        glorot(self.w_j)
        glorot(self.a)

        self.sml_mlp = nn.Sequential(
            nn.PReLU(),
            nn.Linear(self.n_feats, self.n_feats)
        )
    
    def forward(self, data):

        edge_index = data.edge_index
        edge_feats = data.edge_feats
        edge_feats = self.edge_emb(edge_feats)

        deg = degree(edge_index[1], data.x.size(0), dtype=data.x.dtype)

        assert len(edge_index[0]) == len(edge_feats)
        alpha_i = (data.x @ self.w_i)
        alpha_j = (data.x @ self.w_j)
        alpha = alpha_i[edge_index[1]] + alpha_j[edge_index[0]] + self.bias
        alpha = self.sml_mlp(alpha)

        assert alpha.shape == edge_feats.shape
        alpha = (alpha* edge_feats).sum(-1)

        alpha = alpha / (deg[edge_index[0]])
        edge_weights = torch.sigmoid(alpha)

        assert len(edge_weights) == len(edge_index[0])
        edge_attr = data.x[edge_index[0]] * edge_weights.unsqueeze(-1)
        assert len(alpha) == len(edge_attr)
        
        out = edge_attr
        for _ in range(self.n_iter):
            out = scatter(out[data.line_graph_edge_index[0]] , data.line_graph_edge_index[1], dim_size=edge_attr.size(0), dim=0, reduce='add')
            out = edge_attr + (out * edge_weights.unsqueeze(-1))

        x = data.x + scatter(out , edge_index[1], dim_size=data.x.size(0), dim=0, reduce='add')
        x = self.mlp(x)

        return x

    def mlp(self, x): 
        x = self.lin1(x)
        x = (self.lin3(self.lin2(x)) + x) / 2
        x = (self.lin4(x) + x) / 2

        return x


class CustomDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.dropout = (lambda x: x ) if p == 0 else nn.Dropout(p)
    
    def forward(self, input):
        return self.dropout(input)
