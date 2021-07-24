import math

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch, Data
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

import pickle

NUM_FEATURES = None
NUM_EDGE_FEATURES = None 
bipartite_edge_dict = dict()
drug_num_node_indices = dict()

def total_num_rel(name):
    if name.lower() == 'drugbank': return 86
    if name.lower() == 'twosides': return 963
    else: raise NotImplementedError


def split_train_valid(data, fold, val_ratio=0.2):
        cv_split = StratifiedShuffleSplit(n_splits=2, test_size=val_ratio, random_state=fold)

        pos_triples, neg_samples = data
        train_index, val_index = next(iter(cv_split.split(X=pos_triples, y=pos_triples[:, 2])))
        train_pos_triples = pos_triples[train_index]
        val_pos_triples = pos_triples[val_index]
        train_neg_samples = neg_samples[train_index]
        test_neg_samples = neg_samples[val_index]

        train_tup = (train_pos_triples, train_neg_samples)
        val_tup = (val_pos_triples, test_neg_samples)

        return train_tup, val_tup

def load_ddi_data_fold(name, fold, batch_size, data_size_ratio, valid_ratio=0.2):

    def load_split(split_name):
        filename = (f'data/preprocessed/{dataset_name}/pair_pos_neg_triples_{split_name}.csv')
        print(f'\nLoading {filename}...')
        df = pd.read_csv(filename)
        pos_triples = [(d1, d2, r) for d1, d2, r in zip(df['Drug1_ID'], df['Drug2_ID'], df['Y'])]
        neg_samples = [[str(e) for e in neg_s.split('_')] for neg_s in df['Neg samples']]

        return np.array(pos_triples), np.array(neg_samples)

    global NUM_FEATURES
    global NUM_EDGE_FEATURES
    global drug_num_node_indices

    dataset_name = name.lower()
    print(f'Loading {dataset_name}...')

    
    if dataset_name in ('twosides', 'drugbank'):
        filename = f'data/preprocessed/{dataset_name}/drug_data.pkl'
        print('\nLoading ')
        with open(filename, 'rb') as f:
            all_drug_data = pickle.load(f)
        NUM_FEATURES, _, NUM_EDGE_FEATURES = next(iter(all_drug_data.values()))[:3] 
        NUM_FEATURES, NUM_EDGE_FEATURES = NUM_FEATURES.shape[1], NUM_EDGE_FEATURES.shape[1]

        all_drug_data = {drug_id: CustomData(x=data[0], edge_index=data[1], edge_feats=data[2], line_graph_edge_index=data[3])
                    for drug_id, data in all_drug_data.items()}

        ## To speed up training
        drug_num_node_indices = {
            drug_id: torch.zeros(data.x.size(0)).long() for drug_id, data in all_drug_data.items()
        }
                
        train_tup = load_split(f'train_fold{fold}')
        train_tup, val_tup = split_train_valid(train_tup, fold)

        test_tup = load_split(f'test_fold{fold}')
        print(f'{train_tup[1].shape[1]} negative samples on fold {fold}')

        CustomDataset = TwosidesDataset if dataset_name == 'twosides' else DrugDataset

        train_data = CustomDataset(train_tup, all_drug_data, ratio=data_size_ratio, seed=fold)
        val_data = CustomDataset(val_tup, all_drug_data, ratio=data_size_ratio, seed=fold)
        test_data = CustomDataset(test_tup, all_drug_data, ratio=data_size_ratio, seed=fold)
        print(f"\nWill be training on {dataset_name} with {len(train_data)} samples, validating on {len(val_data)}, and testing on {len(test_data)} samples.")

        train_data_loader = DrugDataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_data_loader = DrugDataLoader(val_data, batch_size=batch_size)
        test_data_loader = DrugDataLoader(test_data, batch_size=batch_size)

        return train_data_loader, val_data_loader, test_data_loader, NUM_FEATURES, NUM_EDGE_FEATURES
        
    else:
        raise ValueError(f'{name}: Unrecognized dataset  name. Choose from [drugbank, twosides]')


def load_ddi_data_fold_cold_start(name, fold, batch_size=1024, data_size_ratio=1.0, valid_ratio=0.2):

    def load_split(split_name):
        filename = (f'/preprocessed/cold_start/{dataset_name}/pair_pos_neg_triples-{split_name}.csv')
        print(f'\nCold-start Loading {filename}...')
        df = pd.read_csv(filename)
        pos_triples = [(d1, d2, r) for d1, d2, r in zip(df['Drug1_ID'], df['Drug2_ID'], df['Y'])]
        neg_samples = [[str(e) for e in neg_s.split('_')] for neg_s in df['Neg samples']]

        return np.array(pos_triples), np.array(neg_samples)

    global NUM_FEATURES
    global NUM_EDGE_FEATURES
    global drug_num_node_indices

    dataset_name = name.lower()
    print(f'Loading {dataset_name}...')    

    if name.lower() == 'deepddi':
        raise NotImplementedError
    
    elif dataset_name in ('twosides', 'drugbank'):
        filename = f'data/preprocessed/{dataset_name}/drug_data.pkl'
        print('\nLoading ')
        with open(filename, 'rb') as f:
            all_drug_data = pickle.load(f)
        NUM_FEATURES, _, NUM_EDGE_FEATURES = next(iter(all_drug_data.values()))[:3]  ## TODO
        NUM_FEATURES, NUM_EDGE_FEATURES = NUM_FEATURES.shape[1], NUM_EDGE_FEATURES.shape[1]

        all_drug_data = {drug_id: 
                            CustomData(x=data[0], edge_index=data[1], edge_feats=data[2], line_graph_edge_index=data[3])
                            for drug_id, data in all_drug_data.items()}
        drug_num_node_indices = {
            drug_id: torch.zeros(data.x.size(0)).long() for drug_id, data in all_drug_data.items()
        }
        train_tup = load_split(f'fold{fold}-train')
        M_s1 = load_split(f'fold{fold}-s1')
        M_s2_tup = load_split(f'fold{fold}-s2')
        print(f'{train_tup[1].shape[1]} negative samples')

        CustomDataset = TwosidesDataset if dataset_name == 'twosides' else DrugDataset

        train_data = CustomDataset(train_tup, all_drug_data, ratio=data_size_ratio, seed=fold)
        M_s1_data = CustomDataset(M_s1, all_drug_data,  ratio=data_size_ratio, seed=fold)
        M_s2_data = CustomDataset(M_s2_tup, all_drug_data, ratio=data_size_ratio, seed=fold)
        print(f"\nWill be training on {dataset_name} with {len(train_data)} samples, s1 on {len(M_s1_data)} samples, and s2 on {len(M_s2_data)}")

        train_data_loader = DrugDataLoader(train_data, batch_size=batch_size, shuffle=True)
        M_s2_data_loader = DrugDataLoader(M_s2_data, batch_size=batch_size, shuffle=True)
        M_s1_data_loader = DrugDataLoader(M_s1_data, batch_size=batch_size)

        return train_data_loader, M_s2_data_loader, M_s1_data_loader, NUM_FEATURES, NUM_EDGE_FEATURES
        
    else:
        raise ValueError(f'{name}: Unrecognized dataset  name. Choose from [deepDDI, drugbank, twosides]')


#######    ****** ###############

class DrugDataset(Dataset):

    def __init__(self, pos_neg_tuples, all_drug_data, ratio=1.0, seed=0):
        self.pair_triples = []
        self.ratio = ratio
        self.drug_ids = list(all_drug_data.keys())
        self.all_drug_data = all_drug_data
        self.rng = np.random.RandomState(seed)

        for pos_item, neg_list in zip(*pos_neg_tuples):
            if ((pos_item[0] in self.drug_ids) and (pos_item[1] in self.drug_ids)):
                self.pair_triples.append((pos_item, neg_list))
        
        if ratio != 1.0:
            self.rng.shuffle(self.pair_triples)
            limit = math.ceil(len(self.pair_triples) * ratio)
            self.pair_triples = self.pair_triples[:limit]
    
    def collate_fn(self, batch):
        old_id_to_new_batch_id = {}
        batch_drug_feats = []
        self.node_ind_seqs = []
        self.node_i_ind_seqs_for_pair = []
        self.node_j_ind_seqs_for_pair = []

        combo_indices_pos = []
        combo_indices_neg = []
        already_in_combo = {}
        rels = []
        batch_unique_pairs= []

        for ind, (pos_item, neg_list) in enumerate(batch):
            h, t, r = pos_item[:3]
            idx_h, h_num_nodes = self._get_new_batch_id_and_num_nodes(h, old_id_to_new_batch_id, batch_drug_feats)
            idx_t, t_num_nodes = self._get_new_batch_id_and_num_nodes(t, old_id_to_new_batch_id, batch_drug_feats)
            combo_idx = self._get_combo_index((idx_h, idx_t), (h, t), already_in_combo, batch_unique_pairs, (h_num_nodes, t_num_nodes))
            combo_indices_pos.append(combo_idx)

            rels.append(int(r))

            for neg_s in neg_list:
                s = neg_s.split('$')
                neg_idx, neg_num_nodes = self._get_new_batch_id_and_num_nodes(s[0], old_id_to_new_batch_id, batch_drug_feats)
                if ('h' == s[1].lower()):
                        combo_idx = self._get_combo_index((neg_idx, idx_t), (s[0], t), already_in_combo, batch_unique_pairs, (neg_num_nodes, t_num_nodes))
                else:
                    combo_idx = self._get_combo_index((idx_h, neg_idx), (h, s[0]), already_in_combo, batch_unique_pairs, (h_num_nodes, neg_num_nodes))
                
                combo_indices_neg.append(combo_idx)
        
        batch_drug_data = Batch.from_data_list(batch_drug_feats, follow_batch=['edge_index'])
        batch_drug_pair_indices = torch.LongTensor(combo_indices_pos + combo_indices_neg)
        batch_unique_drug_pair = Batch.from_data_list(batch_unique_pairs, follow_batch=['edge_index'])
        node_j_for_pairs = torch.cat(self.node_j_ind_seqs_for_pair)
        node_i_for_pairs = torch.cat(self.node_i_ind_seqs_for_pair)
        rels = torch.LongTensor(rels)

        return batch_drug_data, batch_unique_drug_pair, rels, batch_drug_pair_indices, node_j_for_pairs, node_i_for_pairs

    def _get_new_batch_id_and_num_nodes(self, old_id, old_id_to_new_batch_id, batch_drug_feats):
        new_id = old_id_to_new_batch_id.get(old_id, -1)
        num_nodes = self.all_drug_data[old_id].x.size(0)
        if new_id == - 1:
            new_id = len(old_id_to_new_batch_id)
            old_id_to_new_batch_id[old_id] = new_id
            batch_drug_feats.append(self.all_drug_data[old_id])
            start = (self.node_ind_seqs[-1][-1] + 1) if len(self.node_ind_seqs) else 0
            self.node_ind_seqs.append(torch.arange(num_nodes) + start)
            
        return new_id, num_nodes

    def _get_combo_index(self, combo, old_combo, already_in_combo, unique_pairs, num_nodes):
        idx = already_in_combo.get(combo, -1)
        if idx == -1:
            idx = len(already_in_combo)
            already_in_combo[combo] = idx
            pair_edge_index = bipartite_edge_dict.get(old_combo)
            if pair_edge_index is None:
                index_j = torch.arange(num_nodes[0]).repeat_interleave(num_nodes[1])
                index_i = torch.arange(num_nodes[1]).repeat(num_nodes[0])
                pair_edge_index = torch.stack([index_j, index_i])
                bipartite_edge_dict[old_combo] = pair_edge_index

            j_num_indices, i_num_indices = drug_num_node_indices[old_combo[0]], drug_num_node_indices[old_combo[1]]
            unique_pairs.append(PairData(j_num_indices, i_num_indices, pair_edge_index))
            self.node_j_ind_seqs_for_pair.append(self.node_ind_seqs[combo[0]])
            self.node_i_ind_seqs_for_pair.append(self.node_ind_seqs[combo[1]])

        return idx


    def __len__(self):
        return len(self.pair_triples)

    def __getitem__(self, index):
        return self.pair_triples[index]
class TwosidesDataset(DrugDataset):

    def collate_fn(self, batch):
        old_id_to_new_batch_id = {}
        batch_drug_feats = []
        self.node_ind_seqs = []
        self.pos_node_i_ind_seqs_for_pair = []
        self.pos_node_j_ind_seqs_for_pair = []

        self.neg_node_i_ind_seqs_for_pair = []
        self.neg_node_j_ind_seqs_for_pair = []

        rels = []
        neg_rels = []
        pair_data_pos = []
        pair_data_neg = []

        for ind, (pos_item, neg_list) in enumerate(batch):
            h, t, r = pos_item[:3]
            idx_h, h_num_nodes = self._get_new_batch_id_and_num_nodes(h, old_id_to_new_batch_id, batch_drug_feats)
            idx_t, t_num_nodes = self._get_new_batch_id_and_num_nodes(t, old_id_to_new_batch_id, batch_drug_feats)
            pair_edge_index = self.get_pair_edge_index((h, t), (h_num_nodes, t_num_nodes))
            j_num_indices, i_num_indices = self.get_num_indices((h, t))
            pair_data_pos.append(PairData(j_num_indices, i_num_indices, pair_edge_index))
            self.set_node_ind_seqs((idx_h, idx_t), 'pos')

            rels.append(int(r))

            for neg_s in neg_list:
                s = neg_s.split('$')
                neg_idx, neg_num_nodes = self._get_new_batch_id_and_num_nodes(s[0], old_id_to_new_batch_id, batch_drug_feats)
                pair_edge_index = self.get_pair_edge_index((h, s[0]), (h_num_nodes, neg_num_nodes))
                j_num_indices, i_num_indices = self.get_num_indices((h, s[0]))
                pair_data_neg.append(PairData(j_num_indices, i_num_indices, pair_edge_index))
                self.set_node_ind_seqs((idx_h, neg_idx), 'neg')
                
            neg_rels.extend([int(r)] * len(neg_list))
        batch_drug_data = Batch.from_data_list(batch_drug_feats, follow_batch=['edge_index'])
        drug_pairs = Batch.from_data_list(pair_data_pos + pair_data_neg, follow_batch=['edge_index'])
        rels = torch.LongTensor(rels + neg_rels)
        node_j_for_pairs = torch.cat(self.pos_node_j_ind_seqs_for_pair + self.neg_node_j_ind_seqs_for_pair)
        node_i_for_pairs = torch.cat(self.pos_node_i_ind_seqs_for_pair + self.neg_node_i_ind_seqs_for_pair)

        return batch_drug_data, drug_pairs, rels, torch.LongTensor([len(batch)]), node_j_for_pairs, node_i_for_pairs

    def get_pair_edge_index(self, combo, num_nodes):
        pair_edge_index = bipartite_edge_dict.get(combo)
        if pair_edge_index is None:
            index_j = torch.arange(num_nodes[0]).repeat_interleave(num_nodes[1])
            index_i = torch.arange(num_nodes[1]).repeat(num_nodes[0])
            pair_edge_index = torch.stack([index_j, index_i])
            bipartite_edge_dict[combo] = pair_edge_index
        return pair_edge_index

    def get_num_indices(self, combo):
        return drug_num_node_indices[combo[0]], drug_num_node_indices[combo[1]]

    def set_node_ind_seqs(self, combo, desc):
        if desc == 'pos':
            self.pos_node_j_ind_seqs_for_pair.append(self.node_ind_seqs[combo[0]])
            self.pos_node_i_ind_seqs_for_pair.append(self.node_ind_seqs[combo[1]])
        else:
            self.neg_node_j_ind_seqs_for_pair.append(self.node_ind_seqs[combo[0]])
            self.neg_node_i_ind_seqs_for_pair.append(self.node_ind_seqs[combo[1]])

class DrugDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)


class PairData(Data):

    def __init__(self, j_indices, i_indices, pair_edge_index):
        super().__init__()
        self.i_indices = i_indices
        self.j_indices = j_indices
        self.edge_index = pair_edge_index
        self.num_nodes = None

    def __inc__(self, key, value):
        if key == 'edge_index':
            return torch.tensor([[self.j_indices.shape[0]], [self.i_indices.shape[0]]])
        if key in ('i_indices', 'j_indices'):
            return 1
        return super().__inc__(key, value)


class CustomData(Data):
    def __inc__(self, key, value):
        if key == 'line_graph_edge_index':
            return self.edge_index.size(1) if self.edge_index.nelement()!=0 else 0
        return super().__inc__(key, value)
