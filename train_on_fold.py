from datetime import datetime
import numpy as np
import torch
from torch import optim
import models
import time
from tqdm import tqdm
from ddi_datasets import load_ddi_data_fold, total_num_rel
from custom_loss import SigmoidLoss
from custom_metrics import do_compute_metrics
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, required=True, choices=['drugbank', 'twosides'])
parser.add_argument('-fold', '--fold', type=int, required=True, help='Fold on which to train on')
parser.add_argument('-n_iter', '--n_iter', type=int, required=True, help='Number of iterations/')
parser.add_argument('-drop', '--dropout', type=float, default=0, help='dropout probability')
parser.add_argument('-b', '--batch_size', type=int, default=512, help='Batch size')

args = parser.parse_args()

print(args)

dataset_name = args.dataset
fold_i = args.fold
dropout = args.dropout
n_iter = args.n_iter
TOTAL_NUM_RELS = total_num_rel(name=dataset_name)
batch_size = args.batch_size
data_size_ratio = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
hid_feats = 64
rel_total = TOTAL_NUM_RELS
lr = 1e-3
weight_decay = 5e-4
n_epochs = 100
kge_feats = 64

def do_compute(model, batch, device): 

        batch = [t.to(device) for t in batch]
        p_score, n_score = model(batch)
        assert p_score.ndim == 2
        assert n_score.ndim == 3
        probas_pred = np.concatenate([torch.sigmoid(p_score.detach()).cpu().mean(dim=-1), torch.sigmoid(n_score.detach()).mean(dim=-1).view(-1).cpu()])
        ground_truth = np.concatenate([np.ones(p_score.shape[0]), np.zeros(n_score.shape[:2]).reshape(-1)])

        return p_score, n_score, probas_pred, ground_truth


def run_batch(model, optimizer, data_loader, epoch_i, desc, loss_fn, device):
        total_loss = 0
        loss_pos = 0
        loss_neg = 0
        probas_pred = []
        ground_truth = []
        
        for batch in tqdm(data_loader, desc= f'{desc} Epoch {epoch_i}'):
            p_score, n_score, batch_probas_pred, batch_ground_truth = do_compute(model, batch, device)

            probas_pred.append(batch_probas_pred)
            ground_truth.append(batch_ground_truth)

            loss, loss_p, loss_n = loss_fn(p_score, n_score)
            if model.training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            loss_pos += loss_p.item()
            loss_neg += loss_n.item() 
        total_loss /= len(data_loader)
        loss_pos /= len(data_loader)
        loss_neg /= len(data_loader)
        
        probas_pred = np.concatenate(probas_pred)
        ground_truth = np.concatenate(ground_truth)

        return total_loss, do_compute_metrics(probas_pred, ground_truth)


def print_metrics(loss, acc, auroc, f1_score, precision, recall, int_ap, ap):
    print(f'loss: {loss:.4f}, acc: {acc:.4f}, roc: {auroc:.4f}, f1: {f1_score:.4f}, ', end='')
    print(f'p: {precision:.4f}, r: {recall:.4f}, int-ap: {int_ap:.4f}, ap: {ap:.4f}')  

    return f1_score


def train(model, train_data_loader, val_data_loader, test_data_loader, loss_fn, optimizer, n_epochs, device, scheduler):
    for epoch_i in range(1, n_epochs+1):
        start = time.time()
        model.train()
        ## Training
        train_loss, train_metrics = run_batch(model, optimizer, train_data_loader, epoch_i,  'train', loss_fn, device)
        if scheduler:
            scheduler.step()

        model.eval()
        with torch.no_grad():

            ## Validation 
            if val_data_loader:
                val_loss , val_metrics = run_batch(model, optimizer, val_data_loader, epoch_i, 'val', loss_fn, device)


        if train_data_loader:
            print(f'\n#### Epoch time {time.time() - start:.4f}s')
            print_metrics(train_loss, *train_metrics)

        if val_data_loader:
            print('#### Validation')
            print_metrics(val_loss, *val_metrics)


train_data_loader, val_data_loader, test_data_loader, NUM_FEATURES, NUM_EDGE_FEATURES = \
    load_ddi_data_fold(dataset_name, fold_i, batch_size=batch_size, data_size_ratio=data_size_ratio)

GmpnnNet = models.GmpnnCSNetDrugBank if dataset_name == 'drugbank' else models.GmpnnCSNetTwosides

model = GmpnnNet(NUM_FEATURES, NUM_EDGE_FEATURES, hid_feats, rel_total, n_iter, dropout)
loss_fn = SigmoidLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** (epoch))

time_stamp = f'{datetime.now()}'.replace(':', '_')


model.to(device=device)
print(f'Training on {device}.')
print(f'Starting fold_{fold_i} at', datetime.now())
train(model, train_data_loader, val_data_loader, test_data_loader, loss_fn, optimizer, n_epochs, device, scheduler)
