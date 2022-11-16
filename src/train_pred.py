import time
import numpy as np
import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import math
import argparse
import os
import sys
import scipy.io as scio

import numpy as np
import data_preprocessing as dpp
import models
import utils
import random
from data_sampler import GraphData
from operator import itemgetter
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

sys.path.append('../')
torch.backends.cudnn.enabled = False


parser = argparse.ArgumentParser(description='Graph counterfactual explanation generation')
parser.add_argument('--nocuda', type=int, default=0, help='Disables CUDA training.')
parser.add_argument('--batch_size', type=int, default=5000, metavar='N',
                    help='input batch size for training (default: 500)')  # community: 500， ogbg: 5000
parser.add_argument('--num_workers', type=int, default=0, metavar='N')
parser.add_argument('--epochs', type=int, default=600, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--trn_rate', type=float, default=0.6, help='training data ratio')
parser.add_argument('--tst_rate', type=float, default=0.2, help='test data ratio')

parser.add_argument('--dim_h', type=int, default=32, metavar='N', help='dimension of h')
parser.add_argument('--dropout', type=float, default=0.1)

parser.add_argument('--dataset', default='ogbg_molhiv', help='dataset to use',
                    choices=['community', 'ogbg_molhiv', 'imdb_m'])
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate for optimizer')  # community: 1e-3
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='weight decay')

args = parser.parse_args()

# select gpu if available
args.cuda = not args.nocuda and torch.cuda.is_available()
device = torch.device("cuda:0" if args.cuda else "cpu")
args.device = device

print('using device: ', device)

# seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def compute_loss(params):
    labels, y_pred = params['labels'], params['y_pred']

    loss = F.nll_loss(F.log_softmax(y_pred, dim=-1), labels.view(-1).long())

    loss_results = {'loss': loss}
    return loss_results

def train(params):
    epochs, model, optimizer, train_loader, val_loader, test_loader, dataset, metrics = \
        params['epochs'], params['model'], params['optimizer'], \
        params['train_loader'], params['val_loader'], params['test_loader'], params['dataset'], params['metrics']
    save_model = params['save_model'] if 'save_model' in params else True
    print("start training!")

    time_begin = time.time()
    best_loss = float('inf')

    for epoch in range(epochs + 1):
        model.train()

        loss = 0.0
        batch_num = 0
        for batch_idx, data in enumerate(train_loader):
            batch_num += 1
            model.zero_grad()

            features = data['features'].float().to(device)
            adj = data['adj'].float().to(device)
            labels = data['labels'].float().to(device)

            optimizer.zero_grad()

            # forward pass
            model_return = model(features, adj)

            # compute loss
            loss_params = {'model': model, 'labels': labels}
            loss_params.update(model_return)

            loss_results = compute_loss(loss_params)
            loss_batch = loss_results['loss']
            loss += loss_batch

        # backward propagation
        (loss / batch_num).backward()
        optimizer.step()

        # evaluate
        if epoch % 100 == 0:
            model.eval()
            eval_params_val = {'model': model, 'data_loader': val_loader, 'dataset': dataset, 'metrics': metrics}
            eval_params_tst = {'model': model, 'data_loader': test_loader, 'dataset': dataset, 'metrics': metrics}
            eval_results_val = test(eval_params_val)
            eval_results_tst = test(eval_params_tst)
            val_loss = eval_results_val['loss']

            metrics_results_val = ""
            metrics_results_tst = ""
            for k in metrics:
                metrics_results_val += f"{k}_val: {eval_results_val[k]:.4f} | "
                metrics_results_tst += f"{k}_tst: {eval_results_tst[k]:.4f} | "

            print(f"[Train] Epoch {epoch}: train_loss: {(loss):.4f} |  "
                  f"val_loss: {(val_loss):.4f} |" + metrics_results_val + metrics_results_tst +
                  f"time: {(time.time() - time_begin):.4f} |")

            # save
            if save_model:
                val_loss = eval_results_val['loss']
                if val_loss < best_loss:
                    best_loss = val_loss
                    path_model = f'../models_save/prediction/weights_graphPred__{args.dataset}.pt'
                    torch.save(model.state_dict(), path_model)
                    print('model saved in ', path_model)

    return


def test(params):
    model, data_loader, dataset, metrics = params['model'], params['data_loader'], params['dataset'], params['metrics']
    model.eval()

    eval_results_all = {k: 0.0 for k in metrics}
    size_all = 0
    loss = 0.0
    batch_num = 0
    for batch_idx, data in enumerate(data_loader):
        batch_num += 1
        batch_size = len(data['labels'])
        size_all += batch_size

        features = data['features'].float().to(device)
        adj = data['adj'].float().to(device)
        u = data['u'].float().to(device)
        labels = data['labels'].float().to(device)
        orin_index = data['index']

        if (labels == 1).sum() == len(labels):
            print('no way!')

        model_return = model(features, adj)
        y_pred = model_return['y_pred']

        # compute loss
        loss_params = {'model': model, 'labels': labels}
        loss_params.update(model_return)

        loss_results = compute_loss(loss_params)
        loss_batch = loss_results['loss']
        loss += loss_batch

        # evaluate metrics
        eval_params = model_return.copy()
        eval_params.update({'y_true': labels, 'y_pred': y_pred, 'metrics': metrics})

        eval_results = evaluate(eval_params)
        # batch -> all, if the metrics is calculated by averaging over all instances
        for k in metrics:
            eval_results_all[k] += (batch_size * eval_results[k])

    for k in metrics:
        eval_results_all[k] /= size_all

    loss = loss / batch_num
    eval_results_all['loss'] = loss
    return eval_results_all

def evaluate(params):
    y_true, y_pred, metrics = params['y_true'], params['y_pred'], params['metrics']
    # y_pred_binary = torch.where(y_pred >= 0.5, torch.tensor(1.0).to(device), torch.tensor(0.0).to(device))
    y_pred = F.softmax(y_pred, dim=-1)
    pred_label = y_pred.argmax(dim=1).view(-1, 1)  # n x 1

    eval_results = {}
    if 'Accuracy' in metrics:
        acc = accuracy_score(y_true.cpu().numpy(), pred_label.cpu().numpy())
        eval_results['Accuracy'] = acc
    if 'AUC-ROC' in metrics:
        auc_roc = roc_auc_score(y_true.cpu().numpy(), y_pred[:,1].detach().cpu().numpy())
        eval_results['AUC-ROC'] = auc_roc
    if 'F1-score' in metrics:
        f1 = f1_score(y_true.cpu().numpy(), pred_label.cpu().numpy())
        eval_results['F1-score'] = f1
    return eval_results

if __name__ == '__main__':
    print('running experiment: ', 'training prediction model')
    data_path_root = '../dataset/'
    model_path = '../models_save/'
    if args.dataset == 'synthetic' or args.dataset == 'ogbg_molhiv' or args.dataset == 'community' or args.dataset == 'imdb_m':
        metrics = ['Accuracy', 'AUC-ROC', 'F1-score']
    else:
        metrics = ['Accuracy']

    data_load = dpp.load_data(data_path_root, args.dataset)
    idx_train_list, idx_val_list, idx_test_list = data_load['idx_train_list'], data_load['idx_val_list'], data_load[
        'idx_test_list']
    data = data_load['data']
    x_dim = data[0]["features"].shape[1]

    n = len(data)
    max_num_nodes = data.max_num_nodes
    unique_class = np.unique(np.array(data.labels_all))
    num_class = len(unique_class)
    print('n ', n, 'x_dim: ', x_dim, ' max_num_nodes: ', max_num_nodes, ' num_class: ', num_class)
    # dpp.data_statistics(data_load)

    results_all_exp = {}
    init_params = {'x_dim': x_dim, 'h_dim': args.dim_h, 'max_num_nodes': max_num_nodes}  # parameters for initialize GraphCFE model

    # load model
    #disable_x = True if args.dataset == 'synthetic' else False
    model = models.Graph_pred_model(x_dim, args.dim_h, num_class, max_num_nodes, args.dataset).to(device)
    # pred_model.load_state_dict(torch.load(model_path + f'weights_graphPred_{args.dataset}' + '.pt'))

    idx_train = idx_train_list[0]
    idx_val = idx_val_list[0]
    idx_test = idx_test_list[0]

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # data loader
    train_loader = utils.select_dataloader(data, idx_train, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = utils.select_dataloader(data, idx_val, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = utils.select_dataloader(data, idx_test, batch_size=args.batch_size, num_workers=args.num_workers)

    if args.cuda:
        model = model.to(device)

    # train
    train_params = {'epochs': args.epochs, 'model': model, 'optimizer': optimizer,
                    'train_loader': train_loader, 'val_loader': val_loader, 'test_loader': test_loader,
                    'dataset': args.dataset, 'metrics': metrics, 'save_model': True}
    train(train_params)

    # test
    # model = models.GraphCFE(init_params=init_params, args=args).to(device)
    # model.load_state_dict(torch.load(model_path + f'weights_graphCFE_{args.dataset}_exp' + str(exp_i) + '.pt'))

    test_params = {'model': model, 'dataset': args.dataset, 'data_loader': test_loader, 'metrics': metrics}
    eval_results = test(test_params)

    for k in eval_results:
        if isinstance(eval_results[k], list):
            print(k, ": ", eval_results[k])
        else:
            print(k, f": {eval_results[k]:.4f}")



