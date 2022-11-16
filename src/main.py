
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
import random
import utils
import plot
import pickle

sys.path.append('../')

font_sz = 28

parser = argparse.ArgumentParser(description='Graph counterfactual explanation generation')
parser.add_argument('--nocuda', type=int, default=0, help='Disables CUDA training.')
parser.add_argument('--batch_size', type=int, default=500, metavar='N',
                    help='input batch size for training (default: 500)')
parser.add_argument('--num_workers', type=int, default=0, metavar='N')
parser.add_argument('--epochs', type=int, default=2000, metavar='N',
                    help='number of epochs to train (default: 2000)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--trn_rate', type=float, default=0.6, help='training data ratio')
parser.add_argument('--tst_rate', type=float, default=0.2, help='test data ratio')

parser.add_argument('--lamda', type=float, default=200, help='weight for CFE loss')
parser.add_argument('--kl_weight', type=float, default=1.0, help='weight for KL loss')
parser.add_argument('--disable_u', type=int, default=0, help='disable u in VAE')
parser.add_argument('--dim_z', type=int, default=16, metavar='N', help='dimension of z')
parser.add_argument('--dim_h', type=int, default=16, metavar='N', help='dimension of h')
parser.add_argument('--dropout', type=float, default=0.1)

parser.add_argument('--dataset', default='imdb_m', help='dataset to use',
                    choices=['community', 'ogbg_molhiv', 'imdb_m'])
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate for optimizer')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='weight decay')

parser.add_argument('--experiment_type', default='train', choices=['train', 'test', 'baseline'],
                    help='train: train CLEAR model; test: load CLEAR from file; baseline: run a baseline')
parser.add_argument('--baseline_type', default='random', choices=['IST', 'random', 'RM'],
                    help='select baseline type: insert, random perturb, or remove edges')

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

def add_list_in_dict(key, dict, elem):
    if key not in dict:
        dict[key] = [elem]
    else:
        dict[key].append(elem)
    return dict

def distance_feature(feat_1, feat_2):
    pdist = nn.PairwiseDistance(p=2)
    output = pdist(feat_1, feat_2) /4
    return output

def distance_graph_prob(adj_1, adj_2_prob):
    dist = F.binary_cross_entropy(adj_2_prob, adj_1)
    return dist

def proximity_feature(feat_1, feat_2, type='cos'):
    if type == 'cos':
        cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        output = cos(feat_1, feat_2)
        output = torch.mean(output)
    return output


def compute_loss(params):
    model, pred_model, z_mu, z_logvar, adj_permuted, features_permuted, adj_reconst, features_reconst, \
    adj_input, features_input, y_cf, z_u_mu, z_u_logvar, z_mu_cf, z_logvar_cf = params['model'], params['pred_model'], params['z_mu'], \
        params['z_logvar'], params['adj_permuted'], params['features_permuted'], params['adj_reconst'], params['features_reconst'], \
        params['adj_input'], params['features_input'], params['y_cf'], params['z_u_mu'], params['z_u_logvar'], params['z_mu_cf'], params['z_logvar_cf']

    # kl loss
    loss_kl = 0.5 * (((z_u_logvar - z_logvar) + ((z_logvar.exp() + (z_mu - z_u_mu).pow(2)) / z_u_logvar.exp())) - 1)
    loss_kl = torch.mean(loss_kl)

    # similarity loss
    size = len(features_permuted)
    dist_x = torch.mean(distance_feature(features_permuted.view(size, -1), features_reconst.view(size, -1)))
    dist_a = distance_graph_prob(adj_permuted, adj_reconst)

    beta = 10

    loss_sim = beta * dist_x + 10 * dist_a

    # CFE loss
    y_pred = pred_model(features_reconst, adj_reconst)['y_pred']  # n x num_class
    loss_cfe = F.nll_loss(F.log_softmax(y_pred, dim=-1), y_cf.view(-1).long())

    # rep loss
    if z_mu_cf is None:
        loss_kl_cf = 0.0
    else:
        loss_kl_cf = 0.5 * (((z_logvar_cf - z_logvar) + ((z_logvar.exp() + (z_mu - z_mu_cf).pow(2)) / z_logvar_cf.exp())) - 1)
        loss_kl_cf = torch.mean(loss_kl_cf)

    loss = 1. * loss_sim + 1 * loss_kl + 1.0 * loss_cfe

    loss_results = {'loss': loss, 'loss_kl': loss_kl, 'loss_sim': loss_sim, 'loss_cfe': loss_cfe, 'loss_kl_cf':loss_kl_cf}
    return loss_results


def train(params):
    epochs, pred_model, model, optimizer, y_cf_all, train_loader, val_loader, test_loader, exp_i, dataset, metrics, variant = \
        params['epochs'], params['pred_model'], params['model'], params['optimizer'], params['y_cf'],\
        params['train_loader'], params['val_loader'], params['test_loader'], params['exp_i'], params['dataset'], params['metrics'], params['variant']
    save_model = params['save_model'] if 'save_model' in params else True
    print("start training!")

    time_begin = time.time()
    best_loss = 100000

    for epoch in range(epochs + 1):
        model.train()

        loss, loss_kl, loss_sim, loss_cfe, loss_kl_cf = 0.0, 0.0, 0.0, 0.0, 0.0
        batch_num = 0
        for batch_idx, data in enumerate(train_loader):
            batch_num += 1

            features = data['features'].float().to(device)
            adj = data['adj'].float().to(device)
            u = data['u'].float().to(device)
            orin_index = data['index']
            y_cf = y_cf_all[orin_index]

            optimizer.zero_grad()

            # forward pass
            model_return = model(features, u, adj, y_cf)

            # z_cf
            z_mu_cf, z_logvar_cf = model.get_represent(model_return['features_reconst'], u, model_return['adj_reconst'], y_cf)

            # compute loss
            loss_params = {'model': model, 'pred_model': pred_model, 'adj_input': adj, 'features_input': features, 'y_cf': y_cf, 'z_mu_cf': z_mu_cf, 'z_logvar_cf':z_logvar_cf}
            loss_params.update(model_return)

            loss_results = compute_loss(loss_params)
            loss_batch, loss_kl_batch, loss_sim_batch, loss_cfe_batch, loss_kl_batch_cf = loss_results['loss'], loss_results['loss_kl'], loss_results['loss_sim'], loss_results['loss_cfe'], loss_results['loss_kl_cf']
            loss += loss_batch
            loss_kl += loss_kl_batch
            loss_sim += loss_sim_batch
            loss_cfe += loss_cfe_batch
            loss_kl_cf += loss_kl_batch_cf

        # backward propagation
        loss, loss_kl, loss_sim, loss_cfe, loss_kl_cf = loss / batch_num, loss_kl/batch_num, loss_sim/batch_num, loss_cfe/batch_num, loss_kl_cf/batch_num

        alpha = 5

        if epoch < 450:
            ((loss_sim + loss_kl + 0* loss_cfe)/ batch_num).backward()
        else:
            ((loss_sim + loss_kl + alpha * loss_cfe)/ batch_num).backward()
        optimizer.step()

        # evaluate
        if epoch % 100 == 0:
            model.eval()
            eval_params_val = {'model': model, 'data_loader': val_loader, 'pred_model': pred_model, 'y_cf': y_cf_all, 'dataset': dataset, 'metrics': metrics}
            eval_params_tst = {'model': model, 'data_loader': test_loader, 'pred_model': pred_model, 'y_cf': y_cf_all, 'dataset': dataset, 'metrics': metrics}
            eval_results_val = test(eval_params_val)
            eval_results_tst = test(eval_params_tst)
            val_loss, val_loss_kl, val_loss_sim, val_loss_cfe = eval_results_val['loss'], eval_results_val['loss_kl'], eval_results_val['loss_sim'], eval_results_val['loss_cfe']

            metrics_results_val = ""
            metrics_results_tst = ""
            for k in metrics:
                metrics_results_val += f"{k}_val: {eval_results_val[k]:.4f} | "
                metrics_results_tst += f"{k}_tst: {eval_results_tst[k]:.4f} | "

            print(f"[Train] Epoch {epoch}: train_loss: {(loss):.4f} |" +
                  metrics_results_val + metrics_results_tst +
                  f"time: {(time.time() - time_begin):.4f} |")

            # save
            if save_model:
                if epoch % 300 == 0 and epoch > 450:
                    CFE_model_path = f'../models_save/weights_graphCFE_{variant}_{args.dataset}_exp' + str(exp_i) + '_epoch'+str(epoch) + '.pt'
                    torch.save(model.state_dict(), CFE_model_path)
                    print('saved CFE model in: ', CFE_model_path)

        # if epoch % 2000 == 0 and args.dataset == 'imdb_m':
        #     x_range = [0.0, 6]
        #     y_range = [0.0, 5]
        #
        #     if epoch == 0:
        #         title = args.dataset + 'origin'
        #         features_permuted = model_return['features_permuted']
        #         adj_permuted = model_return['adj_permuted']
        #
        #         size = len(adj_permuted)
        #         num_nodes = adj_permuted.shape[-1]
        #         ave_x0 = torch.mean(features_permuted[:, :, 0], dim=-1)
        #         ave_degree = (torch.sum(adj_permuted.reshape(size, -1), dim=-1) - num_nodes) / (2 * num_nodes)  # size
        #         plot.draw_scatter(ave_degree.detach().cpu().numpy(), ave_x0.detach().cpu().numpy(),
        #                           c=u.detach().cpu().numpy(),
        #                           x_label='degree',
        #                           y_label='x0',  # title='Community original',
        #                           alpha=0.5,
        #                           x_range=x_range,
        #                           y_range=y_range,
        #                           save_file='../exp_results/' + title + '.pdf'
        #                           )
        #         continue
        #
        #     features_reconst = model_return['features_reconst']
        #     ave_x0_cf = torch.mean(features_reconst[:, :, 0], dim=-1)
        #     adj_reconst = model_return['adj_reconst']
        #     adj_reconst_binary = torch.bernoulli(adj_reconst)
        #
        #     ave_degree_cf_prob = (torch.sum(adj_reconst.reshape(size, -1), dim=-1) - num_nodes) / (2 * num_nodes)
        #     ave_degree_cf = (torch.sum(adj_reconst_binary.reshape(size, -1), dim=-1) - num_nodes) / (2 * num_nodes)
        #
        #     method = 'VAE' if args.disable_u else 'CLEAR'
        #     title = args.dataset + ' CFE' + ', ' + method + ' epoch' + str(epoch)
        #     plot.draw_scatter(ave_degree_cf.detach().cpu().numpy(), ave_x0_cf.detach().cpu().numpy(),
        #                       c=u.detach().cpu().numpy(),
        #                       x_label='degree',
        #                       y_label='x0', title=None, alpha=0.5,
        #                       x_range=x_range,
        #                       y_range=y_range,
        #                       save_file='../exp_results/' + title + '.pdf'
        #                       )
        #     title = args.dataset + ' CFE' + ', prob, ' + method + ' epoch' + str(epoch)
        #     plot.draw_scatter(ave_degree_cf_prob.detach().cpu().numpy(), ave_x0_cf.detach().cpu().numpy(),
        #                       c=u.detach().cpu().numpy(),
        #                       x_label='degree',
        #                       y_label='x0',
        #                       title=None,
        #                       alpha=0.5,
        #                       x_range=x_range,
        #                       y_range=y_range,
        #                       save_file='../exp_results/' + title + '.pdf')
        #
        # if epoch % 3000 == 0 and args.dataset == 'community':
        #     x_range = [0.0, 4.5]
        #     y_range = [0.0, 4.5]
        #
        #     if epoch == 0:
        #         title = 'Community_original'
        #         n0 = 10
        #         n1 = 10
        #         adj_permuted = model_return['adj_permuted']
        #         size = len(adj_permuted)
        #         ave_degree_0 = (torch.sum(adj_permuted[:, :n0, :n0].reshape(size, -1), dim=-1) - n0) / (2 * n0)  # size
        #         ave_degree_1 = (torch.sum(adj_permuted[:, n0:, n0:].reshape(size, -1), dim=-1) - n1) / (2 * n1)  # size
        #         plot.draw_scatter(ave_degree_0.detach().cpu().numpy(), ave_degree_1.detach().cpu().numpy(),
        #                           c=u.detach().cpu().numpy(),
        #                           x_label='degree of community 1',
        #                           y_label='degree of community 2',
        #                           alpha=0.5,
        #                           x_range=x_range,
        #                           y_range=y_range,
        #                           save_file='../exp_results/' + title + '.pdf'
        #                           )
        #         continue

    return

def test(params):
    model, data_loader, pred_model, y_cf_all, dataset, metrics = params['model'], params['data_loader'], params['pred_model'], params['y_cf'], params['dataset'], params['metrics']
    model.eval()
    pred_model.eval()

    eval_results_all = {k: 0.0 for k in metrics}
    size_all = 0
    loss, loss_kl, loss_sim, loss_cfe = 0.0, 0.0, 0.0, 0.0
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
        y_cf = y_cf_all[orin_index]

        model_return = model(features, u, adj, y_cf)
        adj_reconst, features_reconst = model_return['adj_reconst'], model_return['features_reconst']

        adj_reconst_binary = torch.bernoulli(adj_reconst)
        y_cf_pred = pred_model(features_reconst, adj_reconst_binary)['y_pred']
        y_pred = pred_model(features, adj)['y_pred']

        # z_cf
        z_mu_cf, z_logvar_cf = None, None

        # compute loss
        loss_params = {'model': model, 'pred_model': pred_model, 'adj_input': adj, 'features_input': features, 'y_cf': y_cf, 'z_mu_cf': z_mu_cf, 'z_logvar_cf':z_logvar_cf}
        loss_params.update(model_return)

        loss_results = compute_loss(loss_params)
        loss_batch, loss_kl_batch, loss_sim_batch, loss_cfe_batch = loss_results['loss'], loss_results['loss_kl'], \
                                                                    loss_results['loss_sim'], loss_results['loss_cfe']
        loss += loss_batch
        loss_kl += loss_kl_batch
        loss_sim += loss_sim_batch
        loss_cfe += loss_cfe_batch

        # evaluate metrics
        eval_params = model_return.copy()
        eval_params.update({'y_cf': y_cf, 'metrics': metrics, 'y_cf_pred': y_cf_pred, 'dataset': dataset, 'adj_input': adj, 'features_input': features, 'labels':labels, 'u': u, 'y_pred':y_pred})

        eval_results = evaluate(eval_params)
        for k in metrics:
            eval_results_all[k] += (batch_size * eval_results[k])

    for k in metrics:
        eval_results_all[k] /= size_all

    loss, loss_kl, loss_sim, loss_cfe = loss / batch_num, loss_kl / batch_num, loss_sim / batch_num, loss_cfe / batch_num
    eval_results_all['loss'], eval_results_all['loss_kl'], eval_results_all['loss_sim'], eval_results_all['loss_cfe'] = loss, loss_kl, loss_sim, loss_cfe

    return eval_results_all

def evaluate(params):
    adj_permuted, features_permuted, adj_reconst_prob, features_reconst, metrics, dataset, y_cf, y_cf_pred, labels, u, y_pred = \
        params['adj_permuted'], params['features_permuted'], params['adj_reconst'], \
        params['features_reconst'], params['metrics'], params['dataset'], params['y_cf'], params['y_cf_pred'], params['labels'], params['u'], params['y_pred']

    adj_reconst = torch.bernoulli(adj_reconst_prob)
    eval_results = {}
    if 'causality' in metrics:
        score_causal = evaluate_causality(dataset, adj_permuted, features_permuted, adj_reconst, features_reconst,  y_cf, labels, u)
        eval_results['causality'] = score_causal
    if 'proximity' in metrics or 'proximity_x' in metrics or 'proximity_a' in metrics:
        score_proximity, dist_x, dist_a = evaluate_proximity(dataset, adj_permuted, features_permuted, adj_reconst_prob, adj_reconst, features_reconst)
        eval_results['proximity'] = score_proximity
        eval_results['proximity_x'] = dist_x
        eval_results['proximity_a'] = dist_a
    if 'validity' in metrics:
        score_valid = evaluate_validity(y_cf, y_cf_pred)
        eval_results['validity'] = score_valid
    if 'correct' in metrics:
        score_correct = evaluate_correct(dataset, adj_permuted, features_permuted, adj_reconst, features_reconst, y_cf, labels, y_cf_pred, y_pred)
        eval_results['correct'] = score_correct

    return eval_results

def evaluate_validity(y_cf, y_cf_pred):
    y_cf_pred_binary = F.softmax(y_cf_pred, dim=-1)
    y_cf_pred_binary = y_cf_pred_binary.argmax(dim=1).view(-1,1)
    y_eq = torch.where(y_cf == y_cf_pred_binary, torch.tensor(1.0).to(device), torch.tensor(0.0).to(device))
    score_valid = torch.mean(y_eq)
    return score_valid

def evaluate_causality(dataset, adj_permuted, features_permuted, adj_reconst, features_reconst, y_cf, labels, u):
    score_causal = 0.0
    if dataset == 'synthetic' or dataset == 'imdb_m':
        size = len(features_permuted)
        max_num_nodes = adj_reconst.shape[-1]

        # Constraint
        ave_degree = (torch.sum(adj_permuted.view(size, -1), dim=-1) - max_num_nodes) / (2 * max_num_nodes) # size
        ave_degree_cf = (torch.sum(adj_reconst.view(size, -1), dim=-1) - max_num_nodes) / (2 * max_num_nodes)
        ave_x0 = torch.mean(features_permuted[:, :, 0], dim=-1)  # size
        ave_x0_cf = torch.mean(features_reconst[:, :, 0], dim=-1)  # size

        count_good = torch.where(
            (((ave_degree > ave_degree_cf) & (ave_x0 > ave_x0_cf)) |
             ((ave_degree == ave_degree_cf) & (ave_x0 == ave_x0_cf)) |
             ((ave_degree < ave_degree_cf) & (ave_x0 < ave_x0_cf))), torch.tensor(1.0).to(device), torch.tensor(0.0).to(device))

        score_causal = torch.mean(count_good)

    elif dataset == 'ogbg_molhiv':
        ave_x0 = torch.mean(features_permuted[:, :, 0], dim=-1)  # size
        ave_x0_cf = torch.mean(features_reconst[:, :, 0], dim=-1)  # size
        ave_x1 = torch.mean(features_permuted[:, :, 1], dim=-1)  # size
        ave_x1_cf = torch.mean(features_reconst[:, :, 1], dim=-1)  # size

        count_good = torch.where(
            (((ave_x0 > ave_x0_cf) & (ave_x1 > ave_x1_cf)) |
             ((ave_x0 == ave_x0_cf) & (ave_x1 == ave_x1_cf)) |
             ((ave_x0 < ave_x0_cf) & (ave_x1 < ave_x1_cf))), torch.tensor(1.0).to(device),
            torch.tensor(0.0).to(device))
        score_causal = torch.mean(count_good)

    elif dataset == 'community':
        size = len(features_permuted)
        max_num_nodes = adj_reconst.shape[-1]

        # Constraint
        n0 = int(max_num_nodes/2)
        n1 = max_num_nodes - n0

        ave_degree_0 = (torch.sum(adj_permuted[:, :n0, :n0].reshape(size, -1), dim=-1) - n0) / (2 * n0)  # size
        ave_degree_cf_0 = (torch.sum(adj_reconst[:, :n0, :n0].reshape(size, -1), dim=-1) - n0) / (2 * n0)
        ave_degree_1 = (torch.sum(adj_permuted[:, n0:, n0:].reshape(size, -1), dim=-1) - n1) / (2 * n1)  # size
        ave_degree_cf_1 = (torch.sum(adj_reconst[:, n0:, n0:].reshape(size, -1), dim=-1) - n1) / (2 * n1)

        max_dg = ave_degree_1.max().tile(len(ave_degree_1))
        min_dg = ave_degree_1.min().tile(len(ave_degree_1))

        count_good = torch.where(
            (((ave_degree_0 > ave_degree_cf_0) & (((ave_degree_1 < max_dg) & (ave_degree_1 < ave_degree_cf_1)) | (ave_degree_1 == max_dg))) |
             ((ave_degree_0 == ave_degree_cf_0) & (ave_degree_1 == ave_degree_cf_1)) |
             ((ave_degree_0 < ave_degree_cf_0) & (((ave_degree_1 > min_dg) & (ave_degree_1 > ave_degree_cf_1)) | (ave_degree_1 == min_dg)))), torch.tensor(1.0).to(device),
            torch.tensor(0.0).to(device))

        score_causal = torch.mean(count_good)

    return score_causal

def evaluate_proximity(dataset, adj_permuted, features_permuted, adj_reconst_prob, adj_reconst, features_reconst):
    size = len(features_permuted)
    dist_x = torch.mean(distance_feature(features_permuted.view(size, -1), features_reconst.view(size, -1)))
    dist_a = distance_graph_prob(adj_permuted, adj_reconst_prob)
    score = dist_x + dist_a

    proximity_x = proximity_feature(features_permuted, features_reconst, 'cos')

    acc_a = (adj_permuted == adj_reconst).float().mean()
    return score, proximity_x, acc_a

def evaluate_correct(dataset, adj_permuted, features_permuted, adj_reconst, features_reconst, y_cf, labels, y_cf_pred, y_pred):
    y_cf_pred_binary = F.softmax(y_cf_pred, dim=-1)
    y_cf_pred_binary = y_cf_pred_binary.argmax(dim=1).view(-1, 1)
    y_pred_binary = F.softmax(y_pred, dim=-1)
    y_pred_binary = y_pred_binary.argmax(dim=1).view(-1, 1)

    score = -1.0
    if dataset == 'synthetic' or dataset == 'imdb_m':
        size = len(features_permuted)
        max_num_nodes = adj_reconst.shape[-1]
        ave_degree = (torch.sum(adj_permuted.view(size, -1), dim=-1) - max_num_nodes) / (2 * max_num_nodes)  # size
        ave_degree_cf = (torch.sum(adj_reconst.view(size, -1), dim=-1) - max_num_nodes) / (2 * max_num_nodes)

        count_good = torch.where(
            (((ave_degree > ave_degree_cf) & (labels.view(-1) > y_cf.view(-1))) |
            ((ave_degree < ave_degree_cf) & (labels.view(-1) < y_cf.view(-1)))), torch.tensor(1.0).to(device), torch.tensor(0.0).to(device))

        score = torch.sum(count_good)
        all = (labels.view(-1) != y_cf.view(-1)).sum()
        if all.item() == 0:
            return score / (all+1)
        score = score / all
    elif dataset == 'ogbg_molhiv':
        ave_x1 = torch.mean(features_permuted[:, :, 1], dim=-1)  # size
        ave_x1_cf = torch.mean(features_reconst[:, :, 1], dim=-1)  # size

        count_good = torch.where(
            (((ave_x1 > ave_x1_cf) & (y_pred_binary.view(-1) > y_cf_pred_binary.view(-1))) |
             ((ave_x1 < ave_x1_cf) & (y_pred_binary.view(-1) < y_cf_pred_binary.view(-1)))),
            torch.tensor(1.0).to(device),
            torch.tensor(0.0).to(device))

        score = torch.sum(count_good)
        all = (y_pred_binary.view(-1) != y_cf_pred_binary.view(-1)).sum()
        if all.item() == 0:
            return score / (all + 1)
        score = score / all

    elif dataset == 'community':
        size = len(features_permuted)
        max_num_nodes = adj_reconst.shape[-1]

        n0 = int(max_num_nodes / 2)
        n1 = max_num_nodes - n0
        ave_degree_0 = (torch.sum(adj_permuted[:, :n0, :n0].reshape(size, -1), dim=-1) - n0) / (2 * n0)  # size
        ave_degree_cf_0 = (torch.sum(adj_reconst[:, :n0, :n0].reshape(size, -1), dim=-1) - n0) / (2 * n0)

        count_good = torch.where(
            (((ave_degree_0 > ave_degree_cf_0) & (y_pred_binary.view(-1) > y_cf_pred_binary.view(-1))) |
             ((ave_degree_0 < ave_degree_cf_0) & (y_pred_binary.view(-1) < y_cf_pred_binary.view(-1)))), torch.tensor(1.0).to(device),
            torch.tensor(0.0).to(device))
        # score = torch.mean(count_good)
        score = torch.sum(count_good)
        all = (y_pred_binary.view(-1) != y_cf_pred_binary.view(-1)).sum()
        if all.item() == 0:
            return score / (all + 1)
        score = score / all

    return score

def perturb_graph(adj, type='random', num_rounds=1):
    num_node = adj.shape[0]
    num_entry = num_node * num_node
    adj_cf = adj.clone()
    if type == 'random':
        # randomly add/remove edges for T rounds
        for rd in range(num_rounds):
            [row, col] = np.random.choice(num_node, size=2, replace=False)
            adj_cf[row, col] = 1 - adj[row, col]
            adj_cf[col, row] = adj_cf[row, col]

    elif type == 'IST':
        # randomly add edge
        for rd in range(num_rounds):
            idx_select = (adj_cf == 0).nonzero()  # 0
            if len(idx_select) <= 0:
                continue
            ii = np.random.choice(len(idx_select), size=1, replace=False)
            idx = idx_select[ii].view(-1)
            row, col = idx[0], idx[1]
            adj_cf[row, col] = 1
            adj_cf[col, row] = 1

    elif type == 'RM':
        # randomly remove edge
        for rd in range(num_rounds):
            idx_select = adj_cf.nonzero()  # 1
            if len(idx_select) <= 0:
                continue
            ii = np.random.choice(len(idx_select), size=1, replace=False)
            idx = idx_select[ii].view(-1)
            row, col = idx[0], idx[1]
            adj_cf[row, col] = 0
            adj_cf[col, row] = 0

    return adj_cf

def baseline_cf(dataset, data_loader, metrics, y_cf_all, pred_model, num_rounds = 10, type='random'):
    eval_results_all = {k: 0.0 for k in metrics}
    size_all = 0
    batch_num = 0
    for batch_idx, data in enumerate(data_loader):
        batch_num += 1
        batch_size = len(data['labels'])
        size_all += batch_size

        features = data['features'].float().to(device)
        adj = data['adj'].float().to(device)
        u = data['u'].float().to(device)
        labels = data['labels'].float().to(device)
        orin_index = data['index'].to(device)
        y_cf = y_cf_all[orin_index].to(device)

        adj_reconst = adj.clone()

        noise = torch.normal(mean=0.0, std=1, size=features.shape).to(device)  # add a Gaussian noise to node features
        features_reconst = features + noise

        # perturbation on A
        for i in range(batch_size):
            for t in range(num_rounds):
                adj_reconst[i] = perturb_graph(adj_reconst[i], type, num_rounds=1)  # randomly perturb graph
                y_cf_pred_i = pred_model(features_reconst[i].unsqueeze(0), adj_reconst[i].unsqueeze(0))['y_pred'].argmax(dim=1).view(-1,1)  # 1 x 1
                if y_cf_pred_i.item() == y_cf[i].item():  # Stop when f(G^CF) == Y^CF
                    break

        # prediction model
        y_cf_pred = pred_model(features_reconst, adj_reconst)['y_pred']
        y_pred = pred_model(features, adj)['y_pred']

        # evaluate metrics
        eval_params = {}
        eval_params.update(
            {'y_cf': y_cf, 'metrics': metrics, 'y_cf_pred': y_cf_pred, 'dataset': dataset, 'adj_permuted': adj,
             'features_permuted': features, 'adj_reconst': adj_reconst, 'features_reconst': features_reconst, 'labels': labels, 'u': u, 'y_pred':y_pred})

        eval_results = evaluate(eval_params)
        for k in metrics:
            eval_results_all[k] += (batch_size * eval_results[k])

    for k in metrics:
        eval_results_all[k] /= size_all

    return eval_results_all

def run_clear(args, exp_type):
    data_path_root = '../dataset/'
    model_path = '../models_save/'
    assert exp_type == 'train' or exp_type == 'test' or exp_type == 'test_small'
    small_test = 20

    # load data
    data_load = dpp.load_data(data_path_root, args.dataset)
    idx_train_list, idx_val_list, idx_test_list = data_load['idx_train_list'], data_load['idx_val_list'], data_load[
        'idx_test_list']
    data = data_load['data']
    x_dim = data[0]["features"].shape[1]
    u_unique = np.unique(np.array(data.u_all))
    u_dim = len(u_unique)

    n = len(data)
    max_num_nodes = data.max_num_nodes
    unique_class = np.unique(np.array(data.labels_all))
    num_class = len(unique_class)
    print('n ', n, 'x_dim: ', x_dim, ' max_num_nodes: ', max_num_nodes, ' num_class: ', num_class)

    results_all_exp = {}
    exp_num = 3
    init_params = {'vae_type': 'graphVAE', 'x_dim': x_dim, 'u_dim': u_dim,
                   'max_num_nodes': max_num_nodes}  # parameters for initialize GraphCFE model

    # load model
    pred_model = models.Graph_pred_model(x_dim, 32, num_class, max_num_nodes, args.dataset).to(device)
    pred_model.load_state_dict(torch.load(model_path + f'prediction/weights_graphPred__{args.dataset}' + '.pt'))
    pred_model.eval()

    y_cf = 1 - np.array(data.labels_all)
    y_cf = torch.FloatTensor(y_cf).to(device)

    metrics = ['causality', 'validity', 'proximity_x', 'proximity_a']
    time_spent_all = []

    for exp_i in range(0, exp_num):
        print('============================= Start experiment ', str(exp_i),
              ' =============================================')
        idx_train = idx_train_list[exp_i]
        idx_val = idx_val_list[exp_i]
        idx_test = idx_test_list[exp_i]

        if args.disable_u:
            model = models.GraphCFE(init_params=init_params, args=args)
        else:
            model = models.GraphCFE(init_params=init_params, args=args)


        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # data loader
        train_loader = utils.select_dataloader(data, idx_train, batch_size=args.batch_size,
                                               num_workers=args.num_workers)
        val_loader = utils.select_dataloader(data, idx_val, batch_size=args.batch_size, num_workers=args.num_workers)
        test_loader = utils.select_dataloader(data, idx_test, batch_size=args.batch_size, num_workers=args.num_workers)

        if args.cuda:
            model = model.to(device)

        variant = 'VAE' if args.disable_u else 'CLEAR'
        if exp_type == 'train':
            # train
            train_params = {'epochs': args.epochs, 'model': model, 'pred_model': pred_model, 'optimizer': optimizer,
                            'y_cf': y_cf,
                            'train_loader': train_loader, 'val_loader': val_loader, 'test_loader': test_loader,
                            'exp_i': exp_i,
                            'dataset': args.dataset, 'metrics': metrics, 'save_model': False, 'variant': variant}
            train(train_params)
        else:
            # test
            CFE_model_path = model_path + f'weights_graphCFE_{variant}_{args.dataset}_exp' + str(exp_i) +'_epoch'+str(900) + '.pt'
            model.load_state_dict(torch.load(CFE_model_path))
            print('CFE generator loaded from: ' + CFE_model_path)
            if exp_type == 'test_small':
                test_loader = utils.select_dataloader(data, idx_test[:small_test], batch_size=args.batch_size, num_workers=args.num_workers)

        test_params = {'model': model, 'dataset': args.dataset, 'data_loader': test_loader, 'pred_model': pred_model,
                       'metrics': metrics, 'y_cf': y_cf}

        time_begin = time.time()

        eval_results = test(test_params)

        time_end = time.time()
        time_spent = time_end - time_begin
        time_spent = time_spent / small_test
        time_spent_all.append(time_spent)

        for k in metrics:
            results_all_exp = add_list_in_dict(k, results_all_exp, eval_results[k].detach().cpu().numpy())

        print('=========================== Exp ', str(exp_i), ' Results ==================================')
        for k in eval_results:
            if isinstance(eval_results[k], list):
                print(k, ": ", eval_results[k])
            else:
                print(k, f": {eval_results[k]:.4f}")
        print('time: ', time_spent)

    print('============================= Overall Results =============================================')
    record_exp_result = {}  # save in file
    for k in results_all_exp:
        results_all_exp[k] = np.array(results_all_exp[k])
        print(k, f": mean: {np.mean(results_all_exp[k]):.4f} | std: {np.std(results_all_exp[k]):.4f}")
        record_exp_result[k] = {'mean': np.mean(results_all_exp[k]), 'std': np.std(results_all_exp[k])}

    time_spent_all = np.array(time_spent_all)
    record_exp_result['time'] = {'mean': np.mean(time_spent_all), 'std': np.std(time_spent_all)}

    save_result = False
    print("====save in file ====")
    print(record_exp_result)
    if save_result:
        exp_save_path = '../exp_results/'
        if args.disable_u:
            exp_save_path = exp_save_path + 'CVAE' + '.pickle'
        else:
            exp_save_path = exp_save_path + 'CLEAR' + '.pickle'
        with open(exp_save_path, 'wb') as handle:
            pickle.dump(record_exp_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('saved data: ', exp_save_path)
    return

def run_baseline(args, type='random'):
    data_path_root = '../dataset/'
    model_path = '../models_save/'
    small_test = 20
    num_rounds = 150

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

    results_all_exp = {}
    exp_num = 3  # 10

    # load model
    pred_model = models.Graph_pred_model(x_dim, 32, num_class, max_num_nodes, args.dataset).to(device)
    pred_model.load_state_dict(torch.load(model_path + f'prediction/weights_graphPred__{args.dataset}' + '.pt'))
    pred_model.eval()

    y_cf = 1 - np.array(data.labels_all)
    y_cf = torch.FloatTensor(y_cf).to(device)
    metrics = ['causality', 'proximity', 'validity', 'proximity_x', 'proximity_a', 'correct']

    time_spent_all = []

    for exp_i in range(0, exp_num):
        print('============================= Start experiment ', str(exp_i),
              ' =============================================')
        time_begin = time.time()

        idx_test = idx_test_list[exp_i]

        # data loader
        test_loader = utils.select_dataloader(data, idx_test[:small_test], batch_size=args.batch_size, num_workers=args.num_workers)

        # baseline
        eval_results = baseline_cf(args.dataset, test_loader, metrics, y_cf, pred_model, num_rounds=num_rounds, type=type)

        time_end = time.time()
        time_spent = time_end - time_begin
        time_spent = time_spent / small_test
        time_spent_all.append(time_spent)

        for k in metrics:
            results_all_exp = add_list_in_dict(k, results_all_exp, eval_results[k].detach().cpu().numpy())

        print('=========================== Exp ', str(exp_i), ' Results ==================================')
        for k in eval_results:
            if isinstance(eval_results[k], list):
                print(k, ": ", eval_results[k])
            else:
                print(k, f": {eval_results[k]:.4f}")
        print('time: ', time_spent)

    print('============================= Overall Results =============================================')

    for k in results_all_exp:
        results_all_exp[k] = np.array(results_all_exp[k])
        print(k, f": mean: {np.mean(results_all_exp[k]):.4f} | std: {np.std(results_all_exp[k]):.4f}")
    time_spent_all = np.array(time_spent_all)
    print('time', f": mean: {np.mean(time_spent_all):.4f} | std: {np.std(time_spent_all):.4f}")
    return


if __name__ == '__main__':
    experiment_type = args.experiment_type
    print('running experiment: ', experiment_type)

    if experiment_type == 'train':
        run_clear(args, 'train')
    elif experiment_type == 'test':
        run_clear(args, 'test_small')
    elif experiment_type == 'baseline':
        baseline_type = 'random'  # IST, random, RM
        run_baseline(args, baseline_type)

