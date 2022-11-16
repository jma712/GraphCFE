import random
import numpy as np
from data_sampler import GraphData
from operator import itemgetter
import torch

random.seed(20)

def split_data(n, exp_num, rates=[0.6, 0.2, 0.2], labels=None, type='ratio', sorted=False, label_number=1000000):
    idx_train_list = []
    idx_val_list = []
    idx_test_list = []

    trn_rate, val_rate, tst_rate = rates[0], rates[1], rates[2]

    if type == 'ratio':  # follow the original ratio of label distribution, only applicable to binary classification!
        label_idx_0 = np.where(labels == 0)[0]
        label_idx_1 = np.where(labels == 1)[0]

        for i in range(exp_num):
            random.shuffle(label_idx_0)
            random.shuffle(label_idx_1)

            idx_train = np.append(label_idx_0[:min(int(trn_rate * len(label_idx_0)), label_number // 2)],
                                  label_idx_1[:min(int(trn_rate * len(label_idx_1)), label_number // 2)])
            idx_val = np.append(label_idx_0[int(trn_rate * len(label_idx_0)):int((trn_rate + val_rate) * len(label_idx_0))],
                                label_idx_1[int(trn_rate * len(label_idx_1)):int((trn_rate + val_rate) * len(label_idx_1))])
            idx_test = np.append(label_idx_0[int((trn_rate + val_rate) * len(label_idx_0)):], label_idx_1[int((trn_rate + val_rate) * len(label_idx_1)):])

            np.random.shuffle(idx_train)
            np.random.shuffle(idx_val)
            np.random.shuffle(idx_test)

            if sorted:
                idx_train.sort()
                idx_val.sort()
                idx_test.sort()

            idx_train_list.append(idx_train.copy())
            idx_val_list.append(idx_val.copy())
            idx_test_list.append(idx_test.copy())

    elif type == 'random':
        for i in range(exp_num):
            idx_all = np.arange(n)
            idx_train = np.random.choice(n, size=int(trn_rate * n), replace=False)
            idx_left = np.setdiff1d(idx_all, idx_train)
            idx_val = np.random.choice(idx_left, int(val_rate * n), replace=False)
            idx_test = np.setdiff1d(idx_left, idx_val)

            #sorted=True
            if sorted:
                idx_train.sort()
                idx_val.sort()
                idx_test.sort()

            idx_train_list.append(idx_train.copy())
            idx_val_list.append(idx_val.copy())
            idx_test_list.append(idx_test.copy())
    # elif type == "balanced":

    return idx_train_list, idx_val_list, idx_test_list


def get_items_from_list(li, idx_select):
    items = itemgetter(*idx_select)(li)
    return items

def select_dataloader(dataset, idx_select, batch_size=500, num_workers=0):
    dataset_select = GraphData(get_items_from_list(dataset.adj_all,idx_select), get_items_from_list(dataset.feature_all,idx_select),
                               get_items_from_list(dataset.u_all,idx_select), get_items_from_list(dataset.labels_all,idx_select),
                               dataset.max_num_nodes, dataset.padded, index=get_items_from_list(dataset.index,idx_select))
    data_loader_select = torch.utils.data.DataLoader(
        dataset_select,
        batch_size=batch_size,
        num_workers=num_workers)
    return data_loader_select