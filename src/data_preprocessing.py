import numpy as np
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from data_sampler import GraphData
import utils
import networkx as nx
import pickle
import plot

def load_ogbg(dataset, padded=True):
    #data = preprocess_ogbg(dataset, padded=True)

    with open('../dataset/ogbg_molhiv_full.pickle', 'rb') as handle:
        data = pickle.load(handle)['data']
    return data

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def preprocess_ogbg(dataset, padded=True, save_flag=True):
    from ogb.graphproppred import PygGraphPropPredDataset
    data = PygGraphPropPredDataset(name='ogbg-molhiv')

    # filter
    max_num_nodes = 30
    max_num_instances = len(data)

    print('data size: ', len(data))

    adj_all = []  # list; u: n, u_dim;
    features_all = []  # list; n, d;
    u_all = []
    labels_all = []
    degrees = []
    degrees_real_all = []
    num_node_real_all = []
    num_edges_all = []

    u_num = 10
    splt = np.linspace(0.15, 1.0, num=u_num + 1)
    min_1, max_1 = splt[:u_num], splt[1:]  # u_num
    min_0, max_0 = 0.0, 1.0  # splt[:u_num], splt[1:]
    feat_1_all = []
    # u
    for i in range(max_num_instances):
        u = np.random.choice(u_num, size=(1)).astype(float)
        u_all.append(u)

    for i in range(len(data)):
        graph = data[i]
        if graph.num_nodes > max_num_nodes:
            continue

        adj = to_scipy_sparse_matrix(graph.edge_index).toarray()
        np.fill_diagonal(adj, 1.)  # self.loop

        symmetric = check_symmetric(adj)
        if not symmetric:
            print('not symmetric!')

        features = graph.x.float().numpy()
        feat_add = np.random.uniform(min_0, max_0, size=(1))
        feat_1_all.append(feat_add)
        feat_add = feat_add.repeat(len(features)).reshape(-1,1)
        features = np.concatenate([feat_add, features], axis=1)

        adj_all.append(adj)
        features_all.append(features)

        num_node_real = adj.shape[0]
        degrees.append(np.array(float(np.sum(adj) - num_node_real) / (2 * max_num_nodes)))

        edge_num = float(np.sum(adj) - num_node_real) / 2

        degree_real = edge_num / num_node_real
        num_node_real_all.append(num_node_real)
        num_edges_all.append(edge_num)
        degrees_real_all.append(degree_real)

        if len(adj_all) >= max_num_instances:
            break


    degrees = np.array(degrees)
    degrees_real_all = np.array(degrees_real_all)
    num_edges_all = np.array(num_edges_all)
    num_node_real_all = np.array(num_node_real_all)
    degree_ave = np.mean(degrees)

    feat_1_all = np.array(feat_1_all)
    feat_1_ave = np.mean(feat_1_all)

    n = len(adj_all)
    u_all = u_all[:n]

    # label generation
    for i in range(n):
        y = np.array([1.0]) if feat_1_all[i] >= feat_1_ave else np.array([0.0])
        labels_all.append(y)

    data = GraphData(adj_all, features_all, u_all, labels_all, max_num_nodes, padded)

    feat_x0_all = []
    # causal constraint
    for i in range(n):
        # adjustment
        u_i = u_all[i]
        u_i = int(u_i)
        noise_1 = (max_1[u_i] - min_1[u_i]) * np.random.random_sample() + min_1[u_i]
        feat_x0 = noise_1 + 0.5 * feat_1_all[i]
        num_node = len(features_all[i])
        feat_add = feat_x0.repeat(max_num_nodes).reshape(-1, 1)  # max_num_node x 1
        feat_x0_all.append(feat_x0)

        data.feature_all[i] = np.concatenate([feat_add, data.feature_all[i]], axis=1)  # max_num_node x (dim_x + 1)

    # statistics
    num_y1 = np.array(labels_all)
    print('n=', len(adj_all), ' rate y=1: ', np.mean(num_y1), ' degree ave: ', degree_ave)
    feat_x0_all = np.array(feat_x0_all)
    plot.draw_scatter(feat_1_all, feat_x0_all, x_label='x1', y_label='x0',
                      title='ogbg', alpha=0.1)
    print('averaged degree: ', np.mean(degrees), ' std: ', np.std(degrees),
          ' averaged real degree: ', np.mean(degrees_real_all), ' std: ', np.std(degrees_real_all),
          'averaged node num: ', np.mean(num_node_real_all), ' std: ', np.std(num_node_real_all),
        'averaged node edges: ', np.mean(num_edges_all), ' std: ', np.std(num_edges_all))

    # save
    if save_flag:
        path_save = '../dataset/ogbg_molhiv_full.pickle'
        with open(path_save, 'wb') as handle:
            pickle.dump({'data': data}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('saved data: ', path_save)
    return data

def load_imdb(dataset, padded=False):
    #data = preprocess_imdb(dataset, padded, save_flag=True)
    with open('../dataset/imdb_m.pickle', 'rb') as handle:
        data = pickle.load(handle)['data']
    return data

def preprocess_imdb(dataset, padded=False, save_flag=True):
    g_list = []
    label_dict = {}
    feat_dict = {}

    max_num_nodes = 15
    max_num_instances = 100000
    d_x = 8

    adj_all = []  # list; u: n, u_dim;
    features_all = []  # list; n, d;
    u_all = []
    labels_all = []

    num_nodes_all = []
    num_edges_all = []

    degrees = []
    degrees_real_all = []
    num_node_real_all = []

    u_num = 10
    splt = np.linspace(0.15, 1.0, num=u_num + 1)
    min_1, max_1 = splt[:u_num], splt[1:]  # u_num

    # u
    for i in range(max_num_instances):
        u = np.random.choice(u_num, size=(1)).astype(float)
        u_all.append(u)

    import scipy.io
    data = scipy.io.loadmat('../dataset/IMDBMULTI.mat')

    adj_all_orin = data['graph_struct'][0]
    for i in range(len(adj_all_orin)):
        adj = adj_all_orin[i][0]
        adj = adj + np.eye(adj.shape[0])

        num_node_real = adj.shape[0]
        if num_node_real > max_num_nodes:
            continue

        symmetric = check_symmetric(adj)
        if not symmetric:
            print('not symmetric!')

        num_node_real = adj.shape[0]
        degrees.append(np.array(float(np.sum(adj) - num_node_real) / (2 * max_num_nodes)))

        edge_num = float(np.sum(adj) - num_node_real) / 2

        degree_real = edge_num / num_node_real
        num_node_real_all.append(num_node_real)
        num_edges_all.append(edge_num)
        degrees_real_all.append(degree_real)
        features_all.append(np.zeros((num_node_real, 1)))
        adj_all.append(adj)

        if len(adj_all) >= max_num_instances:
            break

    degrees = np.array(degrees)
    degrees_real_all = np.array(degrees_real_all)
    num_edges_all = np.array(num_edges_all)
    num_node_real_all = np.array(num_node_real_all)
    degree_ave = np.mean(degrees)

    n = len(adj_all)
    u_all = u_all[:n]
    feat_x1_all = []

    # label generation
    for i in range(n):
        y = np.array([1.0]) if degrees[i] >= degree_ave else np.array([0.0])
        labels_all.append(y)

    data = GraphData(adj_all, features_all, u_all, labels_all, max_num_nodes, padded)

    # causal constraint
    for i in range(n):
        u_i = u_all[i]
        u_i = int(u_i)
        noise_1 = (max_1[u_i] - min_1[u_i]) * np.random.random_sample() + min_1[u_i]
        feat_x1 = noise_1 + 0.5 * degrees[i]
        num_node = len(features_all[i])
        feat_add = feat_x1.repeat(max_num_nodes).reshape(-1, 1)
        feat_x1_all.append(feat_x1)

        data.feature_all[i] = np.concatenate([feat_add, data.feature_all[i]], axis=1)

    # statistics
    num_y1 = np.array(labels_all)
    print('n=', len(adj_all), ' rate y=1: ', np.mean(num_y1), ' degree ave: ', degree_ave)
    feature_x1_ave = np.array(feat_x1_all)
    plot.draw_scatter(degrees, feature_x1_ave, x_label='degree', y_label='x1',
                      title='imdb_m', alpha=0.1)
    print('averaged degree: ', np.mean(degrees), ' std: ', np.std(degrees),
          ' averaged real degree: ', np.mean(degrees_real_all), ' std: ', np.std(degrees_real_all),
          'averaged node num: ', np.mean(num_node_real_all), ' std: ', np.std(num_node_real_all),
          'averaged node edges: ', np.mean(num_edges_all), ' std: ', np.std(num_edges_all))

    # save
    if save_flag:
        path_save = '../dataset/imdb_m.pickle'
        with open(path_save, 'wb') as handle:
            pickle.dump({'data': data}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('saved data: ', path_save)
    return data

def load_community(dataset, padded=False):
    n = 10000
    max_num_nodes = 20

    with open('../dataset/community_3.pickle', 'rb') as handle:
        data = pickle.load(handle)['data']

    return data

def generate_community(n, max_num_nodes=10, padded=False, save_flag=True):
    adj_all = []
    features_all = []
    u_all = []
    labels_all = []
    d_x = 1

    u_num = 10
    splt = np.linspace(0.15, 1.0, num=u_num + 1)
    min_1, max_1 = splt[:u_num], splt[1:]
    min_0, max_0 = 0.0, 1.0

    # u
    for i in range(n):
        u = np.random.choice(u_num, size=(1)).astype(float)
        u_all.append(u)

    # adj, x
    degrees_0 = []
    degrees_1 = []
    degree_all = []
    num_edges_all = []
    for i in range(n):
        u_i = u_all[i]
        u_i = int(u_i)
        p0 = (max_0 - min_0) * np.random.random_sample() + min_0
        noise_1 = (max_1[u_i] - min_1[u_i]) * np.random.random_sample() + min_1[u_i]
        p1 = noise_1 - 0.15 * p0

        p1 = max(0., p1)
        p1 = min(1., p1)

        n0 = int(max_num_nodes / 2)
        n1 = max_num_nodes - n0
        graph_0 = nx.fast_gnp_random_graph(n0, p0)
        adj_0 = nx.adjacency_matrix(graph_0, nodelist=sorted(graph_0.nodes())).todense() + np.eye(n0)
        graph_1 = nx.fast_gnp_random_graph(n1, p1)
        adj_1 = nx.adjacency_matrix(graph_1, nodelist=sorted(graph_1.nodes())).todense() + np.eye(n1)
        adj = np.zeros((max_num_nodes, max_num_nodes))
        adj[:n0, :n0] = adj_0
        adj[n0:, n0:] = adj_1

        dg0 = [val for (node, val) in graph_0.degree()]
        dg_ave_0 = float(sum(dg0)) / len(dg0)
        dg1 = [val for (node, val) in graph_1.degree()]
        dg_ave_1 = float(sum(dg1)) / len(dg1)

        adj_all.append(adj)
        degrees_0.append(np.array(dg_ave_0))
        degrees_1.append(np.array(dg_ave_1))

        x = np.random.normal(0, 1, (max_num_nodes, d_x))
        features_all.append(x)

        num_edge = (float(adj.sum()) - max_num_nodes) / 2
        degree_all.append(num_edge / max_num_nodes)
        num_edges_all.append(num_edge)

    degrees_0 = np.array(degrees_0)
    degrees_1 = np.array(degrees_1)
    degrees_0_ave = np.mean(degrees_0)
    num_edges_all = np.array(num_edges_all)
    degree_all = np.array(degree_all)

    plot.draw_scatter(degrees_0, degrees_1, x_label='degree of community 0', y_label='degree of community 1', title='Community', alpha=0.1)
    print('averaged degree of community 0: ', np.mean(degrees_0), ' std: ', np.std(degrees_0),
          ' ave degree of community 1: ', np.mean(degrees_1), ' std: ', np.std(degrees_1),
          ' averaged degree: ', np.mean(degree_all), ' std: ', np.std(degree_all), ' averaged num edge: ', np.mean(num_edges_all), ' std：', np.std(num_edges_all))

    # label generation
    for i in range(n):
        y = np.array([1.0]) if degrees_0[i] >= degrees_0_ave else np.array([0.0])
        labels_all.append(y)

    data = GraphData(adj_all, features_all, u_all, labels_all, max_num_nodes, padded)

    # statistics
    num_y1 = np.array(labels_all)
    print('rate y=1: ', np.mean(num_y1))

    # save
    if save_flag:
        path_save = '../dataset/community_3.pickle'
        with open(path_save, 'wb') as handle:
            pickle.dump({'data': data}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('saved data: ', path_save)
    return data

def load_data(path_root, dataset, exp_num=10):

    if dataset == 'community':
        data = load_community(dataset, padded=True)
        n = len(data)

        # ====== if you want to split the data from scratch, use the commented code =======
        # idx_train_list, idx_val_list, idx_test_list = utils.split_data(n, exp_num, rates=[0.6, 0.2, 0.2],
        #                                                                labels=np.array(data.labels_all),
        #                                                                type='ratio', sorted=False, label_number=1000000)

        # path_save = '../dataset/community_datasplit.pickle'
        # with open(path_save, 'wb') as handle:
        #     pickle.dump(
        #         {'idx_train_list': idx_train_list, 'idx_val_list': idx_val_list, 'idx_test_list': idx_test_list},
        #         handle, protocol=pickle.HIGHEST_PROTOCOL)
        # print('saved data: ', path_save)
        path_data = path_root + 'community_datasplit.pickle'
        with open(path_data, 'rb') as handle:
            info_index = pickle.load(handle)
            idx_train_list, idx_val_list, idx_test_list = info_index['idx_train_list'], \
                                                          info_index['idx_val_list'], info_index['idx_test_list']


    elif dataset == 'ogbg_molhiv':
        data = load_ogbg(dataset, padded=True)
        n = len(data)

        # ====== if you want to split the data from scratch, use the commented code =======
        # idx_train_list, idx_val_list, idx_test_list = utils.split_data(n, exp_num, rates=[0.6, 0.2, 0.2],
        #                                                                labels=np.array(data.labels_all),
        #                                                                type='ratio', sorted=False, label_number=1000000)
        #
        # path_save = '../dataset/ogbg_molhiv_datasplit.pickle'
        # with open(path_save, 'wb') as handle:
        #     pickle.dump(
        #         {'idx_train_list': idx_train_list, 'idx_val_list': idx_val_list, 'idx_test_list': idx_test_list},
        #         handle, protocol=pickle.HIGHEST_PROTOCOL)
        # print('saved data: ', path_save)
        path_data = path_root + 'ogbg_molhiv_datasplit.pickle'
        with open(path_data, 'rb') as handle:
            info_index = pickle.load(handle)
            idx_train_list, idx_val_list, idx_test_list = info_index['idx_train_list'], \
                                                          info_index['idx_val_list'], info_index['idx_test_list']

    elif dataset == 'imdb_m':
        data = load_imdb(dataset, padded=True)
        n = len(data)
        # ====== if you want to split the data from scratch, use the commented code =======
        # idx_train_list, idx_val_list, idx_test_list = utils.split_data(n, exp_num, rates=[0.6, 0.2, 0.2],
        #                                                                labels=np.array(data.labels_all),
        #                                                                type='random', sorted=False, label_number=1000000)
        # path_save = '../dataset/imdb_m_datasplit.pickle'
        # with open(path_save, 'wb') as handle:
        #     pickle.dump(
        #         {'idx_train_list': idx_train_list, 'idx_val_list': idx_val_list, 'idx_test_list': idx_test_list},
        #         handle, protocol=pickle.HIGHEST_PROTOCOL)
        # print('saved data: ', path_save)
        path_data = path_root + 'imdb_m_datasplit.pickle'
        with open(path_data, 'rb') as handle:
            info_index = pickle.load(handle)
            idx_train_list, idx_val_list, idx_test_list = info_index['idx_train_list'], \
                                                          info_index['idx_val_list'], info_index['idx_test_list']

    else:
        print('Invalid dataset name!!')
        exit(0)


    print("loaded dataset: ", dataset, "num of instances: ", n)

    data_load = {
        'data': data,
        'idx_train_list': idx_train_list, 'idx_val_list': idx_val_list, 'idx_test_list': idx_test_list
    }

    return data_load

if __name__ == '__main__':
    data_path_root = '../dataset/'
    model_path = '../models_save/'

    dataset = 'community'
    data_load = load_data(data_path_root, dataset)