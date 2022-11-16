import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE as tsn

font_sz = 24
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
matplotlib.rcParams.update({'font.size': font_sz})

def draw_bar(x, y, x_label, y_label=None):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.bar(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def draw_freq(data, x_label=None, bool_discrete = False, title="Title"):
    '''
    :param data: (n,) array or sequence of (n,) arrays
    :param x_label:
    :param bool_discrete:
    :return:
    '''
    fig = plt.figure()
    plt.hist(data, bins=50)

    plt.xlabel(x_label)
    plt.ylabel("Frequency")

    ax = fig.add_subplot(1, 1, 1)

    # Find at most 10 ticks on the y-axis
    if not bool_discrete:
        max_xticks = 10
        xloc = plt.MaxNLocator(max_xticks)
        ax.xaxis.set_major_locator(xloc)

    plt.title(title)
    plt.show()

def draw_scatter(x, y, c=None, x_label=None, y_label=None, s=None, title=None, alpha=1.0, x_range=None, y_range=None, save_file=None):
    if c is not None:
        cmap = 'viridis'
    else:
        cmap = None
    plt.scatter(x, y, alpha=alpha, c=c, cmap=cmap, s=s)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if x_range is not None:
        plt.xlim(x_range[0], x_range[1])
    if y_range is not None:
        plt.ylim(y_range[0], y_range[1])
    if title is not None:
        plt.title(title)
    if c is not None:
        plt.colorbar()

    if save_file:
        plt.savefig(save_file, bbox_inches='tight')
    plt.show()

def plot_cluster(Zt, C, num_cluster, mu_zt_all=None, saving=False, Zt_tsn=None, title=None):
    cluster_color = ['red', 'blue', 'green', 'black', 'yellow', 'purple', 'pink', 'orange', 'cyan', 'brown']

    # print("centroid: ", mu_zt_all)
    fig, ax = plt.subplots()
    if Zt_tsn is None:
        if mu_zt_all is not None:
            Zt_and_center = np.concatenate((Zt, mu_zt_all), axis=0)  # (m + K) x d
        else:
            Zt_and_center = Zt
        Zt_tsn = tsn(n_components=2).fit_transform(Zt_and_center)  # m x d => m x 2
    for k in range(num_cluster):
        idx_k = np.where(C == k)
        if len(idx_k[0]) > 0:
            ax.scatter(Zt_tsn[idx_k, 0], Zt_tsn[idx_k, 1], 3, marker='o', color=cluster_color[k])  # cluster k
            if mu_zt_all is not None:
                ax.scatter(Zt_tsn[k - num_cluster, 0], Zt_tsn[k - num_cluster, 1], 100, marker='D',
                       color=cluster_color[k], alpha = 0.3)  # centroid k

        # plt.xlim(-100, 100)
    if not title is None:
        plt.title(title)

    if saving:
        plt.savefig('./figs/graphcfe_tsne.pdf', bbox_inches='tight')
    else:
        plt.show()
    return Zt_tsn

def plot_cf(x, y, x1, y1, x2, y2, c=None, x_label=None, y_label=None, s=None, title=None, alpha=1.0, x_range=None, y_range=None, save_file=None):
    x_1_pairs = [[x[i], x1[i]] for i in range(len(x))]
    x_2_pairs = [[x[i], x2[i]] for i in range(len(x))]
    y_1_pairs = [[y[i], y1[i]] for i in range(len(y))]
    y_2_pairs = [[y[i], y2[i]] for i in range(len(y))]

    colors = ['#440154FF', '#482878FF', '#3E4A89FF', '#31688EFF', '#26828EFF',
              '#1F9E89FF', '#35B779FF', '#6DCD59FF', '#B4DE2CFF', '#FDE725FF']  # 10 colors in viridis
    markers = ['o', '^', 'D', 'v', '<', 'x', 's', '*', '+', 'H', 'h']
    markers = markers + ['o' for i in range(2000)]
    max_size = min(len(markers), len(x))

    for i in range(len(x)):
        x_1_i = x_1_pairs[i]
        x_2_i = x_2_pairs[i]
        y_1_i = y_1_pairs[i]
        y_2_i = y_2_pairs[i]
        ci = int(c[i])
        plt.plot(x_1_i, y_1_i, color=colors[ci], linestyle='-', marker = markers[i], markersize=12, markevery=1)
        plt.plot(x_2_i, y_2_i, color=colors[ci], linestyle='--', marker = markers[i], markersize=12, markevery=1)

        if i >= max_size -1:
            break

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if x_range is not None:
        plt.xlim(x_range[0], x_range[1])
    if y_range is not None:
        plt.ylim(y_range[0], y_range[1])
    if title is not None:
        plt.title(title)
    # if c is not None:
    cmap = 'viridis'
    #plt.colorbar()

    if save_file:
        plt.savefig(save_file, bbox_inches='tight')
    plt.show()
    return