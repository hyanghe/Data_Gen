import matplotlib.pyplot as plt
import numpy as np
import sys
import time
import dgl
import torch
import pickle
import os
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import interpolate
import itertools
sys.path.insert(0, "/home/lulu/work/deepxde")
import deepxde as dde
import networkx as nx

def construct_data(f, u):
    Ny, Nx = f.shape
    inputs = []
    outputs = []
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            inputs.append(
                [
                    u[j - 1, i - 1],
                    u[j, i - 1],
                    u[j + 1, i - 1],
                    u[j - 1, i],
                    u[j + 1, i],
                    u[j - 1, i + 1],
                    u[j, i + 1],
                    u[j + 1, i + 1],
                    f[j, i],
                ]
            )
            outputs.append([u[j, i]])
    return np.array(inputs), np.array(outputs)

def construct_graph(f, u, dataset, graph_save_dir):
    Ny, Nx = f.shape
    x = np.linspace(0, 1, num=101)
    y = np.linspace(0, 1, num=101)
    x_coord, y_coord = np.meshgrid(x, y)
    node_id_data_dict = {}
    graphs = []
    graph_labels = []
    for i in range(0, Nx):
        for j in range(0, Ny):
            ## x, y, f, u
            node_id_data_dict[j*101 + i] = [x_coord[j, i], y_coord[j, i], f[j, i], u[j, i]]

    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            ctr_node_id = j*101 + i
            src_nodes = [
                ctr_node_id, ctr_node_id, ctr_node_id, ctr_node_id,\
                (j + 1)*101 + i + 1, (j - 1)*101 + i - 1, (j + 1)*101 + i - 1, (j - 1)*101 + i + 1,
                (j + 1)*101 + i + 1, (j - 1)*101 + i - 1, (j + 1)*101 + i - 1, (j - 1)*101 + i + 1,
                (j - 1) * 101 + i, (j + 1) * 101 + i, j * 101 + i - 1, j * 101 + i + 1,
                (j + 1) * 101 + i, j * 101 + i - 1, (j + 1) * 101 + i, j * 101 + i + 1,
                j * 101 + i + 1, (j - 1) * 101 + i, j * 101 + i - 1, (j - 1) * 101 + i
            ]
            tgt_nodes = [
                (j - 1)*101 + i, (j + 1)*101 + i, j*101 + i - 1, j*101 + i + 1,
                (j + 1) * 101 + i, j*101 + i - 1, (j + 1)*101 + i, j*101 + i + 1,
                j * 101 + i + 1, (j - 1)*101 + i, j*101 + i - 1, (j - 1)*101 + i,
                ctr_node_id, ctr_node_id, ctr_node_id, ctr_node_id,
                (j + 1) * 101 + i + 1, (j - 1) * 101 + i - 1, (j + 1) * 101 + i - 1, (j - 1) * 101 + i + 1,
                (j + 1) * 101 + i + 1, (j - 1) * 101 + i - 1, (j + 1) * 101 + i - 1, (j - 1) * 101 + i + 1,
            ]

            temp_dict = {}
            temp_dict[ctr_node_id] = 0
            cnt = 1
            renumbering_to_raw_dict = {}
            renumbering_to_raw_dict[0] = ctr_node_id
            for src_node in src_nodes:
                if src_node in temp_dict:
                    continue
                else:
                    temp_dict[src_node] = cnt
                    renumbering_to_raw_dict[cnt] = src_node
                    cnt += 1

            src_nodes_renumbering = [temp_dict[n] for n in src_nodes]
            tgt_nodes_renumbering = [temp_dict[n] for n in tgt_nodes]

            ## x, y, f, u as target node feature
            node_feature = np.empty((0, 4))
            all_nodes = list(renumbering_to_raw_dict.keys())
            for n in all_nodes:
                node = renumbering_to_raw_dict[n]
                if node == ctr_node_id:
                    node_feature = np.vstack((node_feature,\
                        np.concatenate((np.asarray(node_id_data_dict[node])[[0,1,2]], np.asarray([0])))))
                else:
                    node_feature = np.vstack((node_feature, np.asarray(node_id_data_dict[node])[[0, 1, 2, 3]]))
            node_label = np.asarray([np.asarray(node_id_data_dict[renumbering_to_raw_dict[node]])[[3]] for node in all_nodes])

            g = dgl.graph((src_nodes_renumbering, tgt_nodes_renumbering))

            g.ndata['X'] = torch.tensor(node_feature[:, 0:2])
            g.ndata['f'] = torch.tensor(node_feature[:, 2:3])
            g.ndata['u'] = torch.tensor(node_feature[:, 3:4])
            # g.ndata['label'] = torch.tensor(np.concatenate((src_node_label[None, :], tgt_node_label), axis=0))
            # g.ndata['label'] = src_node_label
            g.edata['e'] = torch.tensor(
                [np.linalg.norm(np.asarray(node_id_data_dict[a])[0:2] - np.asarray(node_id_data_dict[b])[0:2]) for a, b
                 in zip(src_nodes, tgt_nodes)])

            # nx_G = g.to_networkx()
            # pos = nx.kamada_kawai_layout(nx_G)
            # nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])

            graphs.append(g)

            graph_labels.append(node_label[0])
    graph_labels = {"glabel": torch.tensor(graph_labels)}
    dgl.save_graphs(f"{graph_save_dir}{dataset}.bin", graphs, graph_labels)
    print('done')


def train(model):
    model.compile("adam", lr=0.001)
    checkpointer = dde.callbacks.ModelCheckpoint(
        "model/model.ckpt", verbose=1, save_better_only=True
    )
    losshistory, train_state = model.train(epochs=100000, callbacks=[checkpointer])

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)
    model.restore("model/model.ckpt-" + str(train_state.best_step), verbose=1)

    # model.restore("model/model.ckpt-100000")


def main():
    # use_delta = False
    use_delta = True

    # f_random = np.loadtxt("./data/f_random.dat")
    # u_random = np.loadtxt("./data/u_random.dat")

    f_random = np.loadtxt("./data/f_random_use_delta_true.dat")
    u_random = np.loadtxt("./data/u_random_use_delta_true.dat")

    # f_random = np.loadtxt("./data/f_random_use_delta_false.dat")
    # u_random = np.loadtxt("./data/u_random_use_delta_false.dat")
    f0 = np.loadtxt("./data/f0.dat")
    u0 = np.loadtxt("./data/u0.dat")
    f = np.loadtxt("data/f.dat")
    u = np.loadtxt("data/u.dat")

    dataset_f_random = 'f_random'
    dataset_f = 'f'
    dataset_f0 = 'f0'
    graph_save_dir = './graph_data/'
    os.makedirs(graph_save_dir, mode=0o777, exist_ok=True)

    if not use_delta:
        print('do not use delta')
        g_random = construct_graph(f_random, u_random, dataset_f_random, graph_save_dir)
        g_0 = construct_graph(f0, u0, dataset_f0, graph_save_dir)
        g_f = construct_graph(f, u, dataset_f, graph_save_dir)

        d_random = construct_data(f_random, u_random)
        d_f0 = construct_data(f0, u0)
    else:
        print('use delta')
        g_random = construct_graph(f_random - f0, u_random - u0, dataset_f_random, graph_save_dir)
        # g_0 = construct_graph(f0 - f0, u0 - u0, dataset_f0, graph_save_dir)
        # g_f = construct_graph(f - f0, u - u0, dataset_f, graph_save_dir)

        d_random = construct_data(f_random - f0, u_random - u0)
        d_f0 = construct_data(f0 - f0, u0 - u0)
    data = dde.data.DataSet(
        X_train=d_random[0],
        y_train=d_random[1] * 1e2,
        X_test=d_f0[0],
        y_test=d_f0[1] * 1e2,
    )

    layer_size = [9, 1]
    activation = "relu"
    initializer = "He normal"
    net = dde.maps.FNN(layer_size, activation, initializer)
    net.apply_feature_transform(lambda x: x * np.array([1, 1, 1, 1, 1, 1, 1, 1, 1e-3]))
    model = dde.Model(data, net)

    train(model)
    model.print_model()

    f = np.loadtxt("data/f.dat")
    if use_delta:
        f -= f0
    u_true = np.loadtxt("data/u.dat")
    Ny, Nx = f.shape

    errs = []
    if not use_delta:
        u = u0
    else:
        u = np.zeros_like(u0)
    ts = time.time()
    for _ in range(20000):
        inputs = []
        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                inputs.append(
                    [
                        u[j - 1, i - 1],
                        u[j, i - 1],
                        u[j + 1, i - 1],
                        u[j - 1, i],
                        u[j + 1, i],
                        u[j - 1, i + 1],
                        u[j, i + 1],
                        u[j + 1, i + 1],
                        f[j, i],
                    ]
                )
        inputs = np.array(inputs)
        outputs = model.predict(inputs) * 1e-2

        k = 0
        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                u[j, i] = outputs[k, 0]
                k += 1
        if not use_delta:
            err = dde.metrics.l2_relative_error(u_true, u)
        else:
            err = dde.metrics.l2_relative_error(u_true, u + u0)
        errs.append(err)
    print("One-shot took %f s\n" % (time.time() - ts))

    np.savetxt("nn.dat", u + u0)
    plt.figure()
    plt.semilogy(errs)
    plt.show()


if __name__ == "__main__":
    main()
