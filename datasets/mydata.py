from torch_geometric.utils import to_networkx
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from random import randint, sample
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.nn as nn
import networkx as nx
import numpy as np
import torch as th
import dgl
import torch
from sklearn.preprocessing import StandardScaler
import logging
from collections import defaultdict

plt.rcParams ['font.sans-serif']=['SimHei']
plt.rcParams ['axes.unicode_minus']=False

device = ['Humidity_Sensor','Ventilator','CO2_Sensor','CO_Detection_Sensor','media_player','projector','speaker','brightness_sensor','occupancy_sensor','light_controller','motion_sensor','humidity_sensor','heater','air_conditioner','thermometer','window_sensor','fire_alarm','smoke_sensor']
values = [i for i in range(18)]
d2 = {k:v for k, v in zip(device, values)}
def caozuo(lt,c1):
    lt_new = [False for i in range(c1)]
    for i in lt:
        lt_new[i] = True
    return lt_new


def T_or_F(lt, d,a1,b1,c1):
    a, b, c = 0, 0, 0
    for i in lt:
        if i < a1:
            a += 1
        elif i < b1:
            b += 1
        elif i < c1:
            c += 1
        if a >= d and b >= d and c >= d:
            return True
    return False


def train_val_test_split(a, b, c,a1,b1,c1):
    while True:
        lt = sample([i for i in range(c1)], c1)
        train_index = lt[:int(c1 * a)]
        val_index = lt[int(c1 * a):int(c1 * a + c1 * b)]
        test_index = lt[int(c1 * a + c1 * b):]
        if T_or_F(train_index, c,a1,b1,c1) == False:
            continue
        else:
            return caozuo(train_index,c1), caozuo(val_index,c1), caozuo(test_index,c1)
        
class dataset:

    def __init__(self,a,b,c):
        self.data_x = []
        self.data_y = []
        for i in range(a):
            lt = [0 for i in range(c)]
            lt[i] = 1
            self.data_x.append(lt)
            self.data_y.append(0)
        for i in range(a, b):
            lt = [0 for i in range(c)]
            lt[i] = 1
            self.data_x.append(lt)
            self.data_y.append(1)
        for i in range(b, c):
            lt = [0 for i in range(c)]
            lt[i] = 1
            self.data_x.append(lt)
            self.data_y.append(2)
        self.data_edge = [[], []]
        lt1 = [i for i in range(a)]
        lt2 = [i for i in range(a, b)]
        lt3 = [i for i in range(b, c)]
        lt4 = lt2 + lt3
        lt5 = lt1 + lt3
        lt6 = lt1 + lt2
        lt7 = lt1 + lt2 + lt3
        for i in range(a):
            j = randint(1*a/6, 4*a/6)
            for k in range(j):
                self.data_edge[0].append(i)
            self.data_edge[1].extend(sample(lt7[:i] + lt7[i + 1:a], j))
            j = randint(0, a/6)
            for k in range(j):
                self.data_edge[0].append(i)
            self.data_edge[1].extend(sample(lt4, j))
        for i in range(a, b):
            j = randint(a/6, a/2)
            for k in range(j):
                self.data_edge[0].append(i)
            self.data_edge[1].extend(sample(lt7[a:i] + lt7[i + 1:b], j))
            j = randint(0, a/6)
            for k in range(j):
                self.data_edge[0].append(i)
            self.data_edge[1].extend(sample(lt5, j))
        for i in range(b, c):
            j = randint(4, 8)
            for k in range(j):
                self.data_edge[0].append(i)
            self.data_edge[1].extend(sample(lt7[b:i] + lt7[i + 1:], j))
            j = randint(0, a/6)
            for k in range(j):
                self.data_edge[0].append(i)
            self.data_edge[1].extend(sample(lt6, j))

    def x(self):
        return self.data_x

    def edge(self):
        return self.data_edge

    def y(self):
        return self.data_y

def preprocess(graph):
    # make bidirected
    if "feat" in graph.ndata:
        feat = graph.ndata["feat"]
    else:
        feat = None
    src, dst = graph.all_edges()
    # graph.add_edges(dst, src)
    graph = dgl.to_bidirected(graph)
    if feat is not None:
        graph.ndata["feat"] = feat

    # add self-loop
    graph = graph.remove_self_loop().add_self_loop()
    # graph.create_formats_()
    return graph


def scale_feats(x):
    logging.info("### scaling features ###")
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats


def visual(g):
    nx_G = g.to_networkx()
    plt.figure(figsize=(12, 10))
    pos = nx.kamada_kawai_layout(nx_G)
    nx.draw(nx_G, pos, with_labels=True)
    plt.show()

def load_my_dataset():
    data = dataset(18, 36, 54)
    x = th.Tensor(data.x())
    edge = th.LongTensor(data.edge())

    y = th.LongTensor(data.y())
    lt1, lt2, lt3 = train_val_test_split(0.25, 0.25,2, 18, 36, 54)
    g = dgl.graph((edge[0], edge[1]))
    # print(x,"-------------------")

    # visual(g)
    # embed = nn.Embedding(54, 14)  # 34 nodes with embedding dim equal to 5
    # g.ndata['feat'] = embed.weight
    # print(embed.weight)
    g.ndata['train_mask'] = th.BoolTensor(lt1)
    g.ndata['label'] = y

    g.ndata['val_mask'] = th.BoolTensor(lt2)
    g.ndata['test_mask'] = th.BoolTensor(lt3)
    g.ndata['feat'] = th.Tensor(np.random.rand(54, 14))
    # graph = preprocess(g)
    # feat = graph.ndata['feat']
    # feat = scale_feats(feat)
    # g.ndata['feat'] = feat
    graph = g.remove_self_loop()
    g = graph.add_self_loop()

    return g


def load_inductive_dataset(dataset_name):
    if dataset_name == "ppi":
        batch_size = 4
        # define loss function
        # create the dataset
        train_g, _ = dgl.load_graphs('graphs.dgl')
        train_dataloader = [train_g]
        # train_dataset = PPIDataset(mode='train')
        # valid_dataset = PPIDataset(mode='valid')
        # test_dataset = PPIDataset(mode='test')
        # train_dataloader = GraphDataLoader(train_dataset, batch_size=batch_size)
        # valid_dataloader = GraphDataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        # test_dataloader = GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        # eval_train_dataloader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        g = train_dataset[0]
        num_classes = train_dataset.num_labels
        num_features = g.ndata['feat'].shape[1]
    else:
        _args = namedtuple("dt", "dataset")
        dt = _args(dataset_name)
        batch_size = 1
        dataset = load_data(dt)
        num_classes = dataset.num_classes

        g = dataset[0]
        num_features = g.ndata["feat"].shape[1]

        train_mask = g.ndata['train_mask']
        feat = g.ndata["feat"]
        feat = scale_feats(feat)
        g.ndata["feat"] = feat

        g = g.remove_self_loop()
        g = g.add_self_loop()

        train_nid = np.nonzero(train_mask.data.numpy())[0].astype(np.int64)
        train_g = dgl.node_subgraph(g, train_nid)
        train_dataloader = [train_g]
        valid_dataloader = [g]
        test_dataloader = valid_dataloader
        eval_train_dataloader = [train_g]

    return train_dataloader, valid_dataloader, test_dataloader, eval_train_dataloader, num_features, num_classes



if __name__ == "__main__":
    data = dataset(18, 36, 54)
    x = th.Tensor(data.x())
    edge = th.LongTensor(data.edge())
    print(edge)
    y = th.LongTensor(data.y())
    lt1, lt2, lt3 = train_val_test_split(0.25, 0.25, 2, 18, 36, 54)
    print(th.BoolTensor(lt1), lt2, lt3)
    g = dgl.graph((edge[0], edge[1]))

    # visual(g)

    print(x, "-------------------")

    # visual(g)
    g.ndata['feat'] = th.Tensor(np.random.rand(54,14))
    # g.ndata['feat'] = embed.weight
    print(g.ndata['feat'].shape)
    print(g)
    g.ndata['label'] = y
    g.ndata['train_mask'] = th.BoolTensor(lt1)
    g.ndata['val_mask'] = th.BoolTensor(lt2)
    g.ndata['test_mask'] = th.BoolTensor(lt3)
