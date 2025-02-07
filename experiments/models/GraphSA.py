# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 08:43:29 2023

@author: Olivia Zhang
"""
# In[1]
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import copy
import warnings
import sys
import time
from sklearn.metrics import classification_report
from dgl.data.utils import load_graphs

warnings.filterwarnings("ignore")
sys.path.append("../..")
from datasets.graph_gen import create_g,agg_weights,visual,S1set,S3set,S0set,S2set,Allset
from experiments.compare_model.AHP import AHP
Mag = 100
attack_ids = [16,10,13]
test_ids=[16,17,19]
# In[2]

class SageLayer(nn.Module):
    """
    一层SageLayer
    """

    def __init__(self, input_size, out_size, gcn=False):
        super(SageLayer, self).__init__()
        self.input_size = input_size
        self.out_size = out_size
        self.gcn = gcn
        self.weights = 0.5

    def forward(self, self_feats, aggregate_feats, neighs=None):
        """
        Parameters:
            self_feats:源节点的特征向量
            aggregate_feats:聚合后的邻居节点特征
        """
        # print(self.gcn)
        if self.gcn:  # 如果不是gcn的话就要进行concatenate
            # for agg_feat in aggregate_feats:
            #     if len(agg_feat) != 0:
            #
            combined = self_feats * self.weights + aggregate_feats * (1 - self.weights)
            # combined = torch.cat([self_feats, aggregate_feats], dim=1)
        else:
            combined = aggregate_feats
        return combined


class GraphSage(nn.Module):
    """定义一个GraphSage模型"""

    def __init__(self, num_layers, input_size, out_size, raw_features, adj_lists, gcn=False, agg_func='MEAN'):
        super(GraphSage, self).__init__()
        self.input_size = input_size
        self.out_size = out_size
        self.num_layers = num_layers  # Graphsage的层数
        self.gcn = gcn
        self.agg_func = agg_func
        self.raw_features = raw_features
        self.adj_lists = adj_lists
        # 定义每一层的输入和输出
        for index in range(1, num_layers + 1):
            layer_size = out_size if index != 1 else input_size
            setattr(self, 'sage_layer' + str(index),
                    SageLayer(layer_size, out_size, gcn=self.gcn))  # 除了第1层的输入为input_size,其余层的输入和输出均为outsize

    def forward(self, nodes_batch):
        """
        为一批节点生成嵌入表示
        Parameters:
            nodes_batch:目标批次的节点
        """
        # lower_layer_nodes = list(nodes_batch)  # 初始化第一层节点
        # nodes_batch_layers = [(lower_layer_nodes,)]  # 存放每一层的节点信息
        # for i in range(self.num_layers):
        #     lower_samp_neighs, lower_layer_nodes_dict, lower_layer_nodes = self._get_unique_neighs_list(
        #         lower_layer_nodes)  # 根据当前层节点获得下一层节点
        #     nodes_batch_layers.insert(0, (lower_layer_nodes, lower_samp_neighs, lower_layer_nodes_dict))
        # assert len(nodes_batch_layers) == self.num_layers + 1

        pre_hidden_embs = self.raw_features  # 初始化h0
        # print(basescore)
        for index in range(1, self.num_layers + 1):
            aggregate_feats = self.aggregate0(pre_hidden_embs)
            sage_layer = getattr(self, 'sage_layer' + str(index))
            # self.dc.logger.info('sage_layer.')
            cur_hidden_embs = sage_layer(self_feats=pre_hidden_embs,
                                         aggregate_feats=aggregate_feats)
            pre_hidden_embs = cur_hidden_embs
        return pre_hidden_embs

    # 聚合函数 Agg_function
    def aggregate0(self, pre_hidden_embs):
        neigh,weights = agg_weights()
        # print(weights)
        agg_feats = []
        for i in range(len(pre_hidden_embs)):
            if len(neigh[i]) == 0:
                agg_feat = pre_hidden_embs[i]
            else:
                agg_feat = pre_hidden_embs[neigh[i][0]] * weights[i][0]
                for j in range(1,len(neigh[i])):
                    agg_feat += pre_hidden_embs[neigh[i][j]] * weights[i][j]
            agg_feats.append(agg_feat)

        agg_feats = torch.stack(agg_feats)

        return agg_feats

    def min_best(self,X):
        for i in range(len(X)):
            X[i] = max(X)-X[i]
        return X

    def standard(self,X):
        xmin = X.min(axis=0)
        xmax = X.max(axis=0)
        xmaxmin = xmax-xmin
        n, m = X.shape
        for i in range(n):
            for j in range(m):
                X[i,j] = (X[i,j]-xmin[j])/xmaxmin[j]
        return X

    def critic(self,X):
        n,m = X.shape
        M = copy.copy(X)
        M[:,2] = self.min_best(M[:,2])  # 自己的数据根据实际情况
        Z = self.standard(M)  # 标准化X，去量纲
        R = np.array(pd.DataFrame(Z).corr())    #各个指标之间的冲突性，相关系数corr()越大，
        delta = np.zeros(m)
        c = np.zeros(m)
        for j in range(m):
            delta[j] = Z[:,j].std()        #指标的信息量，标准差std越大，包含信息量越大
            c[j] = R.shape[0] - R[:,j].sum()
        C = delta * c
        w = np.round(C/sum(C),7)
        # print(w)
        labels = []
        for label in (torch.from_numpy(X) * torch.from_numpy(w)).tolist():
            labels.append(np.round(sum(label) * Mag, 1))
        return labels

def test(features,labels,id):
    if id == 1:
        output1 = [features.tolist()[0]]
        output1[0].append(labels[0])
        j=1
        for i in range(1,20):
            output1.append(features.tolist()[i])
        for i in range(1,20):
            output1[j].append(labels[i])
            j += 1
    else:

        output1 = [features.tolist()[test_ids[0]]]
        j = 0

        for i in test_ids[1:]:
            output1.append(features.tolist()[i])
        for i in test_ids:
            output1[j].append(labels[i])
            j += 1
    # np.savetxt('Critic'+str(id)+'.csv', output1, delimiter=',')
    np.savetxt('../experiments/excel/output.csv' , output1, delimiter=',')

def visualize(h, color, s):
    z = TSNE(perplexity=10, n_components=2).fit_transform(h.detach().cpu().numpy())
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(16, 9))
    plt.xticks([])
    plt.yticks([])
    plt.scatter(z[:, 0], z[:, 1], s=3, c=color)
    plt.savefig("F:\小论文\安全评估\AHMS_GATSA\datasets\\figure2", dpi=1000)



def visual_0():
    g, adj_lists = create_g("attack")
    feature_data = g.ndata['feat']
    nodes = [i for i in range(feature_data.size(0))]
    graphSage = GraphSage(2, feature_data.size(1), 128, feature_data, adj_lists, gcn='store_true', agg_func='MEAN')
    labels = graphSage.critic(feature_data.numpy())
    features = graphSage(nodes)
    final_labels = graphSage.critic(features.numpy())
    grades = [min(*item) for item in zip(labels, final_labels)]
    visual(adj_lists,grades,attack_ids)
    # visual(adj_lists,grades,[])

# In[3]
def HGL_CC():
    # load graph data
    g, adj_lists = create_g("S1")
    # train_dataloader = S3set(20)
    train_dataloader = load_graphs("../../datasets/test/grade.bin")[0]
    print(len(train_dataloader))

    y_label = []
    y_pred = []
    j = 1
    for subgraph in train_dataloader:
        print("\r", end="")
        print(j)
        print("Test progress: {}%: ".format(j *1/ 24), "▋" * (j // 2), end="")
        j = j + 1
        sys.stdout.flush()
        # graph, adj_lists = create_g()
        feature_data = subgraph.ndata['feat']
        label = subgraph.ndata['label'].numpy().tolist()
        y_label+=label
        # raw_label = graph.ndata['label']
        # print(feature_data)

        nodes = [i for i in range(feature_data.size(0))]
        # load neighbor and weights data
        # for i in test_ids:
        #     print(neigh[i])

        # print(neigh)

        graphSage = GraphSage(2, feature_data.size(1), 128, feature_data, adj_lists, gcn='store_true', agg_func='MEAN')
        labels = graphSage.critic(feature_data.numpy())
        # test(feature_data,labels,1)
        features = graphSage(nodes)
        # security = np.array([[round(features.tolist()[i][j] * Mag) for j in range(0, feature_data.size(1))] for i in range(0, feature_data.size(0))])

        final_labels = graphSage.critic(features.numpy())
        grades = [min(*item) for item in zip(labels, final_labels)]
        safety = np.ones(len(grades))
        for g in range(len(labels)):
            if grades[g] < 60:
                safety[g] = 0
        safety = safety.tolist()
        y_pred+=safety

    accuracy = classification_report(y_label, y_pred, output_dict=True)['accuracy']
    s = classification_report(y_label, y_pred, output_dict=True)['weighted avg']
    precision = s['precision']
    recall = s['recall']
    f1_score = s['f1-score']
    print(accuracy,precision,recall,f1_score)

        # print(accuracy)
        # print(safety)
        # output1.append(safety)

    # np.savetxt('../experiments/excel/output.csv', output1, delimiter=',')
    # grades = [92.50 ,91.90 ,89.60 ,91.90 ,90.00 ,93.30 ,91.80 ,91.60 ,91.60 ,92.60 ,91.20 ,92.50 ,91.80 ,92.90 ,91.70 ,93.30 ,90.40 ,91.80 ,91.50 ,92.80 ,91.50]

    # print(grade)
    # visual(adj_lists,grades,attack_ids)
    # print('graph_grade:'+str(sum(grade)/len(grade)))
    # print(grades)
    # test(features,grades,1)

    # np.array(output).to_csv("F:\小论文\安全评估\AHMS_GATSA\datasets\\results\output.csv")
    # visualize(feature_data, color=["red"] * 7 + ["blue"] * 7 + ["green"] * 7, s=3)
if __name__ == '__main__':
    # visual_0()
    HGL_CC()