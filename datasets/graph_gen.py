import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch as th
import dgl
from dgl.data.utils import save_graphs,load_graphs
from collections import defaultdict
import PIL
import os
import copy
import sys
import random
sys.path.append("..")
from IoT_Attack_Path.src.NetGen import neigh_basescore
from IoT_Attack_Path.src import SecurityEvaluator

Mag = 100

excel_file = '../../datasets/edges.csv'               #导入excel数据
# weights_file = '../datasets/edge_de.csv'               #导入excel数据
device = ['Humidity_Sensor','Ventilator','CO2_Sensor','heater', 'window_sensor0', 'window_sensor1','air_conditioner','CO_Detection_Sensor','Light0','Light1','occupancy_sensor','thermometer', 'fire_alarm','brightness_sensor','motion_sensor','smoke_sensor','media_player','projector','Smart_TV','speaker','Light_controller']
# device_num = np.ones(len(device))
H = ['value','osversion','hardware_health','year','data_importance','physical','digital','realtime_threat','os_security','checkcycle','access_authentication','security_tactic','impact','exploitability','base']
values = [i for i in range(len(device))]
d1 = {v:k for v, k in zip(values, device)}
d2 = {v:k for k, v in zip(values, device)}

attack_id = [16,10,13]
critic_id = [16,17,19]
risk_metric = [12,13,14]
harm_metric = [1,2,5,6,7,8,9,10,11]
plt.rcParams['axes.unicode_minus'] = False
icons = {}
images = {k: PIL.Image.open(fname) for k, fname in icons.items()}
# print(d2)
# security = np.random.uniform(0.0, 1.0, size=(18,15))

def mock(mode,attack_id,critic_id):
    data_x = np.zeros((len(device),len(H)))
    label = np.ones(len(device))
    for i in range(len(device)):
        for j in range(len(H)):
            data_x[i][j] = random.uniform(0.85, 0.99)
    if mode == 'S0':
        return data_x,label,[]
    if mode == 'attack':
        for id in attack_id:
            for j in range(len(H)):
                data_x[id][j] = random.uniform(0.3, 0.4)
        return data_x,label,[]
    if (mode == 'S3') or (mode=='S2') or (mode=='S1'):
        if (mode == 'S2') or (mode=='S3'):
            # for id in critic_id:
            #     for j in risk_metric:
            #         data_x[id][j] = random.uniform(0.3,0.4)
            neigh, _ = neigh_basescore()
            num = random.randint(1, 7)
            negative_list = random.sample(values,num)
            negative_list0 = copy.deepcopy(negative_list)
            # negative_list = [16,10,13]
            for id in negative_list:
                for ne in neigh:
                    if id in ne and len(ne) > 0:
                        nei = ne[ne.index(id):]
                        for i in nei:
                            label[i] = 0
                            negative_list0.append(i)
                            for j in risk_metric:
                                data_x[i][j] = random.uniform(0.1, 0.2)
            negative_list0 = list(dict.fromkeys(negative_list0))
            # print("S23:negative_list------------------",negative_list0,label)
            if (mode == 'S2'):
                return data_x, label,negative_list0
        if (mode == 'S1') or (mode=='S3'):
            # for id in critic_id:
            #     for j in risk_metric:
            #         data_x[id][j] = random.uniform(0.3,0.4)
            num = random.randint(1, 7)
            negative_list = random.sample(values,num)

            # print("S1:negative_list------------------",negative_list)
            for id in negative_list:
                label[id] = 0
                for j in harm_metric:
                    data_x[id][j] = random.uniform(0.1, 0.2)
            return data_x, label,[]
    if mode == 'QT1':
        for j in harm_metric:
            data_x[17][j] = random.uniform(0.3, 0.4)
        return data_x, label, []
    if mode == 'QT2':
        for j in risk_metric:
            data_x[16][j] = random.uniform(0.1, 0.2)
        return data_x, label, []
    if mode == 'QT3':
        for j in risk_metric:
            data_x[16][j] = random.uniform(0.1, 0.2)

        for j in harm_metric:
            data_x[19][j] = random.uniform(0.3, 0.4)
        return data_x, label, []
    return data_x,label,[]

class dataset:

    def __init__(self,mode):
        # self.data_x = np.random.rand(len(device),len(H))
        self.data_x,self.data_y,self.attacked_list = mock(mode,attack_id,critic_id)
        security = np.array([[round(self.data_x[i][j]*Mag) for j in range(0,len(H))] for i in range(0, len(device))])
        # self.data_y = []
        # for i in range(len(self.data_x)):
        #     label = np.argmax(np.bincount(security[i]))
        #     if label == 0:
        #         label = 1
        #     self.data_y.append(label)


#从表中读取邻接矩阵
    def edge(self):
        data = np.loadtxt(excel_file, delimiter=",", dtype=int)
        self.data_edge = [[], []]
        self.adj_lists = defaultdict(set)
        for i in range(len(device)):
            for j in range(len(device)):
                if data[i][j] != 0:
                    self.data_edge[0].append(i)
                    self.data_edge[1].append(j)
                    self.adj_lists[i].add(j)


        return self.data_edge,self.adj_lists

    def x(self):
        return self.data_x

    def y(self):
        return self.data_y
    def attack(self):
        attack = np.zeros(len(device))
        for i in self.attacked_list:
            attack[i] = 1
        return attack
# print(data_edge)

def visual(adj_lists,label,attack_id):
    G = nx.Graph()
    file_name_list = os.listdir("../../icon1")
    weights = np.ones((len(label),len(label)))
    print(attack_id)
    neigh,_ = neigh_basescore()
    print(neigh)

    for id in attack_id:
        for ne in neigh:
            if id in ne and len(ne)>0:
                nei = ne[ne.index(id):]
                if len(nei) > 2:
                    for i in range(len(nei)-1):
                        weights[nei[i]][nei[i+1]]=2
                        weights[nei[i+1]][nei[i]] = 2
                if len(nei) == 2:
                    weights[nei[0]][nei[1]] = 2
                    weights[nei[1]][nei[0]] = 2

    # weights = np.ones((len(label), len(label)))
    # for i in range(21):
    #     for j in range(21):
    #         if weights[i][j]==2.0:
    #             print(str(i)+'---'+str(j))

    for file_name in file_name_list:
        file = str(file_name)[:-4]
        icons[file] = '../../icon1/'+file_name
    images = {k: PIL.Image.open(fname) for k, fname in icons.items()}
    for de in device:
        G.add_node(de,image=images[de],desc=str(label[d2[de]]))
        # G.add_node(de,image=images[de],desc="device:"+de+"\ndevice_id:"+str(device.index(de))+"\nsecurity score:"+str(label[d2[de]]))
        # G.add_node(de,image=images[de],desc="device:"+de+"\nsecurity score:"+str(label[d2[de]]))
        print("device:"+de+"\ndevice_id:"+str(device.index(de))+"\nsecurity score:"+str(label[d2[de]]))
        for neigh in list(adj_lists[d2[de]]):
            # if weights[d2[de]][neigh]==2.0:
            G.add_edge(de, d1[neigh])
            G[de][d1[neigh]]['weight'] = weights[d2[de]][neigh]

    fig, ax = plt.subplots(figsize=(7, 15))

    icon_size = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.07
    icon_center = icon_size / 2.0
    ax.set_title('graph_grade:'+str(round(sum(label)/len(label),2)), loc='center',fontdict={'family': 'serif', 'weight': 'bold', 'size': 18})
    pos = nx.spring_layout(G, seed=1)
    widths = nx.get_edge_attributes(G, 'weight')
    wid = list(widths.values())
    wid = [i*2 for i in wid]
    edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
    # 绘制连接
    # min_source_margin和 min_target_margin调节连接端点到节点的距离

    nx.draw_networkx_edges(G, pos=pos, ax=ax, arrows=True,arrowstyle="-",edgelist=edges, edge_color=weights,edge_cmap=plt.cm.Reds,
                       width=wid,
                       alpha=0.35, min_source_margin=15, min_target_margin=15, )

    # 给每个节点添加各自的图片
    pos_l = pos
    for n in G.nodes:

        xf, yf = ax.transData.transform(pos[n])  # ..data坐标.转..display坐标
        xa, ya = fig.transFigure.inverted().transform((xf, yf))  # dis.lay坐标..转..f.igure坐标
        a = plt.axes([xa - icon_center, ya - icon_center, icon_size, icon_size])
        a.imshow(G.nodes[n]["image"])
        a.axis("off")
        pos_l[n][1] -= icon_size*1.5
    # ax.set_facecolor('dimgray')
    # fig.set_facecolor('dimgray')
    node_labels = nx.get_node_attributes(G, 'desc')
    nx.draw_networkx_labels(G, pos=pos_l, ax=ax, labels=node_labels,alpha=1, font_size=20,font_color='green',bbox=dict(edgecolor='white', alpha=0.1,boxstyle='round,pad=0.2'))
    plt.show()


def create_g(mode):
    data = dataset(mode)
    x = th.Tensor(data.x())
    edge,adj_lists = data.edge()
    edge = th.LongTensor(edge)
    y = th.LongTensor(data.y())
    g = dgl.graph((edge[0], edge[1]))
    g.ndata['feat'] = x
    g.ndata['label'] = y
    attack_id = th.LongTensor(data.attack())
    g.ndata['attack'] = attack_id
    graph = g.remove_self_loop()
    g = graph.add_self_loop()

    return g,adj_lists


def QTset():
    S0_set = []
    for i in range(2):
        g, adj_lists = create_g("S0")
        S0_set.append(g)
    for i in range(2):
        g, adj_lists = create_g("QT1")
        S0_set.append(g)
    for i in range(2):
        g, adj_lists = create_g("S0")
        S0_set.append(g)
    for i in range(4):
        g, adj_lists = create_g("QT2")
        S0_set.append(g)
    for i in range(2):
        g, adj_lists = create_g("QT3")
        S0_set.append(g)
    return S0_set

def IOTAset(a):
    S0_set = []
    for i in range(int(a/5)):
        g, adj_lists = create_g("S2")
        S0_set.append(g)
    for i in range(int(a*4/5)):
        g, adj_lists = create_g("S3")
        S0_set.append(g)
    return S0_set

def SVMset(a):
    S0_set = []
    for i in range(int(a/5)):
        g, adj_lists = create_g("S2")
        S0_set.append(g)
    for i in range(int(a*4/5)):
        g, adj_lists = create_g("S3")
        S0_set.append(g)
    return S0_set

def Allset(a):
    S0_set = []
    positive = 0
    for i in range(int(a * 1 / 5)):
        print("S0",i)
        g, adj_lists = create_g("S0")
        positive+=np.sum(g.ndata['label'].numpy()==1)
        S0_set.append(g)
    for i in range(int(a * 1 / 5)):
        print("S1",i)
        g, adj_lists = create_g("S1")
        positive+=np.sum(g.ndata['label'].numpy()==1)

        S0_set.append(g)
    for i in range(int(a*2 / 5)):
        print("S2",i)
        g, adj_lists = create_g("S2")
        positive+=np.sum(g.ndata['label'].numpy()==1)

        S0_set.append(g)
    for i in range(int(a * 2 / 5)):
        print("S3",i)
        g, adj_lists = create_g("S3")
        positive+=np.sum(g.ndata['label'].numpy()==1)

        S0_set.append(g)
    print('positive---------:',positive)
    return S0_set

def S0set(a):
    S0_set = []
    for i in range(a):
        g, adj_lists = create_g("S0")
        S0_set.append(g)

    return S0_set

def S1set(a):
    train_set = []
    for i in range(a):
        g,adj_lists = create_g('S1')
        train_set.append(g)

    return train_set

def S2set(a):
    S2_set = []
    for i in range(a):
        g, adj_lists = create_g("S2")
        S2_set.append(g)

    return S2_set

def S3set(a):
    S3_set = []
    for i in range(a):
        g, adj_lists = create_g("S3")
        S3_set.append(g)

    return S3_set


def load_dataset():
    train_dataloader = trainset(5)
    valid_dataloader = valset()
    test_dataloader = testset()
    eval_train_dataloader = train_dataloader
    num_features = len(H)
    num_classes = 5
    return train_dataloader, valid_dataloader, test_dataloader, eval_train_dataloader, num_features, num_classes

def agg_weights():
    neigh, basescore = neigh_basescore()
    neighs = []
    weights = []
    for node in device:
        neighbor = []
        weight = []
        bucket = np.zeros(len(device))
        id = device.index(node)
        for ne in neigh:
            if id in ne:
                nei = ne[:ne.index(id)]
                for n in nei:
                    if bucket[n] != 0:
                        bucket[n] += basescore[neigh.index(ne)]/100 * (len(nei)-nei.index(n))*2/(len(nei)*(len(nei)+1))
                        bucket[n] /= 2
                    else:
                        bucket[n] += basescore[neigh.index(ne)]/100 * (len(nei)-nei.index(n))*2/(len(nei)*(len(nei)+1))
        bucket=list(bucket)
        for b in bucket:
            if b != 0:
                neighbor.append(bucket.index(b))
                weight.append(b)
        summ = sum(weight)
        for i in range(len(weight)):
            weight[i] = weight[i]/summ
        neighs.append(neighbor)
        weights.append(weight)
        # print(id,neighbor)

    # for de in neigh:
    #     weight = []
    #     if len(de) > 0:
    #         for ne in de:
    #             weight.append((len(de)-de.index(ne))*2/(len(de)*(len(de)+1)))
    #             # weight.append((de.index(ne)+1)*2*basescore[neigh.index(de)]/(len(de)*(len(de)+1)*200))
    #     # elif len(de) == 1:
    #     #     weight.append(basescore[neigh.index(de)]/100)
    #     # print(weight,sum(weight))
    #     weights.append(weight)

    return neighs,weights



if __name__ == "__main__":
    # train_dataloader = Allset(100)
    # # print(train_dataloader[0].ndata['feat'])
    # save_graphs("../datasets/test/grade.bin", train_dataloader)
    #
    # trains = load_graphs("../datasets/test/grade.bin")[0]
    # # positive = 0
    # for g in trains:
    #     print(g.ndata['feat'])

    g, adj_lists = create_g("S0")
    print(list(adj_lists[1]))
    nei,_ = agg_weights()
    print(nei)
    # for subgraph in trains:
    #     print(subgraph.ndata['feat'])
    # valid_dataloader = valset()
    # test_dataloader = testset()
    # eval_train_dataloader = train_dataloader
    # g,adj_lists = create_g()
    # agg_weights()
    # time0=[]
    # for j in range(30):
    #     time0.append(random.uniform(0.8000, 0.8500))
    # input1 = mock('test',attack_id,critic_id)
    # for i in range(120):
    #     input = mock('normal',attack_id,critic_id)
    #     input1.append(input)
    # print(input1)
    # np.savetxt('../experiments/excel/input1.csv', input1, delimiter=',')
    # grades = [92.50 ,91.90 ,89.60 ,91.90 ,90.00 ,93.30 ,91.80 ,91.60 ,91.60 ,92.60 ,91.20 ,92.50 ,91.80 ,92.90 ,91.70 ,93.30 ,90.40 ,91.80 ,91.50 ,92.80 ,91.50]
    # visual(adj_lists,label=g.ndata['label'].tolist(),attack_id=attack_id)
    # visual(adj_lists, label=grades, attack_id=[])

