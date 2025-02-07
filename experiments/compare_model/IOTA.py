from IoT_Attack_Path.src.NetGen import neigh_basescore
from datasets.graph_gen import create_g,S2set,S1set,S0set,S3set
from sklearn.metrics import classification_report
from dgl.data.utils import load_graphs
import sys
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# 初始化漏洞依赖图
def initialize_vulnerability_dependency_graph(neigh):
    # 从已知的漏洞数据库中获取漏洞信息
    path = []
    # 构建漏洞依赖图
    for p in neigh:
        for d in p:
            path.append(d)
    path = list(dict.fromkeys(path))

    return path




# 主函数
def IOTA(set):
    neigh, _ = neigh_basescore()
    AttackPath = initialize_vulnerability_dependency_graph(neigh)
    j=1
    y_pred = []
    y_label = []
    for subgraph in set:
        print("\r", end="")
        print(j)
        print("Test progress: {}%: ".format(j * 1 / 24), "▋" * (j // 2), end="")
        j = j + 1
        sys.stdout.flush()

        time.sleep(0.05)
        safe = np.ones(len(neigh))

        label = subgraph.ndata['label'].numpy().tolist()

        attack_list = subgraph.ndata['attack'].numpy().tolist()
        c = 0
        for i in range(len(attack_list)):
            if attack_list[i] == 1:
                safe[i] = 0
                c = 1
        # if c == 0:
        #     for d in AttackPath:
        #         safe[d] = 0
        y_label += label
        y_pred += safe.tolist()
    return y_label,y_pred



if __name__ == "__main__":
    # set = S3set(20)
    set = load_graphs("../../datasets/test/grade.bin")[0]
    y_label,y_pred = IOTA(set)

    accuracy = classification_report(y_label, y_pred, output_dict=True)['accuracy']
    s = classification_report(y_label, y_pred, output_dict=True)['weighted avg']
    precision = s['precision']
    recall = s['recall']
    f1_score = s['f1-score']
    print(accuracy, precision, recall, f1_score)
