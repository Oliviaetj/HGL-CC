import random
import warnings
import sys
import time
from sklearn.metrics import classification_report
import numpy as np
from dgl.data.utils import load_graphs

from datasets.graph_gen import S0set,create_g,S2set,S1set,S3set
warnings.filterwarnings("ignore")

# 信息安全风险评估方法伪代码
def calculate_risk_score(indicators, weights):
    """
    计算信息安全风险得分。

    :param indicators: 定性指标的评估值列表
    :param weights: 指标权重列表
    :return: 风险得分
    """
    risk_score = 0
    for i in range(len(indicators)):
        risk_score += indicators[i] * weights[i]
    return risk_score


def AHP(features):
    # 假设有3个定性指标
    qualitative_indicators = [0.8, 0.6, 0.9]

    # 假设权重为 [0.4, 0.3, 0.3]
    weights = []
    for i in range(len(features[0])):
        weights.append(random.randint(1,100))
    s = sum(weights)
    for i in range(len(features[0])):
        weights[i] = weights[i] / s
    # 计算风险得分
    risk_score = []
    for i in range(21):
        indicators = features[i]
        score = 0
        for j in range(len(indicators)):
            score += indicators[j] * weights[j]
        score = score * 2000 / len(indicators)

        risk_score.append(score)

    return risk_score

def QL():
    # train_dataloader = load_graphs("../../datasets/test/grade.bin")[0]
    train_dataloader = S3set(20)
    y_label = []
    y_pred = []
    i = 1
    for subgraph in train_dataloader:
        print("\r", end="")
        print(i)
        print("Test progress: {}%: ".format(i * 1 / 24), "▋" * (i // 2), end="")
        i = i + 1
        sys.stdout.flush()

        time.sleep(0.05)
        feature_data = subgraph.ndata['feat']
        label = subgraph.ndata['label'].numpy().tolist()
        y_label += label
        # raw_label = graph.ndata['label']
        # print(feature_data)

        AHP_grades = AHP(feature_data.numpy().tolist())
        safety = np.ones(len(AHP_grades))
        for g in range(len(AHP_grades)):
            if AHP_grades[g] < 60:
                safety[g] = 0
        safety = safety.tolist()
        y_pred += safety

    accuracy = classification_report(y_label, y_pred, output_dict=True)['accuracy']
    s = classification_report(y_label, y_pred, output_dict=True)['weighted avg']
    precision = s['precision']
    recall = s['recall']
    f1_score = s['f1-score']
    print(accuracy, precision, recall, f1_score)

def QT():
    train_dataloader = load_graphs("../../datasets/test/QT.bin")[0]
    # train_dataloader = trainset(10)
    grade = []
    for subgraph in train_dataloader:


        feature_data = subgraph.ndata['feat']

        AHP_grades = AHP(feature_data.numpy().tolist())
        AHP_grades = [x / 1.1557598918434497 for x in AHP_grades]
        grade.append(AHP_grades[16:20])

    return grade



if __name__ == "__main__":
    QL()
    # grade = QT()
    # np.savetxt('../../experiments/excel/AHP_grade.csv', grade, delimiter=',')
    # train_dataloader = S0set(20)
    #
    # # train_dataloader = load_graphs("../../datasets/test/grade.bin")[0]
    # positive = 0
    # for g in train_dataloader:
    #     print(g.ndata['label'])
    #     positive += np.sum(g.ndata['label'].numpy() == 1)
    # print(positive)