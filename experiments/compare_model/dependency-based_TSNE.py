from experiments.compare_model.IOTA import IOTA,initialize_vulnerability_dependency_graph
from IoT_Attack_Path.src.NetGen import neigh_basescore
from datasets.graph_gen import S1set,create_g,S2set,S3set,S0set,SVMset
import sklearn.metrics as metrics
from sklearn.metrics import classification_report
import numpy as np
import warnings
from dgl.data.utils import load_graphs
import sys
import time
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    # Example usage
    # set = S2set(20)
    set = load_graphs("../../datasets/test/grade.bin")[0]

    y_label,y_pred = IOTA(set)
    i = 0
    for subgraph in set:
        print("\r", end="")
        print(i)
        print("Test progress: {}%: ".format(i * 1 / 24), "▋" * (i // 2), end="")
        sys.stdout.flush()

        time.sleep(0.05)
        feature_data = subgraph.ndata['feat']
        for j in range(21):
            for s in [12,13,14]:
                if feature_data[j][s] < 0.6:
                    y_pred[i*21+j] = 0
        i = i+1
    accuracy = metrics.accuracy_score(y_label, y_pred)
    s = classification_report(y_label, y_pred, output_dict=True)['weighted avg']
    precision = s['precision']
    recall = s['recall']
    f1_score = s['f1-score']
    print(accuracy, precision, recall, f1_score)

