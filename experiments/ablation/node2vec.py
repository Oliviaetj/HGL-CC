import sklearn
import gensim
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.manifold import TSNE
from joblib import Parallel, delayed
from gensim.models import Word2Vec
import numpy as np
import itertools
import dgl

import random
from datasets.graph_gen import create_g

random_state = 2020

def create_alias_table(probs):
    """
    :param probs: sum(probs)=1
    :return: accept,alias
    """
    L = len(probs)
    accept, alias = [0] * L,  [0] * L
    small, large = [], []
    for i, prob in enumerate(probs):
        accept[i] = prob * L
        if prob < 1.0:
            small.append(i)
        else:
            large.append(i)

    while small and large:
        small_idx, large_idx = small.pop(), large.pop()
        accept[small_idx] = probs[small_idx]
        alias[small_idx] = large_idx
        probs[large_idx] = probs[large_idx] - (1 - probs[small_idx])
        if probs[large_idx] < 1.0:
            small.append(large_idx)
        else:
            large.append(large_idx)

    return accept, alias


def alias_sample(accept, alias):
    """
    :param accept:
    :param alias:
    :return: sample index
    """
    N = len(accept)
    i = int(np.random.random()*N)
    r = np.random.random()
    if r < accept[i]:
        return i
    else:
        return alias[i]


class Node2Vec(object):
    def __init__(self, graph, p, q, walk_length, num_walks, workers=1, verbose=0, random_state=None):
        self.G = graph

        self.p = p
        self.q = q
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.workers = workers
        self.alias_nodes = None
        self.alias_edges = None
        self.verbose = verbose

        self.w2v = None
        self.embeddings = None
        self.random_state = (lambda x: x if x else 2020)(random_state)

        self.preprocess_transition_probs()
        self.dataset = self.get_train_data(self.walk_length, self.num_walks, self.workers, self.verbose)

    def fit(self, embed_size=128, window=5, n_jobs=3, epochs=5, **kwargs):
        kwargs["sentences"] = self.dataset
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = embed_size
        kwargs["sg"] = 1  # skip gram
        kwargs["hs"] = 0  # node2vec not use Hierarchical Softmax
        kwargs["workers"] = n_jobs
        kwargs["window"] = window
        kwargs["iter"] = epochs
        kwargs["seed"] = self.random_state

        self.w2v = Word2Vec(**kwargs)

    def get_train_data(self, walk_length, num_walks, workers=1, verbose=0):
        if num_walks % workers == 0:
            num_walks = [num_walks // workers] * workers
        else:
            num_walks = [num_walks // workers] * workers + [num_walks % workers]

        nodes = list(self.G.nodes())

        results = Parallel(n_jobs=workers, verbose=verbose)(
            delayed(self.simulate_walks)(nodes, num, walk_length) for num in num_walks
        )

        dataset = list(itertools.chain(*results))
        return dataset

    def simulate_walks(self,node, num_walks, walk_length, adj_lists):
        walks = []
        for _ in range(num_walks):
            walks.extend(self.node2vec_walk(walk_length=walk_length, start_node=node, adj_lists=adj_lists))
        return walks

    def node2vec_walk(self, walk_length, start_node,adj_lists):

        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = []
        while len(walk) < self.walk_length:
            current_node = start_node
            current_nerghbors = list(adj_lists[current_node])
            if len(current_nerghbors) > 0:
                if len(walk) == 1:
                    walk.append(
                        current_nerghbors[alias_sample(alias_nodes[current_node][0], alias_nodes[current_node][1])]
                    )
                else:
                    previous_node = walk[-2]
                    edge = (previous_node, current_node)
                    next_node = current_nerghbors[
                        alias_sample(alias_edges[edge][0], alias_edges[edge][1])
                    ]
                    walk.append(next_node)
            else:
                break
        return walk

    def get_alias_edge(self, t, v):
        """
        2阶随机游走，顶点间的转移概率
        :param t: 上一顶点
        :param v: 当前顶点
        """
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for x in G.neighbors(v):
            weight = G[v][x].get('weight', 1.0)  # w_vx，无权图权重设为1
            if x == t:  # d_tx == 0
                unnormalized_probs.append(weight / p)
            elif G.has_edge(x, t):  # d_tx == 1
                unnormalized_probs.append(weight)
            else:  # d_tx > 1
                unnormalized_probs.append(weight / q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [
            float(u_prob) / norm_const for u_prob in unnormalized_probs]

        return create_alias_table(normalized_probs)

    def preprocess_transition_probs(self):
        """
        在随机游走之前进行初始化
        """
        G = self.G
        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [
                G[node][nbr].get('weight', 1.0) for nbr in G.neighbors(node)]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [
                float(u_prob) / norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = create_alias_table(normalized_probs)

        alias_edges = {}
        for edge in G.edges():
            alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])

        self.alias_edges = alias_edges
        self.alias_nodes = alias_nodes

    def get_embeddings(self):
        if self.w2v:
            self.embeddings = {}
            for node in self.G.nodes():
                self.embeddings[node] = self.w2v.wv[node]
            return self.embeddings
        else:
            print("Please train the model first")
            return None




def sample_n():
    g1, ad = create_g("S0")
    nodes = [i for i in range(21)]
    neigh = []
    neighs = dgl.sampling.node2vec_random_walk(g1, nodes, 1, 1, walk_length=4)
    for ne in neighs.tolist():
        neigh.append(list(set(ne)))

    return neigh
# model = DeepWalk(G, walk_length=10, num_walks=80, walkers=3, verbose=0, random_state=random_state)
# embeddings = model.get_embeddings()

if __name__ == "__main__":
    nei = sample_n()
    print(nei)
