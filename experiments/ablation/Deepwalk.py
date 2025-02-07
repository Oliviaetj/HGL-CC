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
import random
from datasets.graph_gen import create_g
random_state = 2020


class DeepWalk(object):
    def __init__(self, graph, walk_length, num_walks, walkers=1, verbose=0, random_state=None):
        self.G = graph
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.walkers = walkers
        self.verbose = verbose
        self.w2v = None
        self.embeddings = None
        self.random_state = (lambda x: x if x else 2020)(random_state)
        self.dataset = self.get_train_data(self.walk_length, self.num_walks, self.walkers, self.verbose)

    def fit(self, embed_size=128, window=5, n_jobs=3, epochs=5, **kwargs):
        kwargs["sentences"] = self.dataset
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = embed_size
        kwargs["sg"] = 1  # skip gram
        kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
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

    def simulate_walks(self, nodes, num_walks, walk_length, ):
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                walks.append(self.deep_walk(walk_length=walk_length, start_node=v))
        return walks

    def deep_walk(self, walk_length, start_node):

        G = self.G

        walk = [start_node]
        while len(walk) < self.walk_length:
            current_node = walk[-1]
            current_nerghbors = list(G.neighbors(current_node))
            if len(current_nerghbors) > 0:
                walk.append(random.choice(current_nerghbors))
            else:
                break
        return walk

    def get_embeddings(self):
        if self.w2v:
            self.embeddings = {}
            for node in self.G.nodes():
                self.embeddings[node] = self.w2v.wv[node]
            return self.embeddings
        else:
            print("Please train the model first")
            return None

def simulate_walks(node, num_walks, walk_length,adj_lists):
    walks = []
    for _ in range(num_walks):
        walks.extend(deep_walk(walk_length=walk_length, start_node=node,adj_lists=adj_lists))
    return walks

def deep_walk(walk_length, start_node,adj_lists):
    walk = []
    while len(walk) < walk_length:
        current_node = start_node
        current_nerghbors = list(adj_lists[current_node])
        if len(current_nerghbors) > 0:
            walk.append(random.choice(current_nerghbors))
        else:
            break
    return walk

def sample_d():
    g, adj_lists = create_g("S0")
    nodes = [i for i in range(21)]
    neigh = []
    for node in nodes:
        neigh.append(list(set(simulate_walks(node,10,1,adj_lists))))
    return neigh
# model = DeepWalk(G, walk_length=10, num_walks=80, walkers=3, verbose=0, random_state=random_state)
# embeddings = model.get_embeddings()

if __name__ == "__main__":
    nei = aggregate1()
    print(nei)