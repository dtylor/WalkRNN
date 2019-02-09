from n2v import node2vec
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import networkx as nx
import pandas as pd
import numpy as np
from graphwave import graphwave
from graphwave.utils import utils

path_to_files = "../datasets/MUTAG/"

G = nx.read_edgelist(path_to_files + "MUTAG_A.txt", delimiter=',', nodetype=int, encoding="utf-8")
nodes = pd.read_csv(path_to_files + "MUTAG_graph_indicator.txt", header=None)
nodes.index += 1
nodes = nodes.reset_index().rename(columns={"index":"node", 0:"component"})

subgraphs = nodes.groupby('component')['node'].apply(list) \
    .map(lambda x: G.subgraph(x)).to_dict()

node_labels = pd.read_csv(path_to_files + "MUTAG_node_labels.txt", header=None)
node_labels.index += 1
node_labels = dict(node_labels[0])
nodes['node_label'] =  nodes.node.map(node_labels)

component_labels = pd.read_csv(path_to_files + "MUTAG_graph_labels.txt", header=None)
component_labels.index += 1
component_labels = dict(component_labels[0])
nodes['component_label'] = nodes.component.map(component_labels)

nb_clust = 6
trans_data_all = []
n_components = 4
keys = []
nodes_list = []

# Learning structural signatures using GraphWave 
for key, graph in subgraphs.items():
    chi, heat_print, taus = graphwave.graphwave_alg(
        graph, np.linspace(0, 100, 25), taus='auto', verbose=True)
    if len(graph.nodes) < n_components:
        print("Omitting graph " + key + " with node count: " +
              str(len(graph.nodes)) + " < " + n_components)
    else:
        # Reduce dimensionality to num_nodes x n_components
        pca = PCA(n_components)
        # Scale to accentuate differences
        trans_data = pca.fit_transform(StandardScaler().fit_transform(chi))
        trans_data_all = trans_data_all + trans_data.tolist()
        l = [key] * len(graph.nodes)
        keys = keys + l
        nodes_list = nodes_list + list(graph.nodes())

km = KMeans(n_clusters=nb_clust)
km.fit(trans_data_all)
labels_pred = km.labels_

nodes['graphwave_label'] = labels_pred

nodes['node_label'] = nodes.node_label.map({0: "C",
                                            1: "N",
                                            2: "O",
                                            3: "F",
                                            4: "I",
                                            5: "Cl",
                                            6: "Br"})


e = pd.read_csv(path_to_files+"MUTAG_edge_labels.txt", header=None)
edge_words = e[0].apply(lambda x: "e"+str(x))
edges = pd.read_csv(path_to_files + "MUTAG_A.txt", header=None)
edge_words.index += 1
edges.index += 1
edges['label'] = edge_words

# TODO: label should incorporate node-specific attribute values (OneHot encode categoricals, and cluster continuous)
nodes['label'] = nodes.node_label + " " + nodes.graphwave_label.astype(str)
nodes_indexed = nodes.set_index('node')['label'].T.to_dict()

for start, end, edge in G.edges(data=True):
    edge['weight'] = 1

n2vG = node2vec.Graph(nx_G=G, is_directed=False, p=1, q=.7)
n2vG.preprocess_transition_probs()
num_walks = 20
walk_length = 30
walks = n2vG.simulate_walks(num_walks, walk_length)

edges = edges.set_index([0, 1]).to_dict()['label']

processed_walks = []
for walk in walks:
    rw = ""
    start_node = walk[0]
    rw += nodes_indexed[start_node] + " "
    for step in range(len(walk) - 1):
        start_node = walk[step]
        end_node = walk[step + 1]
        edge_taken = edges[(start_node, end_node)]
        rw += str(edge_taken) + " " + nodes_indexed[end_node] + " "
        step + 1
    processed_walks += [rw]

print(processed_walks)