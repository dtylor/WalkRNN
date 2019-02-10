from n2v import node2vec
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import networkx as nx
import pandas as pd
import numpy as np
from graphwave import graphwave
from graphwave.utils import utils
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def get_structural_signatures(networkXGraph):
    """
    Get structural embeddings using GraphWave.

    Learns structural signatures of each node using GraphWave, and adds these learned structures to the node attributes on the NetworkX graph.

    Example:

    Attributes:
        networkXGraph (networkx.classes.graph.Graph): A graph containing nodes, edges, and their attributes. Nodes must include an attribute called 'component'.

    Returns: a NetworkX graph where structural embeddings are added as `structure` node attributes.

    """
    nb_clust = 4
    trans_data_all = []
    n_components = 4
    keys = []
    nodes_list = []

    components = pd.DataFrame([{"node": k, "component": v} for k, v in nx.get_node_attributes(
        G=networkXGraph, name='component').items()]).groupby('component')['node'].apply(list).to_dict()

    for subgraph_id, nodes in components.items():
        subgraph = networkXGraph.subgraph(nodes)
        chi, heat_print, taus = graphwave.graphwave_alg(
            subgraph, np.linspace(0, 100, 25), taus='auto', verbose=True)
        if len(subgraph.nodes) < n_components:
            print("Omitting graph " + str(subgraph_id) + " with node count: " +
                str(len(subgraph.nodes)) + " < " + str(n_components))
        else:
            pca = PCA(n_components)
            trans_data = pca.fit_transform(StandardScaler().fit_transform(chi))
            trans_data_all = trans_data_all + trans_data.tolist()
            l = [subgraph_id] * len(subgraph.nodes)
            keys = keys + l
            nodes_list = nodes_list + list(subgraph.nodes())

    km = KMeans(n_clusters=nb_clust)
    km.fit(trans_data_all)
    labels_pred = km.labels_

    out = pd.DataFrame(labels_pred.astype(int), index=nodes_list)
    structure_labels = out[0].to_dict()
    nx.set_node_attributes(G=networkXGraph, values=structure_labels, name='structure')

    return networkXGraph

def walk_as_string(networkXGraph, graphComponentLabels, featuresToOmit={"nodes":[], "edges":[]}):
    """
    Generate random walks over a graph.

    Create a language to define nodes/edges and their characteristics, and generate random walks in this language.

    Example:

    Attributes:
        networkXGraph (networkx.classes.graph.Graph): A graph containing nodes, edges, and their attributes. Nodes must include an attribute called 'component'.
        graphComponentLabels (dict): A dictionary where the keys are graph component ids, and the value is the label
            {
                1: -1,
                2: 1,
                3: -1
            }
        featuresToOmit (dict): A dictionary of node and edge features to be omitted

    // return a dataframe where each row contains {`walk`, `graph_label (y)`}

    Returns: a DataFrame containing each walk, and the associated graph label.
    """

    return True
