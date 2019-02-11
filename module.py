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

def _get_components(networkXGraph):
    return pd.DataFrame([{"node": k, "component": v} for k, v in nx.get_node_attributes(
        G=networkXGraph, name='component').items()]).groupby('component')['node'].apply(list).to_dict()


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
    n_components = 6
    keys = []
    nodes_list = []

    trans_data = []

    components = _get_components(networkXGraph)

    heat_signatures = []

    for subgraph_id, nodes in components.items():
        subgraph = networkXGraph.subgraph(nodes)
        chi, heat_print, taus = graphwave.graphwave_alg(
            subgraph, np.linspace(0, 100, 25), taus='auto', verbose=True)
        if len(subgraph.nodes) < n_components:
            print("Omitting graph " + str(subgraph_id) + " with node count: " +
                str(len(subgraph.nodes)) + " < " + str(n_components))
        else:
            heat_signatures += chi.tolist()
            nodes_list += nodes

    pca = PCA(n_components = n_components)
    trans_data_all = pca.fit_transform(StandardScaler().fit_transform(np.array(heat_signatures)))
    km = KMeans(n_clusters = nb_clust).fit(trans_data_all)

    labels_pred = km.labels_

    out = pd.DataFrame(labels_pred.astype(int), index=nodes_list)
    structure_labels = out[0].to_dict()
    nx.set_node_attributes(G=networkXGraph, values=structure_labels, name='structure')

    return networkXGraph, pca, km

def walk_as_string(networkXGraph, graphComponentLabels, featuresToUse={"nodes":['label', 'structure'], "edges":['label']}):
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
        featuresToUse (dict): A dictionary of node and edge features to be expressed in the walk sequence

    // return a dataframe where each row contains {`walk`, `graph_label (y)`}

    Returns: a DataFrame containing each walk, and the associated graph label.
    """
    
    n2vG = node2vec.Graph(nx_G=networkXGraph, is_directed=False, p=1, q=.7)
    n2vG.preprocess_transition_probs()
    num_walks = 20
    walk_length = 30
    walks = n2vG.simulate_walks(num_walks, walk_length)

    def expressNode(node_idx):
        node = networkXGraph.nodes[node_idx]
        result = " ".join([str(node[attribute])
                           for attribute in featuresToUse['nodes']])
        return result
    
    def expressEdge(src_node, dst_node):
        edge = networkXGraph.edges[src_node, dst_node]
        result = " ".join([str(edge[attribute])
                           for attribute in featuresToUse['edges']])
        return result

    sorted_walks = pd.DataFrame(walks).sort_values(0)

    walks = [list(a) for a in sorted_walks.as_matrix()]

    walks_as_words = [" ".join([expressNode(walk[step]) + " " + expressEdge(walk[step], walk[step+1]) + " " +
                       expressNode(walk[step+1]) for step in range(len(walk) - 1)]) for walk in walks]

    result = pd.DataFrame({"walk": walks_as_words, "start_node": np.array(walks)[:,0]})

    result['component'] = result['start_node'].map(nx.get_node_attributes(networkXGraph, name='component'))
    result['label'] = result['component'].map(graphComponentLabels)

    result = result[['walk', 'label', 'start_node', 'component']]

    return result
