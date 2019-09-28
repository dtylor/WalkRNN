from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from os import listdir, sep
import networkx as nx
import module


def copy_graph_nodes_edges(inGraph):
    if (type(inGraph) == nx.Graph):
        G_copy = nx.Graph()
    elif (type(inGraph) == nx.DiGraph):
        G_copy = nx.DiGraph()

    G_copy.add_nodes_from(inGraph.nodes(data=False))
    G_copy.add_edges_from(inGraph.edges(data=False))
    return G_copy


# TODO: parameterize kmeans clusters numbers by attribute name or add elbow method
def process_attributes(FG, G_copy, current_vocab_size=0, is_node=True,nb_clust=6):
    if is_node:
        features = list(set([z for x in list(FG.nodes(data=True)) for z in x[1].keys()]))
        if 'component' in features:
            features.remove('component')
            nx.set_node_attributes(G=G_copy, values=nx.get_node_attributes(FG, 'component'), name='component')
    else:
        features = list(set([z for x in list(FG.edges(data=True)) for z in x[2].keys()]))

    for feat in features:

        if is_node:
            feat_tmp = nx.get_node_attributes(FG, feat)
        else:
            feat_tmp = nx.get_edge_attributes(FG, feat)
        uniq = set(val for val in feat_tmp.values())

        types = set([type(val) for val in uniq])

        if (types.pop() == float):
            # apply kmeans to floats by default
            df = pd.DataFrame.from_dict(feat_tmp, orient='index')
            df.columns = [feat]
            attributes, kmeans_models = transform_features(df,nb_clust)
            feat_tmp = attributes[feat].to_dict()
            uniq = set(val for val in feat_tmp.values())

        dtmp = {ni: indi for indi, ni in enumerate(uniq)}
        vocab = {k: dtmp[v] + current_vocab_size for k, v in feat_tmp.items()}

        current_vocab_size = current_vocab_size + len(uniq)
        if is_node:
            nx.set_node_attributes(G=G_copy, values=vocab, name=feat)
        else:
            nx.set_edge_attributes(G=G_copy, values=vocab, name=feat)
    return G_copy, current_vocab_size


def transform_graph(G_prop,params={'num_kmeans_clusters': 4, "num_pca_components": 4, "num_batch":500, 'num_att_kmeans_clusters': 6}):
    """
    Transforms input networkX property graph into another graph with properties ready for language model analysis.
        Quantitative variables are made word-like/ categorical
        Attribute class vocabularies are separated and made uniform
        Node structural characteristics are described by word-like/ categorical variables

    Attributes:
        G_prop (Graph): A networkX graph containing original properties.

    Returns: A networkX graph with the same graph structure and transformed properties, prepared for language model analysis.

    """
    nb_att_clust = params['num_att_kmeans_clusters']
    G_copy = copy_graph_nodes_edges(G_prop)
    w = nx.get_edge_attributes(G_prop, 'weight')
    nx.set_edge_attributes(G=G_copy, values=w, name='weight')

    current_vocab_size = 0
    # Learn structural signatures of each node and apply to node as an attribute to the original graph
    G_struct, pca, kmeans = module.get_structural_signatures(G_prop, 0,
                                                             params)
    
    # determine vocab representation of attributes of nodes then edges in the original graph and add to the copied graph
    G_new_att, current_vocab_size = process_attributes(G_struct, G_copy, current_vocab_size, is_node=True,nb_clust=nb_att_clust)

    G_new_att, current_vocab_size = process_attributes(G_struct, G_new_att, current_vocab_size, is_node=False,nb_clust=nb_att_clust)
    return G_new_att, current_vocab_size


def transform_features(features_df, nb_clust=6):
    """
    Transforms mixed-type features into categoricals.

    Parses through dtypes of each column, runs KMeans clustering on all columns of type float64, mapping the float values to discrete cluster labels.
    
    Attributes:
        features_df (DataFrame): A dataframe containing feature columns and node (or edge) rows.

    Returns: A DataFrame containing all categorical features, as well as the fitted KMeans model.

    """

    new_features_df = features_df.copy()

    dtypes = new_features_df.dtypes
    # int's are being read in as floats >:(
    dtypes = dtypes[dtypes == 'float64']

    kmeans_models = {}

    for col_name, col_type in dtypes.items():
        kmeans_models[col_name] = KMeans(
            n_clusters=nb_clust).fit(features_df[col_name].values.reshape(-1, 1))
        transformed_column = kmeans_models[col_name].labels_
        new_features_df[col_name] = transformed_column

    return new_features_df, kmeans_models


def load_graph_kernel_graph(path_to_dataset_dir,dataset=None, mappings={}):
    """
    Loads Graph Kernel dataset into a NetworkX graph.

    Loads a dataset into a NetworkX graph, adding the node labels and component labels to each NetworkX node.

    Attributes:
        path_to_dataset_dir (str): Path to the directory containing the dataset files.
        dataset (str): Dataset-specific prefix if more than one dataset is in the directory. (optional)
        mappings (dict): A dictionary describing how to map integer node/edge labels to domain-specific labels (as per dataset README). Also describes how to interpret columns in node_attributes.txt
            AIDS dataset example:
                mappings = {
                    "node_labels": [{
                        "0": "C",
                        "1": "O",
                        "2": "N",
                        "3": "Cl"
                    }, {
                        ...mappings for column 2 of node_labels
                    }],
                    "edge_labels": {
                        "0": "1",
                        "1": "2",
                        "2": "3"
                    },
                    "node_attributes": ["chem", "charge", "x", "y"]
                }


    Returns: A NetworkX graph with node and component labels applied to each node.

    """
    files = listdir(path_to_dataset_dir)
    if path_to_dataset_dir[-1] != sep:
        path_to_dataset_dir += sep
    adjacency = [s for s in files if "_A" in s]
    if len(adjacency) == 0:
        raise RuntimeWarning("No files found!")
    elif len(adjacency) > 1:
        raise RuntimeWarning(
            "More than one GraphKernel dataset in directory; specify by passing a value for the dataset argument.")
        dataset = adjacency[0].split("_A")[0]
        print("Using " + dataset.split("_A")[0])
    else:
        dataset = adjacency[0].split("_A")[0]

    # Load graph kernel csv file(s) into a directed networkx Graph
    G = nx.read_edgelist(path_to_dataset_dir + dataset + "_A.txt",
                         delimiter=',', nodetype=int, encoding="utf-8")

    # Components
    components = pd.read_csv(
        path_to_dataset_dir + dataset + "_graph_indicator.txt", header=None)

    components.index += 1
    components = components.rename(columns={0: "component"})

    components = components.component.to_dict()

    nx.set_node_attributes(G=G, values=components, name='component')


    # Node Labels
    node_labels = pd.read_csv(path_to_dataset_dir +
                              dataset + "_node_labels.txt", header=None)
    node_labels.index += 1

    for column in range(len(node_labels.columns)):
        this_label = node_labels[column].to_dict()
        nx.set_node_attributes(
            G=G, values=this_label, name='label_' + str(column))

    # Edge Labels
    if dataset + "_edge_labels.txt" in listdir(path_to_dataset_dir):
        edges = pd.read_csv(path_to_dataset_dir + dataset + "_A.txt",
                            header=None).rename(columns={0: "src", 1: "dst"})

        edge_labels = pd.read_csv(
            path_to_dataset_dir + dataset + "_edge_labels.txt", header=None)
        edges.index += 1
        edge_labels.index += 1
        edges['label'] = edge_labels[0]

        edges = edges.set_index(['src', 'dst'])['label'].to_dict()

        nx.set_edge_attributes(G=G, values=edges, name='label')

    # Node attributes
    if dataset + "_node_attributes.txt" in listdir(path_to_dataset_dir):
        node_attributes = pd.read_csv(path_to_dataset_dir + dataset + "_node_attributes.txt",
                                      header=None)
        if "node_attributes" in mappings:
            node_attributes = node_attributes.rename(
                columns={x: mappings['node_attributes'][x] for x in range(len(mappings['node_attributes']))})
        else:
            node_attributes = node_attributes.rename(
                columns={x: "attr_" + str(x) for x in range(len(node_attributes.columns))})

        node_attributes.index += 1
        [nx.set_node_attributes(G, node_attributes[col].to_dict(), col) for col in node_attributes.columns]

    # Edge attributes
    if dataset + "_edge_attributes.txt" in listdir(path_to_dataset_dir):
        edge_attributes = pd.read_csv(path_to_dataset_dir + dataset + "_edge_attributes.txt",
                                      header=None)
        if "edge_attributes" in mappings:
            edge_attributes = edge_attributes.rename(
                columns={x: "edge_" + mappings['edge_attributes'][x] for x in range(len(mappings['edge_attributes']))})
        else:
            edge_attributes = edge_attributes.rename(
                columns={x: "edge_attr_" + str(x) for x in range(len(edge_attributes.columns))})

        edge_attributes.index += 1
        edge_attributes.index = edges.keys()
        [nx.set_edge_attributes(G, edge_attributes[col].to_dict(), col)
         for col in edge_attributes.columns]

    print ("DONE")
    return G


def load_graph_kernel_labels(path_to_dataset_dir, dataset=None):
    files = listdir(path_to_dataset_dir)
    if path_to_dataset_dir[-1] != sep:
        path_to_dataset_dir += sep

    adjacency = [s for s in files if "_A" in s]
    if len(adjacency) == 0:
        raise RuntimeWarning("No files found!")
    elif len(adjacency) > 1:
        raise RuntimeWarning(
            "More than one GraphKernel dataset in directory; specify by passing a value for the dataset argument.")
        dataset = adjacency[0].split("_A")[0]
        print("Using " + dataset.split("_A")[0])
    else:
        dataset = adjacency[0].split("_A")[0]

    graphLabels = pd.read_csv(
        path_to_dataset_dir + dataset + '_graph_labels.txt', header=None)

    graphLabels.index += 1
    graphLabels = graphLabels[0].to_dict()

    return graphLabels

# Component Labels
#  e.g. 1-6
# Node Labels
#  e.g. 7-12
# Edge Labels
#  e.g. 13-245
# Node Attributes (transformed - elbow method will dynamically change how many labels to account for)
#  e.g. 246-300
# Edge Attributes(transformed - elbow method will dynamically change how many labels to account for)
#  e.g. 301-320
