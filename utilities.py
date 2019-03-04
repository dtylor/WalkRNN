from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from os import listdir, sep
import networkx as nx


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

def load_graph_kernel_graph(path_to_dataset_dir, dataset=None, mappings={}):
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
        raise RuntimeWarning("More than one GraphKernel dataset in directory; specify by passing a value for the dataset argument.")
        dataset = adjacency[0].split("_A")[0]
        print("Using "+dataset.split("_A")[0])
    else:
        dataset = adjacency[0].split("_A")[0]

    G = nx.read_edgelist(path_to_dataset_dir + dataset + "_A.txt",
                         delimiter=',', nodetype=int, encoding="utf-8")
    

    current_vocab_size = 0

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
        number_of_labels = len(node_labels[column].unique()) + 1
        node_labels[column] += current_vocab_size
        this_label = node_labels[column].to_dict()
        nx.set_node_attributes(
            G=G, values=this_label, name='label_'+str(column))
        current_vocab_size += number_of_labels
        print("Added node_labels[" + str(column) + "]; current_vocab_size = "+str(current_vocab_size))

    # Edge Labels
    if dataset+"_edge_labels.txt" in listdir(path_to_dataset_dir):
        edges = pd.read_csv(path_to_dataset_dir + dataset +"_A.txt",
                            header=None).rename(columns={0: "src", 1: "dst"})

        edge_labels = pd.read_csv(
            path_to_dataset_dir + dataset + "_edge_labels.txt", header=None)
        edges.index += 1
        edge_labels.index += 1
        edges['label'] = edge_labels[0]

        number_of_labels = len(edges['label'].unique()) + 1
        edges['label'] += current_vocab_size
        current_vocab_size += number_of_labels
        print("Added edge_labels; current_vocab_size = "+str(current_vocab_size))

        edges = edges.set_index(['src', 'dst'])['label'].to_dict()

        nx.set_edge_attributes(G=G, values=edges, name='label')

    # Node attributes
    if dataset+"_node_attributes.txt" in listdir(path_to_dataset_dir):
        node_attributes = pd.read_csv(path_to_dataset_dir + dataset + "_node_attributes.txt",
                            header=None)
        if "node_attributes" in mappings:
            node_attributes = node_attributes.rename(
                columns={x: mappings['node_attributes'][x] for x in range(len(mappings['node_attributes']))})
        else:
            node_attributes = node_attributes.rename(columns={x: "attr_"+str(x) for x in range(len(node_attributes.columns))})
        
        node_attributes.index += 1

        # transform here
        node_attributes, kmeans_models = transform_features(node_attributes)
        for col in node_attributes.columns:
            number_of_labels = len(node_attributes[col].unique()) + 1
            node_attributes[col] += current_vocab_size
            current_vocab_size += number_of_labels
            print("Added node_attributes[" + str(col) + "]; current_vocab_size = " +
                  str(current_vocab_size))
        [nx.set_node_attributes(G, node_attributes[col].to_dict(), col) for col in node_attributes.columns]

    # Edge attributes
    if dataset+"_edge_attributes.txt" in listdir(path_to_dataset_dir):
        edge_attributes = pd.read_csv(path_to_dataset_dir + dataset + "_edge_attributes.txt",
                                      header=None)
        if "edge_attributes" in mappings:
            edge_attributes = edge_attributes.rename(
                columns={x: "edge_"+mappings['edge_attributes'][x] for x in range(len(mappings['edge_attributes']))})
        else:
            edge_attributes = edge_attributes.rename(
                columns={x: "edge_attr_"+str(x) for x in range(len(edge_attributes.columns))})

        edge_attributes.index += 1
    
        # transform here
        edge_attributes, kmeans_models = transform_features(edge_attributes)
        for col in edge_attributes.columns:
            number_of_labels = len(edge_attributes[col].unique()) + 1
            edge_attributes[col] += current_vocab_size
            current_vocab_size += number_of_labels
            print("Added edge_attributes[" + str(col) + "]; current_vocab_size = " +
                  str(current_vocab_size))
        edge_attributes.index = edges.keys()
        [nx.set_edge_attributes(G, edge_attributes[col].to_dict(), col)
         for col in edge_attributes.columns]

        print ("DONE")

    return {"G": G, "vocab_size": current_vocab_size}

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
        print("Using "+dataset.split("_A")[0])
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
