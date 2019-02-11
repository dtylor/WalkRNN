import pandas as pd
import numpy as np
from os import listdir, sep
import networkx as nx

def load_graph_kernel_graph(path_to_dataset_dir, dataset=None, mappings={}):
    """
    Loads Graph Kernel dataset into a NetworkX graph.

    Loads a dataset into a NetworkX graph, adding the node labels and component labels to each NetworkX node.

    Attributes:
        path_to_dataset_dir (str): Path to the directory containing the dataset files.
        dataset (str): Dataset-specific prefix if more than one dataset is in the directory. (optional)

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
    

    # Components
    components = pd.read_csv(
        path_to_dataset_dir + dataset + "_graph_indicator.txt", header=None)

    components.index += 1
    components = components.rename(columns={0: "component"})

    if "components" in mappings:
        components['component'] = components.component.map(mappings['components'])
    components = components.component.to_dict()

    nx.set_node_attributes(G=G, values=components, name='component')

    # Node Labels
    node_labels = pd.read_csv(path_to_dataset_dir +
                              dataset + "_node_labels.txt", header=None)
    node_labels.index += 1
    node_labels = node_labels.rename(columns={0: "label"})
    
    if "node_labels" in mappings:
        node_labels['label'] = node_labels.label.map(
            mappings['node_labels'])
    
    node_labels = node_labels['label'].to_dict()

    nx.set_node_attributes(G=G, values=node_labels, name='label')


    # Edge Labels
    if dataset+"_edge_labels.txt" in listdir(path_to_dataset_dir):
        edges = pd.read_csv(path_to_dataset_dir + dataset +"_A.txt",
                            header=None).rename(columns={0: "src", 1: "dst"})

        edge_labels = pd.read_csv(
            path_to_dataset_dir + dataset + "_edge_labels.txt", header=None)
        edges.index += 1
        edge_labels.index += 1
        edges['label'] = edge_labels[0]

        if "edge_labels" in mappings:
            edges['label'] = edges.label.map(
                mappings['edge_labels'])
        else:
            edges['label'] = edges['label'].apply(lambda x: "e"+str(int(x)))

        edges = edges.set_index(['src', 'dst'])['label'].to_dict()

        nx.set_edge_attributes(G=G, values=edges, name='label')


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
        print("Using "+dataset.split("_A")[0])
    else:
        dataset = adjacency[0].split("_A")[0]

    graphLabels = pd.read_csv(
        path_to_dataset_dir + dataset + '_graph_labels.txt', header=None)

    graphLabels.index += 1
    graphLabels = graphLabels[0].to_dict()

    return graphLabels
