import pandas as pd
import numpy as np
from os import listdir, sep
import networkx as nx

def load_graph_kernel_graph(path_to_dataset_dir, dataset=None):
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
    

    # graph_indicator.txt
    components = pd.read_csv(
        path_to_dataset_dir + dataset + "_graph_indicator.txt", header=None)

    components.index += 1
    components = components.rename(columns={0: "component"}).to_dict()['component']

    nx.set_node_attributes(G=G, values=components, name='component')

    # node_labels.txt
    # Node Labels
    node_labels = pd.read_csv(path_to_dataset_dir +
                              dataset + "_node_labels.txt", header=None)
    node_labels.index += 1
    node_labels = node_labels.rename(columns={0: "label"})['label'].to_dict()

    nx.set_node_attributes(G=G, values=node_labels, name='label')


    return G
