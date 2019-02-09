from n2v import node2vec
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import networkx as nx
import pandas as pd
import numpy as np
from graphwave import graphwave
from graphwave.utils import utils

def GetStructuralSignatures(networkXGraph):
    """
    Get structural embeddings using GraphWave.

    Learns structural signatures of each node using GraphWave, and adds these learned structures to the node attributes on the NetworkX graph.

    Example:

    Attributes:
        networkXGraph (networkx.classes.graph.Graph): A graph containing nodes, edges, and their attributes. Nodes must include an attribute called 'component'.

    Returns: a NetworkX graph where structural embeddings are added as `structure` node attributes.

    """
    return True

def WalkAsString(networkXGraph, graphComponentLabels, featuresToOmit={"nodes":[], "edges":[]}):
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
