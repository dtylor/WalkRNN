from utilities import load_graph_kernel_graph
from module import _get_components, get_structural_signatures
import pytest
import networkx as nx
from functools import reduce

def test_load_graph_kernel_graph():
    # with pytest.raises(RuntimeError):
    a = load_graph_kernel_graph("./data/")
    assert len(a.nodes) == 23
    return True


def test__get_components():
    # with pytest.raises(RuntimeError):
    a = load_graph_kernel_graph("./data/")
    b = _get_components(a)
    expected = {0: [1],
        1: [2, 4, 6, 3, 8, 7, 5, 10, 11, 12, 9],
        2: [13, 14, 15, 18, 16, 19, 17, 20, 21, 22, 23]}
    assert b == expected
    return True

def test_get_structural_signatures():
    a = load_graph_kernel_graph('./data/')
    result = get_structural_signatures(a)
    signatures = list(nx.get_node_attributes(result, 'structure').items())
    assert len(signatures) == 22 and reduce((lambda x,y: type(x) and type(y)), signatures)
    return True

if __name__ == "__main__":
    test_load_graph_kernel_graph()
    test__get_components()
    test_get_structural_signatures()
