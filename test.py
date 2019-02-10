from utilities import load_graph_kernel_graph
from module import *
import pytest
import networkx as nx

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
    expected = nx.classes.reportviews.NodeDataView({1: {'component': 0, 'label': 'red'}, 2: {'component': 1, 'label': 'red', 'structure': 3}, 13: {'component': 2, 'label': 'red', 'structure': 0}, 14: {'component': 2, 'label': 'red', 'structure': 0}, 4: {'component': 1, 'label': 'purple', 'structure': 0}, 6: {'component': 1, 'label': 'yellow', 'structure': 2}, 3: {'component': 1, 'label': 'green', 'structure': 0}, 8: {'component': 1, 'label': 'purple', 'structure': 1}, 7: {'component': 1, 'label': 'purple', 'structure': 1}, 5: {'component': 1, 'label': 'green', 'structure': 0}, 10: {'component': 1, 'label': 'green', 'structure': 0}, 11: { \
                                                   'component': 1, 'label': 'green', 'structure': 0}, 12: {'component': 1, 'label': 'green', 'structure': 0}, 9: {'component': 1, 'label': 'green', 'structure': 3}, 15: {'component': 2, 'label': 'purple', 'structure': 1}, 18: {'component': 2, 'label': 'yellow', 'structure': 2}, 16: {'component': 2, 'label': 'green', 'structure': 3}, 19: {'component': 2, 'label': 'purple', 'structure': 1}, 17: {'component': 2, 'label': 'green', 'structure': 0}, 20: {'component': 2, 'label': 'green', 'structure': 0}, 21: {'component': 2, 'label': 'green', 'structure': 0}, 22: {'component': 2, 'label': 'green', 'structure': 0}, 23: {'component': 2, 'label': 'green', 'structure': 3}})
    assert result == expected
