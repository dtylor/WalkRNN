from utilities import load_graph_kernel_graph
from module import _get_components, get_structural_signatures
import unittest
import networkx as nx
from functools import reduce

class TestLoadGraph(unittest.TestCase):
    def setUp(self):
        mappings =  {
            "node_labels": {
                0:       "C",
                1:       "O",
                2:       "N",
                3:       "Cl",
                4:       "F",
                5:       "S",
                6:       "Se",
                7:       "P",
                8:       "Na",
                9:       "I",
                10:      "Co",
                11:      "Br",
                12:      "Li",
                13:      "Si",
                14:      "Mg",
                15:      "Cu",
                16:      "As",
                17:      "B",
                18:      "Pt",
                19:      "Ru",
                20:      "K",
                21:      "Pd",
                22:      "Au",
                23:      "Te",
                24:      "W",
                25:      "Rh",
                26:      "Zn",
                27:      "Bi",
                28:      "Pb",
                29:      "Ge",
                30:      "Sb",
                31:      "Sn",
                32:      "Ga",
                33:      "Hg",
                34:      "Ho",
                35:      "Tl",
                36:      "Ni",
                37:      "Tb"
            },
            "edge_labels": {
                0: 1,
                1: 2,
                2: 3
            },
            "node_attributes": ["chem", "charge", "x", "y"]
        }
        self.graph = load_graph_kernel_graph("./AIDS", mappings=mappings)
    
    def test_load(self):
        self.assertTrue(type(self.graph) == nx.classes.graph.Graph, 'returned object is not a networkX Graph')

    def test_node_labels(self):
        labels = list(set(nx.get_node_attributes(self.graph, 'label').values()))
        self.assertGreater(len(labels), 0, 'loaded Graph has no node labels')

    def test_edge_labels(self):
        labels = list(set(nx.get_edge_attributes(self.graph, 'label').values()))
        self.assertGreater(len(labels), 0, 'loaded Graph has no edge labels')

    def test_node_attributes(self):
        chem = nx.get_node_attributes(self.graph, 'chem')
        self.assertGreater(
            len(chem), 0, 'loaded Graph has no "chem" attributes')
        charge = nx.get_node_attributes(self.graph, 'charge')
        self.assertGreater(
            len(charge), 0, 'loaded Graph has no "charge" attributes')
        x = nx.get_node_attributes(self.graph, 'x')
        self.assertGreater(
            len(x), 0, 'loaded Graph has no "x" attributes')
        y = nx.get_node_attributes(self.graph, 'y')
        self.assertGreater(
            len(y), 0, 'loaded Graph has no "y" attributes')
        # for i in range(1, len(y), int(len(y)/10)):
        #     print(str(chem[i]) + " " + str(charge[i]) + " " + str(x[i]) + " " + str(y[i]))



# def test_load_graph_kernel_graph():
#     a = load_graph_kernel_graph("./AIDS")
#     chems = nx.get_node_attributes(a, 'chem')
#     assert len(list(set(chems.values()))) > 0 
#     charge = nx.get_node_attributes(a, 'charge')
#     assert len(list(set(charge.values()))) > 0 
#     x = nx.get_node_attributes(a, 'x')
#     assert len(list(set(x.values()))) > 0
#     y = nx.get_node_attributes(a, 'y')
#     assert len(list(set(y.values()))) > 0
#     return True


# def test__get_components():
#     a = load_graph_kernel_graph("./data/")
#     b = _get_components(a)
#     expected = {0: [1],
#         1: [2, 4, 6, 3, 8, 7, 5, 10, 11, 12, 9],
#         2: [13, 14, 15, 18, 16, 19, 17, 20, 21, 22, 23]}
#     assert b == expected
#     return True

# def test_get_structural_signatures():
#     a = load_graph_kernel_graph('./data/')
#     result = get_structural_signatures(a)
#     signatures = list(nx.get_node_attributes(result, 'structure').items())
#     assert len(signatures) == 22 and reduce((lambda x,y: type(x) and type(y)), signatures)
#     return True

# if __name__ == "__main__":
#     test_load_graph_kernel_graph()
#     test__get_components()
#     test_get_structural_signatures()
