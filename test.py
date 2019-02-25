from utilities import load_graph_kernel_graph, load_graph_kernel_labels
from module import _get_components, get_structural_signatures, walk_as_string
import unittest
import networkx as nx
from functools import reduce

class TestLoadGraph(unittest.TestCase):
    def setUp(self):
        mappings =  {
            "node_labels": [{
                0:  "depthPoint",
                1:	"tailVertex",
                2:	"leftVertex",
                3:	"rightVertex",
            }, {
                0:	"vertical",
                1:	"Winkelhaken",
                2:	"horizontal"
            }],
            "edge_labels": {
                0:	"wedge",
                1:	"arrangement"
            },
        }
        self.graph = load_graph_kernel_graph("./Cuneiform", mappings=mappings)
        self.y = load_graph_kernel_labels("./Cuneiform")

    def test_load(self):
        self.assertTrue(type(self.graph) == nx.classes.graph.Graph,
                        'returned object is not a networkX Graph')
        self.assertTrue(type(self.y) == dict,
                        'graph labels are not a dict')
        self.assertTrue(len(list(self.y.keys())) == len(list(self.y.values())),
                        'graph labels dict has keys, values mismatch')
        return True

    def test_node_labels(self):
        labels = list(set(nx.get_node_attributes(self.graph, 'label_0').values()))
        self.assertGreater(len(labels), 0, 'loaded Graph has no node labels')
        return True

    def test_edge_labels(self):
        labels = list(set(nx.get_edge_attributes(self.graph, 'label').values()))
        self.assertGreater(len(labels), 0, 'loaded Graph has no edge labels')
        return True

    def test_node_attributes(self):
        attr_0 = nx.get_node_attributes(self.graph, 'attr_0')
        self.assertGreater(
            len(attr_0), 0, 'loaded Graph nodes have no "attr_0" attributes')
        attr_1 = nx.get_node_attributes(self.graph, 'attr_1')
        self.assertGreater(
            len(attr_1), 0, 'loaded Graph nodes have no "attr_1" attributes')
        attr_2 = nx.get_node_attributes(self.graph, 'attr_2')
        self.assertGreater(
            len(attr_2), 0, 'loaded Graph nodes have no "attr_2" attributes')
        return True

    def test_edge_attributes(self):
        attr_0 = nx.get_edge_attributes(self.graph, 'attr_0')
        self.assertGreater(
            len(attr_0), 0, 'loaded Graph edges have no "attr_0" attributes')
        attr_1 = nx.get_edge_attributes(self.graph, 'attr_1')
        self.assertGreater(
            len(attr_1), 0, 'loaded Graph edges have no "attr_1" attributes')
        return True

class TestGetStructuralSignatures(unittest.TestCase):
    def setUp(self):
        mappings = {
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
        self.graph, self.pca, self.km = get_structural_signatures(self.graph)
    
    def test_structural_signatures_assigned(self):
        structs = nx.get_node_attributes(self.graph, 'structure')
        structs = list(set(structs.values()))
        self.assertGreater(
            len(structs), 1, 'structures not assigned as node attributes, or are homogeneous')
        return True

class TestWalkAsString(unittest.TestCase):
    def setUp(self):
        mappings = {
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
                0: "edge_1",
                1: "edge_2",
                2: "edge_3"
            },
            "node_attributes": ["chem", "charge", "x", "y"]
        }
        self.graph = load_graph_kernel_graph("./AIDS", mappings=mappings)
        self.graph, self.pca, self.km = get_structural_signatures(self.graph)
        self.y = load_graph_kernel_labels("./AIDS")
        self.walks = walk_as_string(self.graph, self.y)

    def test_valid_words(self):
        for walk in self.walks['walk']:
            w = walk.split(" ")
            for step in w:
                self.assertTrue(len(step.split("_")) > 1, 'word "' + step + '" does not contain underscore')

