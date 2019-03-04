from utilities import load_graph_kernel_graph, load_graph_kernel_labels
from module import _get_components, get_structural_signatures, walk_as_string
import unittest
import networkx as nx
from functools import reduce

class TestUtilityFunctions(unittest.TestCase):
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
        mappings_2 = {
            "edge_labels": {
                0:	"wedge",
                1:	"arrangement"
            }
        }
        self.graph = load_graph_kernel_graph("./Cuneiform", mappings=mappings)
        self.graph2 = load_graph_kernel_graph("./Cuneiform", mappings=mappings_2)
        self.y = load_graph_kernel_labels("./Cuneiform")

    def test_load(self):
        self.assertTrue(type(self.graph) == nx.classes.graph.Graph,
                        'returned object is not a networkX Graph')
        self.assertTrue(type(self.y) == dict,
                        'graph labels are not a dict')
        self.assertTrue(len(list(self.y.keys())) == len(list(self.y.values())),
                        'graph labels dict has keys, values mismatch')
        print("test_load passes")
        return True

    def test_node_labels(self):
        labels = list(set(nx.get_node_attributes(self.graph, 'label_0').values()))
        self.assertGreater(len(labels), 0, 'loaded Graph has no node labels')
        print("test_node_labels passes")
        return True
    
    def test_no_node_labels(self):
        labels = list(set(nx.get_node_attributes(
            self.graph2, 'label_0').values()))
        self.assertSetEqual(set([0,1,2,3]), set(labels), 'Graph contains node_labels not in the dataset')
        print("test_no_node_labels passes")
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
        attr_0 = nx.get_edge_attributes(self.graph, 'edge_attr_0')
        self.assertGreater(
            len(attr_0), 0, 'loaded Graph edges have no "edge_attr_0" attributes')
        attr_1 = nx.get_edge_attributes(self.graph, 'edge_attr_1')
        self.assertGreater(
            len(attr_1), 0, 'loaded Graph edges have no "edge_attr_1" attributes')
        return True

# class TestModuleFunctions(unittest.TestCase):
#     def setUp(self):
#         mappings = {
#             "node_labels": [{
#                 0:  "depthPoint",
#                 1:	"tailVertex",
#                 2:	"leftVertex",
#                 3:	"rightVertex",
#             }, {
#                 0:	"vertical",
#                 1:	"Winkelhaken",
#                 2:	"horizontal"
#             }],
#             "edge_labels": {
#                 0:	"wedge",
#                 1:	"arrangement"
#             },
#         }
#         self.graph = load_graph_kernel_graph("./Cuneiform", mappings=mappings)
#         self.y = load_graph_kernel_labels("./Cuneiform")
#         self.graph, self.pca, self.km = get_structural_signatures(self.graph)
#         self.walks = walk_as_string(self.graph, self.y)

#     def test_structural_signatures_assigned(self):
#         structs = nx.get_node_attributes(self.graph, 'structure')
#         structs = list(set(structs.values()))
#         self.assertGreater(
#             len(structs), 1, 'structures not assigned as node attributes, or are homogeneous')
#         return True

#     def test_valid_words(self):
#         for walk in self.walks['walk']:
#             w = walk.split(" ")
#             for step in w:
#                 self.assertTrue(len(step.split("_")) > 1, 'word "' +
#                                 step + '" does not contain underscore')

