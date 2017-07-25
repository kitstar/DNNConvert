from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class GraphNode(object):

    def __init__(self, layer):
        self.in_edges = list()
        self.out_edges = list()
        self.layer = layer
        self.left_in_edges = 0



class Graph(object):
 
    @classmethod
    def __init__(self, model):
        # key: layer_name    value: keras layer
        self.layer_map = {}
        self.input_layers = list()
        self.output_layers = list()
        self.layer_name_map = dict()   # maybe re-direct to defuse or fuse node
        self.topological_sort = list()
        self.model = model



    @classmethod
    def build(self):
        self._make_input_layers()
        self._make_output_layers()
        self._get_topological_sort()



    @classmethod
    def _make_input_layers(self):
        for name, layer in self.layer_map.items():
            layer.left_in_edges = len(layer.in_edges)
            if len(layer.in_edges) == 0:
                self.input_layers.append(name)


    @classmethod
    def _make_output_layers(self):
        for name, layer in self.layer_map.items():
            if len(layer.out_edges) == 0:
                self.output_layers.append(name)


    
    @classmethod
    def get_node(self, name):
        if not name in self.layer_map:
            print ("Error: Graph doesn't have node [%s]." % name)
            return None
        else:
            return self.layer_map[name]
        

    @classmethod
    def _make_connection(self, src, dst):
        if src == dst:
#            print ("Warning: Graph Construct a self-loop node {}. Ignored.".format(src))
            return

        if not dst in self.layer_map[src].out_edges:
            self.layer_map[src].out_edges.append(dst)
        if not src in self.layer_map[dst].in_edges:
            self.layer_map[dst].in_edges.append(src)



    @classmethod
    def _get_topological_sort(self):
        self.topological_sort = self.input_layers[:]
        idx = 0
        while idx < len(self.topological_sort):
            current_node = self.get_node(self.topological_sort[idx])
            for next_node in current_node.out_edges:
                next_node_info = self.get_node(next_node)
                next_node_info.left_in_edges -= 1
                if next_node_info.left_in_edges == 0:
                    self.topological_sort.append(next_node)
            idx += 1
