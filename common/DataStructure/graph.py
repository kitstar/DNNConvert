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
        self.layer_name_map = dict()   # maybe re-direct to defuse or fuse node
        self.model = model
    
    def build(self):
        pass


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
    def get_input_layers(self):
        if self.input_layers == None:
            print ("Warning: Keras2Graph has not been built.")
            build(self)
        return self.input_layers



    @classmethod
    def get_output_layers(self):
        if self.output_layers == None:
            print ("Warning: IRGraph has not been built.")
            build(self)
        return self.output_layers


    
    @classmethod
    def get_node(self, name):
        if not name in self.layer_map:
            print ("Error: Keras2Graph doesn't have node [%s]." % name)
            return None
        else:
            return self.layer_map[name]
        

    @classmethod
    def _make_connection(self, src, dst):
        self.layer_map[src].out_edges.append(dst)
        self.layer_map[dst].in_edges.append(src)
