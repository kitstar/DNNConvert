from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import keras as _keras


class Keras2GraphNode(object):        

    def __init__(self, layer):
        self.in_edges = list()
        self.out_edges = list()
        self.layer = layer   



class Keras2Graph(object):

   
    @classmethod
    def __init__(self, model):
        # key: layer_name    value: keras layer
        self.layer_map = {}
        self.input_layers = []

        # sanity check.
        if not (type(model) == _keras.models.Sequential or type(model) == _keras.models.Model):
            raise TypeError("Keras layer of type %s is not supported." % type(model))
        self.model = model
 
   
    def build(self):
        for i, layer in enumerate(self.model.layers):
            self.layer_map[layer.name] = Keras2GraphNode(layer)
            for node in layer.inbound_nodes:
                for pred in node.inbound_layers:
                    if pred.name not in self.layer_map:
                        self.layer_map[pred.name] = Keras2GraphNode(pred)
                    self._make_connection(pred.name, layer.name)

        self._make_input_layers()

    @classmethod
    def _make_input_layers(self):
        for name, layer in self.layer_map.items():
            if len(layer.in_edges) == 0:
                self.input_layers.append(name)

    @classmethod
    def _make_connection(self, src, dst):
        self.layer_map[src].out_edges.append(dst)
        self.layer_map[dst].in_edges.append(src)
