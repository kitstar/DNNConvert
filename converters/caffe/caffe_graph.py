from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from common.DataStructure.graph import GraphNode, Graph


class CaffeGraphNode(GraphNode): 

    @property
    def name(self):
        return self.keras_layer.name



    @property
    def type(self):
        return self.keras_layer.__class__.__name__



class CaffeGraph(Graph):
  
    @classmethod
    def __init__(self, model):
        self.Graph.__init__(model)
 

    @classmethod
    def build(self):
        self.input_layers = list()
        for i, layer in enumerate(self.model.layers):
            self.layer_map[layer.name] = Keras2GraphNode(layer)
            self.layer_name_map[layer.name] = layer.name
            for node in layer.inbound_nodes:
                for pred in node.inbound_layers:
                    if pred.name not in self.layer_map:
                        self.layer_map[pred.name] = Keras2GraphNode(pred)
                        self.layer_name_map[pred.name] = pred.name
                    self._make_connection(pred.name, layer.name)

        self._make_input_layers()
