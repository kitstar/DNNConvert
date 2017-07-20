from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import keras as _keras


class Keras2GraphNode(object):        

    def __init__(self, layer):
        self.in_edges = list()
        self.out_edges = list()
        self.keras_layer = layer
        self.left_in_edges = 0



    @property
    def name(self):
        return self.keras_layer.name



    @property
    def type(self):
        return self.keras_layer.__class__.__name__



class Keras2Graph(object):
  
    @classmethod
    def __init__(self, model):
        # key: layer_name    value: keras layer
        self.layer_map = {}
        self.input_layers = list()
        self.layer_name_map = dict()   # maybe re-direct to defuse or fuse node

        # sanity check.
        if not (type(model) == _keras.models.Sequential or type(model) == _keras.models.Model):
            raise TypeError("Keras layer of type %s is not supported." % type(model))
        self.model = model
  
  
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


    @classmethod
    def _make_input_layers(self):
        for name, layer in self.layer_map.items():
            layer.left_in_edges = len(layer.in_edges)
            if len(layer.in_edges) == 0:
                self.input_layers.append(name)



    @classmethod
    def coreml_make_input_layers(self):
        """
        Extract the ordering of the input layers.
        """
        self.input_layers = []
        if hasattr(self.model, 'input_layers'):
            input_keras_layers = self.model.input_layers[:]
            self.input_layers = [None] * len(input_keras_layers)
            for name, layer in self.layer_map.items():
                if isinstance(layer.layer, _keras.engine.topology.InputLayer):
                    if layer in input_keras_layers:
                        idx = input_keras_layers.index(keras_layer)
                        self.input_layers[idx] = layer
        elif len(self.model.inbound_nodes) <= 1:
            for ts in _to_list(self.model.input):
                # search for the InputLayer that matches this ts
                for l in self.layer_list:
                    kl = self.keras_layer_map[l]
                    if isinstance(kl, _keras.engine.topology.InputLayer) and kl.input == ts:
                        self.input_layers.append(l)
        else:
            raise ValueError("Input values cannot be identified.")


    @classmethod
    def get_input_layers(self):
        if self.input_layers == None:
            print ("Warning: Keras2Graph has not been built.")
            build(self)
        return self.input_layers

    
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
