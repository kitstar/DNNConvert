from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import common.IR.graph_pb2 as graph_pb2
from common.utils import load_protobuf_from_file


class IRGraphNode(object):        

    def __init__(self, layer):
        self.in_edges = list()
        self.out_edges = list()
        self.IR_layer = layer
        self.left_in_edges = 0



    @property
    def name(self):
        return self.IR_layer.name



    @property
    def type(self):
        return self.IR_layer.op



class IRGraph(object):
  
    @staticmethod
    def shapeToStr(tensor_shape):
        ret = ""
        first = True
        for e in tensor_shape.dim:
            if e.size != -1:
                if first == False:
                    ret += ", "
                ret += str(e.size)
                first = False
        return ret

    
    @classmethod
    def __init__(self, filename):
        self.layer_map = {}        # key: layer_name    value: IR layer 
        self.input_layers = list()
        self.output_layers = list()
        self.layer_name_map = dict()   # maybe re-direct to defuse or fuse node

        self.model = graph_pb2.GraphDef()
        load_protobuf_from_file(self.model, filename)
  
  
    @classmethod
    def build(self):
        self.input_layers = list()
        for i, layer in enumerate(self.model.node):
            self.layer_map[layer.name] = IRGraphNode(layer)
            self.layer_name_map[layer.name] = layer.name
            for pred in layer.input:
                self._make_connection(pred, layer.name)

        self._make_input_layers()
        self._make_output_layers()



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
            print ("Warning: IRGraph has not been built.")
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
            print ("Error: IRGraph doesn't have node [%s]." % name)
            return None
        else:
            return self.layer_map[name]
        

    @classmethod
    def _make_connection(self, src, dst):
        self.layer_map[src].out_edges.append(dst)
        self.layer_map[dst].in_edges.append(src)
