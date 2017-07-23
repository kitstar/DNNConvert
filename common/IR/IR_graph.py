from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import common.IR.graph_pb2 as graph_pb2
from common.DataStructure.graph import Graph, GraphNode
from common.utils import load_protobuf_from_file


class IRGraphNode(GraphNode):

    @property
    def IR_layer(self):
        return self.layer

    @property
    def name(self):
        return self.layer.name



    @property
    def type(self):
        return self.layer.op



class IRGraph(Graph):
  
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
