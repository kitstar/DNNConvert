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
        model = graph_pb2.GraphDef()
        load_protobuf_from_file(model, filename)
        super(IRGraph, self).__init__(model)


  
    @classmethod
    def build(self):
        self.input_layers = list()
        for i, layer in enumerate(self.model.node):
            self.layer_map[layer.name] = IRGraphNode(layer)
            self.layer_name_map[layer.name] = layer.name
            for pred in layer.input:
                self._make_connection(pred, layer.name)
        super(IRGraph, self).build()


    @classmethod
    def saveToJson(filename = None):
        json_str = json_format.MessageToJson(parser.IR_graph, preserving_proto_field_name = True)
        if filename != None:
            with open(filename, "wb") as of:
                of.write(json_str)
            print ("IR saved as {}".format(filename))
        return json_str
