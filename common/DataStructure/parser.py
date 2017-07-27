from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
            
import common.IR.graph_pb2 as graph_pb2
from common.IR.graph_pb2 import NodeDef, GraphDef, DataType


class Parser(object):

    @classmethod
    def __init__(self):
        self.IR_graph = GraphDef()


    @classmethod
    def saveToJson(self, filename = None):
        import google.protobuf.json_format as json_format
        
        json_str = json_format.MessageToJson(self.IR_graph, preserving_proto_field_name = True)
        if filename != None:
            with open(filename, "w") as of:
                of.write(json_str)
            print ("IR saved as [{}].".format(filename))
        
        return json_str
