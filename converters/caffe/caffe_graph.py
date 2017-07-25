from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import converters.caffe.caffe_pb2 as caffe_pb2
from common.DataStructure.graph import GraphNode, Graph


class CaffeGraphNode(GraphNode):

    def __init__(self, layer):
        GraphNode.__init__(self, layer)
        self.withRelu = False
        self.withDropout = False
   


    @property
    def name(self):
        return self.layer.name



    @property
    def type(self):
        return self.layer.type



    @property
    def caffe_layer(self):
        return self.layer



class CaffeGraph(Graph):
  
    phase_map = {
            0 : "TRAIN",
            1 : "TEST"
            }
    
    @staticmethod
    def _match_phase(layer, phase):
        if len(layer.include) > 0:
            for p in layer.include:
                if phase == CaffeGraph.phase_map[p.phase]:
                    return True
            return False

        for p in layer.exclude:
            if phase == CaffeGraph.phase_map[p.phase]:
                return False

        # Dropout
        if phase == "TEST":
            return layer.type != "Dropout"
        
        return True
  


    @classmethod
    def __init__(self, model, phase):
        super(CaffeGraph, self).__init__(model)
        self.phase = phase
 

    @classmethod
    def build(self):
        self.input_layers = list()

        todo_layer = list()
        for i, layer in enumerate(self.model.layer):
            if CaffeGraph._match_phase(layer, self.phase) == True:
                if self._same_name_handler(layer) == True:
                    continue

                self.layer_name_map[layer.name] = layer.name
                self.layer_map[layer.name] = CaffeGraphNode(layer) 
                todo_layer.append(layer)
 
        for i, layer in enumerate(todo_layer):    
            for input_name in layer.bottom:
                self._make_connection(input_name, layer.name)
            for output_name in layer.top:
                if not output_name in self.layer_name_map:
                    new_layer = caffe_pb2.LayerParameter()
                    new_layer.name = output_name
                    new_layer.type = layer.type
                    self.layer_map[output_name] = CaffeGraphNode(new_layer)
                    self.layer_name_map[output_name] = output_name 

                self._make_connection(layer.name, output_name)

        super(CaffeGraph, self).build()



    @classmethod
    def _same_name_handler(self, layer):
        if len(layer.bottom) != 1:
            return
        if len(layer.top) != 1:
            return
        if layer.bottom[0] != layer.top[0]:
            return

        if layer.type == "ReLU":
            self.layer_map[layer.top[0]].withRelu = True
        elif layer.type == "Dropout":
            self.layer_map[layer.top[0]].withDropout = True
        else:
            print ("Error! Unknown type [%s] for same name handler." % layer.type)

        return True
