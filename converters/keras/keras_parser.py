from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from converters.keras.keras2_graph import Keras2Graph
import keras as _keras

class KerasParser(object):

    @staticmethod
    def _load_model(model_network_path, model_weight_path):
        """Load a keras model from disk

        Parameters
        ----------
        model_network_path: str
            Path where the model network path is (json file)

        model_weight_path: str
            Path where the model network weights are (hd5 file)

        Returns
        -------
        model: A keras model
        """
        from keras.models import model_from_json
        import json

        # Load the model network
        json_file = open(model_network_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        # Load the model weights
        loaded_model = model_from_json(loaded_model_json)
        if os.path.isfile(model_weight_path) == True:
            loaded_model.load_weights(model_weight_path)
        else:
            print("Warning: Keras Model Weight File [%s] is not found." % (model_weight_path))

        return loaded_model


    @staticmethod
    def _convert(model,
                 input_names = None,
                 output_names = None,
                 image_input_names = None,
                 is_bgr = False,
                 red_bias = 0.0,
                 green_bias = 0.0,
                 blue_bias = 0.0,
                 gray_bias = 0.0,
                 image_scale = 1.0,
                 class_labels = None,
                 predicted_feature_name = None):

        # load model files into Keras graph
        if isinstance(model, basestring):
            model = _keras.models.load_model(model)
        elif isinstance(model, tuple):
            model = KerasParser._load_model(model[0], model[1])

        _keras.utils.plot_model(model, "model.png", show_shapes = True)

        # Build network graph 
        graph =  Keras2Graph(model)        
        graph.build()

        # bfs
        traverse_nodes = graph.input_layers
        enqueued_nodes = set(traverse_nodes)
        while len(traverse_nodes) > 0:
            current_node = traverse_nodes.pop()            
            print (current_node)
            for next_node in graph.layer_map[current_node].out_edges:
                if not next_node in enqueued_nodes:
                    enqueued_nodes.add(next_node)
                    traverse_nodes.append(next_node)


        for node in model.layers:
            node_type = node.__class__.__name__            
            if hasattr(KerasParser, "rename_" + node_type):
                func = getattr(self, "rename_" + node_type)
                func(node)
            else:
                print("KerasParser has not supported operator [%s]." % (node_type))

        print ("finish!")


    def rename_Placeholder(self, node_info):
        print (node_info)

    def rename_Const(self, node_info):
        print (node_info)
      
    def rename_Assign(self, node_info):
        print (node_info)

    def rename_NoOp(self, node_info):
        print ("Ignore node [%s]." % (node_info.op))

    def rename_RestoreV2(self, node_info):
        print ("Ignore node [%s]." % (node_info.op))

    def rename_SaveV2(self, node_info):
        print ("Ignore node [%s]." % (node_info.op))

    def rename_Identity(self, node_info):
        print (node_info)

    def rename_Mean(self, node_info):
        print (node_info)

    def rename_VariableV2(self, node_info):
        print (node_info)

    def rename_reshape(self, node_info):
        print (node_info)

    def rename_ConcatV2(self, node_info):
        print (node_info)

    def rename_Add(self, node_info):
        print (node_info)

    def rename_Sub(self, node_info):
        print (node_info)

    def rename_Reshape(self, node_info):
        print (node_info)

    def rename_Slice(self, node_info):
        print (node_info)

    def rename_MatMul(self, node_info):
        print (node_info)

    def rename_SoftmaxCrossEntropyWithLogits(self, node_info):
        print (node_info)

    def rename_ApplyGradientDescent(self, node_info):
        print ("Ignore node [%s]." % (node_info.op))


'''
   def rename_VariableV2(self, node_info):
        print (node_info)

   def rename_VariableV2(self, node_info):
        print (node_info)

   def rename_VariableV2(self, node_info):
        print (node_info)

   def rename_VariableV2(self, node_info):
        print (node_info)

   def rename_VariableV2(self, node_info):
        print (node_info)

   def rename_VariableV2(self, node_info):
        print (node_info)

   def rename_VariableV2(self, node_info):
        print (node_info)

   def rename_VariableV2(self, node_info):
        print (node_info)

   def rename_VariableV2(self, node_info):
        print (node_info)
'''  
