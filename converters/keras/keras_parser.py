from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import keras as _keras
from converters.keras.keras2_graph import Keras2Graph
import common.IR.graph_pb2 as graph_pb2
from common.IR.graph_pb2 import NodeDef, GraphDef, DataType


class KerasParser(object):
   
    dtype_map = {
            "float16" : graph_pb2.DT_FLOAT16,
            "float32" : graph_pb2.DT_FLOAT32,
            "float64" : graph_pb2.DT_FLOAT64,
            "int16"   : graph_pb2.DT_INT16,
            "int32"   : graph_pb2.DT_INT32,
            "int64"   : graph_pb2.DT_INT64,
            "uint8"   : graph_pb2.DT_UINT8,
            "uint16"  : graph_pb2.DT_UINT16
            }
    

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
        IR_graph = GraphDef()
        traverse_nodes = graph.get_input_layers()
        enqueued_nodes = set(traverse_nodes)
        while len(traverse_nodes) > 0:
            current_node = graph.get_node(traverse_nodes.pop())
            node_type = current_node.keras_layer.__class__.__name__

            if hasattr(KerasParser, "rename_" + node_type):
                func = getattr(KerasParser, "rename_" + node_type)
                new_node = IR_graph.node.add()
                new_node = func(current_node, new_node)
            else:
                print("KerasParser has not supported operator [%s]." % (node_type))

            for next_node in current_node.out_edges:
                if not next_node in enqueued_nodes:
                    enqueued_nodes.add(next_node)
                    traverse_nodes.append(next_node)

        print ("finish!")
        return IR_graph


    @staticmethod
    def _copy_and_reop(source_node, IR_node, new_op = None):
        node_info = source_node.keras_layer
        if new_op == None:
            new_op = node_info.__class__.__name__
        IR_node.name = node_info.name
        IR_node.op = new_op
        for e in source_node.in_edges:
            IR_node.input.append(e)

        if hasattr(node_info, "dtype"):
            IR_node.attr["dtype"].type = KerasParser.dtype_map[node_info.dtype]

        return IR_node

    @staticmethod
    def _copy_shape(source_node, target_node):
        if hasattr(source_node, "output_shape"):
            for dim in source_node.output_shape:
                new_dim = target_node.attr["shape"].shape.dim.add()
                if dim == None:
                    new_dim.size = -1
                else:
                    new_dim.size = dim
        else:
            target_node.attr["shape"].shape.unknown_rank = True

        return target_node

    @staticmethod
    def _convert_dataformat(source_node, target_node):
        if source_node.keras_layer.data_format == 'channel_last':
            target_node.attr["data_format"].s = "NHWC"
        elif source_node.keras_layer.data_format == 'channel_first':
            target_node.attr["data_format"].s = "NCHW"
        else:
            print("Warning: [%s] don't have data format info." % (source_node.keras_layer.name))
        return target_node

    @staticmethod
    def _convert_padding(source_node, target_node):
        if source_node.keras_layer.padding == 'valid':
            target_node.attr["padding"].s = "VALID"
        else:
            target_node.attr["padding"].s = "SAME"
        return target_node


    @staticmethod
    def rename_InputLayer(source_node, IR_node):
        # only for training
        KerasParser._copy_and_reop(source_node, IR_node, "DataInput")
        IR_node = KerasParser._copy_shape(source_node.keras_layer, IR_node)
        return IR_node


    @staticmethod
    def rename_Conv2D(source_node, IR_node):

        print (source_node.keras_layer.activation)
        # name, op and input
        KerasParser._copy_and_reop(source_node, IR_node)
        
        # padding        
        KerasParser._convert_padding(source_node, IR_node)

        # filter
        IR_node.attr["filter"].i = source_node.keras_layer.filters

        # strides
        if isinstance(source_node.keras_layer.strides, tuple) or isinstance(source_node.keras_layer.strides, list):
            sh, sw = source_node.keras_layer.strides
        else:
            sh = source_node.keras_layer.strides
            sw = sh

        IR_node.attr["strides"].list.i.append(sh)
        IR_node.attr["strides"].list.i.append(sw)

        return IR_node
      
    @staticmethod
    def rename_MaxPooling2D(source_node, IR_node):
        # name, op and input
        KerasParser._copy_and_reop(source_node, IR_node, "MaxPool2D")

        # padding
        KerasParser._convert_padding(source_node, IR_node)

        # strides
        if isinstance(source_node.keras_layer.strides, tuple) or isinstance(source_node.keras_layer.strides, list):
            sh, sw = source_node.keras_layer.strides
        else:
            sh = source_node.keras_layer.strides
            sw = sh

        IR_node.attr["strides"].list.i.append(1)
        IR_node.attr["strides"].list.i.append(sh)
        IR_node.attr["strides"].list.i.append(sw)
        IR_node.attr["strides"].list.i.append(1)

        # pool_size
        if isinstance(source_node.keras_layer.pool_size, tuple) or isinstance(source_node.keras_layer.pool_size, list):
            ph, pw = source_node.keras_layer.pool_size
        else:
            ph = source_node.keras_layer.pool_size
            pw = ph
    
        IR_node.attr["ksize"].list.i.append(1)
        IR_node.attr["ksize"].list.i.append(sh)
        IR_node.attr["ksize"].list.i.append(sw)
        IR_node.attr["ksize"].list.i.append(1)


        return IR_node

    @staticmethod
    def rename_Dropout(source_node, IR_node):
        # only for training

        # name, op and input
        KerasParser._copy_and_reop(source_node, IR_node)

        IR_node.attr["keep_prob"].f = source_node.keras_layer.rate
        if source_node.keras_layer.seed != None:
            IR_node.attr["seed"].i = source_node.keras_layer.seed
        return IR_node
  

    @staticmethod
    def rename_Dense(source_node, IR_node):
        # name, op and input
        KerasParser._copy_and_reop(source_node, IR_node, "Fully_connected")
        
        # units
        IR_node.attr["units"].i = source_node.keras_layer.units

        # use_bias
        IR_node.attr["use_bias"].b = source_node.keras_layer.use_bias
        
        return IR_node

    @staticmethod
    def rename_Flatten(source_node, IR_node):
        # name, op and input
        KerasParser._copy_and_reop(source_node, IR_node)
        return IR_node


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
