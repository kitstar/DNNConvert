from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import keras as _keras
from converters.keras.keras2_graph import Keras2Graph
import common.IR.graph_pb2 as graph_pb2
from common.IR.graph_pb2 import NodeDef, GraphDef, DataType


class Keras2Parser(object):
   
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

    activation_map = {
            "relu"    : "Relu",
            'softmax' : "Softmax",
            'sigmoid' : "Sigmoid",
            "tanh"    : "Tanh"
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



    @classmethod
    def __init__(self, model):
        self.IR_graph = GraphDef()
        # load model files into Keras graph
        if isinstance(model, basestring):
            model = _keras.models.load_model(model)
        elif isinstance(model, tuple):
            model = Keras2Parser._load_model(model[0], model[1])

        _keras.utils.plot_model(model, "model.png", show_shapes = True)

        # Build network graph
        self.data_format = _keras.backend.image_data_format()
        self.keras_graph =  Keras2Graph(model)
        self.keras_graph.build()



    @classmethod
    def gen_IR(self):
        # bfs
        traverse_nodes = self.keras_graph.get_input_layers()
        while len(traverse_nodes) > 0:
            current_node = self.keras_graph.get_node(traverse_nodes.pop())
            node_type = current_node.type

            if hasattr(self, "rename_" + node_type):
                func = getattr(self, "rename_" + node_type)
                func(current_node)
            else:
                print("KerasParser has not supported operator [%s]." % (node_type))
                self.rename_UNKNOWN(current_node)

            for next_node in current_node.out_edges:
                next_node_info = self.keras_graph.get_node(next_node)
                next_node_info.left_in_edges -= 1
                if next_node_info.left_in_edges == 0:
                    traverse_nodes.append(next_node)



    @staticmethod
    def _copy_and_reop(source_node, IR_node, new_op = None):
        node_info = source_node.keras_layer
        if new_op == None:
            new_op = source_node.type
        IR_node.name = source_node.name
        IR_node.op = new_op
        if hasattr(node_info, "dtype"):
            IR_node.attr["dtype"].type = Keras2Parser.dtype_map[node_info.dtype]



    @staticmethod
    def _convert_inedge(source_node, IR_node, layer_name_map):
        for e in source_node.in_edges:
            IR_node.input.append(layer_name_map[e])



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


    @staticmethod
    def _convert_dataformat(source_node, target_node):
        if source_node.keras_layer.data_format == 'channels_last':
            target_node.attr["data_format"].s = "NHWC"
        elif source_node.keras_layer.data_format == 'channels_first':
            target_node.attr["data_format"].s = "NCHW"
        else:
            print("Warning: [%s] don't have data format info." % (source_node.keras_layer.name))



    @staticmethod
    def _convert_padding(source_node, target_node):
        if source_node.keras_layer.padding == 'valid':
            target_node.attr["padding"].s = "VALID"
        elif source_node.keras_layer.padding == 'same':
            target_node.attr["padding"].s = "SAME"
        else:
            print ("Error: Invalid embedding [%s]!" % (source_node.keras_layer.padding))



    @classmethod
    def _defuse_activation(self, keras_node):
        if keras_node.keras_layer.activation == None:
            return

        if keras_node.keras_layer.activation.__name__ == "linear":
            return

        IR_node = self.IR_graph.node.add()
        IR_node.name = keras_node.keras_layer.name + "_activation"
        IR_node.op = Keras2Parser.activation_map[keras_node.keras_layer.activation.__name__]
        IR_node.input.append(keras_node.keras_layer.name)
        self.keras_graph.layer_name_map[keras_node.keras_layer.name] = IR_node.name



    @classmethod
    def _convert_convolution(self, keras_node, IR_node, dim):
         # name, op
        Keras2Parser._copy_and_reop(keras_node, IR_node)

        # input edge
        Keras2Parser._convert_inedge(keras_node, IR_node, self.keras_graph.layer_name_map)
        
        # padding        
        Keras2Parser._convert_padding(keras_node, IR_node)

        # filter
        for e in keras_node.keras_layer.kernel_size:
            IR_node.attr["filter"].list.i.append(e)

        if self.data_format == "channels_last":
            IR_node.attr["filter"].list.i.append(keras_node.keras_layer.input_shape[-1])
        else:
            IR_node.attr["filter"].list.i.append(keras_node.keras_layer.input_shape[1])
        IR_node.attr["filter"].list.i.append(keras_node.keras_layer.filters)

        # use_bias
        IR_node.attr["use_bias"].b = keras_node.keras_layer.use_bias

        # strides
        for e in keras_node.keras_layer.strides:
            IR_node.attr["strides"].list.i.append(e)

        while len(IR_node.attr["strides"].list.i) < dim:
            IR_node.attr["strides"].list.i.append(IR_node.attr["strides"].list.i.at(0))

        # activation
        self._defuse_activation(keras_node)


    @classmethod
    def rename_UNKNOWN(self, source_node):
        # only for training
        IR_node = self.IR_graph.node.add()
        
        # name, op
        Keras2Parser._copy_and_reop(source_node, IR_node)
        
        # input edge
        Keras2Parser._convert_inedge(source_node, IR_node, self.keras_graph.layer_name_map)



    @classmethod
    def rename_InputLayer(self, source_node):
        # only for training
        IR_node = self.IR_graph.node.add()
        
        # name, op
        Keras2Parser._copy_and_reop(source_node, IR_node, "DataInput")
        
        # input edge
        Keras2Parser._convert_inedge(source_node, IR_node, self.keras_graph.layer_name_map)

        # shape
        Keras2Parser._copy_shape(source_node.keras_layer, IR_node)



    @classmethod
    def rename_Conv1D(self, keras_node):
        IR_node = self.IR_graph.node.add()        
        self._convert_convolution(keras_node, IR_node, 1)



    @classmethod
    def rename_Conv2D(self, keras_node):
        IR_node = self.IR_graph.node.add()         
        self._convert_convolution(keras_node, IR_node, 2)



    @classmethod
    def rename_Conv3D(self, source_node):
        IR_node = self.IR_graph.node.add()         
        self._convert_convolution(keras_node, IR_node, 3)
       


    @classmethod
    def rename_GlobalMaxPooling1D(self, source_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        Keras2Parser._copy_and_reop(source_node, IR_node, "GlobalMaxPool1D")

        # input edge
        Keras2Parser._convert_inedge(source_node, IR_node, self.keras_graph.layer_name_map)



    @classmethod
    def rename_MaxPooling2D(self, source_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        Keras2Parser._copy_and_reop(source_node, IR_node, "MaxPool2D")

        # input edge
        Keras2Parser._convert_inedge(source_node, IR_node, self.keras_graph.layer_name_map)

        # padding
        Keras2Parser._convert_padding(source_node, IR_node)

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




    @classmethod
    def rename_Dropout(self, source_node):
        # only for training
        IR_node = self.IR_graph.node.add()

        # name, op
        Keras2Parser._copy_and_reop(source_node, IR_node)

        # input edge
        Keras2Parser._convert_inedge(source_node, IR_node, self.keras_graph.layer_name_map)

        IR_node.attr["keep_prob"].f = source_node.keras_layer.rate
        if source_node.keras_layer.seed != None:
            IR_node.attr["seed"].i = source_node.keras_layer.seed
  


    @classmethod
    def rename_Dense(self, source_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        Keras2Parser._copy_and_reop(source_node, IR_node, "Fully_connected")
        
        # input edge
        Keras2Parser._convert_inedge(source_node, IR_node, self.keras_graph.layer_name_map)

        # units
        IR_node.attr["units"].i = source_node.keras_layer.units

        # use_bias
        IR_node.attr["use_bias"].b = source_node.keras_layer.use_bias

        # activation
        self._defuse_activation(source_node)



    @classmethod
    def rename_Flatten(self, source_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        Keras2Parser._copy_and_reop(source_node, IR_node)

        # input edge
        Keras2Parser._convert_inedge(source_node, IR_node, self.keras_graph.layer_name_map)



    @classmethod
    def rename_Activation(self, keras_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        Keras2Parser._copy_and_reop(keras_node, IR_node, self.activation_map[keras_node.keras_layer.activation.__name__])

        # input edge
        Keras2Parser._convert_inedge(keras_node, IR_node, self.keras_graph.layer_name_map)



    @classmethod
    def rename_Embedding(self, source_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        Keras2Parser._copy_and_reop(source_node, IR_node)
        
        # input edge
        Keras2Parser._convert_inedge(source_node, IR_node, self.keras_graph.layer_name_map)

        # input_dim
        IR_node.attr["input_dim"].i = source_node.keras_layer.input_dim

        # output_dim
        IR_node.attr["output_dim"].i = source_node.keras_layer.output_dim

        # mask_zero
        IR_node.attr["mask_zero"].b = source_node.keras_layer.mask_zero



    @classmethod
    def rename_LSTM(self, keras_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        Keras2Parser._copy_and_reop(keras_node, IR_node)
        
        # input edge
        Keras2Parser._convert_inedge(keras_node, IR_node, self.keras_graph.layer_name_map)

        # units
        IR_node.attr["units"].i = keras_node.keras_layer.units

        # use_bias
        IR_node.attr["use_bias"].b = keras_node.keras_layer.use_bias

        # for Keras, drop_out and recurrent_dropout
        IR_node.attr["dropout"].f = keras_node.keras_layer.dropout
        IR_node.attr["recurrent_dropout"].f = keras_node.keras_layer.recurrent_dropout

        # activation
        self._defuse_activation(keras_node)



    @classmethod
    def rename_GRU(self, source_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        Keras2Parser._copy_and_reop(source_node, IR_node)
        
        # input edge
        Keras2Parser._convert_inedge(source_node, IR_node, self.keras_graph.layer_name_map)

        # units
        IR_node.attr["units"].i = source_node.keras_layer.units

        # activation
        self._defuse_activation(source_node)



    @classmethod
    def rename_Add(self, source_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        Keras2Parser._copy_and_reop(source_node, IR_node)
        
        # input edge
        Keras2Parser._convert_inedge(source_node, IR_node, self.keras_graph.layer_name_map)



    @classmethod
    def rename_Concatenate(self, source_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        Keras2Parser._copy_and_reop(source_node, IR_node, 'Concat')
        
        # input edge
        Keras2Parser._convert_inedge(source_node, IR_node, self.keras_graph.layer_name_map)


    @classmethod
    def rename_Reshape(self, source_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        Keras2Parser._copy_and_reop(source_node, IR_node, 'Concat')
        
        # input edge
        Keras2Parser._convert_inedge(source_node, IR_node, self.keras_graph.layer_name_map)

        # for target shape
        for e in source_node.keras_layer.target_shape:
            IR_node.attr["Tshape"].list.i.append(e)


    @classmethod
    def rename_Lambda(self, source_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        Keras2Parser._copy_and_reop(source_node, IR_node, "Keras Lambda")
        
        # input edge
        Keras2Parser._convert_inedge(source_node, IR_node, self.keras_graph.layer_name_map)

        IR_node.attr['function'].s = source_node.keras_layer.function.__name__
        for dim in source_node.keras_layer.output_shape:
            new_dim = IR_node.attr["output_shape"].shape.dim.add()
            if dim == None:
                new_dim.size = -1
            else:
                new_dim.size = dim

        # arguments not implementent
        #print (type(source_node.keras_layer.arguments))



    @classmethod
    def rename_BatchNormalization(self, keras_node):
        IR_node = self.IR_graph.node.add()

        # name, op
        Keras2Parser._copy_and_reop(keras_node, IR_node, 'BatchNorm')
        
        # input edge
        Keras2Parser._convert_inedge(keras_node, IR_node, self.keras_graph.layer_name_map)

        # axis
        IR_node.attr['axis'].i = keras_node.keras_layer.axis
