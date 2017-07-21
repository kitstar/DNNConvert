from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import keras as _keras
from common.IR.IR_graph import IRGraph
import common.IR.graph_pb2 as graph_pb2
from common.IR.graph_pb2 import NodeDef, GraphDef, DataType
from common.utils import listToStr


class Keras2Emitter(object):
   
    dtype_map = {
            graph_pb2.DT_FLOAT16 : "float16",
            graph_pb2.DT_FLOAT32 : "float32",
            graph_pb2.DT_FLOAT64 : "float64",
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
    def __init__(self, filename):
        self.IR_graph = IRGraph(filename)
        self.IR_graph.build()



    @classmethod
    def gen_code(self, output_filename):
        # bfs
        traverse_nodes = self.IR_graph.get_input_layers()[:]
        while len(traverse_nodes) > 0:
            current_node = self.IR_graph.get_node(traverse_nodes.pop())
            node_type = current_node.type

            if hasattr(self, "emit_" + node_type):
                func = getattr(self, "emit_" + node_type)
                line = func(current_node)
                print (line)
            else:
                print("KerasEmitter has not supported operator [%s]." % (node_type))
                self.emit_UNKNOWN(current_node)

            for next_node in current_node.out_edges:
                next_node_info = self.IR_graph.get_node(next_node)
                next_node_info.left_in_edges -= 1
                if next_node_info.left_in_edges == 0:
                    traverse_nodes.append(next_node)


        last_line = "{:<15} = Model(inputs = [{}], outputs = [{}])".format(
                "model",
                listToStr(self.IR_graph.get_input_layers()),
                listToStr(self.IR_graph.get_output_layers()))
        print (last_line)



    @staticmethod
    def _convert_padding(source_node, target_node):
        if source_node.keras_layer.padding == 'valid':
            target_node.attr["padding"].s = "VALID"
        elif source_node.keras_layer.padding == 'same':
            target_node.attr["padding"].s = "SAME"
        else:
            print ("Error: Invalid embedding [%s]!" % (source_node.keras_layer.padding))


    
    @staticmethod
    def _emit_convolution(IR_node):
        dim = len(IR_node.IR_layer.attr["strides"].list.i)

        filter = IR_node.IR_layer.attr["filter"].list.i[dim + 1]
        
        kernel = list()
        for idx in range(0, dim):
            kernel.append(IR_node.IR_layer.attr["filter"].list.i[idx])
        kernel = listToStr(kernel)

        strides = list()
        for e in IR_node.IR_layer.attr["strides"].list.i:
            strides.append(e)
        strides = listToStr(strides)

        use_bias = IR_node.IR_layer.attr["use_bias"].b 

        padding = IR_node.IR_layer.attr["padding"].s
        padding = padding.lower()

        ret = "{:<15} = Conv{}D(filters = {}, kernel_size = ({}), strides = ({}), padding = \'{}\', use_bias = {})({})".format(
                IR_node.name, 
                dim,
                filter,
                kernel,
                strides,
                padding,
                use_bias,
                IR_node.in_edges[0])

        return ret



    @classmethod
    def emit_UNKNOWN(self, IR_node):
        print(IR_node.IR_layer.name)



    @classmethod
    def emit_DataInput(self, IR_node):
        shape_str = IRGraph.shapeToStr(IR_node.IR_layer.attr["shape"].shape)
        code = "{:<15} = Input(shape = ({},), dtype = \"{}\")".format(IR_node.IR_layer.name, shape_str, self.dtype_map[IR_node.IR_layer.attr["dtype"].type])
        return code



    @classmethod
    def emit_Conv1D(self, IR_node):
        return Keras2Emitter._emit_convolution(IR_node)



    @classmethod
    def emit_Conv2D(self, IR_node):
        return Keras2Emitter._emit_convolution(IR_node)



    @classmethod
    def emit_Conv3D(self, source_node):
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
    def emit_Dropout(self, IR_node):
        seed = 'None'
        if 'seed' in IR_node.IR_layer.attr:
            seed = IR_node.IR_layer.attr['seed'].i

        ret = "{:<15} = Dropout(rate = {}, seed = {})({})".format(
                IR_node.name,
                IR_node.IR_layer.attr["keep_prob"].f,
                seed,
                IR_node.in_edges[0])

        return ret
 


    @classmethod
    def emit_Fully_connected(self, IR_node):
        units = IR_node.IR_layer.attr["units"].i
        use_bias = IR_node.IR_layer.attr["use_bias"].b

        ret = "{:<15} = Dense(units = {}, use_bias = {})({})".format(
                IR_node.name, 
                units,
                use_bias,
                IR_node.in_edges[0])

        return ret



    @classmethod
    def emit_Flatten(self, IR_node):
        code = "{:<15} = Flatten()({})".format(
            IR_node.name, IR_node.in_edges[0])
        return code



    @classmethod
    def emit_Relu(self, IR_node):
        code = "{:<15} = Activation(\'relu\')({})".format(
                IR_node.name, IR_node.in_edges[0])
        return code



    @classmethod
    def emit_Softmax(self, IR_node):
        code = "{:<15} = Activation(\'softmax\')({})".format(
                IR_node.name, IR_node.in_edges[0])
        return code



    @classmethod
    def emit_Sigmoid(self, IR_node):
        code = "{:<15} = Activation(\'sigmoid\')({})".format(
                IR_node.name, IR_node.in_edges[0])
        return code




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
    def rename_LSTM(self, source_node):
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

