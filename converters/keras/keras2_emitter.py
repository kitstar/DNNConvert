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
            graph_pb2.DT_INT64 : "int64", 
            "uint8"   : graph_pb2.DT_UINT8,
            "uint16"  : graph_pb2.DT_UINT16
            }

    activation_map = {
            "relu"    : "Relu",
            'softmax' : "Softmax",
            'sigmoid' : "Sigmoid",
            "tanh"    : "Tanh"
            }
    

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
        code = "{:<15} = Input(shape = ({},), dtype = \"{}\")".format(
                IR_node.name, shape_str, 
                self.dtype_map[IR_node.IR_layer.attr["dtype"].type])
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
    def emit_GlobalMaxPool1D(self, IR_node):
        code = "{:<15} = GlobalMaxPooling1D()({})".format(
                IR_node.name, 
                IR_node.in_edges[0])
        return code



    @classmethod
    def emit_MaxPool2D1(self, IR_node):
        code = "no implement"
        return code



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
    def emit_Tanh(self, IR_node):
        code = "{:<15} = Activation(\'tanh\')({})".format(
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
    def emit_Embedding(self, IR_node):
        ret = "{:<15} = Embedding(input_dim = {}, output_dim = {}, mask_zero = {})({})".format(
                IR_node.name, 
                IR_node.IR_layer.attr['input_dim'].i,
                IR_node.IR_layer.attr['output_dim'].i,
                IR_node.IR_layer.attr['mask_zero'].b,
                IR_node.in_edges[0])

        return ret



    @classmethod
    def emit_RNNs(self, IR_node, func):
        # for Keras
        if "dropout" in IR_node.IR_layer.attr:
            dropout_str = ",dropout = {}, recurrent_dropout = {}".format(
                    IR_node.IR_layer.attr['dropout'].f,
                    IR_node.IR_layer.attr['recurrent_dropout'].f)
        else:
            dropout_str = ""
        
        code = "{:<15} = {}(units = {}, use_bias = {} {})({})".format(
                IR_node.name, 
                func,
                IR_node.IR_layer.attr['units'].i,
                IR_node.IR_layer.attr['use_bias'].b,
                dropout_str,
                IR_node.in_edges[0])

        return code



    @classmethod
    def emit_LSTM(self, IR_node):
        return self.emit_RNNs(IR_node, "LSTM")



    @classmethod
    def emit_GRU(self, IR_node):
        return self.emit_RNNs(IR_node, "GRU")



    @classmethod
    def emit_Add(self, IR_node):
        inputs = listToStr(IR_node.in_edges)
        code = "{:<15} = Add()({})".format(
                IR_node.name, 
                inputs)
        return code



    @classmethod
    def emit_Concat(self, IR_node):
        inputs = listToStr(IR_node.in_edges)
        code = "{:<15} = Concatenate()({})".format(
                IR_node.name, 
                inputs)
        return code


    @classmethod
    def emit_BatchNorm(self, IR_node):
        code = "{:<15} = BatchNormalization(name = {}, axis = {})({})".format(
                IR_node.name,
                IR_node.name,
                IR_node.IR_layer.attr['axis'].i,
                IR_node.in_edges[0])
        return code




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
