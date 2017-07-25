# Copyright (c) 2017, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import logging as _logging
import sys as _sys
import google.protobuf.text_format as text_format
# expose files as imports
#from ..models import utils

#from ..models import neural_network
#from ..models import MLModel
import converters
#from .. import proto



def _convert(args):
    if args.srcModelFormat == 'auto':
        if args.srcModelPath.endswith('.caffemodel') or args.caffeProtoTxtPath != '':
            args.srcModelFormat = 'caffe'

        elif args.srcModelPath.endswith('.h5') or args.kerasJsonPath != '':
            args.srcModelFormat = 'keras'

        else:
            print("error: coremlconverter: Unable to auto-detect model format. "
                  "Please specify the model format using the 'srcModelFormat' argument.")
            _sys.exit(1)

    if args.srcModelFormat == 'caffe':
        if args.caffeProtoTxtPath:
            if args.meanImageProtoPath:
                model = (args.srcModelPath, args.caffeProtoTxtPath, args.meanImageProtoPath)
            else:
                model = (args.caffeProtoTxtPath, args.srcModelPath)
        else:
            model = args.srcModelPath

        from converters.caffe.caffe_parser import CaffeParser
        parser = CaffeParser(model, args.caffePhase)
        parser.gen_IR()
        parser.saveToJson("kit_model.json")

        proto_str = parser.IR_graph.SerializeToString()
        with open('kit_model.prototxt', 'wb') as of:
            of.write(proto_str)

        return 0

    elif args.srcModelFormat == 'keras':
#        try:
            if not args.inputNames:
                raise TypeError("Neural network 'inputNames' are required for converting Keras models.")
            if not args.outputNames:
                raise TypeError("Neural network 'outputNames' are required for converting Keras models.")

            if args.kerasJsonPath:
                model = (args.kerasJsonPath, args.srcModelPath)
            else:
                model = args.srcModelPath
            
            from converters.keras.keras2_parser import Keras2Parser
            parser = Keras2Parser(model)
            parser.gen_IR()
            parser.saveToJson("kit_model.json")

            proto_str = parser.IR_graph.SerializeToString()
            with open('kit_model.prototxt', 'wb') as of:
                of.write(proto_str)

            
            from converters.keras.keras2_emitter import Keras2Emitter
            emitter = Keras2Emitter("kit_model.prototxt")
            emitter.gen_code("kit_new_model.py")

            return 0
    else:
        print('error: Invalid srcModelFormat specified.')
        return 1

def _main():
    import argparse

    parser = argparse.ArgumentParser(description='Convert other model file formats to MLKit format (.mlmodel).')
    parser.add_argument('--srcModelFormat', type=unicode, choices=['auto', 'caffe', 'keras'], default='auto', help='Format of model at srcModelPath (default is to auto-detect).')
    parser.add_argument('--srcModelPath', type=unicode, required=True, help='Path to the model file of the external tool (e.g caffe weights proto binary, keras h5 binary')
    parser.add_argument('--dstModelPath', type=unicode, required=True, help='Path to save the model in format .mlmodel')
    parser.add_argument('--caffeProtoTxtPath', type=unicode, default='', help='Path to the .prototxt file if network differs from the source file (optional)')
    parser.add_argument('--meanImageProtoPath', type=unicode, default='', help='Path to the .binaryproto file containing the mean image if required by the network (optional). This requires a prototxt file to be specified.')
    parser.add_argument('--kerasJsonPath', type=unicode, default=None, help='Path to the .json file for keras if the network differs from the weights file (optional)')
    parser.add_argument('--inputNames', type=unicode, nargs='*', help='Names of the feature (input) columns, in order (required for keras models).')
    parser.add_argument('--outputNames', type=unicode, nargs='*', help='Names of the target (output) columns, in order (required for keras models).')
    parser.add_argument('--imageInputNames', type=unicode, default=[], action='append', help='Label the named input as an image. Can be specified more than once for multiple image inputs.')
    parser.add_argument('--isBGR', action='store_true', default=False, help='True if the image data in BGR order (RGB default)')
    parser.add_argument('--redBias', type=float, default=0.0, help='Bias value to be added to the red channel (optional, default 0.0)')
    parser.add_argument('--blueBias', type=float, default=0.0, help='Bias value to be added to the blue channel (optional, default 0.0)')
    parser.add_argument('--greenBias', type=float, default=0.0, help='Bias value to be added to the green channel (optional, default 0.0)')
    parser.add_argument('--grayBias', type=float, default=0.0, help='Bias value to be added to the gray channel for Grayscale images (optional, default 0.0)')
    parser.add_argument('--scale', type=float, default=1.0, help='Value by which the image data must be scaled (optional, default 1.0)')
    parser.add_argument('--classInputPath', type=unicode, default='', help='Path to class labels (ordered new line separated) for treating the neural network as a classifier')
    parser.add_argument('--predictedFeatureName', type=unicode, default='class_output', help='Name of the output feature that captures the class name (for classifiers models).')
    parser.add_argument('--caffePhase', type=unicode, default='TRAIN', help='Convert the specific phase of caffe model.')

    args = parser.parse_args()
    ret = _convert(args)
    _sys.exit(int(ret)) # cast to int or else the exit code is always 1

