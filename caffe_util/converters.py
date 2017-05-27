from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

CAFFE_LOADER = None

class CaffeTensorflowConverter(object):
    
    def __init__(self, src_path, dst_path, phase = 'eval'): 
        if src_path == None:
            fatal_error('Caffe to Tensorflow Converter: source path is empty')
        if dst_path == None:
            fatal_error('Caffe to Tensorflow Converter: destination path is empty')
        self.phase = phase
        self.load(src_path)

    # Building Graph
    def load(self, src_path):
        try:
            import caffe
            self.caffe = caffe
        except ImportError:
            print ("No caffe!")

        if self.caffe:
            self.caffepb = self.caffe.proto.caffe_pb2
        self.NetParameter = self.caffepb.NetParameter

    def transform_model(self):
        pass
    
    def get_caffe_loader():
        global CAFFE_LOADER
        if CAFFE_LOADER is None:
            CAFFE_LOADER = CaffeLoader()
        return CAFFE_LOADER



