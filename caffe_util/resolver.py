from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

CAFFE_LOADER = None

class CaffeLoader(object):
    
    def __init__(self):
        self.caffe = None
        self.import_caffe()

    def import_caffe(self):
        try:
            import caffe
            self.caffe = caffe
        except ImportError:
            print ("No caffe!")

        if self.caffe:
            self.caffepb = self.caffe.proto.caffe_pb2
        self.NetParameter = self.caffepb.NetParameter

    def get_caffe_loader():
        global CAFFE_LOADER
        if CAFFE_LOADER is None:
            CAFFE_LOADER = CaffeLoader()
        return CAFFE_LOADER



