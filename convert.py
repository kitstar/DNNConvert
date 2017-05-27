from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse


FLAGS = None


def fatal_error(msg):
    print(msg)
    exit(-1)


def main():
    if FLAGS.src_tool == 'caffe':
        if FLAGS.dst_tool == 'tf':
            from caffe_util.converters import CaffeTensorflowConverter
            transformer = CaffeTensorflowConverter(FLAGS.src_model_path, FLAGS.dst_model_path, FLAGS.phase)

    elif FLAGS.src_tool == 'tf':
        if FLAGS.dst_tool == 'caffe':
            try:
                from caffe_util.converters import TensorflowCaffeConverter
            except:
                pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument(
            '--src_tool', '-st',
            type=str,
            default='caffe',
            help='source toolkit name: caffe/tf/cntk')

    parser.add_argument(
            '--src_model_path', '-sm',
            type=str,
            default=None,
            help='source model path.')
    
    parser.add_argument(
            '--dst_tool', '-dt',
            type=str,
            default='tf',
            help='destination toolkit name: caffe/tf/cntk')

    parser.add_argument(
            '--dst_model_path', '-dm',
            type=str,
            default='./model/',
            help='destination model path')

    parser.add_argument(
            '--phase', '-p',
            type=str,
            default='eval',
            help='train/eval/all')

    FLAGS = parser.parse_args()
    print("Argument = ", FLAGS)

    main()
