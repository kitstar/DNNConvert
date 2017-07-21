# -*- coding: utf-8 -*-
'''ResNet50 model for Keras.

# Reference:

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

Adapted from code contributed by BigMoyan.
'''
from __future__ import print_function

import numpy as np
import warnings

import keras
from keras.layers import Input, Add
from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs


WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.

    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(include_top=True, weights='imagenet',
             input_tensor=None, input_shape=None,
             pooling=None,
             classes=1000):
    """Instantiates the ResNet50 architecture.

    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format="channels_last"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 244)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 197.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')
        
    input_1         = Input(shape = (224, 224, 3,), dtype = "float32")
    zero_padding2d_1 = ZeroPadding2D(name = "zero_padding2d_1", padding = ((3, 3),(3, 3),))(input_1)
    conv1           = Conv2D(filters = 64, kernel_size = (7, 7), strides = (2, 2), padding = 'valid', use_bias = True)(zero_padding2d_1)
    bn_conv1        = BatchNormalization(name = "bn_conv1", axis = 3)(conv1)
    activation_1    = Activation('relu')(bn_conv1)
    max_pooling2d_1 = MaxPooling2D(name = 'max_pooling2d_1', pool_size = (3, 3), strides = (2, 2), padding = 'valid')(activation_1)
    res2a_branch2a  = Conv2D(filters = 64, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', use_bias = True)(max_pooling2d_1)
    bn2a_branch2a   = BatchNormalization(name = "bn2a_branch2a", axis = 3)(res2a_branch2a)
    activation_2    = Activation('relu')(bn2a_branch2a)
    res2a_branch2b  = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same', use_bias = True)(activation_2)
    bn2a_branch2b   = BatchNormalization(name = "bn2a_branch2b", axis = 3)(res2a_branch2b)
    activation_3    = Activation('relu')(bn2a_branch2b)
    res2a_branch2c  = Conv2D(filters = 256, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', use_bias = True)(activation_3)
    bn2a_branch2c   = BatchNormalization(name = "bn2a_branch2c", axis = 3)(res2a_branch2c)
    res2a_branch1   = Conv2D(filters = 256, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', use_bias = True)(max_pooling2d_1)
    bn2a_branch1    = BatchNormalization(name = "bn2a_branch1", axis = 3)(res2a_branch1)
    add_1           = layers.add([bn2a_branch2c, bn2a_branch1])
    activation_4    = Activation('relu')(add_1)
    res2b_branch2a  = Conv2D(filters = 64, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', use_bias = True)(activation_4)
    bn2b_branch2a   = BatchNormalization(name = "bn2b_branch2a", axis = 3)(res2b_branch2a)
    activation_5    = Activation('relu')(bn2b_branch2a)
    res2b_branch2b  = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same', use_bias = True)(activation_5)
    bn2b_branch2b   = BatchNormalization(name = "bn2b_branch2b", axis = 3)(res2b_branch2b)
    activation_6    = Activation('relu')(bn2b_branch2b)
    res2b_branch2c  = Conv2D(filters = 256, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', use_bias = True)(activation_6)
    bn2b_branch2c   = BatchNormalization(name = "bn2b_branch2c", axis = 3)(res2b_branch2c)
    add_2           = layers.add([bn2b_branch2c, activation_4])
    activation_7    = Activation('relu')(add_2)
    res2c_branch2a  = Conv2D(filters = 64, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', use_bias = True)(activation_7)
    bn2c_branch2a   = BatchNormalization(name = "bn2c_branch2a", axis = 3)(res2c_branch2a)
    activation_8    = Activation('relu')(bn2c_branch2a)
    res2c_branch2b  = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same', use_bias = True)(activation_8)
    bn2c_branch2b   = BatchNormalization(name = "bn2c_branch2b", axis = 3)(res2c_branch2b)
    activation_9    = Activation('relu')(bn2c_branch2b)
    res2c_branch2c  = Conv2D(filters = 256, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', use_bias = True)(activation_9)
    bn2c_branch2c   = BatchNormalization(name = "bn2c_branch2c", axis = 3)(res2c_branch2c)
    add_3           = layers.add([bn2c_branch2c, activation_7])
    activation_10   = Activation('relu')(add_3)
    res3a_branch2a  = Conv2D(filters = 128, kernel_size = (1, 1), strides = (2, 2), padding = 'valid', use_bias = True)(activation_10)
    bn3a_branch2a   = BatchNormalization(name = "bn3a_branch2a", axis = 3)(res3a_branch2a)
    activation_11   = Activation('relu')(bn3a_branch2a)
    res3a_branch2b  = Conv2D(filters = 128, kernel_size = (3, 3), strides = (1, 1), padding = 'same', use_bias = True)(activation_11)
    bn3a_branch2b   = BatchNormalization(name = "bn3a_branch2b", axis = 3)(res3a_branch2b)
    activation_12   = Activation('relu')(bn3a_branch2b)
    res3a_branch2c  = Conv2D(filters = 512, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', use_bias = True)(activation_12)
    bn3a_branch2c   = BatchNormalization(name = "bn3a_branch2c", axis = 3)(res3a_branch2c)
    res3a_branch1   = Conv2D(filters = 512, kernel_size = (1, 1), strides = (2, 2), padding = 'valid', use_bias = True)(activation_10)
    bn3a_branch1    = BatchNormalization(name = "bn3a_branch1", axis = 3)(res3a_branch1)
    add_4           = layers.add([bn3a_branch2c, bn3a_branch1])
    activation_13   = Activation('relu')(add_4)
    res3b_branch2a  = Conv2D(filters = 128, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', use_bias = True)(activation_13)
    bn3b_branch2a   = BatchNormalization(name = "bn3b_branch2a", axis = 3)(res3b_branch2a)
    activation_14   = Activation('relu')(bn3b_branch2a)
    res3b_branch2b  = Conv2D(filters = 128, kernel_size = (3, 3), strides = (1, 1), padding = 'same', use_bias = True)(activation_14)
    bn3b_branch2b   = BatchNormalization(name = "bn3b_branch2b", axis = 3)(res3b_branch2b)
    activation_15   = Activation('relu')(bn3b_branch2b)
    res3b_branch2c  = Conv2D(filters = 512, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', use_bias = True)(activation_15)
    bn3b_branch2c   = BatchNormalization(name = "bn3b_branch2c", axis = 3)(res3b_branch2c)
    add_5           = layers.add([bn3b_branch2c, activation_13])
    activation_16   = Activation('relu')(add_5)
    res3c_branch2a  = Conv2D(filters = 128, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', use_bias = True)(activation_16)
    bn3c_branch2a   = BatchNormalization(name = "bn3c_branch2a", axis = 3)(res3c_branch2a)
    activation_17   = Activation('relu')(bn3c_branch2a)
    res3c_branch2b  = Conv2D(filters = 128, kernel_size = (3, 3), strides = (1, 1), padding = 'same', use_bias = True)(activation_17)
    bn3c_branch2b   = BatchNormalization(name = "bn3c_branch2b", axis = 3)(res3c_branch2b)
    activation_18   = Activation('relu')(bn3c_branch2b)
    res3c_branch2c  = Conv2D(filters = 512, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', use_bias = True)(activation_18)
    bn3c_branch2c   = BatchNormalization(name = "bn3c_branch2c", axis = 3)(res3c_branch2c)
    add_6           = layers.add([bn3c_branch2c, activation_16])
    activation_19   = Activation('relu')(add_6)
    res3d_branch2a  = Conv2D(filters = 128, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', use_bias = True)(activation_19)
    bn3d_branch2a   = BatchNormalization(name = "bn3d_branch2a", axis = 3)(res3d_branch2a)
    activation_20   = Activation('relu')(bn3d_branch2a)
    res3d_branch2b  = Conv2D(filters = 128, kernel_size = (3, 3), strides = (1, 1), padding = 'same', use_bias = True)(activation_20)
    bn3d_branch2b   = BatchNormalization(name = "bn3d_branch2b", axis = 3)(res3d_branch2b)
    activation_21   = Activation('relu')(bn3d_branch2b)
    res3d_branch2c  = Conv2D(filters = 512, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', use_bias = True)(activation_21)
    bn3d_branch2c   = BatchNormalization(name = "bn3d_branch2c", axis = 3)(res3d_branch2c)
    add_7           = layers.add([bn3d_branch2c, activation_19])
    activation_22   = Activation('relu')(add_7)
    res4a_branch2a  = Conv2D(filters = 256, kernel_size = (1, 1), strides = (2, 2), padding = 'valid', use_bias = True)(activation_22)
    bn4a_branch2a   = BatchNormalization(name = "bn4a_branch2a", axis = 3)(res4a_branch2a)
    activation_23   = Activation('relu')(bn4a_branch2a)
    res4a_branch2b  = Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), padding = 'same', use_bias = True)(activation_23)
    bn4a_branch2b   = BatchNormalization(name = "bn4a_branch2b", axis = 3)(res4a_branch2b)
    activation_24   = Activation('relu')(bn4a_branch2b)
    res4a_branch2c  = Conv2D(filters = 1024, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', use_bias = True)(activation_24)
    bn4a_branch2c   = BatchNormalization(name = "bn4a_branch2c", axis = 3)(res4a_branch2c)
    res4a_branch1   = Conv2D(filters = 1024, kernel_size = (1, 1), strides = (2, 2), padding = 'valid', use_bias = True)(activation_22)
    bn4a_branch1    = BatchNormalization(name = "bn4a_branch1", axis = 3)(res4a_branch1)
    add_8           = layers.add([bn4a_branch2c, bn4a_branch1])
    activation_25   = Activation('relu')(add_8)
    res4b_branch2a  = Conv2D(filters = 256, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', use_bias = True)(activation_25)
    bn4b_branch2a   = BatchNormalization(name = "bn4b_branch2a", axis = 3)(res4b_branch2a)
    activation_26   = Activation('relu')(bn4b_branch2a)
    res4b_branch2b  = Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), padding = 'same', use_bias = True)(activation_26)
    bn4b_branch2b   = BatchNormalization(name = "bn4b_branch2b", axis = 3)(res4b_branch2b)
    activation_27   = Activation('relu')(bn4b_branch2b)
    res4b_branch2c  = Conv2D(filters = 1024, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', use_bias = True)(activation_27)
    bn4b_branch2c   = BatchNormalization(name = "bn4b_branch2c", axis = 3)(res4b_branch2c)
    add_9           = layers.add([bn4b_branch2c, activation_25])
    activation_28   = Activation('relu')(add_9)
    res4c_branch2a  = Conv2D(filters = 256, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', use_bias = True)(activation_28)
    bn4c_branch2a   = BatchNormalization(name = "bn4c_branch2a", axis = 3)(res4c_branch2a)
    activation_29   = Activation('relu')(bn4c_branch2a)
    res4c_branch2b  = Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), padding = 'same', use_bias = True)(activation_29)
    bn4c_branch2b   = BatchNormalization(name = "bn4c_branch2b", axis = 3)(res4c_branch2b)
    activation_30   = Activation('relu')(bn4c_branch2b)
    res4c_branch2c  = Conv2D(filters = 1024, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', use_bias = True)(activation_30)
    bn4c_branch2c   = BatchNormalization(name = "bn4c_branch2c", axis = 3)(res4c_branch2c)
    add_10          = layers.add([bn4c_branch2c, activation_28])
    activation_31   = Activation('relu')(add_10)
    res4d_branch2a  = Conv2D(filters = 256, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', use_bias = True)(activation_31)
    bn4d_branch2a   = BatchNormalization(name = "bn4d_branch2a", axis = 3)(res4d_branch2a)
    activation_32   = Activation('relu')(bn4d_branch2a)
    res4d_branch2b  = Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), padding = 'same', use_bias = True)(activation_32)
    bn4d_branch2b   = BatchNormalization(name = "bn4d_branch2b", axis = 3)(res4d_branch2b)
    activation_33   = Activation('relu')(bn4d_branch2b)
    res4d_branch2c  = Conv2D(filters = 1024, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', use_bias = True)(activation_33)
    bn4d_branch2c   = BatchNormalization(name = "bn4d_branch2c", axis = 3)(res4d_branch2c)
    add_11          = layers.add([bn4d_branch2c, activation_31])
    activation_34   = Activation('relu')(add_11)
    res4e_branch2a  = Conv2D(filters = 256, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', use_bias = True)(activation_34)
    bn4e_branch2a   = BatchNormalization(name = "bn4e_branch2a", axis = 3)(res4e_branch2a)
    activation_35   = Activation('relu')(bn4e_branch2a)
    res4e_branch2b  = Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), padding = 'same', use_bias = True)(activation_35)
    bn4e_branch2b   = BatchNormalization(name = "bn4e_branch2b", axis = 3)(res4e_branch2b)
    activation_36   = Activation('relu')(bn4e_branch2b)
    res4e_branch2c  = Conv2D(filters = 1024, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', use_bias = True)(activation_36)
    bn4e_branch2c   = BatchNormalization(name = "bn4e_branch2c", axis = 3)(res4e_branch2c)
    add_12          = layers.add([bn4e_branch2c, activation_34])
    activation_37   = Activation('relu')(add_12)
    res4f_branch2a  = Conv2D(filters = 256, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', use_bias = True)(activation_37)
    bn4f_branch2a   = BatchNormalization(name = "bn4f_branch2a", axis = 3)(res4f_branch2a)
    activation_38   = Activation('relu')(bn4f_branch2a)
    res4f_branch2b  = Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), padding = 'same', use_bias = True)(activation_38)
    bn4f_branch2b   = BatchNormalization(name = "bn4f_branch2b", axis = 3)(res4f_branch2b)
    activation_39   = Activation('relu')(bn4f_branch2b)
    res4f_branch2c  = Conv2D(filters = 1024, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', use_bias = True)(activation_39)
    bn4f_branch2c   = BatchNormalization(name = "bn4f_branch2c", axis = 3)(res4f_branch2c)
    add_13          = layers.add([bn4f_branch2c, activation_37])
    activation_40   = Activation('relu')(add_13)
    res5a_branch2a  = Conv2D(filters = 512, kernel_size = (1, 1), strides = (2, 2), padding = 'valid', use_bias = True)(activation_40)
    bn5a_branch2a   = BatchNormalization(name = "bn5a_branch2a", axis = 3)(res5a_branch2a)
    activation_41   = Activation('relu')(bn5a_branch2a)
    res5a_branch2b  = Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), padding = 'same', use_bias = True)(activation_41)
    bn5a_branch2b   = BatchNormalization(name = "bn5a_branch2b", axis = 3)(res5a_branch2b)
    activation_42   = Activation('relu')(bn5a_branch2b)
    res5a_branch2c  = Conv2D(filters = 2048, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', use_bias = True)(activation_42)
    bn5a_branch2c   = BatchNormalization(name = "bn5a_branch2c", axis = 3)(res5a_branch2c)
    res5a_branch1   = Conv2D(filters = 2048, kernel_size = (1, 1), strides = (2, 2), padding = 'valid', use_bias = True)(activation_40)
    bn5a_branch1    = BatchNormalization(name = "bn5a_branch1", axis = 3)(res5a_branch1)
    add_14          = layers.add([bn5a_branch2c, bn5a_branch1])
    activation_43   = Activation('relu')(add_14)
    res5b_branch2a  = Conv2D(filters = 512, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', use_bias = True)(activation_43)
    bn5b_branch2a   = BatchNormalization(name = "bn5b_branch2a", axis = 3)(res5b_branch2a)
    activation_44   = Activation('relu')(bn5b_branch2a)
    res5b_branch2b  = Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), padding = 'same', use_bias = True)(activation_44)
    bn5b_branch2b   = BatchNormalization(name = "bn5b_branch2b", axis = 3)(res5b_branch2b)
    activation_45   = Activation('relu')(bn5b_branch2b)
    res5b_branch2c  = Conv2D(filters = 2048, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', use_bias = True)(activation_45)
    bn5b_branch2c   = BatchNormalization(name = "bn5b_branch2c", axis = 3)(res5b_branch2c)
    add_15          = layers.add([bn5b_branch2c, activation_43])
    activation_46   = Activation('relu')(add_15)
    res5c_branch2a  = Conv2D(filters = 512, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', use_bias = True)(activation_46)
    bn5c_branch2a   = BatchNormalization(name = "bn5c_branch2a", axis = 3)(res5c_branch2a)
    activation_47   = Activation('relu')(bn5c_branch2a)
    res5c_branch2b  = Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), padding = 'same', use_bias = True)(activation_47)
    bn5c_branch2b   = BatchNormalization(name = "bn5c_branch2b", axis = 3)(res5c_branch2b)
    activation_48   = Activation('relu')(bn5c_branch2b)
    res5c_branch2c  = Conv2D(filters = 2048, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', use_bias = True)(activation_48)
    bn5c_branch2c   = BatchNormalization(name = "bn5c_branch2c", axis = 3)(res5c_branch2c)
    add_16          = layers.add([bn5c_branch2c, activation_46])
    activation_49   = Activation('relu')(add_16)
    avg_pool        = AveragePooling2D(name = 'avg_pool', pool_size = (7, 7), strides = (7, 7), padding = 'valid')(activation_49)
    flatten_1       = Flatten()(avg_pool)
    fc1000          = Dense(units = 1000, use_bias = True)(flatten_1)
    fc1000_activation = Activation('softmax')(fc1000)
    model           = Model(inputs = [input_1], outputs = [fc1000_activation])
    


    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models',
                                    md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='avg_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1000')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    return model


if __name__ == '__main__':
    model = keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet')
    kit_model = ResNet50(include_top = True, weights = 'imagenet')

    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)

    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))

    preds = kit_model.predict(x)
    print('Kit Predicted:', decode_predictions(preds))
