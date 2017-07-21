# -*- coding: utf-8 -*-
'''VGG19 model for Keras.

# Reference:

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

'''
from __future__ import print_function

import numpy as np
import warnings

import keras
from keras.models import Model
from keras.layers import Flatten, Dense, Input, Activation
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs


WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'


def VGG19(include_top=True, weights='imagenet',
          input_tensor=None, input_shape=None,
          pooling=None,
          classes=1000):
    """Instantiates the VGG19 architecture.

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
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 244)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 48.
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
    block1_conv1    = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same', use_bias = True)(input_1)
    block1_conv1_activation = Activation('relu')(block1_conv1)
    block1_conv2    = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same', use_bias = True)(block1_conv1_activation)
    block1_conv2_activation = Activation('relu')(block1_conv2)
    block1_pool     = MaxPooling2D(name = 'block1_pool', pool_size = (2, 2), strides = (2, 2), padding = 'valid')(block1_conv2_activation)
    block2_conv1    = Conv2D(filters = 128, kernel_size = (3, 3), strides = (1, 1), padding = 'same', use_bias = True)(block1_pool)
    block2_conv1_activation = Activation('relu')(block2_conv1)
    block2_conv2    = Conv2D(filters = 128, kernel_size = (3, 3), strides = (1, 1), padding = 'same', use_bias = True)(block2_conv1_activation)
    block2_conv2_activation = Activation('relu')(block2_conv2)
    block2_pool     = MaxPooling2D(name = 'block2_pool', pool_size = (2, 2), strides = (2, 2), padding = 'valid')(block2_conv2_activation)
    block3_conv1    = Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), padding = 'same', use_bias = True)(block2_pool)
    block3_conv1_activation = Activation('relu')(block3_conv1)
    block3_conv2    = Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), padding = 'same', use_bias = True)(block3_conv1_activation)
    block3_conv2_activation = Activation('relu')(block3_conv2)
    block3_conv3    = Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), padding = 'same', use_bias = True)(block3_conv2_activation)
    block3_conv3_activation = Activation('relu')(block3_conv3)
    block3_conv4    = Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), padding = 'same', use_bias = True)(block3_conv3_activation)
    block3_conv4_activation = Activation('relu')(block3_conv4)
    block3_pool     = MaxPooling2D(name = 'block3_pool', pool_size = (2, 2), strides = (2, 2), padding = 'valid')(block3_conv4_activation)
    block4_conv1    = Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), padding = 'same', use_bias = True)(block3_pool)
    block4_conv1_activation = Activation('relu')(block4_conv1)
    block4_conv2    = Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), padding = 'same', use_bias = True)(block4_conv1_activation)
    block4_conv2_activation = Activation('relu')(block4_conv2)
    block4_conv3    = Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), padding = 'same', use_bias = True)(block4_conv2_activation)
    block4_conv3_activation = Activation('relu')(block4_conv3)
    block4_conv4    = Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), padding = 'same', use_bias = True)(block4_conv3_activation)
    block4_conv4_activation = Activation('relu')(block4_conv4)
    block4_pool     = MaxPooling2D(name = 'block4_pool', pool_size = (2, 2), strides = (2, 2), padding = 'valid')(block4_conv4_activation)
    block5_conv1    = Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), padding = 'same', use_bias = True)(block4_pool)
    block5_conv1_activation = Activation('relu')(block5_conv1)
    block5_conv2    = Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), padding = 'same', use_bias = True)(block5_conv1_activation)
    block5_conv2_activation = Activation('relu')(block5_conv2)
    block5_conv3    = Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), padding = 'same', use_bias = True)(block5_conv2_activation)
    block5_conv3_activation = Activation('relu')(block5_conv3)
    block5_conv4    = Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), padding = 'same', use_bias = True)(block5_conv3_activation)
    block5_conv4_activation = Activation('relu')(block5_conv4)
    block5_pool     = MaxPooling2D(name = 'block5_pool', pool_size = (2, 2), strides = (2, 2), padding = 'valid')(block5_conv4_activation)
    flatten         = Flatten()(block5_pool)
    fc1             = Dense(units = 4096, use_bias = True)(flatten)
    fc1_activation  = Activation('relu')(fc1)
    fc2             = Dense(units = 4096, use_bias = True)(fc1_activation)
    fc2_activation  = Activation('relu')(fc2)
    predictions     = Dense(units = 1000, use_bias = True)(fc2_activation)
    predictions_activation = Activation('softmax')(predictions)
    model           = Model(inputs = [input_1], outputs = [predictions_activation])
    
    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('vgg19_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models')
        else:
            weights_path = get_file('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models')
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='block5_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1')
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

    # save as JSON
    json_string = model.to_json()
    with open("vgg19.json", "w") as of:
        of.write(json_string)
                
    return model


if __name__ == '__main__':
    model = keras.applications.vgg19.VGG19(include_top=True, weights='imagenet')
    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)

    preds = model.predict(x)
    print('Src Predicted:', decode_predictions(preds))
    
    kit_model = VGG19(include_top=True, weights='imagenet')
    preds = kit_model.predict(x)
    print('Kit Predicted:', decode_predictions(preds))
 
