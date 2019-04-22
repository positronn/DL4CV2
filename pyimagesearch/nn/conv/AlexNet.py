# -*- coding: utf-8 -*-
# AlexNet.py
'''
Implementation of the AlexNet architecture

The first block of AlexNet applies 96, 11 × 11 kernels with a stride of 4 × 4,
followed by a RELU activation and max pooling with a pool size of 3 × 3 and
strides of 2 × 2, resulting in an output volume of size 55 × 55.

th 1 × 1 strides. After applying max pooling again with a pool size of 3 × 3
and strides of 2 × 2 we are left with a 13 × 13 volume.
Next, we apply (CONV => RELU) * 3 => POOL. The first two CONV layers learn 384,
3 × 3 filters while the final CONV learns 256, 3 × 3 filters.
After another max pooling operation, we reach our two FC layers, each with
4096 nodes and RELU activations in between. The final layer in the network
is our softmax classifier.
'''
from keras import backend as K
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Activation
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization


class AlexNet:
    '''
    '''
    @staticmethod
    def build(width, height, depth, classes, reg = 0.0002):
        '''
        Build the AlexNet architecture given width, height and depth
        as dimensions of the input tensor and the corresponding
        number of classes of the data.
        parameters
        ----------
            width:  width of input images.
            height: height of input images.
            depth:  depth if input images.
            classes: number of classes of the corresponding data.
            reg:    controls the amount of L2 regularization applied to the network
        returns
        -------
            model: the LeNet model compatible with given inputs
                    as a keras sequential model.
        '''
        # initialize the model along with the input shape 
        # to be "channels last" and the channels dimensionn itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
        
        # block #1: first CONV -*-> RELU --> POOL layer set
        model.add(Conv2D(96, (11, 11), strides = (4, 4),
            input_shape = inputShape, padding = "same", kernel_regularizer = l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = chanDim))
        model.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2)))
        model.add(Dropout(0.25))

        # block #2: second CONV -*-> RELU --> POOL layer set
        model.add(Conv2D(256, (5, 5), padding = "same",
            kernel_regularizer = l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = chanDim))
        model.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2)))
        model.add(Dropout(0.25))

        # block #3: CONV -*-> RELU --> CONV -*-> RELU --> CONV -*-> RELU
        model.add(Conv2D(384, (3, 3), padding = "same",
            kernel_regularizer = l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = chanDim))
        model.add(Conv2D(384, (3, 3), padding = "same"))
        model.add(Conv2D(384, (3, 3), padding = "same",
            kernel_regularizer = l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = chanDim))
        model.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2)))
        model.add(Dropout(0.25))

        # block #4: first set of FC --> RELU layers
        model.add(Flatten())
        model.add(Dense(4096, kernel_regularizer = l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # block #5: second set of FC --> RELU layers
        model.add(Dense(4096, kernel_regularizer = l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(classes, kernel_regularizer = l2(reg)))
        model.add(Activation("softmax"))

        # return model
        return model
