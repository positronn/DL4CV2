# -*- coding: utf-8 -*-
# MiniVGGNet.py
'''
Implementation of the MiniVGGNet architecture.

MiniVGGNet architecture is a smaller version of the VGGNet,
presented in a 2014 paper by Simonyan and Zisserman; which
uses (3 * 3) kerneles throughout the entire architecture.
The use of these small kernels is arguably what helps VGGNet generalize to
classification problems outside what the network was originally trained on.

Overall, MiniVGGNet consists of two sets of CONV -*-> RELU --> CONV -*-> RELU --> POOL layers,
followed by a set of FC--> RELU --> FC --> SOFTMAX layers.
'''

from keras import backend as K
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense



class MiniVGGNet:
    '''
    '''
    def build(width:int, height:int, depth:int, classes:int, batchNorm:bool = True):
        '''
        Build the MiniVGGNet architecture given width, height and depth
        as dimensions of the input tensor and the corresponding
        number of classes of the data.

        parameters
        ----------
            width:  width of input images.
            height: height of input images.
            depth:  depth of input images.
            classes: number of classes of the corresponding data.
            batchNorm: indicates whether the Batch Normalization
                is going to be applied to the model or not. 

        returns
        -------
            model: the MiniVGGNet model compatible with given inputs
                    as a keras sequential model.
        '''
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1


        # first CONV -*-> RELU --> CONV -*-> RELU --> POOL layer set
        model.add(Conv2D(32, (3, 3), padding = "same", input_shape = inputShape))
        model.add(Activation("relu"))
        if batchNorm:
            model.add(BatchNormalization(axis = chanDim))
        model.add(Conv2D(32, (3, 3), padding = "same"))
        model.add(Activation("relu"))
        if batchNorm:
            model.add(BatchNormalization(axis = chanDim))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.25))

        # second CONV -*-> RELU --> CONV -*-> RELU --> POOL layer set
        model.add(Conv2D(64, (3, 3), padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = chanDim))
        model.add(Conv2D(64, (3, 3), padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = chanDim))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.25))


        # first (and only) set of FC --> RELU layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model