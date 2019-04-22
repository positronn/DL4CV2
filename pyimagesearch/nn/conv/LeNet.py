# -*- coding: utf-8 -*-
# LeNet.py
'''
Implementation of the LeNet architecture.

LeNet architecture is relatively small and was used
in a seminal paper for OCR in 1998 by LeCun et al.


INPUT --> CONV -*-> TANH --> POOL -|-> CONV -*-> TANH --> POOL -|-> FC --> TANH --> FC
'''


from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.layers.core import Activation
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Conv2D



class LeNet:
    '''
    LeNet Architecture implementation in Keras
    '''

    @staticmethod
    def build(width:int, height:int, depth:int, classes:int):
        '''
        Build the LeNet architecture given width, height and depth
        as dimensions of the input tensor and the corresponding
        number of classes of the data.

        parameters
        ----------
            width:  width of input images.
            height: height of input images.
            depth:  depth if input images.
            classes: number of classes of the corresponding data.

        returns
        -------
            model: the LeNet model compatible with given inputs
                    as a keras sequential model.
        '''
        # initialize model
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # first set of CONV -*-> RELU --> POOL layers
        model.add(Conv2D(20, (5, 5), padding = "same", input_shape = inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))


        # second set of CONV -*-> RELU --> POOL layers
        model.add(Conv2D(50, (5, 5), padding = "same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))


        # first (and only) set of FC --> RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model