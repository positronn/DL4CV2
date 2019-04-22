# MiniGoogLeNet.py

from keras import backend as K
from keras.layers import concatenate
from keras.models import Model
from keras.layers import Input
from keras.layers import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.normalization import BatchNormalization


class MiniGoogLeNet:
    '''
    Implementation of the MiniGoogLeNet architecture: a variant of GoogLeNet architecture
    using the inception module. 
    '''
    @staticmethod
    def _conv_module(x:'keras.layers.convolutional' , K:int, ksize:tuple, stride:tuple, chanDim:int, padding:str = "same"):
        '''
        conv_module static method is responsible for applying convolution,
        followed by a batch normalziation, and then finally an activation.

        parameters
        ----------
            x: the input layer to the function
            K: the number of filters our CONV layer is going to learn
            ksize: the size of each of the K filters that will be learned (kX, kY)
            stride: the stride of the CONV layer
            chanDim: the channel dimension, which is derived from either "channels last"
                or "channels first" ordering
            padding: the type of padding to be applied to CONV layer
        
        returns
        -------
            x: a conv_module with the CONV -*-> BN --> RELU pattern
        '''
        (kX, kY) = ksize

        # define a CONV -*-> BN --> RELU pattern
        x = Conv2D(K, (kX, kY), strides = stride, padding = padding)(x)
        x = BatchNormalization(axis = chanDim)(x)
        x = Activation("relu")(x)

        # return the block
        return x
    

    @staticmethod
    def _inception_module(x:'keras.layers.convolutional', numK1x1:int, numK3x3:int, chanDim:int):
        '''
        Inception module responislbe of performing two sets of convolultions
        in parallel and concatenating the results across the channel dimension.

        parameters
        ----------
            x: the input layer of the function
            numK1x1: the number of (1, 1) kernels in the first convolution layer to learn
            numK3x3: the number of (3, 3) kernels in the second convolution layer to learn
            chanDim: the channel dimension, which is derived from either "channels last"
                or "channels first" ordering

        returns
        -------
            x: a concatenation of the two conv_module
        '''
        # define two CONV modules, the concatenate across the
        # channel dimension
        conv_1x1 = MiniGoogLeNet._conv_module(x, numK1x1, (1, 1), (1, 1), chanDim)
        conv_3x3 = MiniGoogLeNet._conv_module(x, numK3x3, (3, 3), (1, 1), chanDim)
        x = concatenate([conv_1x1, conv_3x3], axis = chanDim)

        # return block
        return x
    

    @staticmethod
    def _downsample_module(x, K, chanDim):
        '''
        downsample module respoisnble for reducing the spatial dimensions of an input volume

        parameters
        ----------
            x: the input layer of the function
            K: the number of filters K our convolutional layer will learn
            chanDim: the channel dimension, which is derived from either "channels last"
                or "channels first" ordering

        returns
        -------
            x: a concatenation of the CONV -*-> POOL pattern
        '''
        # define the CONV module and POOL, then concatenate
        # across the channel dimensions
        conv_3x3 = MiniGoogLeNet._conv_module(x, K, (3, 3), (2, 2), chanDim, padding = "valid")
        pool = MaxPooling2D((3, 3), strides = (2, 2))(x)
        x = concatenate([conv_3x3, pool], axis = chanDim)

        # return the block
        return x
    

    @staticmethod
    def build(width:int , height:int, depth:int, classes:int):
        '''
        Build the MiniGoogLeNet architecture given width, height and depth
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
            model: the MiniGoogLeNet model compatible with given inputs
                    as a keras functional model.
        '''
        # initialize the input shape to be "channels last" and the 
        # channels dimension itself
        inputShape = (height, width, depth)
        chanDim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
        
        # define the model inpput and first CONV module
        inputs = Input(shape = inputShape)
        x = MiniGoogLeNet._conv_module(inputs, 96, (3, 3), (1, 1), chanDim)

        # define the model input and first CONV module
        x = MiniGoogLeNet._inception_module(x, 32, 32, chanDim)
        x = MiniGoogLeNet._inception_module(x, 32, 48, chanDim)
        x = MiniGoogLeNet._downsample_module(x, 80, chanDim)

        # four inception modules followed by a downsample module
        x = MiniGoogLeNet._inception_module(x, 112, 48, chanDim)
        x = MiniGoogLeNet._inception_module(x, 96, 64, chanDim)
        x = MiniGoogLeNet._inception_module(x, 80, 80, chanDim)
        x = MiniGoogLeNet._inception_module(x, 48, 96, chanDim)
        x = MiniGoogLeNet._downsample_module(x, 96, chanDim)

        # two Inception modules followed by global POOL and dropout
        x = MiniGoogLeNet._inception_module(x, 176, 160, chanDim)
        x = MiniGoogLeNet._inception_module(x, 176, 160, chanDim)
        x = AveragePooling2D((7, 7))(x)
        x = Dropout(0.5)(x)

        # softmax classifier
        x = Flatten()(x)
        x = Dense(classes)(x)
        x = Activation("softmax")(x)

        # create the model
        model = Model(inputs, x, name = "minigooglenet")

        # return the constructed network architecture
        return model