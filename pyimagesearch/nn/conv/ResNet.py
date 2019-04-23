# ResNet.py
'''
First introduced by He et al. in their 2015 paper, Deep Residual Learning for
Image Recognition [24], the ResNet architecture has become a seminal work, demonstrating
that extremely deep networks can be trained using standard SGD and a reasonable
initialization function. In order to train networks at depths greater than 50-100
(and in some cases, 1,000) layers, ResNet relies on a micro-architecture called the residual module.

Another interesting component of ResNet is that pooling layers are used
extremely sparingly. Building on the work of Springenberg et al.,
ResNet does not strictly rely on max pooling operations to reduce volume size.
Instead, convolutions with strides > 1 are used to not only learn
weights, but reduce the output volume spatial dimensions. In fact, there are only
two occurrences of pooling being applied in the full implementation of the architecture:

    1. The first (and only) currency of max pooling happens early in the network
    to help reduce spatial dimensions.

    2. The second pooling operation is actually an average pooling layer
    used in place of fully- connected layers, like in GoogLeNet.

Strictly speaking, there is only one max pooling layer – all other reductions in spatial
dimensions are handled by convolutional layers.
'''

from keras import backend as K
from keras.regularizers import l2
from keras.layers import add
from keras.models import Model
from keras.layers import Input
from keras.layers import Flatten
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization


class ResNet:
    '''
    '''
    @staticmethod
    def _residual_module(data:'np.ndarray', K:int, stride, chanDim:int,
        red:bool = False, reg:float = 0.0001, bnEps:float = 2e-5,
        bnMom:float = 0.9):
        '''
        arguments
        ---------
            data: input of the residual module
            K: number of filters that will be learned bu the final
                CONV in the bottleneck
            stride: stride of the convolution
            chanDim: defines the axis which will perform batch normalization
            red: control whether we are reducing spatial dimensions (True) or not (False).
            reg: regularization strength to all CONV layers in the module
            bnEps: batch normalization epsilon to control division by zero
            bnMom: controls the momentum for the moving average

        returns
        -------
            x: a residual module result of the adition of 3 CONV layers and
                the addition node for the output of the layers and the residual input.

        '''
        # the shortcut branch of the ResNet module should be
        # initilialized as the input (identity) data
        shortcut = data

        # the first block of the ResNet module are the 1x1 CONVs
        bn1 = BatchNormalization(axis = chanDim, epsilon = bnEps,
            momentum = bnMom)(data)
        act1 = Activation("relu")(bn1)
        conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias = False,
            kernel_regularizer = l2(reg))(act1)
        
        # the second block of the ResNet module are the 3x3 CONVs
        bn2 = BatchNormalization(axis = chanDim, epsilon = bnEps,
            momentum = bnMom)(conv1)
        act2 = Activation("relu")(bn2)
        conv2 = Conv2D(int(K * 0.25), (3, 3), strides = stride,
            padding = "same", use_bias = False,
            kernel_regularizer = l2(reg))(act2)

        # the third block of the ResNet module is another set of 1x1 CONVs
        bn3 = BatchNormalization(axis = chanDim, epsilon = bnEps,
            momentum = bnMom)(conv2)
        act3 = Activation("relu")(bn3)
        conv3 = Conv2D(K, (1, 1), use_bias = False,
            kernel_regularizer = l2(reg))(act3)

        # if we are to redice the spatial size,
        # apply a CONV layer to the shortcut
        if red:
            shortcut = Conv2D(K, (1, 1), strides = stride,
                use_bias = False, kernel_regularizer = l2(reg))(act1)
        
        # add together the shortcut nd the final CONV
        x = add([conv3, shortcut])

        # return the additiona s the output of the  residual module
        return x


    @staticmethod
    def build(width:int, height:int, depth:int,
        classes:int, stages, filters, reg = 0.0001,
        bnEps = 2e-5, bnMom = 0.9, dataset = "cifar"):
        '''Build the ResNet architecture given width, height and depth
        as dimensions of the input tensor and the corresponding
        number of classes of the data.

        arguments
        ---------
            width:  width of input images.
            height: height of input images.
            depth:  depth of input images.
            classes: number of classes of the corresponding data.
            stages:
            filters:
            reg:
            bnEps:
            bnMom:
            dataset:
        
        returns
        -------

        '''
        # initializr the input shape to be "channels last" ant the
        # channels dimension itself
        inputShape = (height, width, depth)
        chanDim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # set the input and apply BN
        inputs = Input(shape = inputShape)
        x = BatchNormalization(axis = chanDim, epsilon = bnEps,
            momentum = bnMom)(inputs)

        # check if we are utilizing the CIFAR dataset
        if dataset == "cifar":
            # apply  asingle CONV layer
            x = Conv2D(filters[0], (3, 3), use_bias = False,
                padding = "same", kernel_regularizer = l2(reg))(x)
        # or using Tiny ImageNet dataset
        elif dataset == "tiny_imagenet":
            # apply CONV -*-> BN --> ACT --> POOL to reduce spatial size
            x = Conv2D(filters[0], (5, 5), use_bias = False,
                padding = "same", kernel_regularizer = l2(reg))(x)
            x = BatchNormalization(axis = chanDim, epsilon = bnEps,
                momentum = bnMom)(x)
            x = Activation("relu")(x)
            x = ZeroPadding2D((1, 1))(x)
            x = MaxPooling2D((3, 3), stride = (2, 2))(x)

        # loop over the number of stages
        for i in range(0, len(stages)):
            # initiaize the stride, the apply residual module
            # used to reduce the spatial size of the input volume
            stride = (1, 1) if i == 0 else (2, 2)
            x = ResNet._residual_module(x, filters[i + 1],
                stride, chanDim, red = True, bnEps = bnEps, bnMom = bnMom)

            # loop over the number of layers in the stage
            for j in range(0, stages[i] - 1):
                # apply a ResNet module
                x = ResNet._residual_module(x, filters[i + 1],
                    (1, 1), chanDim, bnEps = bnEps, bnMom = bnMom)

        # apply BN --> ACT --> POOL
        x = BatchNormalization(axis = chanDim, epsilon = bnEps,
            momentum = bnMom)(x)
        x = Activation("relu")(x)
        x = AveragePooling2D((8, 8))(x)

        # softmax classifier
        x = Flatten()(x)
        x = Dense(classes, kernel_regularizer = l2(reg))(x)
        x = Activation("softmax")(x)

        # create the model
        model = Model(inputs, x, name = "resnet")

        # return the constructed network architecture
        return model