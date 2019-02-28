# -*- coding: utf-8 -*-
# NeuralNetwork.py
'''
    Vanilla feed-forward fully connected Neural Network with
    backprograpation algorithm implementation.
'''

import numpy as np


class NeuralNetwork:
    '''
        Implementation of a feed-forward fully connected neural network
        trained by the backpropagation algorithm.
    '''
    def __init__(self, layers:int, alpha:float = 0.1):
        '''
        Initialize the list of weights matrices for the
        neural network, then store the network achitecture
        and learning rate.

        parameters
        ----------
            layers: int, number of layers of the neural network
            alpha:  float, learning rate for the training process

        returns
        -------
            None: contructor operator.
        '''
        self.W = []
        self.layers = layers
        self.alpha = alpha

        # start looping from the index of the first layer but
        # stop before we reach the last two layers
        for i in np.arange(0, len(layers) - 2):
            # randomly initialize a weight matrix connecting the
            # number of nodes in each respective layer together,
            # adding an extra node for the bias
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            self.W.append(w / np.sqrt(layers[i]))

        # the last two layers are a special case wher the input
        # connections need a bias term but the output does not
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))


    def __repr__(self):
        '''
        construct and return a string that represents the
        network architecture.

        parameters
        ----------

        returns
        -------
            arch: a str that represents the network architecture
        '''
        return "NeuralNetwork: {}".format("-".join(str(l) for l in self.layers))


    def sigmoid(self, x:float) -> float:
        '''
        Compute and return the sigmoid activation value for a given
        input value x

        parameters
        ----------
            x: a float number

        returns
        -------
            simga(x): a float number, value of the sigmoid activation
                        function for given x
        '''

        return 1.0 / (1 + np.exp(-x))


    def dx_sigmoid(self, x:float) -> float:
        '''
        Compute the derivative of the sigmoid function ASSUMING
        that 'x' has already been passed through the "sigmoid" function

        parameters
        ----------
            x:    a float number, it is assumed that this value has already
                    passed through the sidmoid function (i.e. is a float return
                    value of the function sigmoid(self, x:float) -> float)

        returns
        -------
            sigma'(x): a float number, value of the derivative of the
                        sigmoid activation function for given x 
        '''
        return x * (1 - x)


    def fit(self, X:np.ndarray, y:np.ndarray, epochs:int = 1000, displayUpdate:int = 100):
        '''
        Train the NeuralNetwork

        parameters
        ----------
            X: training data
            y: corresponding class labels for each entry in X
            epochs: number of epochs during training stage
            displayUpdate: number of epochs to pass for displaying progress to terminal

        returns
        -------
            None: trains weights and biass in the weight matrix for the neural network
        '''
        # insert a column of 1's as the last entry in the feature
        # matrix --this little trick allows us to treat the bias
        # as a trainable parameter within the weight matrix
        X = np.c_[X, np.ones((X.shape[0]))]

        # loop over the desired number of epochs
        for epoch in np.arange(0, epochs):
            # loop over each individual data point and train
            # our nwtwork on it
            for (x, target) in zip(X, y):
                self.__fit_partial(x, target)

            # check to see if we should display a training update
            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.calculateLoss(X, y)
                print("[INFO] epoch = {}, loss = {:.7f}".format(epoch + 1, loss))


    def __fit_partial(self, x:np.ndarray, y:np.ndarray):
        '''
        
            ***** intended to be a private method *****

        This method realizes feedforward, backpropagation and weight update
        the vanilla-style, for each input x in the dataset.


        parameters
        ----------
            x: an individual data point from our design matrix (Input matrix)
            y: corresponding class label

        retuns
        ------
            None: trains weights and biass in the weight matrix for the neural network
                    {
                    updates self.W:
                        + feedforward
                        + backpropagation
                        + weight update
                    }
        '''
        # construct our list of output activations for each layer
        # as our data point flows through the network; the first
        # activation is a special case -- it's just the input
        # feature vector itself
        # 
        # variable 'A' is responsible for storing the output activations
        # for each layer as our data point x forward propagates
        # through the network
        A = [np.atleast_2d(x)]

        # ========== feedforward ==========
        # loop over the layers in the network
        for layer in np.arange(0, len(self.W)):
            # feedforward the activation at the current layer by
            # taking the dot product between the activation and
            # the weight matrix -- this is called the "net input"
            # to the current layer
            net = A[layer].dot(self.W[layer])

            # computing the "net output" is simply applying our
            # nonlinear activation function to the net input
            out = self.sigmoid(net)

            # once we have the net output, add it t out list of
            # activations
            A.append(out)

        # ========== backpropagation ==========
        # the first phase of backpropagation is to compute the
        # difference between our *prediction* (the final output
        # activation in the activations list) and the true target value
        error = A[-1] - y

        # apply chain rule and build our list of deltas `D`;
        # the first entry in the deltas is simply the error of the
        # output layer times the derivative of our activation function for the
        # output value
        D = [error * self.dx_sigmoid(A[-1])]

        # loop over the layers in reverse order (ignoring the last two
        # since we already have taken them into account)
        for layer in np.arange(len(A) - 2, 0, -1):
            # the delta for the current layer is equal to the delta
            # of the *previos layer* dotted with the weight matrix
            # of the current layer, followed by multiplying the delta
            # by the derivate of the nonlinear activation function
            # for the activations of the current layer
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.dx_sigmoid(A[layer])
            D.append(delta)

        # since we looped over our layers in reverse order we need
        # to reverse the deltas
        D = D[::-1]

        # ========== weight update phase ==========
        # loop over the layers
        for layer in np.arange(0, len(self.W)):
            # update our weights by taking the dot product of the layer
            # activations with their respective deltas, then multipliying
            # this value by some small learniing rate and adding to our
            # weight matrix -- this is where the actual `learning` takes place
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])


    def predict(self, X:np.ndarray, addBias:bool = True) -> float:
        '''
            Predicts a value according to X, a set of points, or the input
            values of the dataset.

            parameters
            ----------
                X:  np.ndarray, datapoints to predict the class labels for
                addBias: a bool, whether or not we need to add a column
                            of 1’s to X to perform the bias trick.

            returns
            -------
                pred: a float, prediction value for given input

        '''
        # initialize the output prediction as the input features -- this
        # value will be (forward) propagated through the network to obtain the
        # final prediction
        pred = np.atleast_2d(X)

        # check to see if the bias column should be added
        if addBias:
            # insert a column of 1's as the last entry in the feature
            # matrix (bias)
            pred = np.c_[pred, np.ones((pred.shape[0]))]

        # loop over our layers in the network
        for layer in np.arange(0, len(self.W)):
            # computing the output prediciton is as simple as taking
            # the dot product between the current activation value `p`
            # and the weight matrix associated with the current layer,
            # then passing this value through a nonlinear activation
            # function
            pred = self.sigmoid(np.dot(pred, self.W[layer]))

        # return the predicted value
        return pred


    def calculateLoss(self, X:np.ndarray, targets:np.ndarray) -> np.ndarray:
        '''
        Calculate information loss according to the input data
        and their respective true labels.

        parameters
        ----------
            X:          input values
            targets:    true labels of the inputs

        returns
        -------
            loss:       loss value according to:
                            L = (1 / 2) * Sigma[ (y_pred - y_label) ** 2 ]
        '''
        # make predictions for the input data points then compute
        # the loss
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias = False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)

        return loss