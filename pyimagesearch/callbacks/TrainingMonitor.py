# -*- coding: utf-8 -*-
# TrainingMonitor.py

# import matplotlib and change backend
import matplotlib
matplotlib.use("Agg")

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import BaseLogger



class TrainingMonitor(BaseLogger):
    '''
    This monitor serializes the loss and accuracy for both the training and
    validation set to disk, followed by constructing a plot of the data
    when training a network with Keras.
    '''
    def __init__(self, figPath, jsonPath = None, startAt = 0):
        # store the output path fot the figure
        # the path to the JSON serialized file,
        # and the starting epoch
        super(TrainingMonitor, self).__init__()
        self.figPath = figPath
        self.jsonPath = jsonPath
        self.startAt = startAt


    def onTrainBegin(self, logs = {}):
        '''
        '''
        # initialize the history dictionary
        self.H = {}

        # if the JSON history path exists, load the training history
        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                self.H = json.loads(open(self.jsonPath).read())

                # check to see if a starting epoch was supplied
                if self.startAt > 0:
                    # loop over the entries in the history log and
                    # trim any entries that are past the starting
                    # epoch
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.startAt]


    def onEpochEnd(self, epoch, logs = {}):
        '''
        '''
        for (k, v) in logs.items():
            l = self.H.get(k, [])
            l.append(v)
            self.H[k] = l

        # affter this code executes, the dictionary H now has four keys:
        #   - train_loss
        #   - train_acc
        #   - val_loss
        #   - val_acc
        #
        # check to see if the training history should be serialized
        # to file
        if self.jsonPath is not None:
            f = open(self.jsonPath, 'w')
            f.write(json.dumps(self.H))
            f.close()

        # ensure at least two epochs have passed before
        # plotting (epoch starts at zero)
        if len(self.H["loss"]) > 1:
            # plot the training loss and accuracy
            N = np.arange(0, len(self.H["loss"]))
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.H["loss"], label = "train_loss")
            plt.plot(N, self.H["val_loss"], label = "val_loss")
            plt.plot(N, self.H["acc"], label = "train_acc")
            plt.plot(N, self.H["val_acc"], label = "val_acc")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(len(self.H["loss"])))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss / Accuracy")
            plt.legend()

            # save figure
            plt.savefig(self.figPath)
            plt.close()
