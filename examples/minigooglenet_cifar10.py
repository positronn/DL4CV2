# minigooglenet_cifar10.py
import matplotlib
matplotlib.use("Agg")

import os
import argparse
import numpy as np
from keras.datasets import cifar10
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from pyimagesearch.callbacks import TrainingMonitor
from pyimagesearch.nn.conv import MiniGoogLeNet
from sklearn.preprocessing import LabelBinarizer


# we'll be defining a polynomial decay learning rate schedule
# of the form:
#      α = α_0 * (1 - e / e_max) ^ p
# Where α_0 is the initial learning rate, e is the current epoch number, e_max
# is the maximum number of epochs we are going to perform and p is the ower of the polynomial.
# Applying this equation yields the learning rate α for the current epoch

# define the total number of epochs tot rain for along with the
# initial learning rate
NUM_EPOCHS = 70
INIT_LR = 5e-3


def poly_decay(epoch):
    '''
    '''
    # initialize the maximum nubmber of peochs, base learning rate,
    # and power of the polynomial
    maxEpochs = NUM_EPOCHS
    baseLR = INIT_LR
    power = 1.0

    # compute the new learning rate based on polynomial decay
    alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power

    # return new learninig rate
    return alpha


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required = True,
    help = "path to output model")
ap.add_argument("-o", "--output", required = True,
    help = "path to output directory (logs, plots, etc...")
args = vars(ap.parse_args())


# load the training and testinf data, converting the images from integers
# to floats
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float")
testX = testX.astype("float")

# apply mean subtraction to the data
mean = np.mean(trainX, axis = 0)
trainX -= mean
testX -= mean

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

aug = ImageDataGenerator(
    width_shift_range = 0.1, horizontal_flip = True,
    fill_mode = "nearest"
)

# construct the set of callbacks
figPath = os.path.sep.join([args["output"], f"{os.getpid()}.png"])
jsonPath = os.path.sep.join([args["output"], f"{os.getpid()}.json"])
callbacks = [TrainingMonitor(figPath, jsonPath = jsonPath), LearningRateScheduler(poly_decay)]


# initialize the optimizer and model
print("[INFO] compiling model...")
optimizer = SGD(lr = INIT_LR, momentum = 0.9)
model = MiniGoogLeNet.build(width = 32, height = 32, depth = 3, classes = 10)
model.compile(loss = "categorical_crossentropy", optimizer = optimizer, metrics = ["accuracy"])

# trian the network
print("[INFO] training network...")
model.fit_generator(
    aug.flow(trainX, trainY, batch_size = 64),
    validation_data = (testX, testY),
    steps_per_epoch = len(trainX) // 64,
    epochs = NUM_EPOCHS, callbacks = callbacks, verbose = 1
)

# save the network to disk
print("[INFO] serializing network...")
model.save(args["model"])
