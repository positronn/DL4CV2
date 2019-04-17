# train_models.py
import matplotlib
matplotlib.use("Agg")

import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from pyimagesearch.nn.conv import MiniVGGNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.datasets import cifar10


# construct argument parse and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required = True, help = "path to output directory")
ap.add_argument("-m", "--models", required = True, help = "path to output models directory")
ap.add_argument("-n", "--num-models", type = int, default = 5, help = "# of models to train")
args = vars(ap.parse_args())


# load the training and testing data, then scale it into the range [0, 1]
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0


# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

