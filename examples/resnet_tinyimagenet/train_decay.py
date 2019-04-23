# train_decay.py

import matplotlib
matplotlib.use("Agg")

import os
import sys
import json
import argparse
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from pyimagesearch.nn.conv import ResNet
from pyimagesearch.io import HDF5DatasetGenerator
from pyimagesearch.callbacks import TrainingMonitor
from pyimagesearch.preprocessing import MeanPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from config import tiny_imagenet_config as config


# set a high recursion limit so Theano doen't complain
sys.setrecursionlimit(5000)

# define the total number of epochs to train for along with
# the initial learning rate
NUM_EPOCHS = 75
INIT_LR = 1e-1

def poly_decay(epoch):
    # initialize the maximum number of epochs, base learning rate
    # and power of the polynomial
    maxEppchs = NUM_EPOCHS
    baseLR = INIT_LR
    power = 1.0

    # compute the new learning rate based on polynomial decay
    alpha = baseLR * (1 - (epoch / floa(maxEpochs))) ** power

    return alpha


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    "-m", "--model", required = True,
    help = "path to output model"
)
ap.add_argument(
    "-o", "--output", required = True,
    help = "path to output directory (logs, plots, etc.)"
)
args = vars(ap.parse_args())

augmentation = ImageDataGenerator(
    rotation_range = 18,
    zoom_range = 0.15,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.15,
    horizontal_flip = True,
    fill_mode = "nearest"
)

# load the RGB means for the training set
means = json.loads(open(config.DATASET_MEAN).read())

# initialize the image preprocessors
sp = SimplePreprocessor(64, 64)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

# initialize the training and validation dataset generators
trainGen = HDF5DatasetGenerator(
    config.TRAIN_HDF5, 64, aug = augmentation,
    preprocessors = [sp, mp, iap], classes = config.NUM_CLASSES
)

valGen = HDF5DatasetGenerator(
    config.VAL_HDF5, 64
    preprocessors = [sp, mp, iap], classes = config.NUM_CLASSES
)

# constrcut the set of callbacks
figPath = os.path.sep.join(
    [args["output"], "{}.png".format(os.getpid())]
)

jsonPath = os.path.sep.join(
    [args["output"], "{}.json".format(os.getpid())]
)

callbacks = [
    TrainingMonitor(
        figPath, jsonPath = jsonPath
    ),
    LearningRateScheduler(
        poly_decay
    )
]

# initialize the optimizer and model (ResNet-56)
print("[INFO] compiling model...")
model = ResNet.build(
    width = 64, height = 64, depth = 3, classes = config.NUM_CLASSES,
    stages = (3, 4, 6),
    filters = (64, 128, 256, 512),
    reg = 0.0005,
    dataset = "tiny_imagenet"
)

optimizer = SGD(
    lr = INIT_LR, momentum = 0.9
)

model.compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer,
    metrics = ["accuracy"]
)

# train the network
print("[INFO] training network...")
model.fit_generator(
    trainGen.generator(),
    steps_per_epoch = trainGen.numImages // 64,
    validation_data = valGen.generator(),
    validation_steps = valGen.numImages // 64,
    epochs = NUM_EPOCHS,
    max_queue_size = 64 * 2
    callbacks = callbacks,
    verbose = 1
)

# save the netwokr to disk
print("[INFO] serializing network...")
model.save(args["model"])

# close the databases
trainGen.close()
valGen.close()

