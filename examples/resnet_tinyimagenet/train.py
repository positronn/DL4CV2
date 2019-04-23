# train.py

import matplotlib
matplotlib.use("Agg")

# import the necessary packages
import sys
import json
import argparse
import keras.backend as K
from config import tiny_imagenet_config as config
from keras.models import load_model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from pyimagesearch.nn.conv import ResNet
from pyimagesearch.io import HDF5DatasetGenerator
from pyimagesearch.callbacks import EpochCheckpoint
from pyimagesearch.callbacks import TrainingMonitor
from pyimagesearch.preprocessing import MeanPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.preprocessing import ImageToArrayPreprocessor

# set a high recursion limit to Theano doesn't complain
sys.setrecursionlimit(5000)

ap = argparse.ArgumentParser()
ap.add_argument(
    "-c", "--checkpoints", required = True,
    help = "path to output checkpoint directory"
)
ap.add_argument(
    "-m", "--model", type = str,
    help = "path to *specifi* model checkpoint to load"
)
ap.add_argument(
    "-s", "--start-epoch", type = int, default = 0,
    help = "epoch to restart training at"
)
args = vars(ap.parse_args())

# construct th training image generator for data augmentation
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

# initialize the training anda validation dataset generators
trainGen = HDF5DatasetGenerator(
    config.TRAIN_HDF5, 64,
    aug = augmentation,
    preprocessors = [sp, mp, iap],
    classes = config.NUM_CLASSES
)

valGen = HDF5DatasetGenerator(
    config.VAL_HDF5, 64,
    preprocessors = [sp, mp, iap],
    classes = config.NUM_CLASSES
)

# if there is no specific model checkpoint supplied, then initialize
# the network and compile the model
if args["model"] is None:
    print("[INFO] compiling model...")

    model = ResNet.build(
        width = 64, height = 64, depth = 3,
        classes = config.NUM_CLASSES, stages = (3, 4, 6),
        filters = (64, 128, 256, 512), reg = 0.0005,
        dataset = "tiny_imagenet"
    )
    optimizer = SGD(lr = 1e-1, momentum = 0.9)
    model.compile(
        loss = "categorical_crossentropy", optimizer = optimizer,
        metrics = ["accuracy"]
    )
# otherwise, load the checkpoint from disk
else:
    print("[INFO] loading {}...".format(args["model"]))
    model = load_model(args["model"])

    # update the learning rate
    print("[INFO] old learning rate: {}".format(
        K.get_value(model.optimizer.lr)
    ))
    K.set_value(model.optimizer.lr, 1e-5)
    print("[INFO] new learning rate: {}".format(
        K.get_value(model.optimizer.lr)
    ))

# construct the set of callbacks
callbacks = [
    EpochCheckpoint(
        args["checkpoints"], every = 5,
        startAt = args["start_epoch"]
    ),
    TrainingMonitor(
        config.FIG_PATH, jsonPath = config.JSON,
        startAt = args["start_epoch"]
    )
]

# train the network
model.fit_generator(
    trainGen.generator(),
    steps_per_epoch = trainGen.numImages // 64,
    validation_data = valgen.generator(),
    validation_steps = valGen.numImages // 64,
    epochs = 50,
    max_queue = 64 * 2,
    callbacks = callbacks,
    verbose = 1
)

# close the databases
trainGen.close()
valGen.close()