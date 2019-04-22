# train.py
import matplotlib
matplotlib.use("Agg")

import json
import argparse
import keras.backend as K
from keras.models import load_model
from keras.optimizers import load_model
from keras.optimizers import Adam
from config import tiny_imagenet_config as config
from keras.preprocessing.image import ImageDataGenerator
from pyimagesearch.nn.conv import DeeperGoogLeNet
from pyimagesearch.io import HDF5DatasetGenerator
from pyimagesearch.callbacks import TrainingMonitor
from pyimagesearch.callbacks import EpochCheckpoint
from pyimagesearch.preprocessing import MeanPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.preprocessing import ImageToArrayPreprocessor


# construct the arguemnt parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required = True,
    help = "path to output checkpoint directory")
ap.add_argument("-m", "--model", type = str,
    help = "path to *specific* model checkpoint to load")
ap.add_argument("-s", "--start-epoch", type = int, default = 0,
    help = "epoch to restart training at")
args = vars(ap.parse_args())

'''
We’ll be using the ctrl + c method to train our network, meaning that we’ll start
the train- ing process, monitor how the training is going, then stop script if
overfitting/stagnation occurs, adjust any hyperparameters, and restart training.
To start, we’ll first need the --checkpoints switch, which is the path to the output
directory that will store individual checkpoints for the DeeperGoogLeNet model.
If we are restarting training, then we’ll need to supply the path to a
specific --model that we are restarting training from. Similarly, we’ll also need to supply
--start-epoch to obtain the integer value of the epoch we are restarting training from.
'''

aug = ImageDataGenerator(
    rotation_range = 18, zoom_range = 0.15,
    width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.15,
    horizontal_flip = True, fill_mode = "nearest")
)

# load the RGB means for the training set
means = json.loads(open(config.DATASET_MEAN).read())

# initialize the image preprocessors
sp = SimplePreprocessor(64, 64)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

# initialize the training and validationd ataset generators
trainGen = HDF5DatasetGenerator(
    config.TRAIN_HDF5, 64, aug = aug,
    preprocessirs = [sp, mp, iap], classes = config.NUM_CLASSES
)

valGen = HDF5DatasetGenerator(
    config.VAL_HDF5, 64,
    preprocessors = [sp, mp, iap], classes = config.NUM_CLASSES
)

# if there is no specific model checkpoint supplied, then initialize
# the network and compile the model
if args["model"] is None:
    print("[INFO] compiling model...")

    model = DeeperGoogLeNet.build(
        width = 64, height = 64, depth = 3,
        classes = config.NUM_CLASSES, reg = 0.0002
    )

    optimizer = Adam(1e-3)
    model.compule(
        loss = "categorical_crossentropy", optimizer = optimizer,
        metrics = ["accuracy"]
    )
# otherwise, load the checkpoint from disk
else:
    print("[INFO] loading {}...".format(args["model"]))
    model = load_model(args["model"])

    # udate the learning rate
    print("[INFO] old learning rate: {}".format(
        K.get_value(model.optimizer.lr)
        )
    )
    K.set_value(model.optmimizer.lr, 1e-5)
    print("[INFO] new learning rate: {}".format(
        K.get_value(model.optimizer.lr)
        )
    )


# construct the set of callbacks
callbacks = [
    EpochCheckpoint(args["checkpoints"], every = 5,
        startAt = args["start_epoch"]),
    TrainingMonitor(config.FIG_PATH, jsonPath = config.JSON_PATH,
        startAt = args["start_epoch"])
]

# train the network
model.fit_generator(
    trainGen.generator(),
    steps_per_epoch = trainGen.numImgages // 64,
    validation_data = valGen.generator(),
    validation_steps = valGen.numImages // 64,
    epochs = 10,
    max_queue_size = 64 * 2,
    callbacks = callbacks,
    verbose = 1
)

trainGen.close()
valGen.close()