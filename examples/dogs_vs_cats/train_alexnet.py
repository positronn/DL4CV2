# train_alexnet.py

import matplotlib
matplotlib.use("Agg")

import os
import json
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from pyimagesearch.nn.conv import AlexNet
from pyimagesearch.io import HDF5DatasetGenerator
from pyimagesearch.callbacks import TrainingMonitor
from pyimagesearch.preprocessing import PatchPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.preprocessing import MeanPreprocessor
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from config import dogs_vs_cats_config as config


# construct the training image generator for data augmentation
aug = ImageDataGenerator(
    rotation_range = 20, zoom_range = 0.15,
    width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.15,
    horizontal_flip = True, fill_mode = "nearest"
)


# load the RGB means for the training set
means = json.loads(open(config.DATASET_MEAN).read())


# initialize the image preprocessors
sp = SimplePreprocessor(227, 227)
pp = PatchPreprocessor(227, 227)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()


# initialize the training and validation dataset generators
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 64, aug = aug,
    preprocessors = [pp, mp, iap], classes = 2
)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, 64,
    preprocessors = [sp, mp, iap], classes = 2
)

# initialize optimizer
print("[INFO] compiling model...")
optimizer = Adam(lr = 1e-3)
model = AlexNet.build(width = 227, height = 227, depth = 3,
    classes = 2, reg = 0.0002)
model.compile(loss = "binary_crossentropy", optimizer = optimizer , metrics = ["accuracy"])

# construct the set of callbacks
path = os.path.sep.join([config.OUTPUT_PATH, "{}.png".format(os.getpid())])
callbacks = [TrainingMonitor(path)]

# train the network
model.fit_generator(
    trainGen.generator(),
    steps_per_epoch = trainGen.numImages // 64,
    validation_data = valGen.generator(),
    validation_steps = valGen.numImages // 64,
    epochs = 75,
    max_queue_size = 128 * 2,
    callbacks = callbacks,
    verbose = 1
)

# save the model to file
print("[INFO] serializing model...")
model.save(config.MODEL_PATH, overwrite = True)

# close the HDF5 datasets
trainGen.close()
valGen.close()

