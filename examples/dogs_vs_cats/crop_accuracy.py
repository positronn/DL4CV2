# crop_accuracy.py

import json
import progressbar
import numpy as np
from keras.models import load_model
from pyimagesearch.utils.ranked import rank5_accuracy
from pyimagesearch.io import HDF5DatasetGenerator
from pyimagesearch.preprocessing import CropPreprocessor
from pyimagesearch.preprocessing import MeanPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from config import dogs_vs_cats_config as config


# load the TGB means for the training set
means = json.loads(open(config.DATASET_MEAN).read())

# initialize the iamge preprocessors
sp = SimplePreprocessor(227, 227)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
cp = CropPreprocessor(227, 227)
iap = ImageToArrayPreprocessor()

# load the pretrained network
print("[INFO] loading model...")
model = load_model(config.MODEL_PATH)


# initialize the testing dataset generator, then make predictions on
# testing data
print("[INFO] predicting on test data (no crops)...")
testGen = HDF5DatasetGenerator(config.TEST_HDF5, 64, preprocessors = [sp, mp, iap], classes = 2)
predictions = model.predict_generator(testGen.generator(), steps = testGen.numImages // 64, max_queue_size = 64 * 2)

# compute the rank-1 and rank-5 accuracies
(rank1, __) = rank5_accuracy(predictions, testGen.db["labels"])
print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
testGen.close()


# reinitialize the testing set generator, ths time excluding the 
# `SimplePreprocessor`
testGen = HDF5DatasetGenerator(config.TEST_HDF5, 64, preprocessors = [mp], classes = 2)
predictions = []

# initialize the progress bar
widgets = ["Evaluating: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval = testGen.numImages // 64, widgets = widgets).start()

# loop over a single pass of the test data
for (i, (images, labels)) in enumerate(testGen.generator(passes = 1)):
    # loop over each of the individual images
    for image in images:
        # apply the crop preprocessor to the image to generate 10 separate crops,
        # then convert them from images to arrays
        crops = cp.preprocess(images)
        crops = np.array([iap.preprocess(c) for c in crops], dtype = "float32")

        # make predictions on the crops and then average them
        # together to obtatin the final prediction
        pred = model.predict(crops)
        predictions.append(pred.mean(axis = 0))
    
    pbar.update(i)

pbar.finish()
print("[INFO] predictin on test data (with crops)...")
(rank1, __) = rank5_accuracy(predictions, testGen.db["labels"])
print("[INFO] rnak-1: {:.2f}%".format(rank1 * 100))
testGen.close()
