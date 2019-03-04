# extract_features.py

import os
import random
import argparse
import progressbar
import numpy as np
from imutils import paths
from pyimagesearch.io import HDF5DatasetWriter
from sklearn.preprocessing import LabelEncoder
from keras.applications import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils


# argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
        help = "path to input dataset")
ap.add_argument("-o", "--output", required = True,
        help = "pathto output HDF5 file")
ap.add_argument("-b", "--batch-size", type = int, default = 32,
        help = "batch size of iamges to be passed through network")
ap.add_argument("-s", "--buffer-size", type = int, default = 1000,
        help = "size of feature extracttion buffer")
args = vars(ap.parse_args())




# grab the list of images that we'll be describing then randomly
# shuffle them to allow for easy training and testing splits via
# arrat slicing during training time
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
random.shuffle(imagePaths)


# extract the class labels from the image paths then encode the
# labels
labels = [path.split(os.path.sep)[-2] for path in imagePaths]
le = LabelEncoder()
labels = le.fit_transform(labels)


# load the VGG16 Network
print("[INFO] loading network...")
model = VGG16(weights = "imagenet", include_top = False)


# initialize the HDF5 dataset writer, then store the class label
# names in the dataset
dataset = HDF5DatasetWriter(dims = (len(imagePaths), 512 * 7 * 7),
                    outputPath = args["output"],
                    dataKey = "features",
                    bufSize = args["buffer_size"])
dataset.storeClassLabels(le.classes_)


# initialize the progress bar
widgets = ["Extracting Features: ", progressbar.Percentage(), " ",
            progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval = len(imagePaths), widgets = widgets).start()
batchSize = args["batch_size"]
# loop over the images in patches
for i in np.arange(0, len(imagePaths), batchSize):
    # extract the batch of images and labels,
    # then initialize the list of actual iamges that will be
    # passed through the network foe feature extraction
    batchPaths = imagePaths[i:i + batchSize]
    batchLabels = labels[i:i + batchSize]
    batchImages = []

    # loop over the iamges and labels in the current batch
    for (j, imagePath) in enumerate(batchPaths):
        # load the input image using the Keras helper utility
        # while ensuring thenimage is resized to 224 * 224 pixels
        image = load_img(imagePath, target_size = (224, 244))
        image = img_to_array(image)

        # preprocess the image by (1) expanding the dimenions
        # and (2)subtractinng the mean RGB pixel intensity from
        # the ImageNet dataset
        image = np.expand_dims(image, axis = 0)
        image = imagenet_utils.preprocess_input(image)

        # add the image to the batch
        batchImages.append(image)

    # pass the images through the network and use the output as 
    # our actual features
    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages, batch_size = batchSize)

    # reshape the features so that each image is represented by
    # a flattened feature vector of the "MaxPooling2D" outputs
    features = features.reshape((features.shape[0], 512 * 7 * 7))

    # add the features and la els to out HDF5 dataset
    dataset.add(features, batchLabels)
    pbar.update(i)

dataset.close()
pbar.finish()