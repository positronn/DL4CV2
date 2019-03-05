# finetune_flowers17.py

import os
import argparse
import numpy as np
from imutils import paths
from keras.models import Model
from keras.layers import Input
from keras.applications import VGG16
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from pyimagesearch.nn.conv import FCHeadNet
from pyimagesearch.datasets import SimpleDatasetLoader
from pyimagesearch.preprocessing import AspectAwarePreprocessor
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


# argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
        help = "path to input dataset")
ap.add_argument("-m", "--model", required = True,
        help = "path to output model")
args = vars(ap.parse_args())


# construct data augmentation generator
augmentation = ImageDataGenerator(rotation_range = 30,
                    width_shift_range = 0.1,
                    height_shift_range = 0.1,
                    shear_range = 0.2,
                    zoom_range = 0.2,
                    horizontal_flip = True,
                    fill_mode = "nearest"
    )


# grab the lisr of iamges that we'll be describing, then extract
# the class label names rfrom the image paths
print("[INFO] loading iamges...")
imagePaths = list(paths.list_images(args["dataset"]))
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]


# initialize the image preprocessors
aap = AspectAwarePreprocessor(224, 224)
iap = ImageToArrayPreprocessor()


# load the dataset from disk then scale the raw pixel intensities
# to rhe range [0, 1]
sdl = SimpleDatasetLoader(preprocessors = [aap, iap])
(data, labels) = sdl.load(imagePaths, verbose = 500)
data = data.astype("float") / 255.0


# partition the data into training and testing splits
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.25, random_state = 42)


# convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)


# performing network surgery
# load the VGG16 network, ensuring the head FC layer sets are left off
baseModel = VGG16(weights = "imagenet", 
                include_top = False,
                input_tensor = Input(shape = (224, 224, 3))
    )


# initialize the new head of the network, a set of FC layers
# followed by a softmax classifier
headModel = FCHeadNet.build(baseModel, len(classNames), 256)


# place the head FC model on top of the base model -- this will
# become the actual model we will train
model = Model(inputs = baseModel.input, outputs = headModel)


# loop over all layers in the base model
# and freeze them so they will *not* be updated during the training process
for layer in baseModel.layers:
    layer.trainable = False


# compile ur model
print("[INFO] compiling model...")
optimizer = RMSprop(lr = 0.001)
model.compile(loss = "categorical_crossentropy",
            optimizer = optimizer,
            metrics = ["accuracy"]
    )


# train the head of the network for a few epochs (all other layers are frozen)
# -- this will allow the new FC layers to start to become initialized with actual
# *learned* values versus pure random
print("[INFO] training head...")
model.fit_generator(augmentation.flow(trainX, trainY, batch_size = 32),
                validation_data = (testX, testY),
                epochs = 10,
                steps_per_epoch = len(trainX) // 32,
                verbose = 1
    )


# evaluate the network on the fine-tuned model
print("[INFO] evaluating after fine-tuning...")
predictions = model.predict(testX, batch_size = 32)
print(classification_report(testY.argmax(axis = 1), predictions.argmax(axis = 1),
                                                    target_names = classNames))

# save model to disk
print("[INFO] serializing model...")
model.save(args["model"])
