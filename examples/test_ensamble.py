# test_ensamble.py


import os
import glob
import argparse
import numpy as np
from keras.datasets import cifar10
from keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer



# argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--models", required = True,
            help = "path to models directory")
args = vars(ap.parse_args())



# split data
(testX, testY) = cifar10.load_data()[1]
testX = testX.astype("float") / 255.0


# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"]


# convert the labels from integers to vectors
lb = LabelBinarizer()
testY = lb.fit_transform(testY)


# constrcut the path used to collect the models then initialize
# the models list
modelPaths = os.path.sep.join([args["models"], "*.model"])
modelPaths = list(glob.glob(modelPaths))

models = []
# loop over the model paths, loading the model, and adding it
# to the list of models
for (i, modelPath) in enumerate(modelPaths):
    print("[INFO] loading model {}/{}".format(i + 1, len(modelPaths)))
    models.append(load_model(modelPath))


print("[INFO] evaluating ensemble...")
predictions = []


# loop over the models
for model in models:
    # use the current model to make predictions on the testing data,
    # then store these predictions in the aggregate predicitons list
    predictions.append(model.predict(testX, batch_size = 64))


# average the probabilidties across all model predictions,
# then show a classification report
predictions = np.average(predictions, axis = 0)
report = classification_report(testY.argmax(axis = 1),
                predictions.argmax(axis = 1), target_names = labelNames)
print(report)

