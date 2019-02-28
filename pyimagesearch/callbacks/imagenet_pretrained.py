# -*- coding: utf-8 -*-
# imagenet_pretrained.py


import cv2
import argparse
import numpy as np
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img


# construct argument parse
ap.add_argument("-i", "--image", required = True, help = "path to the input image")
ap.add_argument("-m", "--model", type = str, default = "vgg16", help = "name of pre-trained network to use")
args = vars(ap.parse_args())



# define a dictionary that maps model names to their classes
# inside keras
MODELS = {
    "vgg16": VGG16,
    "vgg19": VGG19,
    "inception": InceptionV3,
    "xception": Xception,
    "reset": ResNet50
}


# ensure a valid model name was supplied via command line argument
if args["model"] not in MODELS.keys():
    raise AssertionError("The --model command line argument should be a kei in the `MODELS`dict")


# initialize the input image shape (224 * 224 pixels) along with
# the pre-processing dunction
inputShape = (224, 224)
preprocess = imagenet_urils.preprocess_input

# if we are using the InceptionV2 or Xception networks, then we
# need to set the input shape to (299 * 299)
# and use a different image proccessing function
if ars["model"] in ("inception", "xception"):
    inputShape = (299, 299)
    preprocess = preprocess_input


# load out network weights from disk
print("[INFO] loading {}...".format(args["model"]))
Network = MODELS[args["model"]]
model = Network(weights = "imagenet")


# load the input iamge using keras helper utility
# while ensuring the image is resized to ìnputhspae
# the required input dimensions for the ImageNet pre-trained network
print("[INFO] loading and pre-processing image...")
image = load_img(args["image"], target_size = inputShape)
image = img_to_array(image)


# our input image is now represented as a Numpy array
# of shape (inputSahpe[0], inputShape[1], 3) however we need to
# to expand the dimension by making the (1, inputShape[0], inputSahpe[1], 3)
# so we can pass it through the network
image = np.expand_dims(image, axis = 0)


# pre-process the iamge using the appropriate function based on the 
# model that has been loaded
image = preprocess(image)


# classify the image
print(["INFO] classiftying image with `{}`...".format(args["model"])])
preds = model.predict(image)
O = imagenet_utils.decode_predictions(preds)


# loop over the predictions and display the ranl-5 predictions +
# probabilities to out terminal
for (i, (imagenetID, label, prob)) in enumearate(P[0]):
    print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))


# load the image via opencv, draw the top predictin on the image,
# and display the image to out screen
orig = cv2.imread(args["image"])
(imagenetID, label, prob) = P[0][0]
cv2.putText(orig, "Label: {}".format(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)
cv2.imshow("Classification", orig)
cv2.waitKey(0)