# inspect_model.py

import argparse
from keras.applications import VGG16


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--include-top", type = int, default = 1,
        help = "whether or not to include top of CNN")
args = vars(ap.parse_args())


# load the VGG16 network
print("[INFO] loading network...")
model = VGG16(weights = "imagenet",
            include_top = args["include_top"] > 0)


# loop over the layers in the network and display them
# to the console
print("[INFO] showing layers...")
for (i, layer) in enumerate(model.layers):
    print("[INFO] {}\t{}".format(i, layer.__class__.__name__))