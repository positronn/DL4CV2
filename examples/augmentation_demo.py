# augmentation_demo.py

import argparse
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img


# construct argparse
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
    help = "path tp the input image")
ap.add_argument("-o", "--output", required = True,
    help = "path to output directory to store augmentation examples")
ap.add_argument("-p", "--prefix", type = str, default = "image",
    help = "output filename prefix")
args = vars(ap.parse_args())


# load the input image, convert it to a numpy array
# and then reshape it to have an extra dimension
print("[INFO] loading example image...")
image = load_img(args["image"])
image = img_to_array(image)
image = np.expand_dims(image, axis = 0)


# construct the image generator for data augmentation
# then initialize the total number of iamges generated thus far
augmenation = ImageDataGenerator(rotation_range = 30,
                    width_shift_range = 0.1,
                    height_shift_range = 0.1,
                    shear_range = 0.2,
                    zoom_range = 0.2,
                    horizontal_flip = True, 
                    fill_mode = "nearest")



# construct actual generator
print("[INFO] generating images...")
imageGen = augmenation.flow(image, batch_size = 1,
                save_to_dir = args["output"],
                save_prefix = args["prefix"],
                save_format = "jpg")


total = 0
# loop over eaxmples from our image data augmentation generator
for image in imageGen:
    total += 1

    if total == 10:
        break

