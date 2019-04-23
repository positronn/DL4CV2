# tiny_imagenet.config.py

from os import path


# define the paths to the training and validation directories
TRAIN_IMAGES = "examples/deepergooglenet/datasets/tiny-iamgenet-200/train"
VAL_IMAGES = "examples/deepergooglenet/datasets/tiny-iamgenet-200/val/images"

# define the path tot he file that maps validation filenames to their
# corresponding class labels
VAL_MAPPINGS = "examples/deepergooglenet/datasets/tiny-imagenet-200/val/val_annotations.txt"

# define the paths to the WordNet hierarchy files which are used
# to generate our class labels
WORDNET_IDS = "examples/deepergooglenet/datasets/tiny-imagenet-200/wnids.txt"
WORD_LABELS = "examples/deepergooglenet/datasets/tiny-imagenet-200/nwords.txt"


# since we do not have access to the testing data we need to
# take a number of images dfrom the training data and use it instead
NUM_CLASSES = 200
NUM_TEST_IMAGES = 50 * NUM_CLASSES

# define the path to the output training, validation and testing
# HDF5 files
TRAIN_HDF5 = "examples/deepergooglenet/datasets/tiny-imagenet-200/hdf5/train.hdf5"
VAL_HDF5 = "examples/deepergooglenet/datasets/tiny-imagenet-200/hdf5/val.hdf5"
TEST_HDF5 = "examples/deepergooglenet/datasets/tiny-imagenet-200/hdf5/test.hdf5"

# define the path to the dataset mean
DATASET_MEAN = "output/tiny-imagenet-200-mean.json"

OUTPUT_PATH = "output"
MODEL_PATH = path.sep.join([OUTPUT_PATH, "resnet_tinyimagenet.hdf5"])
FIG_PATH = path.sep.join([OUTPUT_PATH, "resnet56_tinyimagenet.png"])
JSON_PATH = path.sep.join([OUTPUT_PATH, "resnet56_tinyimagenet.json"])

