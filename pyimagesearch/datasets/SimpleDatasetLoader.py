# SimpleDatasetLoader.py

import os
import cv2
import numpy as np


class SimpleDatasetLoader:
    '''
    '''

    def __init__(self, preprocessors = None):
        '''
            Define the preprocessors that are going to be
            applied to images of the dataset while loading.while

            parameters
            ----------
                preprocessors: preprocessing methods and algorithms to be applied

            returns
            -------
                None
        '''
        self.preprocessors = preprocessors

        # if preprocessors is None, initialize them as an
        # empty list
        if self.preprocessors is None:
            self.preprocessors = []


    def load(self, imagePaths, verbose = -1):
        '''
        '''

        # initialize the list of features and labels
        data = []
        labels = []

        # loop over the input images
        for (i, imagePath) in enumerate(imagePaths):
            # load the image and extract the class label assuming
            # that our path has the following format:
            # **  /path/to/dataset/{class}/{image}.jpg  **
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]

            # check if preprocessors are not None
            if self.preprocessors is not None:
                # loop over the preprocessors and apply each to
                # the image
                for p in self.preprocessors:
                    image = p.preprocess(image)

            # treat our preprocessed image as a feature vector
            # by updating the data list followed by the labels
            data.append(image)
            labels.append(label)

            # show and update every {verbose} images
            if verbose > 0:
                if i > 0 and (i + 1) % verbose == 0:
                    print("[INFO] processed {} / {}".format(i + 1, len(imagePaths)))

        # return a tuple of the data and labels
        return (np.array(data), np.array(labels))