# MeanPreprocessor.py

import cv2


class MeanPreprocessor:
    '''
    '''
    def __init__(self, rMean, gMean, bMean):
        # store the Red, Green, and Blue  channel averages across
        # a training set
        self.rMean = rMean
        self.gMean = gMean
        self.bMean = bMean


    def preprocess(self, image):
        # split the image into it respective Red, Green and Blue
        # Channels
        (B, G, R) = cv2.split(image.astype("float32"))


        # substract the means for each channel
        R -= self.rMean
        G -= self.gMean
        B -= self.bMean


        # merge the channels back together and return the image
        return cv2.merge([B, G, R])


