# SimplePreprocessor.py

import cv2


class SimplePreprocessor:
    '''
    '''

    def __init__(self, width, height, inter = cv2.INTER_AREA):
        '''
            Store the target image width, height and interpolation
            method used when resizing.

            parameters
            ----------
                self: self object instance reference
                width:  width of resulting image
                height: height of resulting image
                inter: interpolation method to use when resizing

            returns
            -------
                None
        '''
        self.width = width
        self.height = height
        self.inter = inter


    def preprocess(self, image):
        '''
            Resize the image to a fixed size, ignoring the aspect
            ratio
        '''
        return cv2.resize(image, (self.width, self.height), interpolation = self.inter)

