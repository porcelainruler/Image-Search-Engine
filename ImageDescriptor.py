import numpy as np
import cv2
import imutils

import warnings

warnings.filterwarnings("ignore")
print(cv2.__version__)


class ImageDescriptor:
    def __init__(self, bins):
        super(ImageDescriptor, self).__init__()

        # No. of Bins for all three (h, s, v) Values
        self.bins = bins

    def featureExtracter(self, image):
        # Converting Image to HSV format and initialise feature to Describe / Quatify Image

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features = list()

        h, w = image.shape[:2]
        cX, cY = int(w * 0.5), int(h * 0.5)

        # Divide Image into 4 Segments - [Top Left, Top Right, Bottom Right, Bottom Left]
        segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h), (0, cX, cY, h)]

        # Middle Elliptical Region of Image


imgDesc = ImageDescriptor((2, 3, 4))
