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
        ellipticW, ellipticH = int(w * 0.75) // 2, int(h * 0.75) // 2
        ellipticMask = np.zeros(image.shape[:2], dtype='unit8')
        cv2.ellipse(image, (cX, cY), (ellipticW, ellipticH), 0, 0, 360, 255, -1)

        # Extracting Features for each Frame
        for segment in segments:
            startX, endX, startY, endY = segment

            # Separate / Minus Elliptic Mask from each Quadrant Segment to get Required Area/Region
            quadMask = np.zeros(image.shape[:2], dtype='unit8')
            cv2.rectangle(image, quadMask, (startX, startY), (endX, endY), 255, -1)
            quadMask = cv2.subtract(quadMask, ellipticMask)

            # Extract Colour Feature from Given Segment / Region and Add to Feature map
            feat = self.HSVhistogram(image, quadMask)
            features.extend(feat)

        # Extract Colour Feature from Given Segment / Region and Add to Feature map
        feat = self.HSVhistogram(image, ellipticMask)
        features.extend(feat)

        return features

    def HSVhistogram(self, image, mask):
        # Extract a 3D color histogram from the Masked region of the
        # image, using the available / supplied number of bins per channel
        hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins,
                            [0, 180, 0, 256, 0, 256])

        # Normalize the histogram for OpenCV 2.4
        if imutils.is_cv2():
            hist = cv2.normalize(hist).flatten()

        # Normalize handler for OpenCV 3+
        else:
            hist = cv2.normalize(hist, hist).flatten()

        # Return the histogram / Feature Map
        return hist






imgDesc = ImageDescriptor((2, 3, 4))
