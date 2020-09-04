import numpy as np
import csv

import warnings
warnings.filterwarnings('ignore')

class searchEngine:
    def __init__(self, indexPath):
        super(searchEngine, self).__init__()
        self.indexPath = indexPath

    # Limit Tells No. of Search Results and QueryFeature tells about Feature of Image Provided as Query
    def search(self, QueryFeatures, limit=10):
        # Result Storage
        result = dict()

        with open(self.indexPath) as f:
            # Initialize CSV Reader
            read = csv.reader(csvfile=f)

            # Loop Over CSV File or Feature in Database
            for line in read:
                features = [float(val) for val in line[1:]]
                dist = self.chi2Dist(features, QueryFeatures)

                result[line[0]] = dist

            # Close the File
            f.close()

        result = sorted([(val, key) for (key, val) in result])

        return result[:limit]

    # Chi Square Distance Calculation Implementation
    def chi2Dist(self, feature1, feature2, nonZero=1e-10):
        # Chi Sq Distance Calculation - Averaged over all 5 Regions
        dist = 0.5 * np.sum([((a - b) ** 2) // (a + b + nonZero)] for (a, b) in zip(feature1, feature2))

        return dist


        return 1