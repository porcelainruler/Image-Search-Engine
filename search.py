from ImageDescriptor import ImageDescriptor
from searchEngine import searchEngine
import argparse
import cv2
import config

import warnings
warnings.filterwarnings('ignore')

# Setting Up / Configuring Parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--index', required=True, help='Path where Features/Index of Image in DB will be stored')
parser.add_argument('-q', '--query', required=True, help='Path to Query Image')
parser.add_argument('-r', '--result-path', required=True, help='Path where Result Images to Query will be saved')
args = vars(parser.parse_args())

# Initializing Image Descriptor
imgdesc = ImageDescriptor(config.bins)

# Loading and Extracting Query Image Features
qPath = args['query'] or config.qPath
query = cv2.imread(qPath)
features = imgdesc.featureExtracter(query)

# Performing the Search in DB
idxPath = args['index'] or config.idxPath
limit = args['limit'] or config.limit
searchfunc = searchEngine(idxPath)
results = searchfunc.search(QueryFeatures=features, limit=limit)

# Visualization Part

# Query Display
cv2.imshow(query)

# Result Display
rPath = args['result_path'] or config.rPath
for score, resultID in results:
    result = cv2.imread(args["result_path"] + "/" + resultID)
    cv2.imshow("Result", result)
    cv2.waitKey(0)

