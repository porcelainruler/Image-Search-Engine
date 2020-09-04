from ImageDescriptor import ImageDescriptor
import config
import argparse
import glob
import cv2

# Configuring Argument Parser for parsing the Command Line Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', required=True, help='Path to Directory that contains Images to be Indexed')
parser.add_argument('-i', '--index', required=True, help='Path where Computed Index will be stored')

args = vars(parser.parse_args())

# Make Object of Image Descriptor Class that we have created
imgdesc = ImageDescriptor(config.bins)

# Output File for Writing Image Index / Vector
output = open(args['index'], 'w')

# Looping over Images in Database
for imagePath in glob.glob(args['dataset'] + "/*.png"):
    # First Extract ImageID and then load the Image
    imageID = imagePath[imagePath.rfind("/") + 1:]
    image = cv2.imread(imageID)

    # Features of Image
    features = imgdesc.featureExtracter(image)

    # Write Features of Imae as String in Output
    features = [str(f) for f in features]
    output.write("%s,%s\n" % (imageID, ",".join(features)))

output.close()

