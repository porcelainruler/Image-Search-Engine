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

