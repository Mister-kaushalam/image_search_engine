#Driver implementation 
#import the necessary packages

from __future__ import print_function
from descriptors.detectanddescribe import DetectAndDescribe
from indexer.featureindexer import FeatureIndexer
from imutils.feature import FeatureDetector_create, DescriptorExtractor_create
from imutils import paths
import argparse
import cv2
import imutils


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,help="Path to the directory that contains the images to be indexed")
ap.add_argument("-f", "--features-db", required=True,help="Path to where the features database will be stored") 
#approx nunber of images help estimate the size of our HDF5 dataset when the FeatureIndexer is initialized
ap.add_argument("-a", "--approx-images", type=int, default=500,help="Approximate # of images in the dataset")
#Writing feature vectors to HDF5 one at a time is highly inefficient. Instead, it’s much more effective to collect feature vectors into a large array in memory and then dump them to HDF5 when the buffer is full
ap.add_argument("-b", "--max-buffer-size", type=int, default=50000,help="Maximum buffer size for # of features to be stored in memory")
args = vars(ap.parse_args())

#initialize the key point detector, local invariant descriptor, and the descriptor pipeline

detector =  FeatureDetector_create("SURF")
descriptor = DescriptorExtractor_create("RootSIFT")
dad = DetectAndDescribe(detector, descriptor)

#initialize the feature indexer
fi = FeatureIndexer(args["features_db"], estNumImages = args["approx_images"], maxBufferSize = args["max_buffer_size"], verbose=True)

#loop over the dataset
for (i,imagePath) in enumerate(paths.list_images(args["dataset"])):
    #check to see if progress sould be displayed
    if i>0 and i%10==0:
        fi._debug("processed {} images".format(i), msgType="[PROGRESS]")

    #extract the image filename from the image path and then load the image
    filename = imagePath[imagePath.rfind("/") + 1:]
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=320)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #describe the image
    (kps , desc) = dad.describe(image)

    #if either the keypints or descriptors are None, then ignore the image
    if kps is None or desc is None:
        continue

    #index the features
    fi.add(filename, kps, desc)

#finish the indexing process
fi.finish()