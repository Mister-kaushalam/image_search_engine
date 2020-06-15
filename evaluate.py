#import the necessary packages
from __future__ import print_function
from descriptors.detectanddescribe import DetectAndDescribe
from ir.bagofvisualwords import BagOfVisualWords
from ir.searcher import Searcher
from ir.dists import chi2_distance
from scipy.spatial import distance
from redis import Redis
from imutils.feature import FeatureDetector_create, DescriptorExtractor_create
import numpy as np
import progressbar
import argparse
import pickle
import imutils
import json
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to the directory of indexed images")
ap.add_argument("-f", "--features-db", required=True, help="Path to the features database")
ap.add_argument("-b", "--bovw-db", required=True, help="Path to the bag-of-visual-words database")
ap.add_argument("-c", "--codebook", required=True, help="Path to the codebook")
ap.add_argument("-i", "--idf", type=str, help="Path to inverted document frequencies array")
ap.add_argument("-r", "--relevant", required=True, help="Path to relevant dictionary")
args = vars(ap.parse_args())

#initialize the keypoint detector, local invariant descriptor, descriptor pipeline, distance metric and inverted document frequency
detector = FeatureDetector_create("SURF")
descriptor = DescriptorExtractor_create("RootSIFT")
dad = DetectAndDescribe(detector, descriptor)
distanceMetric = chi2_distance
idf = None

#if the path to inverted document frequency array was supplied then load the idf array and update the distance materics
if args["idf"] is not None:
    idf = pickle.loads(open(args["idf"], "rb").read())
    distanceMetric = distance.cosine

#load the codebook vocabulary and initialize the bog-of-visual-words transformer
vocab = pickle.loads(open(args["codebook"], "rb").read())
bovw = BagOfVisualWords(vocab)

#connect to redis and intialize the searcher
redisDB = Redis(host="localhost", port=6379, db=0)
searcher = Searcher(redisDB, args["bovw_db"], args["features_db"], idf=idf, distanceMetric=distanceMetric)

#load the relevant queries dictionary
relevant = json.loads(open(args["relevant"]).read())
queryIDs = relevant.keys()

#initialize the accuray list and timings list
accuracies = []
timings = []

# initialize the progress bar
widgets = ["Evaluating: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(queryIDs), widgets=widgets).start()

#loop over the images
for (i,queryID) in enumerate(sorted(queryIDs)):
    #lookup the relevant results for query image
    queryRelevant = relevant[queryID]

    #load the image and process it
    p ="{}/{}".format(args["dataset"], queryID)
    queryImage = cv2.imread(p)
    queryImage = imutils.resize(queryImage, width=320)
    queryImage = cv2.cvtColor(queryImage, cv2.COLOR_BGR2RGB)

    #extract features from the query image and construct a bag of visual words from it
    (_, desc) = dad.describe(queryImage)
    hist = bovw.describe(desc).tocoo()

    #perform the search and computer the total number of relevant images in the top-4 tesults
    sr= searcher.search(hist, numResults=4)
    results = set([r[1] for r in sr.results])
    inter = results.intersection(queryRelevant)

    #update the evaluation lists
    accuracies.append(len(inter))
    timings.append(sr.search_time)
    pbar.update(i)

#release any pointers allocated by the search
searcher.finish()
pbar.finish()

#show evaluation information to the user
accuracies = np.array(accuracies)
timings = np.array(timings)
print("[INFO] ACCURACY: u={:.2f}, o={:.2f}".format(accuracies.mean(), accuracies.std()))
print("[INFO] TIMINGS: u={:.2f}, o={:.2f}".format(timings.mean(), timings.std()))