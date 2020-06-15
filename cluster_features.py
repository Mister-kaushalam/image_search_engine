# import the necessary packages
from __future__ import print_function
from ir.vocabulary import Vocabulary
import argparse
import pickle
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--features-db", required=True,help="Path to where the features database will be stored")
ap.add_argument("-c", "--codebook", required=True,help="Path to the output codebook")
ap.add_argument("-k", "--clusters", type=int, default=64,help="# of clusters to generate")
ap.add_argument("-p", "--percentage", type=float, default=0.25,help="Percentage of total features to use when clustering")
args = vars(ap.parse_args())

#create the visual words
voc = Vocabulary(args["features_db"])
vocab = voc.fit(args["clusters"], args["percentage"])

# dump the clusters to file
print("[INFO] storing cluster centers...")
f = open(args["codebook"], "wb")
f.write(pickle.dumps(vocab))
f.close()