#import the necessary packages
from .searchresult import SearchResult
from sklearn.metrics import pairwise
from imutils.feature import DescriptorMatcher_create
import numpy as np
import datetime
import h5py
import cv2

class SpatialVerifier:
    def __init__(self, featuresDBPath, idf, vocab, ratio=0.75, minMatches = 10, reprojThresh=4.0):
        #ratio: David Lowe’s raw feature matching ratio, useful for false-positive match pruning.
        #minMatches required to perform spatial verification
        #reprojThresh: The re-projection threshold for the RANSAC algorithm (i.e., the \epsilon tolerance/”wiggle room” [in pixels] allowed between matched keypoints)
        self.idf = idf
        self.featuresDB = h5py.File(featuresDBPath) #open the feature database
        self.vocab = vocab
        self.ratio = ratio
        self.minMatches = minMatches
        self.reprojThresh = reprojThresh

    def rerank(self, queryKps, queryDescs, searchResult, numResults=10):
        #start the search timer and initialize the re-ranked results dicitonary
        startTime = datetime.datetime.now()
        reranked = {}

        #grab the image indexes from the initial search results and sort them in
        #ascending orfer so the feature indexes can be grabber from HDF5
        
        resultIdxs = np.array([r[-1] for r in searchResult.results])
        resultIdxs.sort()

        #loop over the starting and ending indexes into the features dataset for each image
        for (i, (start, end)) in zip(resultIdxs, self.featuresDB["index"][resultIdxs, ...]):
            #grab the rows from the feature dataset and break the rows into keypoints and feature vectors
            rows = self.featuresDB["features"][int(start): int(end)]
            (kps, descs)= (rows[: , :2], rows[:, 2:])

            #determine the matched inlier keypoints and grab the indexes of the matched keypoints into the bag of visual words
            bovwIdxs = self.match(queryKps, queryDescs.astype("float32"), kps, descs.astype("float32"))

            #provided the at least some keypoints were matched, the final score for the spatial verification is the sim of the idf values for the inlier words
            if bovwIdxs is not None:
                score = self.idf[bovwIdxs].sum()
                reranked[i] = score
            
        #if no spatially verified matched were found, return the initial search result object
        if len(reranked) ==0:
            return searchResult

        #otherwise sort the spatially verified results
        results = sorted([(v, self.featuresDB["image_ids"][k], k) for (k, v) in reranked.items()],reverse=True)

        #loop over the initial search results
        for (score, imageID, imageIdx) in searchResult.results:
            #only add the initial search result to the list of the results if the image has not been spatially verified
            if imageIdx not in reranked:
                results.append((score, imageID, imageIdx))

        #return the spatially verified and reranked results
        return SearchResult(results[:numResults], (datetime.datetime.now() - startTime).total_seconds())

    def match(self, kpsA, featuresA, kpsB, featuresB):
        #computer the raw matches and initialize the list of actual matches and list of inlier indexes
        matcher = DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresB, featuresA, 2)
        matches = []
        inlierIdxs = None

        #loop over the raw matches
        for m in rawMatches:
            #ensure the distance is within a certan ratio of each other
            if len(m) == 2 and m[0].distance < m[1].distance * self.ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # check to see if there are enough matches to process
        if len(matches) >= self.minMatches:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (i, _) in matches])
            ptsB = np.float32([kpsB[j] for (_, j) in matches])

            # compute the homography between the two sets of points and compute
            # the ratio of matched points
            (_, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, self.reprojThresh)

            # grab the inlier points (i.e., where the status > 0), then quantize
            # the features associated with these points
            idxs = np.where(status.flatten() > 0)[0]
            if idxs.shape[0] == 0:
                return None
            inlierIdxs = pairwise.euclidean_distances(featuresB[idxs], Y=self.vocab)
            inlierIdxs = inlierIdxs.argmin(axis=1)

        #return the list of inlier indexes
        return inlierIdxs

    def finish(self):
        self.featuresDB.close()