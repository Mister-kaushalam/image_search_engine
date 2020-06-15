#import necessary packages
from __future__ import print_function
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import datetime
import h5py

class Vocabulary:
    def __init__(self, dbPath, verbose=True):
        self.dbPath = dbPath
        self.verbose = verbose

    def fit(self, numClusters, samplePercent, randomState=None):
        #opent the database and grab the total number of features
        db = h5py.File(self.dbPath)
        totalFeatures = db["features"].shape[0]

        #determince the number of features to sample, generate the indexes of the sample,
        #sorting them in ascending order to speedupp access time from the hdf5 databasee

        sampleSize = int(np.ceil(samplePercent * totalFeatures))

        idxs = np.random.choice(np.arange(0,totalFeatures), (sampleSize), replace=False)
        idxs.sort()
        data = []
        self._debug("starting sampling...")

        #loop over the randomly sampled indexes and accumulate the features to cluster
        for i in idxs:
            data.append(db["features"][i][2:])


        #cluster the data
        self._debug("sampled {:,} features from a population of {:,}".format(len(idxs), totalFeatures))
        self._debug("clustering with k={:,}".format(numClusters))

        clt = MiniBatchKMeans(n_clusters=numClusters, random_state=randomState)
        clt.fit(data)
        self._debug("cluster shape: {}".format(clt.cluster_centers_.shape))

        #close the database
        db.close()

        #return the cluster centroids
        return clt.cluster_centers_

    def _debug(self, msg, msgType="[INFO]"):
        #check to see if message should be printed
        if self.verbose:
            print("{} {} - {}".format(msgType, msg, datetime.datetime.now()))

