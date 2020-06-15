#importing the necessary packages
from .baseindexer import BaseIndexer
import numpy as np
import h5py
import sys

class FeatureIndexer(BaseIndexer):
    def __init__(self, dbPath, estNumImages = 500, maxBufferSize = 50000, dbResizeFactor =2, verbose=True):
        #call the parent constructor
        super(FeatureIndexer, self).__init__(dbPath, estNumImages=estNumImages, maxBufferSize=maxBufferSize, dbResizeFactor=dbResizeFactor, verbose=verbose)

        #open the HDF5 database for writing and initialize the datasets within the group
        self.db = h5py.File(self.dbPath, mode="w")
        self.imageIDDB = None
        self.indexDB = None
        self.featureDB = None

        #initialize the imageIDs buffer, index buffer and the keypoints + feature buffer
        self.imageIDBuffer = []
        self.indexBuffer = []
        self.featuresBuffer = None

        #initialize the total number of features in the buffer along with the indexes dictionary
        self.totalFeatures = 0
        self.idxs = {"index": 0 , "features": 0}

    def add(self, imageID, kps, features):
        #computer the starting and ending index for the features loop
        start = self.idxs["features"] + self.totalFeatures
        end = start + len(features)

        #update the image IDs buffer, features buffer and index buffer
        #followed by incrementing the feature count
        self.imageIDBuffer.append(imageID)
        self.featuresBuffer = BaseIndexer.featureStack(np.hstack([kps, features]), self.featuresBuffer)
        self.indexBuffer.append((start,end))
        self.totalFeatures += len(features)

        #check to see if we have reached the maximum buffer size
        if self.totalFeatures >=self.maxBufferSize:
            #if the databases have not been created yet, create them
            if None in (self.imageIDDB, self.indexDB, self.featureDB):
                self._debug("initial buffer full")
                self._createDatasets()

            #write the buffers to the file
            self._writeBuffers()

    def _createDatasets(self):
        #computer the average number of features extracted from the inital buffer
        #use this number to determine the approximate number of features for the entire dataset

        avgFeatures = self.totalFeatures / float(len(self.imageIDBuffer))
        approxFeatures = int(avgFeatures * self.estNumImages)

        #grab the feature vector size 
        fvectorSize = self.featuresBuffer.shape[1]

        #handle the data type for python 2.7
        if sys.version_info[0] < 3:
            dt = h5py.special_dtype(vlen=unicode)

        #otherwise use a datatype compatible with python 3+
        else:
            dt = h5py.special_dtype(vlen= str)

        #initailize the dataset
        self._debug("creating datasets...")
        #if maxshape is not defined you'll not be able to resize the dataset later. Pass on None if the parameter is unknown
        self.imageIDDB = self.db.create_dataset("image_ids", (self.estNumImages,), maxshape=(None,), dtype=dt) 
        self.indexDB = self.db.create_dataset("index", (self.estNumImages,2), maxshape=(None, 2), dtype=dt)
        self.featureDB = self.db.create_dataset("features", (approxFeatures, fvectorSize), maxshape=(None, fvectorSize), dtype="float")

    def _writeBuffers(self):
        #write the buffers to disk
        self._writeBuffer(self.imageIDDB, "image_ids", self.imageIDBuffer, "index")
        self._writeBuffer(self.indexDB, "index", self.indexBuffer, "index")
        self._writeBuffer(self.featureDB, "features", self.featuresBuffer, "features")

        #increment the indexs
        self.idxs["index"] += len(self.imageIDBuffer)
        self.idxs["features"] += self.totalFeatures

        #reset the buffers and feature counts
        self.imageIDBuffer = []
        self.indexBuffer = []
        self.featuresBuffer = None
        self.totalFeatures = 0

    def finish(self):
        #if the databases have not been initalized then the original bufferes were never filled up
        if None in (self.imageIDDB, self.indexDB, self.featureDB):
            self._debug("minimum init biffer not reached", msgType="[WARN]")
            self._createDatasets

        #write any unempty buufferes to file
        self._debug("writing un-empty buffers...")
        self._writeBuffers()

        #compact datasets
        self._debug("compacting datasets...")
        self._resizeDataset(self.imageIDDB, "image_ids", finished=self.idxs["index"])
        self._resizeDataset(self.indexDB, "index", finished=self.idxs["features"])

        #close the database
        self.db.close()