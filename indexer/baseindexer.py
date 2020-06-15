#import the necessary packages
from __future__ import print_function
import numpy as np
import datetime

class BaseIndexer(object):
    def __init__(self, dbPath, estNumImages=500, maxBufferSize=50000, dbResizeFactor =2, verbose= True):
        #store the database path, estimated number of images in the dataset, max buffer size, the resize factor and the verbosity settings
        self.dbPath = dbPath
        self.estNumImages = estNumImages
        self. maxBufferSize = maxBufferSize
        self.dbResizeFactor = dbResizeFactor
        self.verbose = verbose

        #initalize the indexes dictionary
        #n order to access a given row or insert data into the HDF5 dataset, we need to know the index of the row we are accessing, just like when working with NumPy arrays
        self.idxs = {}


    def _writeBuffers(self):
        #should take all available buffers and flush them; however, since this is the root class, we don’t have any concept on the number of buffers that need to be flushed.
        pass

    def _writeBuffer(self, dataset, datasetName, buf, idxName, sparse= False):

        '''
        accepts a single buffer and writes it to disk
        '''

        #if the buffer is a list, then computer the ending index based on the list length
        if type(buf) is list:
            end= self.idxs[idxName] + len(buf)

        #otherwise, assume that the buffer is a Numpy/scipy array, so 
        #computer the ending index base on the array shape

        else:
            end = self.idxs[idxName] + buf.shape[0]

        #check to see if the dataset needs to be resized
        if end > dataset.shape[0]:
            self._debug("triggering `{}` db resize".format(datasetName))
            self._resizeDataset(dataset, datasetName, baseSize=end)

        #if this is a sparse matrix, then conver the sparse matrix to a dense one so it can be written to file
        if sparse:
            buf= buf.toarray()

        #dump the buffer to file
        self._debug("writing `{}` buffer".format(datasetName))
        dataset[self.idxs[idxName]:end]= buf

    def _resizeDataset(self, dataset, dbName, baseSize=0, finished=0):
        #grab the original size of the dataset
        origSize = dataset.shape[0]

        #check to see if we are finished writting rows to the dataset, and if 
        #so make the new size the current index

        if finished>0:
            newSize = finished

        #otherwise we are enlarging the dataset so calculate the new size of the dataset
        else:
            newSize = baseSize * self.dbResizeFactor

        #determine the shape of (to be) the resized dataset
        shape = list(dataset.shape)
        shape[0] = newSize

        #show old vs new size of the dataset
        dataset.resize(tuple(shape))
        self._debug("old size of `{}`: {:,}; new size: {:,}".format(dbName, origSize,newSize))
 


    def _debug(self, msg, msgType="[INFO]"):

        # check to see the message should be printed
        if self.verbose:
            print("{} {} - {}".format(msgType, msg, datetime.datetime.now()))

    @staticmethod
    def featureStack(array, accum=None, stackMethod=np.vstack):
        # if the accumulated array is None, initialize it
        if accum is None:
            accum = array

        # otherwise, stack the arrays
        else:
            accum = stackMethod([accum, array])

        # return the accumulated array
        return accum