# hdf5DatasetWriter.py

import os
import h5py



class HDF5DatasetWriter:
    '''
    Use to help store data in HDF5 format
    '''
    def __init__(self, dims:tuple, outputPath:str, dataKey:str = "images",
            bufSize:int = 1000):
        '''

        parameters
        ----------
            dims: controls the dimension or shape of the data we will be
                storing in the dataset.
            outputPath: path to where our output HDF5 file will be stored on disk.
            dataKey: name of the dataset that will store the data.
            bufSize:  controls the size of our in-memory buffer.
        '''
        # check to see if the output path exists, and if so,
        # raise an exception
        if os.path.exists(outputPath):
            raise ValueError("The suplpied `outputPath` already" \
                        "exists and cannot be overwritten. Manually delte " \
                        "the file before continuing. ", outputPath)


        # open the HDF5 database for writing and create two datasets:
        # one to store the images/features and another to store the
        # class labels
        self.db = h5py.File(outputPath, "w")
        self.data = self.db.create_dataset(dataKey, dims, dtype = "float")
        self.labels = self.db.create_dataset("labels", (dims[0],), dtype = "int")


        # store the buffer size, then initialize the buffer itself
        # along with the index into the datasets
        self.bufSize = bufSize
        self.buffer = {"data": [], "labels": []}
        self.idx = 0


    def add(self, rows, labels):
        '''
        '''
        # add the wors and labels to the buffer
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)

        # check to see if the byffer need to be flushed to disk
        if len(self.buffer["data"]) >= self.bufSize:
            self.flush()


    def flush(self):
        '''
        Write the buffers to file and reset them.
        '''
        # write the buffers to disk then reset the buffer
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data": [], "labels": []}


    def storeClassLabels(self, classLabels):
        '''
        If called, will store the raw string names of the class labels
        in a separate dataset.
        '''
        # create a dataset to store the actual class label names,
        # then store the class labels
        dt = h5py.special_dtype(vlen = str)
        labelSet = self.db.create_dataset("label_names",
                                (len(classLabels),),
                                dtype = dt)
        labelSet[:] = classLabels


    def close(self):
        '''
        '''
        # check to see if there are any other entries
        # in the buffer that need to be flushed to disk
        if len(self.buffer["data"]) > 0:
            self.flush()

        # close the dataset
        self.db.close()
