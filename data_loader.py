import os
from tensorflow.examples.tutorials import mnist
import scipy.io as sio
import urllib.request
import numpy as np

class DataLoader:
    """
    data_loader is an abstract class intended to be implemented for separately
    for each dataset.
    """
    def __init__(self, data_directory):
        """
        1. Checks if data_directory exists, and creates it if it does not.
        2. Checks if the dataset has been downloaded, and downloads it if it has
        not.
        3. Loads data so that next_batch() can be called immediately.
        """
        raise NotImplementedError("__init__(self, data_directory) not implemented.")

    def next_train_batch(self, batch_size):
        """
        Returns next data batch of size batch_size. If the end of the dataset is
        reached, this function automatically loops around to the first element
        in the dataset to satisfy the batch_size parameter.

        Params:
        batch_size - int: Number of data points to return.

        Return:
        data -  numpy array of size (B x H x W x C) where each value is in the
                range [0, 1]. RGB pixel values have to be normalized by dividing
                by 255
        """
        raise NotImplementedError("next_train_batch(self, batch_size) not implemented.")

    def next_test_batch(self, batch_size):
        """
        Returns next test batch of size batch_size. If the end of the test set
        has been reached, return None.

        Params:
        batch_size - int: Number of data points to return.

        Return:
        data -  numpy array of size (B x H x W x C) where each value is in the
                range [0, 1]. RGB pixel values have to be normalized by dividing
                by 255
        """
        raise NotImplementedError("next_test_batch(self, batch_size) not implemented.")

class MNISTLoader(DataLoader):
    """
    Implementation of data_loader to load mnist data.
    """
    dimensions = (28, 28, 1)

    def __init__(self, data_directory):
        """
        1. Checks if data_directory exists, and creates it if it does not.
        2. Checks if the dataset has been downloaded, and downloads it if it has
        not.
        3. Loads data so that next_batch() can be called immediately.
        """
        mnist_directory = os.path.join(data_directory, "mnist")
        if not os.path.exists(mnist_directory):
        	os.makedirs(mnist_directory)
        data = mnist.input_data.read_data_sets(data_directory, one_hot=True) # binarized (0-1) mnist data
        self._train_data = data.train
        self._test_data = data.test
        self._num_test_images = self._test_data.images.shape[0]
        self._cur_test_index = 0

    def next_train_batch(self, batch_size):
        """
        Returns next data batch of size batch_size. If the end of the dataset is
        reached, this function automatically loops around to the first element
        in the dataset to satisfy the batch_size parameter.

        Params:
        batch_size - int: Number of data points to return.

        Return:
        data -  numpy array of size (B x 28 x 28 x 1) where each entry is a
        binary value (0 or 1)
        """
        x_train, _ = self._train_data.next_batch(batch_size)
        # x_train is a 1D vector, reshape to image dimensions with single channel
        x_train = x_train.reshape((batch_size, 28, 28, 1))
        return x_train

    def next_test_batch(self, batch_size):
        """
        Returns next test batch of size batch_size. If the end of the test set
        has been reached, return None, and reset to start of test set.

        Params:
        batch_size - int: Number of data points to return.

        Return:
        data -  numpy array of size (B x 28 x 28 x 1) where each entry is a
        binary value (0 or 1)
        """
        self._cur_test_index += batch_size
        if self._cur_test_index > self._num_test_images:
            self._cur_test_index = 0
            return None
        x_test, _ = self._test_data.next_batch(batch_size)
        # x_test is a 1D vector, reshape to image dimensions with single channel
        x_test = x_test.reshape((batch_size, 28, 28, 1))
        return x_test

class SVHNLoader(DataLoader):
    """
    Implementation of data_loader to load SVHN data.
    """
    dimensions = (32, 32, 3)

    def __init__(self, data_directory):
        """
        1. Checks if data_directory exists, and creates it if it does not.
        2. Checks if the dataset has been downloaded, and downloads it if it has
        not.
        3. Loads data so that next_batch() can be called immediately.
        """
        svhn_directory = os.path.join(data_directory, "svhn")
        if not os.path.exists(svhn_directory):
        	os.makedirs(svhn_directory)

        svhn_train_file = os.path.join(svhn_directory, "train_32x32.mat")
        svhn_test_file = os.path.join(svhn_directory, "test_32x32.mat")

        if not os.path.exists(svhn_train_file):
            urllib.request.urlretrieve('http://ufldl.stanford.edu/housenumbers/train_32x32.mat', svhn_train_file)
            print("Downloaded: " + svhn_train_file)

        if not os.path.exists(svhn_test_file):
            urllib.request.urlretrieve('http://ufldl.stanford.edu/housenumbers/test_32x32.mat', svhn_test_file)
            print("Downloaded: " + svhn_test_file)

        self._train_data = sio.loadmat(svhn_train_file)['X']
        self._train_data = np.moveaxis(self._train_data, -1, 0) # (num_data_points, 32, 32, 3)
        self._train_data = self._train_data / 255 # Scale to range [0, 1]
        self._test_data = sio.loadmat(svhn_test_file)['X']
        self._test_data = np.moveaxis(self._test_data, -1, 0) # (num_data_points, 32, 32, 3)
        self._test_data = self._test_data / 255 # Scale to range [0, 1]

        self._num_train_images = self._train_data.shape[0]
        self._cur_train_index = 0

        self._num_test_images = self._test_data.shape[0]
        self._cur_test_index = 0

    def next_train_batch(self, batch_size):
        """
        Returns next data batch of size batch_size. If the end of the dataset is
        reached, this function automatically loops around to the first element
        in the dataset to satisfy the batch_size parameter.

        Params:
        batch_size - int: Number of data points to return.

        Return:
        data -  numpy array of size (B x 32 x 32 x 3)
        """
        if self._cur_train_index + batch_size > self._num_train_images:
            self._cur_train_index = 0
        start_index = self._cur_train_index
        self._cur_train_index += batch_size
        x_train = self._train_data[start_index:self._cur_train_index, :, :, :]
        return x_train

    def next_test_batch(self, batch_size):
        """
        Returns next test batch of size batch_size. If the end of the test set
        has been reached, return None, and reset to start of test set.

        Params:
        batch_size - int: Number of data points to return.

        Return:
        data -  numpy array of size (B x 28 x 28 x 1) where each entry is a
        binary value (0 or 1)
        """
        start_index = self._cur_test_index
        self._cur_test_index += batch_size
        if self._cur_test_index > self._num_test_images:
            self._cur_test_index = 0
            return None
        x_test = self._test_data[start_index:self._cur_test_index, :, :, :]
        return x_test
