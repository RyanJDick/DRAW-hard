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
        if self._cur_train_index + batch_size > self._num_train_images:
            self._cur_train_index = 0
        start_index = self._cur_train_index
        self._cur_train_index += batch_size
        x_train = self._train_data[start_index:self._cur_train_index, :, :, :]
        return x_train

    def next_test_batch(self, batch_size):
        """
        Returns next test batch of size batch_size. If the end of the test set
        has been reached, return None, and reset so that the next cal restarts
        at the start of the test set.

        Params:
        batch_size - int: Number of data points to return.

        Return:
        data -  numpy array of size (B x H x W x C) where each value is in the
                range [0, 1]. RGB pixel values have to be normalized by dividing
                by 255
        """
        start_index = self._cur_test_index
        self._cur_test_index += batch_size
        if self._cur_test_index > self._num_test_images:
            self._cur_test_index = 0
            return None
        x_test = self._test_data[start_index:self._cur_test_index, :, :, :]
        return x_test

    def next_val_batch(self, batch_size):
        """
        Returns next validation batch of size batch_size. If the end of the
        validation set has been reached, return None and reset so that the
        following call to this method restarts at the beginning of the
        validtaion set.

        Params:
        batch_size - int: Number of data points to return.

        Return:
        data -  numpy array of size (B x H x W x C) where each value is in the
                range [0, 1]. RGB pixel values have to be normalized by dividing
                by 255
        """
        start_index = self._cur_val_index
        self._cur_val_index += batch_size
        if self._cur_val_index > self._num_val_images:
            self._cur_val_index = 0
            return None
        x_val = self._val_data[start_index:self._cur_val_index, :, :, :]
        return x_val


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
        train_data = data.train.images.reshape((-1, 28, 28, 1))
        self._val_data = train_data[:1000, :, :, :]
        self._train_data = train_data[1000:, :, :, :]
        self._test_data = data.test.images
        self._test_data = self._test_data.reshape((-1, 28, 28, 1))

        self._num_val_images = self._val_data.shape[0]
        self._cur_val_index = 0

        self._num_train_images = self._val_data.shape[0]
        self._cur_train_index = 0

        self._num_test_images = self._test_data.shape[0]
        self._cur_test_index = 0

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
        self._val_data = self._train_data[:1000, :, :, :]
        self._train_data = self._train_data[1000:, :, :, :]

        self._test_data = sio.loadmat(svhn_test_file)['X']
        self._test_data = np.moveaxis(self._test_data, -1, 0) # (num_data_points, 32, 32, 3)
        self._test_data = self._test_data / 255 # Scale to range [0, 1]

        self._num_val_images = self._val_data.shape[0]
        self._cur_val_index = 0

        self._num_train_images = self._train_data.shape[0]
        self._cur_train_index = 0

        self._num_test_images = self._test_data.shape[0]
        self._cur_test_index = 0
