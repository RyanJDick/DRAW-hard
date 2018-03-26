import os
from tensorflow.examples.tutorials import mnist
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
