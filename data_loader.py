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
        """
        raise NotImplementedError("next_train_batch(self, batch_size) not implemented.")

    def next_test_batch(self, batch_size):
        """
        Returns next test batch of size batch_size. If the end of the test set
        has been reached, return None.

        Params:
        batch_size - int: Number of data points to return.
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
        """
        x_train, _ = self._train_data.next_batch(batch_size)
        return x_train

    def next_test_batch(self, batch_size):
        """
        Returns next test batch of size batch_size. If the end of the test set
        has been reached, return NULL.

        Params:
        batch_size - int: Number of data points to return.
        """
        _cur_test_index += batch_size
        if _cur_test_index > _num_test_images:
            return None
        x_test, _ = self._test_data.next_batch(batch_size)
        return x_test
