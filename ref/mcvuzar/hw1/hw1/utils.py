import numpy as np


def sigmoid(x):
    """
     Parameters
     ----------
     x : np.ndarray input data

     Returns
     -------
     np.ndarray
         sigmoid of the input x
     """
    return 1/(1 + np.exp(-x))


def sigmoid_prime(x):
    """
         Parameters
         ----------
         x : np.ndarray input data

         Returns
         -------
         np.ndarray
             derivative of sigmoid of the input x

    """
    return np.exp(-x)/np.square(1 + np.exp(-x))


def random_weights(sizes):
    """
         Parameters
         ----------
         sizes : list of sizes

         Returns
         -------
         list of xavier initialized np arrays weight matrices
    """
    return [xavier_initialization(sizes[i],sizes[i+1]) for i in range(0, len(sizes)-1)]


def zeros_weights(sizes):
    """
         Parameters
         ----------
         sizes : list of sizes

         Returns
         -------
         list of zero np arrays weight matrices
    """
    return [np.zeros((sizes[i],sizes[i+1])) for i in range(0, len(sizes)-1)]


def zeros_biases(sizes):
    """
         Parameters
         ----------
         sizes : list of sizes

         Returns
         -------
         list of zero np arrays bias matrices
    """
    return [np.zeros(sizes[i]) for i in range(0, len(sizes))]


def create_batches(data, labels, batch_size):
    """
         Parameters
         ----------
         data : np.ndarray of input data
         labels : np.ndarray of input labels
         batch_size : int size of batch

         Returns
         -------
         list of tuples of (data batch of batch_size, labels batch of batch_size)
    """
    len_data = len(data)
    return [(data[i:min(i + batch_size, len_data)], labels[i:min(i + batch_size, len_data)]) 
            for i in range(0, len_data, batch_size)]

def add_elementwise(list1, list2):
    """
         Parameters
         ----------
         list1 : np.ndarray of numbers
         list2 : np.ndarray of numbers

         Returns
         -------
         list of sum of each two elements by index
    """
    return np.add(list1, list2)


def xavier_initialization(m, n):
    xavier = 1 / (m ** 0.5)
    return np.random.uniform(low=-xavier, high=xavier, size=(m, n))
