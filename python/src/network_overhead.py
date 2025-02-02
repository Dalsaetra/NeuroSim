import numpy as np

class NeuralNetwork:
    def __init__(self, n_neurons, max_synapses):
        assert max_synapses <= n_neurons

        self.n_neurons = n_neurons
        self.max_synapses = max_synapses

        # Connectivity matrix
        # M[i, j] = k, k is the neuron index where the jth axon of the ith neuron ends up
        self.M = np.zeros((n_neurons, max_synapses), dtype=bool)
        
        # Weight matrix
        self.W = np.zeros((n_neurons, max_synapses))

    # Initialize the connectivity matrix
    def set_connectivity(self, connectivity):
        self.M = connectivity

    # Set connectivity to be random
    def set_random_connectivity(self, replace=False):
        # Every row of M has max_synapses random unique indices from 0 to n_neurons-1
        for i in range(self.n_neurons):
            self.M[i] = np.random.choice(self.n_neurons, self.max_synapses, replace=replace)

    # Initialize the weight matrix
    def set_weights(self, weights):
        self.W = weights

    # Set weights to be random
    def set_random_weights(self, distribution='uniform', **kwargs):
        if distribution == 'uniform':
            self.W = np.random.uniform(**kwargs)
        elif distribution == 'normal':
            self.W = np.random.normal(**kwargs)
        else:
            raise ValueError('Invalid distribution')