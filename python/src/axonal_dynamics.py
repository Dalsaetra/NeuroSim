import numpy as np

# Combination of axonal delay and weight multiplication to get the input to the synaptic dynamics

class AxonalDynamics:
    def __init__(self, connectivity, delays):
        self.connectivity = connectivity
        self.delays = delays

        # Registry contains [post-index, weight (synaptic output), delayed spike time]
        self.registry = []

    def __call__(self, weights, spikes, T):
        # spikes: n_neurons x 1
        # weights: n_neurons x n_synapses
        # delays: n_neurons x n_synapses
        # connectivity: n_neurons x n_synapses
        # Returns: n_neurons x 1
        if spikes.any():
            self.weights = weights
            self.T = T
            np.where(spikes, self.update_registry, None)

        # Get the synaptic input from the registry where the delayed spike time is less than T and the weight is non-zero
        if len(self.registry) > 0:
            registry = np.array(self.registry)
            post_indices = registry[:, 0].astype(int)
            weights = registry[:, 1]
            delayed_times = registry[:, 2]
            synaptic_input = np.zeros_like(spikes)
            mask = (delayed_times <= T) & (weights != 0)
            synaptic_input[post_indices[mask]] = weights[mask]
            self.registry = list(np.array(self.registry)[~mask])
        else:
            synaptic_input = np.zeros_like(spikes)

        return synaptic_input

    def update_registry(self, i):
        # i: neuron index
        # Update the registry for neuron i
        for j in range(self.connectivity.shape[1]):
            if self.connectivity[i, j] != -1:
                self.registry.append(np.array([self.connectivity[i, j], self.weights[i, j], self.T + self.delays[i, j]]))