import numpy as np

from neurons import *
from axonal_dynamics import *
from synaptic_dynamics import *
from network_overhead import *
from neuron_templates import *

class NetworkStats:
    def __init__(self):
        self.V_hist = []
        self.spike_hist = []
        self.I_hist = []


class NeuralSimulator:
    def __init__(self, n_neurons, max_synapses, neuron_params, inhibitory_mask, noise_sigma, dt):

        self.dt = dt

        self.n_neurons = n_neurons
        self.max_synapses = max_synapses
        
        self.neuron_params = neuron_params
        self.inhibitory_mask = inhibitory_mask

        self.noise_sigma = noise_sigma

        self.neuron_stepper = IZ_Neuron_stepper_adapt_deterministic

        self.network = NeuralNetwork(n_neurons, max_synapses)
        self.network.set_random_connectivity()
        self.network.set_random_weights(distribution='normal', clip=[0,1], loc=0.5, scale=0.2)
        
        self.axons = AxonalDynamics(self.network.M)
        self.axons.set_random_delays(distribution='uniform', clip=[0.1,10])

        self.synapses = SynapticDynamics_Optimized(self.inhibitory_mask, dt, tau_ST=5, tau_LT=150)

        self.set_init_state_stable()

        self.stats = NetworkStats()

    def simulate(self, T, I):
        n_steps = int(T / self.dt)

        for i in range(n_steps):
            synaptic_input = self.axons(self.network.W, self.state.spike, i * self.dt)
            synaptic_output = self.synapses(synaptic_input, self.state.V, I) + np.random.normal(0, self.noise_sigma, self.n_neurons)
            self.state = self.neuron_stepper(self.neuron_params, synaptic_output, self.dt)

            self.stats.V_hist.append(self.state.V)
            self.stats.spike_hist.append(self.state.spike)
            self.stats.I_hist.append(synaptic_input)


    def set_init_state_stable(self):
        V = self.neuron_params[5] # Resting potential Vr
        u = np.zeros(self.n_neurons)
        spike = np.zeros(self.n_neurons, dtype=bool)

        self.state = Neuron_State(V, u, spike)

        self.stats.V_hist.append(self.state.V)
        self.stats.spike_hist.append(self.state.spike)
        self.stats.I_hist.append(np.zeros(self.n_neurons))



def neuron_params_classic(type_distribution, excitatory_distribution, inhibitory_distribution, neuron_class="IZ", 
                          delta_V=None, bias=None, threshold_mult=None, threshold_decay=None):
    neuron_params = []
    if neuron_class == "IZ":
        # Excitatory neurons
        for i in range(len(type_distribution)):
            neuron_params += [np.array(neuron_type_IZ[type_distribution[i]]) for _ in range(excitatory_distribution[i])]

        # Inhibitory neurons
        for i in range(len(type_distribution)):
            neuron_params += [np.array(neuron_type_IZ[type_distribution[i]]) for _ in range(inhibitory_distribution[i])]

        neuron_params = np.array(neuron_params)

        if delta_V is not None:
            neuron_params.concatenate(delta_V)
        else:
            neuron_params.concatenate(np.zeros(len(neuron_params)))

        if bias is not None:
            neuron_params.concatenate(bias)
        else:
            neuron_params.concatenate(np.zeros(len(neuron_params)))

        if threshold_mult is not None:
            neuron_params.concatenate(threshold_mult)
        else:
            neuron_params.concatenate(np.ones(len(neuron_params)))

        if threshold_decay is not None:
            neuron_params.concatenate(threshold_decay)
        else:
            neuron_params.concatenate(np.ones(len(neuron_params)))

        
    elif neuron_class == "IZ_simple":
        raise NotImplementedError
    else:
        raise ValueError('Invalid neuron class')

    return neuron_params
