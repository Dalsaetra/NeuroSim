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
    def __init__(self, n_neurons, max_synapses, neuron_params, inhibitory_mask, noise_sigma, dt, weight_mult = 1.0):

        self.dt = dt
        self.ts = []

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

        self.synapses = SynapticDynamics_Optimized(self.inhibitory_mask, dt, tau_ST=5, tau_LT=150, weight_mult=weight_mult)

        self.stats = NetworkStats()

        self.set_init_state_stable()


    def simulate(self, T, I):
        n_steps = int(T / self.dt)

        for i in range(n_steps):
            synaptic_input = self.axons(self.network.W, self.state.spike, i * self.dt)
            synaptic_output = self.synapses(synaptic_input, self.state.V, I[i]) + np.random.normal(0, self.noise_sigma, self.n_neurons)
            self.state = self.neuron_stepper(self.state(), self.neuron_params, synaptic_output, self.dt)

            self.stats.V_hist.append(self.state.V)
            self.stats.spike_hist.append(self.state.spike)
            self.stats.I_hist.append(synaptic_input)
            self.ts.append((i + 1) * self.dt)


    def set_init_state_stable(self):
        V = self.neuron_params[5] # Resting potential Vr
        u = np.zeros(self.n_neurons)
        spike = np.zeros(self.n_neurons, dtype=bool)

        self.state = Neuron_State(V, u, spike)

        self.stats.V_hist.append(self.state.V)
        self.stats.spike_hist.append(self.state.spike)
        self.stats.I_hist.append(np.zeros(self.n_neurons))
        self.ts.append(0)



def neuron_params_classic(type_distribution, excitatory_distribution, inhibitory_distribution, neuron_class="IZ", 
                          delta_V=None, bias=None, threshold_mult=None, threshold_decay=None):
    neuron_params = []
    n_neurons = sum(excitatory_distribution) + sum(inhibitory_distribution)
    if neuron_class == "IZ":
        # Excitatory neurons
        for i in range(len(type_distribution)):
            neuron_params += [np.array(neuron_type_IZ[type_distribution[i]]) for _ in range(excitatory_distribution[i])]

        # Inhibitory neurons
        for i in range(len(type_distribution)):
            neuron_params += [np.array(neuron_type_IZ[type_distribution[i]]) for _ in range(inhibitory_distribution[i])]

        inhibitory_mask = np.zeros(len(neuron_params), dtype=int)
        inhibitory_mask[sum(excitatory_distribution):] = 1.0

        neuron_params = np.array(neuron_params)

        if delta_V is not None:
            neuron_params = np.concatenate([neuron_params, delta_V], axis=1)
        else:
            neuron_params = np.concatenate([neuron_params, np.zeros((n_neurons, 1))], axis=1)

        if bias is not None:
            neuron_params = np.concatenate([neuron_params, bias], axis=1)
        else:
            neuron_params = np.concatenate([neuron_params, np.zeros((n_neurons, 1))], axis=1)

        if threshold_mult is not None:
            neuron_params = np.concatenate([neuron_params, threshold_mult], axis=1)
        else:
            neuron_params = np.concatenate([neuron_params, np.ones((n_neurons, 1))], axis=1)

        if threshold_decay is not None:
            neuron_params = np.concatenate([neuron_params, threshold_decay], axis=1)
        else:
            neuron_params = np.concatenate([neuron_params, np.ones((n_neurons, 1))], axis=1)

        
    elif neuron_class == "IZ_simple":
        raise NotImplementedError
    else:
        raise ValueError('Invalid neuron class')

    return neuron_params.T, inhibitory_mask

if __name__ == '__main__':
    dt = 0.1
    inter_dist = [10, 10, 10, 0, 0]
    inhib_dist = [0, 0, 0, 5, 4]
    type_dist = ["p23", "p23", "p23", "nb", "b"]
    neuron_class = "IZ"

    n_neurons = sum(inter_dist) + sum(inhib_dist)
    max_synapses = n_neurons
    noise_sigma = 100

    thr_mult = np.ones((n_neurons, 1)) * 1.25
    thr_decay = np.ones((n_neurons, 1)) * np.exp(-dt / 50)

    params, inhib_mask = neuron_params_classic(type_dist, inter_dist, inhib_dist, threshold_mult=thr_mult, threshold_decay=thr_decay)

    sim = NeuralSimulator(n_neurons, max_synapses, params, inhib_mask, noise_sigma, dt, weight_mult=10)

    T = 100
    n_steps = int(T / dt)

    t_p = int(50/dt)

    I = np.zeros((n_steps, n_neurons))
    # I[t_p*10:t_p*10 + t_p, 0] = 1000
    I[:, 0] = 500

    sim.simulate(T, I)