import numpy as np

def IZ_Neuron_stepper_euler(states, params, I, dt):
    # states: states_per_neuron x n_neurons, params: params_per_neuron x n_neurons
    # states = [V, u]
    # params = [k, a, b, d, C, Vr, Vt, Vpeak, c, delta_V]
    # Vectorized version of IZ_Neuron.step_euler
    n_neurons = states.shape[1]
    k, a, b, d, C, Vr, Vt, Vpeak, c, delta_V, bias = params
    V, u = states

    spike = np.zeros(n_neurons, dtype=bool)

    dV = (k * (V - Vr) * (V - Vt) - u + I + bias)/C
    du = a * (b * (V - Vr) - u)

    V += dt * dV
    u += dt * du

    spike_prob = dt * np.exp((V - Vpeak) / delta_V)
    spike_rand = np.random.rand(n_neurons)
    spike = spike_rand < spike_prob

    V = np.where(spike, c, V)
    u = np.where(spike, u + d, u)

    return np.array([V, u, spike])