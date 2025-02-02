import numpy as np

# Simulating AMPA/NMDA/GABA_A/GABA_B synapses

class SynapticDynamics:
    def __init__(self, inhibitory_mask, dt, tau_AMPA=5, tau_NMDA=150, tau_GABA_A=6, tau_GABA_B=150, E_AMPA=0, E_NMDA=0, E_GABA_A=-70, E_GABA_B=-90):
        self.inhibitory_mask = inhibitory_mask

        self.dt = dt

        self.tau_AMPA = tau_AMPA
        self.tau_NMDA = tau_NMDA
        self.tau_GABA_A = tau_GABA_A
        self.tau_GABA_B = tau_GABA_B
        self.E_AMPA = E_AMPA
        self.E_NMDA = E_NMDA
        self.E_GABA_A = E_GABA_A
        self.E_GABA_B = E_GABA_B

        self.g_AMPA = np.zeros_like(inhibitory_mask, dtype=float)
        self.g_NMDA = np.zeros_like(inhibitory_mask, dtype=float)
        self.g_GABA_A = np.zeros_like(inhibitory_mask, dtype=float)
        self.g_GABA_B = np.zeros_like(inhibitory_mask, dtype=float)

        self.AMPA_decay = np.exp(-dt / tau_AMPA)
        self.NMDA_decay = np.exp(-dt / tau_NMDA)
        self.GABA_A_decay = np.exp(-dt / tau_GABA_A)
        self.GABA_B_decay = np.exp(-dt / tau_GABA_B)

    def __call__(self, synaptic_input, neurons_V, input):
        # synaptic_input: n_neurons x 1
        # neurons_V: n_neurons x 1
        # input: n_neurons x 1
        # Returns: n_neurons x 1
        self.g_AMPA += synaptic_input
        self.g_NMDA += synaptic_input
        self.g_GABA_A += synaptic_input
        self.g_GABA_B += synaptic_input

        I_AMPA = self.g_AMPA * (self.E_AMPA - neurons_V) * (1 - self.inhibitory_mask)
        V_shifted = (neurons_V + 80) / 60
        NMDA_factor = V_shifted**2 / (1 + V_shifted**2)
        I_NMDA = self.g_NMDA * NMDA_factor * (self.E_NMDA - neurons_V) * (1 - self.inhibitory_mask)
        I_GABA_A = self.g_GABA_A * (self.E_GABA_A - neurons_V) * self.inhibitory_mask
        I_GABA_B = self.g_GABA_B * (self.E_GABA_B - neurons_V) * self.inhibitory_mask

        self.g_AMPA *= self.AMPA_decay
        self.g_NMDA *= self.NMDA_decay
        self.g_GABA_A *= self.GABA_A_decay
        self.g_GABA_B *= self.GABA_B_decay

        return input + I_AMPA + I_NMDA + I_GABA_A + I_GABA_B
    

# Optimized when tau_AMPA=tau_GABA_A, and tau_NMDA=tau_GABA_B
class SynapticDynamics_Optimized:
    def __init__(self, inhibitory_mask, dt, tau_ST=5, tau_LT=150, E_AMPA=0, E_NMDA=0, E_GABA_A=-70, E_GABA_B=-90):
        self.inhibitory_mask = inhibitory_mask
        self.excitatory_mask = ~inhibitory_mask

        self.dt = dt

        self.tau_ST = tau_ST
        self.tau_LT = tau_LT
        self.E_ST = np.where(inhibitory_mask, E_GABA_A, E_AMPA)
        self.E_LT = np.where(inhibitory_mask, E_GABA_B, E_NMDA)

        self.g_ST = np.zeros_like(inhibitory_mask, dtype=float)
        self.g_LT = np.zeros_like(inhibitory_mask, dtype=float)

        self.ST_decay = np.exp(-dt / tau_ST)
        self.LT_decay = np.exp(-dt / tau_LT)


    def __call__(self, synaptic_input, neurons_V, input):
        # synaptic_input: n_neurons x 1
        # neurons_V: n_neurons x 1
        # input: n_neurons x 1
        # Returns: n_neurons x 1
        self.g_ST += synaptic_input
        self.g_LT += synaptic_input

        V_shifted = (neurons_V + 80) / 60
        NMDA_factor = V_shifted**2 / (1 + V_shifted**2)
        I_ST = self.g_ST * (self.E_ST - neurons_V)
        # Combine NMDA and GABA_B calculations
        V_diff = self.E_LT - neurons_V
        I_LT = self.g_LT * V_diff * (NMDA_factor * self.excitatory_mask + self.inhibitory_mask)

        self.g_ST *= self.ST_decay
        self.g_LT *= self.LT_decay

        return input + I_ST + I_LT