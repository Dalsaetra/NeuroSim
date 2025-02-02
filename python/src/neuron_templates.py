# k, a, b, d, C, Vr/v_r/E_m, Vt/v_t, Vpeak, Vmin/c
# or simplified: a, b, c, d
neuron_type_IZ = {
    # From Izhekevich appendix
    "nb1": (0.3, 0.17, 5, 100, 20, -66, -40, 50, -45),
    "p23": (3, 0.01, 5, 400, 100, -60, -50, 40, -57.5),
    "b": (1, 0.15, 8, 200, 20, -55, -40, 25, -55),
    "nb": (1, 0.03, 8, 20, 100, -56, -42, 40, -50),
    "ss4": (3, 0.01, 5, 400, 100, -60, -50, 40, -55),
    "p4": (3, 0.01, 5, 400, 100, -60, -50, 50, -55),
    "p5_p6": (3, 0.01, 5, 400, 100, -60, -50, 40, -55),
    "TC": (1.6, 0.1, 15, 10, 200, -60, -50, 40, -60),
    "TI": (0.5, 0.05, 7, 50, 20, -60, -50, 20, -65),
    "TRN": (0.25, 0.015, 10, 50, 40, -60, -50, 0.0, -55),

    # From BrainPy website https://brainpy-examples.readthedocs.io/en/latest/neurons/Izhikevich_2003_Izhikevich_model.html?utm_source=chatgpt.com
    "Tonic_Spiking": (0.02, 0.04, -65, 2),
    "Phasic_Spiking": (0.02, 0.25, -65, 6),
    "Tonic_Bursting": (0.02, 0.2, -50, 2),
    "Phasic_Bursting": (0.02, 0.25, -55, 0.05),
    "Mixed_Mode": (0.02, 0.2, -55, 4),
    "Spike_Frequency_Adaptation": (0.01, 0.2, -65, 8),
    "Class_1_Excitability": (0.02, -0.1, -55, 6),
    "Class_2_Excitability": (0.2, 0.26, -65, 0),
    "Spike_Latency": (0.02, 0.2, -65, 6),
    "Subthreshold_Oscillations": (0.05, 0.26, -60, 0),
    "Resonator": (0.1, 0.26, -60, -1),
    "Integrator": (0.02, -0.1, -55, 6),
    "Rebound_Spike": (0.03, 0.25, -60, 4),
    "Rebound_Burst": (0.03, 0.25, -52, 0),
    "Threshold_Variability": (0.03, 0.25, -60, 4),
    "Bistability": (1, 1.5, -60, 0),
    "Depolarizing_After_Potentials": (1, 0.2, -60, -21),
    "Accommodation": (0.02, 1, -55, 4),
    "Inhibition_Induced_Spiking": (-0.02, -1, -60, 8),
    "Inhibition_Induced_Bursting": (-0.026, -1, -45, 0),

    # From https://hippocampome.org/php/Izhikevich_model.php?refreshed
    "DG_Granule": (0.45, 0.003, 24.48, 50, 38, -77.4, -44.9, 15.49, -66.47),
    "DG_Hilar_Ectopic_Granule": (2.9, 0.006, 6.17, 186, 483, -70.37, -55.22, -37.82, -62.23),
    "DG_Semilunar_Granule": (0.94, 0, -77.49, 182, 910, -78.38, -58.4, 2.61, -62.9),
    "DG_Mossy": (1.5, 0.004, -20.84, 117, 258, -63.67, -37.11, 28.29, -47.98),
    "DG_AIPRIM": (0.51, 0.002, 0.2, 6, 40, -63.39, -40.19, 7.37, -53.34),
    "DG_Axo_axonic": (1.52, 0.049, 12.27, -6, 77, -65.25, -54.35, 5.07, -64.47),
    "DG_Basket": (0.81, 0.097, 1.89, 553, 208, -61.02, -37.84, 14.08, -36.23),
    "DG_HICAP": (0.5, 0.039, -1.62, 49, 61, -61.28, -35.36, 38.79, -60.77),
    "DG_HIPP": (1.28, 0.006, 57.94, -58, 74, -59.01, -50.53, 0.57, -56.98),
    "DG_HIPROM": (2.16, 0.001, -21.24, 28, 510, -65.4, -45.28, -9.92, -49.31),
    "DG_MOLAX": (0.92, 0.001, -3.12, 8, 236, -54.91, -41.69, -11.61, -48.73),
    "DG_Total_Molecular_Layer": (0.92, 0.001, -3.12, 8, 236, -54.91, -41.69, -11.61, -48.73),
    "DG_MOPP": (0.67, 0.002, -32.42, 163, 250, -74.67, -6.83, 17.03, -42.93),
    "DG_Neurogliaform": (0.67, 0.002, -32.42, 163, 250, -74.67, -6.83, 17.03, -42.93),

    "CA3_Pyramidal": (0.79, 0.008, -42.55, 588, 366, -63.2, -33.6, 35.86, -38.87),
    "CA3c_Pyramidal": (3.01, 0.002, 19.36, 104, 244, -62.29, -45.27, 17.43, -47.37),
    "CA3_Giant": (0.61, 0.004, 1.84, 2, 96, -57.58, -37.12, 36.42, -49.45),
    "CA3_Granule": (2.25, 0.053, 1.54, 443, 241, -78.04, -64.88, -7.73, -68.01),
    "CA3_Axo_axonic": (3.96, 0.005, 8.68, 15, 165, -57.1, -51.72, 27.8, -73.97),
    "CA3_Horizontal_Axo_axonic": (0.63, 0.002, -16.44, 21, 154, -58.52, -33.5, 36.09, -38.5),
    "CA3_Basket": (1, 0.004, 9.26, -6, 45, -57.51, -23.38, 18.45, -47.56),
    "CA3_Basket_CCK+": (0.58, 0.006, -1.24, 54, 135, -59, -39.4, 18.27, -42.77),
    "CA3_LMR_Targeting": (1.1, 0.001, -61.26, 43, 54, -67.73, -24.09, 10.38, -42.74),
    "CA3_Lucidum_ORAX": (1.03, 0.042, -5.07, 68, 67, -60.02, -47.97, 11.09, -59.38),
    "CA3_Spiny_Lucidum": (3.13, 0.003, 10.12, 18, 590, -69.59, -52.97, 34.3, -58.62),
    "CA3_Mossy_Fiber_Associated": (0.55, 0.003, 3.69, 5, 185, -59.41, -36.59, 9.99, -43.55),
    "CA3_Mossy_Fiber_Associated_ORDEN": (1.38, 0.008, 12.93, 0, 209, -57.08, -39.1, 16.31, -40.68),
    "CA3_O_LM": (0.51, 0.01, 2.39, 6, 65, -60.04, -30.87, 19.77, -52.81),
    "CA3_Trilaminar": (0.93, 0, -18.76, 74, 251, -63.13, -55.64, 17.01, -52.62),

    "CA2_Pyramidal": (5.94, 0.001, -15.89, 74, 1630, -72.59, -58.78, 19.99, -62.65),
    "CA2_Basket": (5.12, 0.012, -28.48, 132, 150, -70.31, -49.91, -10.27, -57.08),
    "CA2_Wide_Arbor_Basket": (0.51, 0.006, -2.16, 66, 148, -63.2, -32.84, 35.01, -38.33),
    "CA2_Bistratified": (0.8, 0.008, 4.75, 4, 118, -74.1, -40.19, 11.46, -53.61),
    "CA2_SP_SR": (3.59, 0.003, 48.43, 64, 226, -71.92, -57.05, -6.5, -66.94),

    "CA1_Pyramidal": (1.56, 0, -17.25, 16, 334, -69.36, -53.22, 25.46, -60.22),
    "CA1_Radiatum_Giant": (0.73, 0, 91.37, 100, 725, -66.09, -27.01, 13.37, -46.97),
    "CA1_Axo_axonic": (9.83, 0.002, 14.47, 26, 234, -66.76, -65.78, 3.18, -62.22),
    "CA1_Horizontal_Axo_axonic": (2.15, 0.03, -47.85, 277, 88, -56.45, -41.32, 23.89, -46.01),
    "CA1_Back_Projection": (0.67, 0.004, 15.66, -30, 133, -60.28, -47.12, 0.59, -60.14),
    "CA1_Basket": (1.19, 0.005, 0.22, 2, 114, -57.63, -35.53, 21.72, -48.7),
    "CA1_Horizontal_Basket": (0.83, 0.002, 0.26, 1, 46, -55.72, -27.75, 20.28, -45.42),
    "CA1_Basket_CCK+": (1.06, 0.023, -26.87, 294, 59, -63.18, -38.18, 27.44, -53.32),
    "CA1_Bistratified": (3.94, 0.002, 16.58, 19, 107, -64.67, -58.74, -9.93, -59.7),
    "CA1_Ivy": (1.92, 0.009, 1.91, 45, 364, -70.43, -40.86, -6.92, -53.4),
    "CA1_LMR": (0.65, 0.003, -12.12, 11, 40, -53.95, -43.83, 25.2, -54.05),
    "CA1_Neurogliaform": (2.36, 0.009, 17.56, 40, 254, -63.33, -47.62, 10.7, -50.78),
    "CA1_Neurogliaform_Projecting": (1.86, 0.003, -2.72, 90, 456, -63.35, -56.26, 17.06, -52.77),
    "CA1_O_LM": (4.47, 0.069, 74.3, 299, 73, -60, -56.41, 7.99, -58.16),
    "CA1_Recurrent_O_LM": (0.64, 0.001, 1.93, 20, 210, -68.46, -40.03, 24.6, -68.06),
    "CA1_O_LMR": (0.33, 0.006, 0.4, 48, 96, -56.44, -27.62, 29.48, -51.29),
    "CA1_Oriens_Alveus": (1.21, 0.056, -44.92, 416, 49, -51.65, -41.92, -5.45, -46.01),
    "CA1_Oriens_Bistratified": (2.91, 0.002, 13.67, 35, 841, -57.08, -48.47, 4.15, -52.91),
    "CA1_OR_LM": (0.56, 0.014, 2.09, -15, 248, -57.03, -42.75, 82.73, -45.49),
    "CA1_Perforant_Path_Associated": (4.84, 0.045, -49.31, 194, 12, -55.44, -42.5, -0.05, -51),
    "CA1_Perforant_Path_Associated_QuadD": (4.66, 0.096, -44.34, 290, 12, -64.88, -51.84, -2.99, -62.54),
    "CA1_Quadrilaminar": (1.78, 0.006, -3.45, 52, 186, -73.48, -54.94, 7.07, -64.4),
    "CA1_R_Receiving_Apical_Targeting": (0.67, 0.027, -9.29, 142, 92, -57.28, -43.19, -6.34, -46.58),
    "CA1_Radiatum": (3.34, 0.011, 14.89, 20, 194, -68.98, -58.15, -8.59, -68.03),
    "CA1_Schaffer_Collateral_Associated": (2.99, 0.002, 24.81, 34, 224, -75.31, -64.86, 0.53, -67.72),
    "CA1_SO_SO": (0.32, 0.008, 1.72, -5, 177, -77.24, -54.22, -22, -61.09),
    "CA1_Trilaminar": (0.16, 0.004, 2.72, -3, 38, -60.94, -46.19, -0.92, -55.34),
    "CA1_Radial_Trilaminar": (2.64, 0.009, 3.59, 2, 227, -57.87, -37.93, 4.77, -45.87),

    "Sub_CA1_Projecting_Pyramidal": (4.64, 0, 25.16, 3, 724, -59.4, -45.35, -18.92, -53.45),

    "EC_LI_II_Multipolar_Pyramidal": (0.37, 0.001, -4.08, 13, 204, -70.53, -39.99, 3.96, -54.95),
    "EC_LI_II_Pyramidal_Fan": (0.94, 0.01, -0.32, 283, 539, -50.04, -46.54, 29.22, -50.63),
    "MEC_LII_Oblique_Pyramidal": (0.62, 0.005, 11.69, 3, 118, -58.53, -43.52, 11.48, -49.52),
    "MEC_LII_Stellate": (0.62, 0.005, 11.69, 3, 118, -58.53, -43.52, 11.48, -49.52),
    "MEC_LII_III_Pyramidal_Multiform": (0.37, 0.001, -4.08, 13, 204, -70.53, -39.99, 3.96, -54.95),
    "EC_LII_III_Pyramidal_Tripolar": (0.37, 0.001, -4.08, 13, 204, -70.53, -39.99, 3.96, -54.95),
    "LEC_LIII_Multipolar_Principal": (0.31, 0.001, -3.36, 11, 250, -64.62, -35.32, 5.8, -48.63),
    "MEC_LIII_Multipolar_Principal": (0.95, 0.028, -21.99, 805, 407, -59.01, -39.01, 21.51, -38.58),
    "EC_LIII_Small_Pyramidal": (0.95, 0.028, -21.99, 805, 407, -59.01, -39.01, 21.51, -38.58),
    "LEC_LIII_Complex_Pyramidal": (0.31, 0.001, -3.36, 11, 250, -64.62, -35.32, 5.8, -48.63),
    "MEC_LIII_Complex_Pyramidal": (0.95, 0.028, -21.99, 805, 407, -59.01, -39.01, 21.51, -38.58),
    "MEC_LIII_Bipolar_Complex_Pyramidal": (0.95, 0.028, -21.99, 805, 407, -59.01, -39.01, 21.51, -38.58),
    "EC_LIII_Pyramidal": (0.31, 0.001, -3.36, 11, 250, -64.62, -35.32, 5.8, -48.63),
    "EC_LIII_Stellate": (0.95, 0.028, -21.99, 805, 407, -59.01, -39.01, 21.51, -38.58),
    "EC_LIII_V_Bipolar_Pyramidal": (0.62, 0, -22.09, 37, 240, -67.73, -40.44, 21.38, -48.28),
    "EC_LIV_VI_Deep_Multipolar_Principal": (0.62, 0, -22.09, 37, 240, -67.73, -40.44, 21.38, -48.28),
    "MEC_LV_Multipolar_Pyramidal": (0.21, 0, -16.14, 19, 157, -66.55, -32.89, 13.4, -51.2),
    "MEC_LV_Pyramidal": (0.21, 0, -16.14, 19, 157, -66.55, -32.89, 13.4, -51.2),
    "EC_LV_Deep_Pyramidal": (0.21, 0, -16.14, 19, 157, -66.55, -32.89, 13.4, -51.2),
    "MEC_LV_Superficial_Pyramidal": (0.21, 0, -16.14, 19, 157, -66.55, -32.89, 13.4, -51.2),
    "MEC_LV_VI_Pyramidal_Polymorphic": (0.55, 0, -21.05, 13, 218, -63.16, -39.45, 28.82, -51.14),
    "LEC_LVI_Multipolar_Pyramidal": (0.85, 0.001, -6.98, 21, 380, -63.54, -41.32, 14.2, -49.28),
    
    "MEC_LIII_Multipolar_Interneuron": (2.32, 0.003, 12.27, -2, 115, -57.15, -50.75, 2.43, -60.23),
    "MEC_LIII_Superficial_Multipolar_Interneuron": (3.5, 0.172, -48.95, 295, 29, -62, -41.46, 38.89, -67.53),
    "EC_LIII_Pyramidal_Looking_Interneuron": (1.23, 0.008, 10.68, 25, 521, -71.29, -37.78, 8.3, -50.42),
    "MEC_LIII_Superficial_Trilayered_Interneuron": (1.15, 0.004, 7.39, 1, 152, -59.18, -36.71, 17.73, -54.96),
}