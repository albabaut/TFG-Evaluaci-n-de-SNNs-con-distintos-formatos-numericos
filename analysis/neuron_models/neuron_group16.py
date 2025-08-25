import posit_wrapper
from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

def run_simulation(data_type, I_val):
    if data_type == "posit16":
        conv = lambda x: posit_wrapper.convert16(x)
    elif data_type == "float16":
        conv = lambda x: np.float16(x)
    elif data_type == "float32":
        conv = lambda x: np.float32(x)
    else:
        conv = lambda x: np.float64(x)

    tau = 10 * ms
    V_rest = -70 * mV
    V_reset = -60 * mV
    V_threshold = -50 * mV
    g_leak = 1 * nS

    I_input = conv(I_val) * nA

    eqs = '''
    dv/dt = (-(V_rest - v) + I/g_leak) / tau : volt
    I : amp
    '''

    neuron = NeuronGroup(1, eqs, threshold='v > V_threshold', reset='v = V_reset', method='euler')
    neuron.v = V_rest
    neuron.I = I_input

    monitor = StateMonitor(neuron, 'v', record=True)
    net = Network(neuron, monitor)
    net.run(100 * ms)

    return monitor.v[0] / mV

# Barrido de intensidades
I_vals = np.linspace(0.25, 4.0, 50)
errors_f32 = []
errors_f16 = []
errors_p16 = []

print("Ejecutando barrido de intensidades...")

for I in I_vals:
    v_ref = run_simulation("float64", I)
    v_f32 = run_simulation("float32", I)
    v_f16 = run_simulation("float16", I)
    v_p16 = run_simulation("posit16", I)

    min_len = min(len(v_ref), len(v_f32), len(v_f16), len(v_p16))
    v_ref = v_ref[:min_len]
    v_f32 = v_f32[:min_len]
    v_f16 = v_f16[:min_len]
    v_p16 = v_p16[:min_len]

    cumsum_f32 = np.sum(np.abs(v_f32 - v_ref))
    cumsum_f16 = np.sum(np.abs(v_f16 - v_ref))
    cumsum_p16 = np.sum(np.abs(v_p16 - v_ref))

    errors_f32.append(cumsum_f32)
    errors_f16.append(cumsum_f16)
    errors_p16.append(cumsum_p16)

# 📈 Graficar error acumulado final vs corriente
plt.figure(figsize=(10, 6))
plt.plot(I_vals, errors_f32, label="float32", linewidth=2)
plt.plot(I_vals, errors_f16, label="float16", linewidth=2)
plt.plot(I_vals, errors_p16, label="posit<16,2>", linewidth=2)
plt.xlabel("Corriente de entrada I (nA)")
plt.ylabel("Error acumulado respecto a float64 (mV)")
plt.title("Error acumulado final vs corriente")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("error_acumulado_vs_I.png", dpi=300)
plt.show()
