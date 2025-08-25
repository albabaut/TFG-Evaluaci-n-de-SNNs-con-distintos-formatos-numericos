import posit_wrapper
from brian2 import *
import numpy as np
import pandas as pd

def run_simulation(data_type):
    if data_type == "posit":
        conversion_func = lambda x: posit_wrapper.convert(x)
    elif data_type == "float16":
        conversion_func = lambda x: np.float16(x)
    else:
        conversion_func = lambda x: np.float64(x)

    tau = 10 * ms
    V_rest = -70 * mV
    V_reset = -60 * mV
    V_threshold = -50 * mV
    g_leak = 0.01 * nS

    I_input = conversion_func(1e-9)*amp
    #I_input = I_value * amp

    eqs = '''
    dv/dt = (-(V_rest - v) + I / g_leak) / tau : volt
    I : amp (shared)
    '''

    neuron = NeuronGroup(1, eqs, threshold='v>V_threshold', reset='v = V_reset', method='euler')
    neuron.v = V_rest
    neuron.I = I_input

    monitor = StateMonitor(neuron, 'v', record=True)
    net = Network(neuron, monitor)
    net.run(20 * ms)  # Simulación corta para imprimir valores fácilmente

    voltajes = monitor.v[0][:6] / mV  # primeros 6 valores
    return voltajes

# Ejecutar simulaciones
v64 = run_simulation("float64")
v16 = run_simulation("float16")
vp  = run_simulation("posit")

# Mostrar resultados
print("\nComparación de primeros 6 voltajes (en mV):")
print("{:<10} {:<20} {:<20} {:<20}".format("Paso", "Float64", "Float16", "Posit<16,1>"))
for i in range(6):
    print(f"{i:<10} {v64[i]:<20.8f} {v16[i]:<20.8f} {vp[i]:<20.8f}")
