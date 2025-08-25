import posit_wrapper
from brian2 import *  
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def run_simulation(data_type, csvfile):
    """
    Ejecuta la simulación con el tipo de dato especificado.
    :param data_type: 'posit', 'float32' o 'float64'
    :param title: Título del gráfico
    :param filename: Nombre del archivo de salida
    """
    if data_type == "posit":
        conversion_func = lambda x: posit_wrapper.convert(x)
    else:
        conversion_func = lambda x: np.float32(x)
    
    tau = 10 * ms  
    V_rest = -70 * mV  
    V_reset = -60 * mV  
    V_threshold = -50 * mV  
    g_leak = 0.01 * nS  # Conductancia de fuga en nanoSiemens
    
    I_value = conversion_func(5e-9)  # Convertir al tipo de dato correspondiente
    I_input = I_value * nA  # Convertir a nanoamperios (A)
    
    print(f"Corriente convertida a {data_type}: {I_input}")
    eqs = '''
    dv/dt = (-(V_rest - v) + I / g_leak) / tau : volt
    I : amp  # Definimos I como una variable dentro del modelo
    '''
    
    neuron = NeuronGroup(1, eqs, threshold='v>V_threshold', reset='v = V_reset', method='euler')
    neuron.v = V_rest
    neuron.I = I_input  # Asignamos la corriente convertida a la neurona

    # Monitores
    monitor = StateMonitor(neuron, 'v', record=True)
    spikemon = SpikeMonitor(neuron)

    # Crear y ejecutar la simulación
    net = Network(neuron, monitor, spikemon)  # Incluir todos los monitores
    net.run(500 * ms)

    # Verificar si la neurona dispara
    print(f"Cantidad de spikes: {spikemon.num_spikes}")

    # Guardar los datos en un CSV
    df = pd.DataFrame({
        "Tiempo (ms)": monitor.t / ms,
        "Voltaje (mV)": monitor.v[0] / mV
    })
    df.to_csv(csvfile, index=False)
    print(f"Datos guardados en {csvfile}")

# Ejecutar las simulaciones
run_simulation("posit", "csv_posit.csv")
run_simulation("float32",  "csv_float.csv")


