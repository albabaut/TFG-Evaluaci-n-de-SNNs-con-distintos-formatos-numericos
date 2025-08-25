import posit_wrapper
from brian2 import *  
import matplotlib.pyplot as plt

# Parámetros de la neurona
tau = 10 * ms  
V_rest = -70 * mV  
V_reset = -65 * mV  
V_threshold = -50 * mV  
g_leak = 10 * nS  # Conductancia de fuga en nanoSiemens

# Corriente de entrada utilizando Posit
I_posit_value = posit_wrapper.convert(1.5)  # Convertir a formato Posit
I_posit = I_posit_value * nA  # Convertir a nanoamperios (A)

print(f"Corriente convertida a Posit: {I_posit}")

# Ecuación diferencial de la neurona
eqs = '''
dv/dt = (V_rest - v + I / g_leak) / tau : volt
I : amp  # Definimos I como una variable dentro del modelo
'''

# Crear la neurona
neuron = NeuronGroup(1, eqs, threshold='v>V_threshold', reset='v = V_reset', method='euler')

# Inicializar valores
neuron.v = V_rest
neuron.I = I_posit  # Asignamos la corriente convertida a la neurona

# Monitorear los voltajes
monitor = StateMonitor(neuron, 'v', record=True)

# Ejecutar la simulación
run(100 * ms)

# Graficar los resultados
plt.plot(monitor.t / ms, monitor.v[0] / mV)
plt.xlabel("Tiempo (ms)")
plt.ylabel("Voltaje (mV)")
plt.title("Simulación de una neurona con Posit")
plt.savefig("output.png")
print("Gráfica guardada en 'output.png'. Ábrela manualmente para verla.")

