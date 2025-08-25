import brian2 as b2
import numpy as np
import matplotlib.pyplot as plt
import os
from posit_wrapper import convert16

b2.start_scope()

# ===============================
# PARÁMETROS PARA FORZAR VOLTAJES EN EL RANGO DONDE POSIT16 ES MEJOR
# Rango objetivo: aproximadamente -16 mV a -1 mV
# ===============================
tau = 10 * b2.ms
V_rest = -16 * b2.mV
Vth = -1 * b2.mV
Vreset = -16 * b2.mV
I_ext = 1.5 * b2.mV  # Ajusta si quieres que suba más rápido o lento
simulation_time = 5 * b2.ms  # Suficiente para ver la evolución

# Ecuación LIF
eqs = '''
dv/dt = (-(V_rest - v) + I_ext) / tau : volt (unless refractory)
'''

# --- 1. Simulación float64 (óptimo) ---
b2.device.reinit()
b2.device.activate()
G64 = b2.NeuronGroup(1, eqs, threshold='v > Vth', reset='v = Vreset',
                     refractory=5*b2.ms, method='euler', dtype=np.float64)
G64.v = V_rest
mon64 = b2.StateMonitor(G64, 'v', record=True)
spk64 = b2.SpikeMonitor(G64)
net64 = b2.Network(G64, mon64, spk64)
net64.run(simulation_time, namespace={'I_ext': I_ext, 'tau': tau, 'V_rest': V_rest, 'Vth': Vth, 'Vreset': Vreset})
v64 = mon64.v[0] / b2.mV
times64 = mon64.t / b2.ms

# --- Arrays para guardar los valores de cada paso ---
float16_vals = []
posit16_vals = []

# --- 2. Simulación float16 cuantizado (float32 + conversión a float16 tras cada paso) ---
b2.device.reinit()
b2.device.activate()
G_f16q = b2.NeuronGroup(1, eqs, threshold='v > Vth', reset='v = Vreset',
                        refractory=5*b2.ms, method='euler', dtype=np.float32)
G_f16q.v = V_rest
step_counter_f16 = {'i': 0}
def print_and_convert_to_float16():
    v_before = float(G_f16q.v[0] / b2.mV)
    v_f16 = np.float16(v_before)
    float16_vals.append((step_counter_f16['i'], v_before, float(v_f16)))
    G_f16q.v[0] = float(v_f16) * b2.mV
    step_counter_f16['i'] += 1
float16_converter = b2.NetworkOperation(print_and_convert_to_float16, dt=0.1*b2.ms)
mon_f16q = b2.StateMonitor(G_f16q, 'v', record=True)
spk_f16q = b2.SpikeMonitor(G_f16q)
net_f16q = b2.Network(G_f16q, float16_converter, mon_f16q, spk_f16q)
net_f16q.run(simulation_time, namespace={'I_ext': I_ext, 'tau': tau, 'V_rest': V_rest, 'Vth': Vth, 'Vreset': Vreset})
v_f16q = mon_f16q.v[0] / b2.mV
times_f16q = mon_f16q.t / b2.ms

# --- 3. Simulación posit16 cuantizado (float32 + conversión a posit16 tras cada paso) ---
b2.device.reinit()
b2.device.activate()
G_positq = b2.NeuronGroup(1, eqs, threshold='v > Vth', reset='v = Vreset',
                         refractory=5*b2.ms, method='euler', dtype=np.float32)
G_positq.v = V_rest
step_counter_posit = {'i': 0}
def print_and_convert_to_posit16():
    v_before = float(G_positq.v[0] / b2.mV)
    v_posit = convert16(v_before)
    posit16_vals.append((step_counter_posit['i'], v_before, float(v_posit)))
    G_positq.v[0] = float(v_posit) * b2.mV
    step_counter_posit['i'] += 1
posit16_converter = b2.NetworkOperation(print_and_convert_to_posit16, dt=0.1*b2.ms)
mon_positq = b2.StateMonitor(G_positq, 'v', record=True)
spk_positq = b2.SpikeMonitor(G_positq)
net_positq = b2.Network(G_positq, posit16_converter, mon_positq, spk_positq)
net_positq.run(simulation_time, namespace={'I_ext': I_ext, 'tau': tau, 'V_rest': V_rest, 'Vth': Vth, 'Vreset': Vreset})
v_positq = mon_positq.v[0] / b2.mV
times_positq = mon_positq.t / b2.ms

# --- Sincronizar arrays y crear tabla ---
n_steps = min(len(float16_vals), len(posit16_vals))
tabla = []
for i in range(n_steps):
    paso = float16_vals[i][0]
    v_orig = float16_vals[i][1]
    v_f16 = float16_vals[i][2]
    v_posit = posit16_vals[i][2]
    err_f16 = abs(v_f16 - v_orig)
    err_posit = abs(v_posit - v_orig)
    tabla.append((paso, v_orig, v_f16, v_posit, err_f16, err_posit))

# --- Imprimir tabla (primeros 30 pasos) ---
print("\nTabla de cuantización (primeros 30 pasos):")
print(f"{'Paso':>4} | {'Original':>10} | {'float16':>10} | {'posit16':>10} | {'Err_f16':>10} | {'Err_posit':>10}")
print("-"*68)
for row in tabla[:30]:
    print(f"{row[0]:4d} | {row[1]:10.6f} | {row[2]:10.6f} | {row[3]:10.6f} | {row[4]:10.6f} | {row[5]:10.6f}")

# --- Gráfica de errores de cuantización ---
pasos = [row[0] for row in tabla]
err_f16 = [row[4] for row in tabla]
err_posit = [row[5] for row in tabla]

plt.figure(figsize=(10,5))
plt.plot(pasos, err_f16, 'bo-', label='Error float16', markersize=4)
plt.plot(pasos, err_posit, 'ro-', label='Error posit16', markersize=4)
plt.xlabel('Paso de integración')
plt.ylabel('Error absoluto de cuantización (mV)')
plt.title('Error de cuantización por paso: float16 vs posit16 (rango posit16 mejor)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
os.makedirs('results/analyze_utils', exist_ok=True)
plt.savefig('results/analyze_utils/step_by_step_comparison_posit_better.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Gráfica comparativa ---
plt.figure(figsize=(10,6))
plt.plot(times64, v64, label='float64 (óptimo)', linewidth=2, color='black')
plt.plot(times_f16q, v_f16q, label='float16 cuantizado', linewidth=2, color='blue')
plt.plot(times_positq, v_positq, label='posit16 cuantizado', linewidth=2, color='red')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Voltaje (mV)')
plt.title('Comparación LIF: float64 vs float16 cuantizado vs posit16 cuantizado')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
os.makedirs('results/analyze_utils', exist_ok=True)
plt.savefig('results/analyze_utils/step_by_step_comparison.png', dpi=150, bbox_inches='tight')
plt.show() 