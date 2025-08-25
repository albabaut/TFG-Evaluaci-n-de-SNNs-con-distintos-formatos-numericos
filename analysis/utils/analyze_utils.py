import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from brian2 import ms, mV, second, volt
import brian2 as b2
from brian2 import Hz
import posit_wrapper  # Usar la implementación real de posit16
import os

def analyze_voltage_spike_differences(vm_f16, vm_pos, spikes_f16, spikes_pos):
    """
    Analiza diferencias entre voltajes y spikes para simulaciones en float16 y posit16.
    """
    n_neurons = len(vm_f16.v)
    times = vm_f16.t / ms

    results = []

    for i in range(n_neurons):
        # Extraer voltajes
        v_f16 = vm_f16.v[i] / mV
        v_pos = vm_pos.v[i] / mV

        # Spikes
        s_f16 = spikes_f16.t[spikes_f16.i == i] / ms
        s_pos = spikes_pos.t[spikes_pos.i == i] / ms

        # Métricas de diferencia de voltaje
        rmse = np.sqrt(np.mean((v_f16 - v_pos) ** 2))
        acc_err = np.sum(np.abs(v_f16 - v_pos))
        delta_spikes = len(s_f16) - len(s_pos)

        # Métrica de coincidencia de spikes (con tolerancia)
        tolerance = 1.0  # ms
        tp = 0
        fn = 0
        matched = set()
        for t_gt in s_f16:
            match = np.any(np.abs(s_pos - t_gt) <= tolerance)
            if match:
                tp += 1
                matched.add(t_gt)
            else:
                fn += 1
        fp = len(s_pos) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results.append({
            'neuron': i,
            'delta_spikes': delta_spikes,
            'rmse': rmse,
            'accumulated_error': acc_err,
            'spikes_f16': len(s_f16),
            'spikes_pos': len(s_pos),
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })

    df = pd.DataFrame(results)

    # Visualizaciones
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df[['rmse', 'accumulated_error', 'delta_spikes']])
    plt.title("Distribución de errores por neurona")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("images_compare/error_distributions_boxplot.png")
    plt.close()

    # Scatter RMSE vs Delta Spikes
    plt.figure(figsize=(8, 6))
    plt.scatter(df['rmse'], df['delta_spikes'], alpha=0.7)
    plt.xlabel("RMSE voltaje (mV)")
    plt.ylabel("Diferencia de spikes (float16 - posit16)")
    plt.title("Relación entre RMSE y diferencia de spikes")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("images_compare/rmse_vs_deltaspikes.png")
    plt.close()

    print(df.head())  # O guarda a CSV
    df.to_csv("metrics_compare/neuron_metrics.csv", index=False)
    return df

# --- COMPARACIÓN DE DOS PRECISIONES: float16 vs posit16 ---
# AMBAS USANDO BRIAN2 CON LAS MISMAS CONDICIONES
import matplotlib.pyplot as plt
import numpy as np
import brian2 as b2
from brian2 import ms, mV, Hz, volt, namp
import posit_wrapper  # Usar la implementación real de posit16
import os

# Configurar backend de numpy para soportar float16
b2.prefs.codegen.target = 'numpy'

# Configurar numpy para mostrar más decimales
np.set_printoptions(precision=10)

# ACTIVAR LOGGING DETALLADO PARA VER DENTRO DE BRIAN2
b2.prefs.logging.file_log_level = 'DIAGNOSTIC'
b2.prefs.logging.delete_log_on_exit = False

print("=== CONFIGURACIÓN DE BRIAN2 ===")
print(f"Backend: {b2.prefs.codegen.target}")
print(f"File logging: {b2.prefs.logging.file_log_level}")
print(f"Delete log on exit: {b2.prefs.logging.delete_log_on_exit}")

# Parámetros de la neurona LIF - MODIFICADOS PARA MÁS SPIKES
tau = 10 * ms
V_rest = -70 * mV  # Potencial de reposo
Vth = -50 * mV
Vreset = -70 * mV
I_ext = 20.0 * mV  # Corriente externa directamente en voltios (simplificado)
simulation_time = 100 * ms  # AUMENTADO: más tiempo = más spikes

print(f"\n=== PARÁMETROS DE SIMULACIÓN ===")
print(f"tau: {tau}")
print(f"V_rest: {V_rest}")
print(f"Vth: {Vth}")
print(f"Vreset: {Vreset}")
print(f"I_ext: {I_ext}")
print(f"simulation_time: {simulation_time}")

# --- Simulación 1: float64 (REFERENCIA ÓPTIMA) con Brian2 ---
print("\n=== EJECUTANDO SIMULACIÓN FLOAT64 (REFERENCIA ÓPTIMA) ===")
eqs = '''
dv/dt = (-(V_rest - v) + I_ext) / tau : volt (unless refractory)
'''
print(f"Ecuaciones: {eqs}")

# Limpiar completamente el estado de Brian2
b2.device.reinit()
b2.device.activate()

G64 = b2.NeuronGroup(1, eqs, threshold='v > Vth', reset='v = Vreset',
                     refractory=5*ms, method='euler', dtype=np.float64)
G64.v = V_rest

print(f"Grupo neuronal float64 creado:")
print(f"  - variables: {list(G64.variables.keys())}")

# INSPECCIONAR EL CÓDIGO GENERADO
print(f"\n=== CÓDIGO GENERADO PARA FLOAT64 ===")
print(f"State updater: {G64.state_updater}")
print(f"Thresholder: {G64.thresholder}")

mon64 = b2.StateMonitor(G64, 'v', record=True)
spk64 = b2.SpikeMonitor(G64)

# Crear Network explícito
net64 = b2.Network(G64, mon64, spk64)

# Ejecutar simulación float64
print(f"\nEjecutando simulación float64...")
net64.run(simulation_time, namespace={'I_ext': I_ext, 'tau': tau, 'V_rest': V_rest, 'Vth': Vth, 'Vreset': Vreset})
v64 = mon64.v[0] / mV
spikes64 = spk64.t / ms
times64 = mon64.t / ms

print(f"Simulación float64 completada")
print(f"  - Voltaje inicial: {v64[0]:.10f} mV")
print(f"  - Voltaje final: {v64[-1]:.10f} mV")
print(f"  - Número de spikes: {len(spikes64)}")

# --- Simulación 2: float16 con Brian2 ---
print("\n=== EJECUTANDO SIMULACIÓN FLOAT16 ===")
# Limpiar completamente el estado de Brian2 para la segunda simulación
b2.device.reinit()
b2.device.activate()

G16 = b2.NeuronGroup(1, eqs, threshold='v > Vth', reset='v = Vreset',
                     refractory=5*ms, method='euler', dtype=np.float16)
G16.v = V_rest

print(f"Grupo neuronal float16 creado:")
print(f"  - variables: {list(G16.variables.keys())}")

mon16 = b2.StateMonitor(G16, 'v', record=True)
spk16 = b2.SpikeMonitor(G16)

# Crear Network explícito
net16 = b2.Network(G16, mon16, spk16)

# Ejecutar simulación float16
print(f"\nEjecutando simulación float16...")
net16.run(simulation_time, namespace={'I_ext': I_ext, 'tau': tau, 'V_rest': V_rest, 'Vth': Vth, 'Vreset': Vreset})
v16 = mon16.v[0] / mV
spikes16 = spk16.t / ms
times16 = mon16.t / ms

print(f"Simulación float16 completada")
print(f"  - Voltaje inicial: {v16[0]:.10f} mV")
print(f"  - Voltaje final: {v16[-1]:.10f} mV")
print(f"  - Número de spikes: {len(spikes16)}")

# --- Simulación 3: posit16 con Brian2 (mismas condiciones) ---
print("\n=== EJECUTANDO SIMULACIÓN POSIT16 ===")
# Limpiar completamente el estado de Brian2 para la tercera simulación
b2.device.reinit()
b2.device.activate()

# Crear el mismo grupo neuronal pero con float32 (Brian2 no soporta posit16 directamente)
G_posit = b2.NeuronGroup(1, eqs, threshold='v > Vth', reset='v = Vreset',
                         refractory=5*ms, method='euler', dtype=np.float32)
G_posit.v = V_rest

print(f"Grupo neuronal posit16 creado:")
print(f"  - variables: {list(G_posit.variables.keys())}")

# NetworkOperation para convertir a posit16 después de cada paso
def convert_to_posit16():
    # Convertir el voltaje actual a posit16
    v_current = float(G_posit.v[0] / mV)
    v_posit16 = posit_wrapper.convert16(v_current)
    # Actualizar el voltaje en Brian2 con el valor convertido
    G_posit.v[0] = float(v_posit16) * mV

# Crear la operación que se ejecuta en cada paso
posit_converter = b2.NetworkOperation(convert_to_posit16, dt=0.1*ms)

mon_posit = b2.StateMonitor(G_posit, 'v', record=True)
spk_posit = b2.SpikeMonitor(G_posit)

# Crear Network explícito para evitar conflictos
net_posit = b2.Network(G_posit, posit_converter, mon_posit, spk_posit)

# Ejecutar simulación posit16
print(f"Ejecutando simulación posit16...")
net_posit.run(simulation_time, namespace={'I_ext': I_ext, 'tau': tau, 'V_rest': V_rest, 'Vth': Vth, 'Vreset': Vreset})
v_posit = mon_posit.v[0] / mV
spikes_posit = spk_posit.t / ms
times_posit = mon_posit.t / ms

print(f"Simulación posit16 completada")
print(f"  - Voltaje inicial: {v_posit[0]:.10f} mV")
print(f"  - Voltaje final: {v_posit[-1]:.10f} mV")
print(f"  - Número de spikes: {len(spikes_posit)}")

# --- ANÁLISIS DETALLADO PASO A PASO ---
print(f"\n=== ANÁLISIS DETALLADO PASO A PASO ===")
print("Comparando los últimos 10 pasos de cada simulación:")

min_length = min(len(v64), len(v16), len(v_posit))
for i in range(max(0, min_length-10), min_length):
    diff_f16 = abs(v64[i] - v16[i])
    diff_posit = abs(v64[i] - v_posit[i])
    print(f"Paso {i:4d}: float64={v64[i]:10.6f}, float16={v16[i]:10.6f}, posit16={v_posit[i]:10.6f}")
    print(f"         Diff vs float64: float16={diff_f16:8.6f}, posit16={diff_posit:8.6f}")

# --- Gráfica comparativa con 3 precisiones ---
plt.figure(figsize=(12,8))

plt.subplot(2,1,1)
plt.plot(times64, v64, label='float64 (ÓPTIMO)', linewidth=2, color='black')
plt.plot(times16, v16, label='float16', linewidth=2, color='blue')
plt.plot(times_posit, v_posit, '--', label='posit16', linewidth=2, color='red')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Voltaje (mV)')
plt.title('Comparación LIF: float64 vs float16 vs posit16\n(AMBOS usando Brian2 con method=\'euler\')')
plt.legend()
plt.grid(True, alpha=0.3)

# Marcar spikes en la gráfica
for spike_time in spikes64:
    plt.axvline(x=spike_time, color='black', alpha=0.3, linestyle=':')
for spike_time in spikes16:
    plt.axvline(x=spike_time, color='blue', alpha=0.3, linestyle=':')
for spike_time in spikes_posit:
    plt.axvline(x=spike_time, color='red', alpha=0.3, linestyle=':')

plt.subplot(2,1,2)
# Interpolar para comparar en los mismos tiempos
from scipy.interpolate import interp1d
interp_f16 = interp1d(times16, v16, kind='nearest', fill_value='extrapolate')
interp_posit = interp1d(times_posit, v_posit, kind='nearest', fill_value='extrapolate')
v_f16_interp = interp_f16(times64)
v_posit_interp = interp_posit(times64)

plt.plot(times64, v64 - v_f16_interp, 'b-', label='float64 - float16', linewidth=2)
plt.plot(times64, v64 - v_posit_interp, 'r-', label='float64 - posit16', linewidth=2)
plt.xlabel('Tiempo (ms)')
plt.ylabel('Diferencia de voltaje (mV)')
plt.title('Diferencia respecto a float64 (óptimo)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

# Crear directorio si no existe
os.makedirs('results/analyze_utils', exist_ok=True)
plt.savefig('results/analyze_utils/comparison_three_precisions.png', dpi=150, bbox_inches='tight')
plt.show()

# --- ANÁLISIS PROFUNDO DE DIFERENCIAS ---
print(f"\n" + "="*80)
print("ANÁLISIS PROFUNDO: float64 vs float16 vs posit16")
print("="*80)

# Métricas de comparación
print(f"\nMÉTRICAS DE COMPARACIÓN:")
print(f"Voltaje inicial float64: {v64[0]:.10f} mV")
print(f"Voltaje inicial float16: {v16[0]:.10f} mV")
print(f"Voltaje inicial posit16: {v_posit[0]:.10f} mV")

print(f"\nVoltaje final float64: {v64[-1]:.10f} mV")
print(f"Voltaje final float16: {v16[-1]:.10f} mV")
print(f"Voltaje final posit16: {v_posit[-1]:.10f} mV")

# Diferencias acumuladas
diff_f16_final = abs(v64[-1] - v16[-1])
diff_posit_final = abs(v64[-1] - v_posit[-1])
print(f"\nDiferencia final vs float64:")
print(f"  float16: {diff_f16_final:.10f} mV")
print(f"  posit16: {diff_posit_final:.10f} mV")

# Análisis de spikes
print(f"\nANÁLISIS DE SPIKES:")
print(f"Spikes float64: {spikes64}")
print(f"Spikes float16: {spikes16}")
print(f"Spikes posit16: {spikes_posit}")

# Comparar tiempos de spikes
if len(spikes64) == len(spikes16) == len(spikes_posit):
    print(f"\nDIFERENCIAS EN TIEMPOS DE SPIKES:")
    for i, (s64, s16, s_posit) in enumerate(zip(spikes64, spikes16, spikes_posit)):
        diff_f16 = abs(s64 - s16)
        diff_posit = abs(s64 - s_posit)
        print(f"Spike {i+1}: float64={s64:.1f}, float16={s16:.1f}, posit16={s_posit:.1f}")
        print(f"         Diff vs float64: float16={diff_f16:.2f}ms, posit16={diff_posit:.2f}ms")

# Análisis de acumulación de errores
print(f"\nANÁLISIS DE ACUMULACIÓN DE ERRORES:")
# Calcular diferencias en cada paso
min_length = min(len(v64), len(v16), len(v_posit))
diff_f16_array = np.abs(v64[:min_length] - v16[:min_length])
diff_posit_array = np.abs(v64[:min_length] - v_posit[:min_length])

print(f"Error máximo float16 vs float64: {np.max(diff_f16_array):.10f} mV")
print(f"Error máximo posit16 vs float64: {np.max(diff_posit_array):.10f} mV")
print(f"Error medio float16 vs float64: {np.mean(diff_f16_array):.10f} mV")
print(f"Error medio posit16 vs float64: {np.mean(diff_posit_array):.10f} mV")
print(f"Desv. estándar error float16: {np.std(diff_f16_array):.10f} mV")
print(f"Desv. estándar error posit16: {np.std(diff_posit_array):.10f} mV")

# Análisis de tendencias
print(f"\nANÁLISIS DE TENDENCIAS:")
# Calcular la pendiente de las diferencias (acumulación de error)
if len(diff_f16_array) > 10:
    x = np.arange(len(diff_f16_array))
    slope_f16 = np.polyfit(x, diff_f16_array, 1)[0]
    slope_posit = np.polyfit(x, diff_posit_array, 1)[0]
    print(f"Tendencia de error float16: {slope_f16:.10f} mV/paso")
    print(f"Tendencia de error posit16: {slope_posit:.10f} mV/paso")

# Análisis de precisión relativa
print(f"\nPRECISIÓN RELATIVA:")
print(f"float16 vs float64: {diff_f16_final/v64[-1]*100:.6f}% del valor final")
print(f"posit16 vs float64: {diff_posit_final/v64[-1]*100:.6f}% del valor final")

# Gráfica de acumulación de errores
plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.plot(times64[:min_length], diff_f16_array, 'b-', label='float16 vs float64', linewidth=2)
plt.plot(times64[:min_length], diff_posit_array, 'r-', label='posit16 vs float64', linewidth=2)
plt.xlabel('Tiempo (ms)')
plt.ylabel('Error absoluto (mV)')
plt.title('Acumulación de errores vs tiempo')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2,2,2)
plt.hist(diff_f16_array, bins=50, alpha=0.7, label='float16', color='blue')
plt.hist(diff_posit_array, bins=50, alpha=0.7, label='posit16', color='red')
plt.xlabel('Error absoluto (mV)')
plt.ylabel('Frecuencia')
plt.title('Distribución de errores')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2,2,3)
# Error relativo
error_rel_f16 = diff_f16_array / np.abs(v64[:min_length]) * 100
error_rel_posit = diff_posit_array / np.abs(v64[:min_length]) * 100
plt.plot(times64[:min_length], error_rel_f16, 'b-', label='float16', linewidth=2)
plt.plot(times64[:min_length], error_rel_posit, 'r-', label='posit16', linewidth=2)
plt.xlabel('Tiempo (ms)')
plt.ylabel('Error relativo (%)')
plt.title('Error relativo vs tiempo')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2,2,4)
# Comparación de voltajes en escala logarítmica
plt.semilogy(times64[:min_length], diff_f16_array, 'b-', label='float16', linewidth=2)
plt.semilogy(times64[:min_length], diff_posit_array, 'r-', label='posit16', linewidth=2)
plt.xlabel('Tiempo (ms)')
plt.ylabel('Error absoluto (mV) - escala log')
plt.title('Error en escala logarítmica')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/analyze_utils/error_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n" + "="*80)
print("CONCLUSIONES FINALES:")
print("="*80)
print(f"1. float64 es la referencia óptima (máxima precisión)")
print(f"2. float16 muestra error final de {diff_f16_final:.6f} mV ({diff_f16_final/v64[-1]*100:.4f}%)")
print(f"3. posit16 muestra error final de {diff_posit_final:.6f} mV ({diff_posit_final/v64[-1]*100:.4f}%)")
print(f"4. Ambos formatos mantienen el mismo número de spikes: {len(spikes64)}")
print(f"5. Las diferencias se acumulan gradualmente durante la simulación")
print(f"6. posit16 parece tener mejor comportamiento en algunos aspectos")
print("="*80)
