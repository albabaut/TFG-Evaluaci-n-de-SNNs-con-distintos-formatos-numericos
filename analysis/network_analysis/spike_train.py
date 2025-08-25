import posit_wrapper
from brian2 import *
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Conversión a bfloat16
def to_bfloat16(x):
    x32 = np.array(x, dtype=np.float32)
    bits = x32.view(np.uint32)
    new_bits = (bits >> 16) << 16
    result = new_bits.view(np.float32)
    return float(result)

# Tren binario de picos
def spike_train_binary(spike_times, duration_ms, resolution_ms=1.0):
    train = np.zeros(int(duration_ms / resolution_ms))
    indices = (spike_times / resolution_ms).astype(int)
    indices = indices[indices < len(train)]
    train[indices] = 1
    return train

# Simulación para spike trains
def run_spike_simulation(data_type, I_override=0.1):
    if data_type == "posit16":
        conv = lambda x: posit_wrapper.convert16(x)
    elif data_type == "posit20":
        conv = lambda x: posit_wrapper.convert20(x)
    elif data_type == "posit24":
        conv = lambda x: posit_wrapper.convert24(x)
    elif data_type == "posit32":
        conv = lambda x: posit_wrapper.convert32(x)
    elif data_type == "float16":
        conv = lambda x: np.float16(x)
    elif data_type == "float32":
        conv = lambda x: np.float32(x)
    elif data_type == "bfloat16":
        conv = lambda x: to_bfloat16(x)
    else:
        conv = lambda x: np.float64(x)

    tau = 10 * ms
    V_rest = -70 * mV
    V_reset = -60 * mV
    V_threshold = -50 * mV
    g_leak = 1 * nS
    I_input = conv(I_override) * nA

    eqs = '''
    dv/dt = (-(V_rest - v) + I/g_leak) / tau : volt
    I : amp
    '''

    neuron = NeuronGroup(1, eqs, threshold='v > V_threshold', reset='v = V_reset', method='euler')
    neuron.v = V_rest
    neuron.I = I_input

    spikemon = SpikeMonitor(neuron)
    net = Network(neuron, spikemon)
    net.run(100 * ms)

    return spikemon.t[:] / ms

# Comparación entre tipos en múltiples valores de corriente
types = ["float64", "float32", "float16", "posit16", "posit20", "posit24", "posit32"]
currents = np.linspace(-4, 4, 1000)
duration_ms = 100
results = []

print("\n🔍 Comparación de spike trains binarios respecto a float64 en múltiples corrientes")

for I_val in currents:
    ref_spikes = run_spike_simulation("float64", I_override=I_val)
    binary_ref = spike_train_binary(ref_spikes, duration_ms)

    for dtype in types:
        spikes = run_spike_simulation(dtype, I_override=I_val)
        binary_test = spike_train_binary(spikes, duration_ms)
        rmse = np.sqrt(mean_squared_error(binary_ref, binary_test))
        total_spikes = int(np.sum(binary_test))
        firing_rate = total_spikes / (duration_ms / 1000.0)  # spikes por segundo
        sparsity = 1.0 - (total_spikes / len(binary_test))   # fracción de ceros
        results.append({
            'I': I_val,
            'type': dtype,
            'rmse_spike': rmse,
            'total_spikes': total_spikes,
            'firing_rate': firing_rate,
            'sparsity': sparsity
        })

# Mostrar resumen para una corriente concreta
print("\nTipo de dato       | Corriente (nA) | RMSE spike | Total spikes")
print("-------------------|----------------|------------|--------------")
for r in results:
    if np.isclose(r['I'], 0.1):
        print(f"{r['type']:18} | {r['I']:14.2f} | {r['rmse_spike']:.6f}  | {r['total_spikes']:>13}")

# Guardar resultados a CSV
import pandas as pd
df = pd.DataFrame(results)
df.to_csv("spike_train_results.csv", index=False)

# Visualización: spikes totales vs corriente
plt.figure(figsize=(10, 6))
for dtype in types:
    df_type = df[df['type'] == dtype]
    plt.plot(df_type['I'], df_type['total_spikes'], label=dtype)
plt.title("Número total de spikes vs Corriente")
plt.xlabel("Corriente de entrada I (nA)")
plt.ylabel("Total de spikes en 100 ms")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("spike_totals_vs_current.png", dpi=300)
plt.show()

# Visualización: RMSE de spike trains vs corriente
plt.figure(figsize=(10, 6))
for dtype in types:
    if dtype != "float64":
        df_type = df[df['type'] == dtype]
        plt.plot(df_type['I'], df_type['rmse_spike'], label=dtype)
plt.title("RMSE entre spike trains vs Corriente")
plt.xlabel("Corriente de entrada I (nA)")
plt.ylabel("RMSE con respecto a float64")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("rmse_spike_vs_current.png", dpi=300)
plt.show()

# Visualización: Firing rate vs corriente
plt.figure(figsize=(10, 6))
for dtype in types:
    df_type = df[df['type'] == dtype]
    plt.plot(df_type['I'], df_type['firing_rate'], label=dtype)
plt.title("Firing rate (Hz) vs Corriente")
plt.xlabel("Corriente de entrada I (nA)")
plt.ylabel("Firing rate (Hz)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("firing_rate_vs_current.png", dpi=300)
plt.show()

# Visualización: Sparsity vs corriente
plt.figure(figsize=(10, 6))
for dtype in types:
    df_type = df[df['type'] == dtype]
    plt.plot(df_type['I'], df_type['sparsity'], label=dtype)
plt.title("Sparsity vs Corriente")
plt.xlabel("Corriente de entrada I (nA)")
plt.ylabel("Sparsity (1 - fracción de spikes)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("sparsity_vs_current.png", dpi=300)
plt.show()


# -----------------------------
# 🔢 Imprimir tabla resumen por tipo
# -----------------------------
print("\nResumen completo por tipo:")
print(f"{'Tipo':<10} | {'Corriente (nA)':<15} | {'Firing Rate (Hz)':<17} | {'Sparsity':<10} | {'RMSE Spike':<12}")
print("-" * 70)

for dtype in types:
    df_type = df[df['type'] == dtype]
    for _, row in df_type.iterrows():
        print(f"{dtype:<10} | {row['I']:<15.3f} | {row['firing_rate']:<17.3f} | {row['sparsity']:<10.3f} | {row['rmse_spike']:<12.6f}")
