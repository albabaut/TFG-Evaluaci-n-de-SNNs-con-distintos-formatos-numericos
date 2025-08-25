import posit_wrapper
from brian2 import *
import numpy as np
from sklearn.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import pandas as pd

# Función para convertir un valor a bfloat16 truncando 16 bits LSB
def to_bfloat16(x):
    x32 = np.array(x, dtype=np.float32)
    bits = x32.view(np.uint32)
    new_bits = (bits >> 16) << 16
    result = new_bits.view(np.float32)
    return float(result)

# Nuevas métricas
def relative_error(ref, test, epsilon=1e-8):
    return np.abs((ref - test) / (np.abs(ref) + epsilon))

def spectral_entropy(signal, epsilon=1e-8):
    spectrum = np.abs(np.fft.fft(signal))**2
    P = spectrum / np.sum(spectrum)
    return -np.sum(P * np.log(P + epsilon))


# Simulación
def run_simulation(data_type, I_override=None):
    if data_type == "posit16":
        conv = lambda x: posit_wrapper.convert16(x)
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

    monitor = StateMonitor(neuron, 'v', record=True)
    spikemon = SpikeMonitor(neuron)

    net = Network(neuron, monitor, spikemon)
    net.run(100 * ms)

    return spikemon.num_spikes, monitor.v[0] / mV, spikemon.t[:] / ms

# Barrido
currents = np.linspace(0.2, 1, 500)
results = []


for I_val in currents:
    spikes_ref, v_ref, _ = run_simulation("float64", I_override=I_val)
    spikes_p, v_p, _ = run_simulation("posit16", I_override=I_val)
    spikes_f16, v_f16, _ = run_simulation("float16", I_override=I_val)
    spikes_f32, v_f32, _ = run_simulation("float32", I_override=I_val)
    spikes_bf16, v_bf16, _ = run_simulation("bfloat16", I_override=I_val)

    min_len = min(len(v_ref), len(v_p), len(v_f16), len(v_f32), len(v_bf16))
    v_ref_clip = v_ref[:min_len]
    v_p_clip = v_p[:min_len]
    v_f16_clip = v_f16[:min_len]
    v_f32_clip = v_f32[:min_len]
    v_bf16_clip = v_bf16[:min_len]

    range_v = v_ref_clip.max() - v_ref_clip.min() if v_ref_clip.max() != v_ref_clip.min() else 1.0
    v_norm = lambda x: (x - v_ref_clip.min()) / range_v

    results.append({
        'I': I_val,
        'rmse_f32': np.sqrt(mean_squared_error(v_ref_clip, v_f32_clip)),
        'rmse_f16': np.sqrt(mean_squared_error(v_ref_clip, v_f16_clip)),
        'rmse_posit': np.sqrt(mean_squared_error(v_ref_clip, v_p_clip)),
        'rmse_bf16': np.sqrt(mean_squared_error(v_ref_clip, v_bf16_clip)),

        'psnr_f32': psnr(v_ref_clip, v_f32_clip, data_range=range_v),
        'psnr_f16': psnr(v_ref_clip, v_f16_clip, data_range=range_v),
        'psnr_posit': psnr(v_ref_clip, v_p_clip, data_range=range_v),
        'psnr_bf16': psnr(v_ref_clip, v_bf16_clip, data_range=range_v),

        'ssim_f32': ssim(v_norm(v_ref_clip), v_norm(v_f32_clip), data_range=1.0),
        'ssim_f16': ssim(v_norm(v_ref_clip), v_norm(v_f16_clip), data_range=1.0),
        'ssim_posit': ssim(v_norm(v_ref_clip), v_norm(v_p_clip), data_range=1.0),
        'ssim_bf16': ssim(v_norm(v_ref_clip), v_norm(v_bf16_clip), data_range=1.0),

        'relerr_f16': np.mean(relative_error(v_ref_clip, v_f16_clip)),
        'relerr_bf16': np.mean(relative_error(v_ref_clip, v_bf16_clip)),
        'relerr_posit': np.mean(relative_error(v_ref_clip, v_p_clip)),

        'corr_f16': pearsonr(v_ref_clip, v_f16_clip)[0],
        'corr_bf16': pearsonr(v_ref_clip, v_bf16_clip)[0],
        'corr_posit': pearsonr(v_ref_clip, v_p_clip)[0],

        'specent_f16': spectral_entropy(v_f16_clip),
        'specent_bf16': spectral_entropy(v_bf16_clip),
        'specent_posit': spectral_entropy(v_p_clip)
    })

# Guardar y graficar RMSE vs I
df = pd.DataFrame(results)

plt.figure(figsize=(10, 6))
plt.plot(df['I'], df['rmse_f16'], label='float16')
plt.plot(df['I'], df['rmse_bf16'], label='bfloat16')
plt.plot(df['I'], df['rmse_posit'], label='posit<16,2>')
plt.xlabel("Corriente de entrada I (nA)")
plt.ylabel("RMSE respecto a float64 (voltaje)")
plt.title("RMSE vs Corriente para diferentes tipos de dato")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("rmse_vs_I_extended.png", dpi=300)
plt.show()
# Crear figura con 3 subgráficas
fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# --- 1. Error relativo medio ---
axs[0].plot(df['I'], df['relerr_f16'], label='float16')
axs[0].plot(df['I'], df['relerr_bf16'], label='bfloat16')
axs[0].plot(df['I'], df['relerr_posit'], label='posit<16,2>')
axs[0].set_ylabel("Error relativo medio")
axs[0].set_title("Error relativo medio vs Corriente de entrada (I)")
axs[0].legend()
axs[0].grid(True)

# --- 2. Correlación de Pearson ---
axs[1].plot(df['I'], df['corr_f16'], label='float16')
axs[1].plot(df['I'], df['corr_bf16'], label='bfloat16')
axs[1].plot(df['I'], df['corr_posit'], label='posit<16,2>')
axs[1].set_ylabel("Correlación de Pearson")
axs[1].set_title("Correlación de forma de la señal vs Corriente de entrada (I)")
axs[1].legend()
axs[1].grid(True)

# --- 3. Entropía espectral ---
axs[2].plot(df['I'], df['specent_f16'], label='float16')
axs[2].plot(df['I'], df['specent_bf16'], label='bfloat16')
axs[2].plot(df['I'], df['specent_posit'], label='posit<16,2>')
axs[2].set_ylabel("Entropía espectral")
axs[2].set_xlabel("Corriente de entrada I (nA)")
axs[2].set_title("Entropía espectral vs Corriente de entrada (I)")
axs[2].legend()
axs[2].grid(True)

# Ajustes finales y guardado
plt.tight_layout()
plt.savefig("comparacion_metricas_extended.png", dpi=300)
plt.show()

# Guardar el DataFrame con todas las métricas en un archivo CSV
df.to_csv("resultados_metricas_tipos_dato.csv", index=False)
print("✅ Resultados exportados a 'resultados_metricas_tipos_dato.csv'")
