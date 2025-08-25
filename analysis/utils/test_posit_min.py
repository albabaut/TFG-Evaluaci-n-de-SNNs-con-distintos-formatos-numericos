import numpy as np
from brian2 import *
from sklearn.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import posit_wrapper
import matplotlib.pyplot as plt

defaultclock.dt = 0.01 * ms  # Precisión temporal alta

def run_simulation(data_type, I_val):
    if data_type == "posit":
        conv = lambda x: posit_wrapper.convert(x)
    elif data_type == "float16":
        conv = lambda x: np.float16(x)
    elif data_type == "float32":
        conv = lambda x: np.float32(x)
    else:
        conv = lambda x: np.float64(x)

    scale = 0.01
    tau = 10 * ms
    g_leak = 0.01 * nS
    V_rest = -0.70 * scale * volt
    V_reset = -0.60 * scale * volt
    V_threshold = -0.50 * scale * volt

    I_input = conv(I_val) * amp

    eqs = '''
    dv/dt = (-(V_rest - v) + I / g_leak) / tau : volt
    I : amp (shared)
    '''

    neuron = NeuronGroup(1, eqs, threshold='v > V_threshold', reset='v = V_reset',
                         method='euler', namespace=locals())
    neuron.v = V_rest
    neuron.I = I_input

    monitor = StateMonitor(neuron, 'v', record=True)
    net = Network(neuron, monitor)
    net.run(100 * ms)

    return monitor.v[0] / mV  # sin unidades, para comparación

# ----------- Barrido de valores de I -----------
I_values = np.logspace(-14, -9, 20)
errors = {'I': [], 'RMSE_16': [], 'RMSE_32': [], 'RMSE_p': [],
          'SSIM_16': [], 'SSIM_32': [], 'SSIM_p': [],
          'PSNR_16': [], 'PSNR_32': [], 'PSNR_p': []}

print("Barriendo corriente I...")

for I_val in I_values:
    try:
        v64 = run_simulation("float64", I_val)
        v16 = run_simulation("float16", I_val)
        v32 = run_simulation("float32", I_val)
        vp  = run_simulation("posit", I_val)

        min_len = min(len(v64), len(v16), len(v32), len(vp))
        v64_clip = v64[:min_len]
        v16_clip = v16[:min_len]
        v32_clip = v32[:min_len]
        vp_clip  = vp[:min_len]

        rmse_16 = np.sqrt(mean_squared_error(v64_clip, v16_clip))
        rmse_32 = np.sqrt(mean_squared_error(v64_clip, v32_clip))
        rmse_p  = np.sqrt(mean_squared_error(v64_clip, vp_clip))

        psnr_16 = psnr(v64_clip, v16_clip, data_range=v64_clip.max() - v64_clip.min())
        psnr_32 = psnr(v64_clip, v32_clip, data_range=v64_clip.max() - v64_clip.min())
        psnr_p  = psnr(v64_clip, vp_clip,  data_range=v64_clip.max() - v64_clip.min())

        norm = lambda v: (v - v64_clip.min()) / (v64_clip.max() - v64_clip.min())
        ssim_16 = ssim(norm(v64_clip), norm(v16_clip), data_range=1.0)
        ssim_32 = ssim(norm(v64_clip), norm(v32_clip), data_range=1.0)
        ssim_p  = ssim(norm(v64_clip), norm(vp_clip),  data_range=1.0)

        errors['I'].append(I_val)
        errors['RMSE_16'].append(rmse_16)
        errors['RMSE_32'].append(rmse_32)
        errors['RMSE_p'].append(rmse_p)
        errors['PSNR_16'].append(psnr_16)
        errors['PSNR_32'].append(psnr_32)
        errors['PSNR_p'].append(psnr_p)
        errors['SSIM_16'].append(ssim_16)
        errors['SSIM_32'].append(ssim_32)
        errors['SSIM_p'].append(ssim_p)
        
        print(f"I={I_val:.1e} ✓")

    except Exception as e:
        print(f"I={I_val:.1e} ❌ Error: {e}")

# ----------- Graficar resultados -----------
plt.figure(figsize=(10, 5))
plt.semilogx(errors['I'], errors['RMSE_16'], label='float16')
plt.semilogx(errors['I'], errors['RMSE_32'], label='float32')
plt.semilogx(errors['I'], errors['RMSE_p'], label='posit<16,1>')
plt.ylabel("RMSE vs float64")
plt.xlabel("Corriente I (A)")
plt.title("RMSE en función de I")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("rmse_vs_I.png")

plt.figure(figsize=(10, 5))
plt.semilogx(errors['I'], errors['SSIM_16'], label='float16')
plt.semilogx(errors['I'], errors['SSIM_32'], label='float32')
plt.semilogx(errors['I'], errors['SSIM_p'], label='posit<16,1>')
plt.ylabel("SSIM vs float64")
plt.xlabel("Corriente I (A)")
plt.title("SSIM en función de I")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("ssim_vs_I.png")

plt.figure(figsize=(10, 5))
plt.semilogx(errors['I'], errors['PSNR_16'], label='float16')
plt.semilogx(errors['I'], errors['PSNR_32'], label='float32')
plt.semilogx(errors['I'], errors['PSNR_p'], label='posit<16,1>')
plt.ylabel("PSNR vs float64 (dB)")
plt.xlabel("Corriente I (A)")
plt.title("PSNR en función de I")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("psnr_vs_I.png")

print("\n✅ ¡Listo! Se generaron las gráficas:")
print(" - rmse_vs_I.png")
print(" - ssim_vs_I.png")
print(" - psnr_vs_I.png")
