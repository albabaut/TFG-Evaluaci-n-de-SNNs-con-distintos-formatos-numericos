import posit_wrapper
from brian2 import *
import numpy as np



def run_simulation(data_type, I_override=None):
    if data_type == "posit":
        conv = lambda x: posit_wrapper.convert(x)
    elif data_type == "float16":
        conv = lambda x: np.float16(x)
    elif data_type == "float32":
        conv = lambda x: np.float32(x)
    else:
        conv = lambda x: np.float64(x)

  

    tau = 10 * ms
    V_rest = -70 *  mV
    V_reset = -60 *  mV
    V_threshold = -50 * mV
    I_input = conv(1e-7) * nA
    g_leak = 0.01 * nS


    print(f"{data_type}  : {I_input:.8e}")


    eqs = '''
    dv/dt = (-(V_rest - v) + I / g_leak) / tau : volt
    I : amp (shared)
    '''

    neuron = NeuronGroup(1, eqs, threshold='v > V_threshold', reset='v = V_reset', method='euler')
    neuron.v = V_rest
    neuron.I = I_input

    monitor = StateMonitor(neuron, 'v', record=True)
    spikemon = SpikeMonitor(neuron)

    net = Network(neuron, monitor, spikemon)
    net.run(100 * ms)

    return spikemon.num_spikes, monitor.v[0] / mV, spikemon.t[:] / ms  # <-- devolvemos tiempos



spikes_64, v64, t64 = run_simulation("float64")
spikes_32, v32, t32 = run_simulation("float32")
spikes_16, v16, t16 = run_simulation("float16")
spikes_p,  vp,  tp  = run_simulation("posit")


# Mostrar comparación de spikes
print("\nNúmero de spikes:")
print(f"Float64       : {spikes_64}")
print(f"Float32       : {spikes_32}")
print(f"Float16       : {spikes_16}")
print(f"Posit<16,2>   : {spikes_p}")

# Opcional: comparar RMSE total (solo si quieres)
from sklearn.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

min_len = min(len(v64), len(v32), len(v16), len(vp))
v64_clip = v64[:min_len]
v32_clip = v32[:min_len]
v16_clip = v16[:min_len]
vp_clip  = vp[:min_len]

# RMSE
rmse_32 = np.sqrt(mean_squared_error(v64_clip, v32_clip))
rmse_16 = np.sqrt(mean_squared_error(v64_clip, v16_clip))
rmse_p  = np.sqrt(mean_squared_error(v64_clip, vp_clip))

# PSNR
psnr_32 = psnr(v64_clip, v32_clip, data_range=v64_clip.max() - v64_clip.min())
psnr_16 = psnr(v64_clip, v16_clip, data_range=v64_clip.max() - v64_clip.min())
psnr_p  = psnr(v64_clip, vp_clip,  data_range=v64_clip.max() - v64_clip.min())

# SSIM (normalizar señales entre 0 y 1)
v64_norm = (v64_clip - v64_clip.min()) / (v64_clip.max() - v64_clip.min())
v32_norm = (v32_clip - v64_clip.min()) / (v64_clip.max() - v64_clip.min())
v16_norm = (v16_clip - v64_clip.min()) / (v64_clip.max() - v64_clip.min())
vp_norm  = (vp_clip  - v64_clip.min()) / (v64_clip.max() - v64_clip.min())

ssim_32 = ssim(v64_norm, v32_norm, data_range=1.0)
ssim_16 = ssim(v64_norm, v16_norm, data_range=1.0)
ssim_p  = ssim(v64_norm, vp_norm,  data_range=1.0)


print("\nRMSE respecto a float64:")
print(f"Float32       : {rmse_32:.20f}")
print(f"Float16       : {rmse_16:.20f}")
print(f"Posit<16,2>   : {rmse_p:.20f}")

print("\nPSNR respecto a float64:")
print(f"Float32       : {psnr_32:.20f} dB")
print(f"Float16       : {psnr_16:.20f} dB")
print(f"Posit<16,2>   : {psnr_p:.20f} dB")

print("\nSSIM respecto a float64:")
print(f"Float32       : {ssim_32:.20f}")
print(f"Float16       : {ssim_16:.20f}")
print(f"Posit<16,2>   : {ssim_p:.20f}")

