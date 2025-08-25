# tfg_spike_corr_metrics.py
"""
Métricas avanzadas para la red LIF:
  • Correlación de Pearson
  • Spike‑train RMSE (kernel exponencial 5 ms)
  • Entropía espectral

Script autónomo: replica `run_simulation`, no importa el otro archivo.
"""

import os
from typing import List, Dict

import posit_wrapper
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from brian2 import *  # noqa

# =============== CONFIG ==================================================
N_NEURONS = 10
SIM_TIME  = 100 * ms
I_VALS    = np.linspace(-2, 2.0,400)
SEED      = 42
SAVE_DIR  = os.path.join("figuras", "extra_metrics")
os.makedirs(SAVE_DIR, exist_ok=True)

PAIRS: Dict[str, List[str]] = {
    "16bits": ["float16", "posit16"],
    "24bits": ["float24", "posit20", "posit24"],
    "32bits": ["float32", "posit32"],
}

COLORS = {
    "float16": "tab:blue",  "posit16": "tab:orange",
    "float24": "tab:green", "posit20": "tab:red",
    "posit24": "tab:purple","float32": "tab:brown",
    "posit32": "tab:pink",
}

KERNEL_ALPHA = 5.0  # ms
np.random.seed(SEED)

# =============== CONVERSIÓN DE TIPOS ====================================

def _posit_convert(x, nbits: int, es: int = 2):
    try:
        return posit_wrapper.convert(x, nbits, es)
    except AttributeError:
        return getattr(posit_wrapper, f"convert{nbits}")(x)

def _float24_quantize(x):
    y = np.float32(x)
    ib = y.view(np.uint32)
    ib &= 0xFFFFFF80
    return ib.view(np.float32)

def to_dtype(x, kind: str):
    k = kind.lower()
    if k.startswith("posit"):
        return _posit_convert(x, int(k[5:]), 2)
    if k == "float16":
        return np.float16(x)
    if k == "float32":
        return np.float32(x)
    if k == "float24":
        return _float24_quantize(x)
    return np.float64(x)

# =============== FUNCIÓN DE SIMULACIÓN ==================================

def run_simulation(I_amp: float, kind: str):
    tau, V_rest, V_reset, V_th, g_leak = 10*ms, -70*mV, -60*mV, -50*mV, 1*nS
    I_input = to_dtype(I_amp, kind) * nA

    eqs = (
        "dv/dt = (-(V_rest - v) + I/g_leak) / tau : volt\n"
        "I : amp"
    )
    G = NeuronGroup(N_NEURONS, eqs, threshold="v>V_th", reset="v=V_reset", method="euler")
    G.v = V_rest; G.I = I_input

    S = Synapses(G, G, on_pre="v_post *= 1.2")
    S.connect(condition="i!=j")

    monV  = StateMonitor(G, "v", record=True)
    spMon = SpikeMonitor(G)
    net   = Network(G, S, monV, spMon)
    net.run(SIM_TIME)

    return monV.v/mV, monV.t/ms, spMon

# =============== MÉTRICAS AUXILIARES ====================================

def spike_train(times_ms: np.ndarray, spike_times, T: int):
    """Devuelve vector 0/1 discretizado en la cuadrícula times_ms."""
    if hasattr(spike_times, 'unit'):
        spike_times = (spike_times/ms).ravel()
    else:
        spike_times = np.asarray(spike_times).ravel()
    spike_times = spike_times[(spike_times >= times_ms[0]) & (spike_times <= times_ms[-1])]
    indices = np.searchsorted(times_ms, spike_times)
    vec = np.zeros(T, dtype=np.float32)
    vec[indices] = 1.0
    return vec

def filter_exp(train: np.ndarray, tau_ms: float):
    dt = float(defaultclock.dt/ms)
    alpha = np.exp(-dt / tau_ms)
    return lfilter([1-alpha], [1, -alpha], train)

# =============== EXPERIMENTO ===========================================
records = []
for pair_id, fmts in PAIRS.items():
    for I in I_VALS:
        v_ref, t_vec, spk_ref = run_simulation(I, "float64")
        T = len(t_vec)
        # trenes filtrados referencia
        ref_filt = np.stack([
            filter_exp(spike_train(t_vec, spk_ref.t[spk_ref.i==n], T), KERNEL_ALPHA)
            for n in range(N_NEURONS)
        ])
        for fmt in fmts:
            v_fmt, _, spk_fmt = run_simulation(I, fmt)
            # Pearson
            corr = float(np.corrcoef(v_ref.ravel(), v_fmt.ravel())[0,1])
            # Spike RMSE
            fmt_filt = np.stack([
                filter_exp(spike_train(t_vec, spk_fmt.t[spk_fmt.i==n], T), KERNEL_ALPHA)
                for n in range(N_NEURONS)
            ])
            spike_rmse = float(np.sqrt(np.mean((fmt_filt - ref_filt)**2)))
            # Entropía espectral
            fft = np.abs(np.fft.rfft(v_fmt - v_fmt.mean(axis=-1, keepdims=True), axis=-1))**2
            P = fft/fft.sum(axis=-1, keepdims=True)
            ent = float(-np.sum(P*np.log(P+1e-12), axis=-1).mean())
            records.append(dict(pair=pair_id, fmt=fmt, I=I,
                                pearson=corr, spike_rmse=spike_rmse,
                                spec_entropy=ent))

# =============== GUARDAR Y GRAFICAR ====================================

df = pd.DataFrame(records)
df.to_csv(os.path.join(SAVE_DIR, "spike_corr_metrics.csv"), index=False)

for metric, ylabel in [("pearson", "Correlación de Pearson"),
                       ("spike_rmse", "Spike‑train RMSE"),
                       ("spec_entropy", "Entropía espectral")]:
    for pair_id, fmts in PAIRS.items():
        plt.figure(figsize=(10,6))
        for fmt in fmts:
            sub = df[df.fmt == fmt]
            plt.plot(sub.I, sub[metric], marker="o", label=fmt,
                     color=COLORS.get(fmt))
        if metric == "spike_rmse":
            plt.yscale("log")
        plt.xlabel("Corriente I (nA)")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} – {pair_id}")
        plt.legend(); plt.grid(True, ls=":", lw=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f"{metric}_{pair_id}.png"), dpi=300)
        plt.close()

print(f"Métricas extra guardadas en {SAVE_DIR} (CSV + PNGs)")
