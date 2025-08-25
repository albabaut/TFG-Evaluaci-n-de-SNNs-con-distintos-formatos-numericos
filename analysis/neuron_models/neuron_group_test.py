# tfg_precision_plausible.py
"""
Red LIF *biológicamente plausible* ‑> disparos ≤200 Hz
-----------------------------------------------------------------
* 80 % neuronas excitadoras (E), 20 % inhibitorias (I)
* Peso excitador **+0.2 mV**, peso inhibidor **−0.4 mV**
* Conectividad aleatoria 20 % (sin auto‑sinapsis)
* Refractario absoluto: 2 ms
* Corrientes de entrada I ∈ [0.1 … 1.0] nA (25 pasos)

Se comparan los mismos formatos que antes y se obtienen:
  • RMSE del potencial de membrana (eje log)
  • Firing‑rate medio (Hz)

Las figuras se guardan en `figuras/plausible/`.
"""

import os
from typing import List, Dict

import posit_wrapper
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from brian2 import *  # noqa

# ------------- CONFIGURACIÓN DE LA RED -----------------------------------
N_NEURONS = 100                 # 80E + 20I
FRAC_INH  = 0.2
NE          = int(N_NEURONS * (1 - FRAC_INH))
NI          = N_NEURONS - NE
SIM_TIME    = 500 * ms          # ventana más larga para tasas bajas
I_VALS      = np.linspace(-2, 2, 400)
SEED        = 42
SAVE_DIR    = os.path.join("figuras", "plausible")
os.makedirs(SAVE_DIR, exist_ok=True)

# Pesos
W_E = 0.2 * mV
W_I = -0.4 * mV
P_CONN = 0.2
REFRACT = 2 * ms

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

np.random.seed(SEED)

# ------------- Conversión de formatos ------------------------------------

def _posit_convert(x, nbits: int, es: int = 2):
    try:
        return posit_wrapper.convert(x, nbits, es)
    except AttributeError:
        return getattr(posit_wrapper, f"convert{nbits}")(x)

def _float24_quantize(x):
    y = np.float32(x)
    ib = y.view(np.uint32);
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

# ------------- Función de simulación -------------------------------------

def run_simulation(I_amp: float, kind: str):
    tau, V_rest, V_reset, V_th, g_leak = 10*ms, -70*mV, -60*mV, -50*mV, 1*nS

    I_input = to_dtype(I_amp, kind) * nA

    eqs = (
        "dv/dt = (-(V_rest - v) + I/g_leak) / tau : volt (unless refractory)\n"
        "I : amp"
    )

    G = NeuronGroup(N_NEURONS, eqs, threshold="v > V_th", reset="v = V_reset",
                    method="euler", refractory=REFRACT)
    G.v = V_rest
    G.I = I_input

    # Etiquetas E/I
    E_idx = np.arange(NE)
    I_idx = np.arange(NE, N_NEURONS)

    # sinapsis excitadora (E → todos)
    SE = Synapses(G, G, on_pre="v_post += 0.2*mV")
    SE.connect(condition="i < NE and j != i", p=P_CONN)

    # sinapsis inhibidora (I → todos)
    SI = Synapses(G, G, on_pre="v_post -= 0.4*mV")   # resta 0.4 mV
    SI.connect(condition="i >= NE and j != i", p=P_CONN)


    monV  = StateMonitor(G, "v", record=True)
    spMon = SpikeMonitor(G)

    net = Network(G, SE, SI, monV, spMon)
    net.run(SIM_TIME)

    fr = spMon.count / (SIM_TIME / second)  # Hz por neurona
    return monV.v/mV, monV.t/ms, fr

# ------------- Experimento principal -------------------------------------
records = []
for pair_id, fmts in PAIRS.items():
    for I in I_VALS:
        v_ref, _, fr_ref = run_simulation(I, "float64")
        ref_len = v_ref.shape[1]
        for fmt in fmts:
            v_fmt, _, fr_fmt = run_simulation(I, fmt)
            minT = min(ref_len, v_fmt.shape[1])
            rmse = np.sqrt(np.mean((v_fmt[:, :minT]-v_ref[:, :minT])**2))
            rate = float(np.mean(fr_fmt))
            records.append(dict(pair=pair_id, fmt=fmt, I=I, rmse=rmse, rate_Hz=rate))

# ------------- Guardar y graficar ---------------------------------------

df = pd.DataFrame(records)
df.to_csv(os.path.join(SAVE_DIR, "plausible_metrics.csv"), index=False)

plots = [
    ("rmse", "RMSE (mV)", "rmse", True),
    ("rate_Hz", "Tasa de disparo (Hz)", "rate", False)
]

for col, ylabel, tag, log in plots:
    for pair_id, fmts in PAIRS.items():
        plt.figure(figsize=(10,6))
        for fmt in fmts:
            sub = df[df.fmt==fmt]
            plt.plot(sub.I, sub[col], marker="o", label=fmt,
                     color=COLORS.get(fmt))
        if log:
            plt.yscale("log")
        plt.xlabel("Corriente I (nA)")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} – {pair_id} (red plausible)")
        plt.legend(); plt.grid(True, ls=":", lw=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f"plaus_{tag}_{pair_id}.png"), dpi=300)
        plt.close()

print(f"Figuras y CSV guardados en ➜ {SAVE_DIR}/")
