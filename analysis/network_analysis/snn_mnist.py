# snn_mnist_precision.py
"""
Extiende el experimento MNIST-STDP para evaluar el impacto de precisión reducida:
- float16, float32, float64
- posit16, posit24 (si está disponible en posit_wrapper)

Se entrena una SNN con STDP sobre MNIST (clases 0, 1, 8), genera features (spike-counts)
y evalúa precisión de RandomForest. Repite el mismo proceso con cada tipo de dato.
"""

import numpy as np
from keras.datasets import mnist
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from brian2 import *
import posit_wrapper  # wrapper tuyo para posit<16>, <24>, etc.

np.random.seed(0)

# Cargar datos y filtrar clases
(X_train, y_train), (X_test, y_test) = mnist.load_data()
keep = lambda y: (y == 0) | (y == 1) | (y == 8)
X_train, y_train = X_train[keep(y_train)], y_train[keep(y_train)]
X_test, y_test = X_test[keep(y_test)], y_test[keep(y_test)]
X_train = X_train / 4  # rate encoding (Hz)
X_test = X_test / 4

n_input = 28 * 28
n_e = 100
n_i = 100

# Parámetros del modelo
v_rest_e = -60.*mV
v_reset_e = -65.*mV
v_thresh_e = -52.*mV
v_rest_i = -60.*mV
v_reset_i = -45.*mV
v_thresh_i = -40.*mV

# STDP
taupre = 20*ms
taupost = 20*ms
gmax = 0.05
dApre = 0.01 * gmax
dApost = -dApre * taupre / taupost * 1.05

stdp = '''w : 1
          lr : 1 (shared)
          dApre/dt = -Apre / taupre : 1 (event-driven)
          dApost/dt = -Apost / taupost : 1 (event-driven)'''
pre = '''ge += w
        Apre += dApre
        w = clip(w + lr*Apost, 0, gmax)'''
post = '''Apost += dApost
         w = clip(w + lr*Apre, 0, gmax)'''

# Función para conversión según tipo

def to_dtype(x, kind):
    if kind == 'float16': return np.float16(x)
    if kind == 'float32': return np.float32(x)
    if kind == 'float64': return np.float64(x)
    if kind == 'posit16': return np.array([posit_wrapper.convert16(float(xi)) for xi in x])
    if kind == 'posit24': return np.array([posit_wrapper.convert20(float(xi)) for xi in x])
    raise ValueError(f"Formato desconocido: {kind}")

# Modelo SNN (adaptado)

def train_eval_model(X, y, Xtest, ytest, dtype="float32", n_train=1000, n_eval=500):
    defaultclock.dt = 0.1*ms
    net = Network()

    PG = PoissonGroup(n_input, rates=np.zeros(n_input)*Hz)

    rates_array = X[:n_train]
    

    EG = NeuronGroup(n_e, '''
        dv/dt = (ge*(0*mV-v) + gi*(-100*mV-v) + (v_rest_e-v)) / (100*ms) : volt
        dge/dt = -ge / (5*ms) : 1
        dgi/dt = -gi / (10*ms) : 1
        ''', threshold='v>v_thresh_e', reset='v=v_reset_e', method='euler')
    EG.v = v_rest_e - 20*mV

    IG = NeuronGroup(n_i, '''
        dv/dt = (ge*(0*mV-v) + (v_rest_i-v)) / (10*ms) : volt
        dge/dt = -ge / (5*ms) : 1
        ''', threshold='v>v_thresh_i', reset='v=v_reset_i', method='euler')
    IG.v = v_rest_i - 20*mV

    S1 = Synapses(PG, EG, model=stdp, on_pre=pre, on_post=post, method='euler')
    S1.connect()
    S1.w = 'rand()*gmax'
    S1.lr = 1

    S2 = Synapses(EG, IG, 'w : 1', on_pre='ge += w')
    S2.connect(j='i')
    S2.w = 3

    S3 = Synapses(IG, EG, 'w : 1', on_pre='gi += w')
    S3.connect(condition='i!=j')
    S3.w = 0.03

    net.add([PG, EG, IG, S1, S2, S3])

    # ENTRENAMIENTO
    net.run(n_train * 0.5 * second)  # 0.35 + 0.15 por muestra
    S1.lr = 0  # freeze STDP

    # EVALUACIÓN
    features = []
    for i in range(n_eval):
        spike_mon = SpikeMonitor(EG)
        net.add(spike_mon)

     
        PG.rates = to_dtype(X[i].ravel(), dtype) * Hz
        net.run(0.35*second)
        PG.rates = np.zeros(n_input)*Hz
        net.run(0.15*second)

        features.append(np.array(spike_mon.count).astype(int))

        net.remove(spike_mon)

    clf = RandomForestClassifier(max_depth=4, random_state=0)
    clf.fit(features, ytest[:n_eval])
    pred = clf.predict(features)
    acc = accuracy_score(pred, ytest[:n_eval])
    print(f"[{dtype}] Accuracy (n={n_eval}):", acc)
    return acc

# Ejecutar para distintos formatos
for fmt in ["float16", "float32", "float64", "posit16", "posit24"]:
    acc = train_eval_model(X_train, y_train, X_test, y_test, dtype=fmt, n_train=500, n_eval=300)
