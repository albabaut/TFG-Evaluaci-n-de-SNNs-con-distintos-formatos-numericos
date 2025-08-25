import brian2 as b2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import logging
from posit_wrapper import convert16
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stress_test_cases.log'),
        logging.StreamHandler()
    ]
)

class StressTestCases:
    def __init__(self, dt=0.1*b2.ms):
        """
        Inicializa las pruebas de estrés para casos límite
        
        Args:
            dt: Paso de tiempo
        """
        self.dt = dt
        
        # Parámetros de la neurona LIF
        self.tau = 10 * b2.ms
        self.V_rest = -70 * b2.mV
        self.Vth = -50 * b2.mV
        self.Vreset = -70 * b2.mV
        
        # Configurar Brian2
        b2.prefs.codegen.target = 'numpy'
        
        logging.info(f"StressTestCases inicializado")
    
    def test_threshold_current(self, I_ext_range, precision_types=['float64', 'float32', 'float16', 'posit16']):
        """
        Prueba corrientes justo en el umbral de disparo
        
        Args:
            I_ext_range: Rango de corrientes a probar
            precision_types: Tipos de precisión a comparar
        """
        logging.info("=== PRUEBA: CORRIENTE EN EL UMBRAL ===")
        
        results = []
        
        for I_ext in I_ext_range:
            logging.info(f"Probando corriente: {I_ext}")
            
            for precision in precision_types:
                # Limpiar estado
                b2.device.reinit()
                b2.device.activate()
                
                # Ecuaciones
                eqs = '''
                dv/dt = (-(V_rest - v) + I_ext) / tau : volt (unless refractory)
                '''
                
                # Crear grupo neuronal
                if precision == 'float64':
                    dtype = np.float64
                elif precision == 'float32':
                    dtype = np.float32
                elif precision == 'float16':
                    dtype = np.float16
                elif precision == 'posit16':
                    dtype = np.float32
                
                G = b2.NeuronGroup(1, eqs, threshold='v > Vth', reset='v = Vreset',
                                  refractory=5*b2.ms, method='euler', dtype=dtype)
                G.v = self.V_rest
                
                # Monitores
                mon = b2.StateMonitor(G, 'v', record=True)
                spk = b2.SpikeMonitor(G)
                
                # Network
                net = b2.Network(G, mon, spk)
                
                # Ejecutar simulación
                simulation_time = 1000 * b2.ms
                namespace = {
                    'I_ext': I_ext, 'tau': self.tau, 'V_rest': self.V_rest, 
                    'Vth': self.Vth, 'Vreset': self.Vreset
                }
                
                net.run(simulation_time, namespace=namespace)
                
                # Analizar resultados
                voltage = mon.v[0] / b2.mV
                time = mon.t / b2.ms
                spikes = spk.t / b2.ms
                
                # Calcular métricas
                max_voltage = np.max(voltage)
                min_voltage = np.min(voltage)
                voltage_range = max_voltage - min_voltage
                n_spikes = len(spikes)
                
                # Detectar si la neurona alcanzó el umbral
                reached_threshold = max_voltage >= float(self.Vth / b2.mV)
                
                result = {
                    'I_ext': float(I_ext / b2.mV),
                    'precision': precision,
                    'n_spikes': n_spikes,
                    'max_voltage': max_voltage,
                    'min_voltage': min_voltage,
                    'voltage_range': voltage_range,
                    'reached_threshold': reached_threshold,
                    'first_spike_time': spikes[0] if len(spikes) > 0 else np.nan
                }
                
                results.append(result)
                
                logging.info(f"  {precision}: {n_spikes} spikes, max_v={max_voltage:.3f}mV, threshold={reached_threshold}")
        
        # Crear DataFrame y guardar
        df = pd.DataFrame(results)
        os.makedirs('results/stress_tests', exist_ok=True)
        df.to_csv('results/stress_tests/threshold_current_test.csv', index=False)
        
        # Visualizar resultados
        self.plot_threshold_test_results(df)
        
        return df
    
    def test_noise_amplification(self, I_ext, noise_levels, precision_types=['float64', 'float16']):
        """
        Prueba la amplificación de diferencias por ruido aleatorio
        
        Args:
            I_ext: Corriente externa
            noise_levels: Lista de niveles de ruido
            precision_types: Tipos de precisión a comparar
        """
        logging.info("=== PRUEBA: AMPLIFICACIÓN POR RUIDO ===")
        
        results = []
        
        for noise_std in noise_levels:
            logging.info(f"Probando ruido: {noise_std}")
            
            for precision in precision_types:
                for seed in range(5):  # 5 semillas diferentes
                    # Limpiar estado
                    b2.device.reinit()
                    b2.device.activate()
                    
                    # Configurar semilla
                    np.random.seed(seed)
                    
                    # Ecuaciones con ruido
                    eqs = '''
                    dv/dt = (-(V_rest - v) + I_ext) / tau + xi * noise_std / sqrt(tau) : volt (unless refractory)
                    '''
                    
                    # Crear grupo neuronal
                    if precision == 'float64':
                        dtype = np.float64
                    elif precision == 'float16':
                        dtype = np.float16
                    
                    G = b2.NeuronGroup(1, eqs, threshold='v > Vth', reset='v = Vreset',
                                      refractory=5*b2.ms, method='euler', dtype=dtype)
                    G.v = self.V_rest
                    
                    # Monitores
                    mon = b2.StateMonitor(G, 'v', record=True)
                    spk = b2.SpikeMonitor(G)
                    
                    # Network
                    net = b2.Network(G, mon, spk)
                    
                    # Ejecutar simulación
                    simulation_time = 1000 * b2.ms
                    namespace = {
                        'I_ext': I_ext, 'tau': self.tau, 'V_rest': self.V_rest, 
                        'Vth': self.Vth, 'Vreset': self.Vreset, 'noise_std': noise_std
                    }
                    
                    net.run(simulation_time, namespace=namespace)
                    
                    # Analizar resultados
                    voltage = mon.v[0] / b2.mV
                    spikes = spk.t / b2.ms
                    
                    # Calcular métricas
                    n_spikes = len(spikes)
                    voltage_std = np.std(voltage)
                    voltage_mean = np.mean(voltage)
                    
                    # Calcular variabilidad de intervalos inter-spike
                    if len(spikes) > 1:
                        isi = np.diff(spikes)
                        isi_cv = np.std(isi) / np.mean(isi)  # Coeficiente de variación
                    else:
                        isi_cv = np.nan
                    
                    result = {
                        'noise_std': noise_std,
                        'precision': precision,
                        'seed': seed,
                        'n_spikes': n_spikes,
                        'voltage_std': voltage_std,
                        'voltage_mean': voltage_mean,
                        'isi_cv': isi_cv
                    }
                    
                    results.append(result)
                    
                    logging.info(f"  {precision} (seed {seed}): {n_spikes} spikes, voltage_std={voltage_std:.3f}mV")
        
        # Crear DataFrame y guardar
        df = pd.DataFrame(results)
        df.to_csv('results/stress_tests/noise_amplification_test.csv', index=False)
        
        # Visualizar resultados
        self.plot_noise_test_results(df)
        
        return df
    
    def test_long_simulation(self, I_ext, simulation_time=100*b2.second, precision_types=['float64', 'float16']):
        """
        Prueba simulaciones largas para detectar acumulación de error
        
        Args:
            I_ext: Corriente externa
            simulation_time: Tiempo de simulación
            precision_types: Tipos de precisión a comparar
        """
        logging.info("=== PRUEBA: SIMULACIÓN LARGA ===")
        logging.info(f"Tiempo de simulación: {simulation_time}")
        
        results = []
        
        for precision in precision_types:
            logging.info(f"Ejecutando simulación larga con {precision}")
            
            # Limpiar estado
            b2.device.reinit()
            b2.device.activate()
            
            # Ecuaciones
            eqs = '''
            dv/dt = (-(V_rest - v) + I_ext) / tau : volt (unless refractory)
            '''
            
            # Crear grupo neuronal
            if precision == 'float64':
                dtype = np.float64
            elif precision == 'float16':
                dtype = np.float16
            
            G = b2.NeuronGroup(1, eqs, threshold='v > Vth', reset='v = Vreset',
                              refractory=5*b2.ms, method='euler', dtype=dtype)
            G.v = self.V_rest
            
            # Monitores con muestreo reducido para ahorrar memoria
            mon = b2.StateMonitor(G, 'v', record=True, dt=1*b2.ms)
            spk = b2.SpikeMonitor(G)
            
            # Network
            net = b2.Network(G, mon, spk)
            
            # Ejecutar simulación
            namespace = {
                'I_ext': I_ext, 'tau': self.tau, 'V_rest': self.V_rest, 
                'Vth': self.Vth, 'Vreset': self.Vreset
            }
            
            net.run(simulation_time, namespace=namespace)
            
            # Analizar resultados
            voltage = mon.v[0] / b2.mV
            time = mon.t / b2.ms
            spikes = spk.t / b2.ms
            
            # Calcular métricas
            n_spikes = len(spikes)
            total_time = float(simulation_time / b2.second)
            firing_rate = n_spikes / total_time
            
            # Analizar estabilidad temporal
            if len(spikes) > 10:
                # Dividir en ventanas de tiempo
                n_windows = 10
                window_size = total_time / n_windows
                rates_per_window = []
                
                for i in range(n_windows):
                    start_time = i * window_size
                    end_time = (i + 1) * window_size
                    spikes_in_window = spikes[(spikes >= start_time * 1000) & (spikes < end_time * 1000)]
                    rate = len(spikes_in_window) / window_size
                    rates_per_window.append(rate)
                
                rate_stability = np.std(rates_per_window)
            else:
                rate_stability = np.nan
            
            # Detectar anomalías
            voltage_range = np.max(voltage) - np.min(voltage)
            voltage_anomaly = voltage_range > 100  # Voltaje fuera de rango normal
            
            result = {
                'precision': precision,
                'simulation_time_s': total_time,
                'n_spikes': n_spikes,
                'firing_rate_hz': firing_rate,
                'rate_stability': rate_stability,
                'voltage_range': voltage_range,
                'voltage_anomaly': voltage_anomaly,
                'max_voltage': np.max(voltage),
                'min_voltage': np.min(voltage)
            }
            
            results.append(result)
            
            logging.info(f"  {precision}: {n_spikes} spikes, tasa {firing_rate:.2f} Hz, estabilidad {rate_stability:.3f}")
        
        # Crear DataFrame y guardar
        df = pd.DataFrame(results)
        df.to_csv('results/stress_tests/long_simulation_test.csv', index=False)
        
        # Visualizar resultados
        self.plot_long_simulation_results(df)
        
        return df
    
    def test_random_initial_conditions(self, I_ext, n_neurons=10, precision_types=['float64', 'float16']):
        """
        Prueba con condiciones iniciales aleatorias
        
        Args:
            I_ext: Corriente externa
            n_neurons: Número de neuronas
            precision_types: Tipos de precisión a comparar
        """
        logging.info("=== PRUEBA: CONDICIONES INICIALES ALEATORIAS ===")
        
        results = []
        
        for precision in precision_types:
            logging.info(f"Probando {precision} con {n_neurons} neuronas")
            
            # Limpiar estado
            b2.device.reinit()
            b2.device.activate()
            
            # Ecuaciones
            eqs = '''
            dv/dt = (-(V_rest - v) + I_ext) / tau : volt (unless refractory)
            '''
            
            # Crear grupo neuronal
            if precision == 'float64':
                dtype = np.float64
            elif precision == 'float16':
                dtype = np.float16
            
            G = b2.NeuronGroup(n_neurons, eqs, threshold='v > Vth', reset='v = Vreset',
                              refractory=5*b2.ms, method='euler', dtype=dtype)
            
            # Condiciones iniciales aleatorias
            np.random.seed(42)  # Semilla fija para reproducibilidad
            initial_voltages = np.random.uniform(-75, -65, n_neurons) * b2.mV
            G.v = initial_voltages
            
            # Monitores
            mon = b2.StateMonitor(G, 'v', record=True)
            spk = b2.SpikeMonitor(G)
            
            # Network
            net = b2.Network(G, mon, spk)
            
            # Ejecutar simulación
            simulation_time = 1000 * b2.ms
            namespace = {
                'I_ext': I_ext, 'tau': self.tau, 'V_rest': self.V_rest, 
                'Vth': self.Vth, 'Vreset': self.Vreset
            }
            
            net.run(simulation_time, namespace=namespace)
            
            # Analizar resultados por neurona
            for i in range(n_neurons):
                voltage = mon.v[i] / b2.mV
                neuron_spikes = spk.t[spk.i == i] / b2.ms
                
                # Calcular métricas
                n_spikes = len(neuron_spikes)
                firing_rate = n_spikes / (float(simulation_time / b2.second))
                voltage_std = np.std(voltage)
                
                # Calcular entropía de la secuencia de spikes
                if len(neuron_spikes) > 1:
                    # Crear señal binaria
                    time_bins = np.arange(0, float(simulation_time / b2.ms), 1.0)
                    spike_signal = np.zeros(len(time_bins))
                    
                    for spike_time in neuron_spikes:
                        bin_idx = int(spike_time)
                        if bin_idx < len(spike_signal):
                            spike_signal[bin_idx] = 1
                    
                    # Calcular entropía
                    if np.sum(spike_signal) > 0:
                        spike_entropy = entropy(np.bincount(spike_signal.astype(int)))
                    else:
                        spike_entropy = 0.0
                else:
                    spike_entropy = 0.0
                
                result = {
                    'precision': precision,
                    'neuron_id': i,
                    'initial_voltage': float(initial_voltages[i] / b2.mV),
                    'n_spikes': n_spikes,
                    'firing_rate_hz': firing_rate,
                    'voltage_std': voltage_std,
                    'spike_entropy': spike_entropy
                }
                
                results.append(result)
            
            # Calcular estadísticas globales
            neuron_spikes = [len(spk.t[spk.i == i]) for i in range(n_neurons)]
            total_spikes = sum(neuron_spikes)
            spike_variance = np.var(neuron_spikes)
            
            logging.info(f"  {precision}: {total_spikes} spikes totales, varianza {spike_variance:.2f}")
        
        # Crear DataFrame y guardar
        df = pd.DataFrame(results)
        df.to_csv('results/stress_tests/random_initial_conditions_test.csv', index=False)
        
        # Visualizar resultados
        self.plot_random_conditions_results(df)
        
        return df
    
    def plot_threshold_test_results(self, df):
        """Visualiza resultados de la prueba de corriente umbral"""
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Número de spikes vs corriente
        plt.subplot(2, 3, 1)
        for precision in df['precision'].unique():
            data = df[df['precision'] == precision]
            plt.plot(data['I_ext'], data['n_spikes'], 'o-', label=precision)
        plt.xlabel('Corriente externa (mV)')
        plt.ylabel('Número de spikes')
        plt.title('Spikes vs Corriente')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Voltaje máximo vs corriente
        plt.subplot(2, 3, 2)
        for precision in df['precision'].unique():
            data = df[df['precision'] == precision]
            plt.plot(data['I_ext'], data['max_voltage'], 'o-', label=precision)
        plt.axhline(y=float(self.Vth / b2.mV), color='r', linestyle='--', label='Umbral')
        plt.xlabel('Corriente externa (mV)')
        plt.ylabel('Voltaje máximo (mV)')
        plt.title('Voltaje Máximo vs Corriente')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Tiempo del primer spike
        plt.subplot(2, 3, 3)
        for precision in df['precision'].unique():
            data = df[df['precision'] == precision]
            data = data.dropna(subset=['first_spike_time'])
            if len(data) > 0:
                plt.plot(data['I_ext'], data['first_spike_time'], 'o-', label=precision)
        plt.xlabel('Corriente externa (mV)')
        plt.ylabel('Tiempo primer spike (ms)')
        plt.title('Latencia vs Corriente')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Rango de voltaje
        plt.subplot(2, 3, 4)
        for precision in df['precision'].unique():
            data = df[df['precision'] == precision]
            plt.plot(data['I_ext'], data['voltage_range'], 'o-', label=precision)
        plt.xlabel('Corriente externa (mV)')
        plt.ylabel('Rango de voltaje (mV)')
        plt.title('Rango de Voltaje vs Corriente')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 5: Proporción de neuronas que alcanzan umbral
        plt.subplot(2, 3, 5)
        threshold_data = []
        for I_ext in df['I_ext'].unique():
            for precision in df['precision'].unique():
                data = df[(df['I_ext'] == I_ext) & (df['precision'] == precision)]
                if len(data) > 0:
                    threshold_ratio = np.mean(data['reached_threshold'])
                    threshold_data.append({
                        'I_ext': I_ext,
                        'precision': precision,
                        'threshold_ratio': threshold_ratio
                    })
        
        threshold_df = pd.DataFrame(threshold_data)
        for precision in threshold_df['precision'].unique():
            data = threshold_df[threshold_df['precision'] == precision]
            plt.plot(data['I_ext'], data['threshold_ratio'], 'o-', label=precision)
        plt.xlabel('Corriente externa (mV)')
        plt.ylabel('Proporción que alcanza umbral')
        plt.title('Eficacia de Disparo vs Corriente')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/stress_tests/threshold_test_results.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_noise_test_results(self, df):
        """Visualiza resultados de la prueba de ruido"""
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Número de spikes vs ruido
        plt.subplot(2, 3, 1)
        for precision in df['precision'].unique():
            data = df[df['precision'] == precision]
            mean_spikes = data.groupby('noise_std')['n_spikes'].mean()
            std_spikes = data.groupby('noise_std')['n_spikes'].std()
            plt.errorbar(mean_spikes.index, mean_spikes.values, yerr=std_spikes.values, 
                        label=precision, marker='o')
        plt.xlabel('Nivel de ruido')
        plt.ylabel('Número de spikes')
        plt.title('Spikes vs Ruido')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Variabilidad de voltaje vs ruido
        plt.subplot(2, 3, 2)
        for precision in df['precision'].unique():
            data = df[df['precision'] == precision]
            mean_voltage_std = data.groupby('noise_std')['voltage_std'].mean()
            plt.plot(mean_voltage_std.index, mean_voltage_std.values, 'o-', label=precision)
        plt.xlabel('Nivel de ruido')
        plt.ylabel('Desviación estándar del voltaje (mV)')
        plt.title('Variabilidad de Voltaje vs Ruido')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Coeficiente de variación ISI vs ruido
        plt.subplot(2, 3, 3)
        for precision in df['precision'].unique():
            data = df[df['precision'] == precision]
            data = data.dropna(subset=['isi_cv'])
            if len(data) > 0:
                mean_isi_cv = data.groupby('noise_std')['isi_cv'].mean()
                plt.plot(mean_isi_cv.index, mean_isi_cv.values, 'o-', label=precision)
        plt.xlabel('Nivel de ruido')
        plt.ylabel('Coeficiente de variación ISI')
        plt.title('Regularidad de Spikes vs Ruido')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/stress_tests/noise_test_results.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_long_simulation_results(self, df):
        """Visualiza resultados de simulaciones largas"""
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Tasa de disparo
        plt.subplot(2, 2, 1)
        precisions = df['precision'].values
        rates = df['firing_rate_hz'].values
        plt.bar(precisions, rates)
        plt.ylabel('Tasa de disparo (Hz)')
        plt.title('Tasa de Disparo en Simulación Larga')
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Estabilidad de tasa
        plt.subplot(2, 2, 2)
        stabilities = df['rate_stability'].values
        plt.bar(precisions, stabilities)
        plt.ylabel('Estabilidad de tasa (Hz)')
        plt.title('Estabilidad Temporal')
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Rango de voltaje
        plt.subplot(2, 2, 3)
        voltage_ranges = df['voltage_range'].values
        plt.bar(precisions, voltage_ranges)
        plt.ylabel('Rango de voltaje (mV)')
        plt.title('Rango de Voltaje')
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Anomalías
        plt.subplot(2, 2, 4)
        anomalies = df['voltage_anomaly'].values
        plt.bar(precisions, anomalies)
        plt.ylabel('Anomalía detectada')
        plt.title('Detección de Anomalías')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/stress_tests/long_simulation_results.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_random_conditions_results(self, df):
        """Visualiza resultados de condiciones iniciales aleatorias"""
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Distribución de spikes por neurona
        plt.subplot(2, 3, 1)
        for precision in df['precision'].unique():
            data = df[df['precision'] == precision]
            plt.hist(data['n_spikes'], alpha=0.7, label=precision, bins=10)
        plt.xlabel('Número de spikes')
        plt.ylabel('Frecuencia')
        plt.title('Distribución de Spikes por Neurona')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Tasa de disparo vs voltaje inicial
        plt.subplot(2, 3, 2)
        for precision in df['precision'].unique():
            data = df[df['precision'] == precision]
            plt.scatter(data['initial_voltage'], data['firing_rate_hz'], alpha=0.7, label=precision)
        plt.xlabel('Voltaje inicial (mV)')
        plt.ylabel('Tasa de disparo (Hz)')
        plt.title('Tasa vs Voltaje Inicial')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Entropía de spikes
        plt.subplot(2, 3, 3)
        for precision in df['precision'].unique():
            data = df[df['precision'] == precision]
            plt.hist(data['spike_entropy'], alpha=0.7, label=precision, bins=10)
        plt.xlabel('Entropía de spikes')
        plt.ylabel('Frecuencia')
        plt.title('Distribución de Entropía')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Variabilidad de voltaje
        plt.subplot(2, 3, 4)
        for precision in df['precision'].unique():
            data = df[df['precision'] == precision]
            plt.hist(data['voltage_std'], alpha=0.7, label=precision, bins=10)
        plt.xlabel('Desviación estándar del voltaje (mV)')
        plt.ylabel('Frecuencia')
        plt.title('Variabilidad de Voltaje')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/stress_tests/random_conditions_results.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def run_all_stress_tests(self):
        """Ejecuta todas las pruebas de estrés"""
        logging.info("=== EJECUTANDO TODAS LAS PRUEBAS DE ESTRÉS ===")
        
        # 1. Prueba de corriente umbral
        I_ext_range = np.arange(1.0, 5.0, 0.1) * b2.mV
        threshold_results = self.test_threshold_current(I_ext_range)
        
        # 2. Prueba de ruido
        noise_levels = [0.0, 0.1, 0.5, 1.0, 2.0]
        noise_results = self.test_noise_amplification(20.0 * b2.mV, noise_levels)
        
        # 3. Prueba de simulación larga
        long_sim_results = self.test_long_simulation(20.0 * b2.mV)
        
        # 4. Prueba de condiciones iniciales aleatorias
        random_conditions_results = self.test_random_initial_conditions(20.0 * b2.mV)
        
        logging.info("=== TODAS LAS PRUEBAS COMPLETADAS ===")
        
        return {
            'threshold': threshold_results,
            'noise': noise_results,
            'long_simulation': long_sim_results,
            'random_conditions': random_conditions_results
        }

def main():
    """Función principal"""
    print("=== PRUEBAS DE ESTRÉS PARA CASOS LÍMITE ===")
    print("Implementando técnicas de prueba de estrés avanzadas")
    print("1. Corriente justo en el umbral")
    print("2. Amplificación por ruido aleatorio")
    print("3. Simulaciones largas")
    print("4. Condiciones iniciales aleatorias")
    print()
    
    # Crear instancia de pruebas de estrés
    stress_tester = StressTestCases(dt=0.1*b2.ms)
    
    # Ejecutar todas las pruebas
    results = stress_tester.run_all_stress_tests()
    
    print("\n=== PRUEBAS COMPLETADAS ===")
    print("Revisa los archivos generados:")
    print("- stress_test_cases.log: Log detallado")
    print("- results/stress_tests/threshold_current_test.csv: Prueba de umbral")
    print("- results/stress_tests/noise_amplification_test.csv: Prueba de ruido")
    print("- results/stress_tests/long_simulation_test.csv: Simulación larga")
    print("- results/stress_tests/random_initial_conditions_test.csv: Condiciones aleatorias")
    print("- results/stress_tests/*.png: Gráficas de resultados")

if __name__ == "__main__":
    main() 