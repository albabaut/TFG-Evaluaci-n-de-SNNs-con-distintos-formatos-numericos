import brian2 as b2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import logging
from posit_wrapper import convert16
from scipy import signal
from scipy.stats import entropy
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('advanced_numerical_comparison.log'),
        logging.StreamHandler()
    ]
)

class AdvancedNumericalComparison:
    def __init__(self, dt=0.1*b2.ms, simulation_time=1000*b2.ms):
        """
        Inicializa el comparador avanzado de formatos numéricos
        
        Args:
            dt: Paso de tiempo
            simulation_time: Tiempo total de simulación
        """
        self.dt = dt
        self.simulation_time = simulation_time
        
        # Parámetros de la neurona LIF
        self.tau = 10 * b2.ms
        self.V_rest = -70 * b2.mV
        self.Vth = -50 * b2.mV
        self.Vreset = -70 * b2.mV
        
        # Almacenamiento de resultados
        self.results = {}
        
        # Configurar Brian2
        b2.prefs.codegen.target = 'numpy'
        
        logging.info(f"AdvancedNumericalComparison inicializado")
    
    def run_simulation_with_precision(self, precision_type, I_ext, noise_std=0.0, seed=42):
        """
        Ejecuta simulación con precisión específica
        
        Args:
            precision_type: 'float64', 'float32', 'float16', 'posit16'
            I_ext: Corriente externa
            noise_std: Desviación estándar del ruido
            seed: Semilla para reproducibilidad
        """
        logging.info(f"Ejecutando simulación {precision_type} con I_ext={I_ext}")
        
        # Limpiar estado de Brian2
        b2.device.reinit()
        b2.device.activate()
        
        # Configurar semilla
        np.random.seed(seed)
        
        # Ecuaciones con ruido opcional
        if noise_std > 0:
            eqs = '''
            dv/dt = (-(V_rest - v) + I_ext) / tau + xi * noise_std / sqrt(tau) : volt (unless refractory)
            '''
        else:
            eqs = '''
            dv/dt = (-(V_rest - v) + I_ext) / tau : volt (unless refractory)
            '''
        
        # Crear grupo neuronal con precisión específica
        if precision_type == 'float64':
            dtype = np.float64
        elif precision_type == 'float32':
            dtype = np.float32
        elif precision_type == 'float16':
            dtype = np.float16
        elif precision_type == 'posit16':
            dtype = np.float32  # Usar float32 como base, luego convertir
        
        G = b2.NeuronGroup(1, eqs, threshold='v > Vth', reset='v = Vreset',
                          refractory=5*b2.ms, method='euler', dtype=dtype)
        G.v = self.V_rest
        
        # Monitores
        mon = b2.StateMonitor(G, 'v', record=True)
        spk = b2.SpikeMonitor(G)
        
        # Network
        net = b2.Network(G, mon, spk)
        
        # Ejecutar simulación
        namespace = {
            'I_ext': I_ext, 'tau': self.tau, 'V_rest': self.V_rest, 
            'Vth': self.Vth, 'Vreset': self.Vreset, 'noise_std': noise_std
        }
        
        net.run(self.simulation_time, namespace=namespace)
        
        # Extraer datos
        voltage = mon.v[0] / b2.mV
        time = mon.t / b2.ms
        spikes = spk.t / b2.ms
        
        # Aplicar conversión de precisión si es necesario
        if precision_type == 'posit16':
            voltage = np.array([convert16(v) for v in voltage])
        
        self.results[precision_type] = {
            'voltage': voltage,
            'time': time,
            'spikes': spikes,
            'I_ext': I_ext,
            'noise_std': noise_std
        }
        
        logging.info(f"Simulación {precision_type} completada. {len(spikes)} spikes")
        
        return voltage, time, spikes
    
    def calculate_spike_jitter(self, reference_spikes, test_spikes, tolerance=1.0):
        """
        Calcula el jitter de spikes entre dos simulaciones
        
        Args:
            reference_spikes: Spikes de referencia (float64)
            test_spikes: Spikes de prueba
            tolerance: Tolerancia para emparejar spikes (ms)
        
        Returns:
            jitter_times: Lista de diferencias temporales
            matched_pairs: Pares de spikes emparejados
        """
        if len(reference_spikes) == 0 or len(test_spikes) == 0:
            return [], []
        
        jitter_times = []
        matched_pairs = []
        used_test_spikes = set()
        
        for ref_spike in reference_spikes:
            # Encontrar el spike más cercano en test_spikes
            distances = np.abs(test_spikes - ref_spike)
            min_idx = np.argmin(distances)
            min_distance = distances[min_idx]
            
            if min_distance <= tolerance and min_idx not in used_test_spikes:
                jitter_times.append(test_spikes[min_idx] - ref_spike)
                matched_pairs.append((ref_spike, test_spikes[min_idx]))
                used_test_spikes.add(min_idx)
        
        return jitter_times, matched_pairs
    
    def calculate_firing_latency(self, spikes, stimulus_start=0.0):
        """
        Calcula la latencia del primer spike después del estímulo
        
        Args:
            spikes: Tiempos de spike
            stimulus_start: Tiempo de inicio del estímulo
        
        Returns:
            latency: Latencia del primer spike (ms)
        """
        if len(spikes) == 0:
            return np.nan
        
        # Encontrar el primer spike después del estímulo
        post_stimulus_spikes = spikes[spikes >= stimulus_start]
        
        if len(post_stimulus_spikes) == 0:
            return np.nan
        
        return post_stimulus_spikes[0] - stimulus_start
    
    def calculate_victor_purpura_distance(self, spikes1, spikes2, cost_factor=1.0):
        """
        Calcula la distancia Victor-Purpura entre dos trenes de spikes
        
        Args:
            spikes1: Primer tren de spikes
            spikes2: Segundo tren de spikes
            cost_factor: Factor de costo temporal
        
        Returns:
            distance: Distancia Victor-Purpura
        """
        if len(spikes1) == 0 and len(spikes2) == 0:
            return 0.0
        elif len(spikes1) == 0:
            return len(spikes2)
        elif len(spikes2) == 0:
            return len(spikes1)
        
        # Implementación simplificada de Victor-Purpura
        # Para una implementación completa, usar librería Elephant
        
        # Calcular matriz de costos
        n1, n2 = len(spikes1), len(spikes2)
        cost_matrix = np.zeros((n1 + 1, n2 + 1))
        
        # Inicializar primera fila y columna
        for i in range(n1 + 1):
            cost_matrix[i, 0] = i
        for j in range(n2 + 1):
            cost_matrix[0, j] = j
        
        # Llenar matriz de costos
        for i in range(1, n1 + 1):
            for j in range(1, n2 + 1):
                # Costo de inserción/eliminación
                insert_cost = cost_matrix[i, j-1] + 1
                delete_cost = cost_matrix[i-1, j] + 1
                
                # Costo de movimiento
                time_cost = cost_factor * abs(spikes1[i-1] - spikes2[j-1])
                move_cost = cost_matrix[i-1, j-1] + time_cost
                
                cost_matrix[i, j] = min(insert_cost, delete_cost, move_cost)
        
        return cost_matrix[n1, n2]
    
    def calculate_isi_distance(self, spikes1, spikes2):
        """
        Calcula la distancia de intervalos inter-spike
        
        Args:
            spikes1: Primer tren de spikes
            spikes2: Segundo tren de spikes
        
        Returns:
            distance: Distancia ISI
        """
        def get_isis(spikes):
            if len(spikes) < 2:
                return np.array([])
            return np.diff(spikes)
        
        isi1 = get_isis(spikes1)
        isi2 = get_isis(spikes2)
        
        if len(isi1) == 0 and len(isi2) == 0:
            return 0.0
        elif len(isi1) == 0:
            return np.sum(isi2)
        elif len(isi2) == 0:
            return np.sum(isi1)
        
        # Calcular distancia usando correlación
        min_len = min(len(isi1), len(isi2))
        if min_len == 0:
            return np.inf
        
        # Normalizar y calcular correlación
        isi1_norm = (isi1[:min_len] - np.mean(isi1[:min_len])) / np.std(isi1[:min_len])
        isi2_norm = (isi2[:min_len] - np.mean(isi2[:min_len])) / np.std(isi2[:min_len])
        
        correlation = np.corrcoef(isi1_norm, isi2_norm)[0, 1]
        
        # Convertir correlación a distancia
        distance = 1 - correlation if not np.isnan(correlation) else 1.0
        
        return distance
    
    def calculate_spectral_entropy(self, spikes, time_window, bin_size=1.0):
        """
        Calcula la entropía espectral de un tren de spikes
        
        Args:
            spikes: Tiempos de spike
            time_window: Ventana de tiempo (ms)
            bin_size: Tamaño del bin (ms)
        
        Returns:
            entropy: Entropía espectral
        """
        if len(spikes) == 0:
            return 0.0
        
        # Crear señal binaria
        n_bins = int(time_window / bin_size)
        signal_binary = np.zeros(n_bins)
        
        for spike in spikes:
            if spike < time_window:
                bin_idx = int(spike / bin_size)
                if bin_idx < n_bins:
                    signal_binary[bin_idx] = 1
        
        # Calcular espectro de potencia
        if np.sum(signal_binary) == 0:
            return 0.0
        
        # Aplicar FFT
        fft = np.fft.fft(signal_binary)
        power_spectrum = np.abs(fft) ** 2
        
        # Normalizar
        power_spectrum = power_spectrum / np.sum(power_spectrum)
        
        # Calcular entropía
        # Evitar log(0)
        power_spectrum = power_spectrum[power_spectrum > 0]
        
        if len(power_spectrum) == 0:
            return 0.0
        
        entropy_val = -np.sum(power_spectrum * np.log2(power_spectrum))
        
        return entropy_val
    
    def calculate_firing_rate(self, spikes, time_window):
        """
        Calcula la tasa de disparo en una ventana de tiempo
        
        Args:
            spikes: Tiempos de spike
            time_window: Ventana de tiempo (ms)
        
        Returns:
            rate: Tasa de disparo (Hz)
        """
        if len(spikes) == 0:
            return 0.0
        
        # Contar spikes en la ventana
        spikes_in_window = spikes[spikes <= time_window]
        rate = len(spikes_in_window) / (time_window / 1000.0)  # Convertir a Hz
        
        return rate
    
    def run_comprehensive_comparison(self, I_ext_values, noise_levels=[0.0]):
        """
        Ejecuta comparación comprehensiva entre formatos numéricos
        
        Args:
            I_ext_values: Lista de corrientes externas a probar
            noise_levels: Lista de niveles de ruido a probar
        """
        logging.info("=== INICIANDO COMPARACIÓN COMPREHENSIVA ===")
        
        precision_types = ['float64', 'float32', 'float16', 'posit16']
        comparison_results = []
        
        for I_ext in I_ext_values:
            for noise_std in noise_levels:
                logging.info(f"Probando I_ext={I_ext}, noise_std={noise_std}")
                
                # Ejecutar simulaciones para todas las precisiones
                for precision in precision_types:
                    self.run_simulation_with_precision(precision, I_ext, noise_std)
                
                # Calcular métricas de comparación
                reference_spikes = self.results['float64']['spikes']
                
                for test_precision in ['float32', 'float16', 'posit16']:
                    test_spikes = self.results[test_precision]['spikes']
                    test_voltage = self.results[test_precision]['voltage']
                    reference_voltage = self.results['float64']['voltage']
                    
                    # Métricas de spike
                    jitter_times, matched_pairs = self.calculate_spike_jitter(reference_spikes, test_spikes)
                    latency_ref = self.calculate_firing_latency(reference_spikes)
                    latency_test = self.calculate_firing_latency(test_spikes)
                    
                    # Distancias entre trenes
                    vp_distance = self.calculate_victor_purpura_distance(reference_spikes, test_spikes)
                    isi_distance = self.calculate_isi_distance(reference_spikes, test_spikes)
                    
                    # Entropía espectral
                    spectral_entropy_ref = self.calculate_spectral_entropy(reference_spikes, float(self.simulation_time/b2.ms))
                    spectral_entropy_test = self.calculate_spectral_entropy(test_spikes, float(self.simulation_time/b2.ms))
                    
                    # Tasas de disparo
                    firing_rate_ref = self.calculate_firing_rate(reference_spikes, float(self.simulation_time/b2.ms))
                    firing_rate_test = self.calculate_firing_rate(test_spikes, float(self.simulation_time/b2.ms))
                    
                    # Error de voltaje
                    rmse_voltage = np.sqrt(np.mean((reference_voltage - test_voltage) ** 2))
                    
                    # Almacenar resultados
                    result = {
                        'I_ext': float(I_ext / b2.mV),
                        'noise_std': noise_std,
                        'precision': test_precision,
                        'n_spikes_ref': len(reference_spikes),
                        'n_spikes_test': len(test_spikes),
                        'jitter_mean': np.mean(jitter_times) if len(jitter_times) > 0 else np.nan,
                        'jitter_std': np.std(jitter_times) if len(jitter_times) > 0 else np.nan,
                        'latency_diff': latency_test - latency_ref,
                        'vp_distance': vp_distance,
                        'isi_distance': isi_distance,
                        'spectral_entropy_diff': spectral_entropy_test - spectral_entropy_ref,
                        'firing_rate_diff': firing_rate_test - firing_rate_ref,
                        'rmse_voltage': rmse_voltage
                    }
                    
                    comparison_results.append(result)
        
        # Crear DataFrame
        df = pd.DataFrame(comparison_results)
        
        # Guardar resultados
        os.makedirs('results/advanced_comparison', exist_ok=True)
        df.to_csv('results/advanced_comparison/comprehensive_metrics.csv', index=False)
        
        logging.info(f"Comparación completada. {len(df)} métricas calculadas")
        
        return df
    
    def create_advanced_visualizations(self, df):
        """
        Crea visualizaciones avanzadas de los resultados
        
        Args:
            df: DataFrame con resultados de comparación
        """
        logging.info("=== CREANDO VISUALIZACIONES AVANZADAS ===")
        
        os.makedirs('results/advanced_comparison', exist_ok=True)
        
        # 1. Gráfica de jitter de spikes
        plt.figure(figsize=(15, 12))
        
        plt.subplot(3, 3, 1)
        for precision in ['float32', 'float16', 'posit16']:
            data = df[df['precision'] == precision]['jitter_mean'].dropna()
            if len(data) > 0:
                plt.hist(data, alpha=0.7, label=precision, bins=20)
        plt.xlabel('Jitter medio (ms)')
        plt.ylabel('Frecuencia')
        plt.title('Distribución de Jitter de Spikes')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Latencia de disparo vs corriente
        plt.subplot(3, 3, 2)
        for precision in ['float32', 'float16', 'posit16']:
            data = df[df['precision'] == precision]
            plt.plot(data['I_ext'], data['latency_diff'], 'o-', label=precision)
        plt.xlabel('Corriente externa (mV)')
        plt.ylabel('Diferencia de latencia (ms)')
        plt.title('Latencia de Disparo vs Corriente')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Distancia Victor-Purpura
        plt.subplot(3, 3, 3)
        for precision in ['float32', 'float16', 'posit16']:
            data = df[df['precision'] == precision]
            plt.plot(data['I_ext'], data['vp_distance'], 'o-', label=precision)
        plt.xlabel('Corriente externa (mV)')
        plt.ylabel('Distancia Victor-Purpura')
        plt.title('Distancia Victor-Purpura vs Corriente')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Distancia ISI
        plt.subplot(3, 3, 4)
        for precision in ['float32', 'float16', 'posit16']:
            data = df[df['precision'] == precision]
            plt.plot(data['I_ext'], data['isi_distance'], 'o-', label=precision)
        plt.xlabel('Corriente externa (mV)')
        plt.ylabel('Distancia ISI')
        plt.title('Distancia ISI vs Corriente')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. Entropía espectral
        plt.subplot(3, 3, 5)
        for precision in ['float32', 'float16', 'posit16']:
            data = df[df['precision'] == precision]
            plt.plot(data['I_ext'], data['spectral_entropy_diff'], 'o-', label=precision)
        plt.xlabel('Corriente externa (mV)')
        plt.ylabel('Diferencia de entropía espectral')
        plt.title('Entropía Espectral vs Corriente')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. Tasa de disparo
        plt.subplot(3, 3, 6)
        for precision in ['float32', 'float16', 'posit16']:
            data = df[df['precision'] == precision]
            plt.plot(data['I_ext'], data['firing_rate_diff'], 'o-', label=precision)
        plt.xlabel('Corriente externa (mV)')
        plt.ylabel('Diferencia de tasa de disparo (Hz)')
        plt.title('Tasa de Disparo vs Corriente')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 7. RMSE de voltaje
        plt.subplot(3, 3, 7)
        for precision in ['float32', 'float16', 'posit16']:
            data = df[df['precision'] == precision]
            plt.plot(data['I_ext'], data['rmse_voltage'], 'o-', label=precision)
        plt.xlabel('Corriente externa (mV)')
        plt.ylabel('RMSE voltaje (mV)')
        plt.title('RMSE Voltaje vs Corriente')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 8. Número de spikes
        plt.subplot(3, 3, 8)
        for precision in ['float32', 'float16', 'posit16']:
            data = df[df['precision'] == precision]
            plt.plot(data['I_ext'], data['n_spikes_test'], 'o-', label=precision)
        plt.xlabel('Corriente externa (mV)')
        plt.ylabel('Número de spikes')
        plt.title('Número de Spikes vs Corriente')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 9. Heatmap de correlación de métricas
        plt.subplot(3, 3, 9)
        metrics_cols = ['jitter_mean', 'latency_diff', 'vp_distance', 'isi_distance', 
                       'spectral_entropy_diff', 'firing_rate_diff', 'rmse_voltage']
        correlation_matrix = df[metrics_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlación entre Métricas')
        
        plt.tight_layout()
        plt.savefig('results/advanced_comparison/advanced_metrics_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Gráfica adicional: Comparación de voltajes superpuestos
        self.plot_voltage_comparison()
        
        # Gráfica adicional: Raster plots comparativos
        self.plot_raster_comparison()
    
    def plot_voltage_comparison(self):
        """Crea gráfica de comparación de voltajes superpuestos"""
        if 'float64' not in self.results or 'float16' not in self.results:
            return
        
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Voltajes superpuestos
        plt.subplot(2, 2, 1)
        time = self.results['float64']['time']
        v64 = self.results['float64']['voltage']
        v16 = self.results['float16']['voltage']
        
        plt.plot(time, v64, 'b-', label='float64', linewidth=2)
        plt.plot(time, v16, 'r--', label='float16', linewidth=1, alpha=0.7)
        
        # Marcar spikes
        spikes64 = self.results['float64']['spikes']
        spikes16 = self.results['float16']['spikes']
        
        plt.vlines(spikes64, -80, -40, colors='blue', alpha=0.5, linewidth=0.5)
        plt.vlines(spikes16, -80, -40, colors='red', alpha=0.5, linewidth=0.5)
        
        plt.xlabel('Tiempo (ms)')
        plt.ylabel('Voltaje (mV)')
        plt.title('Comparación de Voltajes Superpuestos')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Diferencia acumulada
        plt.subplot(2, 2, 2)
        diff = v64 - v16
        plt.plot(time, diff, 'g-', label='float64 - float16')
        plt.xlabel('Tiempo (ms)')
        plt.ylabel('Diferencia (mV)')
        plt.title('Diferencia Acumulada de Voltaje')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Error cuadrático
        plt.subplot(2, 2, 3)
        squared_error = diff ** 2
        plt.plot(time, squared_error, 'm-', label='Error cuadrático')
        plt.xlabel('Tiempo (ms)')
        plt.ylabel('Error cuadrático (mV²)')
        plt.title('Error Cuadrático vs Tiempo')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Histograma de diferencias
        plt.subplot(2, 2, 4)
        plt.hist(diff, bins=50, alpha=0.7, color='orange')
        plt.xlabel('Diferencia (mV)')
        plt.ylabel('Frecuencia')
        plt.title('Histograma de Diferencias')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/advanced_comparison/voltage_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_raster_comparison(self):
        """Crea raster plots comparativos"""
        if 'float64' not in self.results or 'float16' not in self.results:
            return
        
        plt.figure(figsize=(12, 8))
        
        # Raster plot float64
        plt.subplot(2, 1, 1)
        spikes64 = self.results['float64']['spikes']
        plt.vlines(spikes64, 0, 1, colors='blue', linewidth=2)
        plt.ylabel('float64')
        plt.title('Raster Plots Comparativos')
        plt.xlim(0, float(self.simulation_time/b2.ms))
        plt.grid(True, alpha=0.3)
        
        # Raster plot float16
        plt.subplot(2, 1, 2)
        spikes16 = self.results['float16']['spikes']
        plt.vlines(spikes16, 0, 1, colors='red', linewidth=2)
        plt.ylabel('float16')
        plt.xlabel('Tiempo (ms)')
        plt.xlim(0, float(self.simulation_time/b2.ms))
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/advanced_comparison/raster_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def run_stress_tests(self):
        """Ejecuta pruebas de estrés en casos límite"""
        logging.info("=== EJECUTANDO PRUEBAS DE ESTRÉS ===")
        
        # 1. Corriente justo en el umbral
        threshold_current = 2.0 * b2.mV  # Ajustar según sea necesario
        logging.info(f"Probando corriente en el umbral: {threshold_current}")
        
        for precision in ['float64', 'float32', 'float16', 'posit16']:
            self.run_simulation_with_precision(precision, threshold_current)
            spikes = self.results[precision]['spikes']
            logging.info(f"{precision}: {len(spikes)} spikes con corriente umbral")
        
        # 2. Simulación larga
        logging.info("Ejecutando simulación larga (10 segundos)")
        long_simulation_time = 10 * b2.second
        
        # Guardar tiempo original
        original_time = self.simulation_time
        self.simulation_time = long_simulation_time
        
        for precision in ['float64', 'float16']:
            self.run_simulation_with_precision(precision, 20.0 * b2.mV)
            spikes = self.results[precision]['spikes']
            rate = len(spikes) / (float(long_simulation_time/b2.second))
            logging.info(f"{precision}: {len(spikes)} spikes, tasa {rate:.2f} Hz en simulación larga")
        
        # Restaurar tiempo original
        self.simulation_time = original_time
        
        # 3. Con ruido
        logging.info("Ejecutando simulaciones con ruido")
        for noise_std in [0.1, 0.5, 1.0]:
            for precision in ['float64', 'float16']:
                self.run_simulation_with_precision(precision, 20.0 * b2.mV, noise_std)
                spikes = self.results[precision]['spikes']
                logging.info(f"{precision} con ruido {noise_std}: {len(spikes)} spikes")

def main():
    """Función principal"""
    print("=== COMPARACIÓN AVANZADA DE FORMATOS NUMÉRICOS ===")
    print("Implementando métricas y visualizaciones sofisticadas")
    print("1. Jitter de spikes y latencia de disparo")
    print("2. Distancias Victor-Purpura e ISI")
    print("3. Entropía espectral y tasas de disparo")
    print("4. Visualizaciones superpuestas y raster plots")
    print("5. Pruebas de estrés en casos límite")
    print()
    
    # Crear instancia del comparador
    comparator = AdvancedNumericalComparison(dt=0.1*b2.ms, simulation_time=1000*b2.ms)
    
    # Corrientes a probar
    I_ext_values = [10.0, 15.0, 20.0, 25.0, 30.0] * b2.mV
    
    # Ejecutar comparación comprehensiva
    df = comparator.run_comprehensive_comparison(I_ext_values)
    
    # Crear visualizaciones
    comparator.create_advanced_visualizations(df)
    
    # Ejecutar pruebas de estrés
    comparator.run_stress_tests()
    
    print("\n=== ANÁLISIS COMPLETADO ===")
    print("Revisa los archivos generados:")
    print("- advanced_numerical_comparison.log: Log detallado")
    print("- results/advanced_comparison/comprehensive_metrics.csv: Métricas completas")
    print("- results/advanced_comparison/advanced_metrics_analysis.png: Análisis de métricas")
    print("- results/advanced_comparison/voltage_comparison.png: Comparación de voltajes")
    print("- results/advanced_comparison/raster_comparison.png: Raster plots")

if __name__ == "__main__":
    main() 