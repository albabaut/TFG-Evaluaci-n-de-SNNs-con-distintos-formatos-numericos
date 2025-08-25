import brian2 as b2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import logging
from posit_wrapper import convert16
import seaborn as sns
from scipy import signal
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('balanced_network_analysis.log'),
        logging.StreamHandler()
    ]
)

class BalancedNetworkAnalysis:
    def __init__(self, dt=0.1*b2.ms, simulation_time=2000*b2.ms):
        """
        Inicializa el análisis de redes balanceadas E/I
        
        Args:
            dt: Paso de tiempo
            simulation_time: Tiempo de simulación
        """
        self.dt = dt
        self.simulation_time = simulation_time
        
        # Parámetros de la neurona LIF
        self.tau = 10 * b2.ms
        self.V_rest = -70 * b2.mV
        self.Vth = -50 * b2.mV
        self.Vreset = -70 * b2.mV
        
        # Configurar Brian2
        b2.prefs.codegen.target = 'numpy'
        
        logging.info(f"BalancedNetworkAnalysis inicializado")
    
    def create_balanced_network(self, n_excitatory=100, n_inhibitory=25, precision_type='float64', 
                               connectivity=0.1, external_input=None):
        """
        Crea una red balanceada E/I
        
        Args:
            n_excitatory: Número de neuronas excitatorias
            n_inhibitory: Número de neuronas inhibitorias
            precision_type: Tipo de precisión numérica
            connectivity: Probabilidad de conexión
            external_input: Input externo (opcional)
        
        Returns:
            network_components: Diccionario con componentes de la red
        """
        logging.info(f"Creando red balanceada: {n_excitatory}E/{n_inhibitory}I con precisión {precision_type}")
        
        # Configurar dtype según precisión
        if precision_type == 'float64':
            dtype = np.float64
        elif precision_type == 'float32':
            dtype = np.float32
        elif precision_type == 'float16':
            dtype = np.float16
        elif precision_type == 'posit16':
            dtype = np.float32  # Base, luego convertir
        
        # Ecuaciones de la neurona
        eqs = '''
        dv/dt = (-(V_rest - v) + I_syn + I_ext) / tau : volt (unless refractory)
        I_syn : volt
        I_ext : volt
        '''
        
        # Crear grupos neuronales
        excitatory = b2.NeuronGroup(n_excitatory, eqs, threshold='v > Vth', reset='v = Vreset',
                                   refractory=5*b2.ms, method='euler', dtype=dtype)
        inhibitory = b2.NeuronGroup(n_inhibitory, eqs, threshold='v > Vth', reset='v = Vreset',
                                   refractory=5*b2.ms, method='euler', dtype=dtype)
        
        # Condiciones iniciales
        excitatory.v = self.V_rest
        inhibitory.v = self.V_rest
        
        # Aplicar conversión de precisión si es necesario
        if precision_type == 'posit16':
            excitatory.v = np.array([convert16(float(v / b2.mV)) * b2.mV for v in excitatory.v])
            inhibitory.v = np.array([convert16(float(v / b2.mV)) * b2.mV for v in inhibitory.v])
        
        # Input externo
        if external_input is not None:
            excitatory.I_ext = external_input
            inhibitory.I_ext = external_input
        else:
            excitatory.I_ext = 0 * b2.mV
            inhibitory.I_ext = 0 * b2.mV
        
        # Crear sinapsis
        # E->E
        ee_synapses = b2.Synapses(excitatory, excitatory,
                                 on_pre='I_syn_post += 0.1*mV',
                                 dtype=dtype)
        ee_synapses.connect(p=connectivity)
        
        # E->I
        ei_synapses = b2.Synapses(excitatory, inhibitory,
                                 on_pre='I_syn_post += 0.1*mV',
                                 dtype=dtype)
        ei_synapses.connect(p=connectivity)
        
        # I->E
        ie_synapses = b2.Synapses(inhibitory, excitatory,
                                 on_pre='I_syn_post -= 0.4*mV',
                                 dtype=dtype)
        ie_synapses.connect(p=connectivity)
        
        # I->I
        ii_synapses = b2.Synapses(inhibitory, inhibitory,
                                 on_pre='I_syn_post -= 0.4*mV',
                                 dtype=dtype)
        ii_synapses.connect(p=connectivity)
        
        # Monitores
        e_monitor = b2.StateMonitor(excitatory, 'v', record=True)
        i_monitor = b2.StateMonitor(inhibitory, 'v', record=True)
        e_spikes = b2.SpikeMonitor(excitatory)
        i_spikes = b2.SpikeMonitor(inhibitory)
        
        network_components = {
            'excitatory': excitatory,
            'inhibitory': inhibitory,
            'ee_synapses': ee_synapses,
            'ei_synapses': ei_synapses,
            'ie_synapses': ie_synapses,
            'ii_synapses': ii_synapses,
            'e_monitor': e_monitor,
            'i_monitor': i_monitor,
            'e_spikes': e_spikes,
            'i_spikes': i_spikes,
            'precision': precision_type
        }
        
        return network_components
    
    def run_balanced_simulation(self, n_excitatory=100, n_inhibitory=25, external_input=None,
                               precision_types=['float64', 'float32', 'float16', 'posit16']):
        """
        Ejecuta simulación de red balanceada con diferentes precisiones
        
        Args:
            n_excitatory: Número de neuronas excitatorias
            n_inhibitory: Número de neuronas inhibitorias
            external_input: Input externo
            precision_types: Tipos de precisión a probar
        
        Returns:
            results: Diccionario con resultados por precisión
        """
        logging.info("=== EJECUTANDO SIMULACIÓN DE RED BALANCEADA ===")
        
        results = {}
        
        for precision in precision_types:
            logging.info(f"Ejecutando red balanceada con precisión: {precision}")
            
            # Limpiar estado de Brian2
            b2.device.reinit()
            b2.device.activate()
            
            # Crear red
            network = self.create_balanced_network(n_excitatory, n_inhibitory, precision, external_input=external_input)
            
            # Crear Network de Brian2
            all_objects = [
                network['excitatory'], network['inhibitory'],
                network['ee_synapses'], network['ei_synapses'],
                network['ie_synapses'], network['ii_synapses'],
                network['e_monitor'], network['i_monitor'],
                network['e_spikes'], network['i_spikes']
            ]
            
            net = b2.Network(all_objects)
            
            # Ejecutar simulación
            namespace = {
                'tau': self.tau, 'V_rest': self.V_rest, 
                'Vth': self.Vth, 'Vreset': self.Vreset
            }
            
            net.run(self.simulation_time, namespace=namespace)
            
            # Extraer resultados
            e_voltage = network['e_monitor'].v / b2.mV
            i_voltage = network['i_monitor'].v / b2.mV
            time = network['e_monitor'].t / b2.ms
            
            e_spikes = network['e_spikes'].t / b2.ms
            e_spike_neurons = network['e_spikes'].i
            i_spikes = network['i_spikes'].t / b2.ms
            i_spike_neurons = network['i_spikes'].i
            
            # Calcular métricas de actividad
            e_firing_rates = np.bincount(e_spike_neurons, minlength=n_excitatory) / (float(self.simulation_time / b2.second))
            i_firing_rates = np.bincount(i_spike_neurons, minlength=n_inhibitory) / (float(self.simulation_time / b2.second))
            
            # Calcular métricas globales
            e_mean_rate = np.mean(e_firing_rates)
            i_mean_rate = np.mean(i_firing_rates)
            e_rate_std = np.std(e_firing_rates)
            i_rate_std = np.std(i_firing_rates)
            
            # Calcular sincronía
            e_synchrony = self.calculate_synchrony(e_spikes, e_spike_neurons, n_excitatory)
            i_synchrony = self.calculate_synchrony(i_spikes, i_spike_neurons, n_inhibitory)
            
            # Calcular oscilaciones globales
            e_oscillations = self.calculate_oscillations(e_spikes, time)
            i_oscillations = self.calculate_oscillations(i_spikes, time)
            
            results[precision] = {
                'e_voltage': e_voltage,
                'i_voltage': i_voltage,
                'time': time,
                'e_spikes': e_spikes,
                'e_spike_neurons': e_spike_neurons,
                'i_spikes': i_spikes,
                'i_spike_neurons': i_spike_neurons,
                'e_firing_rates': e_firing_rates,
                'i_firing_rates': i_firing_rates,
                'e_mean_rate': e_mean_rate,
                'i_mean_rate': i_mean_rate,
                'e_rate_std': e_rate_std,
                'i_rate_std': i_rate_std,
                'e_synchrony': e_synchrony,
                'i_synchrony': i_synchrony,
                'e_oscillations': e_oscillations,
                'i_oscillations': i_oscillations,
                'network': network
            }
            
            logging.info(f"  {precision}: E={len(e_spikes)} spikes, I={len(i_spikes)} spikes")
            logging.info(f"    E rate: {e_mean_rate:.2f} Hz, I rate: {i_mean_rate:.2f} Hz")
        
        return results
    
    def calculate_synchrony(self, spikes, spike_neurons, n_neurons):
        """
        Calcula la sincronía de la población
        
        Args:
            spikes: Tiempos de spike
            spike_neurons: Neuronas que dispararon
            n_neurons: Número total de neuronas
        
        Returns:
            synchrony: Medida de sincronía
        """
        if len(spikes) == 0:
            return 0.0
        
        # Crear señal de población
        time_bins = np.arange(0, float(self.simulation_time / b2.ms), 1.0)
        population_signal = np.zeros(len(time_bins))
        
        for spike_time in spikes:
            bin_idx = int(spike_time)
            if bin_idx < len(population_signal):
                population_signal[bin_idx] += 1
        
        # Normalizar
        population_signal = population_signal / n_neurons
        
        # Calcular varianza temporal
        synchrony = np.var(population_signal)
        
        return synchrony
    
    def calculate_oscillations(self, spikes, time):
        """
        Calcula las oscilaciones en la actividad
        
        Args:
            spikes: Tiempos de spike
            time: Vector de tiempo
        
        Returns:
            oscillations: Información sobre oscilaciones
        """
        if len(spikes) == 0:
            return {'dominant_freq': 0.0, 'power_spectrum': None}
        
        # Crear señal de actividad
        activity_signal = np.zeros(len(time))
        
        for spike_time in spikes:
            time_idx = np.argmin(np.abs(time - spike_time))
            if time_idx < len(activity_signal):
                activity_signal[time_idx] += 1
        
        # Calcular espectro de potencia
        if len(activity_signal) > 1:
            freqs, power = signal.welch(activity_signal, fs=1000.0/float(self.dt/b2.ms))
            
            # Encontrar frecuencia dominante
            dominant_freq_idx = np.argmax(power)
            dominant_freq = freqs[dominant_freq_idx]
            
            return {
                'dominant_freq': dominant_freq,
                'power_spectrum': power,
                'frequencies': freqs
            }
        else:
            return {'dominant_freq': 0.0, 'power_spectrum': None}
    
    def analyze_network_differences(self, results):
        """
        Analiza las diferencias entre redes con diferentes precisiones
        
        Args:
            results: Resultados de las simulaciones
        
        Returns:
            analysis: Análisis de diferencias
        """
        logging.info("=== ANALIZANDO DIFERENCIAS DE RED BALANCEADA ===")
        
        # Usar float64 como referencia
        reference = results['float64']
        analysis = {}
        
        for precision in ['float32', 'float16', 'posit16']:
            if precision not in results:
                continue
            
            test = results[precision]
            precision_analysis = {}
            
            # Comparar tasas de disparo
            e_rate_diff = test['e_mean_rate'] - reference['e_mean_rate']
            i_rate_diff = test['i_mean_rate'] - reference['i_mean_rate']
            
            e_rate_correlation = np.corrcoef(test['e_firing_rates'], reference['e_firing_rates'])[0, 1]
            i_rate_correlation = np.corrcoef(test['i_firing_rates'], reference['i_firing_rates'])[0, 1]
            
            # Comparar sincronía
            e_sync_diff = test['e_synchrony'] - reference['e_synchrony']
            i_sync_diff = test['i_synchrony'] - reference['i_synchrony']
            
            # Comparar oscilaciones
            e_osc_diff = test['e_oscillations']['dominant_freq'] - reference['e_oscillations']['dominant_freq']
            i_osc_diff = test['i_oscillations']['dominant_freq'] - reference['i_oscillations']['dominant_freq']
            
            # Comparar número de spikes
            e_spike_diff = len(test['e_spikes']) - len(reference['e_spikes'])
            i_spike_diff = len(test['i_spikes']) - len(reference['i_spikes'])
            
            # Calcular balance E/I
            ref_ei_ratio = reference['e_mean_rate'] / (reference['i_mean_rate'] + 1e-6)
            test_ei_ratio = test['e_mean_rate'] / (test['i_mean_rate'] + 1e-6)
            ei_ratio_diff = test_ei_ratio - ref_ei_ratio
            
            precision_analysis = {
                'e_rate_diff': e_rate_diff,
                'i_rate_diff': i_rate_diff,
                'e_rate_correlation': e_rate_correlation,
                'i_rate_correlation': i_rate_correlation,
                'e_sync_diff': e_sync_diff,
                'i_sync_diff': i_sync_diff,
                'e_osc_diff': e_osc_diff,
                'i_osc_diff': i_osc_diff,
                'e_spike_diff': e_spike_diff,
                'i_spike_diff': i_spike_diff,
                'ei_ratio_diff': ei_ratio_diff,
                'ref_ei_ratio': ref_ei_ratio,
                'test_ei_ratio': test_ei_ratio
            }
            
            analysis[precision] = precision_analysis
        
        return analysis
    
    def create_balanced_visualizations(self, results, analysis):
        """
        Crea visualizaciones para el análisis de red balanceada
        
        Args:
            results: Resultados de las simulaciones
            analysis: Análisis de diferencias
        """
        logging.info("=== CREANDO VISUALIZACIONES DE RED BALANCEADA ===")
        
        os.makedirs('results/balanced_network_analysis', exist_ok=True)
        
        # 1. Raster plots por precisión
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for i, precision in enumerate(['float64', 'float32', 'float16', 'posit16']):
            if precision not in results:
                continue
            
            ax = axes[i//2, i%2]
            
            # Spikes excitatorios
            e_spikes = results[precision]['e_spikes']
            e_spike_neurons = results[precision]['e_spike_neurons']
            ax.scatter(e_spikes, e_spike_neurons, s=1, alpha=0.7, color='blue', label='Excitatory')
            
            # Spikes inhibitorios
            i_spikes = results[precision]['i_spikes']
            i_spike_neurons = results[precision]['i_spike_neurons']
            n_excitatory = len(results[precision]['e_firing_rates'])
            ax.scatter(i_spikes, i_spike_neurons + n_excitatory, s=1, alpha=0.7, color='red', label='Inhibitory')
            
            ax.set_xlabel('Tiempo (ms)')
            ax.set_ylabel('Neurona')
            ax.set_title(f'Raster Plot - {precision}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/balanced_network_analysis/raster_plots.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # 2. Tasas de disparo
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for i, precision in enumerate(['float64', 'float32', 'float16', 'posit16']):
            if precision not in results:
                continue
            
            ax = axes[i//2, i%2]
            
            # Histograma de tasas excitatorias
            e_rates = results[precision]['e_firing_rates']
            i_rates = results[precision]['i_firing_rates']
            
            ax.hist(e_rates, alpha=0.7, label='Excitatory', bins=20, color='blue')
            ax.hist(i_rates, alpha=0.7, label='Inhibitory', bins=20, color='red')
            
            ax.set_xlabel('Tasa de disparo (Hz)')
            ax.set_ylabel('Frecuencia')
            ax.set_title(f'Distribución de Tasas - {precision}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/balanced_network_analysis/firing_rates.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # 3. Comparación de métricas
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        metrics = ['e_rate_diff', 'i_rate_diff', 'e_sync_diff', 'i_sync_diff', 'e_osc_diff', 'ei_ratio_diff']
        metric_names = ['Diff E Rate', 'Diff I Rate', 'Diff E Sync', 'Diff I Sync', 'Diff E Osc', 'Diff E/I Ratio']
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i//3, i%3]
            
            precisions = []
            values = []
            
            for precision in ['float32', 'float16', 'posit16']:
                if precision in analysis:
                    precisions.append(precision)
                    values.append(analysis[precision][metric])
            
            if values:
                bars = ax.bar(precisions, values, alpha=0.8)
                ax.set_ylabel(name)
                ax.set_title(f'Comparación de {name}')
                ax.grid(True, alpha=0.3)
                
                # Agregar valores en las barras
                for bar, value in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(values),
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('results/balanced_network_analysis/metrics_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # 4. Espectros de potencia
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for i, precision in enumerate(['float64', 'float32', 'float16', 'posit16']):
            if precision not in results:
                continue
            
            ax = axes[i//2, i%2]
            
            # Espectro excitatorio
            e_osc = results[precision]['e_oscillations']
            if e_osc['power_spectrum'] is not None:
                ax.plot(e_osc['frequencies'], e_osc['power_spectrum'], label='Excitatory', color='blue')
            
            # Espectro inhibitorio
            i_osc = results[precision]['i_oscillations']
            if i_osc['power_spectrum'] is not None:
                ax.plot(i_osc['frequencies'], i_osc['power_spectrum'], label='Inhibitory', color='red')
            
            ax.set_xlabel('Frecuencia (Hz)')
            ax.set_ylabel('Potencia')
            ax.set_title(f'Espectro de Potencia - {precision}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 50)  # Limitar a frecuencias bajas
        
        plt.tight_layout()
        plt.savefig('results/balanced_network_analysis/power_spectra.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # 5. Evolución temporal de actividad
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for i, precision in enumerate(['float64', 'float32', 'float16', 'posit16']):
            if precision not in results:
                continue
            
            ax = axes[i//2, i%2]
            
            time = results[precision]['time']
            
            # Actividad excitatoria promedio
            e_voltage = results[precision]['e_voltage']
            e_voltage_mean = np.mean(e_voltage, axis=0)
            ax.plot(time, e_voltage_mean, label='Excitatory', color='blue', alpha=0.8)
            
            # Actividad inhibitoria promedio
            i_voltage = results[precision]['i_voltage']
            i_voltage_mean = np.mean(i_voltage, axis=0)
            ax.plot(time, i_voltage_mean, label='Inhibitory', color='red', alpha=0.8)
            
            ax.set_xlabel('Tiempo (ms)')
            ax.set_ylabel('Voltaje promedio (mV)')
            ax.set_title(f'Actividad Temporal - {precision}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/balanced_network_analysis/temporal_activity.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def create_balanced_report(self, results, analysis):
        """
        Crea reporte detallado de la red balanceada
        
        Args:
            results: Resultados de las simulaciones
            analysis: Análisis de diferencias
        """
        logging.info("=== CREANDO REPORTE DE RED BALANCEADA ===")
        
        # Crear DataFrame con métricas por precisión
        report_data = []
        
        for precision in results.keys():
            result = results[precision]
            
            row = {
                'precision': precision,
                'e_mean_rate': result['e_mean_rate'],
                'i_mean_rate': result['i_mean_rate'],
                'e_rate_std': result['e_rate_std'],
                'i_rate_std': result['i_rate_std'],
                'e_synchrony': result['e_synchrony'],
                'i_synchrony': result['i_synchrony'],
                'e_osc_freq': result['e_oscillations']['dominant_freq'],
                'i_osc_freq': result['i_oscillations']['dominant_freq'],
                'e_n_spikes': len(result['e_spikes']),
                'i_n_spikes': len(result['i_spikes']),
                'ei_ratio': result['e_mean_rate'] / (result['i_mean_rate'] + 1e-6)
            }
            
            report_data.append(row)
        
        df = pd.DataFrame(report_data)
        
        # Crear DataFrame de comparación
        comparison_data = []
        for precision in analysis.keys():
            metrics = analysis[precision]
            row = {
                'precision': precision,
                'e_rate_diff': metrics['e_rate_diff'],
                'i_rate_diff': metrics['i_rate_diff'],
                'e_rate_correlation': metrics['e_rate_correlation'],
                'i_rate_correlation': metrics['i_rate_correlation'],
                'e_sync_diff': metrics['e_sync_diff'],
                'i_sync_diff': metrics['i_sync_diff'],
                'e_osc_diff': metrics['e_osc_diff'],
                'i_osc_diff': metrics['i_osc_diff'],
                'e_spike_diff': metrics['e_spike_diff'],
                'i_spike_diff': metrics['i_spike_diff'],
                'ei_ratio_diff': metrics['ei_ratio_diff']
            }
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Guardar reportes
        df.to_csv('results/balanced_network_analysis/network_metrics.csv', index=False)
        comparison_df.to_csv('results/balanced_network_analysis/network_comparison.csv', index=False)
        
        # Imprimir resumen
        print("\n=== RESUMEN DE RED BALANCEADA ===")
        for precision in results.keys():
            result = results[precision]
            print(f"\n{precision}:")
            print(f"  E rate: {result['e_mean_rate']:.2f} ± {result['e_rate_std']:.2f} Hz")
            print(f"  I rate: {result['i_mean_rate']:.2f} ± {result['i_rate_std']:.2f} Hz")
            print(f"  E/I ratio: {result['e_mean_rate']/(result['i_mean_rate']+1e-6):.2f}")
            print(f"  E synchrony: {result['e_synchrony']:.4f}")
            print(f"  I synchrony: {result['i_synchrony']:.4f}")
            print(f"  E oscillations: {result['e_oscillations']['dominant_freq']:.1f} Hz")
            print(f"  I oscillations: {result['i_oscillations']['dominant_freq']:.1f} Hz")
        
        return df, comparison_df
    
    def run_comprehensive_balanced_analysis(self):
        """
        Ejecuta análisis comprehensivo de redes balanceadas
        """
        logging.info("=== INICIANDO ANÁLISIS COMPREHENSIVO DE REDES BALANCEADAS ===")
        
        # Configuraciones a probar
        configs = [
            {'name': 'small', 'n_excitatory': 50, 'n_inhibitory': 12, 'external_input': None},
            {'name': 'medium', 'n_excitatory': 100, 'n_inhibitory': 25, 'external_input': None},
            {'name': 'large', 'n_excitatory': 200, 'n_inhibitory': 50, 'external_input': None},
            {'name': 'with_input', 'n_excitatory': 100, 'n_inhibitory': 25, 'external_input': 5.0 * b2.mV}
        ]
        
        all_results = {}
        
        for config in configs:
            logging.info(f"Probando configuración: {config['name']}")
            
            # Ejecutar simulación
            results = self.run_balanced_simulation(
                n_excitatory=config['n_excitatory'],
                n_inhibitory=config['n_inhibitory'],
                external_input=config['external_input']
            )
            
            # Analizar diferencias
            analysis = self.analyze_network_differences(results)
            
            # Crear visualizaciones
            self.create_balanced_visualizations(results, analysis)
            
            # Crear reporte
            df, comparison_df = self.create_balanced_report(results, analysis)
            
            # Almacenar resultados
            all_results[config['name']] = {
                'config': config,
                'results': results,
                'analysis': analysis,
                'metrics_report': df,
                'comparison_report': comparison_df
            }
        
        logging.info("=== ANÁLISIS DE REDES BALANCEADAS COMPLETADO ===")
        
        return all_results

def main():
    """Función principal"""
    print("=== ANÁLISIS DE REDES BALANCEADAS E/I ===")
    print("Estudiando sensibilidad a precisión en redes dinámicamente complejas")
    print("1. Redes balanceadas excitatorias/inhibitorias")
    print("2. Análisis de sincronía y oscilaciones")
    print("3. Comparación de dinámicas globales")
    print("4. Reporte de estabilidad y balance")
    print()
    
    # Crear instancia del analizador
    analyzer = BalancedNetworkAnalysis(dt=0.1*b2.ms, simulation_time=2000*b2.ms)
    
    # Ejecutar análisis comprehensivo
    results = analyzer.run_comprehensive_balanced_analysis()
    
    print("\n=== ANÁLISIS COMPLETADO ===")
    print("Revisa los archivos generados:")
    print("- balanced_network_analysis.log: Log detallado")
    print("- results/balanced_network_analysis/raster_plots.png: Raster plots")
    print("- results/balanced_network_analysis/firing_rates.png: Tasas de disparo")
    print("- results/balanced_network_analysis/metrics_comparison.png: Comparación de métricas")
    print("- results/balanced_network_analysis/network_metrics.csv: Métricas de red")
    print("- results/balanced_network_analysis/network_comparison.csv: Comparación de redes")

if __name__ == "__main__":
    main() 